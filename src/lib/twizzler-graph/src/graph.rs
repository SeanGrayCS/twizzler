//! The `Graph` engine: open/create a graph and run vertex/edge operations.
//!
//! `Graph` owns the cross-cutting orchestration and the registries (vertices,
//! edges, labels). Vertex-centric traversal lives on [`VertexView`] in
//! `vertex.rs`; edge/vertex record types live in their own modules.
//!
//! The registries are segmented vectors ([`SegVec`]) so they outgrow a single
//! object; lookups by id index them directly (ids are append indices, and
//! segments are uniformly sized, so id -> (segment, offset) is O(1)). The
//! `(label, name) -> vertex` point lookup uses a persistent `hachage` index.
//! `vertices_by_label` is still a scan.

use naming::{static_naming_factory, GetFlags};
use twizzler::{
    collections::hachage::{PersistentHashMap, PersistentHashMapBase},
    marker::{BaseType, Invariant},
    object::{MapFlags, ObjID, Object, ObjectBuilder, TypedObject},
};
use twizzler_rt_abi::error::ArgumentError;

use crate::{
    arena_store::{ArenaStore, FillTo},
    edge::{EdgeId, EdgeInfo, EdgeRef},
    error::{GraphError, Result},
    name::NameKey,
    props::{self, PropValue},
    reclaim,
    segvec::SegVec,
    vertex::{Labels, VertexId, VertexInfo, VertexRef, VertexView},
};

pub(crate) const MAGIC: u64 = 0x4731_5457_5A47_5248; // "G1TWZGRH"

/// Magic of a root whose graph has been [`destroyed`](Graph::destroy).
///
/// A destroyed root cannot simply be zeroed: `data/` names cannot be unbound on
/// this build, so the root outlives the graph, and a zeroed one is
/// indistinguishable from "not our object" — which made `reset` refuse it and
/// the name unusable forever, across boots, since the root persists in the
/// disk image. A distinct marker keeps three states apart: a live graph, our
/// destroyed root (rebuildable in place), and something that was never ours
/// (must not be touched).
pub(crate) const MAGIC_DESTROYED: u64 = MAGIC ^ 0xFFFF_FFFF_FFFF_FFFF;

/// On-disk format 5 ("v3"): segmented registries, one vertex object plus two
/// adjacency objects per vertex, one object per edge.
///
/// 1. The number 5 is burned. Version numbers must never be recycled — a
///    future layout reusing 5 would be read as v3 by any build still carrying
///    this guard, which is the misread-as-garbage failure the whole scheme
///    exists to prevent.
/// 2. It documents what `version_supported` is rejecting when an old image
///    turns up.
///
/// The disk image survives between QEMU runs, which is what turns "I changed a
/// struct" into "the next boot hangs". Any change to a persisted record's
/// layout — size, field order, or alignment — must bump this, so the guard
/// rejects the old graph loudly instead of misreading it.
#[allow(dead_code)] // see above: reserved, not obsolete
pub(crate) const VERSION: u32 = 5;

/// On-disk format 7 ("v4"): vertices and adjacency live in packed arenas
/// ([`ArenaStore`]) instead of three objects per vertex plus one per edge.
pub(crate) const VERSION_ARENA: u32 = 7;

/// Whether this build understands a graph in the given on-disk format.
fn version_supported(v: u32) -> bool {
    v == VERSION_ARENA
}
const TOMBSTONE: u32 = 1; // `flags` bit 0: record is deleted

/// Default per-segment registry capacity. Registry records are plain data
/// (~100–150 B, no `InvPtr`s), so 4096 entries keep a segment well under the
/// object size limit while amortizing segment creation.
pub(crate) const DEFAULT_SEG_CAP: usize = 4096;

/// Default vertices packed per arena on the VERSION 4 layout.
///
/// The benefit saturates once the interconnected set fits in one arena:
/// 16384 and 65536 were indistinguishable on every metric, because both held
/// all 6 001 phase-A vertices in a single arena. Cap only matters up to the
/// working set being connected.
///
/// # This value is PROVISIONAL
pub const DEFAULT_ARENA_CAP: usize = 16384;

/// Read/write/persist map flags for reopening mutable registries.
fn rw() -> MapFlags {
    MapFlags::READ | MapFlags::WRITE | MapFlags::PERSIST
}

/// Base of the graph root object: format guard, the registry segment
/// capacity, and the registry ObjIDs (raw, so the on-disk format is
/// backend-agnostic and relocatable). The registry ids point at `SegVec`
/// directory objects as of version 3.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct GraphRoot {
    pub(crate) magic: u64,
    pub(crate) version: u32,
    pub(crate) seg_cap: u32,
    pub(crate) verts_raw: u128,
    pub(crate) edges_raw: u128,
    pub(crate) labels_raw: u128,
    pub(crate) vindex_raw: u128,
    /// VERSION 4 only: the [`ArenaStore`]'s arena directory and location
    /// registry. Both zero in a v3 graph, and only read when `version` says 4 —
    /// appended at the end so a v3 root stays byte-compatible.
    pub(crate) arena_dir_raw: u128,
    pub(crate) arena_locs_raw: u128,
}
unsafe impl Invariant for GraphRoot {}
impl BaseType for GraphRoot {}

/// An interned label (string ↔ small id).
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct LabelEntry {
    pub(crate) id: u32,
    pub(crate) name: NameKey,
}
unsafe impl Invariant for LabelEntry {}

/// Key for the vertex index: a (label id, name) pair.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
struct VKey {
    label: u32,
    name: NameKey,
}
unsafe impl Invariant for VKey {}

/// Persistent index from (label, name) to vertex id.
type VIndex = PersistentHashMap<VKey, u64>;

/// An open graph: the root id, the registries, the vertex index, and the store.
pub struct Graph {
    root_id: ObjID,
    /// Vestigial, and genuinely unread — the `dead_code` warning this raises
    /// is deliberate. Do not silence it; it is the reminder that this field
    /// still needs removing.
    ///
    /// Vertices live in `store`; this registry is always empty. It is still
    /// *allocated* because `GraphRoot.verts_raw` is a persisted field, and still
    /// *inventoried* so `reset`/`destroy` free its objects rather than orphaning
    /// them — but both go through raw ids, not this handle. The only code that
    /// touches the handle is `owned_object_ids`, which is `#[cfg(test)]`, so a
    /// normal build reads it nowhere.
    verts: SegVec<VertexRef>,
    edges: SegVec<EdgeRef>,
    labels: SegVec<LabelEntry>,
    vindex: VIndex,
    /// Vertices and adjacency — the whole graph, in packed arenas.
    store: ArenaStore,
}

impl Graph {

    pub fn arena_count(&self) -> usize {
        self.store.arena_count()
    }

    pub fn arena_sync_count(&self) -> usize {
        self.store.sync_count()
    }

    /// Diagnostic pass-through to [`ArenaStore::arena_vertex_counts`]:
    /// `(policy view, ground truth)` vertices per arena.
    pub fn arena_vertex_counts(&self) -> (Vec<usize>, Vec<usize>) {
        self.store.arena_vertex_counts()
    }
}

impl Graph {
    /// Open the graph registered at `data/<name>`, or create and register a
    /// fresh one at [`DEFAULT_ARENA_CAP`]. If an existing graph has an
    /// incompatible format (magic/version mismatch — which now includes every
    /// v3 graph) this returns [`GraphError::StaleVersion`] and leaves the
    /// existing graph intact; use [`Graph::reset`] to discard it.
    pub fn open_or_create(name: &str) -> Result<Graph> {
        Self::open_or_create_with_capacity(name, DEFAULT_SEG_CAP)
    }

    /// Like [`Graph::open_or_create`], with an explicit registry segment
    /// capacity. The capacity is used only when *creating* a graph; an
    /// existing graph always keeps the capacity recorded in its root, since
    /// segment geometry must stay uniform for the graph's lifetime. (Small
    /// capacities let tests force segment rollover cheaply.)
    pub fn open_or_create_with_capacity(name: &str, cap: usize) -> Result<Graph> {
        Self::open_inner(name, cap, DEFAULT_ARENA_CAP)
    }

    /// Open or create a graph packing `arena_cap` vertices per arena object.
    ///
    /// The name is now redundant (every graph is an arena graph); it stays to
    /// avoid churning ~40 call sites, and should become
    /// `open_or_create_with_arena_cap` when something else touches them.
    pub fn open_or_create_arena(name: &str, arena_cap: usize) -> Result<Graph> {
        Self::open_inner(name, DEFAULT_SEG_CAP, arena_cap)
    }

    /// [`Graph::open_or_create_arena`] with an explicit registry segment
    /// capacity, for tests that force segment rollover cheaply.
    pub fn open_or_create_arena_with_capacity(
        name: &str,
        cap: usize,
        arena_cap: usize,
    ) -> Result<Graph> {
        Self::open_inner(name, cap, arena_cap)
    }

    /// Every graph created here is VERSION 4; `arena_cap` sets placement at
    /// creation and is ignored (see above) when opening an existing graph.
    fn open_inner(name: &str, cap: usize, arena_cap: usize) -> Result<Graph> {
        if cap == 0 || cap > u32::MAX as usize {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        if arena_cap == 0 {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        let mut namer = static_naming_factory().expect("naming service available");
        let path = format!("data/{name}");

        if let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) {
            let root =
                Object::<GraphRoot>::map(node.id.into(), MapFlags::READ | MapFlags::PERSIST)?;
            let (magic, version, seg_cap, verts_raw, edges_raw, labels_raw, vindex_raw) = {
                let r = root.base();
                (
                    r.magic,
                    r.version,
                    r.seg_cap,
                    r.verts_raw,
                    r.edges_raw,
                    r.labels_raw,
                    r.vindex_raw,
                )
            };
            if magic == MAGIC && version_supported(version) {
                // The persisted capacity governs, not the caller's.
                let cap = seg_cap as usize;
                let vbacking: Object<PersistentHashMapBase<VKey, u64>> =
                    Object::map(ObjID::new(vindex_raw), rw())?;
                // `version_supported` above already rejected anything but
                // VERSION_ARENA, so there is exactly one layout to open.
                let store = {
                    let (dir, locs) = {
                        let r = root.base();
                        (r.arena_dir_raw, r.arena_locs_raw)
                    };
                    ArenaStore::open(dir, locs, Box::new(FillTo { cap: arena_cap }), cap)?
                };
                return Ok(Graph {
                    root_id: node.id.into(),
                    verts: SegVec::open(verts_raw, cap)?,
                    edges: SegVec::open(edges_raw, cap)?,
                    labels: SegVec::open(labels_raw, cap)?,
                    vindex: PersistentHashMap::from(vbacking),
                    store,
                });
            }
            // Incompatible/stale format: do NOT touch the existing graph.
            // A v3 graph lands here now — deliberately, see `version_supported`.
            return Err(GraphError::StaleVersion {
                found: version,
                expected: VERSION_ARENA,
            });
        }

        let verts = SegVec::create(cap)?;
        let edges = SegVec::create(cap)?;
        let labels = SegVec::create(cap)?;
        let vindex = VIndex::new_persist()?;
        let store = ArenaStore::create(Box::new(FillTo { cap: arena_cap }), cap)?;
        let (arena_dir_raw, arena_locs_raw) = store.ids();

        let root = ObjectBuilder::<GraphRoot>::default()
            .persist(true)
            .build(GraphRoot {
                magic: MAGIC,
                version: VERSION_ARENA,
                seg_cap: cap as u32,
                verts_raw: verts.dir_raw(),
                edges_raw: edges.dir_raw(),
                labels_raw: labels.dir_raw(),
                vindex_raw: vindex.object().id().raw(),
                arena_dir_raw,
                arena_locs_raw,
            })?;

        let _ = namer.remove(&path);
        namer.put(&path, root.id())?;

        Ok(Graph {
            root_id: root.id(),
            verts,
            edges,
            labels,
            vindex,
            store,
        })
    }

    /// The graph root's ObjID.
    pub fn root_id(&self) -> ObjID {
        self.root_id
    }

    /// Reset a graph to empty, reusing its registration. No-op if no such graph
    /// is registered.
    ///
    /// This does not remove the `data/<name>` entry: removing a name under the
    /// persistent `data/` namespace is unsupported on the current Twizzler build
    /// (the pager's external unlink is unimplemented). Instead it rewrites the
    /// existing root object in place to point at fresh, empty registries. Old
    /// registry objects are orphaned; reclamation and true unregistration are
    /// future work.
    pub fn reset(name: &str) -> Result<()> {
        Self::reset_inner(name, None)
    }

    /// Like [`Graph::reset`], but the rebuilt graph uses the given registry
    /// segment capacity instead of keeping the existing one. No-op if no such
    /// graph is registered — pair it with
    /// [`Graph::open_or_create_with_capacity`] so both paths agree on `cap`.
    pub fn reset_with_capacity(name: &str, cap: usize) -> Result<()> {
        if cap == 0 || cap > u32::MAX as usize {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        Self::reset_inner(name, Some(cap))
    }

    /// Discard a graph and rebuild it empty, packing `arena_cap` vertices per
    /// arena object.
    ///
    /// Tests need this to be idempotent across runs: the disk image survives
    /// between QEMU invocations, so a graph left behind by an earlier run would
    /// otherwise be re-opened in whatever format it was written in. Unbinding
    /// the name is not an option — `data/` names cannot be removed on this
    /// build — so the root is rewritten in place, exactly as `reset` does.
    pub fn reset_arena(name: &str, arena_cap: usize) -> Result<()> {
        if arena_cap == 0 {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        Self::reset_inner_fmt(name, None, arena_cap)
    }

    fn reset_inner(name: &str, cap: Option<usize>) -> Result<()> {
        Self::reset_inner_fmt(name, cap, DEFAULT_ARENA_CAP)
    }

    /// The name is not unbound, because `data/` entries cannot be removed
    /// on this build. The root object is retained and its magic cleared, which
    /// makes it (a) unmistakably not a graph, so a later `open_or_create`
    /// refuses rather than reading freed ids, and (b) one leaked object per
    /// destroyed name instead of a whole graph. Re-using the name needs an
    /// explicit `reset`/`reset_arena`, which rebuilds in place.
    pub fn destroy(name: &str) -> Result<usize> {
        let mut namer = static_naming_factory().expect("naming service available");
        let path = format!("data/{name}");
        let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) else {
            return Ok(0); // nothing registered
        };
        let mut root = Object::<GraphRoot>::map(node.id.into(), rw())?;
        let (is_graph, version, cap, v, e, l, x, ad, al) = {
            let r = root.base();
            (
                r.magic == MAGIC,
                r.version,
                r.seg_cap as usize,
                r.verts_raw,
                r.edges_raw,
                r.labels_raw,
                r.vindex_raw,
                r.arena_dir_raw,
                r.arena_locs_raw,
            )
        };
        if !is_graph {
            // Already destroyed (magic is `MAGIC_DESTROYED`), or never ours.
            // Idempotent either way, and safe: a destroyed root's registry ids
            // are zeroed, so there is nothing left to chase.
            return Ok(0);
        }
        if !version_supported(version) || cap == 0 {
            // Refuse rather than delete objects named by a layout we cannot
            // read correctly.
            return Err(GraphError::StaleVersion {
                found: version,
                expected: VERSION_ARENA,
            });
        }

        // `version_supported` above admits only `VERSION_ARENA`, so there is one
        // layout to walk.
        let mut ids = {
            let mut ids = Vec::new();
            if let Ok(store) = ArenaStore::open(
                ad,
                al,
                Box::new(FillTo {
                    cap: DEFAULT_ARENA_CAP,
                }),
                cap,
            ) {
                ids.extend(store.owned_object_ids());
            }
            ids.extend(edge_registry_ids(e, cap));
            if let Ok(sv) = SegVec::<VertexRef>::open(v, cap) {
                ids.extend(sv.object_ids());
            }
            if let Ok(sv) = SegVec::<LabelEntry>::open(l, cap) {
                ids.extend(sv.object_ids());
            }
            ids.push(x);
            ids
        };
        // Guard against a double-free: an id reachable two ways (a shared
        // property object, say) would otherwise be deleted twice.
        ids.sort_unstable();
        ids.dedup();

        // Mark the root dead *before* freeing, so an interruption leaves a root
        // that refuses to open rather than one naming freed objects.
        root.with_tx(|tx| {
            let mut b = tx.base_mut();
            b.magic = MAGIC_DESTROYED;
            b.verts_raw = 0;
            b.edges_raw = 0;
            b.labels_raw = 0;
            b.vindex_raw = 0;
            b.arena_dir_raw = 0;
            b.arena_locs_raw = 0;
            Ok(())
        })?;

        Ok(reclaim::delete_all(ids))
    }

    /// Rebuild empty on VERSION 4, packing `arena_cap` vertices per arena.
    ///
    /// The *outgoing* graph may still be v3 — a disk image outlives the code
    /// that wrote it — so the inventory below keeps its v3 arm even though
    /// nothing creates v3 any more. That arm is what stops a stale image from
    /// leaking a graph's worth of objects on the first reset after the upgrade.
    fn reset_inner_fmt(name: &str, cap: Option<usize>, arena_cap: usize) -> Result<()> {
        let mut namer = static_naming_factory().expect("naming service available");
        let path = format!("data/{name}");
        let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) else {
            return Ok(()); // nothing registered
        };

        let mut root = Object::<GraphRoot>::map(node.id.into(), rw())?;
        // Only clobber something that is actually one of our graphs; trust the
        // stored capacity only if the root has the current layout.
        // A destroyed root is ours and rebuildable in place; only a root that
        // was never ours is refused. Without this, `destroy` would burn the
        // name permanently — the root survives in the disk image, so the next
        // boot inherits the refusal too.
        let (is_graph, old_version, old_cap) = {
            let r = root.base();
            (
                r.magic == MAGIC || r.magic == MAGIC_DESTROYED,
                r.version,
                r.seg_cap,
            )
        };
        let was_destroyed = root.base().magic == MAGIC_DESTROYED;
        if !is_graph {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        let cap = cap.unwrap_or(DEFAULT_SEG_CAP);

        let old_ids = {
            let (v, e, l, x, ad, al) = {
                let r = root.base();
                (
                    r.verts_raw,
                    r.edges_raw,
                    r.labels_raw,
                    r.vindex_raw,
                    r.arena_dir_raw,
                    r.arena_locs_raw,
                )
            };
            match old_version {
                // A destroyed root already freed everything and zeroed its
                // registry ids; walking them would chase freed objects.
                _ if was_destroyed => Vec::new(),
                // v4: arenas plus registries. The `verts` registry exists but
                // is unused, and is freed with the rest.
                VERSION_ARENA if old_cap != 0 => {
                    let cap = old_cap as usize;
                    let mut ids = Vec::new();
                    if let Ok(store) = ArenaStore::open(
                        ad,
                        al,
                        Box::new(FillTo {
                            cap: DEFAULT_ARENA_CAP,
                        }),
                        cap,
                    ) {
                        ids.extend(store.owned_object_ids());
                    }
                    ids.extend(edge_registry_ids(e, cap));
                    if let Ok(sv) = SegVec::<VertexRef>::open(v, cap) {
                        ids.extend(sv.object_ids());
                    }
                    if let Ok(sv) = SegVec::<LabelEntry>::open(l, cap) {
                        ids.extend(sv.object_ids());
                    }
                    ids.push(x);
                    ids
                }
                // An unrecognised format is left alone rather than guessed at:
                // deleting objects named by a layout we cannot read is how a
                // live graph gets destroyed.
                _ => Vec::new(),
            }
        };

        // Fresh, empty registries.
        let verts = SegVec::<VertexRef>::create(cap)?;
        let edges = SegVec::<EdgeRef>::create(cap)?;
        let labels = SegVec::<LabelEntry>::create(cap)?;
        let vindex = VIndex::new_persist()?;
        let (verts_raw, edges_raw, labels_raw, vindex_raw) = (
            verts.dir_raw(),
            edges.dir_raw(),
            labels.dir_raw(),
            vindex.object().id().raw(),
        );
        let store = ArenaStore::create(Box::new(FillTo { cap: arena_cap }), cap)?;
        let (arena_dir_raw, arena_locs_raw) = store.ids();

        // Rewrite the root transactionally so the change is synced to the
        // backing store; a raw write would be lost on reboot.
        root.with_tx(|tx| {
            let mut b = tx.base_mut();
            b.magic = MAGIC;
            b.version = VERSION_ARENA;
            b.seg_cap = cap as u32;
            b.verts_raw = verts_raw;
            b.edges_raw = edges_raw;
            b.labels_raw = labels_raw;
            b.vindex_raw = vindex_raw;
            b.arena_dir_raw = arena_dir_raw;
            b.arena_locs_raw = arena_locs_raw;
            Ok(())
        })?;

        reclaim::delete_all(old_ids);
        Ok(())
    }

    pub fn add_vertex(&mut self, label: &str, name: &str, target: ObjID) -> Result<VertexId> {
        let lbl = self.intern_label(label)?;

        // One arena allocation, shared with `cap-1` other vertices, instead of
        // the three whole objects v3 used (vertex + two adjacency `VecObject`s).
        let id = self.store.add_vertex(lbl, name, target.raw())?;
        self.vindex.insert(
            VKey {
                label: lbl,
                name: NameKey::new(name),
            },
            id,
        )?;
        Ok(VertexId(id))
    }

    /// Add a typed edge `from -> to`: append an adjacency entry to `from`'s
    /// outgoing chain and `to`'s incoming chain. The edge itself is only a
    /// registry row — v3's separate edge object is gone.
    pub fn add_edge(&mut self, from: VertexId, label: &str, to: VertexId) -> Result<EdgeId> {
        let lbl = self.intern_label(label)?;

        // No edge object at all. The adjacency entry carries the edge id and
        // label, and its `InvPtr` neighbour costs no FOT entry when both
        // endpoints share an arena.
        let id = self.edges.len() as u64;
        if !self.store.is_alive(from.0) || !self.store.is_alive(to.0) {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        self.store.add_edge(from.0, to.0, id, lbl)?;
        self.edges.push_nosync(EdgeRef {
            id,
            label: lbl,
            from_id: from.0,
            to_id: to.0,
            eobj_raw: 0, // no edge object on this layout
            props_raw: 0,
            flags: 0,
        })?;
        Ok(EdgeId(id))
    }

    /// Find a vertex by (label, name) via the persistent index.
    pub fn find_vertex(&self, label: &str, name: &str) -> Option<VertexId> {
        let lbl = self.find_label(label)?;
        let key = VKey {
            label: lbl,
            name: NameKey::new(name),
        };
        let id = VertexId(*self.vindex.get(&key)?);
        if self.is_vertex_alive(id) {
            Some(id)
        } else {
            None
        }
    }

    /// A traversal handle for a vertex, or `None` if it is deleted.
    pub fn vertex_view(&self, id: VertexId) -> Option<VertexView<'_>> {
        if !self.is_vertex_alive(id) {
            return None;
        }
        Some(VertexView { graph: self, id })
    }

    /// Convenience: outgoing/incoming/both neighbors of `id` (no predicate).
    pub fn out_neighbors(&self, id: VertexId, labels: Labels) -> Vec<VertexId> {
        self.arena_neighbors(id, labels, true, false)
    }
    pub fn in_neighbors(&self, id: VertexId, labels: Labels) -> Vec<VertexId> {
        self.arena_neighbors(id, labels, false, true)
    }
    pub fn both_neighbors(&self, id: VertexId, labels: Labels) -> Vec<VertexId> {
        self.arena_neighbors(id, labels, true, true)
    }

    // --- arena seams for the DSL (`vertex.rs`) ------------------------------

    /// `(edge_id, edge_label, neighbour_id)` in traversal order. `VertexView`
    /// uses this instead of mapping adjacency objects.
    pub(crate) fn arena_adjacency(
        &self,
        id: VertexId,
        out: bool,
        inc: bool,
    ) -> Vec<(u64, u32, u64)> {
        // The store hides tombstoned *vertices* but knows nothing about the
        // edge registry, so a deleted edge would still yield its neighbour.
        // Filtering it out has to happen here, since `Graph` owns the registry.
        // Missing it would diverge only on deletes — a bug that shows up as a
        // wrong query result long after the change that caused it.
        self.store
            .adjacency(id.0, out, inc)
            .into_iter()
            .filter(|(eid, _, _)| self.is_edge_alive(EdgeId(*eid)))
            .collect()
    }

    /// A neighbour's `(label, name)` for predicate evaluation — no `String`
    /// allocation, since most candidates are filtered out.
    pub(crate) fn arena_vertex_key(&self, id: u64) -> Option<(u32, NameKey)> {
        self.store.vertex_key(id)
    }

    /// An edge's label and endpoints from the registry. Used to build
    /// `EdgeHandle`s without an edge *object*, which this layout does not have.
    pub(crate) fn edge_endpoints(&self, e: EdgeId) -> Option<(u32, VertexId, VertexId)> {
        let r = self.edges.get_ref(e.0 as usize)?;
        if r.id != e.0 || r.flags & TOMBSTONE != 0 {
            return None;
        }
        Some((r.label, VertexId(r.from_id), VertexId(r.to_id)))
    }

    fn arena_neighbors(
        &self,
        id: VertexId,
        labels: Labels,
        out: bool,
        inc: bool,
    ) -> Vec<VertexId> {
        // Routed through `arena_adjacency` rather than the store's own
        // `neighbors_labeled` so the dead-edge filter applies here too — the
        // two must not have separate notions of which entries count.
        let filter = self.resolve_labels(labels);
        self.arena_adjacency(id, out, inc)
            .into_iter()
            .filter(|(_, l, _)| filter.as_ref().map_or(true, |ls| ls.contains(l)))
            .map(|(_, _, nb)| VertexId(nb))
            .collect()
    }

    /// All live vertex ids in the graph. Linear scan.
    pub fn vertices(&self) -> Vec<VertexId> {
        self.store.vertices().into_iter().map(VertexId).collect()
    }

    /// An edge's label and endpoints by id, or `None` if the edge is deleted.
    /// O(1): ids are append indices, so the record is at position `id`.
    pub fn edge_info(&self, id: EdgeId) -> Option<EdgeInfo> {
        if !self.is_edge_alive(id) {
            return None;
        }
        let r = self.edges.get_ref(id.0 as usize)?;
        Some(EdgeInfo {
            label: self.label_name(r.label).unwrap_or_default(),
            from: VertexId(r.from_id),
            to: VertexId(r.to_id),
        })
    }

    /// Delete a vertex (tombstone). Its incident edges become hidden too, since
    /// an edge is alive only while both endpoints are. No-op if already gone.
    ///
    /// v3 additionally freed the vertex's two adjacency objects and its property
    /// object here. v4 has no per-vertex adjacency objects at all, and the
    /// property object is deliberately left named by the tombstoned record: it
    /// is unreachable through the graph but still allocated, so
    /// `owned_object_ids` must keep reporting it or nothing will ever free it.
    pub fn delete_vertex(&mut self, id: VertexId) -> Result<()> {
        // `?` rather than a bare tail: the store speaks `TwzError`, the graph
        // API speaks `GraphError`.
        self.store.delete_vertex(id.0)?;
        Ok(())
    }

    /// Delete an edge (tombstone). No-op if already gone.
    pub fn delete_edge(&mut self, id: EdgeId) -> Result<()> {
        let idx = id.0 as usize;
        if idx >= self.edges.len() {
            return Ok(());
        }
        self.edges.with_mut_at(idx, |r| {
            if r.id == id.0 && r.flags & TOMBSTONE == 0 {
                r.flags |= TOMBSTONE;
            }
            Ok(())
        })?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn owned_object_ids(&self) -> Vec<u128> {
        let mut ids = Vec::new();
        ids.extend(self.store.owned_object_ids());
        // Edges: no object to map, and the property id lives in the mirror.
        ids.extend(self.edges.object_ids());
        for i in 0..self.edges.len() {
            if let Some(r) = self.edges.get_ref(i) {
                if r.props_raw != 0 {
                    ids.push(r.props_raw);
                }
            }
        }
        // `verts` is vestigial on v4 but still allocated, and still freed.
        ids.extend(self.verts.object_ids());
        ids.extend(self.labels.object_ids());
        ids.push(self.vindex.object().id().raw());
        ids
    }

    #[cfg(test)]
    pub(crate) fn vertex_props_raw(&self, v: VertexId) -> Option<u128> {
        self.store.props_raw(v.0)
    }

    #[cfg(test)]
    pub(crate) fn edge_props_raw(&self, e: EdgeId) -> Option<u128> {
        self.edges.get_ref(e.0 as usize).map(|r| r.props_raw)
    }

    /// Set a property on a vertex; errors if it is missing or tombstoned.
    /// Creates the vertex's property object on first use and records its id in
    /// the vertex's arena record.
    pub fn set_vertex_prop(&mut self, v: VertexId, key: &str, val: PropValue) -> Result<()> {
        if !self.is_vertex_alive(v) {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        let cur = self.store.props_raw(v.0).unwrap_or(0);
        let new_raw = props::set_in(cur, key, val)?;
        if new_raw != cur {
            self.store.set_props_raw(v.0, new_raw)?;
        }
        Ok(())
    }

    /// A vertex property, or `None` if unset or the vertex is dead.
    pub fn get_vertex_prop(&self, v: VertexId, key: &str) -> Option<PropValue> {
        if !self.is_vertex_alive(v) {
            return None;
        }
        props::get_in(self.store.props_raw(v.0)?, key)
    }

    /// All of a vertex's properties in insertion order (empty if dead/unset).
    pub fn vertex_props(&self, v: VertexId) -> Vec<(String, PropValue)> {
        if !self.is_vertex_alive(v) {
            return Vec::new();
        }
        self.store.props_raw(v.0).map_or(Vec::new(), props::list_in)
    }

    /// Set a property on an edge; errors if it is missing, tombstoned, or has
    /// a dead endpoint. The property-object id lives in the edge registry,
    /// since there is no edge object to hold it.
    pub fn set_edge_prop(&mut self, e: EdgeId, key: &str, val: PropValue) -> Result<()> {
        if !self.is_edge_alive(e) {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        let idx = e.0 as usize;
        let cur = self.edges.get_ref(idx).map(|r| r.props_raw).unwrap_or(0);
        let new_raw = props::set_in(cur, key, val)?;
        if new_raw != cur {
            self.edges.with_mut_at(idx, |r| {
                r.props_raw = new_raw;
                Ok(())
            })?;
        }
        Ok(())
    }

    /// An edge property, or `None` if unset or the edge is dead.
    pub fn get_edge_prop(&self, e: EdgeId, key: &str) -> Option<PropValue> {
        if !self.is_edge_alive(e) {
            return None;
        }
        props::get_in(self.edges.get_ref(e.0 as usize)?.props_raw, key)
    }

    /// All of an edge's properties in insertion order (empty if dead/unset).
    pub fn edge_props(&self, e: EdgeId) -> Vec<(String, PropValue)> {
        if !self.is_edge_alive(e) {
            return Vec::new();
        }
        match self.edges.get_ref(e.0 as usize) {
            Some(r) => props::list_in(r.props_raw),
            None => Vec::new(),
        }
    }

    /// Whether a vertex exists and is not tombstoned.
    pub(crate) fn is_vertex_alive(&self, id: VertexId) -> bool {
        self.store.is_alive(id.0)
    }

    /// Make the graph durable — one sync per arena plus the registries.
    pub fn sync(&mut self) -> Result<()> {
        self.store.sync_all()?;
        self.edges.flush()?;
        self.labels.flush()?;
        Ok(())
    }

    /// Whether an edge exists, is not tombstoned, and both endpoints are alive.
    pub(crate) fn is_edge_alive(&self, id: EdgeId) -> bool {
        let Some(r) = self.edges.get_ref(id.0 as usize) else {
            return false;
        };
        if r.id != id.0 || r.flags & TOMBSTONE != 0 {
            return false;
        }
        self.is_vertex_alive(VertexId(r.from_id)) && self.is_vertex_alive(VertexId(r.to_id))
    }

    /// All vertices with the given label. Linear scan.
    pub fn vertices_by_label(&self, label: &str) -> Vec<VertexId> {
        let Some(lbl) = self.find_label(label) else {
            return Vec::new();
        };
        self.store
            .vertices_by_label(lbl)
            .into_iter()
            .map(VertexId)
            .collect()
    }

    /// Read back a vertex's data from the registry, or `None` if it is deleted.
    /// O(1): ids are append indices, so the record is at position `id`.
    pub fn vertex_info(&self, id: VertexId) -> Option<VertexInfo> {
        let (lbl, name, target) = self.store.vertex_info(id.0)?;
        Some(VertexInfo {
            label: self.label_name(lbl).unwrap_or_default(),
            name,
            target: ObjID::new(target),
        })
    }

    /// Diagnostic, temporary — pass-through to
    /// [`ArenaStore::debug_liveness`].
    pub fn debug_liveness(&self, id: VertexId) -> Option<String> {
        Some(self.store.debug_liveness(id.0))
    }

    /// Resolve a [`Labels`] filter to label ids. `None` means "any".
    pub(crate) fn resolve_labels(&self, labels: Labels) -> Option<Vec<u32>> {
        match labels {
            Labels::Any => None,
            Labels::These(names) => {
                let mut ids = Vec::new();
                for n in names {
                    if let Some(id) = self.find_label(n) {
                        ids.push(id);
                    }
                }
                Some(ids)
            }
        }
    }

    // --- private lookup helpers ---

    /// O(1): ids are append indices, so the record is at position `id`.

    // The content-keyed lookups below are linear scans (where a hachage index
    // would later go): find_label by name, find_vertex by (label, name), and
    // vertices_by_label.

    fn find_label(&self, name: &str) -> Option<u32> {
        find_label_in(&self.labels, name)
    }

    fn label_name(&self, id: u32) -> Option<String> {
        for i in 0..self.labels.len() {
            let e = self.labels.get_ref(i)?;
            if e.id == id {
                return Some(e.name.as_str().to_string());
            }
        }
        None
    }

    fn intern_label(&mut self, name: &str) -> Result<u32> {
        if let Some(id) = self.find_label(name) {
            return Ok(id);
        }
        let id = self.labels.len() as u32;
        self.labels.push(LabelEntry {
            id,
            name: NameKey::new(name),
        })?;
        Ok(id)
    }

}

// --- shared lookup helpers -------------------------------------------------

/// Label lookup by name over the label registry.
fn find_label_in(labels: &SegVec<LabelEntry>, name: &str) -> Option<u32> {
    for i in 0..labels.len() {
        let e = labels.get_ref(i)?;
        if e.name.eq_str(name) {
            return Some(e.id);
        }
    }
    None
}

/// The edge registry's own objects plus every edge property object.
///
/// v4 keeps `props_raw` in `EdgeRef`, so this needs no edge *objects* — which
/// is why it is separate from [`inventory`], whose v3 form has to map each edge
/// object to find the same id.
fn edge_registry_ids(edges_raw: u128, cap: usize) -> Vec<u128> {
    let Ok(edges) = SegVec::<EdgeRef>::open(edges_raw, cap) else {
        return Vec::new();
    };
    let mut ids = edges.object_ids();
    for i in 0..edges.len() {
        if let Some(r) = edges.get_ref(i) {
            if r.props_raw != 0 {
                ids.push(r.props_raw);
            }
        }
    }
    ids
}
