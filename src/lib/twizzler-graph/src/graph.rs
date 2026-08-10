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
    arena_store::{ArenaStore, FillTo, PropSlot},
    edge::{EdgeId, EdgeInfo},
    error::{GraphError, Result},
    name::NameKey,
    props::PropValue,
    reclaim,
    segvec::SegVec,
    vertex::{Labels, VertexId, VertexInfo, VertexView},
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

/// On-disk format 8 (the arena layout): vertices and adjacency live in
/// packed arenas ([`ArenaStore`]) instead of three objects per vertex plus one
/// per edge.
///
/// Reading a format-8 graph with this build would interpret padding as a
/// generation, mismatch every adjacency entry, and silently return a graph with
/// no edges. Clear the disk image.
pub(crate) const VERSION_ARENA: u32 = 11;

/// No longer reclaimable as of format 9. It was, while 8 differed from 7
/// only in a trailing root field; format 9 moved the *record* layout, and the
/// inventory walk reads records. Kept for the same reason as [`VERSION`]: the
/// number is burned and must never be recycled.
#[allow(dead_code)] // reserved, not obsolete — see above
pub(crate) const VERSION_ARENA_NOCAP: u32 = 7;

/// Whether this build can *operate* on a graph in the given on-disk format —
/// read it, write it, hand it to a caller.
///
/// Strict on purpose: misreading a layout yields garbage rather than an error.
fn version_supported(v: u32) -> bool {
    v == VERSION_ARENA
}

/// Whether this build can *free* a graph in the given format — deliberately
/// broader than [`version_supported`].
///
/// Reading and reclaiming are different questions. Reading needs every field
/// to mean what the code thinks it means. Reclaiming only needs to find the
/// object ids, so a predecessor format qualifies whenever its *object graph* is
/// unchanged, whatever happened to the interpretation of individual records.
fn version_reclaimable(v: u32) -> bool {
    // This is the rule from the 7 → 8 mistake applied in the other direction:
    // extend `version_reclaimable` when placement is untouched, and *don't*
    // when it isn't.
    v == VERSION_ARENA
}

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
/// Memory is not the constraint — per-vertex overhead is `1454/cap` frames,
/// so this raise takes it from ~0.36 to ~0.089.
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
    pub(crate) labels_raw: u128,
    pub(crate) vindex_raw: u128,
    /// The [`ArenaStore`]'s arena directory and location registry.
    pub(crate) arena_dir_raw: u128,
    pub(crate) arena_locs_raw: u128,
    /// Read on open in preference to the caller's argument. `seg_cap` above is
    /// the precedent: geometry that must stay uniform for the graph's lifetime
    /// belongs in the root.
    pub(crate) arena_cap: u32,
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
    labels: SegVec<LabelEntry>,
    vindex: VIndex,
    /// Vertices and adjacency — the whole graph, in packed arenas.
    store: ArenaStore,
}

impl Graph {

    pub fn record_count(&self) -> usize {
        self.store.record_count()
    }

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
        if arena_cap == 0 || arena_cap > u32::MAX as usize {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        let mut namer = static_naming_factory().expect("naming service available");
        let path = format!("data/{name}");

        if let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) {
            let root =
                Object::<GraphRoot>::map(node.id.into(), MapFlags::READ | MapFlags::PERSIST)?;
            let (magic, version, seg_cap, labels_raw, vindex_raw) = {
                let r = root.base();
                (
                    r.magic,
                    r.version,
                    r.seg_cap,
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
                    let (dir, locs, stored_cap) = {
                        let r = root.base();
                        (r.arena_dir_raw, r.arena_locs_raw, r.arena_cap)
                    };
                    let cap_for_placement = if stored_cap == 0 {
                        DEFAULT_ARENA_CAP
                    } else {
                        stored_cap as usize
                    };
                    ArenaStore::open(
                        dir,
                        locs,
                        Box::new(FillTo {
                            cap: cap_for_placement,
                        }),
                        cap,
                    )?
                };
                return Ok(Graph {
                    root_id: node.id.into(),
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
                labels_raw: labels.dir_raw(),
                vindex_raw: vindex.object().id().raw(),
                arena_dir_raw,
                arena_locs_raw,
                arena_cap: arena_cap as u32,
            })?;

        let _ = namer.remove(&path);
        namer.put(&path, root.id())?;

        Ok(Graph {
            root_id: root.id(),
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
        if arena_cap == 0 || arena_cap > u32::MAX as usize {
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
        let (is_graph, version, cap, l, x, ad, al) = {
            let r = root.base();
            (
                r.magic == MAGIC,
                r.version,
                r.seg_cap as usize,
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
        // `version_reclaimable`, not `version_supported`: destroy's job is to
        // free what the graph owns, and that only needs the object graph to be
        // walkable. Refusing here on a merely-outdated format would strand the
        // graph forever — the name cannot be unbound on this build, so nothing
        // would ever come back to free it.
        if !version_reclaimable(version) || cap == 0 {
            // Still refuse a format whose objects we cannot locate: deleting ids
            // read out of a layout we do not understand is how a live graph gets
            // destroyed.
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
            b.labels_raw = 0;
            b.vindex_raw = 0;
            b.arena_dir_raw = 0;
            b.arena_locs_raw = 0;
            b.arena_cap = 0;
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
            let (l, x, ad, al) = {
                let r = root.base();
                (
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
                // Arenas, the label registry and the index. Matches any
                // *reclaimable* format, not just the current one — see
                // `version_reclaimable`.
                ver if version_reclaimable(ver) && old_cap != 0 => {
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
        let labels = SegVec::<LabelEntry>::create(cap)?;
        let vindex = VIndex::new_persist()?;
        let (labels_raw, vindex_raw) = (
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
            b.labels_raw = labels_raw;
            b.vindex_raw = vindex_raw;
            b.arena_dir_raw = arena_dir_raw;
            b.arena_locs_raw = arena_locs_raw;
            b.arena_cap = arena_cap as u32;
            Ok(())
        })?;

        reclaim::delete_all(old_ids);
        Ok(())
    }

    /// The set supplied here is fixed for the record's lifetime, because the
    /// record's size is: every inbound `AdjRef.neighbor` holds its arena offset,
    /// so a record that grew would have to have all of them rewritten. Anything
    /// added later via [`Graph::set_vertex_prop`] becomes a *data* property,
    /// which lives behind one indirection and can move freely.
    ///
    /// Choosing is the user's job, and the criterion is access pattern: put
    /// a property here if traversals *filter* on it, since inline slots sit in
    /// cache lines a walk has already paid for. Put everything else in data
    /// properties — inline slots widen every record, and record width is what
    /// sets page density.
    ///
    /// Keys are interned through the same table as labels. They cannot collide:
    /// a label id is read from `record.label` and a key id from `slot.key_id`,
    /// which are different fields consulted in different contexts.
    pub fn add_vertex_with_props(
        &mut self,
        label: &str,
        name: &str,
        target: ObjID,
        props: &[(&str, PropValue)],
    ) -> Result<VertexId> {
        let lbl = self.intern_label(label)?;
        let mut slots = Vec::with_capacity(props.len());
        for (k, v) in props {
            slots.push(PropSlot {
                key_id: self.intern_label(k)?,
                _pad: 0,
                val: *v,
            });
        }
        let id = self
            .store
            .add_record(lbl, name, target.raw(), &slots, false)?;
        self.vindex.insert(
            VKey {
                label: lbl,
                name: NameKey::new(name),
            },
            id,
        )?;
        Ok(VertexId(id))
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

        let id = self.store.add_edge_record(lbl, from.0, to.0)?;
        Ok(EdgeId(id))
    }

    /// Attach a further participant to an existing edge, making it a hyperedge.
    ///
    /// `out = true` adds `vertex` as another target of `edge`, `false` as
    /// another source. Edges are records, so this needs no machinery beyond one
    /// more link in each direction.
    pub fn add_edge_endpoint(&mut self, edge: EdgeId, vertex: VertexId, out: bool) -> Result<()> {
        self.store.add_edge_endpoint(edge.0, vertex.0, out)?;
        Ok(())
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
        // Worth stating because deleting a filter is exactly the change that
        // looks like a regression later: the guarantee moved rather than went
        // away, and `deleting_an_edge_agrees` in `graph-eval` is what would
        // notice if it had gone away.
        //
        // It is also a small win. The old filter ran `is_edge_alive` per entry,
        // which now resolves records; keeping it would have put that on the hot
        // path for no benefit.
        self.store.neighbors_via_edges(id.0, out, inc)
    }

    /// A neighbour's `(label, name)` for predicate evaluation — no `String`
    /// allocation, since most candidates are filtered out.
    pub(crate) fn arena_vertex_key(&self, id: u64) -> Option<(u32, NameKey)> {
        self.store.vertex_key(id)
    }

    /// An edge's label and endpoints, read from its record. Used to build
    /// `EdgeHandle`s.
    ///
    /// A hyperedge has several of each endpoint; this returns the first, which
    /// is what an `EdgeHandle` can represent. Callers needing the general shape
    /// walk the edge's chains instead.
    pub(crate) fn edge_endpoints(&self, e: EdgeId) -> Option<(u32, VertexId, VertexId)> {
        let label = self.store.vertex_label(e.0)?;
        let (from, to) = self.store.edge_endpoints(e.0)?;
        Some((label, VertexId(from), VertexId(to)))
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

    /// An edge's label and endpoints by id, or `None` if it is deleted or is
    /// not an edge.
    ///
    /// That second case is new with the unified id space and is the runtime
    /// check replacing what the type system used to give us: `EdgeId` and
    /// `VertexId` are the same type now, so "edge passed where a vertex belongs"
    /// cannot be rejected at compile time. `IS_EDGE` does the rejecting instead,
    /// and `edge_info` on a vertex record must return `None`.
    pub fn edge_info(&self, id: EdgeId) -> Option<EdgeInfo> {
        if !self.is_edge_alive(id) {
            return None;
        }
        let (label, from, to) = self.edge_endpoints(id)?;
        Some(EdgeInfo {
            label: self.label_name(label).unwrap_or_default(),
            from,
            to,
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
        if !self.store.is_edge(id.0) {
            return Ok(()); // not an edge record: no-op, as for an unknown id
        }
        self.store.delete_vertex(id.0)?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn owned_object_ids(&self) -> Vec<u128> {
        let mut ids = Vec::new();
        ids.extend(self.store.owned_object_ids());
        // Edges: no object to map, and the property id lives in the mirror.
        // `verts` is vestigial on v4 but still allocated, and still freed.
        ids.extend(self.labels.object_ids());
        ids.push(self.vindex.object().id().raw());
        ids
    }

    #[cfg(test)]
    pub(crate) fn record_touches(&self) -> usize {
        self.store.record_touches()
    }

    #[cfg(test)]
    pub(crate) fn reset_record_touches(&self) {
        self.store.reset_record_touches();
    }

    #[cfg(test)]
    pub(crate) fn vertex_props_raw(&self, v: VertexId) -> Option<u128> {
        self.store.live_record(v.0).map(|_| 0)
    }

    #[cfg(test)]
    pub(crate) fn data_block_reads(&self) -> usize {
        self.store.data_block_reads()
    }

    #[cfg(test)]
    pub(crate) fn edge_props_raw(&self, e: EdgeId) -> Option<u128> {
        self.store.live_record(e.0).map(|_| 0)
    }

    /// Set a property on a vertex; errors if it is missing or tombstoned.
    /// Creates the vertex's property object on first use and records its id in
    /// the vertex's arena record.
    pub fn set_vertex_prop(&mut self, v: VertexId, key: &str, val: PropValue) -> Result<()> {
        if !self.is_vertex_alive(v) {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        // If the key is already an inline traversal slot, update it there.
        // Writing the side object instead would leave two values for one key,
        // with readers preferring the stale inline one — a divergence invisible
        // from outside, since a wrong property reads exactly like a right one.
        // Updating in place is sound because a slot is fixed-size; it is
        // *adding* a key that the format forbids, not changing one.
        let key_id = self.intern_label(key)?;
        if self.store.set_traversal_prop(v.0, key_id, val) == Some(true) {
            return Ok(());
        }
        self.store.set_data_prop(v.0, key_id, val)?;
        Ok(())
    }

    /// A vertex property, or `None` if unset or the vertex is dead.
    pub fn get_vertex_prop(&self, v: VertexId, key: &str) -> Option<PropValue> {
        if !self.is_vertex_alive(v) {
            return None;
        }
        let key_id = self.find_label(key)?;
        if let Some(val) = self.store.traversal_prop(v.0, key_id) {
            return Some(val);
        }
        self.store
            .data_props(v.0)?
            .into_iter()
            .find(|s| s.key_id == key_id)
            .map(|s| s.val)
    }

    /// All of a vertex's properties (empty if dead/unset).
    ///
    /// Inline traversal properties first, in slot order, then data properties in
    /// insertion order. The two sets are disjoint by construction.
    pub fn vertex_props(&self, v: VertexId) -> Vec<(String, PropValue)> {
        if !self.is_vertex_alive(v) {
            return Vec::new();
        }
        let mut slots = self.store.traversal_props(v.0).unwrap_or_default();
        slots.extend(self.store.data_props(v.0).unwrap_or_default());
        slots
            .into_iter()
            .filter_map(|s| self.label_name(s.key_id).map(|k| (k, s.val)))
            .collect()
    }

    /// Set a property on an edge; errors if it is missing, tombstoned, or has
    /// a dead endpoint. The property-object id lives in the edge registry,
    /// since there is no edge object to hold it.
    pub fn set_edge_prop(&mut self, e: EdgeId, key: &str, val: PropValue) -> Result<()> {
        if !self.is_edge_alive(e) {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        self.set_vertex_prop(VertexId(e.0), key, val)
    }

    /// An edge property, or `None` if unset or the edge is dead.
    pub fn get_edge_prop(&self, e: EdgeId, key: &str) -> Option<PropValue> {
        if !self.is_edge_alive(e) {
            return None;
        }
        self.get_vertex_prop(VertexId(e.0), key)
    }

    /// All of an edge's properties in insertion order (empty if dead/unset).
    pub fn edge_props(&self, e: EdgeId) -> Vec<(String, PropValue)> {
        if !self.is_edge_alive(e) {
            return Vec::new();
        }
        self.vertex_props(VertexId(e.0))
    }

    /// Whether a vertex exists and is not tombstoned.
    pub(crate) fn is_vertex_alive(&self, id: VertexId) -> bool {
        self.store.is_alive(id.0)
    }

    /// Make the graph durable — one sync per arena plus the registries.
    pub fn sync(&mut self) -> Result<()> {
        self.store.sync_all()?;
        self.labels.flush()?;
        Ok(())
    }

    /// Whether `id` names a live edge record whose endpoints are both live.
    ///
    /// The `is_edge` test is what keeps the unified id space honest: without it
    /// every live vertex would answer "yes" and `edge_info`/`get_edge_prop`
    /// would happily treat a vertex as an edge.
    pub(crate) fn is_edge_alive(&self, id: EdgeId) -> bool {
        if !self.store.is_edge(id.0) || !self.store.is_alive(id.0) {
            return false;
        }
        // An edge is alive only while both endpoints are — unchanged semantics,
        // now read from the edge's own chains rather than a registry mirror.
        match self.store.edge_endpoints(id.0) {
            Some((f, t)) => self.store.is_alive(f) && self.store.is_alive(t),
            None => false,
        }
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
        // An edge record is not a vertex. The mirror of the `IS_EDGE` check
        // in `edge_info`, and it was missing: with one id space the type system
        // no longer separates the two, so every accessor has to reject the
        // wrong kind at runtime or it will happily describe an edge as a vertex.
        // `gstress verify` found this by reading an id that had silently become
        // an edge's.
        if self.store.is_edge(id.0) {
            return None;
        }
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

#[cfg(test)]
mod tests {
    //! Unit tests for the format guards and the geometry validation.
    //!
    //! Pure functions and early-return argument checks only — nothing here
    //! creates an object, so these cost the shared boot nothing. The
    //! behavioural counterparts live in `tests/arena_graph.rs`.

    use super::*;

    /// Reading is gated strictly: exactly one format, and no predecessor.
    #[test]
    fn only_the_current_format_is_readable() {
        assert!(version_supported(VERSION_ARENA));
        assert!(!version_supported(VERSION_ARENA_NOCAP), "v7 is not readable");
        assert!(!version_supported(VERSION), "v3 is not readable");
        // Every number *except* the current one. Written as a filter over a
        // range rather than a literal list, which is how `9` ended up in a
        // "must not be readable" list on the very build that made 9 current.
        for v in (0..=32u32).filter(|v| *v != VERSION_ARENA) {
            assert!(!version_supported(v), "version {v} must not be readable");
        }
    }

    /// Freeing is gated on whether the *object graph* is walkable — a weaker
    /// condition than readability, but not a free pass.
    ///
    /// Currently no predecessor qualifies, and that is a statement about
    /// format 9 rather than a permanent one: 7 qualified while 8 was current,
    /// because 7 → 8 moved only a trailing root field. The rule is what to
    /// assert, not the membership.
    #[test]
    fn reclaimability_tracks_whether_records_are_still_walkable() {
        assert!(version_reclaimable(VERSION_ARENA));
        // Format 7 was reclaimable while 8 was current, because 7 → 8 touched
        // only a trailing root field. Format 9 moved the record layout, and the
        // inventory walk reads records — so 7 dropped out, deliberately.
        assert!(
            !version_reclaimable(VERSION_ARENA_NOCAP),
            "a format whose record layout we can no longer read must not be \
             walked for object ids: leaking beats mis-freeing"
        );
    }

    #[test]
    fn v3_is_not_reclaimable() {
        assert!(!version_reclaimable(VERSION));
        for v in (0..=32u32).filter(|v| *v != VERSION_ARENA) {
            assert!(!version_reclaimable(v), "version {v} must not be freed");
        }
    }

    /// The invariant that the 7 → 8 bump broke. Anything this build can
    /// read, it must also be able to free — otherwise opening a graph and then
    /// resetting it leaks the very objects it was just using. Asserting the
    /// relationship rather than two enumerations is what makes this survive the
    /// next bump: a new format added to `version_supported` alone fails here.
    #[test]
    fn everything_readable_is_also_reclaimable() {
        for v in 0..=32u32 {
            assert!(
                !version_supported(v) || version_reclaimable(v),
                "version {v} is readable but not reclaimable — a reset would \
                 leak a graph this build had open"
            );
        }
    }

    #[test]
    fn arena_cap_is_rejected_outside_the_persisted_range() {
        let too_big = u32::MAX as usize + 1;
        assert!(Graph::open_or_create_arena("t-cap-guard", 0).is_err());
        assert!(Graph::open_or_create_arena("t-cap-guard", too_big).is_err());
        assert!(Graph::reset_arena("t-cap-guard", 0).is_err());
        assert!(Graph::reset_arena("t-cap-guard", too_big).is_err());
        assert!(Graph::open_or_create_arena_with_capacity("t-cap-guard", 4, 0).is_err());
    }

    /// The registry segment capacity has the same persisted-as-`u32` constraint.
    #[test]
    fn seg_cap_is_rejected_outside_the_persisted_range() {
        assert!(Graph::open_or_create_with_capacity("t-seg-guard", 0).is_err());
        assert!(
            Graph::open_or_create_with_capacity("t-seg-guard", u32::MAX as usize + 1).is_err()
        );
    }
}
