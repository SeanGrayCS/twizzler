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
//! `vertices_by_label` is still a scan. Adjacency is held per-vertex (see
//! `vertex.rs`), not in the registries.

use std::collections::HashMap;

use naming::{static_naming_factory, GetFlags};
use twizzler::{
    collections::{
        hachage::{PHMsession, PersistentHashMap, PersistentHashMapBase},
        vec::{Vec as TwzVec, VecObject, VecObjectAlloc},
    },
    error::TwzError,
    marker::{BaseType, Invariant},
    object::{MapFlags, ObjID, Object, ObjectBuilder, TypedObject},
    ptr::InvPtr,
};
use twizzler_rt_abi::error::ArgumentError;

use crate::{
    arena_store::{ArenaStore, FillTo},
    edge::{Edge, EdgeId, EdgeInfo, EdgeRef},
    error::{GraphError, Result},
    name::NameKey,
    props::{self, PropValue},
    reclaim,
    segvec::{vec_new_nosync, vec_push_ctor_nosync, SegVec},
    vertex::{AdjEntry, Labels, Vertex, VertexId, VertexInfo, VertexRef, VertexView},
};

pub(crate) const MAGIC: u64 = 0x4731_5457_5A47_5248; // "G1TWZGRH"

/// On-disk format 5: segmented registries, one vertex object plus two adjacency
/// objects per vertex, one object per edge.
///
/// The disk image survives between QEMU runs, which is what turns "I changed a
/// struct" into "the next boot hangs". Any change to a persisted record's
/// layout — size, field order, or alignment — must bump this, so the guard
/// rejects the old graph loudly instead of misreading it.
pub(crate) const VERSION: u32 = 5;

/// On-disk format 6: vertices and adjacency live in packed arenas
/// ([`ArenaStore`]) instead of three objects per vertex plus one per edge.
pub(crate) const VERSION_ARENA: u32 = 6;

/// Whether this build understands a graph in the given on-disk format.
fn version_supported(v: u32) -> bool {
    v == VERSION || v == VERSION_ARENA
}
const TOMBSTONE: u32 = 1; // `flags` bit 0: record is deleted

/// Default per-segment registry capacity. Registry records are plain data
/// (~100–150 B, no `InvPtr`s), so 4096 entries keep a segment well under the
/// object size limit while amortizing segment creation.
pub(crate) const DEFAULT_SEG_CAP: usize = 4096;

pub const DEFAULT_ARENA_CAP: usize = 4096;

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

/// An open graph: the root id, the registries, and the vertex index.
pub struct Graph {
    root_id: ObjID,
    verts: SegVec<VertexRef>,
    edges: SegVec<EdgeRef>,
    labels: SegVec<LabelEntry>,
    vindex: VIndex,
    /// `Some` on a VERSION 4 graph: vertices and adjacency live here instead of
    /// in `verts` and per-vertex objects. When set, `verts` stays empty and
    /// every vertex path defers to the store.
    store: Option<ArenaStore>,
}

impl Graph {
    /// Whether this graph uses the arena layout (VERSION 4).
    pub fn is_arena(&self) -> bool {
        self.store.is_some()
    }

    pub fn arena_count(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.arena_count())
    }

    pub fn arena_sync_count(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.sync_count())
    }
}

impl Graph {
    /// Open the graph registered at `data/<name>`, or create and register a
    /// fresh one. If an existing graph has an incompatible format
    /// (magic/version mismatch), this returns [`GraphError::StaleVersion`] and
    /// leaves the existing graph intact; use [`Graph::reset`] to discard it.
    pub fn open_or_create(name: &str) -> Result<Graph> {
        Self::open_or_create_with_capacity(name, DEFAULT_SEG_CAP)
    }

    /// Like [`Graph::open_or_create`], with an explicit registry segment
    /// capacity. The capacity is used only when *creating* a graph; an
    /// existing graph always keeps the capacity recorded in its root, since
    /// segment geometry must stay uniform for the graph's lifetime. (Small
    /// capacities let tests force segment rollover cheaply.)
    pub fn open_or_create_with_capacity(name: &str, cap: usize) -> Result<Graph> {
        Self::open_inner(name, cap, None)
    }

    /// Open or create a graph on the arena layout (VERSION 4), packing
    /// `arena_cap` vertices per object.
    pub fn open_or_create_arena(name: &str, arena_cap: usize) -> Result<Graph> {
        Self::open_inner(name, DEFAULT_SEG_CAP, Some(arena_cap))
    }

    /// [`Graph::open_or_create_arena`] with an explicit registry segment
    /// capacity, for tests that force segment rollover cheaply.
    pub fn open_or_create_arena_with_capacity(
        name: &str,
        cap: usize,
        arena_cap: usize,
    ) -> Result<Graph> {
        Self::open_inner(name, cap, Some(arena_cap))
    }

    /// `arena` is `Some(cap)` to *create* a VERSION 4 graph; it does not affect
    /// opening an existing one.
    fn open_inner(name: &str, cap: usize, arena: Option<usize>) -> Result<Graph> {
        if cap == 0 || cap > u32::MAX as usize {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        if matches!(arena, Some(0)) {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        // Packing is what the arena layout buys, so an arena graph reopened
        // without a stated cap still needs one for future inserts; the store's
        // existing contents are unaffected by the choice.
        let arena_cap = arena.unwrap_or(DEFAULT_ARENA_CAP);
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
                // The *stored* version decides the layout, not the caller: a
                // graph opened by name must come back the way it was written.
                let store = if version == VERSION_ARENA {
                    let (dir, locs) = {
                        let r = root.base();
                        (r.arena_dir_raw, r.arena_locs_raw)
                    };
                    Some(ArenaStore::open(
                        dir,
                        locs,
                        Box::new(FillTo { cap: arena_cap }),
                        cap,
                    )?)
                } else {
                    None
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
            return Err(GraphError::StaleVersion {
                found: version,
                expected: VERSION,
            });
        }

        let verts = SegVec::create(cap)?;
        let edges = SegVec::create(cap)?;
        let labels = SegVec::create(cap)?;
        let vindex = VIndex::new_persist()?;
        let store = match arena {
            Some(_) => Some(ArenaStore::create(
                Box::new(FillTo { cap: arena_cap }),
                cap,
            )?),
            None => None,
        };
        let (arena_dir_raw, arena_locs_raw) =
            store.as_ref().map_or((0, 0), |s| s.ids());

        let root = ObjectBuilder::<GraphRoot>::default()
            .persist(true)
            .build(GraphRoot {
                magic: MAGIC,
                version: if arena.is_some() {
                    VERSION_ARENA
                } else {
                    VERSION
                },
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

    #[cfg(test)]
    pub(crate) fn registry_segments(&self) -> (usize, usize, usize) {
        (
            self.verts.segments(),
            self.edges.segments(),
            self.labels.segments(),
        )
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

    /// Discard a graph and rebuild it empty on the arena layout
    /// (VERSION 4), packing `arena_cap` vertices per object.
    ///
    /// Arena tests need this to be idempotent across runs: the disk image
    /// survives between QEMU invocations, so a graph left behind by an earlier
    /// run would otherwise be re-opened in whatever format it was written in,
    /// and a plain [`Graph::reset`] would rebuild it as v3. Unbinding the name
    /// is not an option — `data/` names cannot be removed on this build — so
    /// the root is rewritten in place, exactly as `reset` does.
    pub fn reset_arena(name: &str, arena_cap: usize) -> Result<()> {
        if arena_cap == 0 {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        Self::reset_inner_fmt(name, None, Some(arena_cap))
    }

    fn reset_inner(name: &str, cap: Option<usize>) -> Result<()> {
        Self::reset_inner_fmt(name, cap, None)
    }

    /// `arena` is `Some(cap)` to rebuild on VERSION 4, `None` for VERSION 3.
    fn reset_inner_fmt(name: &str, cap: Option<usize>, arena: Option<usize>) -> Result<()> {
        let mut namer = static_naming_factory().expect("naming service available");
        let path = format!("data/{name}");
        let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) else {
            return Ok(()); // nothing registered
        };

        let mut root = Object::<GraphRoot>::map(node.id.into(), rw())?;
        // Only clobber something that is actually one of our graphs; trust the
        // stored capacity only if the root has the current layout.
        let (is_graph, old_version, old_cap) = {
            let r = root.base();
            (r.magic == MAGIC, r.version, r.seg_cap)
        };
        if !is_graph {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        let cap = cap.unwrap_or(if old_version == VERSION && old_cap != 0 {
            old_cap as usize
        } else {
            DEFAULT_SEG_CAP
        });

        // Re-enable this together with the delete, not before: the inventory is
        // only worth its page-in cost if something frees what it finds.
        let old_ids: Vec<u128> = Vec::new();

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
        let store = match arena {
            Some(ac) => Some(ArenaStore::create(Box::new(FillTo { cap: ac }), cap)?),
            None => None,
        };
        let (arena_dir_raw, arena_locs_raw) = store.as_ref().map_or((0, 0), |s| s.ids());

        // Rewrite the root transactionally so the change is synced to the
        // backing store; a raw write would be lost on reboot.
        root.with_tx(|tx| {
            let mut b = tx.base_mut();
            b.magic = MAGIC;
            b.version = if arena.is_some() {
                VERSION_ARENA
            } else {
                VERSION
            };
            b.seg_cap = cap as u32;
            b.verts_raw = verts_raw;
            b.edges_raw = edges_raw;
            b.labels_raw = labels_raw;
            b.vindex_raw = vindex_raw;
            // Zero on a v3 rebuild: a v3 root must not name arenas it does not
            // own, and a previous v4 incarnation would have left ids here.
            b.arena_dir_raw = arena_dir_raw;
            b.arena_locs_raw = arena_locs_raw;
            Ok(())
        })?;

        let _ = old_ids;
        Ok(())
    }

    pub fn add_vertex(&mut self, label: &str, name: &str, target: ObjID) -> Result<VertexId> {
        let lbl = self.intern_label(label)?;

        // VERSION 4: one arena allocation, shared with `cap-1` other vertices,
        // instead of three whole objects (vertex + two adjacency `VecObject`s).
        if self.store.is_some() {
            let id = {
                let store = self.store.as_mut().expect("arena graph");
                store.add_vertex(lbl, name, target.raw())?
            };
            self.vindex.insert(
                VKey {
                    label: lbl,
                    name: NameKey::new(name),
                },
                id,
            )?;
            return Ok(VertexId(id));
        }

        let id = self.verts.len() as u64;

        // Per-vertex adjacency lists, split by direction.
        let out_adj: VecObject<AdjEntry, VecObjectAlloc> =
            VecObject::new(ObjectBuilder::default().persist(true))?;
        let in_adj: VecObject<AdjEntry, VecObjectAlloc> =
            VecObject::new(ObjectBuilder::default().persist(true))?;
        let out_raw = out_adj.object().id().raw();
        let in_raw = in_adj.object().id().raw();

        let vobj = ObjectBuilder::<Vertex>::default()
            .persist(true)
            .build(Vertex {
                id,
                label: lbl,
                name: NameKey::new(name),
                target_raw: target.raw(),
                props_raw: 0,
                flags: 0,
                out_raw,
                in_raw,
            })?;

        self.verts.push(VertexRef {
            id,
            label: lbl,
            name: NameKey::new(name),
            target_raw: target.raw(),
            props_raw: 0,
            flags: 0,
            vobj_raw: vobj.id().raw(),
            out_raw,
            in_raw,
        })?;

        self.vindex.insert(
            VKey {
                label: lbl,
                name: NameKey::new(name),
            },
            id,
        )?;
        Ok(VertexId(id))
    }

    /// Add a typed edge `from -> to`: create the edge object (endpoints as
    /// invariant pointers), append to `from`'s outgoing list and `to`'s incoming
    /// list.
    pub fn add_edge(&mut self, from: VertexId, label: &str, to: VertexId) -> Result<EdgeId> {
        let lbl = self.intern_label(label)?;

        // VERSION 4: no edge object at all. The adjacency entry carries the
        // edge id and label, and its `InvPtr` neighbour costs no FOT entry when
        // both endpoints share an arena.
        if self.store.is_some() {
            let id = self.edges.len() as u64;
            {
                let store = self.store.as_mut().expect("arena graph");
                if !store.is_alive(from.0) || !store.is_alive(to.0) {
                    return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
                }
                store.add_edge(from.0, to.0, id, lbl)?;
            }
            self.edges.push_nosync(EdgeRef {
                id,
                label: lbl,
                from_id: from.0,
                to_id: to.0,
                eobj_raw: 0, // no edge object on this layout
                props_raw: 0,
                flags: 0,
            })?;
            return Ok(EdgeId(id));
        }
        let (from_vobj, from_out, _from_in) = self
            .vertex_locs(from)
            .ok_or(TwzError::from(ArgumentError::InvalidArgument))?;
        let (to_vobj, _to_out, to_in) = self
            .vertex_locs(to)
            .ok_or(TwzError::from(ArgumentError::InvalidArgument))?;

        let fobj =
            Object::<Vertex>::map(ObjID::new(from_vobj), MapFlags::READ | MapFlags::PERSIST)?;
        let tobj = Object::<Vertex>::map(ObjID::new(to_vobj), MapFlags::READ | MapFlags::PERSIST)?;

        let id = self.edges.len() as u64;

        // The edge object: both endpoints are invariant pointers.
        let eobj = ObjectBuilder::<Edge>::default()
            .persist(true)
            .build_inplace(|tx| {
                let e = Edge {
                    id,
                    label: lbl,
                    from_id: from.0,
                    to_id: to.0,
                    from: InvPtr::new(&tx, fobj.base_ref())?,
                    to: InvPtr::new(&tx, tobj.base_ref())?,
                    props_raw: 0,
                    flags: 0,
                };
                tx.write(e)
            })?;

        // `from` outgoing list: neighbor is `to`.
        {
            let mut adj = VecObject::<AdjEntry, VecObjectAlloc>::from(Object::<
                TwzVec<AdjEntry, VecObjectAlloc>,
            >::map(
                ObjID::new(from_out),
                rw(),
            )?);
            adj.push_ctor(|place| {
                let a = AdjEntry {
                    label: lbl,
                    edge: InvPtr::new(&place, eobj.base_ref())?,
                    neighbor: InvPtr::new(&place, tobj.base_ref())?,
                };
                Ok(place.write(a))
            })?;
        }

        // `to` incoming list: neighbor is `from`.
        {
            let mut adj = VecObject::<AdjEntry, VecObjectAlloc>::from(Object::<
                TwzVec<AdjEntry, VecObjectAlloc>,
            >::map(
                ObjID::new(to_in), rw()
            )?);
            adj.push_ctor(|place| {
                let a = AdjEntry {
                    label: lbl,
                    edge: InvPtr::new(&place, eobj.base_ref())?,
                    neighbor: InvPtr::new(&place, fobj.base_ref())?,
                };
                Ok(place.write(a))
            })?;
        }

        self.edges.push(EdgeRef {
            id,
            label: lbl,
            from_id: from.0,
            to_id: to.0,
            eobj_raw: eobj.id().raw(),
            props_raw: 0, // v3 keeps the id in the edge object itself
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
        // On v4 there are no per-direction adjacency objects to name: the
        // collectors branch on `is_arena` and walk the arena's chunk chain, so
        // these two are unused. They stay zero rather than becoming an
        // `Option`, which would add a match to every v3 read for no gain.
        if self.store.is_some() {
            return Some(VertexView {
                graph: self,
                id,
                out_raw: 0,
                in_raw: 0,
            });
        }
        let (_vobj, out_raw, in_raw) = self.vertex_locs(id)?;
        Some(VertexView {
            graph: self,
            id,
            out_raw,
            in_raw,
        })
    }

    /// Convenience: outgoing/incoming/both neighbors of `id` (no predicate).
    pub fn out_neighbors(&self, id: VertexId, labels: Labels) -> Vec<VertexId> {
        if self.store.is_some() {
            return self.arena_neighbors(id, labels, true, false);
        }
        self.vertex_view(id)
            .map(|v| v.out_neighbors(labels))
            .unwrap_or_default()
    }
    pub fn in_neighbors(&self, id: VertexId, labels: Labels) -> Vec<VertexId> {
        if self.store.is_some() {
            return self.arena_neighbors(id, labels, false, true);
        }
        self.vertex_view(id)
            .map(|v| v.in_neighbors(labels))
            .unwrap_or_default()
    }
    pub fn both_neighbors(&self, id: VertexId, labels: Labels) -> Vec<VertexId> {
        if self.store.is_some() {
            return self.arena_neighbors(id, labels, true, true);
        }
        self.vertex_view(id)
            .map(|v| v.both_neighbors(labels))
            .unwrap_or_default()
    }

    // --- arena seams for the DSL (`vertex.rs`) ------------------------------

    /// `(edge_id, edge_label, neighbour_id)` in traversal order, or empty on
    /// v3. `VertexView` uses this instead of mapping adjacency objects.
    pub(crate) fn arena_adjacency(
        &self,
        id: VertexId,
        out: bool,
        inc: bool,
    ) -> Vec<(u64, u32, u64)> {
        let Some(store) = self.store.as_ref() else {
            return Vec::new();
        };
        // The store hides tombstoned *vertices* but knows nothing about the
        // edge registry, so a deleted edge would still yield its neighbour.
        // v3 filters on `is_edge_alive` per entry; this is where v4 has to do
        // the same, since `Graph` owns the registry. Missing it would diverge
        // only on deletes — a bug that shows up as a wrong query result long
        // after the change that caused it.
        store
            .adjacency(id.0, out, inc)
            .into_iter()
            .filter(|(eid, _, _)| self.is_edge_alive(EdgeId(*eid)))
            .collect()
    }

    /// A neighbour's `(label, name)` for predicate evaluation — no `String`
    /// allocation, since most candidates are filtered out.
    pub(crate) fn arena_vertex_key(&self, id: u64) -> Option<(u32, NameKey)> {
        self.store.as_ref()?.vertex_key(id)
    }

    /// An edge's label and endpoints from the registry, which both layouts
    /// share. Used to build `EdgeHandle`s without an edge *object*, which v4
    /// does not have.
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
        if let Some(store) = &self.store {
            return store.vertices().into_iter().map(VertexId).collect();
        }
        let mut out = Vec::new();
        for i in 0..self.verts.len() {
            if let Some(r) = self.verts.get_ref(i) {
                if r.flags & TOMBSTONE == 0 {
                    out.push(VertexId(r.id));
                }
            }
        }
        out
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
    pub fn delete_vertex(&mut self, id: VertexId) -> Result<()> {
        if let Some(store) = self.store.as_mut() {
            store.delete_vertex(id.0)?;
            return Ok(());
        }
        let idx = id.0 as usize;
        if idx >= self.verts.len() {
            return Ok(());
        }
        let freeable = self.verts.with_mut_at(idx, |r| {
            if r.id != id.0 || r.flags & TOMBSTONE != 0 {
                return Ok(Vec::new()); // unknown id, or already deleted
            }
            r.flags |= TOMBSTONE;
            let ids = vec![r.out_raw, r.in_raw, r.props_raw];
            // Clear the ids in the same transaction that sets the tombstone, so
            // the record never names an object that no longer exists.
            r.out_raw = 0;
            r.in_raw = 0;
            r.props_raw = 0;
            Ok(ids)
        })?;
        let _ = freeable;
        Ok(())
    }

    /// Delete an edge (tombstone). No-op if already gone.
    pub fn delete_edge(&mut self, id: EdgeId) -> Result<()> {
        let idx = id.0 as usize;
        if idx >= self.edges.len() {
            return Ok(());
        }
        let eobj_raw = self.edges.with_mut_at(idx, |r| {
            if r.id != id.0 || r.flags & TOMBSTONE != 0 {
                return Ok(0);
            }
            r.flags |= TOMBSTONE;
            Ok(r.eobj_raw)
        })?;
        if eobj_raw != 0 {
            // The property id lives in the edge object, not the registry.
            if let Ok(eo) =
                Object::<Edge>::map(ObjID::new(eobj_raw), MapFlags::READ | MapFlags::PERSIST)
            {
                reclaim::delete_raw(eo.base().props_raw);
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn owned_object_ids(&self) -> Vec<u128> {
        inventory(
            &self.verts,
            &self.edges,
            &self.labels,
            self.vindex.object().id().raw(),
        )
    }

    #[cfg(test)]
    pub(crate) fn vertex_object_ids(&self, v: VertexId) -> Option<(u128, u128, u128, u128)> {
        let r = self.verts.get_ref(v.0 as usize)?;
        Some((r.vobj_raw, r.out_raw, r.in_raw, r.props_raw))
    }

    #[cfg(test)]
    pub(crate) fn edge_object_ids(&self, e: EdgeId) -> Option<(u128, u128)> {
        let eobj_raw = self.edges.get_ref(e.0 as usize)?.eobj_raw;
        let eo =
            Object::<Edge>::map(ObjID::new(eobj_raw), MapFlags::READ | MapFlags::PERSIST).ok()?;
        let props_raw = eo.base().props_raw;
        Some((eobj_raw, props_raw))
    }

    /// Set a property on a vertex; errors if it is missing or tombstoned.
    /// Creates the vertex's property object on first use and records it in
    /// the registry mirror (`VertexRef.props_raw`).
    pub fn set_vertex_prop(&mut self, v: VertexId, key: &str, val: PropValue) -> Result<()> {
        if !self.is_vertex_alive(v) {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        if self.store.is_some() {
            let cur = self
                .store
                .as_ref()
                .and_then(|s| s.props_raw(v.0))
                .unwrap_or(0);
            let new_raw = props::set_in(cur, key, val)?;
            if new_raw != cur {
                self.store
                    .as_mut()
                    .expect("arena graph")
                    .set_props_raw(v.0, new_raw)?;
            }
            return Ok(());
        }
        let idx = v.0 as usize;
        let cur = self.verts.get_ref(idx).map(|r| r.props_raw).unwrap_or(0);
        let new_raw = props::set_in(cur, key, val)?;
        if new_raw != cur {
            self.verts.with_mut_at(idx, |r| {
                r.props_raw = new_raw;
                Ok(())
            })?;
        }
        Ok(())
    }

    /// A vertex property, or `None` if unset or the vertex is dead.
    pub fn get_vertex_prop(&self, v: VertexId, key: &str) -> Option<PropValue> {
        if !self.is_vertex_alive(v) {
            return None;
        }
        if let Some(store) = &self.store {
            return props::get_in(store.props_raw(v.0)?, key);
        }
        let raw = self.verts.get_ref(v.0 as usize)?.props_raw;
        props::get_in(raw, key)
    }

    /// All of a vertex's properties in insertion order (empty if dead/unset).
    pub fn vertex_props(&self, v: VertexId) -> Vec<(String, PropValue)> {
        if !self.is_vertex_alive(v) {
            return Vec::new();
        }
        if let Some(store) = &self.store {
            return store.props_raw(v.0).map_or(Vec::new(), props::list_in);
        }
        match self.verts.get_ref(v.0 as usize) {
            Some(r) => props::list_in(r.props_raw),
            None => Vec::new(),
        }
    }

    /// Set a property on an edge; errors if it is missing, tombstoned, or has
    /// a dead endpoint. Edge props live in the edge *object* (`EdgeRef` has
    /// no `props_raw` field; see `props.rs`).
    pub fn set_edge_prop(&mut self, e: EdgeId, key: &str, val: PropValue) -> Result<()> {
        if !self.is_edge_alive(e) {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        // VERSION 4 keeps the property-object id in the registry, since there
        // is no edge object to hold it.
        if self.store.is_some() {
            let idx = e.0 as usize;
            let cur = self.edges.get_ref(idx).map(|r| r.props_raw).unwrap_or(0);
            let new_raw = props::set_in(cur, key, val)?;
            if new_raw != cur {
                self.edges.with_mut_at(idx, |r| {
                    r.props_raw = new_raw;
                    Ok(())
                })?;
            }
            return Ok(());
        }
        let eobj_raw = self
            .edges
            .get_ref(e.0 as usize)
            .map(|r| r.eobj_raw)
            .ok_or(TwzError::from(ArgumentError::InvalidArgument))?;
        let mut eo = Object::<Edge>::map(ObjID::new(eobj_raw), rw())?;
        let cur = eo.base().props_raw;
        let new_raw = props::set_in(cur, key, val)?;
        if new_raw != cur {
            eo.with_tx(|tx| {
                tx.base_mut().props_raw = new_raw;
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
        if self.store.is_some() {
            return props::get_in(self.edges.get_ref(e.0 as usize)?.props_raw, key);
        }
        let eobj_raw = self.edges.get_ref(e.0 as usize)?.eobj_raw;
        let eo = Object::<Edge>::map(
            ObjID::new(eobj_raw),
            MapFlags::READ | MapFlags::PERSIST,
        )
        .ok()?;
        let raw = eo.base().props_raw;
        props::get_in(raw, key)
    }

    /// All of an edge's properties in insertion order (empty if dead/unset).
    pub fn edge_props(&self, e: EdgeId) -> Vec<(String, PropValue)> {
        if !self.is_edge_alive(e) {
            return Vec::new();
        }
        let Some(r) = self.edges.get_ref(e.0 as usize) else {
            return Vec::new();
        };
        if self.store.is_some() {
            return props::list_in(r.props_raw);
        }
        let eobj_raw = r.eobj_raw;
        drop(r);
        let Ok(eo) = Object::<Edge>::map(
            ObjID::new(eobj_raw),
            MapFlags::READ | MapFlags::PERSIST,
        ) else {
            return Vec::new();
        };
        let raw = eo.base().props_raw;
        props::list_in(raw)
    }

    /// Whether a vertex exists and is not tombstoned.
    pub(crate) fn is_vertex_alive(&self, id: VertexId) -> bool {
        if let Some(store) = &self.store {
            return store.is_alive(id.0);
        }
        match self.verts.get_ref(id.0 as usize) {
            Some(r) if r.id == id.0 => r.flags & TOMBSTONE == 0,
            _ => false,
        }
    }

    /// Make the arena layout durable — one sync per arena plus the registries.
    /// A no-op on v3, where every write already synced as it went.
    pub fn sync(&mut self) -> Result<()> {
        if let Some(store) = self.store.as_mut() {
            store.sync_all()?;
        }
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
        if let Some(store) = &self.store {
            return store
                .vertices_by_label(lbl)
                .into_iter()
                .map(VertexId)
                .collect();
        }
        let mut out = Vec::new();
        for i in 0..self.verts.len() {
            if let Some(r) = self.verts.get_ref(i) {
                if r.flags & TOMBSTONE == 0 && r.label == lbl {
                    out.push(VertexId(r.id));
                }
            }
        }
        out
    }

    /// Read back a vertex's data from the registry, or `None` if it is deleted.
    /// O(1): ids are append indices, so the record is at position `id`.
    pub fn vertex_info(&self, id: VertexId) -> Option<VertexInfo> {
        if let Some(store) = &self.store {
            let (lbl, name, target) = store.vertex_info(id.0)?;
            return Some(VertexInfo {
                label: self.label_name(lbl).unwrap_or_default(),
                name,
                target: ObjID::new(target),
            });
        }
        let r = self.verts.get_ref(id.0 as usize)?;
        if r.id != id.0 || r.flags & TOMBSTONE != 0 {
            return None;
        }
        Some(VertexInfo {
            label: self.label_name(r.label).unwrap_or_default(),
            name: r.name.as_str().to_string(),
            target: ObjID::new(r.target_raw),
        })
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
    fn vertex_locs(&self, v: VertexId) -> Option<(u128, u128, u128)> {
        vertex_locs_in(&self.verts, v)
    }

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

    pub fn bulk<R>(&mut self, f: impl FnOnce(&mut BulkSession<'_>) -> Result<R>) -> Result<R> {
        if self.store.is_some() {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        let Graph {
            verts,
            edges,
            labels,
            vindex,
            ..
        } = self;
        let mut s = BulkSession {
            verts,
            edges,
            labels,
            vsession: vindex.write_session()?,
            created_verts: Vec::new(),
            created_edges: Vec::new(),
            adj: HashMap::new(),
            vmaps: HashMap::new(),
        };
        let r = f(&mut s)?;
        s.finish()?;
        Ok(r)
    }
}

// --- shared lookup helpers (Graph + BulkSession) ---------------------------

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

fn intern_label_nosync(labels: &mut SegVec<LabelEntry>, name: &str) -> Result<u32> {
    if let Some(id) = find_label_in(labels, name) {
        return Ok(id);
    }
    let id = labels.len() as u32;
    labels.push_nosync(LabelEntry {
        id,
        name: NameKey::new(name),
    })?;
    Ok(id)
}

/// Vertex object/adjacency locations by id. O(1): ids are append indices.
fn vertex_locs_in(verts: &SegVec<VertexRef>, v: VertexId) -> Option<(u128, u128, u128)> {
    let r = verts.get_ref(v.0 as usize)?;
    if r.id != v.0 {
        return None;
    }
    Some((r.vobj_raw, r.out_raw, r.in_raw))
}

/// Every object a graph owns, given its open registries: the per-vertex
/// objects (vertex, both adjacency lists, properties), the per-edge objects
/// (edge, properties), and the three registries plus the vertex index.
///
/// Tombstoned records are included deliberately — their objects are exactly
/// the ones nobody will ever free otherwise. The graph root is *not* included:
/// `reset` retains and repoints it, and it is the name-registered identity of
/// the graph.
#[allow(dead_code)]
fn inventory(
    verts: &SegVec<VertexRef>,
    edges: &SegVec<EdgeRef>,
    labels: &SegVec<LabelEntry>,
    vindex_raw: u128,
) -> Vec<u128> {
    let mut ids = Vec::new();
    for i in 0..verts.len() {
        if let Some(r) = verts.get_ref(i) {
            ids.extend([r.vobj_raw, r.out_raw, r.in_raw]);
            if r.props_raw != 0 {
                ids.push(r.props_raw);
            }
        }
    }
    for i in 0..edges.len() {
        if let Some(r) = edges.get_ref(i) {
            ids.push(r.eobj_raw);
            if let Ok(eo) = Object::<Edge>::map(
                ObjID::new(r.eobj_raw),
                MapFlags::READ | MapFlags::PERSIST,
            ) {
                let p = eo.base().props_raw;
                if p != 0 {
                    ids.push(p);
                }
            }
        }
    }
    ids.extend(verts.object_ids());
    ids.extend(edges.object_ids());
    ids.extend(labels.object_ids());
    ids.push(vindex_raw);
    ids
}

/// [`inventory`] for a graph we only have raw ids for (the outgoing graph in
/// `reset`). Opens the old registries just long enough to walk them, then
/// drops the handles so the ids can be deleted. Anything that fails to open is
/// skipped rather than guessed at.
// As above — and this one is deliberately *not* called by `reset` any more,
// because opening the outgoing graph to inventory it pages the whole thing in.
#[allow(dead_code)]
fn old_inventory(
    verts_raw: u128,
    edges_raw: u128,
    labels_raw: u128,
    vindex_raw: u128,
    cap: usize,
) -> Vec<u128> {
    let (Ok(verts), Ok(edges), Ok(labels)) = (
        SegVec::<VertexRef>::open(verts_raw, cap),
        SegVec::<EdgeRef>::open(edges_raw, cap),
        SegVec::<LabelEntry>::open(labels_raw, cap),
    ) else {
        return Vec::new();
    };
    inventory(&verts, &edges, &labels, vindex_raw)
}

/// A batched-insert session; see [`Graph::bulk`]. Mirrors the semantics of
/// [`Graph::add_vertex`]/[`Graph::add_edge`] exactly, with durability
/// deferred to one sync per touched object when the batch closes.
pub struct BulkSession<'a> {
    verts: &'a mut SegVec<VertexRef>,
    edges: &'a mut SegVec<EdgeRef>,
    labels: &'a mut SegVec<LabelEntry>,
    /// Index write session: one tx over the table, synced once on drop.
    vsession: PHMsession<'a, VKey, u64>,
    /// Vertex/edge objects created without their initial sync.
    created_verts: Vec<Object<Vertex>>,
    created_edges: Vec<Object<Edge>>,
    /// Adjacency lists touched this batch: cached handles (no per-edge
    /// remapping), each synced once at finish.
    adj: HashMap<u128, VecObject<AdjEntry, VecObjectAlloc>>,
    vmaps: HashMap<u128, Object<Vertex>>,
}

impl BulkSession<'_> {
    /// Batched [`Graph::add_vertex`].
    pub fn add_vertex(&mut self, label: &str, name: &str, target: ObjID) -> Result<VertexId> {
        let lbl = intern_label_nosync(self.labels, label)?;
        let id = self.verts.len() as u64;

        let out_adj: VecObject<AdjEntry, VecObjectAlloc> =
            vec_new_nosync(ObjectBuilder::default().persist(true))?;
        let in_adj: VecObject<AdjEntry, VecObjectAlloc> =
            vec_new_nosync(ObjectBuilder::default().persist(true))?;
        let out_raw = out_adj.object().id().raw();
        let in_raw = in_adj.object().id().raw();
        // Cache the handles: created nosync (dirty from birth), and they may
        // receive edge pushes later in this batch.
        self.adj.insert(out_raw, out_adj);
        self.adj.insert(in_raw, in_adj);

        // Create the vertex object without its initial sync: the ctor aborts
        // the tx (public API; suppresses sync-on-drop, no rollback exists —
        // see segvec.rs nosync-primitives note), and finish() syncs it once.
        let vobj = ObjectBuilder::<Vertex>::default()
            .persist(true)
            .build_inplace(|tx| {
                let mut done = tx.write(Vertex {
                    id,
                    label: lbl,
                    name: NameKey::new(name),
                    target_raw: target.raw(),
                    props_raw: 0,
                    flags: 0,
                    out_raw,
                    in_raw,
                })?;
                done.abort();
                Ok(done)
            })?;

        self.verts.push_nosync(VertexRef {
            id,
            label: lbl,
            name: NameKey::new(name),
            target_raw: target.raw(),
            props_raw: 0,
            flags: 0,
            vobj_raw: vobj.id().raw(),
            out_raw,
            in_raw,
        })?;
        self.created_verts.push(vobj);

        self.vsession.insert(
            VKey {
                label: lbl,
                name: NameKey::new(name),
            },
            id,
        )?;
        Ok(VertexId(id))
    }

    /// Batched [`Graph::add_edge`].
    pub fn add_edge(&mut self, from: VertexId, label: &str, to: VertexId) -> Result<EdgeId> {
        let lbl = intern_label_nosync(self.labels, label)?;
        let (from_vobj, from_out, _from_in) = vertex_locs_in(self.verts, from)
            .ok_or(TwzError::from(ArgumentError::InvalidArgument))?;
        let (to_vobj, _to_out, to_in) =
            vertex_locs_in(self.verts, to).ok_or(TwzError::from(ArgumentError::InvalidArgument))?;

        let fobj = Self::vmap_handle(&mut self.vmaps, from_vobj)?;
        let tobj = Self::vmap_handle(&mut self.vmaps, to_vobj)?;

        let id = self.edges.len() as u64;
        let eobj = ObjectBuilder::<Edge>::default()
            .persist(true)
            .build_inplace(|tx| {
                let e = Edge {
                    id,
                    label: lbl,
                    from_id: from.0,
                    to_id: to.0,
                    from: InvPtr::new(&tx, fobj.base_ref())?,
                    to: InvPtr::new(&tx, tobj.base_ref())?,
                    props_raw: 0,
                    flags: 0,
                };
                // abort = suppress sync-on-drop (public API; no rollback
                // exists — see the nosync-primitives note in segvec.rs).
                let mut done = tx.write(e)?;
                done.abort();
                Ok(done)
            })?;

        {
            let adj = Self::adj_handle(&mut self.adj, from_out)?;
            vec_push_ctor_nosync(adj, |place| {
                let a = AdjEntry {
                    label: lbl,
                    edge: InvPtr::new(&place, eobj.base_ref())?,
                    neighbor: InvPtr::new(&place, tobj.base_ref())?,
                };
                Ok(place.write(a))
            })?;
        }
        {
            let adj = Self::adj_handle(&mut self.adj, to_in)?;
            vec_push_ctor_nosync(adj, |place| {
                let a = AdjEntry {
                    label: lbl,
                    edge: InvPtr::new(&place, eobj.base_ref())?,
                    neighbor: InvPtr::new(&place, fobj.base_ref())?,
                };
                Ok(place.write(a))
            })?;
        }

        self.edges.push_nosync(EdgeRef {
            id,
            label: lbl,
            from_id: from.0,
            to_id: to.0,
            eobj_raw: eobj.id().raw(),
            props_raw: 0, // v3 keeps the id in the edge object itself
            flags: 0,
        })?;
        self.created_edges.push(eobj);
        Ok(EdgeId(id))
    }

    /// Cached endpoint vertex handle, mapped read-only on first touch.
    /// Returns a clone of the handle (cheap: reference-counted), which keeps
    /// borrows simple across the adjacency pushes below.
    fn vmap_handle(
        vmaps: &mut HashMap<u128, Object<Vertex>>,
        raw: u128,
    ) -> Result<Object<Vertex>> {
        use std::collections::hash_map::Entry;
        Ok(match vmaps.entry(raw) {
            Entry::Occupied(o) => o.get().clone(),
            Entry::Vacant(v) => v
                .insert(Object::<Vertex>::map(
                    ObjID::new(raw),
                    MapFlags::READ | MapFlags::PERSIST,
                )?)
                .clone(),
        })
    }

    /// Cached adjacency handle, mapping the object on first touch.
    fn adj_handle(
        adj: &mut HashMap<u128, VecObject<AdjEntry, VecObjectAlloc>>,
        raw: u128,
    ) -> Result<&mut VecObject<AdjEntry, VecObjectAlloc>> {
        use std::collections::hash_map::Entry;
        Ok(match adj.entry(raw) {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => {
                v.insert(VecObject::from(
                    Object::<TwzVec<AdjEntry, VecObjectAlloc>>::map(ObjID::new(raw), rw())?,
                ))
            }
        })
    }

    /// Close the batch: one sync per touched object. Referents first (vertex
    /// and edge objects), then referrers (adjacency lists holding `InvPtr`s),
    /// then the registries; the index table syncs when `vsession` drops.
    fn finish(self) -> Result<()> {
        // Safety (for each `as_mut().sync()`): the engine is single-threaded
        // per graph handle; nothing else mutates these objects concurrently.
        for o in &self.created_verts {
            unsafe { o.as_mut()?.sync()? };
        }
        for o in &self.created_edges {
            unsafe { o.as_mut()?.sync()? };
        }
        for v in self.adj.values() {
            unsafe { v.object().as_mut()?.sync()? };
        }
        self.verts.flush()?;
        self.edges.flush()?;
        self.labels.flush()?;
        drop(self.vsession); // one sync of the index table
        Ok(())
    }
}
