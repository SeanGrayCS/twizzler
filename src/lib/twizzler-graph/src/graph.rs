//! The `Graph` engine: open/create a graph and run vertex/edge operations.
//!
//! `Graph` owns the cross-cutting orchestration and the registries (vertices,
//! edges, labels). Vertex-centric traversal lives on [`VertexView`] in
//! `vertex.rs`; edge/vertex record types live in their own modules.
//!
//! The registries are `VecObject`s; lookups by id index them directly (ids are
//! append indices), while the `(label, name) -> vertex` point lookup uses a
//! persistent `hachage` index. `vertices_by_label` is still a scan.
//! Adjacency is held per-vertex (see `vertex.rs`), not in the registries.

use naming::{static_naming_factory, GetFlags};
use twizzler::{
    collections::hachage::{PersistentHashMap, PersistentHashMapBase},
    collections::vec::{Vec as TwzVec, VecObject, VecObjectAlloc},
    error::TwzError,
    marker::{BaseType, Invariant},
    object::{MapFlags, ObjID, Object, ObjectBuilder, TypedObject},
    ptr::InvPtr,
};
use twizzler_rt_abi::error::ArgumentError;

use crate::error::{GraphError, Result};
use crate::edge::{Edge, EdgeId, EdgeInfo, EdgeRef};
use crate::name::NameKey;
use crate::vertex::{AdjEntry, Labels, Vertex, VertexId, VertexInfo, VertexRef, VertexView};

const MAGIC: u64 = 0x4731_5457_5A47_5248; // "G1TWZGRH"
const VERSION: u32 = 2; // on-disk format version
const TOMBSTONE: u32 = 1; // `flags` bit 0: record is deleted

/// Read/write/persist map flags for reopening mutable registries.
fn rw() -> MapFlags {
    MapFlags::READ | MapFlags::WRITE | MapFlags::PERSIST
}

/// Base of the graph root object: format guard + the registry ObjIDs (raw, so
/// the on-disk format is backend-agnostic and relocatable).
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct GraphRoot {
    pub(crate) magic: u64,
    pub(crate) version: u32,
    pub(crate) verts_raw: u128,
    pub(crate) edges_raw: u128,
    pub(crate) labels_raw: u128,
    pub(crate) vindex_raw: u128,
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
    verts: VecObject<VertexRef, VecObjectAlloc>,
    edges: VecObject<EdgeRef, VecObjectAlloc>,
    labels: VecObject<LabelEntry, VecObjectAlloc>,
    vindex: VIndex,
}

impl Graph {
    /// Open the graph registered at `data/<name>`, or create and register a
    /// fresh one. If an existing graph has an incompatible format
    /// (magic/version mismatch), this returns [`GraphError::StaleVersion`] and
    /// leaves the existing graph intact; use [`Graph::reset`] to discard it.
    pub fn open_or_create(name: &str) -> Result<Graph> {
        let mut namer = static_naming_factory().expect("naming service available");
        let path = format!("data/{name}");

        if let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) {
            let root = Object::<GraphRoot>::map(node.id.into(), MapFlags::READ | MapFlags::PERSIST)?;
            let (magic, version, verts_raw, edges_raw, labels_raw, vindex_raw) = {
                let r = root.base();
                (
                    r.magic,
                    r.version,
                    r.verts_raw,
                    r.edges_raw,
                    r.labels_raw,
                    r.vindex_raw,
                )
            };
            if magic == MAGIC && version == VERSION {
                let vbacking: Object<PersistentHashMapBase<VKey, u64>> =
                    Object::map(ObjID::new(vindex_raw), rw())?;
                return Ok(Graph {
                    root_id: node.id.into(),
                    verts: VecObject::from(Object::<TwzVec<VertexRef, VecObjectAlloc>>::map(
                        ObjID::new(verts_raw),
                        rw(),
                    )?),
                    edges: VecObject::from(Object::<TwzVec<EdgeRef, VecObjectAlloc>>::map(
                        ObjID::new(edges_raw),
                        rw(),
                    )?),
                    labels: VecObject::from(Object::<TwzVec<LabelEntry, VecObjectAlloc>>::map(
                        ObjID::new(labels_raw),
                        rw(),
                    )?),
                    vindex: PersistentHashMap::from(vbacking),
                });
            }
            // Incompatible/stale format: do NOT touch the existing graph.
            return Err(GraphError::StaleVersion {
                found: version,
                expected: VERSION,
            });
        }

        let verts: VecObject<VertexRef, VecObjectAlloc> =
            VecObject::new(ObjectBuilder::default().persist(true))?;
        let edges: VecObject<EdgeRef, VecObjectAlloc> =
            VecObject::new(ObjectBuilder::default().persist(true))?;
        let labels: VecObject<LabelEntry, VecObjectAlloc> =
            VecObject::new(ObjectBuilder::default().persist(true))?;
        let vindex = VIndex::new_persist()?;

        let root = ObjectBuilder::<GraphRoot>::default()
            .persist(true)
            .build(GraphRoot {
                magic: MAGIC,
                version: VERSION,
                verts_raw: verts.object().id().raw(),
                edges_raw: edges.object().id().raw(),
                labels_raw: labels.object().id().raw(),
                vindex_raw: vindex.object().id().raw(),
            })?;

        let _ = namer.remove(&path);
        namer.put(&path, root.id())?;

        Ok(Graph {
            root_id: root.id(),
            verts,
            edges,
            labels,
            vindex,
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
        let mut namer = static_naming_factory().expect("naming service available");
        let path = format!("data/{name}");
        let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) else {
            return Ok(()); // nothing registered
        };

        let mut root = Object::<GraphRoot>::map(node.id.into(), rw())?;
        // Only clobber something that is actually one of our graphs.
        let is_graph = root.base().magic == MAGIC;
        if !is_graph {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }

        // Fresh, empty registries.
        let verts: VecObject<VertexRef, VecObjectAlloc> =
            VecObject::new(ObjectBuilder::default().persist(true))?;
        let edges: VecObject<EdgeRef, VecObjectAlloc> =
            VecObject::new(ObjectBuilder::default().persist(true))?;
        let labels: VecObject<LabelEntry, VecObjectAlloc> =
            VecObject::new(ObjectBuilder::default().persist(true))?;
        let vindex = VIndex::new_persist()?;
        let (verts_raw, edges_raw, labels_raw, vindex_raw) = (
            verts.object().id().raw(),
            edges.object().id().raw(),
            labels.object().id().raw(),
            vindex.object().id().raw(),
        );

        // Rewrite the root transactionally so the change is synced to the
        // backing store; a raw write would be lost on reboot.
        root.with_tx(|tx| {
            let mut b = tx.base_mut();
            b.magic = MAGIC;
            b.version = VERSION;
            b.verts_raw = verts_raw;
            b.edges_raw = edges_raw;
            b.labels_raw = labels_raw;
            b.vindex_raw = vindex_raw;
            Ok(())
        })?;
        Ok(())
    }

    pub fn add_vertex(&mut self, label: &str, name: &str, target: ObjID) -> Result<VertexId> {
        let lbl = self.intern_label(label)?;
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
        let (from_vobj, from_out, _from_in) = self
            .vertex_locs(from)
            .ok_or(TwzError::from(ArgumentError::InvalidArgument))?;
        let (to_vobj, _to_out, to_in) = self
            .vertex_locs(to)
            .ok_or(TwzError::from(ArgumentError::InvalidArgument))?;

        let fobj = Object::<Vertex>::map(ObjID::new(from_vobj), MapFlags::READ | MapFlags::PERSIST)?;
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
            let mut adj = VecObject::<AdjEntry, VecObjectAlloc>::from(
                Object::<TwzVec<AdjEntry, VecObjectAlloc>>::map(ObjID::new(from_out), rw())?,
            );
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
            let mut adj = VecObject::<AdjEntry, VecObjectAlloc>::from(
                Object::<TwzVec<AdjEntry, VecObjectAlloc>>::map(ObjID::new(to_in), rw())?,
            );
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
        self.vertex_view(id)
            .map(|v| v.out_neighbors(labels))
            .unwrap_or_default()
    }
    pub fn in_neighbors(&self, id: VertexId, labels: Labels) -> Vec<VertexId> {
        self.vertex_view(id)
            .map(|v| v.in_neighbors(labels))
            .unwrap_or_default()
    }
    pub fn both_neighbors(&self, id: VertexId, labels: Labels) -> Vec<VertexId> {
        self.vertex_view(id)
            .map(|v| v.both_neighbors(labels))
            .unwrap_or_default()
    }

    /// All live vertex ids in the graph. Linear scan.
    pub fn vertices(&self) -> Vec<VertexId> {
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
        let idx = id.0 as usize;
        if idx >= self.verts.len() {
            return Ok(());
        }
        self.verts.with_mut_slice(idx..idx + 1, |s| {
            if s[0].id == id.0 {
                s[0].flags |= TOMBSTONE;
            }
            Ok(())
        })?;
        Ok(())
    }

    /// Delete an edge (tombstone). No-op if already gone.
    pub fn delete_edge(&mut self, id: EdgeId) -> Result<()> {
        let idx = id.0 as usize;
        if idx >= self.edges.len() {
            return Ok(());
        }
        self.edges.with_mut_slice(idx..idx + 1, |s| {
            if s[0].id == id.0 {
                s[0].flags |= TOMBSTONE;
            }
            Ok(())
        })?;
        Ok(())
    }

    /// Whether a vertex exists and is not tombstoned.
    pub(crate) fn is_vertex_alive(&self, id: VertexId) -> bool {
        match self.verts.get_ref(id.0 as usize) {
            Some(r) if r.id == id.0 => r.flags & TOMBSTONE == 0,
            _ => false,
        }
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
        let r = self.verts.get_ref(v.0 as usize)?;
        if r.id != v.0 {
            return None;
        }
        Some((r.vobj_raw, r.out_raw, r.in_raw))
    }

    // The content-keyed lookups below are linear scans (where a hachage index
    // would later go): find_label by name, find_vertex by (label, name), and
    // vertices_by_label.

    fn find_label(&self, name: &str) -> Option<u32> {
        for i in 0..self.labels.len() {
            let e = self.labels.get_ref(i)?;
            if e.name.eq_str(name) {
                return Some(e.id);
            }
        }
        None
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
