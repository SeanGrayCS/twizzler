//! Vertices and vertex-centric traversal.
//!
//! Each vertex is its own persistent object owning two adjacency lists,
//! outgoing and incoming, each a `VecObject<AdjEntry>` of invariant pointers to
//! incident edges and neighbors. Direction is structural: `out_*`/`in_*` read
//! one list, `both_*` reads both. Traversal is O(degree) pointer-chasing.
//!
//! The traversal API mirrors Gremlin: `out_neighbors`/`in_neighbors`/`both_neighbors`
//! and `out_edges`/`in_edges`/`both_edges`, each with a label filter, plus
//! `_where` variants taking a user predicate over a [`VertexHandle`]/[`EdgeHandle`]
//! (e.g. to skip already-visited vertices during recursive traversal).
//!
//! `VertexRef` is the registry mirror for enumeration and key lookup.

use twizzler::{
    collections::vec::{Vec as TwzVec, VecObject, VecObjectAlloc},
    marker::{BaseType, Invariant},
    object::{MapFlags, ObjID, Object},
    ptr::InvPtr,
};

use crate::{
    edge::{Edge, EdgeHandle, EdgeId},
    graph::Graph,
    name::NameKey,
};

/// Public vertex id (an append index).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexId(pub u64);

/// Label filter for traversal: any label, or a specific set (so one query can
/// span multiple relation types).
#[derive(Clone, Copy)]
pub enum Labels<'a> {
    Any,
    These(&'a [&'a str]),
}
impl<'a> Labels<'a> {
    pub fn any() -> Self {
        Labels::Any
    }
    pub fn these(s: &'a [&'a str]) -> Self {
        Labels::These(s)
    }
}

/// Which adjacency list(s) a traversal reads.
#[derive(Clone, Copy)]
enum Which {
    Out,
    In,
    Both,
}

/// One adjacency entry: invariant pointers to an incident edge and the neighbor
/// vertex. Direction is implied by which list (out vs in) the entry lives in.
#[repr(C)]
pub(crate) struct AdjEntry {
    pub(crate) label: u32,
    pub(crate) edge: InvPtr<Edge>,
    pub(crate) neighbor: InvPtr<Vertex>,
}
unsafe impl Invariant for AdjEntry {}

/// A vertex's persistent object: identity, label, properties, and the ObjIDs of
/// its outgoing and incoming adjacency lists.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct Vertex {
    pub(crate) id: u64,
    pub(crate) label: u32,
    pub(crate) name: NameKey,
    pub(crate) target_raw: u128,
    pub(crate) props_raw: u128, // reserved (0 = none)
    pub(crate) flags: u32,      // reserved (bit 0 = tombstone)
    pub(crate) out_raw: u128,   // ObjID of outgoing adjacency VecObject
    pub(crate) in_raw: u128,    // ObjID of incoming adjacency VecObject
}
unsafe impl Invariant for Vertex {}
impl BaseType for Vertex {}

/// Registry mirror of a vertex (enumeration / key lookup). Carries the adjacency
/// ObjIDs so traversal needs no vertex-object map.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct VertexRef {
    pub(crate) id: u64,
    pub(crate) label: u32,
    pub(crate) name: NameKey,
    pub(crate) target_raw: u128,
    pub(crate) props_raw: u128,
    pub(crate) flags: u32,
    pub(crate) vobj_raw: u128,
    pub(crate) out_raw: u128,
    pub(crate) in_raw: u128,
}
unsafe impl Invariant for VertexRef {}

/// A snapshot of a vertex's data for callers.
#[derive(Clone)]
pub struct VertexInfo {
    pub label: String,
    pub name: String,
    pub target: ObjID,
}

/// Vertex element passed to user filter predicates.
#[derive(Clone, Copy)]
pub struct VertexHandle {
    id: VertexId,
    label: u32,
    name: NameKey,
}
impl VertexHandle {
    pub fn id(&self) -> VertexId {
        self.id
    }
    pub fn label_id(&self) -> u32 {
        self.label
    }
    pub fn name(&self) -> &str {
        self.name.as_str()
    }
}

/// A handle for vertex-centric traversal: a vertex id plus the locations of its
/// outgoing/incoming adjacency lists, bound to a graph.
pub struct VertexView<'a> {
    pub(crate) graph: &'a Graph,
    pub(crate) id: VertexId,
    pub(crate) out_raw: u128,
    pub(crate) in_raw: u128,
}

impl<'a> VertexView<'a> {
    pub fn id(&self) -> VertexId {
        self.id
    }

    // --- neighbors -----------------------------------------------------------

    pub fn out_neighbors(&self, labels: Labels) -> Vec<VertexId> {
        self.collect_neighbors(Which::Out, labels, |_| true)
    }
    pub fn in_neighbors(&self, labels: Labels) -> Vec<VertexId> {
        self.collect_neighbors(Which::In, labels, |_| true)
    }
    pub fn both_neighbors(&self, labels: Labels) -> Vec<VertexId> {
        self.collect_neighbors(Which::Both, labels, |_| true)
    }
    pub fn out_neighbors_where(
        &self,
        labels: Labels,
        pred: impl Fn(&VertexHandle) -> bool,
    ) -> Vec<VertexId> {
        self.collect_neighbors(Which::Out, labels, pred)
    }
    pub fn in_neighbors_where(
        &self,
        labels: Labels,
        pred: impl Fn(&VertexHandle) -> bool,
    ) -> Vec<VertexId> {
        self.collect_neighbors(Which::In, labels, pred)
    }
    pub fn both_neighbors_where(
        &self,
        labels: Labels,
        pred: impl Fn(&VertexHandle) -> bool,
    ) -> Vec<VertexId> {
        self.collect_neighbors(Which::Both, labels, pred)
    }

    // --- edges ---------------------------------------------------------------

    pub fn out_edges(&self, labels: Labels) -> Vec<EdgeId> {
        self.collect_edges(Which::Out, labels, |_| true)
    }
    pub fn in_edges(&self, labels: Labels) -> Vec<EdgeId> {
        self.collect_edges(Which::In, labels, |_| true)
    }
    pub fn both_edges(&self, labels: Labels) -> Vec<EdgeId> {
        self.collect_edges(Which::Both, labels, |_| true)
    }
    pub fn out_edges_where(
        &self,
        labels: Labels,
        pred: impl Fn(&EdgeHandle) -> bool,
    ) -> Vec<EdgeId> {
        self.collect_edges(Which::Out, labels, pred)
    }
    pub fn in_edges_where(
        &self,
        labels: Labels,
        pred: impl Fn(&EdgeHandle) -> bool,
    ) -> Vec<EdgeId> {
        self.collect_edges(Which::In, labels, pred)
    }
    pub fn both_edges_where(
        &self,
        labels: Labels,
        pred: impl Fn(&EdgeHandle) -> bool,
    ) -> Vec<EdgeId> {
        self.collect_edges(Which::Both, labels, pred)
    }

    // --- internals -----------------------------------------------------------

    fn lists(&self, which: Which) -> Vec<u128> {
        match which {
            Which::Out => vec![self.out_raw],
            Which::In => vec![self.in_raw],
            Which::Both => vec![self.out_raw, self.in_raw],
        }
    }

    /// Which direction(s) `which` selects, as `(out, in)`.
    fn dirs(which: Which) -> (bool, bool) {
        match which {
            Which::Out => (true, false),
            Which::In => (false, true),
            Which::Both => (true, true),
        }
    }

    fn collect_neighbors<F: Fn(&VertexHandle) -> bool>(
        &self,
        which: Which,
        labels: Labels,
        pred: F,
    ) -> Vec<VertexId> {
        let filter = self.graph.resolve_labels(labels);

        // VERSION 4: adjacency is a chunk chain inside the vertex's arena, so
        // there is no per-direction object to map and no `InvPtr` to resolve
        // for a neighbour in the same arena. Liveness of both the source and
        // each neighbour is already applied by the store's walk.
        if self.graph.is_arena() {
            let (o, i) = Self::dirs(which);
            let mut out = Vec::new();
            for (_, elabel, nb) in self.graph.arena_adjacency(self.id, o, i) {
                if !label_matches(&filter, elabel) {
                    continue;
                }
                let Some((label, name)) = self.graph.arena_vertex_key(nb) else {
                    continue;
                };
                let h = VertexHandle {
                    id: VertexId(nb),
                    label,
                    name,
                };
                if pred(&h) {
                    out.push(h.id);
                }
            }
            return out;
        }

        let mut out = Vec::new();
        for raw in self.lists(which) {
            let Ok(obj) = Object::<TwzVec<AdjEntry, VecObjectAlloc>>::map(
                ObjID::new(raw),
                MapFlags::READ | MapFlags::PERSIST,
            ) else {
                continue;
            };
            let adj = VecObject::from(obj);
            for i in 0..adj.len() {
                if let Some(e) = adj.get_ref(i) {
                    if !label_matches(&filter, e.label) {
                        continue;
                    }
                    let edge = unsafe { e.edge.resolve() };
                    if !self.graph.is_edge_alive(EdgeId(edge.id)) {
                        continue;
                    }
                    let nb = unsafe { e.neighbor.resolve() };
                    let h = VertexHandle {
                        id: VertexId(nb.id),
                        label: nb.label,
                        name: nb.name,
                    };
                    if pred(&h) {
                        out.push(h.id);
                    }
                }
            }
        }
        out
    }

    fn collect_edges<F: Fn(&EdgeHandle) -> bool>(
        &self,
        which: Which,
        labels: Labels,
        pred: F,
    ) -> Vec<EdgeId> {
        let filter = self.graph.resolve_labels(labels);

        // VERSION 4: the adjacency entry carries the edge id and label, and the
        // endpoints come from the shared registry — there is no edge *object*
        // on this layout to resolve them from.
        if self.graph.is_arena() {
            let (o, i) = Self::dirs(which);
            let mut out = Vec::new();
            for (eid, elabel, _) in self.graph.arena_adjacency(self.id, o, i) {
                if !label_matches(&filter, elabel) {
                    continue;
                }
                let Some((label, from, to)) = self.graph.edge_endpoints(EdgeId(eid)) else {
                    continue;
                };
                let h = EdgeHandle::new(EdgeId(eid), label, from, to);
                if pred(&h) {
                    out.push(h.id());
                }
            }
            return out;
        }

        let mut out = Vec::new();
        for raw in self.lists(which) {
            let Ok(obj) = Object::<TwzVec<AdjEntry, VecObjectAlloc>>::map(
                ObjID::new(raw),
                MapFlags::READ | MapFlags::PERSIST,
            ) else {
                continue;
            };
            let adj = VecObject::from(obj);
            for i in 0..adj.len() {
                if let Some(e) = adj.get_ref(i) {
                    if !label_matches(&filter, e.label) {
                        continue;
                    }
                    let edge = unsafe { e.edge.resolve() };
                    if !self.graph.is_edge_alive(EdgeId(edge.id)) {
                        continue;
                    }
                    let h = EdgeHandle::new(
                        EdgeId(edge.id),
                        edge.label,
                        VertexId(edge.from_id),
                        VertexId(edge.to_id),
                    );
                    if pred(&h) {
                        out.push(h.id());
                    }
                }
            }
        }
        out
    }
}

fn label_matches(filter: &Option<Vec<u32>>, label: u32) -> bool {
    match filter {
        None => true,
        Some(ids) => ids.contains(&label),
    }
}
