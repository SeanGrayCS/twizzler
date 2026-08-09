//! Vertices and vertex-centric traversal.
//!
//! A vertex is a fixed-size record inside a shared *arena* object, and its
//! adjacency is a chunk chain in the same arena — see `arena_store.rs`.
//! Traversal is O(degree) and, for neighbours in the same arena, stays within
//! one mapping: no `InvPtr` resolution and no FOT entry. Direction is
//! structural: `out_*`/`in_*` read one direction, `both_*` reads both.
//!
//! The traversal API mirrors Gremlin: `out_neighbors`/`in_neighbors`/`both_neighbors`
//! and `out_edges`/`in_edges`/`both_edges`, each with a label filter, plus
//! `_where` variants taking a user predicate over a [`VertexHandle`]/[`EdgeHandle`]
//! (e.g. to skip already-visited vertices during recursive traversal).

use twizzler::{marker::Invariant, object::ObjID};

use crate::{
    edge::{EdgeHandle, EdgeId},
    graph::Graph,
    name::NameKey,
};

/// A re-export, not `type VertexId = RecordId`. A type alias binds only the
/// *type* namespace, so `VertexId(id)` — the tuple-struct constructor, which
/// lives in the value namespace — fails to resolve (E0423, 28 sites). `use ...
/// as` imports every namespace the path resolves in, so both the type and the
/// constructor come across.
pub use crate::record::RecordId as VertexId;

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

/// Vestigial. The v3 vertex registry row. Nothing writes one: vertices live
/// in [`crate::arena_store::ArenaStore`] and `Graph::verts` is always empty.
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

/// A handle for vertex-centric traversal: a vertex id bound to a graph.
///
/// *v3 also carried the ObjIDs of the two adjacency lists; the arena layout has
/// no such objects, so the walk goes through `Graph::arena_adjacency`.*
pub struct VertexView<'a> {
    pub(crate) graph: &'a Graph,
    pub(crate) id: VertexId,
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

        // Adjacency is a chunk chain inside the vertex's arena, so there is no
        // per-direction object to map and no `InvPtr` to resolve for a
        // neighbour in the same arena. Liveness of the source, each neighbour
        // and each edge is already applied by `Graph::arena_adjacency`.
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
        out
    }

    fn collect_edges<F: Fn(&EdgeHandle) -> bool>(
        &self,
        which: Which,
        labels: Labels,
        pred: F,
    ) -> Vec<EdgeId> {
        let filter = self.graph.resolve_labels(labels);

        // The adjacency entry carries the edge id and label, and the endpoints
        // come from the shared registry — there is no edge *object* on this
        // layout to resolve them from.
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
        out
    }
}

fn label_matches(filter: &Option<Vec<u32>>, label: u32) -> bool {
    match filter {
        None => true,
        Some(ids) => ids.contains(&label),
    }
}
