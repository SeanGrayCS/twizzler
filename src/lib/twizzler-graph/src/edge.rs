//! Edges.
//!
//! An edge has no object of its own: it is a record in the arena, marked
//! `IS_EDGE` and drawn from the same id space as vertices — so edge
//! properties, deletion and traversal are the vertex paths rather than
//! parallel implementations. Its endpoints are its own adjacency chains, which
//! is what makes a hyperedge need no new machinery.
//!
//! `EdgeHandle` is the element passed to user filter predicates.

use crate::vertex::VertexId;

/// Public edge id (an append index) — `RecordId` under another name. Edges are
/// records in the shared registry, so edge ids and vertex ids come from one
/// append sequence and `EdgeId(n)` names the same record as `VertexId(n)`.
/// Don't build an `EdgeId` by value; capture what `add_edge` returns. See
/// `record.rs`.
///
/// Re-exported rather than aliased so the tuple-struct constructor resolves;
/// see the note on `VertexId`.
pub use crate::record::RecordId as EdgeId;

/// An edge's label and endpoints, returned by `Graph::edge_info`.
#[derive(Clone)]
pub struct EdgeInfo {
    pub label: String,
    pub from: VertexId,
    pub to: VertexId,
}

/// Edge element passed to user filter predicates.
#[derive(Clone, Copy)]
pub struct EdgeHandle {
    id: EdgeId,
    label: u32,
    from: VertexId,
    to: VertexId,
}
impl EdgeHandle {
    pub(crate) fn new(id: EdgeId, label: u32, from: VertexId, to: VertexId) -> Self {
        EdgeHandle {
            id,
            label,
            from,
            to,
        }
    }
    pub fn id(&self) -> EdgeId {
        self.id
    }
    pub fn label_id(&self) -> u32 {
        self.label
    }
    pub fn from(&self) -> VertexId {
        self.from
    }
    pub fn to(&self) -> VertexId {
        self.to
    }
}
