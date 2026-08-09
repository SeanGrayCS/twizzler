//! Edges.
//!
//! An edge has no object of its own. `EdgeRef` — the registry row — *is* the
//! edge: it carries the label, both endpoint ids, the property-object id and the
//! tombstone bit. Traversal reaches an edge through an adjacency entry in the
//! source vertex's arena, which stores the edge id inline, so following an edge
//! is an index into the registry rather than a pointer chase.
//!
//! `EdgeHandle` is the element passed to user filter predicates.

use twizzler::marker::Invariant;

use crate::vertex::VertexId;

/// Until the layout lands, `EdgeId(n)` built by value is a latent bug — it
/// compiles under either regime but means different records. Capture what
/// `add_edge` returns instead. See `record.rs`.
///
/// Re-exported rather than aliased so the tuple-struct constructor resolves;
/// see the note on `VertexId`.
pub use crate::record::RecordId as EdgeId;

/// The edge itself: label, endpoints, properties, tombstone. Lives in the edge
/// registry; there is no separate edge object to mirror.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct EdgeRef {
    pub(crate) id: u64,
    pub(crate) label: u32,
    pub(crate) from_id: u64,
    pub(crate) to_id: u64,
    pub(crate) eobj_raw: u128,
    /// VERSION 4: the edge's property object. On v3 this stays 0 and the id
    /// lives in the edge *object* instead — v4 has no edge object, so the
    /// registry has to carry it.
    pub(crate) props_raw: u128,
    pub(crate) flags: u32,
}
unsafe impl Invariant for EdgeRef {}

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
