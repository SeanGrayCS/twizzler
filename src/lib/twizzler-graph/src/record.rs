//! One id space for every record.
//!
//! Vertices and edges are records in the same registry: `add_vertex` and
//! `add_edge` both return the record id and the caller tracks it. `VertexId`
//! and `EdgeId` are re-exports of [`RecordId`], so `EdgeId(n)` and
//! `VertexId(n)` name the same record. Don't construct an id by value —
//! capture what `add_vertex`/`add_edge` return.
//!
//! One id space makes the operations generalise: `out_neighbors` of an edge
//! record is the edge's targets, which is why hyperedges need no new
//! machinery. The cost is that "edge passed where a vertex belongs" cannot be
//! rejected at compile time, so `IS_EDGE` (a `flags` bit, mirrored into
//! `VertexLoc`) also serves runtime kind checks: `edge_info` on a vertex
//! record must return `None`.

/// A record's id — an append index into the location registry, shared by every
/// record whatever its kind.
///
/// Ids are never reused: deletion tombstones, and `add_*` always appends.
/// Code may rely on that.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RecordId(pub u64);

impl RecordId {
    /// The raw index. Prefer this to `.0` in new code so the field can become
    /// private later.
    pub fn raw(self) -> u64 {
        self.0
    }
}

impl From<u64> for RecordId {
    fn from(v: u64) -> Self {
        RecordId(v)
    }
}

impl core::fmt::Display for RecordId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}
