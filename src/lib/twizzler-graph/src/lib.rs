//! A property-graph database engine on Twizzler objects.
//!
//! Vertices and edges are fixed-size records in one id space, packed into
//! shared *arena* objects, with adjacency held beside each record: an inline
//! first entry per direction naming its neighbour by record id, then chunk
//! chains of invariant pointers in the same arena. Traversal from a vertex is
//! O(degree) and, for chunked same-arena neighbours, stays inside one mapping.
//! A `Graph` root owns the label registry, the index and blob families, and
//! the arena store, and is registered under the pager-backed `data/`
//! namespace, so a graph re-opens by name — including across a reboot.
//!
//! The registries are segmented (`SegVec`) so a graph can outgrow a single
//! object. `find_vertex` answers through the graph's index schema — by default
//! a volatile per-label map built on first lookup — while `vertices_by_label`
//! and label lookups are linear scans.

mod edge;
mod error;
mod graph;
mod arena_store;
mod blobstore;
mod index;
mod name;
mod props;
mod reclaim;
mod record;
mod segvec;
mod traversal;
mod vertex;

#[cfg(test)]
mod tests;

pub use edge::{EdgeHandle, EdgeId, EdgeInfo};
pub use error::GraphError;
pub use graph::{
    BulkInsert, DestroyReport, Graph, StructPages, DEFAULT_ARENA_CAP, DEFAULT_SEG_CAP,
};
// The index is a schema decision, not a constant; see `index.rs`.
pub use index::{IndexSchema, IndexStrategy, Lookup, RebuildSource, UnindexedLookup};
pub use props::PropValue;
// Exported so the stress harness can measure the arena layout directly.
pub use arena_store::{
    record_size_for, ArenaStat, ArenaStore, FillTo, OnePerArena, Placement, MAX_TEXT_LEN,
    RECORD_SIZE_NO_PROPS,
};
pub use name::NameKey;

/// Force the kernel to re-run `scan_deleted`, by creating and deleting one
/// volatile throwaway object; reports whether the sweep ran.
///
/// `Delete` marks an object and calls `scan_deleted`, which reaps only objects
/// that are pending-delete and mapped nowhere, while the syscall reports
/// success after the mark. An object still mapped when its delete lands
/// therefore stays in the object table with its pages until some later
/// `Delete` sweeps the map — and `scan_deleted` walks the whole map, not just
/// the id being deleted, which is what makes this usable as a probe from
/// userspace: it separates what a delete returns immediately from what it
/// returns eventually.
pub fn sweep_deleted_objects() -> bool {
    reclaim::sweep_deleted()
}

/// Resident pages over a set of raw object ids: `(ids_still_resolving, pages)`.
/// Ids the kernel no longer knows contribute to neither figure. Same lower-bound
/// caveat as [`Graph::resident_pages`].
pub fn pages_of_ids(ids: &[u128]) -> (usize, usize) {
    reclaim::pages_of(ids.iter().copied())
}

/// One id space for every record: `VertexId` and `EdgeId` are re-exports of
/// this type.
pub use record::RecordId;
pub use traversal::{
    EdgeTraversal, Path, PathElem, Paths, Repeat, StrictRepeat, TraversalSource, VertexTraversal,
    DEFAULT_MAX_DEPTH,
};
pub use vertex::{Labels, VertexHandle, VertexId, VertexInfo, VertexView};
