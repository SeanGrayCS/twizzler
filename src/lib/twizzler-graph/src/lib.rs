//! A property-graph database engine on Twizzler objects.
//!
//! Vertices are fixed-size records packed into shared *arena* objects, with
//! adjacency held as a chunk chain in the same arena; an edge is a registry row
//! with no object of its own. Traversal from a vertex is O(degree) and stays
//! inside one mapping whenever the neighbour shares an arena — index-free
//! adjacency with no global scan and no index lookup. A `Graph` root owns the
//! edge/label registries and the arena store, and is registered under the
//! pager-backed `data/` namespace, so a graph re-opens by name.

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
pub use graph::{BulkInsert, Graph, DEFAULT_ARENA_CAP, DEFAULT_SEG_CAP};
pub use index::{IndexSchema, IndexStrategy, Lookup, RebuildSource, UnindexedLookup};
pub use props::PropValue;
pub use arena_store::{
    record_size_for, ArenaStat, ArenaStore, FillTo, OnePerArena, Placement, MAX_TEXT_LEN,
    RECORD_SIZE_NO_PROPS,
};
pub use name::NameKey;
pub use record::RecordId;
pub use traversal::{
    EdgeTraversal, Path, PathElem, Paths, Repeat, StrictRepeat, TraversalSource, VertexTraversal,
    DEFAULT_MAX_DEPTH,
};
pub use vertex::{Labels, VertexHandle, VertexId, VertexInfo, VertexView};
