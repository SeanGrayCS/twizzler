//! A property-graph database engine on Twizzler objects.
//!
//! A vertex is its own persistent object owning adjacency lists of invariant
//! pointers to its incident edges and neighbors; an edge is its own persistent
//! object holding an `InvPtr<Vertex>` to each endpoint. Traversal from a vertex
//! is O(degree) pointer-chasing, with no global scan or index lookup. A `Graph`
//! root owns the vertex/edge/label registries and is registered under the
//! pager-backed `data/` namespace, so a graph re-opens by name after a reboot.

mod edge;
mod error;
mod graph;
mod name;
mod segvec;
mod traversal;
mod vertex;

#[cfg(test)]
mod tests;

pub use edge::{EdgeHandle, EdgeId, EdgeInfo};
pub use error::GraphError;
pub use graph::{BulkSession, Graph};
pub use name::NameKey;
pub use traversal::{EdgeTraversal, TraversalSource, VertexTraversal};
pub use vertex::{Labels, VertexHandle, VertexId, VertexInfo, VertexView};
