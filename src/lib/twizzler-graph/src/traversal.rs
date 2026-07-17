//! A Gremlin-subset traversal DSL over [`Graph`].
//!
//! `graph.traversal()` returns a source; from a set of starting vertices a
//! traversal chains steps and ends in a terminal:
//!
//! ```ignore
//! let projects = g.traversal()
//!     .v(file)
//!     .out(Labels::these(&["tagged"]))
//!     .out(Labels::these(&["in_project"]))
//!     .dedup()
//!     .to_ids();
//! ```
//!
//! The current element set is held eagerly as a list of ids (with duplicates,
//! as in Gremlin, until `dedup`). The DSL uses only the public engine API.

use std::collections::HashSet;

use crate::{EdgeId, Graph, Labels, VertexId, VertexInfo};

impl Graph {
    /// Start a traversal.
    pub fn traversal(&self) -> TraversalSource<'_> {
        TraversalSource { graph: self }
    }
}

/// Entry point for building a traversal.
pub struct TraversalSource<'a> {
    graph: &'a Graph,
}

impl<'a> TraversalSource<'a> {
    /// Start from every vertex.
    pub fn vertices(self) -> VertexTraversal<'a> {
        let current = self.graph.vertices();
        VertexTraversal {
            graph: self.graph,
            current,
        }
    }
    /// Start from every vertex with the given label.
    pub fn with_label(self, label: &str) -> VertexTraversal<'a> {
        let current = self.graph.vertices_by_label(label);
        VertexTraversal {
            graph: self.graph,
            current,
        }
    }
    /// Start from one vertex (empty if it is deleted).
    pub fn v(self, id: VertexId) -> VertexTraversal<'a> {
        let current = if self.graph.is_vertex_alive(id) {
            vec![id]
        } else {
            Vec::new()
        };
        VertexTraversal {
            graph: self.graph,
            current,
        }
    }
    /// Start from a set of vertices (deleted ones are dropped).
    pub fn vs(self, ids: &[VertexId]) -> VertexTraversal<'a> {
        let current = ids
            .iter()
            .copied()
            .filter(|id| self.graph.is_vertex_alive(*id))
            .collect();
        VertexTraversal {
            graph: self.graph,
            current,
        }
    }
}

/// A traversal whose current elements are vertices.
pub struct VertexTraversal<'a> {
    graph: &'a Graph,
    current: Vec<VertexId>,
}

impl<'a> VertexTraversal<'a> {
    /// Move to outgoing neighbors over edges matching `labels`.
    pub fn out(mut self, labels: Labels) -> Self {
        let g = self.graph;
        let mut next = Vec::new();
        for v in &self.current {
            next.extend(g.out_neighbors(*v, labels));
        }
        self.current = next;
        self
    }
    /// Move to incoming neighbors over edges matching `labels`.
    pub fn in_(mut self, labels: Labels) -> Self {
        let g = self.graph;
        let mut next = Vec::new();
        for v in &self.current {
            next.extend(g.in_neighbors(*v, labels));
        }
        self.current = next;
        self
    }
    /// Move to neighbors in either direction over edges matching `labels`.
    pub fn both(mut self, labels: Labels) -> Self {
        let g = self.graph;
        let mut next = Vec::new();
        for v in &self.current {
            next.extend(g.both_neighbors(*v, labels));
        }
        self.current = next;
        self
    }

    /// Move to outgoing edges matching `labels`.
    pub fn out_e(self, labels: Labels) -> EdgeTraversal<'a> {
        self.edges_in(Dir::Out, labels)
    }
    /// Move to incoming edges matching `labels`.
    pub fn in_e(self, labels: Labels) -> EdgeTraversal<'a> {
        self.edges_in(Dir::In, labels)
    }
    /// Move to incident edges in either direction matching `labels`.
    pub fn both_e(self, labels: Labels) -> EdgeTraversal<'a> {
        self.edges_in(Dir::Both, labels)
    }

    /// Keep vertices with the given label.
    pub fn has_label(mut self, label: &str) -> Self {
        let g = self.graph;
        self.current
            .retain(|v| g.vertex_info(*v).map_or(false, |i| i.label == label));
        self
    }
    /// Keep vertices with the given name.
    pub fn has_name(mut self, name: &str) -> Self {
        let g = self.graph;
        self.current
            .retain(|v| g.vertex_info(*v).map_or(false, |i| i.name == name));
        self
    }
    /// Keep vertices for which the predicate returns true.
    pub fn filter(mut self, pred: impl Fn(&VertexInfo) -> bool) -> Self {
        let g = self.graph;
        self.current
            .retain(|v| g.vertex_info(*v).map_or(false, |i| pred(&i)));
        self
    }
    /// Remove duplicate vertices, preserving first-seen order.
    pub fn dedup(mut self) -> Self {
        let mut seen = HashSet::new();
        self.current.retain(|v| seen.insert(v.0));
        self
    }
    /// Keep at most `n` vertices.
    pub fn limit(mut self, n: usize) -> Self {
        self.current.truncate(n);
        self
    }

    /// Collect the current vertex ids.
    pub fn to_ids(self) -> Vec<VertexId> {
        self.current
    }
    /// Collect the current vertices' info.
    pub fn to_infos(self) -> Vec<VertexInfo> {
        let g = self.graph;
        self.current
            .into_iter()
            .filter_map(|v| g.vertex_info(v))
            .collect()
    }
    /// Number of current vertices.
    pub fn count(self) -> usize {
        self.current.len()
    }
    /// The first current vertex, if any.
    pub fn first(self) -> Option<VertexId> {
        self.current.into_iter().next()
    }

    fn edges_in(self, dir: Dir, labels: Labels) -> EdgeTraversal<'a> {
        let g = self.graph;
        let mut edges = Vec::new();
        for v in &self.current {
            if let Some(view) = g.vertex_view(*v) {
                match dir {
                    Dir::Out => edges.extend(view.out_edges(labels)),
                    Dir::In => edges.extend(view.in_edges(labels)),
                    Dir::Both => edges.extend(view.both_edges(labels)),
                }
            }
        }
        EdgeTraversal {
            graph: g,
            current: edges,
        }
    }
}

enum Dir {
    Out,
    In,
    Both,
}

/// A traversal whose current elements are edges.
pub struct EdgeTraversal<'a> {
    graph: &'a Graph,
    current: Vec<EdgeId>,
}

impl<'a> EdgeTraversal<'a> {
    /// Move to each edge's source vertex.
    pub fn out_v(self) -> VertexTraversal<'a> {
        let g = self.graph;
        let mut vs = Vec::new();
        for e in &self.current {
            if let Some(i) = g.edge_info(*e) {
                vs.push(i.from);
            }
        }
        VertexTraversal {
            graph: g,
            current: vs,
        }
    }
    /// Move to each edge's target vertex.
    pub fn in_v(self) -> VertexTraversal<'a> {
        let g = self.graph;
        let mut vs = Vec::new();
        for e in &self.current {
            if let Some(i) = g.edge_info(*e) {
                vs.push(i.to);
            }
        }
        VertexTraversal {
            graph: g,
            current: vs,
        }
    }
    /// Move to both endpoint vertices of each edge.
    pub fn both_v(self) -> VertexTraversal<'a> {
        let g = self.graph;
        let mut vs = Vec::new();
        for e in &self.current {
            if let Some(i) = g.edge_info(*e) {
                vs.push(i.from);
                vs.push(i.to);
            }
        }
        VertexTraversal {
            graph: g,
            current: vs,
        }
    }

    /// Keep edges with the given label.
    pub fn has_label(mut self, label: &str) -> Self {
        let g = self.graph;
        self.current
            .retain(|e| g.edge_info(*e).map_or(false, |i| i.label == label));
        self
    }
    /// Remove duplicate edges, preserving first-seen order.
    pub fn dedup(mut self) -> Self {
        let mut seen = HashSet::new();
        self.current.retain(|e| seen.insert(e.0));
        self
    }
    /// Keep at most `n` edges.
    pub fn limit(mut self, n: usize) -> Self {
        self.current.truncate(n);
        self
    }

    /// Collect the current edge ids.
    pub fn to_ids(self) -> Vec<EdgeId> {
        self.current
    }
    /// Number of current edges.
    pub fn count(self) -> usize {
        self.current.len()
    }
    /// The first current edge, if any.
    pub fn first(self) -> Option<EdgeId> {
        self.current.into_iter().next()
    }
}
