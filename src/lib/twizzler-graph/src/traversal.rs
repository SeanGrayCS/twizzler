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
//!
//! Steps include movement (`out`/`in_`/`both`, `out_e`/`in_e`/`both_e`,
//! `out_v`/`in_v`/`both_v`), filters (`has_label`, `has_name`, `filter`, and
//! property `has(key, value)` on both vertices and edges), ordering
//! (`order_by_name{,_desc}`, `order_by_prop{,_desc}`), reshaping (`dedup`,
//! `limit`), and terminals (`to_ids`, `to_infos`, `values(key)`, `path`,
//! `count`, `first`).
//!
//! Paths. Every traversal records, for each current element, the vertices
//! it came through; [`VertexTraversal::path`] returns them. Paths are
//! vertex-only: an edge hop (`out_e(..).in_v()`) carries the path through
//! and records the destination vertex, not the edge — Gremlin records both,
//! and narrowing that keeps the path type `Vec<VertexId>`. Tracking is always
//! on rather than opt-in, so `path()` works at the end of any chain; the cost
//! is one `Vec` per element, which suits the eager, QEMU-sized model.

use std::collections::HashSet;

use crate::{EdgeId, Graph, Labels, PropValue, VertexId, VertexInfo};

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

/// One path per element: the vertices traversed to reach it.
type Paths = Vec<Vec<VertexId>>;

/// Seed a path for each starting vertex.
fn seed_paths(current: &[VertexId]) -> Paths {
    current.iter().map(|v| vec![*v]).collect()
}

impl<'a> TraversalSource<'a> {
    /// Start from every vertex.
    pub fn vertices(self) -> VertexTraversal<'a> {
        let current = self.graph.vertices();
        let paths = seed_paths(&current);
        VertexTraversal {
            graph: self.graph,
            current,
            paths,
        }
    }
    /// Start from every vertex with the given label.
    pub fn with_label(self, label: &str) -> VertexTraversal<'a> {
        let current = self.graph.vertices_by_label(label);
        let paths = seed_paths(&current);
        VertexTraversal {
            graph: self.graph,
            current,
            paths,
        }
    }
    /// Start from one vertex (empty if it is deleted).
    pub fn v(self, id: VertexId) -> VertexTraversal<'a> {
        let current = if self.graph.is_vertex_alive(id) {
            vec![id]
        } else {
            Vec::new()
        };
        let paths = seed_paths(&current);
        VertexTraversal {
            graph: self.graph,
            current,
            paths,
        }
    }
    /// Start from a set of vertices (deleted ones are dropped).
    pub fn vs(self, ids: &[VertexId]) -> VertexTraversal<'a> {
        let current: Vec<VertexId> = ids
            .iter()
            .copied()
            .filter(|id| self.graph.is_vertex_alive(*id))
            .collect();
        let paths = seed_paths(&current);
        VertexTraversal {
            graph: self.graph,
            current,
            paths,
        }
    }
}

/// A traversal whose current elements are vertices.
pub struct VertexTraversal<'a> {
    graph: &'a Graph,
    current: Vec<VertexId>,
    paths: Paths,
}

impl<'a> VertexTraversal<'a> {
    /// Move to outgoing neighbors over edges matching `labels`.
    pub fn out(self, labels: Labels) -> Self {
        self.hop(Dir::Out, labels)
    }
    /// Move to incoming neighbors over edges matching `labels`.
    pub fn in_(self, labels: Labels) -> Self {
        self.hop(Dir::In, labels)
    }
    /// Move to neighbors in either direction over edges matching `labels`.
    pub fn both(self, labels: Labels) -> Self {
        self.hop(Dir::Both, labels)
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
        self.retain(|v| g.vertex_info(v).map_or(false, |i| i.label == label));
        self
    }
    /// Keep vertices with the given name.
    pub fn has_name(mut self, name: &str) -> Self {
        let g = self.graph;
        self.retain(|v| g.vertex_info(v).map_or(false, |i| i.name == name));
        self
    }
    /// Keep vertices for which the predicate returns true.
    pub fn filter(mut self, pred: impl Fn(&VertexInfo) -> bool) -> Self {
        let g = self.graph;
        self.retain(|v| g.vertex_info(v).map_or(false, |i| pred(&i)));
        self
    }
    /// Keep vertices whose property `key` equals `value` (≈ Gremlin `has`).
    /// Vertices lacking the key are dropped.
    pub fn has(mut self, key: &str, value: PropValue) -> Self {
        let g = self.graph;
        self.retain(|v| g.get_vertex_prop(v, key) == Some(value));
        self
    }
    /// Collect the current vertices' values for property `key`, in traversal
    /// order, skipping vertices that lack it (≈ Gremlin `values`).
    pub fn values(&self, key: &str) -> Vec<PropValue> {
        let g = self.graph;
        self.current
            .iter()
            .filter_map(|v| g.get_vertex_prop(*v, key))
            .collect()
    }

    /// Sort by vertex name, ascending; ties (and unreadable vertices) break
    /// by id, so the order is total and reproducible.
    pub fn order_by_name(mut self) -> Self {
        let g = self.graph;
        self.sort_by_key_opt(|v| g.vertex_info(v).map(|i| i.name), false);
        self
    }
    /// Sort by vertex name, descending. Vertices without readable info sort
    /// last in *both* directions, so `limit(n)` never surfaces them first.
    pub fn order_by_name_desc(mut self) -> Self {
        let g = self.graph;
        self.sort_by_key_opt(|v| g.vertex_info(v).map(|i| i.name), true);
        self
    }
    /// Sort by the value of property `key`, ascending; vertices lacking the
    /// key sort last, ties break by id.
    pub fn order_by_prop(mut self, key: &str) -> Self {
        let g = self.graph;
        self.sort_by_key_opt(|v| g.get_vertex_prop(v, key), false);
        self
    }
    /// Sort by the value of property `key`, descending; vertices lacking the
    /// key still sort last (the "newest first, missing dates last" shape the
    /// LDBC short reads want), ties break by id.
    pub fn order_by_prop_desc(mut self, key: &str) -> Self {
        let g = self.graph;
        self.sort_by_key_opt(|v| g.get_vertex_prop(v, key), true);
        self
    }

    /// Remove duplicate vertices, preserving first-seen order (and that
    /// occurrence's path).
    pub fn dedup(mut self) -> Self {
        let mut seen = HashSet::new();
        self.retain(move |v| seen.insert(v.0));
        self
    }
    /// Keep at most `n` vertices.
    pub fn limit(mut self, n: usize) -> Self {
        self.current.truncate(n);
        self.paths.truncate(n);
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
    /// The vertices traversed to reach each current element, one path per
    /// element, aligned with [`Self::to_ids`]. See the module docs for the
    /// vertex-only convention.
    pub fn path(self) -> Paths {
        self.paths
    }
    /// Number of current vertices.
    pub fn count(self) -> usize {
        self.current.len()
    }
    /// The first current vertex, if any.
    pub fn first(self) -> Option<VertexId> {
        self.current.into_iter().next()
    }

    // --- internals ---------------------------------------------------------

    /// Keep elements (and their paths) for which `keep` holds.
    fn retain(&mut self, mut keep: impl FnMut(VertexId) -> bool) {
        let mut current = Vec::with_capacity(self.current.len());
        let mut paths = Vec::with_capacity(self.paths.len());
        for (v, p) in self.current.drain(..).zip(self.paths.drain(..)) {
            if keep(v) {
                current.push(v);
                paths.push(p);
            }
        }
        self.current = current;
        self.paths = paths;
    }

    /// Sort elements and paths together by an optional key: `None` sorts last
    /// regardless of direction, ties break by vertex id.
    fn sort_by_key_opt<K: Ord>(&mut self, key: impl Fn(VertexId) -> Option<K>, desc: bool) {
        let mut zipped: Vec<(VertexId, Vec<VertexId>)> = self
            .current
            .drain(..)
            .zip(self.paths.drain(..))
            .collect();
        zipped.sort_by(|(a, _), (b, _)| {
            let (ka, kb) = (key(*a), key(*b));
            match (ka, kb) {
                (Some(x), Some(y)) => {
                    let ord = if desc { y.cmp(&x) } else { x.cmp(&y) };
                    ord.then(a.0.cmp(&b.0))
                }
                (Some(_), None) => core::cmp::Ordering::Less,
                (None, Some(_)) => core::cmp::Ordering::Greater,
                (None, None) => a.0.cmp(&b.0),
            }
        });
        for (v, p) in zipped {
            self.current.push(v);
            self.paths.push(p);
        }
    }

    /// Neighbor hop: each neighbor inherits its source's path, extended.
    fn hop(mut self, dir: Dir, labels: Labels) -> Self {
        let g = self.graph;
        let mut next = Vec::new();
        let mut next_paths = Vec::new();
        for (v, p) in self.current.iter().zip(self.paths.iter()) {
            let neighbors = match dir {
                Dir::Out => g.out_neighbors(*v, labels),
                Dir::In => g.in_neighbors(*v, labels),
                Dir::Both => g.both_neighbors(*v, labels),
            };
            for n in neighbors {
                let mut path = p.clone();
                path.push(n);
                next.push(n);
                next_paths.push(path);
            }
        }
        self.current = next;
        self.paths = next_paths;
        self
    }

    fn edges_in(self, dir: Dir, labels: Labels) -> EdgeTraversal<'a> {
        let g = self.graph;
        let mut edges = Vec::new();
        let mut paths = Vec::new();
        for (v, p) in self.current.iter().zip(self.paths.iter()) {
            if let Some(view) = g.vertex_view(*v) {
                let found = match dir {
                    Dir::Out => view.out_edges(labels),
                    Dir::In => view.in_edges(labels),
                    Dir::Both => view.both_edges(labels),
                };
                for e in found {
                    edges.push(e);
                    // Vertex-only paths: the edge itself is not recorded.
                    paths.push(p.clone());
                }
            }
        }
        EdgeTraversal {
            graph: g,
            current: edges,
            paths,
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
    paths: Paths,
}

impl<'a> EdgeTraversal<'a> {
    /// Move to each edge's source vertex.
    pub fn out_v(self) -> VertexTraversal<'a> {
        self.endpoints(End::From)
    }
    /// Move to each edge's target vertex.
    pub fn in_v(self) -> VertexTraversal<'a> {
        self.endpoints(End::To)
    }
    /// Move to both endpoint vertices of each edge.
    pub fn both_v(self) -> VertexTraversal<'a> {
        self.endpoints(End::Both)
    }

    /// Keep edges with the given label.
    pub fn has_label(mut self, label: &str) -> Self {
        let g = self.graph;
        self.retain(|e| g.edge_info(e).map_or(false, |i| i.label == label));
        self
    }
    /// Keep edges whose property `key` equals `value`; edges lacking the key
    /// are dropped.
    pub fn has(mut self, key: &str, value: PropValue) -> Self {
        let g = self.graph;
        self.retain(|e| g.get_edge_prop(e, key) == Some(value));
        self
    }
    /// Collect the current edges' values for property `key`, in order,
    /// skipping edges that lack it.
    pub fn values(&self, key: &str) -> Vec<PropValue> {
        let g = self.graph;
        self.current
            .iter()
            .filter_map(|e| g.get_edge_prop(*e, key))
            .collect()
    }
    /// Remove duplicate edges, preserving first-seen order.
    pub fn dedup(mut self) -> Self {
        let mut seen = HashSet::new();
        self.retain(move |e| seen.insert(e.0));
        self
    }
    /// Keep at most `n` edges.
    pub fn limit(mut self, n: usize) -> Self {
        self.current.truncate(n);
        self.paths.truncate(n);
        self
    }

    /// Collect the current edge ids.
    pub fn to_ids(self) -> Vec<EdgeId> {
        self.current
    }
    /// The vertex paths that reached each current edge (the edge itself is
    /// not recorded — see the module docs).
    pub fn path(self) -> Paths {
        self.paths
    }
    /// Number of current edges.
    pub fn count(self) -> usize {
        self.current.len()
    }
    /// The first current edge, if any.
    pub fn first(self) -> Option<EdgeId> {
        self.current.into_iter().next()
    }

    // --- internals ---------------------------------------------------------

    fn retain(&mut self, mut keep: impl FnMut(EdgeId) -> bool) {
        let mut current = Vec::with_capacity(self.current.len());
        let mut paths = Vec::with_capacity(self.paths.len());
        for (e, p) in self.current.drain(..).zip(self.paths.drain(..)) {
            if keep(e) {
                current.push(e);
                paths.push(p);
            }
        }
        self.current = current;
        self.paths = paths;
    }

    fn endpoints(self, end: End) -> VertexTraversal<'a> {
        let g = self.graph;
        let mut vs = Vec::new();
        let mut paths = Vec::new();
        for (e, p) in self.current.iter().zip(self.paths.iter()) {
            let Some(info) = g.edge_info(*e) else { continue };
            let mut push = |v: VertexId| {
                let mut path = p.clone();
                path.push(v);
                vs.push(v);
                paths.push(path);
            };
            match end {
                End::From => push(info.from),
                End::To => push(info.to),
                End::Both => {
                    push(info.from);
                    push(info.to);
                }
            }
        }
        VertexTraversal {
            graph: g,
            current: vs,
            paths,
        }
    }
}

enum End {
    From,
    To,
    Both,
}
