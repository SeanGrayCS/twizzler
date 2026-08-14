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

use crate::error::{GraphError, Result};
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
            truncated: false,
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
            truncated: false,
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
            truncated: false,
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
            truncated: false,
        }
    }
}

/// A traversal whose current elements are vertices.
pub struct VertexTraversal<'a> {
    graph: &'a Graph,
    current: Vec<VertexId>,
    paths: Paths,
    /// Sticky. Every step carries it forward, because a truncated walk taints
    /// everything computed from it — clearing it on the next `.out()` would
    /// recreate the silent-short-answer problem one step removed. The cost is
    /// that it names the chain rather than the step: you learn *that* something
    /// truncated, not which repeat did.
    truncated: bool,
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

    pub fn has_text(mut self, key: &str, value: &str) -> Self {
        let g = self.graph;
        self.retain(|v| g.text_eq(v, key, value));
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
    ///
    /// Decorate–sort–undecorate rather than `slice::sort_by_cached_key`,
    /// because the order is not `K`'s natural one: `None` sorts last in *both*
    /// directions and the id tiebreak is always ascending. Expressing that
    /// through a cached-key sort needs a wrapper type whose `Ord` depends on
    /// `desc` — more machinery for the same `n` reads.
    fn sort_by_key_opt<K: Ord>(&mut self, key: impl Fn(VertexId) -> Option<K>, desc: bool) {
        let mut zipped: Vec<(Option<K>, VertexId, Vec<VertexId>)> = self
            .current
            .drain(..)
            .zip(self.paths.drain(..))
            .map(|(v, p)| (key(v), v, p))
            .collect();
        zipped.sort_by(|(ka, a, _), (kb, b, _)| match (ka, kb) {
            (Some(x), Some(y)) => {
                let ord = if desc { y.cmp(x) } else { x.cmp(y) };
                ord.then(a.0.cmp(&b.0))
            }
            (Some(_), None) => core::cmp::Ordering::Less,
            (None, Some(_)) => core::cmp::Ordering::Greater,
            (None, None) => a.0.cmp(&b.0),
        });
        for (_, v, p) in zipped {
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
            truncated: false,
        }
    }
}

enum End {
    From,
    To,
    Both,
}

// Not Gremlin's higher-order `repeat(step)`. An anonymous step fights an
// ownership model where every step consumes `self`; `repeat_out(labels)` plus a
// builder expresses every criterion, including IS6's
// `repeat(out(replyOf)).until(no outgoing replyOf)`. The closure form stays open
// if a query ever needs a compound per-hop step.

enum HopFilter<'a> {
    Label(String),
    Prop(String, PropValue),
    Pred(Box<dyn Fn(&VertexInfo) -> bool + 'a>),
}

/// How a walk stops.
enum Stop<'a> {
    /// Exactly k hops.
    Times(usize),
    Until(Box<dyn Fn(&VertexInfo) -> bool + 'a>),
    /// Until nothing new is reachable. Returns the last non-empty frontier —
    /// the end of the chain, which is what IS6 wants.
    Exhausted,
}

/// Default depth cap. Deep enough for any realistic `replyOf` chain (LDBC's are
/// single digits), shallow enough that a cyclic or adversarial graph stops
/// promptly. The visited set already guarantees termination on a finite graph;
/// this is the second line, against depth rather than repetition.
pub const DEFAULT_MAX_DEPTH: usize = 64;

struct RepeatCfg<'a> {
    dir: Dir,
    labels: Labels<'a>,
    emit: bool,
    max_depth: usize,
    filters: Vec<HopFilter<'a>>,
}

impl<'a> RepeatCfg<'a> {
    fn passes(&self, g: &Graph, v: VertexId) -> bool {
        if self.filters.is_empty() {
            return true;
        }
        let Some(info) = g.vertex_info(v) else {
            return false;
        };
        self.filters.iter().all(|f| match f {
            HopFilter::Label(l) => info.label == *l,
            HopFilter::Prop(k, val) => g.get_vertex_prop(v, k) == Some(*val),
            HopFilter::Pred(p) => p(&info),
        })
    }
}

/// Builder for a recursive walk. Configure, then terminate with `times`,
/// `until`, or `until_exhausted`.
pub struct Repeat<'a> {
    base: VertexTraversal<'a>,
    cfg: RepeatCfg<'a>,
}

/// As [`Repeat`], but truncation at the depth cap is an error rather than a
/// queryable flag.
///
/// A distinct type on purpose: if strictness were a flag on `Repeat`, a caller
/// could set it and then call a terminator that cannot fail, silently getting
/// the lax behaviour they explicitly asked against. Here the terminators return
/// `Result`, so the choice cannot be ignored.
pub struct StrictRepeat<'a> {
    inner: Repeat<'a>,
}

impl<'a> VertexTraversal<'a> {
    /// Walk outgoing edges repeatedly. See [`Repeat`].
    pub fn repeat_out(self, labels: Labels<'a>) -> Repeat<'a> {
        Repeat::new(self, Dir::Out, labels)
    }
    /// Walk incoming edges repeatedly.
    pub fn repeat_in(self, labels: Labels<'a>) -> Repeat<'a> {
        Repeat::new(self, Dir::In, labels)
    }
    /// Walk edges in either direction repeatedly.
    pub fn repeat_both(self, labels: Labels<'a>) -> Repeat<'a> {
        Repeat::new(self, Dir::Both, labels)
    }

    pub fn hit_depth_cap(&self) -> bool {
        self.truncated
    }
}

impl<'a> Repeat<'a> {
    fn new(base: VertexTraversal<'a>, dir: Dir, labels: Labels<'a>) -> Self {
        Repeat {
            base,
            cfg: RepeatCfg {
                dir,
                labels,
                emit: false,
                max_depth: DEFAULT_MAX_DEPTH,
                filters: Vec::new(),
            },
        }
    }

    /// Collect every vertex visited, in first-visit order, rather than only the
    /// final frontier. The start vertices are not re-emitted.
    pub fn emit(mut self) -> Self {
        self.cfg.emit = true;
        self
    }

    /// Bound the walk. Truncation is reported by
    /// [`VertexTraversal::hit_depth_cap`]; see [`Repeat::strict_depth`] to make
    /// it an error instead.
    pub fn max_depth(mut self, n: usize) -> Self {
        self.cfg.max_depth = n;
        self
    }

    /// Bound the walk and make truncation an error.
    pub fn strict_depth(mut self, n: usize) -> StrictRepeat<'a> {
        self.cfg.max_depth = n;
        StrictRepeat { inner: self }
    }

    /// Keep only vertices with this label, per hop.
    pub fn has_label(mut self, label: &str) -> Self {
        self.cfg.filters.push(HopFilter::Label(label.to_string()));
        self
    }

    /// Keep only vertices with this property value, per hop.
    pub fn has(mut self, key: &str, value: PropValue) -> Self {
        self.cfg
            .filters
            .push(HopFilter::Prop(key.to_string(), value));
        self
    }

    /// Keep only vertices satisfying `pred`, per hop.
    pub fn filter(mut self, pred: impl Fn(&VertexInfo) -> bool + 'a) -> Self {
        self.cfg.filters.push(HopFilter::Pred(Box::new(pred)));
        self
    }

    /// Apply the hop exactly `k` times.
    pub fn times(self, k: usize) -> VertexTraversal<'a> {
        walk(self.base, self.cfg, Stop::Times(k))
    }

    /// Walk until a frontier contains a vertex satisfying `pred`, returning
    /// only the matching vertices. If nothing ever matches the result is
    /// empty; if that was because the cap was reached, `hit_depth_cap` says so.
    pub fn until(self, pred: impl Fn(&VertexInfo) -> bool + 'a) -> VertexTraversal<'a> {
        walk(self.base, self.cfg, Stop::Until(Box::new(pred)))
    }

    /// Walk until nothing new is reachable, returning the last non-empty
    /// frontier — or, with `emit`, everything visited.
    pub fn until_exhausted(self) -> VertexTraversal<'a> {
        walk(self.base, self.cfg, Stop::Exhausted)
    }
}

impl<'a> StrictRepeat<'a> {
    pub fn emit(mut self) -> Self {
        self.inner = self.inner.emit();
        self
    }
    pub fn has_label(mut self, label: &str) -> Self {
        self.inner = self.inner.has_label(label);
        self
    }
    pub fn has(mut self, key: &str, value: PropValue) -> Self {
        self.inner = self.inner.has(key, value);
        self
    }
    pub fn filter(mut self, pred: impl Fn(&VertexInfo) -> bool + 'a) -> Self {
        self.inner = self.inner.filter(pred);
        self
    }

    pub fn times(self, k: usize) -> Result<VertexTraversal<'a>> {
        check(self.inner.times(k))
    }
    pub fn until(self, pred: impl Fn(&VertexInfo) -> bool + 'a) -> Result<VertexTraversal<'a>> {
        check(self.inner.until(pred))
    }
    pub fn until_exhausted(self) -> Result<VertexTraversal<'a>> {
        check(self.inner.until_exhausted())
    }
}

fn check(t: VertexTraversal<'_>) -> Result<VertexTraversal<'_>> {
    if t.truncated {
        return Err(GraphError::WalkTruncated);
    }
    Ok(t)
}

/// The walk itself: breadth-first, one visited set, per-hop filtering before
/// expansion.
fn walk<'a>(mut base: VertexTraversal<'a>, cfg: RepeatCfg<'a>, stop: Stop<'a>) -> VertexTraversal<'a> {
    let g = base.graph;
    // Seeded with the start, so a cycle back to it neither loops nor re-emits.
    let mut visited: HashSet<u64> = base.current.iter().map(|v| v.0).collect();
    let mut frontier = std::mem::take(&mut base.current);
    let mut fpaths = std::mem::take(&mut base.paths);
    let mut last_nonempty = (frontier.clone(), fpaths.clone());
    let mut emitted: Vec<VertexId> = Vec::new();
    let mut epaths: Paths = Vec::new();

    let limit = match stop {
        Stop::Times(k) => k.min(cfg.max_depth),
        _ => cfg.max_depth,
    };

    let mut depth = 0usize;
    while depth < limit && !frontier.is_empty() {
        let mut next = Vec::new();
        let mut npaths = Vec::new();
        for (v, p) in frontier.iter().zip(fpaths.iter()) {
            let neighbors = match cfg.dir {
                Dir::Out => g.out_neighbors(*v, cfg.labels),
                Dir::In => g.in_neighbors(*v, cfg.labels),
                Dir::Both => g.both_neighbors(*v, cfg.labels),
            };
            for n in neighbors {
                if !visited.insert(n.0) {
                    continue;
                }
                if !cfg.passes(g, n) {
                    continue;
                }
                let mut path = p.clone();
                path.push(n);
                next.push(n);
                npaths.push(path);
            }
        }
        depth += 1;
        frontier = next;
        fpaths = npaths;
        if !frontier.is_empty() {
            last_nonempty = (frontier.clone(), fpaths.clone());
        }
        if cfg.emit {
            emitted.extend(frontier.iter().copied());
            epaths.extend(fpaths.iter().cloned());
        }

        if let Stop::Until(pred) = &stop {
            let hits: Vec<usize> = frontier
                .iter()
                .enumerate()
                .filter(|(_, v)| g.vertex_info(**v).map(|i| pred(&i)).unwrap_or(false))
                .map(|(i, _)| i)
                .collect();
            if !hits.is_empty() {
                base.current = hits.iter().map(|&i| frontier[i]).collect();
                base.paths = hits.iter().map(|&i| fpaths[i].clone()).collect();
                return base;
            }
        }
    }

    if depth == cfg.max_depth && !frontier.is_empty() {
        base.truncated = true;
    }

    let (cur, ps) = if cfg.emit {
        (emitted, epaths)
    } else {
        match stop {
            // The last non-empty frontier: the end of the chain.
            Stop::Exhausted => last_nonempty,
            // `until` that never matched yields nothing; `hit_depth_cap`
            // distinguishes "no match" from "gave up".
            Stop::Until(_) => (Vec::new(), Vec::new()),
            Stop::Times(_) => (frontier, fpaths),
        }
    };
    base.current = cur;
    base.paths = ps;
    base
}
