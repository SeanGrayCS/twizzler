//! Engine integration tests, run on Twizzler via `cargo start-qemu --tests`.
//!
//! Each test creates a persistent graph registered under `data/<name>`. To stay
//! idempotent across runs every test resets its graph first (`fresh`), and graph
//! names are unique per test.

use twizzler::object::ObjID;

use crate::{Graph, GraphError, Labels};

/// Clear any existing graph of this name, then open a clean one.
fn fresh(name: &str) -> Graph {
    Graph::reset(name).expect("reset graph");
    Graph::open_or_create(name).expect("create graph")
}

#[test]
fn create_and_vertex_info() {
    let mut g = fresh("t-civ");
    let v = g.add_vertex("file", "doc", ObjID::new(0)).unwrap();
    let info = g.vertex_info(v).expect("vertex info");
    assert_eq!(info.label, "file");
    assert_eq!(info.name, "doc");
}

#[test]
fn edges_and_neighbors() {
    let mut g = fresh("t-en");
    let a = g.add_vertex("file", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("file", "b", ObjID::new(0)).unwrap();
    let t = g.add_vertex("tag", "thesis", ObjID::new(0)).unwrap();
    g.add_edge(a, "tagged", t).unwrap();
    g.add_edge(b, "tagged", t).unwrap();

    // Incoming `tagged` edges into the tag → both files.
    assert_eq!(g.in_neighbors(t, Labels::these(&["tagged"])).len(), 2);
    // Outgoing from a file → the tag.
    let outs = g.out_neighbors(a, Labels::any());
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0], t);
    // The tag has no outgoing edges.
    assert_eq!(g.out_neighbors(t, Labels::any()).len(), 0);
}

#[test]
fn label_filter_single_and_multi() {
    let mut g = fresh("t-lbl");
    let a = g.add_vertex("file", "a", ObjID::new(0)).unwrap();
    let t = g.add_vertex("tag", "t", ObjID::new(0)).unwrap();
    let u = g.add_vertex("user", "u", ObjID::new(0)).unwrap();
    g.add_edge(a, "tagged", t).unwrap();
    g.add_edge(a, "authored_by", u).unwrap();

    assert_eq!(g.out_neighbors(a, Labels::these(&["tagged"])).len(), 1);
    assert_eq!(
        g.out_neighbors(a, Labels::these(&["tagged", "authored_by"]))
            .len(),
        2
    );
    assert_eq!(g.out_neighbors(a, Labels::any()).len(), 2);
    // A label that does not exist matches nothing.
    assert_eq!(g.out_neighbors(a, Labels::these(&["nope"])).len(), 0);
}

#[test]
fn predicate_filter_skips_vertices() {
    let mut g = fresh("t-pred");
    let a = g.add_vertex("file", "a", ObjID::new(0)).unwrap();
    let keep = g.add_vertex("tag", "keep", ObjID::new(0)).unwrap();
    let _skip = g.add_vertex("tag", "skip", ObjID::new(0)).unwrap();
    g.add_edge(a, "rel", keep).unwrap();
    g.add_edge(a, "rel", _skip).unwrap();

    let view = g.vertex_view(a).unwrap();
    let kept = view.out_neighbors_where(Labels::any(), |h| h.name() != "skip");
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0], keep);
}

#[test]
fn directionality_out_in_both() {
    let mut g = fresh("t-dir");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let c = g.add_vertex("n", "c", ObjID::new(0)).unwrap();
    g.add_edge(a, "e", b).unwrap(); // a -> b
    g.add_edge(c, "e", a).unwrap(); // c -> a

    let view = g.vertex_view(a).unwrap();
    assert_eq!(view.out_neighbors(Labels::any()), vec![b]);
    assert_eq!(view.in_neighbors(Labels::any()), vec![c]);
    assert_eq!(view.both_neighbors(Labels::any()).len(), 2);
}

#[test]
fn edge_api() {
    let mut g = fresh("t-edges");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let e = g.add_edge(a, "e", b).unwrap();

    let va = g.vertex_view(a).unwrap();
    let oe = va.out_edges(Labels::any());
    assert_eq!(oe.len(), 1);
    assert_eq!(oe[0], e);
    assert_eq!(va.in_edges(Labels::any()).len(), 0);

    let vb = g.vertex_view(b).unwrap();
    assert_eq!(vb.in_edges(Labels::any()).len(), 1);
}

#[test]
fn find_and_by_label() {
    let mut g = fresh("t-find");
    let t = g.add_vertex("tag", "thesis", ObjID::new(0)).unwrap();
    assert_eq!(g.find_vertex("tag", "thesis"), Some(t));
    assert_eq!(g.find_vertex("tag", "missing"), None);
    g.add_vertex("tag", "other", ObjID::new(0)).unwrap();
    assert_eq!(g.vertices_by_label("tag").len(), 2);
    assert_eq!(g.vertices_by_label("file").len(), 0);
}

#[test]
fn reopen_within_boot_persists() {
    let name = "t-reopen";
    let _ = Graph::reset(name);
    let v = {
        let mut g = Graph::open_or_create(name).unwrap();
        g.add_vertex("file", "persisted", ObjID::new(0)).unwrap()
    };
    // Reopen the same graph by name and confirm the vertex is still there.
    let g2 = Graph::open_or_create(name).unwrap();
    assert_eq!(g2.vertex_info(v).unwrap().name, "persisted");
    let _ = Graph::reset(name);
}

#[test]
fn reset_clears_and_is_idempotent() {
    let name = "t-reset";
    let _ = Graph::reset(name);
    {
        let mut g = Graph::open_or_create(name).unwrap();
        g.add_vertex("file", "x", ObjID::new(0)).unwrap();
    }
    Graph::reset(name).unwrap();
    // Resetting again is a no-op (not an error).
    Graph::reset(name).unwrap();
    // After reset, the registration is reused and the graph is empty.
    let g = Graph::open_or_create(name).unwrap();
    assert_eq!(g.vertices_by_label("file").len(), 0);
    let _ = Graph::reset(name);
}

#[test]
fn multi_hop_reachability() {
    let mut g = fresh("t-hops");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let c = g.add_vertex("n", "c", ObjID::new(0)).unwrap();
    let d = g.add_vertex("n", "d", ObjID::new(0)).unwrap();
    g.add_edge(a, "e", b).unwrap();
    g.add_edge(b, "e", c).unwrap();
    g.add_edge(c, "e", d).unwrap();

    // Walk the path one hop at a time.
    let h1 = g.out_neighbors(a, Labels::any());
    assert_eq!(h1, vec![b]);
    let h2 = g.out_neighbors(h1[0], Labels::any());
    assert_eq!(h2, vec![c]);
    let h3 = g.out_neighbors(h2[0], Labels::any());
    assert_eq!(h3, vec![d]);
    assert!(g.out_neighbors(h3[0], Labels::any()).is_empty());
}

#[test]
fn recursive_traversal_with_visited_set() {
    use std::collections::HashSet;

    let mut g = fresh("t-bfs");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let c = g.add_vertex("n", "c", ObjID::new(0)).unwrap();
    g.add_edge(a, "e", b).unwrap();
    g.add_edge(b, "e", c).unwrap();
    g.add_edge(c, "e", a).unwrap(); // cycle a -> b -> c -> a

    // Traverse outward from `a`, using the predicate filter to skip vertices
    // already visited so the cycle does not loop forever.
    let mut visited = HashSet::new();
    visited.insert(a.0);
    let mut frontier = vec![a];
    while let Some(v) = frontier.pop() {
        let view = g.vertex_view(v).unwrap();
        for n in view.out_neighbors_where(Labels::any(), |h| !visited.contains(&h.id().0)) {
            visited.insert(n.0);
            frontier.push(n);
        }
    }
    assert_eq!(visited.len(), 3);
}

#[test]
fn dsl_multi_hop_and_filter() {
    let mut g = fresh("t-dsl");
    let a = g.add_vertex("file", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("file", "b", ObjID::new(0)).unwrap();
    let t = g.add_vertex("tag", "thesis", ObjID::new(0)).unwrap();
    let proj = g.add_vertex("project", "p", ObjID::new(0)).unwrap();
    g.add_edge(a, "tagged", t).unwrap();
    g.add_edge(b, "tagged", t).unwrap();
    g.add_edge(t, "in_project", proj).unwrap();

    // Files tagged thesis: incoming `tagged` edges into the tag.
    let files = g.traversal().v(t).in_(Labels::these(&["tagged"])).to_ids();
    assert_eq!(files.len(), 2);

    // Two hops: file -> tag -> project.
    let projects = g
        .traversal()
        .v(a)
        .out(Labels::these(&["tagged"]))
        .out(Labels::these(&["in_project"]))
        .to_ids();
    assert_eq!(projects, vec![proj]);

    // Both files reach the same tag; dedup collapses, has_label confirms kind.
    let tags = g
        .traversal()
        .vs(&[a, b])
        .out(Labels::these(&["tagged"]))
        .dedup()
        .has_label("tag")
        .to_ids();
    assert_eq!(tags, vec![t]);

    assert_eq!(g.traversal().with_label("file").count(), 2);
}

#[test]
fn dsl_edge_steps() {
    let mut g = fresh("t-dsl-e");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    g.add_edge(a, "e", b).unwrap();

    let vs = g.traversal().v(a).out_e(Labels::any()).in_v().to_ids();
    assert_eq!(vs, vec![b]);
}

#[test]
fn add_edge_to_missing_vertex_errors() {
    let mut g = fresh("t-bad");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let bogus = crate::VertexId(9999);
    match g.add_edge(a, "e", bogus) {
        Err(GraphError::Twz(_)) => {}
        other => panic!("expected an error for missing endpoint, got {other:?}"),
    }
}
