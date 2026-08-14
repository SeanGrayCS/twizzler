//! Adjacency semantics: directionality, label and predicate filters,
//! multi-hop walks, self-loops, and parallel edges.

use twizzler::object::ObjID;

use super::fresh;
use crate::Labels;

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
    let ab = g.add_edge(a, "e", b).unwrap(); // a -> b
    let ca = g.add_edge(c, "e", a).unwrap(); // c -> a

    let view = g.vertex_view(a).unwrap();
    assert_eq!(view.out_neighbors(Labels::any()), vec![b]);
    assert_eq!(view.in_neighbors(Labels::any()), vec![c]);
    assert_eq!(view.both_neighbors(Labels::any()).len(), 2);

    // The edge-carrying form names the edge crossed, in each direction.
    assert_eq!(g.out_neighbors_with_edges(a, Labels::any()), vec![(ab, b)]);
    assert_eq!(g.in_neighbors_with_edges(a, Labels::any()), vec![(ca, c)]);
    // `both` is out then in, matching the plain form's documented ordering.
    assert_eq!(
        g.both_neighbors_with_edges(a, Labels::any()),
        vec![(ab, b), (ca, c)]
    );

    // The plain form is a projection of the pair form: same elements, same
    // order.
    for (pairs, plain) in [
        (
            g.out_neighbors_with_edges(a, Labels::any()),
            g.out_neighbors(a, Labels::any()),
        ),
        (
            g.in_neighbors_with_edges(a, Labels::any()),
            g.in_neighbors(a, Labels::any()),
        ),
        (
            g.both_neighbors_with_edges(a, Labels::any()),
            g.both_neighbors(a, Labels::any()),
        ),
    ] {
        assert_eq!(pairs.iter().map(|(_, n)| *n).collect::<Vec<_>>(), plain);
    }

    // Label filtering selects entries, so it reaches the pair form too.
    assert!(g
        .out_neighbors_with_edges(a, Labels::these(&["nope"]))
        .is_empty());
    assert_eq!(
        g.out_neighbors_with_edges(a, Labels::these(&["e"])),
        vec![(ab, b)]
    );
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

/// A self-loop appears once in the out list and once in the in list, so
/// `both` reports it twice; deleting the loop edge clears both directions.
#[test]
fn self_loop_edge() {
    let mut g = fresh("t-selfloop");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let e = g.add_edge(a, "e", a).unwrap();

    assert_eq!(g.out_neighbors(a, Labels::any()), vec![a]);
    assert_eq!(g.in_neighbors(a, Labels::any()), vec![a]);
    assert_eq!(g.both_neighbors(a, Labels::any()).len(), 2);
    let info = g.edge_info(e).unwrap();
    assert_eq!((info.from, info.to), (a, a));

    g.delete_edge(e).unwrap();
    assert!(g.out_neighbors(a, Labels::any()).is_empty());
    assert!(g.in_neighbors(a, Labels::any()).is_empty());
}

/// Parallel edges are distinct records, each with its own adjacency entry;
/// `dedup` collapses them at query level; deleting one leaves the other.
#[test]
fn parallel_edges() {
    let mut g = fresh("t-paredge");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let e1 = g.add_edge(a, "e", b).unwrap();
    let e2 = g.add_edge(a, "e", b).unwrap();
    assert_ne!(e1, e2);

    assert_eq!(g.out_neighbors(a, Labels::any()), vec![b, b]);
    assert_eq!(g.vertex_view(a).unwrap().out_edges(Labels::any()).len(), 2);
    assert_eq!(
        g.traversal().v(a).out(Labels::any()).dedup().to_ids(),
        vec![b]
    );

    // The plain form cannot say which edge reached which `b`; the pair form
    // can, in insertion order.
    assert_eq!(
        g.out_neighbors_with_edges(a, Labels::any()),
        vec![(e1, b), (e2, b)]
    );

    g.delete_edge(e1).unwrap();
    assert_eq!(g.out_neighbors(a, Labels::any()), vec![b]);
    assert!(g.edge_info(e2).is_some());
    // The survivor is identified, not merely counted.
    assert_eq!(
        g.out_neighbors_with_edges(a, Labels::any()),
        vec![(e2, b)],
        "the deleted edge leaves the pair form too"
    );
}
