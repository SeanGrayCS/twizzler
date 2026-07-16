//! Gremlin-subset DSL: step chaining, terminals, and multiset semantics.

use twizzler::object::ObjID;

use super::fresh;
use crate::Labels;

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
fn dsl_limit_first_and_infos() {
    let mut g = fresh("t-dsl-lim");
    for i in 0..5 {
        g.add_vertex("n", &format!("v{i}"), ObjID::new(0)).unwrap();
    }

    assert_eq!(g.traversal().with_label("n").limit(3).count(), 3);

    // Registry order is insertion order, so `first` is the earliest insert.
    let first = g.traversal().with_label("n").first().unwrap();
    assert_eq!(g.vertex_info(first).unwrap().name, "v0");

    let infos = g.traversal().with_label("n").limit(2).to_infos();
    assert_eq!(infos.len(), 2);
    assert_eq!(infos[0].name, "v0");
    assert_eq!(infos[1].name, "v1");

    assert_eq!(g.traversal().with_label("n").limit(0).count(), 0);
    assert!(g.traversal().with_label("missing").first().is_none());
}

#[test]
fn dsl_both_and_edge_directions() {
    let mut g = fresh("t-dsl-both");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let c = g.add_vertex("n", "c", ObjID::new(0)).unwrap();
    g.add_edge(a, "x", b).unwrap(); // a -> b
    g.add_edge(c, "y", a).unwrap(); // c -> a

    // `both` from a: outgoing neighbor first, then incoming.
    assert_eq!(g.traversal().v(a).both(Labels::any()).to_ids(), vec![b, c]);

    assert_eq!(g.traversal().v(a).in_e(Labels::any()).count(), 1);
    assert_eq!(g.traversal().v(a).both_e(Labels::any()).count(), 2);

    // Both endpoints of a's outgoing edge, from-then-to.
    let endpoints = g.traversal().v(a).out_e(Labels::any()).both_v().to_ids();
    assert_eq!(endpoints, vec![a, b]);

    // `out_v` of incoming edges returns the sources.
    assert_eq!(
        g.traversal().v(a).in_e(Labels::any()).out_v().to_ids(),
        vec![c]
    );
}

#[test]
fn dsl_has_name_and_filter() {
    let mut g = fresh("t-dsl-has");
    let d = g.add_vertex("file", "doc", ObjID::new(0)).unwrap();
    let i = g.add_vertex("file", "img", ObjID::new(0)).unwrap();
    let t = g.add_vertex("tag", "doc", ObjID::new(0)).unwrap();

    // has_name filters the current set; labels distinguish the two "doc"s.
    assert_eq!(g.traversal().vertices().has_name("doc").count(), 2);
    assert_eq!(
        g.traversal().with_label("file").has_name("doc").to_ids(),
        vec![d]
    );
    assert_eq!(
        g.traversal().with_label("tag").has_name("doc").to_ids(),
        vec![t]
    );

    // Arbitrary predicate over VertexInfo.
    let imgs = g
        .traversal()
        .with_label("file")
        .filter(|v| v.name.starts_with("i"))
        .to_ids();
    assert_eq!(imgs, vec![i]);

    // vs() keeps duplicates (multiset) until dedup.
    assert_eq!(g.traversal().vs(&[d, d]).count(), 2);
    assert_eq!(g.traversal().vs(&[d, d]).dedup().count(), 1);
}
