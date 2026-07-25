//! Gremlin-subset DSL: step chaining, terminals, and multiset semantics.

use twizzler::object::ObjID;

use super::fresh;
use crate::{Graph, Labels, PropValue};

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
fn dsl_has_and_values() {
    let mut g = fresh("t-dsl-prop");
    let a = g.add_vertex("file", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("file", "b", ObjID::new(0)).unwrap();
    let c = g.add_vertex("file", "c", ObjID::new(0)).unwrap();
    g.set_vertex_prop(a, "size", PropValue::U64(4096)).unwrap();
    g.set_vertex_prop(b, "size", PropValue::U64(4096)).unwrap();
    g.set_vertex_prop(c, "size", PropValue::U64(99)).unwrap();
    // `a` also carries a different key; `c`'s size differs; nothing on the
    // "kind" key except `a`.
    g.set_vertex_prop(a, "kind", PropValue::str("doc")).unwrap();

    // has keeps exactly the size==4096 files.
    let m = g
        .traversal()
        .with_label("file")
        .has("size", PropValue::U64(4096))
        .to_ids();
    assert_eq!(m, vec![a, b]);

    // A key some vertices lack: only `a` has "kind".
    assert_eq!(
        g.traversal()
            .with_label("file")
            .has("kind", PropValue::str("doc"))
            .to_ids(),
        vec![a]
    );

    // values collects present values in order; `c` has a value too.
    let vals = g.traversal().with_label("file").values("size");
    assert_eq!(
        vals,
        vec![PropValue::U64(4096), PropValue::U64(4096), PropValue::U64(99)]
    );

    // values skips elements missing the key.
    assert_eq!(
        g.traversal().with_label("file").values("kind"),
        vec![PropValue::str("doc")]
    );
}

#[test]
fn dsl_has_composes_in_chain() {
    let mut g = fresh("t-dsl-chain");
    let f1 = g.add_vertex("file", "f1", ObjID::new(0)).unwrap();
    let f2 = g.add_vertex("file", "f2", ObjID::new(0)).unwrap();
    let t = g.add_vertex("tag", "thesis", ObjID::new(0)).unwrap();
    let p = g.add_vertex("project", "p", ObjID::new(0)).unwrap();
    g.add_edge(f1, "tagged", t).unwrap();
    g.add_edge(f2, "tagged", t).unwrap();
    g.add_edge(t, "in_project", p).unwrap();
    g.set_vertex_prop(p, "active", PropValue::Bool(true)).unwrap();

    // f1,f2 -> tag -> project, dedup the shared project, keep active ones.
    let active = g
        .traversal()
        .vs(&[f1, f2])
        .out(Labels::these(&["tagged"]))
        .out(Labels::these(&["in_project"]))
        .dedup()
        .has("active", PropValue::Bool(true))
        .to_ids();
    assert_eq!(active, vec![p]);

    // If the project were inactive, the chain yields nothing.
    g.set_vertex_prop(p, "active", PropValue::Bool(false)).unwrap();
    assert_eq!(
        g.traversal()
            .v(f1)
            .out(Labels::these(&["tagged"]))
            .out(Labels::these(&["in_project"]))
            .has("active", PropValue::Bool(true))
            .limit(5)
            .count(),
        0
    );
}

#[test]
fn dsl_has_after_reopen() {
    let name = "t-dsl-prop-reopen";
    let _ = Graph::reset(name);
    let ids = {
        let mut g = Graph::open_or_create(name).unwrap();
        let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
        let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
        g.set_vertex_prop(a, "k", PropValue::I64(1)).unwrap();
        g.set_vertex_prop(b, "k", PropValue::I64(2)).unwrap();
        (a, b)
    };
    let g = Graph::open_or_create(name).unwrap();
    assert_eq!(
        g.traversal().vertices().has("k", PropValue::I64(2)).to_ids(),
        vec![ids.1]
    );
    assert_eq!(
        g.traversal().vertices().values("k"),
        vec![PropValue::I64(1), PropValue::I64(2)]
    );
    let _ = Graph::reset(name);
}

#[test]
fn dsl_edge_has_and_values() {
    let mut g = fresh("t-dsl-eprop");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let c = g.add_vertex("n", "c", ObjID::new(0)).unwrap();
    let e1 = g.add_edge(a, "rel", b).unwrap();
    let _e2 = g.add_edge(a, "rel", c).unwrap();
    g.set_edge_prop(e1, "weight", PropValue::I64(5)).unwrap();
    // e2 has no weight.

    let heavy = g
        .traversal()
        .v(a)
        .out_e(Labels::any())
        .has("weight", PropValue::I64(5))
        .to_ids();
    assert_eq!(heavy, vec![e1]);

    assert_eq!(
        g.traversal().v(a).out_e(Labels::any()).values("weight"),
        vec![PropValue::I64(5)]
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
