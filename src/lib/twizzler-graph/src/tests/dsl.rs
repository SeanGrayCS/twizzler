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
fn dsl_path_records_traversal() {
    let mut g = fresh("t-dsl-path");
    let a = g.add_vertex("file", "a", ObjID::new(0)).unwrap();
    let t1 = g.add_vertex("tag", "t1", ObjID::new(0)).unwrap();
    let t2 = g.add_vertex("tag", "t2", ObjID::new(0)).unwrap();
    let p = g.add_vertex("project", "p", ObjID::new(0)).unwrap();
    g.add_edge(a, "tagged", t1).unwrap();
    g.add_edge(a, "tagged", t2).unwrap();
    g.add_edge(t1, "in_project", p).unwrap();
    g.add_edge(t2, "in_project", p).unwrap();

    // Single branch first: a -> t1 -> p.
    let paths = g
        .traversal()
        .v(a)
        .out(Labels::these(&["tagged"]))
        .has_name("t1")
        .out(Labels::these(&["in_project"]))
        .path();
    assert_eq!(paths, vec![vec![a, t1, p]]);

    // Fork: both two-hop routes reach p, each with its own path.
    let paths = g
        .traversal()
        .v(a)
        .out(Labels::these(&["tagged"]))
        .out(Labels::these(&["in_project"]))
        .path();
    assert_eq!(paths, vec![vec![a, t1, p], vec![a, t2, p]]);
}

#[test]
fn dsl_path_stays_aligned_through_dedup_and_limit() {
    let mut g = fresh("t-dsl-path2");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let hub = g.add_vertex("n", "hub", ObjID::new(0)).unwrap();
    g.add_edge(a, "e", hub).unwrap();
    g.add_edge(b, "e", hub).unwrap();

    // Both starts reach hub: two elements, two paths.
    let t = g.traversal().vs(&[a, b]).out(Labels::any());
    assert_eq!(t.path(), vec![vec![a, hub], vec![b, hub]]);

    // dedup keeps the first occurrence — and its path.
    let paths = g
        .traversal()
        .vs(&[a, b])
        .out(Labels::any())
        .dedup()
        .path();
    assert_eq!(paths, vec![vec![a, hub]]);

    // limit truncates elements and paths together.
    let paths = g
        .traversal()
        .vs(&[a, b])
        .out(Labels::any())
        .limit(1)
        .path();
    assert_eq!(paths, vec![vec![a, hub]]);

    // A filter that drops everything leaves no paths.
    let paths = g
        .traversal()
        .vs(&[a, b])
        .out(Labels::any())
        .has_name("nope")
        .path();
    assert!(paths.is_empty());
}

#[test]
fn dsl_order_by_name() {
    let mut g = fresh("t-dsl-order");
    // Inserted out of order; note two vertices share the name "dup".
    let c = g.add_vertex("n", "cherry", ObjID::new(0)).unwrap();
    let a = g.add_vertex("n", "apple", ObjID::new(0)).unwrap();
    let d1 = g.add_vertex("n", "dup", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "banana", ObjID::new(0)).unwrap();
    let d2 = g.add_vertex("n", "dup", ObjID::new(0)).unwrap();

    let ids = g.traversal().with_label("n").order_by_name().to_ids();
    assert_eq!(ids, vec![a, b, c, d1, d2], "name asc, ties by id");

    let names: Vec<String> = g
        .traversal()
        .with_label("n")
        .order_by_name()
        .to_infos()
        .into_iter()
        .map(|i| i.name)
        .collect();
    assert_eq!(names, vec!["apple", "banana", "cherry", "dup", "dup"]);

    // Top-2.
    assert_eq!(
        g.traversal().with_label("n").order_by_name().limit(2).to_ids(),
        vec![a, b]
    );
    // Descending reverses the *names* only: the id tiebreak stays ascending,
    // so the two "dup" vertices keep insertion order. This is LDBC's own
    // convention ("... desc, then id asc"), and it keeps the tiebreak's
    // meaning independent of the sort direction.
    assert_eq!(
        g.traversal().with_label("n").order_by_name_desc().to_ids(),
        vec![d1, d2, c, b, a]
    );
    assert_eq!(
        g.traversal()
            .with_label("n")
            .order_by_name_desc()
            .limit(1)
            .to_ids(),
        vec![d1],
        "first of the tied 'dup' pair, not the last"
    );
}

#[test]
fn dsl_order_by_prop() {
    let mut g = fresh("t-dsl-orderp");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let c = g.add_vertex("n", "c", ObjID::new(0)).unwrap();
    let none = g.add_vertex("n", "none", ObjID::new(0)).unwrap();
    g.set_vertex_prop(a, "at", PropValue::U64(30)).unwrap();
    g.set_vertex_prop(b, "at", PropValue::U64(10)).unwrap();
    g.set_vertex_prop(c, "at", PropValue::U64(20)).unwrap();
    // `none` deliberately has no "at" property.

    assert_eq!(
        g.traversal().with_label("n").order_by_prop("at").to_ids(),
        vec![b, c, a, none],
        "ascending, missing key last"
    );
    assert_eq!(
        g.traversal()
            .with_label("n")
            .order_by_prop_desc("at")
            .to_ids(),
        vec![a, c, b, none],
        "descending, missing key still last"
    );
    // Newest-first with a cap — the LDBC short-read shape.
    assert_eq!(
        g.traversal()
            .with_label("n")
            .order_by_prop_desc("at")
            .limit(2)
            .to_ids(),
        vec![a, c]
    );
}

#[test]
fn dsl_path_edge_cases_and_edge_steps() {
    let mut g = fresh("t-dsl-path3");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    g.add_edge(a, "e", b).unwrap();

    // Empty traversal.
    assert!(g.traversal().with_label("missing").path().is_empty());

    // Vertex-only paths: the edge hop adds only the destination vertex.
    assert_eq!(
        g.traversal().v(a).out_e(Labels::any()).in_v().path(),
        vec![vec![a, b]]
    );

    // Deleted start vertex yields nothing at all.
    g.delete_vertex(a).unwrap();
    assert!(g.traversal().v(a).path().is_empty());
    assert!(g.traversal().v(a).out(Labels::any()).path().is_empty());
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
