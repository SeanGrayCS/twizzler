//! Gremlin-subset DSL: step chaining, terminals, and multiset semantics.

use twizzler::object::ObjID;

use super::fresh;
use crate::{Graph, Labels, PathElem, PropValue};

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

/// `limit`, `first`, and `to_infos` terminals; empty-set behavior.
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

/// `both`, `in_e`/`both_e`, and edge-to-vertex steps (`out_v`, `both_v`) with
/// their ordering (out list before in list).
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

/// `has(key, value)` keeps exactly the matching vertices; `values(key)`
/// collects present values in traversal order, skipping absent ones.
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

/// `has` composes with `out`/`dedup`/`limit`: files tagged X whose project has
/// a property.
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

/// Property steps read persistent objects, so they behave identically after
/// reopen-by-name.
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

/// `has`/`values` on an edge traversal.
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

/// `vertex_path()` records the vertices traversed, one path per current
/// element, in first-seen order — including across a fork.
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
        .vertex_path();
    assert_eq!(paths, vec![vec![a, t1, p]]);

    // Fork: both two-hop routes reach p, each with its own path.
    let paths = g
        .traversal()
        .v(a)
        .out(Labels::these(&["tagged"]))
        .out(Labels::these(&["in_project"]))
        .vertex_path();
    assert_eq!(paths, vec![vec![a, t1, p], vec![a, t2, p]]);
}

/// Paths stay aligned with the current elements through filtering steps —
/// `dedup` and `limit` drop the corresponding paths, not others.
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
    assert_eq!(t.vertex_path(), vec![vec![a, hub], vec![b, hub]]);

    // dedup keeps the first occurrence — and its path.
    let paths = g
        .traversal()
        .vs(&[a, b])
        .out(Labels::any())
        .dedup()
        .vertex_path();
    assert_eq!(paths, vec![vec![a, hub]]);

    // limit truncates elements and paths together.
    let paths = g
        .traversal()
        .vs(&[a, b])
        .out(Labels::any())
        .limit(1)
        .vertex_path();
    assert_eq!(paths, vec![vec![a, hub]]);

    // A filter that drops everything leaves no paths.
    let paths = g
        .traversal()
        .vs(&[a, b])
        .out(Labels::any())
        .has_name("nope")
        .vertex_path();
    assert!(paths.is_empty());
}

/// `order_by_name` sorts ascending with id as the tiebreak, and composes with
/// `limit` to give top-N semantics.
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
    // Descending reverses the names only: the id tiebreak stays ascending, so
    // the two "dup" vertices keep insertion order and the tiebreak's meaning
    // is independent of the sort direction.
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

/// Ordering by a property value: elements missing the key sort last in both
/// directions, so `limit(n)` never surfaces them ahead of real data.
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
    // Newest-first with a cap.
    assert_eq!(
        g.traversal()
            .with_label("n")
            .order_by_prop_desc("at")
            .limit(2)
            .to_ids(),
        vec![a, c]
    );
}

/// Empty and deleted-start traversals have no paths, and an edge hop projected
/// to vertices contributes only the destination.
#[test]
fn dsl_path_edge_cases_and_edge_steps() {
    let mut g = fresh("t-dsl-path3");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    g.add_edge(a, "e", b).unwrap();

    // Empty traversal.
    assert!(g.traversal().with_label("missing").vertex_path().is_empty());

    // Projected to vertices, the edge hop contributes only the destination.
    assert_eq!(
        g.traversal().v(a).out_e(Labels::any()).in_v().vertex_path(),
        vec![vec![a, b]]
    );

    // Deleted start vertex yields nothing at all.
    g.delete_vertex(a).unwrap();
    assert!(g.traversal().v(a).vertex_path().is_empty());
    assert!(g
        .traversal()
        .v(a)
        .out(Labels::any())
        .vertex_path()
        .is_empty());
}

/// `has_name`, arbitrary `filter` predicates, and `vs` multiset semantics
/// (duplicates persist until `dedup`, as in Gremlin).
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

// ---------------------------------------------------------------------------
// Edge-carrying paths
// ---------------------------------------------------------------------------

/// Every path alternates vertex, edge, vertex, …, starting with a vertex — so a
/// path ending on a vertex has odd length and one ending on an edge has even
/// length. Checked structurally rather than against a literal, because the
/// invariant is what the other assertions are entitled to assume.
fn assert_alternates(paths: &[Vec<PathElem>]) {
    for p in paths {
        for (i, el) in p.iter().enumerate() {
            let ok = match el {
                PathElem::Vertex(_) => i % 2 == 0,
                PathElem::Edge(_) => i % 2 == 1,
            };
            assert!(ok, "path does not alternate at index {i}: {p:?}");
        }
    }
}

/// `path()` records the edges crossed as well as the vertices reached, and
/// `vertex_path()` projects back to the vertex-only answer.
#[test]
fn dsl_path_carries_edges() {
    let mut g = fresh("t-dsl-b6-carry");
    let a = g.add_vertex("file", "a", ObjID::new(0)).unwrap();
    let t = g.add_vertex("tag", "t", ObjID::new(0)).unwrap();
    let p = g.add_vertex("project", "p", ObjID::new(0)).unwrap();
    let e1 = g.add_edge(a, "tagged", t).unwrap();
    let e2 = g.add_edge(t, "in_project", p).unwrap();

    let paths = g
        .traversal()
        .v(a)
        .out(Labels::these(&["tagged"]))
        .out(Labels::these(&["in_project"]))
        .path();
    assert_eq!(
        paths,
        vec![vec![
            PathElem::Vertex(a),
            PathElem::Edge(e1),
            PathElem::Vertex(t),
            PathElem::Edge(e2),
            PathElem::Vertex(p),
        ]]
    );
    assert_alternates(&paths);
    assert_eq!(paths[0].len(), 5, "two hops: odd length, ends on a vertex");

    // The same chain, projected: exactly the vertex-only answer.
    assert_eq!(
        g.traversal()
            .v(a)
            .out(Labels::these(&["tagged"]))
            .out(Labels::these(&["in_project"]))
            .vertex_path(),
        vec![vec![a, t, p]]
    );

    // A chain ending on an edge has even length.
    let epaths = g.traversal().v(a).out_e(Labels::any()).path();
    assert_alternates(&epaths);
    assert_eq!(epaths, vec![vec![PathElem::Vertex(a), PathElem::Edge(e1)]]);

    // Degenerate cases.
    assert_eq!(g.traversal().v(a).path(), vec![vec![PathElem::Vertex(a)]]);
    assert_eq!(g.traversal().v(a).vertex_path(), vec![vec![a]]);
    assert!(g.traversal().with_label("missing").path().is_empty());
}

/// Parallel edges give paths that differ in their edge element — the case a
/// vertex-only path cannot express. A deleted edge leaves subsequent paths.
#[test]
fn dsl_path_distinguishes_parallel_edges() {
    let mut g = fresh("t-dsl-b6-parallel");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let e1 = g.add_edge(a, "reads", b).unwrap();
    let e2 = g.add_edge(a, "writes", b).unwrap();
    assert_ne!(e1, e2, "distinct edges, or the test proves nothing");

    let paths = g.traversal().v(a).out(Labels::any()).path();
    assert_alternates(&paths);
    assert_eq!(paths.len(), 2);

    // Vertex-only, the two are indistinguishable.
    assert_eq!(
        g.traversal().v(a).out(Labels::any()).vertex_path(),
        vec![vec![a, b], vec![a, b]]
    );

    // Carrying edges, they are not.
    let mut edges: Vec<_> = paths
        .iter()
        .map(|p| p[1].as_edge().expect("element 1 should be an edge"))
        .collect();
    edges.sort();
    let mut expected = vec![e1, e2];
    expected.sort();
    assert_eq!(edges, expected, "both edges appear, and they differ");

    // The deleted edge leaves the paths; the surviving one stays.
    g.delete_edge(e1).unwrap();
    let paths = g.traversal().v(a).out(Labels::any()).path();
    assert_eq!(
        paths,
        vec![vec![
            PathElem::Vertex(a),
            PathElem::Edge(e2),
            PathElem::Vertex(b)
        ]]
    );
}

/// After `both(..)` the crossed edge's endpoints say which way the hop went,
/// and `out_e().in_v()` produces the same path as `out()`.
#[test]
fn dsl_edge_path_direction_and_edge_steps() {
    let mut g = fresh("t-dsl-b6-dir");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let c = g.add_vertex("n", "c", ObjID::new(0)).unwrap();
    let out_e = g.add_edge(a, "e", b).unwrap(); // a -> b
    let in_e = g.add_edge(c, "e", a).unwrap(); // c -> a

    let paths = g.traversal().v(a).both(Labels::any()).path();
    assert_alternates(&paths);
    assert_eq!(paths.len(), 2);

    // Direction is recoverable from the edge element alone.
    let mut saw_out = false;
    let mut saw_in = false;
    for p in &paths {
        let PathElem::Edge(e) = p[1] else {
            panic!("element 1 should be an edge: {p:?}")
        };
        let info = g.edge_info(e).expect("edge is live");
        if info.from == a {
            saw_out = true;
            assert_eq!(e, out_e);
            assert_eq!(p[2], PathElem::Vertex(b));
        } else {
            saw_in = true;
            assert_eq!(info.to, a);
            assert_eq!(e, in_e);
            assert_eq!(p[2], PathElem::Vertex(c));
        }
    }
    assert!(saw_out && saw_in, "both directions crossed through both()");

    // The edge step ends the path on the edge, and extending it to the far
    // endpoint agrees with the plain neighbour hop.
    let via_edge = g
        .traversal()
        .v(a)
        .out_e(Labels::any())
        .in_v()
        .path();
    let via_hop = g.traversal().v(a).out(Labels::any()).path();
    assert_eq!(via_edge, via_hop, "two routes, one path");
    assert_eq!(
        via_hop,
        vec![vec![
            PathElem::Vertex(a),
            PathElem::Edge(out_e),
            PathElem::Vertex(b)
        ]]
    );
}

/// `repeat` carries edges but still dedups by vertex, so parallel edges to the
/// same neighbour yield one path, not two; truncation reporting is unaffected.
#[test]
fn dsl_repeat_carries_edges() {
    let mut g = fresh("t-dsl-b6-repeat");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let c = g.add_vertex("n", "c", ObjID::new(0)).unwrap();
    let e1 = g.add_edge(a, "e", b).unwrap();
    let _e2 = g.add_edge(a, "e", b).unwrap(); // parallel: same endpoints
    let e3 = g.add_edge(b, "e", c).unwrap();

    // One path per vertex reached, even though two edges reach `b`.
    let t = g.traversal().v(a).repeat_out(Labels::any()).times(1);
    let paths = t.path();
    assert_eq!(
        paths.len(),
        1,
        "repeat dedups by vertex (B3-AC3), so parallel edges collapse"
    );
    assert_alternates(&paths);
    assert_eq!(
        paths[0],
        vec![
            PathElem::Vertex(a),
            PathElem::Edge(e1),
            PathElem::Vertex(b)
        ],
        "the first edge in adjacency order wins — arbitrary but deterministic"
    );

    // Two hops carries both edges.
    assert_eq!(
        g.traversal()
            .v(a)
            .repeat_out(Labels::any())
            .times(2)
            .path(),
        vec![vec![
            PathElem::Vertex(a),
            PathElem::Edge(e1),
            PathElem::Vertex(b),
            PathElem::Edge(e3),
            PathElem::Vertex(c),
        ]]
    );

    // The depth cap still reports truncation, with edges in the paths.
    let t = g
        .traversal()
        .v(a)
        .repeat_out(Labels::any())
        .max_depth(1)
        .times(5);
    assert!(t.hit_depth_cap(), "stopped at the cap with somewhere to go");
    assert_alternates(&t.path());

    // And the strict form is still an error rather than a short answer.
    assert!(g
        .traversal()
        .v(a)
        .repeat_out(Labels::any())
        .strict_depth(1)
        .times(5)
        .is_err());
}
