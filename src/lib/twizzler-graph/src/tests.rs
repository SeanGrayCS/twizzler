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

/// Like `fresh`, but with a forced registry segment capacity, so sharding
/// tests can trigger rollover with a handful of inserts.
fn fresh_cap(name: &str, cap: usize) -> Graph {
    Graph::reset_with_capacity(name, cap).expect("reset graph");
    Graph::open_or_create_with_capacity(name, cap).expect("create graph")
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
fn vertex_index_find_delete_and_persist() {
    let name = "t-vindex";
    let _ = Graph::reset(name);
    let (t, deleted) = {
        let mut g = Graph::open_or_create(name).unwrap();
        let t = g.add_vertex("tag", "thesis", ObjID::new(0)).unwrap();
        let d = g.add_vertex("tag", "gone", ObjID::new(0)).unwrap();
        // Index lookup.
        assert_eq!(g.find_vertex("tag", "thesis"), Some(t));
        // Deleted vertices are not returned by the index lookup.
        g.delete_vertex(d).unwrap();
        assert_eq!(g.find_vertex("tag", "gone"), None);
        (t, d)
    };
    // The index persists: reopen and look up again.
    let g = Graph::open_or_create(name).unwrap();
    assert_eq!(g.find_vertex("tag", "thesis"), Some(t));
    assert_eq!(g.find_vertex("tag", "gone"), None);
    assert!(g.vertex_info(deleted).is_none());
    let _ = Graph::reset(name);
}

#[test]
fn delete_vertex_hides_it_and_incident_edges() {
    let mut g = fresh("t-delv");
    let a = g.add_vertex("file", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("file", "b", ObjID::new(0)).unwrap();
    let t = g.add_vertex("tag", "x", ObjID::new(0)).unwrap();
    g.add_edge(a, "tagged", t).unwrap();
    g.add_edge(b, "tagged", t).unwrap();

    g.delete_vertex(a).unwrap();
    assert!(g.vertex_info(a).is_none());
    assert_eq!(g.vertices_by_label("file"), vec![b]);
    // a's incident edge is hidden, so only b is tagged now.
    assert_eq!(g.in_neighbors(t, Labels::these(&["tagged"])), vec![b]);
    // Traversal from a deleted vertex yields nothing.
    assert!(g.vertex_view(a).is_none());
    assert_eq!(g.traversal().v(a).count(), 0);
}

#[test]
fn delete_edge_hides_it() {
    let mut g = fresh("t-dele");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let e = g.add_edge(a, "e", b).unwrap();

    g.delete_edge(e).unwrap();
    assert!(g.edge_info(e).is_none());
    assert!(g.out_neighbors(a, Labels::any()).is_empty());
    assert!(g
        .vertex_view(a)
        .unwrap()
        .out_edges(Labels::any())
        .is_empty());
}

#[test]
fn delete_persists_on_reopen() {
    let name = "t-delp";
    let _ = Graph::reset(name);
    let a = {
        let mut g = Graph::open_or_create(name).unwrap();
        let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
        g.delete_vertex(a).unwrap();
        a
    };
    let g = Graph::open_or_create(name).unwrap();
    assert!(g.vertex_info(a).is_none());
    let _ = Graph::reset(name);
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

#[test]
fn registry_rollover_segments_and_lookup() {
    let mut g = fresh_cap("t-shardv", 4);
    let mut ids = Vec::new();
    for i in 0..10 {
        ids.push(g.add_vertex("n", &format!("v{i}"), ObjID::new(0)).unwrap());
    }
    // ceil(10/4) = 3 vertex-registry segments.
    assert_eq!(g.registry_segments().0, 3);
    // Segment interior, last-slot, first-slot-of-next, and tail.
    for i in [0usize, 3, 4, 9] {
        let info = g.vertex_info(ids[i]).expect("vertex info");
        assert_eq!(info.name, format!("v{i}"));
    }
    assert_eq!(g.find_vertex("n", "v7"), Some(ids[7]));
    assert_eq!(g.vertices().len(), 10);
}

#[test]
fn registry_sharding_persists_and_appends() {
    let name = "t-shardp";
    Graph::reset_with_capacity(name, 4).expect("reset");
    let ids = {
        let mut g = Graph::open_or_create_with_capacity(name, 4).unwrap();
        (0..10)
            .map(|i| g.add_vertex("n", &format!("v{i}"), ObjID::new(0)).unwrap())
            .collect::<Vec<_>>()
    };

    // Reopen WITHOUT the capacity parameter: the persisted cap (4) governs.
    let mut g = Graph::open_or_create(name).unwrap();
    assert_eq!(g.registry_segments().0, 3);
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(g.vertex_info(*id).unwrap().name, format!("v{i}"));
    }

    // 10 -> 12 fills segment 3 exactly; the 13th opens segment 4.
    let v10 = g.add_vertex("n", "v10", ObjID::new(0)).unwrap();
    let v11 = g.add_vertex("n", "v11", ObjID::new(0)).unwrap();
    assert_eq!(g.registry_segments().0, 3);
    let v12 = g.add_vertex("n", "v12", ObjID::new(0)).unwrap();
    assert_eq!(g.registry_segments().0, 4);
    assert_eq!(g.find_vertex("n", "v10"), Some(v10));
    assert_eq!(g.find_vertex("n", "v11"), Some(v11));
    assert_eq!(g.vertex_info(v12).unwrap().name, "v12");
    let _ = Graph::reset(name);
}

#[test]
fn edge_and_label_registries_shard() {
    let mut g = fresh_cap("t-sharde", 4);
    let hub = g.add_vertex("hub", "h", ObjID::new(0)).unwrap();
    let mut spokes = Vec::new();
    let mut edges = Vec::new();
    for i in 0..6 {
        let s = g.add_vertex("spoke", &format!("s{i}"), ObjID::new(0)).unwrap();
        // Distinct edge labels push the label registry over a boundary too:
        // 2 vertex labels + 6 edge labels = 8 = 2 segments of 4.
        edges.push(g.add_edge(hub, &format!("l{i}"), s).unwrap());
        spokes.push(s);
    }
    assert_eq!(g.registry_segments().1, 2, "edge registry segments");
    assert_eq!(g.registry_segments().2, 2, "label registry segments");
    for i in [0usize, 3, 4, 5] {
        let info = g.edge_info(edges[i]).expect("edge info");
        assert_eq!(info.label, format!("l{i}"));
        assert_eq!(info.from, hub);
        assert_eq!(info.to, spokes[i]);
    }
    // Traversal liveness checks consult the sharded edge registry.
    assert_eq!(g.out_neighbors(hub, Labels::any()).len(), 6);
    assert_eq!(g.out_neighbors(hub, Labels::these(&["l5"])), vec![spokes[5]]);
}

#[test]
fn tombstones_across_segments() {
    let mut g = fresh_cap("t-shardd", 4);
    let mut vs = Vec::new();
    for i in 0..10 {
        vs.push(g.add_vertex("n", &format!("v{i}"), ObjID::new(0)).unwrap());
    }
    let mut es = Vec::new();
    for i in 0..5 {
        es.push(g.add_edge(vs[i], "e", vs[i + 1]).unwrap());
    }

    // Vertex id 8 lives in the third segment.
    g.delete_vertex(vs[8]).unwrap();
    assert!(g.vertex_info(vs[8]).is_none());
    assert_eq!(g.find_vertex("n", "v8"), None);
    assert_eq!(g.vertices().len(), 9);

    // Edge id 4 lives in the second segment.
    g.delete_edge(es[4]).unwrap();
    assert!(g.edge_info(es[4]).is_none());
    assert!(g.out_neighbors(vs[4], Labels::any()).is_empty());

    // Surrounding records are untouched.
    assert_eq!(g.edge_info(es[3]).unwrap().to, vs[4]);
    assert_eq!(g.vertex_info(vs[9]).unwrap().name, "v9");
}

#[test]
fn stale_v2_root_detected_and_resettable() {
    use naming::{static_naming_factory, GetFlags};
    use twizzler::object::{MapFlags, Object, ObjectBuilder};

    use crate::graph::{GraphRoot, MAGIC, VERSION};

    let name = "t-stalev2";
    let path = format!("data/{name}");
    let mut namer = static_naming_factory().expect("naming service available");
    let rw = MapFlags::READ | MapFlags::WRITE | MapFlags::PERSIST;

    // Plant a version-2 root at data/<name>. If a previous run left a graph
    // registered here, rewrite it in place (data/ names cannot be removed).
    if let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) {
        let mut root = Object::<GraphRoot>::map(node.id.into(), rw).unwrap();
        root.with_tx(|tx| {
            let mut b = tx.base_mut();
            b.magic = MAGIC;
            b.version = 2;
            Ok(())
        })
        .unwrap();
    } else {
        let root = ObjectBuilder::<GraphRoot>::default()
            .persist(true)
            .build(GraphRoot {
                magic: MAGIC,
                version: 2,
                seg_cap: 0,
                verts_raw: 0,
                edges_raw: 0,
                labels_raw: 0,
                vindex_raw: 0,
            })
            .unwrap();
        namer.put(&path, root.id()).unwrap();
    }

    // The guard refuses and reports both versions; the graph is not touched.
    match Graph::open_or_create(name) {
        Err(GraphError::StaleVersion { found, expected }) => {
            assert_eq!(found, 2);
            assert_eq!(expected, VERSION);
        }
        Ok(_) => panic!("expected StaleVersion, but the stale graph opened"),
        Err(e) => panic!("expected StaleVersion, got {e:?}"),
    }

    // Discarding is explicit — and works on the stale root.
    Graph::reset(name).expect("reset stale graph");
    let g = Graph::open_or_create(name).expect("open after reset");
    assert!(g.vertices().is_empty());
}

/// Unit: plain `reset` preserves the graph's stored segment capacity.
#[test]
fn reset_preserves_capacity() {
    let name = "t-capkeep";
    let _ = fresh_cap(name, 4);
    Graph::reset(name).expect("plain reset");
    let mut g = Graph::open_or_create(name).expect("reopen");
    for i in 0..5 {
        g.add_vertex("n", &format!("v{i}"), ObjID::new(0)).unwrap();
    }
    // Still capacity 4: 5 vertices -> 2 segments.
    assert_eq!(g.registry_segments().0, 2);
}

/// Unit: `reset_with_capacity` rebuilds the graph with the new capacity.
#[test]
fn reset_with_capacity_overrides() {
    let name = "t-capswap";
    let _ = fresh_cap(name, 4);
    Graph::reset_with_capacity(name, 2).expect("reset to cap 2");
    let mut g = Graph::open_or_create(name).expect("reopen");
    for i in 0..5 {
        g.add_vertex("n", &format!("v{i}"), ObjID::new(0)).unwrap();
    }
    // Now capacity 2: 5 vertices -> 3 segments.
    assert_eq!(g.registry_segments().0, 3);
}

#[test]
fn zero_capacity_rejected() {
    match Graph::open_or_create_with_capacity("t-zerocap", 0) {
        Err(GraphError::Twz(_)) => {}
        Ok(_) => panic!("capacity 0 must be rejected"),
        Err(e) => panic!("expected an argument error, got {e:?}"),
    }
    match Graph::reset_with_capacity("t-zerocap", 0) {
        Err(GraphError::Twz(_)) => {}
        Ok(_) => panic!("capacity 0 must be rejected"),
        Err(e) => panic!("expected an argument error, got {e:?}"),
    }
}
