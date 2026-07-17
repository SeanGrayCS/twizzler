use twizzler::object::ObjID;

use super::fresh_cap;
use crate::{Graph, GraphError, Labels};

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
        let s = g
            .add_vertex("spoke", &format!("s{i}"), ObjID::new(0))
            .unwrap();
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
    assert_eq!(
        g.out_neighbors(hub, Labels::these(&["l5"])),
        vec![spokes[5]]
    );
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
