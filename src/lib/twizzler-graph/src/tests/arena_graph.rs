//! `Graph` on the arena layout: read paths, mutation, and packing.
//!
//! Graphs here use a deliberately tiny `ARENA_CAP` so a handful of inserts
//! crosses an arena boundary; that is the seam most likely to break.

use twizzler::object::ObjID;

use crate::{Graph, Labels, PropValue, VertexId};

const ARENA_CAP: usize = 4;

/// A clean arena graph.
///
/// Uses [`Graph::reset_arena`] rather than [`Graph::reset`] deliberately: the
/// disk image survives between QEMU runs, and a plain reset would rebuild a
/// leftover graph in whatever format it already had.
fn fresh_arena(tag: &str) -> Graph {
    let name = format!("t-ab-{tag}");
    Graph::reset_arena(&name, ARENA_CAP).expect("reset v4");
    let g = Graph::open_or_create_arena(&name, ARENA_CAP).expect("open v4");
    assert!(g.is_arena(), "subject must be on the arena layout");
    g
}

/// A hub with four labelled spokes, plus a chain among the spokes. Returns the
/// ids so callers can compare across layouts — ids are append indices on both,
/// so they line up.
fn build(g: &mut Graph) -> (VertexId, Vec<VertexId>) {
    let hub = g.add_vertex("hub", "h", ObjID::new(0)).unwrap();
    let mut spokes = Vec::new();
    for i in 0..4 {
        let s = g
            .add_vertex("spoke", &format!("s{i}"), ObjID::new(7 + i as u128))
            .unwrap();
        let label = if i % 2 == 0 { "even" } else { "odd" };
        g.add_edge(hub, label, s).unwrap();
        spokes.push(s);
    }
    for i in 0..3 {
        g.add_edge(spokes[i], "next", spokes[i + 1]).unwrap();
    }
    (hub, spokes)
}

#[test]
fn arena_read_paths_return_the_expected_shape() {
    let mut g = fresh_arena("reads");
    let (hub, spokes) = build(&mut g);

    // Whole-graph enumeration, in insertion order.
    let all: Vec<VertexId> = std::iter::once(hub).chain(spokes.iter().copied()).collect();
    assert_eq!(g.vertices(), all);
    assert_eq!(g.vertices_by_label("spoke"), spokes);
    assert_eq!(g.vertices_by_label("hub"), vec![hub]);
    assert!(g.vertices_by_label("nonesuch").is_empty());

    // Per-vertex records, including the target ObjID the arena record absorbed.
    let h = g.vertex_info(hub).expect("hub info");
    assert_eq!((h.label.as_str(), h.name.as_str()), ("hub", "h"));
    assert_eq!(h.target, ObjID::new(0));
    for (i, &s) in spokes.iter().enumerate() {
        let info = g.vertex_info(s).expect("spoke info");
        assert_eq!(info.label, "spoke");
        assert_eq!(info.name, format!("s{i}"));
        assert_eq!(
            info.target,
            ObjID::new(7 + i as u128),
            "the arena record carries target_raw inline"
        );
    }

    // Name lookup through the index.
    assert_eq!(g.find_vertex("hub", "h"), Some(hub));
    for (i, &s) in spokes.iter().enumerate() {
        assert_eq!(g.find_vertex("spoke", &format!("s{i}")), Some(s));
    }
    assert_eq!(g.find_vertex("spoke", "missing"), None);
    assert_eq!(g.find_vertex("nonesuch", "h"), None, "label is part of the key");

    // Adjacency. The hub points at every spoke; the spokes form a chain.
    assert_eq!(g.out_neighbors(hub, Labels::any()), spokes);
    assert!(g.in_neighbors(hub, Labels::any()).is_empty());
    for (i, &s) in spokes.iter().enumerate() {
        let out = g.out_neighbors(s, Labels::any());
        let expected_out: Vec<VertexId> = spokes.get(i + 1).copied().into_iter().collect();
        assert_eq!(out, expected_out, "chain step from s{i}");

        // Hub first (added in the spoke loop), then the chain predecessor.
        let mut expected_in = vec![hub];
        if i > 0 {
            expected_in.push(spokes[i - 1]);
        }
        assert_eq!(g.in_neighbors(s, Labels::any()), expected_in, "in of s{i}");

        // `both` is out-then-in, and that ordering is part of the contract.
        let mut expected_both = expected_out.clone();
        expected_both.extend(expected_in);
        assert_eq!(g.both_neighbors(s, Labels::any()), expected_both);
    }

    // Label filters, including one that matches nothing.
    assert_eq!(
        g.out_neighbors(hub, Labels::these(&["even"])),
        vec![spokes[0], spokes[2]]
    );
    assert_eq!(
        g.out_neighbors(hub, Labels::these(&["odd"])),
        vec![spokes[1], spokes[3]]
    );
    assert_eq!(
        g.out_neighbors(hub, Labels::these(&["even", "odd"])),
        spokes,
        "a multi-label filter unions, preserving insertion order"
    );
    assert!(g.out_neighbors(hub, Labels::these(&["next"])).is_empty());
    assert!(g.out_neighbors(hub, Labels::these(&["absent"])).is_empty());
}

/// Properties and tombstones — the paths most likely to break, since the arena
/// record holds `props_raw` inline where v3 kept it in a registry mirror.
#[test]
fn arena_properties_and_vertex_deletes() {
    let mut g = fresh_arena("mutate");
    let (hub, spokes) = build(&mut g);

    g.set_vertex_prop(hub, "age", PropValue::I64(30)).unwrap();
    g.set_vertex_prop(hub, "ok", PropValue::Bool(true)).unwrap();
    g.set_vertex_prop(spokes[1], "age", PropValue::I64(7))
        .unwrap();

    assert_eq!(g.get_vertex_prop(hub, "age"), Some(PropValue::I64(30)));
    assert_eq!(g.get_vertex_prop(hub, "ok"), Some(PropValue::Bool(true)));
    assert_eq!(g.get_vertex_prop(hub, "absent"), None);
    assert_eq!(g.get_vertex_prop(spokes[1], "age"), Some(PropValue::I64(7)));
    assert_eq!(
        g.vertex_props(hub).len(),
        2,
        "both keys, and no leakage from the other vertex's property object"
    );

    g.delete_vertex(spokes[1]).unwrap();

    assert!(g.vertex_info(spokes[1]).is_none());
    assert_eq!(g.find_vertex("spoke", "s1"), None);
    assert_eq!(
        g.vertices(),
        vec![hub, spokes[0], spokes[2], spokes[3]],
        "the tombstoned spoke leaves a gap rather than renumbering"
    );
    assert_eq!(
        g.out_neighbors(hub, Labels::any()),
        vec![spokes[0], spokes[2], spokes[3]],
        "A4-F2: a dead *neighbour* is hidden from the hub's adjacency"
    );
    assert!(
        g.out_neighbors(spokes[0], Labels::any()).is_empty(),
        "s0's only out-edge pointed at the deleted s1"
    );
    assert!(g.vertex_props(spokes[1]).is_empty());
    assert!(g.get_vertex_prop(spokes[1], "age").is_none());
}

/// Edge deletion. The arena store hides tombstoned *vertices* but knows nothing
/// about the edge registry, so without an explicit `is_edge_alive` filter in
/// `Graph` a deleted edge would still yield its neighbour — a wrong answer
/// visible only after a delete.
#[test]
fn arena_hides_a_deleted_edge() {
    let mut g = fresh_arena("deledge");
    let (hub, spokes) = build(&mut g);

    let e0 = g
        .vertex_view(hub)
        .expect("hub view")
        .out_edges(Labels::any())
        .first()
        .copied()
        .expect("hub has an outgoing edge");
    let info = g.edge_info(e0).expect("edge 0 is live");
    assert_eq!((info.from, info.to), (hub, spokes[0]));

    g.delete_edge(e0).unwrap();

    assert!(g.edge_info(e0).is_none());
    assert_eq!(
        g.out_neighbors(hub, Labels::any()),
        spokes[1..].to_vec(),
        "the deleted edge's neighbour is dropped from the hub"
    );
    assert!(
        g.in_neighbors(spokes[0], Labels::any()).is_empty(),
        "and from the far side's inbound list"
    );

    // The endpoints themselves survive — deleting an edge is not deleting a
    // vertex, which is the mistake the tombstone semantics invite.
    assert_eq!(
        g.vertices(),
        std::iter::once(hub).chain(spokes.iter().copied()).collect::<Vec<_>>()
    );
    assert!(g.vertex_info(spokes[0]).is_some());
}

#[test]
fn arena_survives_bulk_vertex_deletion() {
    const N: usize = 70;
    let mut g = fresh_arena("bulkdel");

    for i in 0..N {
        g.add_vertex("v", &format!("v{i}"), ObjID::new(i as u128))
            .unwrap();
    }
    // Chain every vertex to one well outside its own arena, so records carry
    // cross-arena adjacency at delete time.
    for i in 0..N - 1 {
        g.add_edge(VertexId(i as u64), "next", VertexId(((i + 37) % N) as u64))
            .unwrap();
    }
    for i in (0..N).step_by(7) {
        g.delete_vertex(VertexId(i as u64)).unwrap();
    }

    for i in 0..N {
        let deleted = i % 7 == 0;
        let got = g.vertex_info(VertexId(i as u64)).is_some();
        assert_eq!(
            got,
            !deleted,
            "vertex_info for v{i}: got alive={got}, expected {}",
            !deleted
        );
    }
    let expected: Vec<VertexId> = (0..N)
        .filter(|i| i % 7 != 0)
        .map(|i| VertexId(i as u64))
        .collect();
    assert_eq!(g.vertices(), expected, "the scan agrees with the bookkeeping");

    // A dead vertex is hidden from its *neighbours'* lists too — the direction
    // that only breaks after a delete.
    for i in (0..N).step_by(7).take(4) {
        let src = (i + N - 37) % N;
        if src % 7 == 0 || src >= N - 1 {
            continue;
        }
        assert!(
            !g.out_neighbors(VertexId(src as u64), Labels::any())
                .contains(&VertexId(i as u64)),
            "v{src} still lists deleted v{i}"
        );
    }
}

#[test]
fn arena_packs_vertices_and_adds_no_object_per_edge() {
    let name = "t-ab-objects";
    Graph::reset_arena(name, ARENA_CAP).expect("reset v4");
    let mut g = Graph::open_or_create_arena(name, ARENA_CAP).expect("open v4");

    let mut ids = Vec::new();
    for i in 0..12 {
        ids.push(
            g.add_vertex("n", &format!("v{i}"), ObjID::new(0))
                .unwrap(),
        );
    }
    assert_eq!(
        g.arena_count(),
        3,
        "12 vertices / cap {ARENA_CAP} = 3 arenas (v3 would be 36 objects)"
    );

    for i in 0..11 {
        g.add_edge(ids[i], "e", ids[i + 1]).unwrap();
    }
    assert_eq!(
        g.arena_count(),
        3,
        "edges allocate inside existing arenas — v3 would have added 11 objects"
    );

    assert_eq!(g.arena_sync_count(), 0, "nothing synced before sync()");
    g.sync().unwrap();
    assert_eq!(g.arena_sync_count(), 3, "one sync per arena, not per record");
}

/// Ids are append indices and continue without gaps — the invariant that
/// `bulk_is_refused_on_the_arena_layout` used to guard from the other side.
#[test]
fn arena_ids_are_gapless_append_indices() {
    let name = "t-ab-ids";
    Graph::reset_arena(name, ARENA_CAP).expect("reset v4");
    let mut g = Graph::open_or_create_arena(name, ARENA_CAP).expect("open v4");

    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    assert_eq!(g.vertices(), vec![a]);
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    assert_eq!(b.0, a.0 + 1, "ids continue without gaps");
    assert_eq!(g.find_vertex("n", "b"), Some(b));

    // A delete tombstones rather than freeing the id, so the next insert does
    // not reuse it. `gstress` asserts the same property as "vertex id drift".
    g.delete_vertex(a).unwrap();
    let c = g.add_vertex("n", "c", ObjID::new(0)).unwrap();
    assert_eq!(c.0, b.0 + 1, "a deleted id is never reused");
}

#[test]
fn destroy_frees_the_graph_and_refuses_reopen() {
    let name = "t-ab-destroy";
    Graph::reset_arena(name, ARENA_CAP).expect("reset v4");
    let arenas = {
        let mut g = Graph::open_or_create_arena(name, ARENA_CAP).expect("open v4");
        let (hub, spokes) = build(&mut g);
        g.set_vertex_prop(hub, "k", PropValue::I64(1)).unwrap();
        let e0 = g
            .vertex_view(hub)
            .expect("hub view")
            .out_edges(Labels::any())
            .first()
            .copied()
            .expect("hub has an outgoing edge");
        g.set_edge_prop(e0, "w", PropValue::I64(2)).unwrap();
        assert!(!spokes.is_empty());
        g.sync().unwrap();
        g.arena_count()
    };
    assert!(arenas >= 1);

    let freed = Graph::destroy(name).expect("destroy");
    // Arenas, both store registries, the edge/label/vertex registries, the
    // index, and both property objects — comfortably more than the arenas.
    assert!(
        freed > arenas,
        "destroy freed {freed} objects, expected more than {arenas} arenas"
    );

    // The name is still bound (data/ entries cannot be removed), but the root
    // is no longer a graph, so opening refuses instead of reading freed ids.
    assert!(
        Graph::open_or_create(name).is_err(),
        "a destroyed graph must not open"
    );
    // Idempotent.
    assert_eq!(Graph::destroy(name).expect("second destroy"), 0);

    // The name is reusable via an explicit reset, which rebuilds in place.
    // This is why `destroy` marks the root rather than zeroing it: the root
    // survives in the disk image, so a zeroed one would burn the name in every
    // future boot too.
    Graph::reset_arena(name, ARENA_CAP).expect("rebuild after destroy");
    let g = Graph::open_or_create(name).expect("reopen after rebuild");
    assert!(g.is_arena());
    assert!(g.vertices().is_empty());
}

#[test]
fn destroy_cycles_do_not_accumulate() {
    let name = "t-ab-cycle";
    let mut freed_each = Vec::new();
    for _ in 0..4 {
        Graph::reset_arena(name, ARENA_CAP).expect("reset");
        {
            let mut g = Graph::open_or_create_arena(name, ARENA_CAP).expect("open");
            build(&mut g);
            g.sync().unwrap();
        }
        freed_each.push(Graph::destroy(name).expect("destroy"));
    }
    // Every cycle frees the same amount: the workload is identical, so a
    // growing figure would mean a cycle is inheriting the previous one's
    // objects instead of freeing its own.
    assert!(
        freed_each.windows(2).all(|w| w[0] == w[1]),
        "per-cycle frees drifted: {freed_each:?}"
    );
    assert!(freed_each[0] > 0);
}

#[test]
fn arena_graph_reopens_in_its_own_format() {
    let name = "t-ab-reopen";
    Graph::reset_arena(name, ARENA_CAP).expect("reset v4");
    let ids = {
        let mut g = Graph::open_or_create_arena(name, ARENA_CAP).expect("open v4");
        let (hub, spokes) = build(&mut g);
        g.set_vertex_prop(hub, "k", PropValue::I64(1)).unwrap();
        g.sync().unwrap();
        (hub, spokes)
    };

    // Re-opened through the *plain* constructor: the root's version decides.
    let g = Graph::open_or_create(name).expect("reopen by name");
    assert!(g.is_arena(), "stored format governs, not the constructor");
    assert_eq!(g.vertex_info(ids.0).unwrap().name, "h");
    assert_eq!(g.out_neighbors(ids.0, Labels::any()).len(), 4);
    assert_eq!(g.get_vertex_prop(ids.0, "k"), Some(PropValue::I64(1)));
    assert_eq!(g.in_neighbors(ids.1[3], Labels::any()).len(), 2);
}
