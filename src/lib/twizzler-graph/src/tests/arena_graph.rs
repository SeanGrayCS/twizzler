//! Arena graphs are reset with [`Graph::reset_arena`] rather than
//! [`Graph::reset`]: the disk image survives between QEMU runs, and a plain
//! reset would rebuild a leftover graph as v3, so the second run of this file
//! would silently be testing v3 against v3 and passing.

use twizzler::object::ObjID;

use crate::{Graph, Labels, PropValue, VertexId};

const ARENA_CAP: usize = 4;

/// A clean graph on each layout, under distinct names.
fn pair(tag: &str) -> (Graph, Graph) {
    let legacy_name = format!("t-ab-v3-{tag}");
    let arena_name = format!("t-ab-v4-{tag}");
    Graph::reset(&legacy_name).expect("reset v3");
    Graph::reset_arena(&arena_name, ARENA_CAP).expect("reset v4");
    let legacy = Graph::open_or_create(&legacy_name).expect("open v3");
    let arena = Graph::open_or_create_arena(&arena_name, ARENA_CAP).expect("open v4");
    assert!(!legacy.is_arena(), "control must be on the v3 layout");
    assert!(arena.is_arena(), "subject must be on the v4 layout");
    (legacy, arena)
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
fn arena_matches_legacy_on_every_read_path() {
    let (mut legacy, mut arena) = pair("reads");
    let (hub_a, spokes_a) = build(&mut legacy);
    let (hub_b, spokes_b) = build(&mut arena);
    assert_eq!(hub_a, hub_b, "ids are append indices on both layouts");
    assert_eq!(spokes_a, spokes_b);

    // Whole-graph enumeration.
    assert_eq!(legacy.vertices(), arena.vertices());
    assert_eq!(
        legacy.vertices_by_label("spoke"),
        arena.vertices_by_label("spoke")
    );
    assert_eq!(legacy.vertices_by_label("hub"), arena.vertices_by_label("hub"));
    assert!(arena.vertices_by_label("nonesuch").is_empty());

    // Per-vertex records, including the target ObjID the arena record absorbed.
    for v in std::iter::once(hub_a).chain(spokes_a.iter().copied()) {
        let l = legacy.vertex_info(v).expect("v3 info");
        let a = arena.vertex_info(v).expect("v4 info");
        assert_eq!((l.label, l.name, l.target), (a.label, a.name, a.target));
    }

    // Name lookup through the index.
    for name in ["h", "s0", "s3"] {
        let lbl = if name == "h" { "hub" } else { "spoke" };
        assert_eq!(legacy.find_vertex(lbl, name), arena.find_vertex(lbl, name));
    }
    assert_eq!(arena.find_vertex("spoke", "missing"), None);

    // Adjacency, unfiltered and filtered, in both directions.
    for v in std::iter::once(hub_a).chain(spokes_a.iter().copied()) {
        assert_eq!(
            legacy.out_neighbors(v, Labels::any()),
            arena.out_neighbors(v, Labels::any()),
            "out-neighbours of {v:?}"
        );
        assert_eq!(
            legacy.in_neighbors(v, Labels::any()),
            arena.in_neighbors(v, Labels::any()),
            "in-neighbours of {v:?}"
        );
        assert_eq!(
            legacy.both_neighbors(v, Labels::any()),
            arena.both_neighbors(v, Labels::any()),
            "both-neighbours of {v:?} — including out-then-in ordering"
        );
    }
    for filter in [
        Labels::these(&["even"]),
        Labels::these(&["odd"]),
        Labels::these(&["even", "odd"]),
        Labels::these(&["next"]),
        Labels::these(&["absent"]),
    ] {
        assert_eq!(
            legacy.out_neighbors(hub_a, filter),
            arena.out_neighbors(hub_a, filter)
        );
    }
}

#[test]
fn arena_matches_legacy_on_properties_and_deletes() {
    let (mut legacy, mut arena) = pair("mutate");
    let (hub, spokes) = build(&mut legacy);
    build(&mut arena);

    for g in [&mut legacy, &mut arena] {
        g.set_vertex_prop(hub, "age", PropValue::I64(30)).unwrap();
        g.set_vertex_prop(hub, "ok", PropValue::Bool(true)).unwrap();
        g.set_vertex_prop(spokes[1], "age", PropValue::I64(7))
            .unwrap();
    }

    assert_eq!(
        legacy.get_vertex_prop(hub, "age"),
        arena.get_vertex_prop(hub, "age")
    );
    assert_eq!(legacy.vertex_props(hub), arena.vertex_props(hub));
    assert_eq!(
        legacy.get_vertex_prop(hub, "absent"),
        arena.get_vertex_prop(hub, "absent")
    );

    legacy.delete_vertex(spokes[1]).unwrap();
    arena.delete_vertex(spokes[1]).unwrap();

    assert!(legacy.vertex_info(spokes[1]).is_none());
    assert!(arena.vertex_info(spokes[1]).is_none());
    assert_eq!(legacy.vertices(), arena.vertices());
    assert_eq!(
        legacy.find_vertex("spoke", "s1"),
        arena.find_vertex("spoke", "s1")
    );
    assert_eq!(
        legacy.out_neighbors(hub, Labels::any()),
        arena.out_neighbors(hub, Labels::any()),
        "the deleted spoke is hidden from the hub on both layouts"
    );
    assert_eq!(
        legacy.vertex_props(spokes[1]),
        arena.vertex_props(spokes[1])
    );
    assert!(arena.get_vertex_prop(spokes[1], "age").is_none());
}

#[test]
fn arena_matches_legacy_when_an_edge_is_deleted() {
    let (mut legacy, mut arena) = pair("deledge");
    let (hub, spokes) = build(&mut legacy);
    build(&mut arena);

    // The hub->spokes[0] edge is edge id 0 on both layouts (append indices).
    let e0 = crate::EdgeId(0);
    assert_eq!(legacy.edge_info(e0).is_some(), arena.edge_info(e0).is_some());

    legacy.delete_edge(e0).unwrap();
    arena.delete_edge(e0).unwrap();

    assert!(legacy.edge_info(e0).is_none());
    assert!(arena.edge_info(e0).is_none());
    assert_eq!(
        legacy.out_neighbors(hub, Labels::any()),
        arena.out_neighbors(hub, Labels::any()),
        "the deleted edge's neighbour is dropped on both layouts"
    );
    assert_eq!(
        legacy.in_neighbors(spokes[0], Labels::any()),
        arena.in_neighbors(spokes[0], Labels::any())
    );
    // The endpoints themselves survive.
    assert_eq!(legacy.vertices(), arena.vertices());
    assert!(arena.vertex_info(spokes[0]).is_some());
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

#[test]
fn bulk_is_refused_on_the_arena_layout() {
    let name = "t-ab-bulk";
    Graph::reset_arena(name, ARENA_CAP).expect("reset v4");
    let mut g = Graph::open_or_create_arena(name, ARENA_CAP).expect("open v4");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();

    let r = g.bulk(|b| b.add_vertex("n", "b", ObjID::new(0)));
    assert!(r.is_err(), "bulk on v4 must error, not silently misplace");

    // The refusal leaves the graph untouched — no half-written vertex, and the
    // next real insert still gets the next id.
    assert_eq!(g.vertices(), vec![a]);
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    assert_eq!(b.0, a.0 + 1, "ids continue normally after a refused bulk");
    assert_eq!(g.find_vertex("n", "b"), Some(b));
}

#[test]
fn destroy_frees_the_graph_and_refuses_reopen() {
    let name = "t-ab-destroy";
    Graph::reset_arena(name, ARENA_CAP).expect("reset v4");
    let arenas = {
        let mut g = Graph::open_or_create_arena(name, ARENA_CAP).expect("open v4");
        let (hub, spokes) = build(&mut g);
        g.set_vertex_prop(hub, "k", PropValue::I64(1)).unwrap();
        g.set_edge_prop(crate::EdgeId(0), "w", PropValue::I64(2))
            .unwrap();
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
