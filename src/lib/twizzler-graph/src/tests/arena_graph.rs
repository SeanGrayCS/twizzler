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
    Graph::open_or_create_arena(&name, ARENA_CAP).expect("open v4")
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

/// Edge deletion — a deleted edge must not yield its neighbour, a wrong answer
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
fn arena_packs_records_including_edges() {
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
    // 12 vertices + 11 edges = 23 records, at cap 4 = 6 arenas.
    assert_eq!(
        g.arena_count(),
        6,
        "an edge is a record and occupies a slot: ceil((12+11)/{ARENA_CAP})"
    );
    assert!(
        g.arena_count() * ARENA_CAP >= 23,
        "arenas must cover every record"
    );
    assert!(
        g.arena_count() < 47,
        "still far below v3's object-per-entity cost — 6 against 47"
    );

    assert_eq!(g.arena_sync_count(), 0, "nothing synced before sync()");
    let arenas = g.arena_count();
    g.sync().unwrap();
    assert_eq!(
        g.arena_sync_count(),
        arenas,
        "one sync per arena, not per record"
    );
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

/// Asserting it now means a v5 regression is attributable to v5. Asserting it
/// afterwards would only tell us the property is absent, not when it went.
///
/// The counter is the whole point: "does not resolve the record" is otherwise a
/// claim about mechanism that passes by inspection.
#[test]
fn scanning_vertices_does_not_touch_any_record() {
    let mut g = fresh_arena("scan-cost");
    let (hub, spokes) = build(&mut g);
    g.set_vertex_prop(hub, "k", PropValue::I64(1)).unwrap();

    // Deletes matter here: a tombstoned record is the case most likely to
    // tempt an implementation into resolving, since liveness is what the scan
    // filters on.
    g.delete_vertex(spokes[3]).unwrap();

    g.reset_record_touches();
    let live = g.vertices();
    assert_eq!(
        g.record_touches(),
        0,
        "vertices() resolved {} record(s); liveness must come from the mirror",
        g.record_touches()
    );
    assert_eq!(live.len(), 4, "hub plus three surviving spokes");

    // Control. `vertices_by_label` filters on the label, which the mirror
    // does not carry, so it resolves every record — `vertex_label` goes through
    // `with_vertex`. Without this arm a zero above would be unfalsifiable: it
    // would look identical whether the scan avoids records or the counter is
    // simply never incremented.
    //
    // The label must be one `build` actually interned ("spoke", not "n"):
    // `vertices_by_label` returns early on an unknown label without reaching
    // the store, which would make the control pass for the wrong reason.
    g.reset_record_touches();
    let spoke_ids = g.vertices_by_label("spoke");
    assert_eq!(spoke_ids.len(), 3, "one of the four spokes is deleted");
    assert!(
        g.record_touches() >= 4,
        "control: a label scan resolves records (touched {}), so the zero \
         above is a real property and not a dead counter",
        g.record_touches()
    );
}

/// What a build must do with a predecessor format it cannot walk.
///
/// This test has inverted once already, and the inversion is the lesson.
///
/// It was written when format 7 *was* reclaimable — 7 → 8 moved only a trailing
/// `GraphRoot` field, leaving every object exactly where the walker expected —
/// and it existed because that bump had silently turned reclaim into a leak:
/// the inventory arm matched `VERSION_ARENA` by name, so changing the constant
/// stopped it matching, with no error.
///
/// Format 9 moved the *record* layout, and the inventory walk reads records.
/// So 7 stopped being walkable and the correct answer flipped from "free it" to
/// "refuse". The rule was never about which versions are in the set — it is
/// that a build must never guess. Three outcomes are acceptable in principle
/// and only two are acceptable in practice:
///
/// - free it, when the object graph is genuinely walkable;
/// - refuse loudly, when it is not;
/// - and never `Ok(0)`, which reads as "there was nothing to free" and is how
///   the original regression hid.
///
/// The complement matters as much: `reset` must still succeed, or the name is
/// stranded forever. `data/` entries cannot be unbound on this build, so a
/// format the engine can neither open nor reset is a permanently burned name,
/// in this boot and every future one.
///
/// The graph is built normally and then *downgraded* by rewriting its root's
/// version, which is the only way to obtain a previous-format graph from a
/// build that can no longer write one.
#[test]
fn an_unwalkable_predecessor_is_refused_loudly_and_stays_recoverable() {
    use naming::{static_naming_factory, GetFlags};
    use twizzler::object::{MapFlags, Object};

    use crate::graph::{GraphRoot, VERSION_ARENA_NOCAP};

    let name = "t-ab-reclaim-old";
    Graph::reset_arena(name, ARENA_CAP).expect("reset");
    let owned = {
        let mut g = Graph::open_or_create_arena(name, ARENA_CAP).expect("open");
        let (hub, spokes) = build(&mut g);
        g.set_vertex_prop(hub, "k", PropValue::I64(1)).unwrap();
        g.set_vertex_prop(spokes[0], "k", PropValue::I64(2)).unwrap();
        g.sync().unwrap();
        g.owned_object_ids().len()
    };
    assert!(owned > 0, "the graph owns something to reclaim");

    // Downgrade the root in place: same objects, previous format.
    let path = format!("data/{name}");
    let mut namer = static_naming_factory().expect("naming service available");
    let node = namer.get(&path, GetFlags::FOLLOW_SYMLINK).expect("registered");
    let mut root = Object::<GraphRoot>::map(
        node.id.into(),
        MapFlags::READ | MapFlags::WRITE | MapFlags::PERSIST,
    )
    .expect("map root");
    root.with_tx(|tx| {
        tx.base_mut().version = VERSION_ARENA_NOCAP;
        Ok(())
    })
    .expect("downgrade");

    let _ = owned;

    // Reading it must refuse — that guard is meant to be strict.
    assert!(
        Graph::open_or_create(name).is_err(),
        "a previous format must not open"
    );

    // `destroy` must refuse *loudly*, not return Ok(0). Format 9 moved the
    // record layout and the inventory walk reads records, so walking a format-7
    // graph would free ids read at the wrong offsets — mis-freeing, which is
    // strictly worse than leaking. But a silent `Ok(0)` would be worse still:
    // indistinguishable from "there was nothing to free", which is how the
    // 7 → 8 bump turned into a leak nobody noticed.
    match Graph::destroy(name) {
        Err(crate::GraphError::StaleVersion { found, expected }) => {
            assert_eq!(found, VERSION_ARENA_NOCAP);
            assert_eq!(expected, crate::graph::VERSION_ARENA);
        }
        Ok(n) => panic!("destroy silently reported {n} objects freed"),
        Err(e) => panic!("expected StaleVersion, got {e:?}"),
    }

    Graph::reset_arena(name, ARENA_CAP).expect("reset must recover the name");
    let g = Graph::open_or_create_arena(name, ARENA_CAP).expect("reopen after reset");
    assert!(g.vertices().is_empty(), "rebuilt empty and usable");
}

/// The load-bearing assertion is about where new records go, not where old
/// ones are. Existing records never move regardless, so an implementation that
/// persisted nothing would still pass a test that only re-read old placement.
#[test]
fn arena_cap_survives_a_reopen_that_does_not_supply_one() {
    let name = "t-ab-cap-persist";
    Graph::reset_arena(name, ARENA_CAP).expect("reset");

    let before = {
        let mut g = Graph::open_or_create_arena(name, ARENA_CAP).expect("open");
        for i in 0..10 {
            g.add_vertex("n", &format!("v{i}"), ObjID::new(0))
                .expect("add");
        }
        g.sync().expect("sync");
        // `FillTo` packs `cap` per arena before opening another: 4 + 4 + 2.
        assert_eq!(g.arena_count(), 3, "10 vertices at cap 4 fill 3 arenas");
        g.arena_vertex_counts().1
    };

    // Reopened through the constructor that supplies no cap.
    let mut g = Graph::open_or_create(name).expect("reopen without a cap");
    assert_eq!(
        g.arena_vertex_counts().1,
        before,
        "existing records do not move on reopen"
    );

    // The third arena holds 2 of 4. Four more inserts must fill it and then
    // roll into a fourth. Under the defect the store reopens at
    // DEFAULT_ARENA_CAP (16384), so all four land in arena 2 and the count
    // stays at 3 — which is exactly the silent relayout being pinned here.
    for i in 10..14 {
        g.add_vertex("n", &format!("w{i}"), ObjID::new(0))
            .expect("add after reopen");
    }
    assert_eq!(
        g.arena_count(),
        4,
        "inserts after reopen must follow the persisted cap, not DEFAULT_ARENA_CAP"
    );
    assert_eq!(
        g.arena_vertex_counts().1,
        vec![4, 4, 4, 2],
        "placement policy is continuous across the reopen"
    );
}

#[test]
fn graph_reopens_by_name_with_contents_intact() {
    let name = "t-ab-reopen";
    Graph::reset_arena(name, ARENA_CAP).expect("reset v4");
    let ids = {
        let mut g = Graph::open_or_create_arena(name, ARENA_CAP).expect("open v4");
        let (hub, spokes) = build(&mut g);
        g.set_vertex_prop(hub, "k", PropValue::I64(1)).unwrap();
        g.sync().unwrap();
        (hub, spokes)
    };

    let g = Graph::open_or_create(name).expect("reopen by name");
    assert_eq!(g.vertex_info(ids.0).unwrap().name, "h");
    assert_eq!(g.out_neighbors(ids.0, Labels::any()).len(), 4);
    assert_eq!(g.get_vertex_prop(ids.0, "k"), Some(PropValue::I64(1)));
    assert_eq!(g.in_neighbors(ids.1[3], Labels::any()).len(), 2);
}

/// The contrast is the test. Either assertion alone is worthless — zero could
/// mean "free" or "counter never incremented", and non-zero could mean anything.
#[test]
fn inline_properties_are_free_and_data_properties_are_not() {
    let name = "t-ab-propcost";
    Graph::reset_arena(name, ARENA_CAP).expect("reset");
    let mut g = Graph::open_or_create_arena(name, ARENA_CAP).expect("open");

    // `hot` is supplied at insert, so it is inline. `cold` is added afterwards,
    // so it is a data property — that is the entire user-facing rule.
    let v = g
        .add_vertex_with_props("n", "v", ObjID::new(0), &[("hot", PropValue::I64(1))])
        .expect("add with inline props");
    g.set_vertex_prop(v, "cold", PropValue::I64(2)).unwrap();

    // Both are readable, and neither shadows the other.
    assert_eq!(g.get_vertex_prop(v, "hot"), Some(PropValue::I64(1)));
    assert_eq!(g.get_vertex_prop(v, "cold"), Some(PropValue::I64(2)));

    g.reset_record_touches();
    assert_eq!(g.get_vertex_prop(v, "hot"), Some(PropValue::I64(1)));
    assert_eq!(
        g.data_block_reads(),
        0,
        "an inline property must not touch the data block"
    );

    g.reset_record_touches();
    assert_eq!(g.get_vertex_prop(v, "cold"), Some(PropValue::I64(2)));
    assert!(
        g.data_block_reads() > 0,
        "control: a data property does dereference the block, so the zero \
         above means 'free' rather than 'counter never fires'"
    );

    // Updating an inline key must stay inline rather than forking a second copy
    // into the data block — the two-sources-of-truth hazard.
    g.set_vertex_prop(v, "hot", PropValue::I64(9)).unwrap();
    assert_eq!(g.get_vertex_prop(v, "hot"), Some(PropValue::I64(9)));
    g.reset_record_touches();
    assert_eq!(g.get_vertex_prop(v, "hot"), Some(PropValue::I64(9)));
    assert_eq!(
        g.data_block_reads(),
        0,
        "updating an inline key must not migrate it to the data block"
    );

    // Growth: enough data properties to force at least one block reallocation.
    for i in 0..12 {
        g.set_vertex_prop(v, &format!("d{i}"), PropValue::I64(i as i64))
            .unwrap();
    }
    for i in 0..12 {
        assert_eq!(
            g.get_vertex_prop(v, &format!("d{i}")),
            Some(PropValue::I64(i as i64)),
            "d{i} survived the block reallocations"
        );
    }
    assert_eq!(g.get_vertex_prop(v, "cold"), Some(PropValue::I64(2)));
    assert_eq!(g.get_vertex_prop(v, "hot"), Some(PropValue::I64(9)));
}
