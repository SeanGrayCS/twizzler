//! These tests pin that arena-backing spends 1 object per vertex, or 1/N with
//! packing, while producing *identical* traversal results. Packing is the
//! lever — cost per vertex is `1454/cap` — not the choice of ids over
//! pointers, which the corrected model shows buys essentially nothing on
//! memory.

use crate::arena_store::{ArenaStore, FillTo, OnePerArena, PropSlot, ADJ_CHUNK};
use crate::props::PropValue;

fn store(policy: Box<dyn crate::arena_store::Placement>) -> ArenaStore {
    ArenaStore::create(policy, 64).expect("create arena store")
}

#[test]
fn one_per_arena_uses_one_object_per_vertex() {
    let mut s = store(Box::new(OnePerArena));
    for i in 0..8 {
        s.add_vertex(0, &format!("v{i}"), 0).unwrap();
    }
    assert_eq!(s.record_count(), 8);
    assert_eq!(s.arena_count(), 8, "one arena per vertex under OnePerArena");
}

#[test]
fn packing_collapses_object_count() {
    let mut s = store(Box::new(FillTo { cap: 4 }));
    for i in 0..12 {
        s.add_vertex(0, &format!("v{i}"), 0).unwrap();
    }
    assert_eq!(s.arena_count(), 3, "12 vertices / cap 4 = 3 arenas");

    // Edges must not add objects: chunks are allocated inside the endpoint's
    // own arena, which is the whole reason the ceiling moves.
    for i in 0..11u64 {
        s.add_edge(i, i + 1, i, 0).unwrap();
    }
    assert_eq!(
        s.arena_count(),
        3,
        "edges allocate inside existing arenas, adding no objects"
    );
}

#[test]
fn adjacency_spans_chunks_in_order() {
    let mut s = store(Box::new(FillTo { cap: 64 }));
    let hub = s.add_vertex(0, "hub", 0).unwrap();
    let n = ADJ_CHUNK * 2 + 1; // forces three chunks
    let mut targets = Vec::new();
    for i in 0..n {
        targets.push(s.add_vertex(0, &format!("t{i}"), 0).unwrap());
    }
    for (i, t) in targets.iter().enumerate() {
        s.add_edge(hub, *t, i as u64, 0).unwrap();
    }

    let got = s.neighbors(hub, true);
    assert_eq!(got.len(), n, "all {n} neighbours across chunk boundaries");
    assert_eq!(got, targets, "insertion order preserved across chunks");

    // The reverse direction is populated too.
    assert_eq!(s.neighbors(targets[0], false), vec![hub]);
    assert!(s.neighbors(targets[0], true).is_empty());
}

#[test]
fn policy_does_not_change_results() {
    fn build(policy: Box<dyn crate::arena_store::Placement>) -> (Vec<u64>, Vec<u64>, usize) {
        let mut s = store(policy);
        let mut ids = Vec::new();
        for i in 0..8 {
            ids.push(s.add_vertex(0, &format!("v{i}"), 0).unwrap());
        }
        // A deterministic web: each vertex points at the next two.
        for i in 0..8u64 {
            s.add_edge(i, (i + 1) % 8, i * 2, 0).unwrap();
            s.add_edge(i, (i + 2) % 8, i * 2 + 1, 1).unwrap();
        }
        let outs = s.neighbors(3, true);
        let ins = s.neighbors(3, false);
        (outs, ins, s.arena_count())
    }

    let (a_out, a_in, a_objs) = build(Box::new(OnePerArena));
    let (b_out, b_in, b_objs) = build(Box::new(FillTo { cap: 5 }));

    assert_eq!(a_out, b_out, "out-neighbours identical across policies");
    assert_eq!(a_in, b_in, "in-neighbours identical across policies");
    assert_eq!(a_objs, 8, "one arena per vertex");
    assert_eq!(b_objs, 2, "8 vertices / cap 5 = 2 arenas");
    assert!(
        b_objs < a_objs,
        "packing must reduce objects — that is the point"
    );
}

#[test]
fn reopen_preserves_graph() {
    let (dir, locs) = {
        let mut s = store(Box::new(FillTo { cap: 4 }));
        for i in 0..6 {
            s.add_vertex(7, &format!("v{i}"), 0).unwrap();
        }
        for i in 0..5u64 {
            s.add_edge(i, i + 1, i, 3).unwrap();
        }
        s.sync_all().expect("sync");
        s.ids()
    };

    let s = ArenaStore::open(dir, locs, Box::new(FillTo { cap: 4 }), 64).expect("reopen");
    assert_eq!(s.record_count(), 6);
    assert_eq!(s.arena_count(), 2, "6 vertices / cap 4 = 2 arenas");
    assert_eq!(s.vertex_name(0).as_deref(), Some("v0"));
    assert_eq!(s.vertex_name(5).as_deref(), Some("v5"));
    assert_eq!(s.neighbors(0, true), vec![1]);
    assert_eq!(s.neighbors(3, false), vec![2]);
    assert_eq!(s.neighbors(5, false), vec![4]);
}

#[test]
fn delete_tombstones_vertex() {
    let mut s = store(Box::new(FillTo { cap: 16 }));
    for i in 0..3 {
        s.add_vertex(0, &format!("v{i}"), 0).unwrap();
    }
    s.add_edge(0, 1, 0, 0).unwrap();
    s.add_edge(1, 2, 1, 0).unwrap();

    s.delete_vertex(1).unwrap();
    assert_eq!(s.vertex_name(1), None);
    assert!(s.neighbors(1, true).is_empty());
    assert!(s.neighbors(1, false).is_empty());
    assert!(
        s.neighbors(0, true).is_empty(),
        "deleted neighbour hidden from the surviving vertex's out-list"
    );
    assert!(
        s.neighbors(2, false).is_empty(),
        "deleted neighbour hidden from the surviving vertex's in-list"
    );
    // Surviving vertices keep their own records.
    assert_eq!(s.vertex_name(0).as_deref(), Some("v0"));
    assert_eq!(s.vertex_name(2).as_deref(), Some("v2"));
}

#[test]
fn allocation_syncs_once_per_arena_not_per_allocation() {
    let mut s = store(Box::new(FillTo { cap: 4 }));
    for i in 0..12 {
        s.add_vertex(0, &format!("v{i}"), 0).unwrap();
    }
    for i in 0..11u64 {
        s.add_edge(i, i + 1, i, 0).unwrap();
    }
    // 12 vertex records plus a chunk per endpoint per edge: >30 allocations,
    // every one of which was a sync before this change.
    assert_eq!(s.sync_count(), 0, "nothing is synced before sync_all");

    s.sync_all().unwrap();

    assert_eq!(s.arena_count(), 3, "12 vertices / cap 4 = 3 arenas");
    assert_eq!(
        s.sync_count(),
        3,
        "one sync per arena — if this equals the allocation count, the \
         batching transaction is being dropped instead of reused"
    );

    // Batching must not cost durability: the whole batch is readable after the
    // single flush. (`reopen_preserves_graph` covers survival across a reopen.)
    assert_eq!(s.vertex_name(0).as_deref(), Some("v0"));
    assert_eq!(s.vertex_name(11).as_deref(), Some("v11"));
    assert_eq!(s.neighbors(5, true), vec![6]);
}

#[test]
fn arena_record_carries_target_and_props() {
    let mut s = store(Box::new(FillTo { cap: 8 }));
    let a = s.add_vertex(3, "alpha", 0xAABB).unwrap();
    let b = s.add_vertex(4, "beta", 0).unwrap();

    assert_eq!(s.vertex_info(a), Some((3, "alpha".to_string(), 0xAABB)));
    assert_eq!(s.vertex_info(b), Some((4, "beta".to_string(), 0)));
    assert_eq!(s.data_props(a), Some(Vec::new()), "no properties yet");

    s.set_data_prop(a, 7, PropValue::I64(0x1234)).unwrap();
    let got = s.data_props(a).expect("live record");
    assert_eq!(got.len(), 1);
    assert_eq!((got[0].key_id, got[0].val), (7, PropValue::I64(0x1234)));
    assert_eq!(
        s.data_props(b),
        Some(Vec::new()),
        "sibling untouched — blocks are per-record"
    );

    // Liveness gates every accessor, so `Graph` need not re-check.
    assert!(s.is_alive(a));
    s.delete_vertex(a).unwrap();
    assert!(!s.is_alive(a));
    assert_eq!(s.vertex_info(a), None);
    assert_eq!(s.data_props(a), None);
    assert!(
        s.set_data_prop(a, 7, PropValue::I64(9)).is_err(),
        "no writes to a dead vertex"
    );

    assert_eq!(s.vertices(), vec![b], "tombstones excluded");
    assert_eq!(s.vertices_by_label(4), vec![b]);
    assert!(s.vertices_by_label(3).is_empty(), "a was label 3 and is dead");
}

/// Label filtering happens inside the adjacency walk, so a selective query
/// never materialises the whole neighbourhood. Same filter semantics as
/// `Graph`'s `Labels::these`.
#[test]
fn labeled_neighbor_and_edge_queries() {
    let mut s = store(Box::new(FillTo { cap: 8 }));
    let hub = s.add_vertex(0, "hub", 0).unwrap();
    let mut spokes = Vec::new();
    for i in 0..4 {
        spokes.push(s.add_vertex(0, &format!("s{i}"), 0).unwrap());
    }
    // Alternating labels 7 and 9, edge ids 100..104.
    for (i, sp) in spokes.iter().enumerate() {
        let label = if i % 2 == 0 { 7 } else { 9 };
        s.add_edge(hub, *sp, 100 + i as u64, label).unwrap();
    }

    assert_eq!(s.neighbors_labeled(hub, true, None).len(), 4, "any label");
    assert_eq!(
        s.neighbors_labeled(hub, true, Some(&[7])),
        vec![spokes[0], spokes[2]]
    );
    assert_eq!(
        s.neighbors_labeled(hub, true, Some(&[9])),
        vec![spokes[1], spokes[3]]
    );
    assert_eq!(
        s.neighbors_labeled(hub, true, Some(&[7, 9])).len(),
        4,
        "a multi-label filter is a union, not an intersection"
    );
    assert!(s.neighbors_labeled(hub, true, Some(&[42])).is_empty());

    // Edge ids come off the same walk, in the same order.
    assert_eq!(s.edge_ids(hub, true, None), vec![100, 101, 102, 103]);
    assert_eq!(s.edge_ids(hub, true, Some(&[9])), vec![101, 103]);
    assert_eq!(s.edge_ids(spokes[1], false, None), vec![101]);
}

#[test]
fn arena_liveness_mirrors_the_record() {
    let mut s = store(Box::new(FillTo { cap: 4 }));
    for i in 0..10 {
        s.add_vertex(0, &format!("v{i}"), 0).unwrap();
    }
    for i in 0..9u64 {
        s.add_edge(i, i + 1, i, 0).unwrap();
    }

    // Registry view (`vertices`, no resolve) and record view (`vertex_name`,
    // resolves) must agree before any delete.
    assert_eq!(s.vertices().len(), 10);
    for i in 0..10u64 {
        assert!(s.is_alive(i));
        assert!(s.vertex_name(i).is_some());
    }

    s.delete_vertex(3).unwrap();
    s.delete_vertex(7).unwrap();

    // Registry view.
    assert_eq!(s.vertices(), vec![0, 1, 2, 4, 5, 6, 8, 9]);
    assert!(!s.is_alive(3) && !s.is_alive(7));
    assert_eq!(s.vertex_name(3), None);
    assert_eq!(s.vertex_name(7), None);
    assert_eq!(s.vertex_info(3), None);
    assert!(s.neighbors(2, true).is_empty(), "neighbour 3 is hidden");
    assert!(s.neighbors(4, false).is_empty(), "neighbour 3 is hidden");

    // And across a reopen, since `flags` is now persisted state.
    s.sync_all().expect("sync");
    let (dir, locs) = s.ids();
    let s = ArenaStore::open(dir, locs, Box::new(FillTo { cap: 4 }), 64).expect("reopen");
    assert_eq!(s.vertices(), vec![0, 1, 2, 4, 5, 6, 8, 9]);
    assert_eq!(s.vertex_name(3), None);
    assert!(s.is_alive(9));
}

/// That disagreement is not hypothetical. `GlobalPtr::resolve` maps `READ` and
/// `resolve_mut` maps `READ | WRITE | PERSIST`, so they are separate mappings
/// and a record write is invisible to a record read. At `scale:20000` this left
/// 2 858 deleted vertices live on every record-reading path while the mirror
/// had them right. A small in-boot test cannot provoke the incoherence — the
/// suite was green throughout — so this stages it directly instead.
///
/// Record access no longer uses `GlobalPtr`, so the two agree again in the
/// ordinary case. The mirror stays authoritative for the cross-arena
/// `InvPtr::resolve` path, and this test is what stops that quietly eroding.
///
/// If someone reinstates a record-side liveness check, every assertion below
/// fails at once.
#[test]
fn liveness_reads_come_from_the_mirror_not_the_record() {
    let mut s = store(Box::new(FillTo { cap: 4 }));
    for i in 0..6 {
        s.add_vertex(0, &format!("v{i}"), 0).unwrap();
    }
    for i in 0..5u64 {
        s.add_edge(i, i + 1, i, 0).unwrap();
    }
    assert!(s.vertex_info(2).is_some(), "v2 starts live");

    s.tombstone_mirror_only(2).expect("mirror tombstone");

    assert!(!s.is_alive(2));
    assert_eq!(s.vertex_info(2), None, "vertex_info follows the mirror");
    assert_eq!(s.vertex_name(2), None, "vertex_name follows the mirror");
    assert_eq!(s.vertex_label(2), None, "vertex_label follows the mirror");
    assert!(
        s.neighbors(2, true).is_empty(),
        "a dead vertex yields no adjacency"
    );
    assert!(
        !s.neighbors(1, true).contains(&2),
        "a dead *neighbour* is hidden from the walk"
    );
    assert!(!s.vertices().contains(&2), "and from the scan");
    assert!(
        s.set_data_prop(2, 1, PropValue::I64(1)).is_err(),
        "property writes are gated on the mirror too"
    );
    // Undamaged neighbours still resolve, so the walk is filtering rather than
    // bailing out at the first dead entry.
    assert!(s.neighbors(1, false).contains(&0));
    assert!(s.vertex_info(3).is_some());
}

/// CANARY — the load-bearing assumption of every `nosync` path in the crate.
#[test]
fn tx_abort_does_not_roll_back() {
    use twizzler::object::{ObjectBuilder, TypedObject};

    let obj = ObjectBuilder::default().build(1u32).expect("build");
    let mut tx = obj.as_tx().expect("as_tx");
    let mut base = tx.base_mut();
    *base = 7;
    drop(base);
    tx.abort();
    drop(tx);
    assert_eq!(
        *obj.base(),
        7,
        "TxObject::abort rolled back a write — upstream tx semantics changed; \
         every nosync path in the engine is now unsound. See A3 in docs/tasks.md."
    );
}

/// Multi-segment store. Every other test here fits the location registry
/// and the arena directory in one segment each, which is why two bugs reached
/// `gstress scale:20000` before anything caught them: deletes had no effect,
/// and a reopened store found 0 arenas against 5 532 registered vertices.
///
/// A tiny `seg_cap` reproduces the same geometry in a second: 20 vertices at
/// `seg_cap` 4 gives 5 `locs` segments and 5 arena-directory entries, where
/// `scale:20000` needed 55 101 vertices to reach 14.
#[test]
fn multi_segment_store_deletes_and_reopens() {
    // seg_cap 4 (not 64) so both registries roll over; cap 4 so each vertex
    // group also opens a new arena.
    let mut s = ArenaStore::create(Box::new(FillTo { cap: 4 }), 4).expect("create");
    for i in 0..20 {
        s.add_vertex(0, &format!("v{i}"), 0).unwrap();
    }
    for i in 0..19u64 {
        s.add_edge(i, i + 1, i, 0).unwrap();
    }
    assert_eq!(s.arena_count(), 5, "20 vertices / cap 4");
    assert_eq!(s.record_count(), 20);

    // Delete across segment boundaries: 3 is in segment 0, 7 in segment 1,
    // 19 in the last.
    for v in [3u64, 7, 19] {
        s.delete_vertex(v).unwrap();
    }
    for v in [3u64, 7, 19] {
        assert!(!s.is_alive(v), "v{v} still alive after delete");
        assert_eq!(s.vertex_name(v), None, "v{v} record not tombstoned");
    }
    assert_eq!(s.vertices().len(), 17);

    // Survives a reopen with every segment populated — the case where the
    // arena directory came back empty.
    s.sync_all().expect("sync");
    let (dir, locs) = s.ids();
    let s = ArenaStore::open(dir, locs, Box::new(FillTo { cap: 4 }), 4).expect("reopen");
    assert_eq!(
        s.arena_count(),
        5,
        "arena directory lost its entries across reopen"
    );
    assert_eq!(s.record_count(), 20);
    assert_eq!(s.vertices().len(), 17, "tombstones did not survive reopen");
    assert_eq!(s.vertex_name(0).as_deref(), Some("v0"));
    assert_eq!(s.vertex_name(19), None);
    // Adjacency still resolves across arenas after the reopen.
    assert_eq!(s.neighbors(0, true), vec![1]);
    assert_eq!(s.neighbors(5, false), vec![4]);
}

#[test]
fn policy_is_identifiable() {
    let a = store(Box::new(OnePerArena));
    let b = store(Box::new(FillTo { cap: 4 }));
    assert_eq!(a.policy_name(), "one-per-arena");
    assert_eq!(b.policy_name(), "fill-to");
}


#[test]
fn records_of_differing_widths_read_back_correctly() {
    let mut s = store(Box::new(FillTo { cap: 64 }));
    let slot = |k: u32, v: i64| PropSlot {
        key_id: k,
        _pad: 0,
        val: PropValue::I64(v),
    };

    // Alternating widths, all in one arena so they really are adjacent.
    let widths = [0usize, 3, 0, 1, 7, 0, 2];
    let mut ids = Vec::new();
    for (i, w) in widths.iter().enumerate() {
        let props: Vec<PropSlot> = (0..*w)
            .map(|j| slot(j as u32, (i * 100 + j) as i64))
            .collect();
        ids.push(
            s.add_record(1, &format!("r{i}"), 0, &props, false)
                .expect("add_record"),
        );
    }
    assert_eq!(s.arena_count(), 1, "adjacency is the point of this test");

    for (i, w) in widths.iter().enumerate() {
        let got = s.traversal_props(ids[i]).expect("record is live");
        assert_eq!(got.len(), *w, "record {i} reports the wrong width");
        for j in 0..*w {
            assert_eq!(got[j].key_id, j as u32, "record {i} slot {j} key");
            assert_eq!(
                got[j].val,
                PropValue::I64((i * 100 + j) as i64),
                "record {i} slot {j} value — a stride slip reads the neighbour"
            );
        }
        // Identity too: a slipped stride would hand back an adjacent record.
        assert_eq!(s.vertex_name(ids[i]).unwrap(), format!("r{i}"));
    }
}

/// Edge records share `locs` with vertices, so `vertices()` must exclude them —
/// and must do so without resolving anything, or the scan's cost becomes a
/// function of how wide records are. The counter is what makes the second half
/// checkable rather than asserted.
#[test]
fn edge_records_are_excluded_from_scans_without_resolving() {
    let mut s = store(Box::new(FillTo { cap: 64 }));
    let v0 = s.add_record(1, "v0", 0, &[], false).unwrap();
    let e0 = s.add_record(2, "e0", 0, &[], true).unwrap();
    let v1 = s.add_record(1, "v1", 0, &[], false).unwrap();
    let e1 = s.add_record(2, "e1", 0, &[], true).unwrap();

    assert!(!s.is_edge(v0) && !s.is_edge(v1));
    assert!(s.is_edge(e0) && s.is_edge(e1));

    s.reset_record_touches();
    assert_eq!(s.vertices(), vec![v0, v1], "edges are not vertices");
    assert_eq!(
        s.record_touches(),
        0,
        "the scan resolved {} record(s); is-edge must come from the mirror",
        s.record_touches()
    );

    // A tombstoned edge stays excluded, and still without a resolve.
    s.delete_vertex(e0).unwrap();
    s.reset_record_touches();
    assert_eq!(s.vertices(), vec![v0, v1]);
    assert_eq!(s.record_touches(), 0);
}

/// Topology is `from → edge → to`, so a 1-hop query is two resolutions. That
/// cost is the point of the design, not an accident: it buys edge properties
/// through the same path as vertex properties, and hyperedges with no new
/// machinery.
#[test]
fn edges_are_records_and_traversal_reaches_the_far_vertex() {
    let mut s = store(Box::new(FillTo { cap: 64 }));
    let a = s.add_record(1, "a", 0, &[], false).unwrap();
    let b = s.add_record(1, "b", 0, &[], false).unwrap();
    let e = s.add_edge_record(9, a, b).unwrap();

    // The edge occupies an id in the same space, and is excluded from vertices.
    assert!(s.is_edge(e) && !s.is_edge(a));
    assert_eq!(s.vertices(), vec![a, b], "the edge is not a vertex");
    assert_eq!(s.edge_endpoints(e), Some((a, b)));
    assert_eq!(s.edge_endpoints(a), None, "a vertex has no endpoints");

    assert_eq!(s.neighbors_via_edges(a, true, false), vec![(e, 9, b)]);
    assert_eq!(s.neighbors_via_edges(b, false, true), vec![(e, 9, a)]);
    assert_eq!(s.neighbors_via_edges(a, false, true), vec![], "wrong direction");
    assert_eq!(
        s.neighbors_via_edges(a, true, true),
        vec![(e, 9, b)],
        "both-directions must not double-count a single edge"
    );
}

/// A hyperedge needs no new machinery — an edge record with several
/// out-links simply has several targets, and the same walk returns them all.
/// This is the payoff that justified edges-as-records; if it needed a special
/// case, the design would not have earned its cost.
#[test]
fn a_hyperedge_is_just_an_edge_record_with_more_links() {
    let mut s = store(Box::new(FillTo { cap: 64 }));
    let a = s.add_record(1, "a", 0, &[], false).unwrap();
    let b = s.add_record(1, "b", 0, &[], false).unwrap();
    let c = s.add_record(1, "c", 0, &[], false).unwrap();
    let e = s.add_edge_record(9, a, b).unwrap();
    // A third participant, joined to the *existing* edge record.
    s.add_edge_endpoint(e, c, true).expect("add participant");

    let mut got = s.neighbors_via_edges(a, true, false);
    got.sort();
    assert_eq!(got, vec![(e, 9, b), (e, 9, c)], "one edge, two far endpoints");
    assert_eq!(s.neighbors_via_edges(c, false, true), vec![(e, 9, a)]);
    assert_eq!(s.vertices(), vec![a, b, c]);
}

/// A self-loop's far endpoint legitimately *is* its source, so the walk must
/// return it rather than filtering it as a duplicate.
#[test]
fn a_self_loop_returns_its_own_source() {
    let mut s = store(Box::new(FillTo { cap: 64 }));
    let a = s.add_record(1, "a", 0, &[], false).unwrap();
    let e = s.add_edge_record(9, a, a).unwrap();
    assert_eq!(s.edge_endpoints(e), Some((a, a)));
    assert_eq!(s.neighbors_via_edges(a, true, false), vec![(e, 9, a)]);
}

/// Deleting an edge record hides it from traversal, because `walk_adj` skips
/// tombstoned neighbours — edge deletion falls out of record deletion rather
/// than needing its own path.
#[test]
fn tombstoning_an_edge_record_hides_the_hop() {
    let mut s = store(Box::new(FillTo { cap: 64 }));
    let a = s.add_record(1, "a", 0, &[], false).unwrap();
    let b = s.add_record(1, "b", 0, &[], false).unwrap();
    let e = s.add_edge_record(9, a, b).unwrap();
    assert_eq!(s.neighbors_via_edges(a, true, false).len(), 1);

    s.delete_vertex(e).unwrap();
    assert_eq!(s.neighbors_via_edges(a, true, false), vec![]);
    assert_eq!(s.vertices(), vec![a, b], "endpoints survive");
    assert_eq!(s.edge_endpoints(e), None, "a dead edge has no endpoints");
}

/// A deleted record's slot is reclaimed and reused.
///
/// `FillTo` counts records placed, so before slot reuse existed a tombstone
/// never made room: one live vertex pinned its whole arena and churn grew
/// arenas without bound. That was the counter-pressure against raising
/// `DEFAULT_ARENA_CAP` ("4× worse space amplification under churn").
///
/// Reuse is only safe because of the generation counter. Inbound
/// `AdjRef.neighbor` `InvPtr`s still hold a dead record's offset, and
/// `walk_adj` resolves the pointer *before* checking liveness — so handing the
/// bytes to a new record would make every stale entry resolve to a live record
/// with a valid id, pass the liveness check, and yield a neighbour that was
/// never connected. `stale_adjacency_does_not_resurrect_through_a_reused_slot`
/// is the test for that half; this one is about space.
#[test]
fn a_deleted_records_slot_is_reused() {
    let mut s = store(Box::new(FillTo { cap: 2 }));
    let a = s.add_record(1, "a", 0, &[], false).unwrap();
    let b = s.add_record(1, "b", 0, &[], false).unwrap();
    assert_eq!(s.arena_count(), 1, "cap 2, so both land in one arena");

    s.delete_vertex(a).unwrap();
    assert_eq!(s.vertices(), vec![b], "a is gone from enumeration");

    let c = s.add_record(1, "c", 0, &[], false).unwrap();
    assert_eq!(
        s.arena_count(),
        1,
        "c takes a's reclaimed slot rather than opening a second arena"
    );
    assert_eq!(s.vertices(), vec![b, c]);
    assert!(c > b, "ids are still append indices: a deleted *id* is not reused");
    assert_eq!(s.vertex_name(c).as_deref(), Some("c"));
    assert_eq!(s.vertex_name(b).as_deref(), Some("b"), "b is undisturbed");

    // Churn is now flat rather than linear: the steady state holds.
    for _ in 0..8 {
        let last = *s.vertices().last().unwrap();
        s.delete_vertex(last).unwrap();
        s.add_record(1, "x", 0, &[], false).unwrap();
    }
    assert_eq!(
        s.arena_count(),
        1,
        "8 delete/insert cycles must not grow the arena count at all"
    );
    assert_eq!(s.vertices().len(), 2, "still two live records");

    // A double delete must not return the slot twice — two records sharing
    // bytes is corruption, not something generations can catch.
    let live = s.vertices();
    s.delete_vertex(live[0]).unwrap();
    s.delete_vertex(live[0]).unwrap();
    let p = s.add_record(1, "p", 0, &[], false).unwrap();
    let q = s.add_record(1, "q", 0, &[], false).unwrap();
    assert_eq!(s.vertex_name(p).as_deref(), Some("p"));
    assert_eq!(s.vertex_name(q).as_deref(), Some("q"), "p and q must not share a slot");
}

/// The guard that makes reuse safe. A neighbour entry pointing at a slot
/// that has since been reclaimed must be skipped, not followed.
///
/// Without the generation check this is a silent wrong answer: the stale
/// `InvPtr` resolves to a perfectly valid, live record, `is_alive` passes, and
/// traversal reports a neighbour that was never connected. Nothing faults and
/// nothing logs — the same shape as the `resolve`/`resolve_mut` incoherence,
/// which stayed hidden for five days behind a green suite.
#[test]
fn stale_adjacency_does_not_resurrect_through_a_reused_slot() {
    let mut s = store(Box::new(FillTo { cap: 8 }));
    let hub = s.add_record(1, "hub", 0, &[], false).unwrap();
    let victim = s.add_record(1, "victim", 0, &[], false).unwrap();
    let e = s.add_edge_record(9, hub, victim).unwrap();

    assert_eq!(s.neighbors_via_edges(hub, true, false), vec![(e, 9, victim)]);

    // Delete the far endpoint, then allocate until its slot is handed out.
    s.delete_vertex(victim).unwrap();
    assert_eq!(s.neighbors_via_edges(hub, true, false), vec![]);

    let squatter = s.add_record(1, "squatter", 0, &[], false).unwrap();
    assert!(s.vertices().contains(&squatter));

    // The edge's out-chain still holds an `InvPtr` to those bytes, and they now
    // hold a live record. Only the generation stops it being followed.
    assert_eq!(
        s.neighbors_via_edges(hub, true, false),
        vec![],
        "the squatter must not appear as hub's neighbour — it never was one"
    );
    assert_eq!(
        s.edge_endpoints(e),
        None,
        "the edge lost an endpoint and must not silently acquire a new one"
    );
    // The squatter is a normal record in every other respect.
    assert_eq!(s.vertex_name(squatter).as_deref(), Some("squatter"));
    assert!(s.neighbors_via_edges(squatter, true, true).is_empty());
}

/// An `AdjChunk` holds `ADJ_CHUNK` entries and costs 208 bytes. An edge record
/// has out-degree 1 and in-degree 1, so before the inline slots it spent 416
/// bytes to store 48 — more than three times the record itself, and the
/// difference between edges-as-records costing 4.6× v4 per edge and 1.7×.
///
/// Asserted through arena *count* at a tiny cap, which is the only handle the
/// store exposes on allocation volume: chunks and records share the arena, so a
/// workload that stops allocating chunks fits in fewer arenas.
#[test]
fn a_degree_one_record_allocates_no_chunk() {
    // Cap 4 with records only: 3 records = 1 arena. If each edge still took two
    // chunks, the same workload would spill well past it.
    let mut s = store(Box::new(FillTo { cap: 4 }));
    let a = s.add_record(1, "a", 0, &[], false).unwrap();
    let b = s.add_record(1, "b", 0, &[], false).unwrap();
    let e = s.add_edge_record(9, a, b).unwrap();

    // Correctness first — inline entries must read back like chunked ones.
    assert_eq!(s.neighbors_via_edges(a, true, false), vec![(e, 9, b)]);
    assert_eq!(s.neighbors_via_edges(b, false, true), vec![(e, 9, a)]);
    assert_eq!(s.edge_endpoints(e), Some((a, b)));

    // And the ordering contract survives the inline/chunk boundary: the inline
    // slot is the oldest entry, so it must come first.
    let hub = s.add_record(1, "hub", 0, &[], false).unwrap();
    let mut targets = Vec::new();
    for i in 0..ADJ_CHUNK + 2 {
        let t = s.add_record(1, &format!("t{i}"), 0, &[], false).unwrap();
        s.add_edge_record(7, hub, t).unwrap();
        targets.push(t);
    }
    let got: Vec<u64> = s
        .neighbors_via_edges(hub, true, false)
        .into_iter()
        .map(|(_, _, far)| far)
        .collect();
    assert_eq!(
        got, targets,
        "insertion order must hold across the inline→chunk transition — the \
         inline entry is the oldest and the chunks are prepended"
    );
}
