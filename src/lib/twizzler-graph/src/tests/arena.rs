//! Arena-backed storage with a pluggable placement policy.
//!
//! These tests pin that arena-backing spends one object per vertex, or 1/N
//! with packing, while producing identical traversal results. Counts stay
//! small: the assertions are about ratios, not magnitudes.

use crate::arena_store::{ArenaStore, FillTo, OnePerArena, PropSlot, ADJ_CHUNK};
use crate::props::PropValue;

fn store(policy: Box<dyn crate::arena_store::Placement>) -> ArenaStore {
    ArenaStore::create(policy, 64).expect("create arena store")
}

/// `OnePerArena` places each vertex in its own arena.
#[test]
fn one_per_arena_uses_one_object_per_vertex() {
    let mut s = store(Box::new(OnePerArena));
    for i in 0..8 {
        s.add_vertex(0, &format!("v{i}"), 0).unwrap();
    }
    assert_eq!(s.record_count(), 8);
    assert_eq!(s.arena_count(), 8, "one arena per vertex under OnePerArena");
}

/// `FillTo` packs `cap` vertices per arena, and edges allocate inside
/// existing arenas without adding objects.
#[test]
fn packing_collapses_object_count() {
    let mut s = store(Box::new(FillTo { cap: 4 }));
    for i in 0..12 {
        s.add_vertex(0, &format!("v{i}"), 0).unwrap();
    }
    assert_eq!(s.arena_count(), 3, "12 vertices / cap 4 = 3 arenas");

    // Edges must not add objects: chunks are allocated inside the endpoint's
    // own arena.
    for i in 0..11u64 {
        s.add_edge(i, i + 1, i, 0).unwrap();
    }
    assert_eq!(
        s.arena_count(),
        3,
        "edges allocate inside existing arenas, adding no objects"
    );
}

/// Adjacency survives chunk rollover in insertion order. Chunks are prepended
/// internally, so readers must recover the original order.
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

/// Placement policy changes object count, never traversal results.
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

/// Durability: sync, reopen from the recorded ids, read back.
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

/// Tombstones hide a vertex and its adjacency.
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
    // A tombstoned vertex must also vanish from its neighbours' lists.
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

/// Allocation is batched: a batch costs one sync and one transaction per
/// arena, not one per allocation.
#[test]
fn allocation_syncs_once_per_arena_not_per_allocation() {
    let mut s = store(Box::new(FillTo { cap: 4 }));
    for i in 0..12 {
        s.add_vertex(0, &format!("v{i}"), 0).unwrap();
    }
    for i in 0..11u64 {
        s.add_edge(i, i + 1, i, 0).unwrap();
    }
    // 12 vertex records plus a chunk per endpoint per edge: >30 allocations.
    assert_eq!(s.sync_count(), 0, "nothing is synced before sync_all");

    s.sync_all().unwrap();

    assert_eq!(s.arena_count(), 3, "12 vertices / cap 4 = 3 arenas");
    assert_eq!(
        s.sync_count(),
        3,
        "one sync per arena — if this equals the allocation count, the \
         batching transaction is being dropped instead of reused"
    );
    // `syncs` increments only inside `sync_all`, so a revert to per-allocation
    // drop-syncs would leave the sync assertions above green. Transaction opens
    // are counted at the open site, so that regression reads as one open per
    // record allocation (12 here) instead of one per arena per batch (3).
    assert_eq!(
        s.tx_opens(),
        3,
        "one batching transaction per arena — more means batching has \
         regressed toward transaction-per-allocation"
    );

    // Batching must not cost durability: the whole batch is readable after the
    // single flush. (`reopen_preserves_graph` covers survival across a reopen.)
    assert_eq!(s.vertex_name(0).as_deref(), Some("v0"));
    assert_eq!(s.vertex_name(11).as_deref(), Some("v11"));
    assert_eq!(s.neighbors(5, true), vec![6]);
}

/// A vertices-only batch must mark its arena dirty: `add_record` writes
/// through the arena's own handle, not `record_ptr`, and `sync_all` skips an
/// unmarked arena. The free-list reuse branch must mark too.
#[test]
fn a_vertex_only_batch_syncs_its_arena() {
    let mut s = store(Box::new(FillTo { cap: 64 }));
    s.add_vertex(0, "v0", 0).unwrap();
    s.sync_all().unwrap(); // spends the fresh arena's own mark
    let base = s.sync_count();

    s.add_vertex(0, "v1", 0).unwrap(); // pure add_record into a clean arena
    s.sync_all().unwrap();
    assert_eq!(
        s.sync_count() - base,
        1,
        "a vertices-only batch left its arena unmarked: sync_all skipped the \
         arena while locs.flush() recorded the new record as live"
    );

    // The free-list reuse branch writes through its own `lea_mut` and must
    // mark too. Spend the delete's mark (`record_ptr`) first, so the only
    // thing between the two counts is the reusing insert.
    let v = s.add_vertex(0, "victim", 0).unwrap();
    s.delete_vertex(v).unwrap();
    s.sync_all().unwrap();
    let base = s.sync_count();
    s.add_vertex(0, "reuser", 0).unwrap(); // exact-stride match: takes v's slot
    s.sync_all().unwrap();
    assert_eq!(
        s.sync_count() - base,
        1,
        "the reuse branch of alloc_record_bytes left its arena unmarked"
    );

    // And a no-op batch still costs nothing — sync_all must not sync
    // unconditionally.
    let base = s.sync_count();
    s.sync_all().unwrap();
    assert_eq!(s.sync_count(), base, "an empty batch synced something");
}

/// The arena record carries label, name, target, and data properties, and
/// liveness gates every accessor.
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

/// Label filtering happens inside the adjacency walk, with the same filter
/// semantics as `Graph`'s `Labels::these`.
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

/// Liveness is mirrored into `VertexLoc` so a scan need not resolve each
/// record. The mirror and the records must agree after every mutation, and
/// across a reopen.
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
    // Record-resolving read paths consult the mirror for liveness too.
    assert_eq!(s.vertex_name(3), None);
    assert_eq!(s.vertex_name(7), None);
    assert_eq!(s.vertex_info(3), None);
    assert!(s.neighbors(2, true).is_empty(), "neighbour 3 is hidden");
    assert!(s.neighbors(4, false).is_empty(), "neighbour 3 is hidden");

    // And across a reopen: `flags` is persisted state.
    s.sync_all().expect("sync");
    let (dir, locs) = s.ids();
    let s = ArenaStore::open(dir, locs, Box::new(FillTo { cap: 4 }), 64).expect("reopen");
    assert_eq!(s.vertices(), vec![0, 1, 2, 4, 5, 6, 8, 9]);
    assert_eq!(s.vertex_name(3), None);
    assert!(s.is_alive(9));
}

/// Liveness comes from the `locs` mirror, never from the record. The test
/// forces the two out of agreement — mirror dead, record live — and checks
/// that every read path follows the mirror.
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

/// Canary for every `nosync` path in the crate: `TxObject::abort` suppresses
/// only the sync-on-drop — upstream has no rollback, so aborted writes remain
/// in memory. If this fails, upstream semantics changed and every batched
/// write path is unsound until it moves to a proper upstream nosync API.
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

/// Multi-segment store: deletes and reopen still work when the location
/// registry and the arena directory each span several segments. A tiny
/// `seg_cap` forces both to roll over.
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

    // Survives a reopen with every segment populated.
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

/// Each policy reports an identifying name.
#[test]
fn policy_is_identifiable() {
    let a = store(Box::new(OnePerArena));
    let b = store(Box::new(FillTo { cap: 4 }));
    assert_eq!(a.policy_name(), "one-per-arena");
    assert_eq!(b.policy_name(), "fill-to");
}

/// A record's extent is derived from its own `nprops`, so records of different
/// widths sit back-to-back. A stride slip reads a neighbour's bytes instead of
/// failing, which is why zero-slot and many-slot records are interleaved here.
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

/// Edge records share `locs` with vertices, so `vertices()` must exclude them
/// from the mirror alone, without resolving any record.
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

/// An edge is a record: topology is `from → edge → to`, and the two-hop walk
/// returns the far vertex.
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

/// A hyperedge is an edge record with several out-links; the same walk
/// returns every far endpoint.
#[test]
fn a_hyperedge_is_just_an_edge_record_with_more_links() {
    let mut s = store(Box::new(FillTo { cap: 64 }));
    let a = s.add_record(1, "a", 0, &[], false).unwrap();
    let b = s.add_record(1, "b", 0, &[], false).unwrap();
    let c = s.add_record(1, "c", 0, &[], false).unwrap();
    let e = s.add_edge_record(9, a, b).unwrap();
    // A third participant, joined to the existing edge record.
    s.add_edge_endpoint(e, c, true).expect("add participant");

    let mut got = s.neighbors_via_edges(a, true, false);
    got.sort();
    assert_eq!(got, vec![(e, 9, b), (e, 9, c)], "one edge, two far endpoints");
    assert_eq!(s.neighbors_via_edges(c, false, true), vec![(e, 9, a)]);
    assert_eq!(s.vertices(), vec![a, b, c]);
}

/// A self-loop's far endpoint is its source, so the walk returns it rather
/// than filtering it as a duplicate.
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

/// A deleted record's slot is reclaimed and reused, so churn does not grow
/// the arena count. The generation counter is what makes reuse safe;
/// `stale_adjacency_does_not_resurrect_through_a_reused_slot` covers that half.
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

    // Steady-state churn: the arena count stays flat.
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

/// A twice-tombstoned offset (delete, in-session reuse, delete again — legal,
/// since `delete_vertex` preserves `off` and `locs` is append-only) enters the
/// rebuilt free list once on reopen, so later allocations do not share a slot.
#[test]
fn a_twice_tombstoned_offset_is_freed_once_across_reopen() {
    let (dir, locs) = {
        let mut s = store(Box::new(FillTo { cap: 4 }));
        let a = s.add_record(1, "a", 0, &[], false).unwrap();
        s.add_record(1, "keep", 0, &[], false).unwrap();
        s.delete_vertex(a).unwrap(); // tombstone №1 at offset X
        let c = s.add_record(1, "c", 0, &[], false).unwrap(); // in-session reuse of X
        s.delete_vertex(c).unwrap(); // tombstone №2 at X
        s.sync_all().unwrap();
        s.ids()
    };

    let mut s =
        ArenaStore::open(dir, locs, Box::new(FillTo { cap: 4 }), 64).expect("reopen");
    // The post-reopen inserts. With X pushed twice, p and q both land on X.
    let p = s.add_record(1, "p", 0, &[], false).unwrap();
    let q = s.add_record(1, "q", 0, &[], false).unwrap();
    assert_eq!(
        s.vertex_name(p).as_deref(),
        Some("p"),
        "q was handed p's bytes: the reopen rebuild pushed one offset twice"
    );
    assert_eq!(s.vertex_name(q).as_deref(), Some("q"));
    let live = s.vertices();
    assert_eq!(live.len(), 3, "keep, p and q live; a and c stay dead: {live:?}");
}

/// Vertex labels, edge labels, and property keys intern through one table, so
/// one label id can name both kinds. A by-label vertex scan must still exclude
/// edge records.
#[test]
fn a_label_shared_by_vertices_and_edges_yields_no_edge_records() {
    let mut s = store(Box::new(FillTo { cap: 8 }));
    let a = s.add_record(1, "a", 0, &[], false).unwrap();
    let b = s.add_record(1, "b", 0, &[], false).unwrap();
    // Same label id as the vertices — legal, and what a user gets by using
    // one string for both kinds.
    let e = s.add_edge_record(1, a, b).unwrap();

    assert_eq!(
        s.vertices_by_label(1),
        vec![a, b],
        "edge record {e} leaked into a by-label vertex scan"
    );
    // The mask must not damage the edge itself: it still resolves as an edge.
    assert_eq!(s.edge_endpoints(e), Some((a, b)));
    // Tombstones are masked by the same mirror test.
    s.delete_vertex(b).unwrap();
    assert_eq!(s.vertices_by_label(1), vec![a]);
    assert_eq!(s.edge_endpoints(e), None, "b is gone, so the edge hides too");
}

/// A neighbour entry pointing at a reclaimed slot is skipped, not followed:
/// without the generation check the stale `InvPtr` resolves to the slot's new
/// live record and traversal reports a neighbour that was never connected.
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

/// A degree-1 record allocates no adjacency chunk — its entries fit in the
/// inline slots — asserted through the chunk-allocation counter. Ordering
/// holds across the inline-to-chunk transition.
#[test]
fn a_degree_one_record_allocates_no_chunk() {
    let mut s = store(Box::new(FillTo { cap: 4 }));
    let a = s.add_record(1, "a", 0, &[], false).unwrap();
    let b = s.add_record(1, "b", 0, &[], false).unwrap();
    let e = s.add_edge_record(9, a, b).unwrap();

    // Zero chunks so far: both endpoints and both of the edge record's own
    // directions fit in inline slots.
    assert_eq!(
        s.chunk_allocs(),
        0,
        "a degree-1 edge must live entirely in inline slots — 416 bytes of \
         chunk for 48 bytes of entries is the cost A7-AC5 removes"
    );

    // Correctness — inline entries must read back like chunked ones.
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
    // And past degree 1 the counter must move: the hub spilled to chunks
    // (its inline out slot holds entry 0; the rest chunk), while each edge
    // record itself stayed inline. If this reads 0, the counter is not wired
    // to the allocation site and the assertion above is vacuous.
    assert!(
        s.chunk_allocs() > 0,
        "a degree-{} hub must have spilled past its inline slot",
        ADJ_CHUNK + 2
    );
}

/// `add_edge_endpoint` appends entries without creating records or deduping,
/// so a legal chain can far exceed the record count. `walk_adj`'s corruption
/// bound is a revisited-offset check, which no chain length trips.
#[test]
fn a_wide_hyperedge_walks_without_tripping_the_corruption_bound() {
    let mut s = store(Box::new(FillTo { cap: 64 }));
    let a = s.add_record(1, "a", 0, &[], false).unwrap();
    let b = s.add_record(1, "b", 0, &[], false).unwrap();
    let e = s.add_edge_record(9, a, b).unwrap();
    // 60 repeated joins of the same participant: entries grow, `locs` does
    // not, so the out-chain runs many chunks against three records.
    for _ in 0..60 {
        s.add_edge_endpoint(e, b, true).expect("repeated participant");
    }

    let got = s.neighbors_via_edges(a, true, false);
    assert_eq!(got.len(), 61, "the original endpoint plus 60 repeats, none lost");
    assert!(
        got.iter().all(|&(ee, l, nb)| ee == e && l == 9 && nb == b),
        "every entry names the same edge and participant"
    );
    // The reverse chain grew identically and must walk too.
    let back = s.neighbors_via_edges(b, false, true);
    assert_eq!(back.len(), 61);
    assert!(back.iter().all(|&(ee, _, nb)| ee == e && nb == a));
}

/// Data-property growth allocates a fresh block and repoints `data_props`;
/// the record itself never moves, so its `locs` entry stays put and inbound
/// edges keep resolving.
#[test]
fn data_prop_growth_moves_neither_record_nor_inbound_edges() {
    let mut s = store(Box::new(FillTo { cap: 8 }));
    let fan = s.add_record(1, "fan", 0, &[], false).unwrap();
    let hub = s.add_record(1, "hub", 0, &[], false).unwrap();
    let e = s.add_edge_record(9, fan, hub).unwrap();
    let before = s.record_loc(hub).expect("hub is live");

    // 12 keys: first block (cap 4), then a doubled one, then another — three
    // allocations, two copies, two repoints. The record itself must not move
    // by a byte.
    for k in 0..12u32 {
        s.set_data_prop(hub, 1000 + k, PropValue::I64(k as i64))
            .expect("set data prop");
    }
    assert_eq!(
        s.record_loc(hub),
        Some(before),
        "data-property growth moved the record's locs entry — every inbound \
         AdjRef offset into it is now stale"
    );
    let props = s.data_props(hub).expect("hub is live");
    for k in 0..12u32 {
        assert_eq!(
            props.iter().find(|p| p.key_id == 1000 + k).map(|p| p.val),
            Some(PropValue::I64(k as i64)),
            "key {k} lost across block growth"
        );
    }
    // The inbound edge resolves through the unmoved offset.
    assert_eq!(s.neighbors_via_edges(fan, true, false), vec![(e, 9, hub)]);
    assert_eq!(s.edge_endpoints(e), Some((fan, hub)));
}

/// A hyperedge with two sources and two sinks survives a within-boot reopen.
#[test]
fn a_two_in_two_out_hyperedge_survives_a_reopen() {
    let (dir, locs, a, b, c, d, e) = {
        let mut s = store(Box::new(FillTo { cap: 8 }));
        let a = s.add_record(1, "a", 0, &[], false).unwrap();
        let b = s.add_record(1, "b", 0, &[], false).unwrap();
        let c = s.add_record(1, "c", 0, &[], false).unwrap();
        let d = s.add_record(1, "d", 0, &[], false).unwrap();
        let e = s.add_edge_record(9, a, b).unwrap();
        s.add_edge_endpoint(e, c, true).expect("second sink");
        s.add_edge_endpoint(e, d, false).expect("second source");
        s.sync_all().expect("sync");
        let (dir, locs) = s.ids();
        (dir, locs, a, b, c, d, e)
    };

    let s = ArenaStore::open(dir, locs, Box::new(FillTo { cap: 8 }), 64).expect("reopen");
    let mut from_a = s.neighbors_via_edges(a, true, false);
    from_a.sort();
    assert_eq!(from_a, vec![(e, 9, b), (e, 9, c)], "first source reaches both sinks");
    let mut from_d = s.neighbors_via_edges(d, true, false);
    from_d.sort();
    assert_eq!(from_d, vec![(e, 9, b), (e, 9, c)], "second source reaches both sinks");
    let mut into_b = s.neighbors_via_edges(b, false, true);
    into_b.sort();
    assert_eq!(into_b, vec![(e, 9, a), (e, 9, d)], "first sink sees both sources");
    let mut into_c = s.neighbors_via_edges(c, false, true);
    into_c.sort();
    assert_eq!(into_c, vec![(e, 9, a), (e, 9, d)], "second sink sees both sources");
}

// --- cross-arena adjacency -------------------------------------------------
//
// Most tests above run in one arena. These use `FillTo { cap: 2 }` so that
// almost every reference crosses an arena, and they assert they are actually
// crossing: a test that quietly fell back to one arena would pass while
// covering nothing.

/// `a → edge → b` with all three records in different arenas resolves at
/// every hop.
#[test]
fn cross_arena_edge_traversal_is_correct_at_every_hop() {
    let mut s = store(Box::new(FillTo { cap: 2 }));
    let a = s.add_record(1, "a", 0, &[], false).unwrap();
    let b = s.add_record(1, "b", 0, &[], false).unwrap();
    let e = s.add_edge_record(9, a, b).unwrap();

    // The premise. Without this the test could pass by not crossing anything.
    assert!(
        s.arena_count() >= 2,
        "cap 2 must push the edge record out of a and b's arena; got {} arena(s)",
        s.arena_count()
    );

    let _ = s.take_diag();
    assert_eq!(s.neighbors_via_edges(a, true, false), vec![(e, 9, b)]);
    let d = s.take_diag();
    assert!(
        d.inline > 0,
        "the edge is degree-1, so its entry must be inline — that is the path \
         that failed cross-arena"
    );
    assert_eq!(d.skipped_dead, 0, "no live neighbour may read as dead");
    assert_eq!(d.skipped_gen, 0, "no live neighbour may fail the generation check");

    assert_eq!(s.neighbors_via_edges(b, false, true), vec![(e, 9, a)]);
    assert_eq!(s.edge_endpoints(e), Some((a, b)));
    assert_eq!(s.vertices(), vec![a, b]);
}

/// Chunk entries across arenas, past the inline slot — the control showing
/// the inline test above measures the inline path rather than cross-arena
/// traversal in general.
#[test]
fn cross_arena_chunk_entries_resolve_and_keep_order() {
    let mut s = store(Box::new(FillTo { cap: 2 }));
    let hub = s.add_record(1, "hub", 0, &[], false).unwrap();
    let mut targets = Vec::new();
    for i in 0..ADJ_CHUNK + 3 {
        let t = s.add_record(1, &format!("t{i}"), 0, &[], false).unwrap();
        s.add_edge_record(7, hub, t).unwrap();
        targets.push(t);
    }
    assert!(s.arena_count() > 4, "many arenas, so references cross freely");

    let _ = s.take_diag();
    let got: Vec<u64> = s
        .neighbors_via_edges(hub, true, false)
        .into_iter()
        .map(|(_, _, far)| far)
        .collect();
    let d = s.take_diag();

    assert_eq!(got, targets, "every far endpoint, in insertion order");
    assert!(
        d.cross_arena > 0,
        "the hub's chunk entries must actually cross an arena boundary"
    );
    assert!(d.inline > 0, "and the first entry is still inline");
    assert_eq!(d.skipped_dead, 0);
    assert_eq!(d.skipped_gen, 0);
}

/// Liveness across arenas: a tombstoned far vertex must disappear from a walk
/// that reaches it through another arena, and its neighbours must not.
#[test]
fn cross_arena_deletes_hide_exactly_one_neighbour() {
    let mut s = store(Box::new(FillTo { cap: 2 }));
    let hub = s.add_record(1, "hub", 0, &[], false).unwrap();
    let mut targets = Vec::new();
    for i in 0..5 {
        let t = s.add_record(1, &format!("t{i}"), 0, &[], false).unwrap();
        s.add_edge_record(7, hub, t).unwrap();
        targets.push(t);
    }
    assert!(s.arena_count() > 2);

    s.delete_vertex(targets[2]).unwrap();

    let got: Vec<u64> = s
        .neighbors_via_edges(hub, true, false)
        .into_iter()
        .map(|(_, _, far)| far)
        .collect();
    let expected: Vec<u64> = targets
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 2)
        .map(|(_, t)| *t)
        .collect();
    assert_eq!(got, expected, "exactly the deleted target disappears");
    assert!(!s.vertices().contains(&targets[2]));
}

/// Slot reuse across arenas: the generation guard must still reject a stale
/// entry when the reused slot is in a different arena from the walker.
#[test]
fn cross_arena_slot_reuse_does_not_resurrect_a_neighbour() {
    let mut s = store(Box::new(FillTo { cap: 2 }));
    let hub = s.add_record(1, "hub", 0, &[], false).unwrap();
    let victim = s.add_record(1, "victim", 0, &[], false).unwrap();
    let e = s.add_edge_record(9, hub, victim).unwrap();
    assert!(s.arena_count() >= 2);
    assert_eq!(s.neighbors_via_edges(hub, true, false), vec![(e, 9, victim)]);

    s.delete_vertex(victim).unwrap();
    let squatter = s.add_record(1, "squatter", 0, &[], false).unwrap();

    assert_eq!(
        s.neighbors_via_edges(hub, true, false),
        vec![],
        "the squatter was never hub's neighbour, whatever arena it landed in"
    );
    assert_eq!(s.vertex_name(squatter).as_deref(), Some("squatter"));
}
