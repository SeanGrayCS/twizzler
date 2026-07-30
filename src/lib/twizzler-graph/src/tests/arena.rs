//! These tests pin that arena-backing spends 1 object per vertex, or 1/N with
//! packing, while producing *identical* traversal results. Packing is the
//! lever — cost per vertex is `1454/cap` — not the choice of ids over
//! pointers, which the corrected model shows buys essentially nothing on
//! memory.

use crate::arena_store::{ArenaStore, FillTo, OnePerArena, ADJ_CHUNK};

fn store(policy: Box<dyn crate::arena_store::Placement>) -> ArenaStore {
    ArenaStore::create(policy, 64).expect("create arena store")
}

#[test]
fn one_per_arena_uses_one_object_per_vertex() {
    let mut s = store(Box::new(OnePerArena));
    for i in 0..8 {
        s.add_vertex(0, &format!("v{i}"), 0).unwrap();
    }
    assert_eq!(s.vertex_count(), 8);
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
    assert_eq!(s.vertex_count(), 6);
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

/// VERSION 4: the arena record carries everything the retiring `VertexRef`
/// mirror did, so `Graph` can drop the `verts` registry instead of keeping two
/// structures that assign ids in lockstep by convention.
#[test]
fn arena_record_carries_target_and_props() {
    let mut s = store(Box::new(FillTo { cap: 8 }));
    let a = s.add_vertex(3, "alpha", 0xAABB).unwrap();
    let b = s.add_vertex(4, "beta", 0).unwrap();

    assert_eq!(s.vertex_info(a), Some((3, "alpha".to_string(), 0xAABB)));
    assert_eq!(s.vertex_info(b), Some((4, "beta".to_string(), 0)));
    assert_eq!(s.props_raw(a), Some(0), "no property object yet");

    s.set_props_raw(a, 0x1234).unwrap();
    assert_eq!(s.props_raw(a), Some(0x1234));
    assert_eq!(s.props_raw(b), Some(0), "sibling untouched");

    // Liveness gates every accessor, so `Graph` need not re-check.
    assert!(s.is_alive(a));
    s.delete_vertex(a).unwrap();
    assert!(!s.is_alive(a));
    assert_eq!(s.vertex_info(a), None);
    assert_eq!(s.props_raw(a), None);
    assert!(s.set_props_raw(a, 9).is_err(), "no writes to a dead vertex");

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
fn policy_is_identifiable() {
    let a = store(Box::new(OnePerArena));
    let b = store(Box::new(FillTo { cap: 4 }));
    assert_eq!(a.policy_name(), "one-per-arena");
    assert_eq!(b.policy_name(), "fill-to");
}
