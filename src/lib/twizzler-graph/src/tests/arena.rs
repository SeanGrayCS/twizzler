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
        s.add_vertex(0, &format!("v{i}")).unwrap();
    }
    assert_eq!(s.vertex_count(), 8);
    assert_eq!(s.arena_count(), 8, "one arena per vertex under OnePerArena");
}

#[test]
fn packing_collapses_object_count() {
    let mut s = store(Box::new(FillTo { cap: 4 }));
    for i in 0..12 {
        s.add_vertex(0, &format!("v{i}")).unwrap();
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
    let hub = s.add_vertex(0, "hub").unwrap();
    let n = ADJ_CHUNK * 2 + 1; // forces three chunks
    let mut targets = Vec::new();
    for i in 0..n {
        targets.push(s.add_vertex(0, &format!("t{i}")).unwrap());
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
            ids.push(s.add_vertex(0, &format!("v{i}")).unwrap());
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
            s.add_vertex(7, &format!("v{i}")).unwrap();
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
        s.add_vertex(0, &format!("v{i}")).unwrap();
    }
    s.add_edge(0, 1, 0, 0).unwrap();
    s.add_edge(1, 2, 1, 0).unwrap();

    s.delete_vertex(1).unwrap();
    assert_eq!(s.vertex_name(1), None);
    assert!(s.neighbors(1, true).is_empty());
    assert!(s.neighbors(1, false).is_empty());
    // Surviving vertices keep their own records.
    assert_eq!(s.vertex_name(0).as_deref(), Some("v0"));
    assert_eq!(s.vertex_name(2).as_deref(), Some("v2"));
}

#[test]
fn allocation_syncs_once_per_arena_not_per_allocation() {
    let mut s = store(Box::new(FillTo { cap: 4 }));
    for i in 0..12 {
        s.add_vertex(0, &format!("v{i}")).unwrap();
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
fn policy_is_identifiable() {
    let a = store(Box::new(OnePerArena));
    let b = store(Box::new(FillTo { cap: 4 }));
    assert_eq!(a.policy_name(), "one-per-arena");
    assert_eq!(b.policy_name(), "fill-to");
}
