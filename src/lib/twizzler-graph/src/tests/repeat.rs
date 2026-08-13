//! The DSL is fixed-depth today: each `.out()` is exactly one hop, so a k-hop
//! query is k chained steps and an unbounded one cannot be written at all. The
//! engine API can already express recursion by hand — see
//! `adjacency::recursive_traversal_with_visited_set` — but a caller has to bring
//! their own `visited` set and loop.

use twizzler::object::ObjID;

use super::fresh;
use crate::{Graph, Labels, PropValue, VertexId};

/// `a -> b -> c -> d`, a linear chain over `e`.
fn chain(tag: &str) -> (Graph, Vec<VertexId>) {
    let mut g = fresh(tag);
    let ids: Vec<VertexId> = ["a", "b", "c", "d"]
        .iter()
        .map(|n| g.add_vertex("n", n, ObjID::new(0)).unwrap())
        .collect();
    for w in ids.windows(2) {
        g.add_edge(w[0], "e", w[1]).unwrap();
    }
    (g, ids)
}

#[test]
fn times_matches_hand_chained_hops() {
    let (g, ids) = chain("t-b3-times");
    let e = Labels::these(&["e"]);

    let chained = g.traversal().v(ids[0]).out(e).out(e).to_ids();
    let repeated = g.traversal().v(ids[0]).repeat_out(e).times(2).to_ids();
    assert_eq!(repeated, chained);
    assert_eq!(repeated, vec![ids[2]]);
}

#[test]
fn until_returns_the_matching_frontier() {
    let (g, ids) = chain("t-b3-until");
    let found = g
        .traversal()
        .v(ids[0])
        .repeat_out(Labels::these(&["e"]))
        .until(|info| info.name == "d")
        .to_ids();
    assert_eq!(found, vec![ids[3]]);
}

#[test]
fn until_exhausted_reaches_the_chain_root() {
    let (g, ids) = chain("t-b3-exhaust");
    let root = g
        .traversal()
        .v(ids[0])
        .repeat_out(Labels::these(&["e"]))
        .until_exhausted()
        .to_ids();
    assert_eq!(root, vec![ids[3]], "the last vertex with no outgoing `e`");
}

#[test]
fn cycle_terminates_and_never_revisits() {
    let mut g = fresh("t-b3-cycle");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let c = g.add_vertex("n", "c", ObjID::new(0)).unwrap();
    g.add_edge(a, "e", b).unwrap();
    g.add_edge(b, "e", c).unwrap();
    g.add_edge(c, "e", a).unwrap();

    let seen = g
        .traversal()
        .v(a)
        .repeat_out(Labels::these(&["e"]))
        .emit()
        .until_exhausted()
        .to_ids();

    let mut sorted = seen.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), seen.len(), "no vertex is visited twice");

    // `a` is absent, and that is the point. `emit` does not re-emit the
    // start, so reaching `a` again on hop 3 must produce nothing — which is
    // precisely the evidence the visited set stopped the cycle. Without it the
    // walk would step back onto `a` and run to the depth cap.
    assert_eq!(sorted, vec![b, c], "every reachable vertex once; not the start");
    assert!(
        !seen.contains(&a),
        "returning to the start must not re-emit it"
    );
}

#[test]
fn emit_collects_the_path_not_just_the_frontier() {
    let (g, ids) = chain("t-b3-emit");
    let e = Labels::these(&["e"]);

    let emitted = g
        .traversal()
        .v(ids[0])
        .repeat_out(e)
        .emit()
        .times(3)
        .to_ids();
    assert_eq!(
        emitted,
        vec![ids[1], ids[2], ids[3]],
        "every vertex stepped onto, in first-visit order; the start is not re-emitted"
    );

    let frontier = g.traversal().v(ids[0]).repeat_out(e).times(3).to_ids();
    assert_eq!(frontier, vec![ids[3]], "without emit, only the frontier");
}

#[test]
fn per_hop_filter_prunes_expansion() {
    let (mut g, ids) = chain("t-b3-prune");
    for (i, v) in ids.iter().enumerate() {
        g.set_vertex_prop(*v, "ok", PropValue::U64(if i == 1 { 0 } else { 1 }))
            .unwrap();
    }
    let reached = g
        .traversal()
        .v(ids[0])
        .repeat_out(Labels::these(&["e"]))
        .has("ok", PropValue::U64(1))
        .emit()
        .until_exhausted()
        .to_ids();
    assert!(
        reached.is_empty(),
        "b fails the filter, so c and d are never expanded to: {reached:?}"
    );
}

#[test]
fn depth_cap_bounds_the_walk_and_is_observable() {
    let (g, ids) = chain("t-b3-cap");
    let e = Labels::these(&["e"]);

    let capped = g
        .traversal()
        .v(ids[0])
        .repeat_out(e)
        .max_depth(1)
        .emit()
        .until_exhausted();
    assert!(
        capped.hit_depth_cap(),
        "the walk stopped at the cap, and must report that"
    );
    assert_eq!(capped.to_ids(), vec![ids[1]]);

    // A walk that ends because the frontier emptied is *not* truncated.
    let complete = g
        .traversal()
        .v(ids[0])
        .repeat_out(e)
        .max_depth(64)
        .emit()
        .until_exhausted();
    assert!(
        !complete.hit_depth_cap(),
        "ending because there is nowhere left to go is not truncation"
    );
}

#[test]
fn composes_with_downstream_steps_and_reopen() {
    let name = "t-b3-compose";
    let ids = {
        let (mut g, ids) = chain(name);
        g.sync().unwrap();
        drop(g);
        ids
    };
    let g = Graph::open_or_create_arena(name, super::TEST_ARENA_CAP).expect("reopen");
    let n = g
        .traversal()
        .v(ids[0])
        .repeat_out(Labels::these(&["e"]))
        .emit()
        .until_exhausted()
        .has_label("n")
        .dedup()
        .count();
    assert_eq!(n, 3);
}

/// This is the criterion worth the most. Every other test here compares the
/// implementation against expectations written at the same time as the code; this
/// one compares it against an *independent* implementation that predates it and
/// is retained precisely so it can serve as an oracle.
#[test]
fn agrees_with_the_hand_rolled_visited_set_walk() {
    use std::collections::HashSet;

    let mut g = fresh("t-b3-differential");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let c = g.add_vertex("n", "c", ObjID::new(0)).unwrap();
    let d = g.add_vertex("n", "d", ObjID::new(0)).unwrap();
    g.add_edge(a, "e", b).unwrap();
    g.add_edge(b, "e", c).unwrap();
    g.add_edge(c, "e", a).unwrap(); // cycle
    g.add_edge(b, "e", d).unwrap(); // branch

    let mut visited: HashSet<u64> = HashSet::new();
    visited.insert(a.0);
    let mut frontier = vec![a];
    while let Some(v) = frontier.pop() {
        let view = g.vertex_view(v).unwrap();
        for nb in view.out_neighbors_where(Labels::any(), |h| !visited.contains(&h.id().0)) {
            visited.insert(nb.0);
            frontier.push(nb);
        }
    }

    let mut dsl: Vec<u64> = g
        .traversal()
        .v(a)
        .repeat_out(Labels::any())
        .emit()
        .until_exhausted()
        .to_ids()
        .into_iter()
        .map(|v| v.0)
        .collect();
    dsl.push(a.0); // the manual walk seeds `visited` with the start; emit does not
    dsl.sort();
    dsl.dedup();

    let mut manual: Vec<u64> = visited.into_iter().collect();
    manual.sort();
    assert_eq!(dsl, manual);
}
