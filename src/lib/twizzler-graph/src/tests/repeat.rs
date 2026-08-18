//! Recursive / variable-length traversal: `repeat_out` with `times`, `until`,
//! `until_exhausted`, `emit`, per-hop filters, and depth caps.
//!
//! `repeat_out(labels)` returns a `Repeat` builder rather than Gremlin's
//! higher-order `repeat(step)`: an anonymous step fights this DSL's ownership
//! model, where every step consumes `self`. The visited set is always on, so
//! cycles terminate.

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

/// `times(k)` is exactly k chained hops. Checked against the hand-chained form
/// rather than a written-down expectation, so the two cannot drift.
#[test]
fn times_matches_hand_chained_hops() {
    let (g, ids) = chain("t-b3-times");
    let e = Labels::these(&["e"]);

    let chained = g.traversal().v(ids[0]).out(e).out(e).to_ids();
    let repeated = g.traversal().v(ids[0]).repeat_out(e).times(2).to_ids();
    assert_eq!(repeated, chained);
    assert_eq!(repeated, vec![ids[2]]);
}

/// `until` walks to the first frontier satisfying the predicate.
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

/// `until_exhausted` walks to the end of a chain: the terminating condition is
/// "no outgoing edge of this label", not a property.
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

/// A cycle terminates: the visited set stops the walk once every reachable
/// vertex has been seen, and no vertex is visited twice.
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

    // `a` is absent on purpose: `emit` does not re-emit the start, so reaching
    // `a` again on hop 3 produces nothing — the visited set stopped the cycle.
    // Without it the walk would step back onto `a` and run to the depth cap.
    assert_eq!(sorted, vec![b, c], "every reachable vertex once; not the start");
    assert!(
        !seen.contains(&a),
        "returning to the start must not re-emit it"
    );
}

/// `emit` collects everything visited, in first-visit order; without it only
/// the final frontier comes back.
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

/// A per-hop filter prunes the frontier, so failing vertices do not expand:
/// blocking `b` also makes `c` and `d` unreachable.
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

/// The depth cap bounds the walk and reports that it did; a walk that ends
/// because the frontier emptied does not report truncation.
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

    // A walk that ends because the frontier emptied is not truncated.
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

/// `emit` and `until` compose as in Gremlin's `repeat().emit().until()`: stop
/// at the first matching frontier, return everything emitted through it.
#[test]
fn emit_and_until_compose() {
    let (g, ids) = chain("t-b3-emituntil");
    let e = Labels::these(&["e"]);

    // Match at c: the result is everything visited through c's frontier —
    // b then c, first-visit order — and the walk stopped there (no d).
    let composed = g
        .traversal()
        .v(ids[0])
        .repeat_out(e)
        .emit()
        .until(|i| i.name == "c")
        .to_ids();
    assert_eq!(
        composed,
        vec![ids[1], ids[2]],
        "emitted set through the matching frontier, matchers included, d unreached"
    );

    // Without emit, the matchers-only behaviour is unchanged.
    let matchers = g
        .traversal()
        .v(ids[0])
        .repeat_out(e)
        .until(|i| i.name == "c")
        .to_ids();
    assert_eq!(matchers, vec![ids[2]]);

    // `until` that never matches, with emit: the visits are still the result,
    // and the flag still distinguishes exhaustion from the cap.
    let unmatched = g
        .traversal()
        .v(ids[0])
        .repeat_out(e)
        .emit()
        .until(|i| i.name == "never");
    assert!(!unmatched.hit_depth_cap(), "exhausted, not capped");
    assert_eq!(unmatched.to_ids(), vec![ids[1], ids[2], ids[3]]);
}

/// `times(k)` at `k == max_depth` is completion, not truncation. Only a
/// `times` that asks for more hops than the cap allows is truncated.
#[test]
fn times_at_the_cap_is_complete_not_truncated() {
    let (g, ids) = chain("t-b3-capk");
    let e = Labels::these(&["e"]);

    // k == cap, frontier still non-empty afterwards: complete.
    let at_cap = g.traversal().v(ids[0]).repeat_out(e).max_depth(2).times(2);
    assert!(
        !at_cap.hit_depth_cap(),
        "all k requested hops were performed — nothing was cut short"
    );
    assert_eq!(at_cap.to_ids(), vec![ids[2]]);
    // And strict mode agrees: a complete answer is not an error.
    let strict = g
        .traversal()
        .v(ids[0])
        .repeat_out(e)
        .strict_depth(2)
        .times(2)
        .expect("k == cap with all hops performed must not be WalkTruncated");
    assert_eq!(strict.to_ids(), vec![ids[2]]);

    // k > cap: the request was cut short, and that IS truncation.
    let over = g.traversal().v(ids[0]).repeat_out(e).max_depth(2).times(3);
    assert!(over.hit_depth_cap(), "3 hops requested, 2 allowed, more to walk");
    assert_eq!(over.to_ids(), vec![ids[2]], "the frontier where the cap stopped it");
}

/// The truncation flag is sticky: an edge step after a truncated repeat
/// carries it forward, and so does the step back to vertices.
#[test]
fn truncation_survives_an_edge_step() {
    let (g, ids) = chain("t-b3-launder");
    let e = Labels::these(&["e"]);

    let truncated = g.traversal().v(ids[0]).repeat_out(e).max_depth(1).times(3);
    assert!(truncated.hit_depth_cap(), "1 of 3 requested hops performed");
    assert_eq!(truncated.to_ids(), vec![ids[1]]);

    let through_edges = g
        .traversal()
        .v(ids[0])
        .repeat_out(e)
        .max_depth(1)
        .times(3)
        .out_e(e);
    assert!(
        through_edges.hit_depth_cap(),
        "the edge form must carry the flag, not launder it"
    );
    let back_on_vertices = through_edges.in_v();
    assert!(
        back_on_vertices.hit_depth_cap(),
        "and hand it back to the vertex form"
    );
    assert_eq!(back_on_vertices.to_ids(), vec![ids[2]]);
}

/// The default cap bounds depth on a long acyclic chain: a walk asked for more
/// hops than `DEFAULT_MAX_DEPTH` stops at the cap and reports truncation.
#[test]
fn a_request_past_the_default_cap_stops_at_it_and_reports() {
    use crate::DEFAULT_MAX_DEPTH;
    let mut g = fresh("t-b3-defcap");
    let n = DEFAULT_MAX_DEPTH + 2;
    let ids: Vec<VertexId> = (0..n)
        .map(|i| g.add_vertex("n", &format!("c{i}"), ObjID::new(0)).unwrap())
        .collect();
    for w in ids.windows(2) {
        g.add_edge(w[0], "e", w[1]).unwrap();
    }

    let walked = g
        .traversal()
        .v(ids[0])
        .repeat_out(Labels::these(&["e"]))
        .times(DEFAULT_MAX_DEPTH + 1); // no max_depth() call: the default governs
    assert!(
        walked.hit_depth_cap(),
        "one more hop was requested than DEFAULT_MAX_DEPTH allows"
    );
    assert_eq!(
        walked.to_ids(),
        vec![ids[DEFAULT_MAX_DEPTH]],
        "stopped exactly at the default cap"
    );
}

/// `repeat` composes with downstream steps, and survives a reopen.
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

/// The DSL agrees with a hand-rolled visited-set walk over a graph with a
/// cycle and a branch — an independent implementation serving as an oracle.
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
