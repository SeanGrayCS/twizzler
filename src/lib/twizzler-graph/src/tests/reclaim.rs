//! The obvious fix — `sys_object_ctrl(id, Delete)` — turned out to be unsafe
//! in this build: deleting a pager-backed object livelocks the pager, which
//! answers the next `ObjectInfoReq` for that id with "uncategorized error: 0"
//! and retries forever. So the deletes are withheld and only the parts that
//! are safe and useful remain:
//!
//! - the inventory (what a graph owns), which is what any reclaim or
//!   eviction scheme needs and which nothing else in the engine could report;
//! - the unreachability work in `delete_vertex` — tombstone plus clearing
//!   the adjacency and property ids — which is what makes those objects
//!   eligible for reclaim in the first place;
//! - the semantics guarantee, that none of this changes an answer.
//!
//! ---
//!
//! Everything above this line is stale and is kept only for the trail.
//!
//! The load-bearing error was the frame-accounting paragraph, which read:

use crate::Lookup;
use twizzler::object::ObjID;

use super::{fresh, TEST_ARENA_CAP};
use crate::{reclaim, Graph, Labels, PropValue};

#[test]
fn inventory_covers_everything_the_graph_owns() {
    let mut g = fresh("t-reclaim-inv");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let e = g.add_edge(a, "knows", b).unwrap();
    g.set_vertex_prop(a, "age", PropValue::I64(30)).unwrap();
    g.set_edge_prop(e, "since", PropValue::I64(2020)).unwrap();

    let ids = g.owned_object_ids();

    // Every id is real, and none is the root: `reset` retains and repoints the
    // root, so listing it would invite deleting the graph's identity.
    assert!(ids.iter().all(|r| *r != 0), "no null ids in the inventory");
    assert!(!ids.contains(&g.root_id().raw()), "root is not owned");

    assert_eq!(
        g.vertex_props_raw(a),
        Some(0),
        "a property must no longer allocate an object"
    );
    assert_eq!(g.edge_props_raw(e), Some(0), "nor an edge property");

    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), ids.len(), "inventory contains no duplicates");

    // Fill the first arena and spill into a second. Exactly one new id should
    // appear: the new arena. Anything else means the inventory is tracking
    // something it should not, and a miss means arenas are invisible to
    // reclaim — the bug this rewrite exists to close.
    let before = ids.len();
    assert_eq!(g.arena_count(), 1, "two vertices fit in one arena");
    for i in 0..TEST_ARENA_CAP {
        g.add_vertex("n", &format!("f{i}"), ObjID::new(0)).unwrap();
    }
    assert_eq!(g.arena_count(), 2, "spilled into a second arena");
    assert_eq!(
        g.owned_object_ids().len(),
        before + 1,
        "the second arena, and only it, joins the inventory"
    );
}

#[test]
fn delete_vertex_leaves_no_object_to_reclaim() {
    let mut g = fresh("t-reclaim-delv");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    g.add_edge(a, "knows", b).unwrap();
    g.set_vertex_prop(a, "age", PropValue::I64(30)).unwrap();

    let before = g.owned_object_ids().len();
    assert_eq!(
        g.vertex_props_raw(a),
        Some(0),
        "properties are arena bytes now, not an object"
    );

    g.delete_vertex(a).unwrap();

    // The record survives as a tombstone — neighbours' adjacency entries still
    // point into it, and traversal resolves those before checking liveness —
    // but it no longer names the property object.
    assert!(
        g.vertex_info(a).is_none(),
        "tombstoned vertex reads as absent"
    );
    assert_eq!(
        g.vertex_props_raw(a),
        None,
        "a tombstoned record is not reachable through the liveness mirror"
    );
    assert_eq!(g.get_vertex_prop(a, "age"), None, "properties unreadable");

    assert_eq!(
        g.owned_object_ids().len(),
        before,
        "a delete must not change the object inventory: records own no objects"
    );
}

#[test]
fn reclaim_does_not_change_answers() {
    let mut g = fresh("t-reclaim-answers");
    let hub = g.add_vertex("n", "hub", ObjID::new(0)).unwrap();
    let mut spokes = Vec::new();
    for i in 0..4 {
        let s = g.add_vertex("n", &format!("s{i}"), ObjID::new(0)).unwrap();
        g.add_edge(hub, "e", s).unwrap();
        spokes.push(s);
    }
    g.set_vertex_prop(spokes[1], "k", PropValue::I64(9)).unwrap();

    g.delete_vertex(spokes[1]).unwrap();

    // The deleted vertex is invisible in every way.
    assert!(g.vertex_info(spokes[1]).is_none());
    assert_eq!(g.find_vertex("n", "s1"), Lookup::NotFound);
    assert!(g.out_neighbors(spokes[1], Labels::any()).is_empty());
    assert!(g.in_neighbors(spokes[1], Labels::any()).is_empty());
    assert_eq!(g.get_vertex_prop(spokes[1], "k"), None);
    assert!(g.vertex_props(spokes[1]).is_empty());

    // Everything else is untouched.
    let mut nbrs = g.out_neighbors(hub, Labels::any());
    nbrs.sort_by_key(|v| v.0);
    assert_eq!(nbrs, vec![spokes[0], spokes[2], spokes[3]]);
    assert_eq!(g.vertices().len(), 4);
    for i in [0usize, 2, 3] {
        assert_eq!(g.vertex_info(spokes[i]).unwrap().name, format!("s{i}"));
        assert_eq!(g.in_neighbors(spokes[i], Labels::any()), vec![hub]);
    }
}

#[test]
fn deletes_stay_idempotent() {
    let mut g = fresh("t-reclaim-idem");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let e = g.add_edge(a, "knows", b).unwrap();

    g.delete_vertex(a).expect("first delete");
    g.delete_vertex(a)
        .expect("second delete is a no-op, not an error");
    g.delete_edge(e).expect("first edge delete");
    g.delete_edge(e).expect("second edge delete is a no-op");

    g.delete_vertex(crate::VertexId(999)).expect("unknown vertex");
    g.delete_edge(crate::EdgeId(999)).expect("unknown edge");

    assert!(g.vertex_info(a).is_none());
    assert_eq!(g.vertices(), vec![b]);
}

/// `reset` still produces a working, empty graph that keeps its identity —
/// the property the withheld reclaim must not have broken.
#[test]
fn reset_leaves_a_working_empty_graph() {
    let name = "t-reclaim-reset";
    let root = {
        let mut g = fresh(name);
        let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
        let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
        g.add_edge(a, "knows", b).unwrap();
        g.set_vertex_prop(a, "age", PropValue::I64(30)).unwrap();
        g.root_id()
    };

    // Explicit caps, not `reset`/`open_or_create`: those default to
    // `DEFAULT_ARENA_CAP` (16 384), and an arena is sized by its cap, so the
    // convenience constructors would allocate a production-sized arena inside
    // the shared test boot for a graph that holds one vertex.
    Graph::reset_arena(name, TEST_ARENA_CAP).expect("reset");

    let mut g =
        Graph::open_or_create_arena(name, TEST_ARENA_CAP).expect("reopen after reset");
    assert_eq!(g.root_id(), root, "reset keeps the graph's identity");
    assert!(g.vertices().is_empty());
    assert_eq!(g.find_vertex("n", "a"), Lookup::NotFound);
    // And it is usable again: ids restart from 0.
    let v = g.add_vertex("n", "fresh", ObjID::new(0)).unwrap();
    assert_eq!(v.0, 0);
}

/// Deliberately the least interesting test in the file, and deliberately first.
/// Every conclusion downstream divides by `pages_before`, so a silent zero
/// there would turn "no frames came back" and "the counter returns nothing"
/// into the same observation — which is the shape of mistake that produced the
/// 38%. Asserting the denominator before anyone reads the ratio is the whole
/// point.
#[test]
fn object_stat_reports_pages_for_a_live_arena() {
    let mut g = fresh("t-reclaim-stat");
    // Spill past one arena, so the arena group holds more than a single id and
    // a per-arena zero cannot hide inside a non-zero total.
    for i in 0..(TEST_ARENA_CAP + 2) {
        g.add_vertex("n", &format!("v{i}"), ObjID::new(0)).unwrap();
    }
    assert!(g.arena_count() >= 2, "spilled into a second arena");
    g.sync().unwrap();

    let ids = g.owned_object_ids();
    assert!(!ids.is_empty(), "a populated graph owns objects");

    let unknown: Vec<u128> = ids
        .iter()
        .copied()
        .filter(|r| reclaim::object_pages(*r).is_none())
        .collect();
    assert!(
        unknown.is_empty(),
        "{} owned ids do not resolve via sys_object_stat: {unknown:?}",
        unknown.len()
    );

    let (objects, pages) = g.resident_pages();
    assert_eq!(objects, ids.len() + 1, "every owned id, plus the root");
    assert!(
        pages > 0,
        "sys_object_stat reported 0 resident pages across {objects} live \
         objects. AC13a is *untestable* in this state, not met: the page \
         counter is the denominator of every figure A5-AC13 reports."
    );
}

/// What this test can and cannot see. `sys_object_stat` returns `Ok` for an
/// object that is in the manager's map, and a marked-but-unreaped object is
/// still in the map. So `Ok`/`Err` cannot distinguish "never deleted" from
/// "deleted, not yet reaped" — which is why this test now *reports* rather than
/// asserts, and why the sweep below exists to separate them.
///
/// The root is checked in the opposite direction on purpose: it is the one
/// object `destroy` deliberately retains, because the naming service cannot
/// unbind on this build.
#[test]
fn destroy_defers_reaping_until_the_mapping_drops() {
    let name = "t-reclaim-gone";
    let (ids, root) = {
        let mut g = fresh(name);
        for i in 0..(TEST_ARENA_CAP + 2) {
            g.add_vertex("n", &format!("v{i}"), ObjID::new(0)).unwrap();
        }
        g.sync().unwrap();
        (g.owned_object_ids(), g.root_id())
    };
    assert!(!ids.is_empty());

    let rep = Graph::destroy_measured(name).expect("destroy");
    assert!(rep.measured, "destroy_measured must set the measured flag");
    assert!(rep.accepted > 0, "the kernel accepted no deletes at all");

    let resolving = |ids: &[u128]| -> usize {
        ids.iter()
            .filter(|r| reclaim::object_pages(**r).is_some())
            .count()
    };
    let immediately = resolving(&ids);

    let swept = reclaim::sweep_deleted();
    let after_sweep = resolving(&ids);

    println!(
        "A5-AC13b: {} ids, destroy accepted {}, still resolving {} immediately, \
         {} after a sweep (sweep ran: {swept})",
        ids.len(),
        rep.accepted,
        immediately,
        after_sweep
    );
    println!(
        "  reading: {}",
        if after_sweep == 0 {
            "DEFERRED — reaping happens once the mapping drops; the residue is a \
             lag, not a leak"
        } else if after_sweep < immediately {
            "PARTIALLY DEFERRED — some ids were still mapped at destroy time and \
             some still are; report both numbers, do not average them"
        } else {
            "BLOCKED — a context outlives the delete, so scan_deleted can never \
             reap these. Engine- or runtime-side, and §5.5.2's 'the residue is \
             platform-side' does not survive as written"
        }
    );

    // Deliberately no assertion on `after_sweep`. Every value of it is a result,
    // and the one thing this file must not do again is assert a predicted
    // outcome and read the failure as a defect rather than as data.
    assert!(
        after_sweep <= immediately,
        "a sweep made *more* ids resolve ({immediately} -> {after_sweep}), which \
         is not a thing scan_deleted can do; suspect the harness"
    );

    // The deliberate leak, asserted as deliberate.
    assert!(
        reclaim::object_pages(root.raw()).is_some(),
        "the root must survive destroy — the name cannot be unbound on this \
         build, so deleting it would leave data/{name} pointing at a deleted \
         object"
    );
}

/// The only assertion is that destroying an object never *increases* its
/// resident pages. That is deliberate and it is the point of the test.
#[test]
fn destroy_returns_pages() {
    let name = "t-reclaim-pages";
    let ids = {
        let mut g = fresh(name);
        for i in 0..(TEST_ARENA_CAP * 2) {
            g.add_vertex("n", &format!("v{i}"), ObjID::new(0)).unwrap();
        }
        g.sync().unwrap();
        g.owned_object_ids()
    };

    let rep = Graph::destroy_measured(name).expect("destroy");
    assert!(rep.pages_before > 0, "AC13a again, on this graph's ids");
    assert!(
        !rep.grew(),
        "resident pages rose across a destroy: {} -> {}",
        rep.pages_before,
        rep.pages_after
    );

    // The report is the deliverable. Printed rather than asserted, so a change
    // in the platform's behaviour shows up as a changed number in the log
    // instead of a red test nobody can interpret.
    println!("A5-AC13 destroy report for `{name}`:");
    for line in rep.report_lines() {
        println!("{line}");
    }
    match rep.returned_fraction() {
        Some(f) => println!(
            "  returned {}/{} pages = {:.1}% (§5.5 inferred ~38% from stall \
             points; this is a measurement of a different quantity and the two \
             are not comparable without AC14)",
            rep.returned_pages(),
            rep.pages_before,
            f * 100.0
        ),
        None => println!("  returned fraction: undefined (nothing resident)"),
    }

    // The same figure after a sweep, because the first run of this file
    // established that `destroy` marks and `scan_deleted` reaps, and the two do
    // not happen at the same moment. `rep` is therefore the *instantaneous*
    // fraction, which may be structurally near zero without telling us anything
    // about whether the frames ever come back. This is the eventual one.
    let swept = reclaim::sweep_deleted();
    let (present, pages) = reclaim::pages_of(ids.iter().copied());
    println!(
        "  after sweep (ran: {swept}): {present} ids still resolve, {pages} \
         pages resident, i.e. {} of {} pages returned eventually ({})",
        rep.pages_before.saturating_sub(pages),
        rep.pages_before,
        if rep.pages_before > 0 {
            format!(
                "{:.1}%",
                rep.pages_before.saturating_sub(pages) as f64 * 100.0
                    / rep.pages_before as f64
            )
        } else {
            "undefined".to_string()
        }
    );
    println!(
        "  **This pair is the A5 headline, not either number alone.** A large \
         gap between the instantaneous and eventual fractions means the §5.5 \
         residue is a deferral; no gap means it is a retention. §5.5 could not \
         tell these apart because it measured neither."
    );
}
