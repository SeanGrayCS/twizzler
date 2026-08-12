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

use crate::Lookup;
use twizzler::object::ObjID;

use super::{fresh, TEST_ARENA_CAP};
use crate::{Graph, Labels, PropValue};

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
