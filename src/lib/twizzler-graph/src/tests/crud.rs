//! Create/read/delete semantics, id behavior, and lookup by key.

use twizzler::object::ObjID;

use super::fresh;
use crate::{EdgeId, GraphError, Labels, VertexId};

#[test]
fn create_and_vertex_info() {
    let mut g = fresh("t-civ");
    let v = g.add_vertex("file", "doc", ObjID::new(0)).unwrap();
    let info = g.vertex_info(v).expect("vertex info");
    assert_eq!(info.label, "file");
    assert_eq!(info.name, "doc");
}

#[test]
fn find_and_by_label() {
    let mut g = fresh("t-find");
    let t = g.add_vertex("tag", "thesis", ObjID::new(0)).unwrap();
    assert_eq!(g.find_vertex("tag", "thesis"), Some(t));
    assert_eq!(g.find_vertex("tag", "missing"), None);
    g.add_vertex("tag", "other", ObjID::new(0)).unwrap();
    assert_eq!(g.vertices_by_label("tag").len(), 2);
    assert_eq!(g.vertices_by_label("file").len(), 0);
}

#[test]
fn add_edge_to_missing_vertex_errors() {
    let mut g = fresh("t-bad");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let bogus = VertexId(9999);
    match g.add_edge(a, "e", bogus) {
        Err(GraphError::Twz(_)) => {}
        other => panic!("expected an error for missing endpoint, got {other:?}"),
    }
}

#[test]
fn delete_vertex_hides_it_and_incident_edges() {
    let mut g = fresh("t-delv");
    let a = g.add_vertex("file", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("file", "b", ObjID::new(0)).unwrap();
    let t = g.add_vertex("tag", "x", ObjID::new(0)).unwrap();
    g.add_edge(a, "tagged", t).unwrap();
    g.add_edge(b, "tagged", t).unwrap();

    g.delete_vertex(a).unwrap();
    assert!(g.vertex_info(a).is_none());
    assert_eq!(g.vertices_by_label("file"), vec![b]);
    // a's incident edge is hidden, so only b is tagged now.
    assert_eq!(g.in_neighbors(t, Labels::these(&["tagged"])), vec![b]);
    // Traversal from a deleted vertex yields nothing.
    assert!(g.vertex_view(a).is_none());
    assert_eq!(g.traversal().v(a).count(), 0);
}

#[test]
fn delete_edge_hides_it() {
    let mut g = fresh("t-dele");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let e = g.add_edge(a, "e", b).unwrap();

    g.delete_edge(e).unwrap();
    assert!(g.edge_info(e).is_none());
    assert!(g.out_neighbors(a, Labels::any()).is_empty());
    assert!(g
        .vertex_view(a)
        .unwrap()
        .out_edges(Labels::any())
        .is_empty());
}

#[test]
fn out_of_range_ids_are_none() {
    let mut g = fresh("t-oob");
    g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    assert!(g.vertex_info(VertexId(9999)).is_none());
    assert!(g.edge_info(EdgeId(9999)).is_none());
    assert!(g.vertex_view(VertexId(9999)).is_none());
    // Deletes past the end are Ok(()) by contract (idempotent delete space).
    g.delete_vertex(VertexId(9999)).unwrap();
    g.delete_edge(EdgeId(9999)).unwrap();
}

#[test]
fn delete_is_idempotent_and_ids_are_not_reused() {
    let mut g = fresh("t-delid");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    g.delete_vertex(a).unwrap();
    g.delete_vertex(a).unwrap(); // second delete: no error, no effect
    let c = g.add_vertex("n", "c", ObjID::new(0)).unwrap();
    // c takes the next append index; a's id is not recycled.
    assert_eq!(c, VertexId(2));
    assert!(g.vertex_info(a).is_none());
    assert_eq!(g.vertex_info(b).unwrap().name, "b");
    assert_eq!(g.vertex_info(c).unwrap().name, "c");
    assert_eq!(g.vertices(), vec![b, c]);
}
