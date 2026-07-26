//! D2b acceptance tests: IndraDB semantics through `Database`, backed by
//! `TwizzlerDatastore`. These drive the public API a consumer (and E3) uses,
//! not the `Transaction` methods directly.

use indradb::{
    Database, Edge, Identifier, Json, QueryExt, QueryOutputValue, SpecificEdgeQuery,
    SpecificVertexQuery, Vertex,
};
use serde_json::json;
use uuid::Uuid;

use crate::TwizzlerDatastore;

fn ident(s: &str) -> Identifier {
    Identifier::new(s).expect("valid identifier")
}

fn db() -> Database<TwizzlerDatastore> {
    TwizzlerDatastore::new_db().expect("create datastore")
}

/// Two vertices joined by an edge, returned as (a, b, edge).
fn seed(db: &Database<TwizzlerDatastore>) -> (Uuid, Uuid, Edge) {
    let a = db.create_vertex_from_type(ident("person")).expect("vertex a");
    let b = db.create_vertex_from_type(ident("person")).expect("vertex b");
    let e = Edge::new(a, ident("knows"), b);
    assert!(db.create_edge(&e).expect("create edge"));
    (a, b, e)
}

fn vertices_of(out: &[QueryOutputValue]) -> Vec<Vertex> {
    match out.last().expect("an output value") {
        QueryOutputValue::Vertices(vs) => vs.clone(),
        other => panic!("expected Vertices, got {other:?}"),
    }
}

fn edges_of(out: &[QueryOutputValue]) -> Vec<Edge> {
    match out.last().expect("an output value") {
        QueryOutputValue::Edges(es) => es.clone(),
        other => panic!("expected Edges, got {other:?}"),
    }
}

/// D2b-AC2: vertex round-trip by id, and vertex creation is id-unique.
#[test]
fn vertex_roundtrip_and_uniqueness() {
    let db = db();
    let a = db.create_vertex_from_type(ident("person")).unwrap();

    let out = db.get(SpecificVertexQuery::single(a)).unwrap();
    let vs = vertices_of(&out);
    assert_eq!(vs.len(), 1);
    assert_eq!(vs[0].id, a);
    assert_eq!(vs[0].t, ident("person"));

    // Re-creating the same uuid reports "not created".
    let dup = Vertex::with_id(a, ident("person"));
    assert!(!db.create_vertex(&dup).unwrap(), "uuid already taken");

    // A vertex that was never created isn't returned.
    let out = db.get(SpecificVertexQuery::single(Uuid::from_u128(999))).unwrap();
    assert!(vertices_of(&out).is_empty());
}

/// D2b-AC3: edges are reachable from both endpoints (forward index and
/// reverse index), and edges to missing vertices are refused.
#[test]
fn edge_roundtrip_both_directions() {
    let db = db();
    let (a, b, e) = seed(&db);

    // Outbound from a.
    let out = db.get(SpecificVertexQuery::single(a).outbound().unwrap()).unwrap();
    let es = edges_of(&out);
    assert_eq!(es.len(), 1);
    assert_eq!(es[0], e);

    // Inbound into b — exercises the reverse index.
    let out = db.get(SpecificVertexQuery::single(b).inbound().unwrap()).unwrap();
    let es = edges_of(&out);
    assert_eq!(es.len(), 1);
    assert_eq!(es[0], e);

    // Specific edge query finds it; a fabricated edge is absent.
    let out = db.get(SpecificEdgeQuery::single(e.clone())).unwrap();
    assert_eq!(edges_of(&out).len(), 1);
    let ghost = Edge::new(a, ident("knows"), Uuid::from_u128(12345));
    let out = db.get(SpecificEdgeQuery::single(ghost.clone())).unwrap();
    assert!(edges_of(&out).is_empty());

    // Creating an edge to a missing vertex is refused, not an error.
    assert!(!db.create_edge(&ghost).unwrap());
}

/// D2b-AC4: vertex and edge properties round-trip (JSON values).
#[test]
fn properties_roundtrip() {
    let db = db();
    let (a, _b, e) = seed(&db);
    let size = ident("size");
    let weight = ident("weight");

    db.set_properties(
        SpecificVertexQuery::single(a),
        size,
        &Json::new(json!({"bytes": 4096})),
    )
    .unwrap();
    db.set_properties(
        SpecificEdgeQuery::single(e.clone()),
        weight,
        &Json::new(json!(7)),
    )
    .unwrap();

    let out = db
        .get(SpecificVertexQuery::single(a).properties().unwrap())
        .unwrap();
    match out.last().unwrap() {
        QueryOutputValue::VertexProperties(props) => {
            assert_eq!(props.len(), 1);
            assert_eq!(props[0].props.len(), 1);
            assert_eq!(props[0].props[0].name, size);
            assert_eq!(*props[0].props[0].value.0, json!({"bytes": 4096}));
        }
        other => panic!("expected VertexProperties, got {other:?}"),
    }

    let out = db
        .get(SpecificEdgeQuery::single(e).properties().unwrap())
        .unwrap();
    match out.last().unwrap() {
        QueryOutputValue::EdgeProperties(props) => {
            assert_eq!(props.len(), 1);
            assert_eq!(props[0].props[0].name, weight);
            assert_eq!(*props[0].props[0].value.0, json!(7));
        }
        other => panic!("expected EdgeProperties, got {other:?}"),
    }
}

/// D2b-AC5: deleting a vertex removes it, its properties, and its incident
/// edges (IndraDB's semantics, which differ from our engine's tombstones).
#[test]
fn delete_vertex_cascades_to_edges() {
    let db = db();
    let (a, b, e) = seed(&db);
    db.set_properties(
        SpecificVertexQuery::single(a),
        ident("k"),
        &Json::new(json!("v")),
    )
    .unwrap();

    db.delete(SpecificVertexQuery::single(a)).unwrap();

    // Vertex gone.
    let out = db.get(SpecificVertexQuery::single(a)).unwrap();
    assert!(vertices_of(&out).is_empty());
    // Its edge is gone from both directions.
    let out = db.get(SpecificEdgeQuery::single(e)).unwrap();
    assert!(edges_of(&out).is_empty());
    let out = db.get(SpecificVertexQuery::single(b).inbound().unwrap()).unwrap();
    assert!(edges_of(&out).is_empty());
    // The surviving endpoint is untouched.
    let out = db.get(SpecificVertexQuery::single(b)).unwrap();
    assert_eq!(vertices_of(&out).len(), 1);
}

/// D2b-AC5 (edges): deleting an edge leaves its endpoints alone.
#[test]
fn delete_edge_keeps_vertices() {
    let db = db();
    let (a, b, e) = seed(&db);
    db.delete(SpecificEdgeQuery::single(e.clone())).unwrap();

    let out = db.get(SpecificEdgeQuery::single(e)).unwrap();
    assert!(edges_of(&out).is_empty());
    for id in [a, b] {
        let out = db.get(SpecificVertexQuery::single(id)).unwrap();
        assert_eq!(vertices_of(&out).len(), 1, "endpoint survives");
    }
}

/// D2b-AC1: the datastore is durable — reopen by name and everything is
/// still there. (Same contract as the engine's `Graph::open_or_create`.)
#[test]
fn reopen_by_name_persists() {
    let name = "idb-reopen";
    let (a, b, e) = {
        let db = TwizzlerDatastore::open_db(name).expect("create by name");
        let (a, b, e) = seed(&db);
        db.set_properties(
            SpecificVertexQuery::single(a),
            ident("k"),
            &Json::new(json!(42)),
        )
        .unwrap();
        (a, b, e)
    };

    let db = TwizzlerDatastore::open_db(name).expect("reopen by name");
    let out = db.get(SpecificVertexQuery::single(a)).unwrap();
    assert_eq!(vertices_of(&out).len(), 1, "vertex survived reopen");
    let out = db.get(SpecificEdgeQuery::single(e)).unwrap();
    assert_eq!(edges_of(&out).len(), 1, "edge survived reopen");
    let out = db.get(SpecificVertexQuery::single(b).inbound().unwrap()).unwrap();
    assert_eq!(edges_of(&out).len(), 1, "reverse index survived reopen");
    let out = db
        .get(SpecificVertexQuery::single(a).properties().unwrap())
        .unwrap();
    match out.last().unwrap() {
        QueryOutputValue::VertexProperties(props) => {
            assert_eq!(*props[0].props[0].value.0, json!(42));
        }
        other => panic!("expected VertexProperties, got {other:?}"),
    }
}

/// Property-index queries: unindexed names report "not indexed" (IndraDB
/// surfaces our `Ok(None)` as `NotIndexed`); indexed ones resolve.
#[test]
fn property_index_declared_then_queryable() {
    let db = db();
    let (a, _b, _e) = seed(&db);
    let name = ident("tag");
    db.set_properties(
        SpecificVertexQuery::single(a),
        name,
        &Json::new(json!("thesis")),
    )
    .unwrap();

    // Before indexing, a property query is rejected.
    let q = indradb::VertexWithPropertyValueQuery::new(name, Json::new(json!("thesis")));
    assert!(db.get(q).is_err(), "unindexed property query must fail");

    // After declaring the index, it resolves.
    db.index_property(name).unwrap();
    let q = indradb::VertexWithPropertyValueQuery::new(name, Json::new(json!("thesis")));
    let out = db.get(q).expect("indexed property query");
    let vs = vertices_of(&out);
    assert_eq!(vs.len(), 1);
    assert_eq!(vs[0].id, a);
}
