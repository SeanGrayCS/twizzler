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

/// D2b-AC1: the datastore state survives reopen by name. (Same contract as
/// the engine's `Graph::open_or_create`.)
///
/// **Within-boot only, and post-D2c that distinction is load-bearing**
/// (2026-08-18 audit, Part 2 item 9): writes are `nosync` now, so without the
/// `sync` below this test passed purely on same-boot shared object state and
/// evidenced nothing about durability. The KV layer's flush tests carry the
/// in-boot durability property; the cross-*reboot* half is
/// `gstress indradb-seed <N>` → reboot → `gstress indradb-verify <N>`, added
/// the same day, because a within-boot reopen maps pages that are still
/// resident and cannot tell durable from merely mapped (F1-AC0).
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
        // Post-D2c the durability point is `sync`, not `put` — a reopen test
        // that never syncs is testing the page cache.
        db.sync().expect("sync");
        (a, b, e)
    };

    let db = TwizzlerDatastore::open_db(name).expect("reopen by name");
    let out = db.get(SpecificVertexQuery::single(a)).unwrap();
    assert_eq!(vertices_of(&out).len(), 1, "vertex survived reopen");
    // D1-AC2's missed half (2026-08-18 audit): the second vertex was never
    // queried by its own id — only through the edge's reverse index.
    let out = db.get(SpecificVertexQuery::single(b)).unwrap();
    assert_eq!(
        vertices_of(&out).len(),
        1,
        "the second vertex, addressed by its own id"
    );
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

/// **D2c-AC5: an indexed property is answered from the index.**
///
/// The pre-D2c adapter recorded the declaration and scanned every vertex
/// property at query time, so this passed while doing exactly the work the
/// index exists to avoid. It cannot assert "no scan happened" through the
/// public API, so it asserts the observable consequence instead: the answer
/// must be exact even when many vertices carry the same property name with
/// *different* values, which is the case a scan-and-filter gets right and a
/// mis-keyed index gets wrong.
#[test]
fn indexed_property_value_query_is_exact() {
    let db = db();
    let name = ident("ldbcId");
    db.index_property(name).expect("index");

    let mut want = Vec::new();
    for i in 0..25u32 {
        let v = db.create_vertex_from_type(ident("person")).expect("vertex");
        // Two vertices share id "7"; the rest are distinct. A prefix-keyed
        // index that forgets its value terminator would also return "70".
        let id = if i == 7 || i == 19 { "7".to_string() } else { i.to_string() };
        db.set_properties(
            SpecificVertexQuery::single(v),
            name,
            &Json::new(id.clone().into()),
        )
        .expect("set");
        if id == "7" {
            want.push(v);
        }
    }
    let _ = db.create_vertex_from_type(ident("person")).expect("bare");

    let q = indradb::VertexWithPropertyValueQuery::new(name, Json::new("7".into()));
    let mut got: Vec<Uuid> = vertices_of(&db.get(q).expect("query")).iter().map(|v| v.id).collect();
    got.sort();
    want.sort();
    assert_eq!(got, want, "exactly the two vertices with ldbcId=7");
}

/// **D2c-AC6: the index stays true under mutation.**
///
/// Each of these leaves a stale entry if maintenance is missed, and a stale
/// entry is worse than no index: it resurrects deleted data in query results.
#[test]
fn property_index_maintained_on_overwrite_and_delete() {
    let db = db();
    let name = ident("ldbcId");
    db.index_property(name).expect("index");

    let a = db.create_vertex_from_type(ident("person")).expect("a");
    let b = db.create_vertex_from_type(ident("person")).expect("b");
    let set = |v: Uuid, s: &str| {
        db.set_properties(SpecificVertexQuery::single(v), name, &Json::new(s.into()))
            .expect("set");
    };
    let find = |s: &str| -> Vec<Uuid> {
        let q = indradb::VertexWithPropertyValueQuery::new(name, Json::new(s.into()));
        vertices_of(&db.get(q).expect("query")).iter().map(|v| v.id).collect()
    };

    set(a, "old");
    set(b, "keep");
    assert_eq!(find("old"), vec![a]);

    // Overwrite: the old value must stop matching.
    set(a, "new");
    assert!(find("old").is_empty(), "stale entry for the overwritten value");
    assert_eq!(find("new"), vec![a]);

    // Deleting the vertex must take its index entry with it.
    db.delete(SpecificVertexQuery::single(a)).expect("delete vertex");
    assert!(find("new").is_empty(), "deleted vertex still in the index");
    assert_eq!(find("keep"), vec![b], "denial is scoped to the deleted vertex");
}

/// **D2c-AC6: `index_property` backfills.** IndraDB allows declaring an index
/// after the data exists; the LDBC loader happens to declare first, which is
/// exactly why this would otherwise go untested.
#[test]
fn index_property_backfills_existing_vertices() {
    let db = db();
    let name = ident("late");
    let v = db.create_vertex_from_type(ident("person")).expect("vertex");
    db.set_properties(SpecificVertexQuery::single(v), name, &Json::new("x".into()))
        .expect("set before index");

    db.index_property(name).expect("index after the fact");

    let q = indradb::VertexWithPropertyValueQuery::new(name, Json::new("x".into()));
    let got: Vec<Uuid> = vertices_of(&db.get(q).expect("query")).iter().map(|v| v.id).collect();
    assert_eq!(got, vec![v], "pre-existing vertex was not backfilled");
}

/// **D2c-AC1/AC2 at the datastore level:** `sync` is the durability point, and
/// it must be reachable through the public API without error.
#[test]
fn sync_is_a_real_flush() {
    let db = db();
    let (a, _b, _e) = seed(&db);
    db.sync().expect("sync must succeed, not merely be overridden away");
    let out = db.get(SpecificVertexQuery::single(a)).expect("get after sync");
    assert_eq!(vertices_of(&out).len(), 1);
}
