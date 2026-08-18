//! Typed properties on vertices and edges: round-trips, overwrite,
//! enumeration order, persistence, and error cases.

use twizzler::object::ObjID;

use super::fresh;
use crate::{EdgeId, Graph, PropValue, VertexId};

/// A fresh vertex has no props; a set round-trips.
#[test]
fn vertex_prop_roundtrip_and_missing() {
    let mut g = fresh("t-prop1");
    let v = g.add_vertex("file", "doc", ObjID::new(0)).unwrap();
    assert_eq!(g.get_vertex_prop(v, "size"), None);
    g.set_vertex_prop(v, "size", PropValue::U64(4096)).unwrap();
    assert_eq!(g.get_vertex_prop(v, "size"), Some(PropValue::U64(4096)));
    // Unrelated key remains unset.
    assert_eq!(g.get_vertex_prop(v, "kind"), None);
}

/// Overwrite replaces in place (position preserved); distinct keys coexist;
/// enumeration returns insertion order.
#[test]
fn prop_overwrite_multiple_keys_and_enumeration() {
    let mut g = fresh("t-prop2");
    let v = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    g.set_vertex_prop(v, "a", PropValue::U64(1)).unwrap();
    g.set_vertex_prop(v, "b", PropValue::I64(-2)).unwrap();
    g.set_vertex_prop(v, "c", PropValue::Bool(true)).unwrap();
    g.set_vertex_prop(v, "d", PropValue::str("x")).unwrap();
    // Overwrite the first key after others were added.
    g.set_vertex_prop(v, "a", PropValue::U64(100)).unwrap();

    assert_eq!(g.get_vertex_prop(v, "a"), Some(PropValue::U64(100)));
    assert_eq!(g.get_vertex_prop(v, "b"), Some(PropValue::I64(-2)));
    assert_eq!(g.get_vertex_prop(v, "c"), Some(PropValue::Bool(true)));
    assert_eq!(g.get_vertex_prop(v, "d"), Some(PropValue::str("x")));

    let all = g.vertex_props(v);
    assert_eq!(all.len(), 4);
    // Insertion order, with "a" still first despite the late overwrite.
    let keys: Vec<&str> = all.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, vec!["a", "b", "c", "d"]);
    assert_eq!(all[0].1, PropValue::U64(100));
}

/// Edge properties round-trip like vertex properties.
#[test]
fn edge_prop_roundtrip() {
    let mut g = fresh("t-prop3");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let e = g.add_edge(a, "rel", b).unwrap();

    assert_eq!(g.get_edge_prop(e, "weight"), None);
    g.set_edge_prop(e, "weight", PropValue::I64(7)).unwrap();
    assert_eq!(g.get_edge_prop(e, "weight"), Some(PropValue::I64(7)));
    assert_eq!(g.edge_props(e).len(), 1);
    // Vertex props and edge props are separate stores.
    assert_eq!(g.get_vertex_prop(a, "weight"), None);
}

/// Properties persist across reopen-by-name.
#[test]
fn props_persist_on_reopen() {
    let name = "t-prop4";
    let _ = Graph::reset(name);
    let (v, e) = {
        let mut g = Graph::open_or_create(name).unwrap();
        let v = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
        let w = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
        let e = g.add_edge(v, "rel", w).unwrap();
        g.set_vertex_prop(v, "size", PropValue::U64(9)).unwrap();
        g.set_edge_prop(e, "weight", PropValue::I64(-3)).unwrap();
        (v, e)
    };
    let g = Graph::open_or_create(name).unwrap();
    assert_eq!(g.get_vertex_prop(v, "size"), Some(PropValue::U64(9)));
    assert_eq!(g.get_edge_prop(e, "weight"), Some(PropValue::I64(-3)));
    let _ = Graph::reset(name);
}

/// Tombstoned elements hide their properties.
#[test]
fn deleted_element_props_hidden() {
    let mut g = fresh("t-prop5");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let e = g.add_edge(a, "rel", b).unwrap();
    g.set_vertex_prop(a, "k", PropValue::Bool(true)).unwrap();
    g.set_edge_prop(e, "k", PropValue::Bool(false)).unwrap();

    g.delete_edge(e).unwrap();
    assert_eq!(g.get_edge_prop(e, "k"), None);
    assert!(g.edge_props(e).is_empty());

    g.delete_vertex(a).unwrap();
    assert_eq!(g.get_vertex_prop(a, "k"), None);
    assert!(g.vertex_props(a).is_empty());
}

/// A graph written without any props reopens normally — no `StaleVersion`,
/// and reads return `None`.
#[test]
fn propless_graph_compat() {
    let name = "t-prop6";
    let _ = Graph::reset(name);
    let v = {
        let mut g = Graph::open_or_create(name).unwrap();
        g.add_vertex("n", "plain", ObjID::new(0)).unwrap()
    };
    let g = Graph::open_or_create(name).expect("no StaleVersion");
    assert_eq!(g.get_vertex_prop(v, "anything"), None);
    assert!(g.vertex_props(v).is_empty());
    assert_eq!(g.vertex_info(v).unwrap().name, "plain");
    let _ = Graph::reset(name);
}

/// Str values truncate byte-wise at 31, exactly like `NameKey`.
#[test]
fn str_props_truncate_like_namekey() {
    let mut g = fresh("t-prop7");
    let v = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let long = "z".repeat(40);
    g.set_vertex_prop(v, "s", PropValue::str(&long)).unwrap();
    assert_eq!(
        g.get_vertex_prop(v, "s"),
        Some(PropValue::str(&"z".repeat(31)))
    );
    // Keys truncate the same way: a 40-byte key and its 31-byte prefix
    // collide (same NameKey semantics as vertex names).
    g.set_vertex_prop(v, &long, PropValue::U64(1)).unwrap();
    assert_eq!(
        g.get_vertex_prop(v, &"z".repeat(31)),
        Some(PropValue::U64(1))
    );
}

/// Every variant round-trips: missing → set → overwritten per variant,
/// including cross-variant overwrite.
#[test]
fn all_variants_roundtrip() {
    let mut g = fresh("t-prop8");
    let v = g.add_vertex("n", "a", ObjID::new(0)).unwrap();

    let cases: Vec<(&str, PropValue)> = vec![
        ("i", PropValue::I64(-42)),
        ("u", PropValue::U64(u64::MAX)),
        ("bt", PropValue::Bool(true)),
        ("bf", PropValue::Bool(false)),
        ("o", PropValue::ObjId(0xDEAD_BEEF_0000_0001_u128)),
        ("s", PropValue::str("hello")),
    ];
    for (k, val) in &cases {
        assert_eq!(g.get_vertex_prop(v, k), None, "missing before set: {k}");
        g.set_vertex_prop(v, k, *val).unwrap();
        assert_eq!(g.get_vertex_prop(v, k), Some(*val), "set: {k}");
    }
    // Cross-variant overwrite: U64 -> Str, Str -> I64.
    g.set_vertex_prop(v, "u", PropValue::str("now-a-string")).unwrap();
    assert_eq!(
        g.get_vertex_prop(v, "u"),
        Some(PropValue::str("now-a-string"))
    );
    g.set_vertex_prop(v, "s", PropValue::I64(0)).unwrap();
    assert_eq!(g.get_vertex_prop(v, "s"), Some(PropValue::I64(0)));
    assert_eq!(g.vertex_props(v).len(), cases.len());
}

/// Setting on missing or tombstoned elements errors; the graph stays usable
/// afterward.
#[test]
fn set_on_missing_or_deleted_errors() {
    let mut g = fresh("t-prop9");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let e = g.add_edge(a, "rel", b).unwrap();

    assert!(g
        .set_vertex_prop(VertexId(9999), "k", PropValue::Bool(true))
        .is_err());
    assert!(g
        .set_edge_prop(EdgeId(9999), "k", PropValue::Bool(true))
        .is_err());

    g.delete_vertex(b).unwrap();
    assert!(g.set_vertex_prop(b, "k", PropValue::Bool(true)).is_err());
    // Edge died with its endpoint; setting on it errors too.
    assert!(g.set_edge_prop(e, "k", PropValue::Bool(true)).is_err());

    // Still usable.
    g.set_vertex_prop(a, "k", PropValue::U64(1)).unwrap();
    assert_eq!(g.get_vertex_prop(a, "k"), Some(PropValue::U64(1)));
}
