//! NameKey and label/name edge cases: truncation, UTF-8 boundaries, empty
//! strings, and duplicate (label, name) pairs.

use crate::Lookup;
use twizzler::object::ObjID;

use super::fresh;

/// Names longer than 31 bytes truncate byte-wise on store and on lookup, so
/// the full string still finds the vertex and prefix-sharing names collide.
#[test]
fn long_names_truncate_to_31_bytes() {
    let mut g = fresh("t-longname");
    let long = "a".repeat(40);
    let v = g.add_vertex("n", &long, ObjID::new(0)).unwrap();

    // Stored name is the 31-byte prefix.
    assert_eq!(g.vertex_info(v).unwrap().name, "a".repeat(31));
    // Lookup with the full string hits: the key truncates identically.
    assert_eq!(g.find_vertex("n", &long), Lookup::Found(v));
    // ...which means prefix-sharing names collide by design.
    assert_eq!(g.find_vertex("n", &"a".repeat(35)), Lookup::Found(v));
}

/// A multibyte char spanning byte 31 leaves invalid UTF-8 in the key, so
/// `NameKey::as_str` degrades to ""; byte-wise equality still finds the
/// vertex.
#[test]
fn multibyte_char_at_truncation_boundary() {
    let mut g = fresh("t-utf8name");
    // 30 ASCII bytes then a 2-byte 'é': byte 31 splits the char.
    let name = format!("{}é", "x".repeat(30));
    let v = g.add_vertex("n", &name, ObjID::new(0)).unwrap();

    assert_eq!(g.vertex_info(v).unwrap().name, "");
    assert_eq!(g.find_vertex("n", &name), Lookup::Found(v));
}

/// Empty labels and names are legal keys.
#[test]
fn empty_label_and_name() {
    let mut g = fresh("t-empty");
    let v = g.add_vertex("", "", ObjID::new(0)).unwrap();
    assert_eq!(g.find_vertex("", ""), Lookup::Found(v));
    let info = g.vertex_info(v).unwrap();
    assert_eq!(info.label, "");
    assert_eq!(info.name, "");
    assert_eq!(g.vertices_by_label(""), vec![v]);
}

/// `NameKey` orders by content, not length: a derived `Ord` would compare
/// `len` before `bytes` and put `"z"` before `"aa"`. Mixed lengths catch that.
#[test]
fn namekey_orders_by_content_not_length() {
    use crate::{NameKey, PropValue};

    // Direct: content order, where length-first order would invert.
    assert!(NameKey::new("aa") < NameKey::new("z"), "content order: aa < z");
    assert!(NameKey::new("ab") < NameKey::new("b"));
    assert!(NameKey::new("b") < NameKey::new("ba"));
    // And through PropValue::Str, which is what order_by_prop compares.
    assert!(PropValue::str("aa") < PropValue::str("z"));

    // Graph-level: order_by_name over mixed lengths.
    let mut g = fresh("t-namekey-order");
    let z = g.add_vertex("n", "z", ObjID::new(0)).unwrap();
    let aa = g.add_vertex("n", "aa", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    let ordered = g.traversal().vs(&[z, aa, b]).order_by_name().to_ids();
    assert_eq!(
        ordered,
        vec![aa, b, z],
        "order_by_name must sort by string content; a length-first order \
         would put z before aa"
    );
}

/// Duplicate (label, name) pairs are allowed; the index binds the last write,
/// and deleting the bound vertex does not resurface the shadowed one.
#[test]
fn duplicate_label_name_pairs() {
    let mut g = fresh("t-dupname");
    let v1 = g.add_vertex("tag", "x", ObjID::new(0)).unwrap();
    let v2 = g.add_vertex("tag", "x", ObjID::new(0)).unwrap();

    // Both records exist independently...
    assert!(g.vertex_info(v1).is_some());
    assert!(g.vertex_info(v2).is_some());
    assert_eq!(g.vertices_by_label("tag").len(), 2);
    // ...but the index resolves to the most recent insert.
    assert_eq!(g.find_vertex("tag", "x"), Lookup::Found(v2));

    // Deleting the bound one leaves the key unbound (not versioned).
    g.delete_vertex(v2).unwrap();
    assert_eq!(g.find_vertex("tag", "x"), Lookup::NotFound);
    // The shadowed vertex is still alive and enumerable.
    assert_eq!(g.vertices_by_label("tag"), vec![v1]);
}
