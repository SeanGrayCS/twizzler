//! NameKey and label/name edge cases: truncation, UTF-8 boundaries, empty
//! strings, and duplicate (label, name) pairs.

use twizzler::object::ObjID;

use super::fresh;

#[test]
fn long_names_truncate_to_31_bytes() {
    let mut g = fresh("t-longname");
    let long = "a".repeat(40);
    let v = g.add_vertex("n", &long, ObjID::new(0)).unwrap();

    // Stored name is the 31-byte prefix.
    assert_eq!(g.vertex_info(v).unwrap().name, "a".repeat(31));
    // Lookup with the full string hits: the key truncates identically.
    assert_eq!(g.find_vertex("n", &long), Some(v));
    // ...which means prefix-sharing names collide by design.
    assert_eq!(g.find_vertex("n", &"a".repeat(35)), Some(v));
}

#[test]
fn multibyte_char_at_truncation_boundary() {
    let mut g = fresh("t-utf8name");
    // 30 ASCII bytes then a 2-byte 'é': byte 31 splits the char.
    let name = format!("{}é", "x".repeat(30));
    let v = g.add_vertex("n", &name, ObjID::new(0)).unwrap();

    assert_eq!(g.vertex_info(v).unwrap().name, "");
    assert_eq!(g.find_vertex("n", &name), Some(v));
}

#[test]
fn empty_label_and_name() {
    let mut g = fresh("t-empty");
    let v = g.add_vertex("", "", ObjID::new(0)).unwrap();
    assert_eq!(g.find_vertex("", ""), Some(v));
    let info = g.vertex_info(v).unwrap();
    assert_eq!(info.label, "");
    assert_eq!(info.name, "");
    assert_eq!(g.vertices_by_label(""), vec![v]);
}

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
    assert_eq!(g.find_vertex("tag", "x"), Some(v2));

    // Deleting the bound one leaves the key unbound (not versioned).
    g.delete_vertex(v2).unwrap();
    assert_eq!(g.find_vertex("tag", "x"), None);
    // The shadowed vertex is still alive and enumerable.
    assert_eq!(g.vertices_by_label("tag"), vec![v1]);
}
