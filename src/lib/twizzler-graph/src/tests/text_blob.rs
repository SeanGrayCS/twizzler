//! Long text values and out-of-line blobs.
//!
//! Text past the 31-byte `PropValue::Str` tier stays queryable; blobs are
//! write-and-read-back only, with deliberately no filter method.

use twizzler::object::ObjID;

use super::fresh;
use crate::{Graph, Labels, PropValue};

/// `n` bytes of distinguishable text.
fn long(n: usize) -> String {
    ('a'..='z').cycle().take(n).collect()
}

/// Text values from empty to 255 bytes round-trip exactly.
#[test]
fn text_survives_past_the_old_31_byte_cap() {
    let mut g = fresh("t-a9-text");
    let v = g.add_vertex("n", "v", ObjID::new(0)).unwrap();

    // The lengths cross the 31/32 boundary of the inline `Str` tier.
    for n in [0usize, 1, 31, 32, 124, 255] {
        let s = long(n);
        g.set_vertex_text(v, "t", &s).expect("set text");
        assert_eq!(
            g.get_vertex_text(v, "t").as_deref(),
            Some(s.as_str()),
            "{n}-byte value did not round-trip"
        );
    }
}

/// Text over the 255-byte limit is an error, not a truncation.
#[test]
fn text_over_the_limit_is_refused_not_truncated() {
    let mut g = fresh("t-a9-limit");
    let v = g.add_vertex("n", "v", ObjID::new(0)).unwrap();
    assert!(
        g.set_vertex_text(v, "t", &long(256)).is_err(),
        "256 bytes must be refused; truncating is the bug A9 exists to fix"
    );
    assert_eq!(g.get_vertex_text(v, "t"), None, "a refused write stores nothing");
}

/// Long text is queryable: `has_text` matches the full value, not a prefix.
#[test]
fn long_text_filters_exactly() {
    let mut g = fresh("t-a9-filter");
    let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("n", "b", ObjID::new(0)).unwrap();
    // Differ only in the last byte, well past 31: a prefix comparison would
    // match both.
    let base = long(120);
    g.set_vertex_text(a, "name", &format!("{base}X")).unwrap();
    g.set_vertex_text(b, "name", &format!("{base}Y")).unwrap();

    let hits = g
        .traversal()
        .vertices()
        .has_text("name", &format!("{base}X"))
        .to_ids();
    assert_eq!(hits, vec![a], "must not match on a truncated prefix");
}

/// A blob holds far more than a text value and comes back byte-exact.
#[test]
fn blob_round_trips_multi_kilobyte_bytes() {
    let mut g = fresh("t-a9-blob");
    let v = g.add_vertex("n", "v", ObjID::new(0)).unwrap();
    let payload: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
    g.set_vertex_blob(v, "content", &payload).expect("set blob");
    assert_eq!(g.get_vertex_blob(v, "content").as_deref(), Some(&payload[..]));
}

/// A blob and a text property are different things and are not readable as
/// each other.
#[test]
fn blob_and_text_do_not_alias() {
    let mut g = fresh("t-a9-alias");
    let v = g.add_vertex("n", "v", ObjID::new(0)).unwrap();
    g.set_vertex_blob(v, "content", &vec![7u8; 1000]).unwrap();
    assert_eq!(g.get_vertex_text(v, "content"), None, "a blob is not text");

    g.set_vertex_text(v, "title", "short").unwrap();
    assert_eq!(g.get_vertex_blob(v, "title"), None, "text is not a blob");
}

/// Blobs share backing objects rather than allocating one object each.
#[test]
fn blobs_share_backing_objects() {
    let mut g = fresh("t-a9-blob-objects");
    let before = g.blob_object_count();
    for i in 0..256 {
        let v = g
            .add_vertex("n", &format!("v{i}"), ObjID::new(0))
            .unwrap();
        g.set_vertex_blob(v, "content", &vec![(i % 256) as u8; 512])
            .unwrap();
    }
    let after = g.blob_object_count();
    assert!(
        after - before <= 4,
        "256 blobs (128 KB) took {} objects; blobs must share backing storage",
        after - before
    );
}

/// A record carrying no long value does not grow by a byte.
#[test]
fn records_without_long_values_do_not_grow() {
    assert_eq!(
        crate::record_size_for(0),
        crate::RECORD_SIZE_NO_PROPS,
        "the zero-property record stride changed; every edge record pays for this"
    );
}

/// Text and blob values survive a within-boot reopen; the cross-boot half
/// lives in `gstress seed`/`verify`.
#[test]
fn text_and_blob_survive_reopen() {
    let name = "t-a9-reopen";
    let text = long(200);
    let blob: Vec<u8> = (0..3000u32).map(|i| (i % 253) as u8).collect();
    let v = {
        let mut g = fresh(name);
        let v = g.add_vertex("n", "v", ObjID::new(0)).unwrap();
        g.set_vertex_text(v, "name", &text).unwrap();
        g.set_vertex_blob(v, "content", &blob).unwrap();
        g.sync().unwrap();
        v
    };
    let g = Graph::open_or_create_arena(name, super::TEST_ARENA_CAP).expect("reopen");
    assert_eq!(g.get_vertex_text(v, "name").as_deref(), Some(text.as_str()));
    assert_eq!(g.get_vertex_blob(v, "content").as_deref(), Some(&blob[..]));
}

/// Short values keep the plain `PropValue::Str` path and semantics.
#[test]
fn short_propvalue_path_is_unchanged() {
    let mut g = fresh("t-a9-compat");
    let v = g.add_vertex("n", "v", ObjID::new(0)).unwrap();
    g.set_vertex_prop(v, "k", PropValue::str("still short"))
        .unwrap();
    assert_eq!(
        g.get_vertex_prop(v, "k"),
        Some(PropValue::str("still short"))
    );
    let hit = g
        .traversal()
        .vertices()
        .has("k", PropValue::str("still short"))
        .to_ids();
    assert_eq!(hit, vec![v]);
    let _ = Labels::any();
}
