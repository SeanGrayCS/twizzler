//! Key encoding for the KV-backed IndraDB datastore (board task D2b).
//!
//! Everything IndraDB stores becomes a byte key in one flat sorted space
//! (see [`crate::kv`]), partitioned by a one-byte namespace tag. The encoding
//! must make **byte order equal IndraDB's own `Ord`**, because
//! `Transaction::range_vertices` / `range_edges` / `range_reversed_edges`
//! promise "everything at or after this value" in that order. Tests below pin
//! that equivalence against the real `Uuid`/`Edge` orderings rather than
//! trusting it.
//!
//! ```text
//! [0x01][uuid:16]                                   -> vertex type bytes
//! [0x02][out:16][type][0x00][in:16]                 -> (empty) edge exists
//! [0x03][in:16][type][0x00][out:16]                 -> (empty) reverse index
//! [0x04][uuid:16][name][0x00]                       -> vertex property JSON
//! [0x05][out:16][type][0x00][in:16][name][0x00]     -> edge property JSON
//! [0x06][name][0x00]                                -> (empty) property is indexed
//! ```
//!
//! **Why a `0x00` terminator rather than a length prefix.** `Identifier` is
//! restricted to letters, digits, dashes and underscores, so `0x00` cannot
//! occur inside one and sorts below every byte that can. That makes the
//! prefix case order correctly: `"a" < "ab"` encodes as `a\0…` vs `ab…`, and
//! `0x00 < b'b'`. A length prefix would invert it (`1a` > `2ab` … wrong), and
//! silently corrupt every range query.

use indradb::{Edge, Identifier};
use uuid::Uuid;

const NS_VERTEX: u8 = 0x01;
const NS_EDGE: u8 = 0x02;
const NS_REV_EDGE: u8 = 0x03;
const NS_VERTEX_PROP: u8 = 0x04;
const NS_EDGE_PROP: u8 = 0x05;
const NS_INDEXED: u8 = 0x06;

/// Namespace tags for whole-namespace scans (property-index queries sweep
/// every property key, since indexes are declared rather than built).
pub(crate) const VERTEX_PROP_TAG: u8 = NS_VERTEX_PROP;
pub(crate) const EDGE_PROP_TAG: u8 = NS_EDGE_PROP;

/// Terminator for variable-length `Identifier` segments; see module docs.
const TERM: u8 = 0x00;

fn push_ident(out: &mut Vec<u8>, t: &Identifier) {
    out.extend_from_slice(t.as_str().as_bytes());
    out.push(TERM);
}

/// Split at the first terminator: (segment, rest-after-terminator).
fn split_ident(bytes: &[u8]) -> Option<(Identifier, &[u8])> {
    let end = bytes.iter().position(|b| *b == TERM)?;
    let s = core::str::from_utf8(&bytes[..end]).ok()?;
    let ident = Identifier::new(s).ok()?;
    Some((ident, &bytes[end + 1..]))
}

fn take_uuid(bytes: &[u8]) -> Option<(Uuid, &[u8])> {
    if bytes.len() < 16 {
        return None;
    }
    let arr: [u8; 16] = bytes[..16].try_into().ok()?;
    Some((Uuid::from_bytes(arr), &bytes[16..]))
}

// --- vertices --------------------------------------------------------------

pub(crate) fn vertex_key(id: Uuid) -> Vec<u8> {
    let mut k = Vec::with_capacity(17);
    k.push(NS_VERTEX);
    k.extend_from_slice(id.as_bytes());
    k
}

/// Prefix matching every vertex — the scan start for `all_vertices`.
pub(crate) fn vertex_prefix() -> Vec<u8> {
    vec![NS_VERTEX]
}

pub(crate) fn decode_vertex_key(k: &[u8]) -> Option<Uuid> {
    let rest = k.strip_prefix(&[NS_VERTEX])?;
    let (id, tail) = take_uuid(rest)?;
    tail.is_empty().then_some(id)
}

// --- edges -----------------------------------------------------------------

pub(crate) fn edge_key(e: &Edge) -> Vec<u8> {
    let mut k = Vec::with_capacity(40);
    k.push(NS_EDGE);
    k.extend_from_slice(e.outbound_id.as_bytes());
    push_ident(&mut k, &e.t);
    k.extend_from_slice(e.inbound_id.as_bytes());
    k
}

/// The same edge keyed for the reverse index: (inbound, type, outbound).
pub(crate) fn rev_edge_key(e: &Edge) -> Vec<u8> {
    let mut k = Vec::with_capacity(40);
    k.push(NS_REV_EDGE);
    k.extend_from_slice(e.inbound_id.as_bytes());
    push_ident(&mut k, &e.t);
    k.extend_from_slice(e.outbound_id.as_bytes());
    k
}

pub(crate) fn edge_prefix() -> Vec<u8> {
    vec![NS_EDGE]
}

pub(crate) fn rev_edge_prefix() -> Vec<u8> {
    vec![NS_REV_EDGE]
}

fn decode_edge_body(rest: &[u8]) -> Option<(Uuid, Identifier, Uuid)> {
    let (first, rest) = take_uuid(rest)?;
    let (t, rest) = split_ident(rest)?;
    let (second, tail) = take_uuid(rest)?;
    tail.is_empty().then_some((first, t, second))
}

pub(crate) fn decode_edge_key(k: &[u8]) -> Option<Edge> {
    let rest = k.strip_prefix(&[NS_EDGE])?;
    let (out, t, inb) = decode_edge_body(rest)?;
    Some(Edge::new(out, t, inb))
}

/// Decode a reverse-index key. Returns the edge in **reversed form** (the
/// stored orientation), which is what `range_reversed_edges` yields; flip it
/// with [`flip`] to recover the real edge.
pub(crate) fn decode_rev_edge_key(k: &[u8]) -> Option<Edge> {
    let rest = k.strip_prefix(&[NS_REV_EDGE])?;
    let (inb, t, out) = decode_edge_body(rest)?;
    Some(Edge::new(inb, t, out))
}

/// Swap an edge's endpoints (real form <-> reversed form).
pub(crate) fn flip(e: &Edge) -> Edge {
    Edge::new(e.inbound_id, e.t.clone(), e.outbound_id)
}

// --- properties ------------------------------------------------------------

pub(crate) fn vertex_prop_key(id: Uuid, name: &Identifier) -> Vec<u8> {
    let mut k = Vec::with_capacity(24);
    k.push(NS_VERTEX_PROP);
    k.extend_from_slice(id.as_bytes());
    push_ident(&mut k, name);
    k
}

/// Prefix matching every property of one vertex.
pub(crate) fn vertex_props_prefix(id: Uuid) -> Vec<u8> {
    let mut k = Vec::with_capacity(17);
    k.push(NS_VERTEX_PROP);
    k.extend_from_slice(id.as_bytes());
    k
}

pub(crate) fn decode_vertex_prop_key(k: &[u8]) -> Option<(Uuid, Identifier)> {
    let rest = k.strip_prefix(&[NS_VERTEX_PROP])?;
    let (id, rest) = take_uuid(rest)?;
    let (name, tail) = split_ident(rest)?;
    tail.is_empty().then_some((id, name))
}

pub(crate) fn edge_prop_key(e: &Edge, name: &Identifier) -> Vec<u8> {
    let mut k = Vec::with_capacity(48);
    k.push(NS_EDGE_PROP);
    k.extend_from_slice(e.outbound_id.as_bytes());
    push_ident(&mut k, &e.t);
    k.extend_from_slice(e.inbound_id.as_bytes());
    push_ident(&mut k, name);
    k
}

/// Prefix matching every property of one edge.
pub(crate) fn edge_props_prefix(e: &Edge) -> Vec<u8> {
    let mut k = Vec::with_capacity(40);
    k.push(NS_EDGE_PROP);
    k.extend_from_slice(e.outbound_id.as_bytes());
    push_ident(&mut k, &e.t);
    k.extend_from_slice(e.inbound_id.as_bytes());
    k
}

pub(crate) fn decode_edge_prop_key(k: &[u8]) -> Option<(Edge, Identifier)> {
    let rest = k.strip_prefix(&[NS_EDGE_PROP])?;
    let (out, t, rest) = {
        let (out, rest) = take_uuid(rest)?;
        let (t, rest) = split_ident(rest)?;
        (out, t, rest)
    };
    let (inb, rest) = take_uuid(rest)?;
    let (name, tail) = split_ident(rest)?;
    tail.is_empty()
        .then_some((Edge::new(out, t, inb), name))
}

// --- indexed-property registry ---------------------------------------------

pub(crate) fn indexed_key(name: &Identifier) -> Vec<u8> {
    let mut k = Vec::with_capacity(16);
    k.push(NS_INDEXED);
    push_ident(&mut k, name);
    k
}

/// Prefix matching every declared property index.
///
/// Unused by the datastore today — `index_property` writes one key and
/// `is_indexed` reads one key, neither needing to enumerate. Kept (with its
/// decoder) because listing declared indexes is exactly what the capability
/// matrix and any future index-rebuild would ask for, and a codec that can
/// write a namespace but not read it back is half a codec.
#[allow(dead_code)]
pub(crate) fn indexed_prefix() -> Vec<u8> {
    vec![NS_INDEXED]
}

#[allow(dead_code)]
pub(crate) fn decode_indexed_key(k: &[u8]) -> Option<Identifier> {
    let rest = k.strip_prefix(&[NS_INDEXED])?;
    let (name, tail) = split_ident(rest)?;
    tail.is_empty().then_some(name)
}

#[cfg(test)]
mod tests {
    //! D2b-1: round-trips, and — the load-bearing part — that encoded byte
    //! order equals IndraDB's own `Ord` for `Uuid` and `Edge`.

    use indradb::{Edge, Identifier};
    use uuid::Uuid;

    use super::*;

    fn ident(s: &str) -> Identifier {
        Identifier::new(s).expect("valid identifier")
    }

    fn uuid(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    #[test]
    fn vertex_key_roundtrip() {
        let id = uuid(0x1234_5678_9abc_def0_1122_3344_5566_7788);
        let k = vertex_key(id);
        assert!(k.starts_with(&vertex_prefix()));
        assert_eq!(decode_vertex_key(&k), Some(id));
        // Cross-namespace decodes reject.
        assert_eq!(decode_vertex_key(&edge_key(&Edge::new(id, ident("t"), id))), None);
    }

    /// Load-bearing: vertex key order == `Uuid` order, so `range_vertices`
    /// returns what IndraDB promises.
    #[test]
    fn vertex_key_order_matches_uuid_order() {
        let mut ids: Vec<Uuid> = vec![
            uuid(u128::MAX),
            uuid(0),
            uuid(1),
            uuid(0x0100_0000_0000_0000_0000_0000_0000_0000),
            uuid(0xff),
        ];
        let mut by_key = ids.clone();
        by_key.sort_by_key(|i| vertex_key(*i));
        ids.sort();
        assert_eq!(by_key, ids, "encoded order must equal Uuid Ord");
    }

    #[test]
    fn edge_key_roundtrip_and_flip() {
        let e = Edge::new(uuid(7), ident("knows"), uuid(9));
        let k = edge_key(&e);
        assert!(k.starts_with(&edge_prefix()));
        assert_eq!(decode_edge_key(&k), Some(e.clone()));

        // Reverse index stores the flipped orientation.
        let rk = rev_edge_key(&e);
        assert!(rk.starts_with(&rev_edge_prefix()));
        let stored = decode_rev_edge_key(&rk).expect("decode reversed");
        assert_eq!(stored.outbound_id, e.inbound_id);
        assert_eq!(stored.inbound_id, e.outbound_id);
        assert_eq!(flip(&stored), e, "flip recovers the real edge");
    }

    /// Load-bearing: edge key order == `Edge` order — including the type
    /// prefix case (`"ab"` vs `"b"`) that a length prefix would get wrong.
    #[test]
    fn edge_key_order_matches_edge_order() {
        let mut edges = vec![
            Edge::new(uuid(1), ident("b"), uuid(1)),
            Edge::new(uuid(1), ident("ab"), uuid(2)),
            Edge::new(uuid(1), ident("a"), uuid(3)),
            Edge::new(uuid(0), ident("z"), uuid(0)),
            Edge::new(uuid(1), ident("a"), uuid(1)),
            Edge::new(uuid(2), ident("a"), uuid(0)),
        ];
        let mut by_key = edges.clone();
        by_key.sort_by_key(edge_key);
        edges.sort();
        assert_eq!(by_key, edges, "encoded order must equal Edge Ord");

        // Spot-check the prefix case directly.
        let a = Edge::new(uuid(1), ident("a"), uuid(9));
        let ab = Edge::new(uuid(1), ident("ab"), uuid(0));
        assert!(edge_key(&a) < edge_key(&ab), "'a' sorts before 'ab'");
    }

    /// Reverse keys order by (inbound, type, outbound) — the ordering
    /// `range_reversed_edges` walks.
    #[test]
    fn rev_edge_key_orders_by_inbound_first() {
        let e1 = Edge::new(uuid(9), ident("t"), uuid(1)); // inbound 1
        let e2 = Edge::new(uuid(0), ident("t"), uuid(2)); // inbound 2
        assert!(
            rev_edge_key(&e1) < rev_edge_key(&e2),
            "reverse index sorts by inbound id first"
        );
        // ...whereas the forward index sorts the other way for this pair.
        assert!(edge_key(&e2) < edge_key(&e1));
    }

    #[test]
    fn vertex_prop_key_roundtrip_and_prefix_scoping() {
        let v1 = uuid(1);
        let v2 = uuid(2);
        let name = ident("size");
        let k = vertex_prop_key(v1, &name);
        assert_eq!(decode_vertex_prop_key(&k), Some((v1, name.clone())));

        // One vertex's prefix does not match another's properties.
        assert!(k.starts_with(&vertex_props_prefix(v1)));
        assert!(!k.starts_with(&vertex_props_prefix(v2)));

        // Property names that share a prefix stay distinct.
        let short = vertex_prop_key(v1, &ident("s"));
        let long = vertex_prop_key(v1, &ident("size"));
        assert_ne!(short, long);
        assert!(short < long, "terminator keeps 's' before 'size'");
    }

    #[test]
    fn edge_prop_key_roundtrip_and_prefix_scoping() {
        let e = Edge::new(uuid(3), ident("rel"), uuid(4));
        let other = Edge::new(uuid(3), ident("rel"), uuid(5));
        let name = ident("weight");
        let k = edge_prop_key(&e, &name);
        assert_eq!(decode_edge_prop_key(&k), Some((e.clone(), name)));
        assert!(k.starts_with(&edge_props_prefix(&e)));
        assert!(!k.starts_with(&edge_props_prefix(&other)));
    }

    #[test]
    fn indexed_key_roundtrip() {
        let name = ident("indexed_prop");
        let k = indexed_key(&name);
        assert!(k.starts_with(&indexed_prefix()));
        assert_eq!(decode_indexed_key(&k), Some(name));
    }

    /// Namespaces are disjoint: no key from one decodes as another, and the
    /// tags keep the whole space partitioned.
    #[test]
    fn namespaces_are_disjoint() {
        let id = uuid(5);
        let e = Edge::new(uuid(5), ident("t"), uuid(6));
        let name = ident("p");
        let keys = vec![
            vertex_key(id),
            edge_key(&e),
            rev_edge_key(&e),
            vertex_prop_key(id, &name),
            edge_prop_key(&e, &name),
            indexed_key(&name),
        ];
        // Distinct leading tags.
        let mut tags: Vec<u8> = keys.iter().map(|k| k[0]).collect();
        tags.sort_unstable();
        tags.dedup();
        assert_eq!(tags.len(), keys.len(), "one namespace tag per key kind");

        // Each decoder accepts only its own namespace.
        assert!(decode_vertex_key(&keys[1]).is_none());
        assert!(decode_edge_key(&keys[0]).is_none());
        assert!(decode_rev_edge_key(&keys[1]).is_none());
        assert!(decode_vertex_prop_key(&keys[4]).is_none());
        assert!(decode_edge_prop_key(&keys[3]).is_none());
        assert!(decode_indexed_key(&keys[0]).is_none());
    }
}
