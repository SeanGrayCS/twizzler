//! Ordering steps: their cost, asserted as a property-read count, and their
//! comparator semantics (missing keys last, ties broken by ascending id).

use twizzler::object::ObjID;

use super::fresh;
use crate::{Graph, Labels, PropValue, VertexId};

/// Enough elements that `n` and `2n·log₂n` read counts cannot be confused,
/// small enough for the shared boot's frame budget (33 vertices + 32 edges).
const N: usize = 32;

/// A hub with `N` spokes pointing at it, each carrying property `d`.
///
/// `d` counts down as the id counts up, so neither id order nor insertion
/// order matches the value order; a sort that ignores the property shows.
fn hub_and_spokes(name: &str) -> (Graph, VertexId, Vec<VertexId>) {
    let mut g = fresh(name);
    let hub = g.add_vertex("hub", "h", ObjID::new(0)).unwrap();
    let mut spokes = Vec::new();
    for i in 0..N {
        let s = g.add_vertex("spoke", &format!("s{i}"), ObjID::new(0)).unwrap();
        // Zero-padded so lexicographic order is numeric order.
        g.set_vertex_prop(s, "d", PropValue::str(&format!("{:04}", N - i)))
            .unwrap();
        g.add_edge(s, "e", hub).unwrap();
        spokes.push(s);
    }
    (g, hub, spokes)
}

/// Ordering by a property reads that property once per candidate.
#[test]
fn order_by_prop_reads_each_candidate_once() {
    let (g, hub, _) = hub_and_spokes("t-ord-cost");

    g.reset_prop_reads();
    let ids = g
        .traversal()
        .v(hub)
        .in_(Labels::these(&["e"]))
        .order_by_prop_desc("d")
        .limit(10)
        .to_ids();
    let reads = g.prop_reads();

    assert_eq!(ids.len(), 10, "limit(10) over {N} candidates");
    // Exact equality: the hop and `limit` perform no property reads, so the
    // ordering step's budget is the whole count.
    assert_eq!(
        reads, N,
        "ordering {N} candidates must cost {N} property reads, not {reads} \
         (~2n·log2(n) = {} is the pre-fix figure)",
        2 * N * 5
    );
}

/// The ascending step also reads each candidate's property exactly once.
#[test]
fn order_by_prop_asc_reads_each_candidate_once() {
    let (g, hub, _) = hub_and_spokes("t-ord-cost-asc");

    g.reset_prop_reads();
    let n = g
        .traversal()
        .v(hub)
        .in_(Labels::these(&["e"]))
        .order_by_prop("d")
        .count();
    assert_eq!(n, N);
    assert_eq!(g.prop_reads(), N);
}

/// `order_by_name` sorts on the record head, already in hand from the walk,
/// so it costs no property reads.
#[test]
fn order_by_name_reads_no_properties() {
    let (g, hub, _) = hub_and_spokes("t-ord-name");

    g.reset_prop_reads();
    let n = g
        .traversal()
        .v(hub)
        .in_(Labels::these(&["e"]))
        .order_by_name_desc()
        .count();
    assert_eq!(n, N);
    assert_eq!(g.prop_reads(), 0);
}

/// Descending by property then `limit(k)` yields the `k` largest values in
/// descending order.
#[test]
fn order_by_prop_desc_then_limit_is_top_k() {
    let (g, hub, _) = hub_and_spokes("t-ord-topk");

    let got: Vec<PropValue> = g
        .traversal()
        .v(hub)
        .in_(Labels::these(&["e"]))
        .order_by_prop_desc("d")
        .limit(5)
        .values("d");

    // `d` runs from 0001 (last spoke) to 0032 (first), so the top five are the
    // five earliest-inserted spokes, in descending value order.
    let want: Vec<PropValue> = (0..5)
        .map(|i| PropValue::str(&format!("{:04}", N - i)))
        .collect();
    assert_eq!(got, want);
}

/// Vertices lacking the key sort last in both directions, and ties break by
/// ascending id in both, so descending is not the reverse of ascending.
#[test]
fn missing_keys_sort_last_in_both_directions() {
    let mut g = fresh("t-ord-missing");
    let hub = g.add_vertex("hub", "h", ObjID::new(0)).unwrap();

    // Two vertices share a value, one has none. Insertion order is id order.
    let a = g.add_vertex("spoke", "a", ObjID::new(0)).unwrap();
    let b = g.add_vertex("spoke", "b", ObjID::new(0)).unwrap();
    let c = g.add_vertex("spoke", "c", ObjID::new(0)).unwrap();
    g.set_vertex_prop(a, "d", PropValue::str("0002")).unwrap();
    g.set_vertex_prop(b, "d", PropValue::str("0001")).unwrap();
    g.set_vertex_prop(c, "d", PropValue::str("0002")).unwrap();
    for s in [a, b, c] {
        g.add_edge(s, "e", hub).unwrap();
    }

    let desc = g
        .traversal()
        .v(hub)
        .in_(Labels::these(&["e"]))
        .order_by_prop_desc("d")
        .to_ids();
    assert_eq!(desc, vec![a, c, b], "0002 (a before c, id asc), then 0001");

    let asc = g
        .traversal()
        .v(hub)
        .in_(Labels::these(&["e"]))
        .order_by_prop("d")
        .to_ids();
    assert_eq!(asc, vec![b, a, c], "0001, then 0002 (a before c, id asc)");

    // The unset key: last in both directions, not first in one of them.
    let d = g.add_vertex("spoke", "d", ObjID::new(0)).unwrap();
    g.add_edge(d, "e", hub).unwrap();
    for order in [true, false] {
        let t = g.traversal().v(hub).in_(Labels::these(&["e"]));
        let ids = if order {
            t.order_by_prop_desc("d").to_ids()
        } else {
            t.order_by_prop("d").to_ids()
        };
        assert_eq!(*ids.last().unwrap(), d, "missing key sorts last (desc={order})");
    }
}

/// Long values are not orderable: `get_vertex_prop` filters `TextRef` and
/// `BlobRef` out, so a value promoted past 31 bytes sorts as missing — last.
#[test]
fn long_values_are_not_orderable() {
    let mut g = fresh("t-ord-text");
    let hub = g.add_vertex("hub", "h", ObjID::new(0)).unwrap();
    let short = g.add_vertex("spoke", "s", ObjID::new(0)).unwrap();
    let long = g.add_vertex("spoke", "l", ObjID::new(0)).unwrap();
    g.set_vertex_prop(short, "d", PropValue::str("0001")).unwrap();
    // 40 bytes: past the 31-byte `PropValue::Str` tier, so it lands in the
    // byte store. If it were visible it would sort first in descending order.
    g.set_vertex_text(long, "d", &"z".repeat(40)).unwrap();
    g.add_edge(short, "e", hub).unwrap();
    g.add_edge(long, "e", hub).unwrap();

    let ids = g
        .traversal()
        .v(hub)
        .in_(Labels::these(&["e"]))
        .order_by_prop_desc("d")
        .to_ids();
    assert_eq!(ids, vec![short, long], "the long value sorts as missing");
}
