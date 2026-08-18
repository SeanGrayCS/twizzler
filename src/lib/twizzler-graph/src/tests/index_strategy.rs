//! Pluggable index strategy: per-label opt-in, the volatile lazily-built
//! default, rebuild sources, scan policy, and schema-bits round-tripping.
//!
//! `find_vertex` returns `Lookup` because there are two distinct negative
//! answers — "no such vertex" and "this label is not indexed, I did not
//! look" — and `Option` cannot carry the difference. Unindexed lookups default
//! to `Refuse`, because a per-query full scan is invisible at the call site;
//! `Scan` allows them, and scans are counted.

use twizzler::object::ObjID;

use super::super::*;

/// Fresh graph under a given strategy. Each test uses its own name so a stale
/// `target/disk-*.img` cannot make one test observe another's records.
fn fresh(name: &str, strategy: IndexStrategy) -> Graph {
    fresh_with(name, IndexSchema::new(strategy))
}

fn fresh_with(name: &str, schema: IndexSchema) -> Graph {
    Graph::reset_arena_with_index(name, DEFAULT_ARENA_CAP, schema).expect("reset");
    Graph::open_or_create_arena_with_index(name, DEFAULT_ARENA_CAP, schema).expect("open")
}

/// The strategy is a property of the graph, not of the call that opened it: a
/// reopen with no strategy argument sees the stored one.
#[test]
fn strategy_persists_across_reopen() {
    let name = "t-a8-persist";
    {
        let g = fresh(name, IndexStrategy::None);
        assert_eq!(g.index_strategy(), IndexStrategy::None);
    }
    let g = Graph::open_or_create_arena(name, DEFAULT_ARENA_CAP).expect("reopen");
    assert_eq!(
        g.index_strategy(),
        IndexStrategy::None,
        "reopen must honour the stored strategy, not the build default"
    );
}

/// Per-label opt-in: an undeclared label produces no index entries.
#[test]
fn only_declared_labels_are_indexed() {
    let mut g = fresh("t-a8-optin", IndexStrategy::LazyLabel);
    g.set_label_indexed("person", true).expect("declare");

    let p = g.add_vertex("person", "alice", ObjID::new(0)).expect("add");
    g.add_vertex("comment", "c1", ObjID::new(0)).expect("add");

    assert!(g.is_label_indexed("person"));
    assert!(!g.is_label_indexed("comment"));
    assert_eq!(g.find_vertex("person", "alice"), Lookup::Found(p));
    assert_eq!(g.indexed_entry_count(), 1, "only the declared label is indexed");
}

/// The default index is volatile: it owns no object, so there is nothing to
/// sync and nothing to reclaim.
#[test]
fn default_index_owns_no_object() {
    let mut g = fresh("t-a8-volatile", IndexStrategy::LazyLabel);
    g.set_label_indexed("n", true).expect("declare");
    for i in 0..64 {
        g.add_vertex("n", &format!("v{i}"), ObjID::new(0))
            .expect("add");
    }
    let _ = g.find_vertex("n", "v0");
    assert_eq!(
        g.index_object_ids(),
        Vec::<u128>::new(),
        "a volatile index must not appear in the owned-object inventory"
    );
}

/// The index is built on the first lookup, not on insert, and later lookups
/// reuse it. A load that never looks up never pays the build.
#[test]
fn index_is_not_built_until_first_lookup() {
    let mut g = fresh("t-a8-lazy", IndexStrategy::LazyLabel);
    g.set_label_indexed("n", true).expect("declare");
    for i in 0..32 {
        g.add_vertex("n", &format!("v{i}"), ObjID::new(0))
            .expect("add");
    }
    assert_eq!(g.index_builds(), 0, "inserting must not build the index");
    let _ = g.find_vertex("n", "v3");
    assert_eq!(g.index_builds(), 1, "first lookup builds it");
    let _ = g.find_vertex("n", "v4");
    assert_eq!(g.index_builds(), 1, "subsequent lookups reuse it");
}

/// Lookup parity across a reopen: the rebuild reconstructs the index from
/// records alone, so records carry enough to regenerate it.
#[test]
fn lookup_parity_after_reopen() {
    let name = "t-a8-parity";
    let (a, b) = {
        let mut g = fresh(name, IndexStrategy::LazyLabel);
        g.set_label_indexed("n", true).expect("declare");
        let a = g.add_vertex("n", "alpha", ObjID::new(0)).expect("add");
        let b = g.add_vertex("n", "beta", ObjID::new(0)).expect("add");
        g.sync().expect("sync");
        (a, b)
    };
    let g = Graph::open_or_create_arena(name, DEFAULT_ARENA_CAP).expect("reopen");
    assert_eq!(g.find_vertex("n", "alpha"), Lookup::Found(a));
    assert_eq!(g.find_vertex("n", "beta"), Lookup::Found(b));
    assert_eq!(
        g.find_vertex("n", "absent"),
        Lookup::NotFound,
        "an indexed label gives an authoritative negative, not NotIndexed"
    );
}

/// A rebuild respects tombstones: the `locs` mirror is authoritative for
/// liveness, so deleted names do not come back after a reopen.
#[test]
fn rebuild_does_not_resurrect_deleted_vertices() {
    let name = "t-a8-tombstone";
    {
        let mut g = fresh(name, IndexStrategy::LazyLabel);
        g.set_label_indexed("n", true).expect("declare");
        let doomed = g.add_vertex("n", "doomed", ObjID::new(0)).expect("add");
        g.add_vertex("n", "kept", ObjID::new(0)).expect("add");
        g.delete_vertex(doomed).expect("delete");
        g.sync().expect("sync");
    }
    let g = Graph::open_or_create_arena(name, DEFAULT_ARENA_CAP).expect("reopen");
    assert_eq!(g.find_vertex("n", "doomed"), Lookup::NotFound);
    assert!(g.find_vertex("n", "kept").found().is_some());
}

/// An unindexed lookup answers `NotIndexed`, not `NotFound`: the vertex can
/// exist without the index having looked. Under `Refuse`, nothing scans.
#[test]
fn unindexed_lookup_reports_not_indexed_not_not_found() {
    let mut g = fresh("t-a8-loud", IndexStrategy::LazyLabel);
    g.add_vertex("comment", "c1", ObjID::new(0)).expect("add");
    assert_eq!(
        g.find_vertex("comment", "c1"),
        Lookup::NotIndexed,
        "the vertex exists; reporting NotFound here would be false"
    );
    assert_eq!(g.scans_performed(), 0, "Refuse must not scan");
}

/// The roots list holds only indexed records, not every record in the graph.
#[test]
fn roots_list_covers_only_indexed_records() {
    let mut g = fresh_with(
        "t-a8-roots-scope",
        IndexSchema::new(IndexStrategy::LazyLabel).rebuild(RebuildSource::Roots),
    );
    g.set_label_indexed("person", true).expect("declare");
    for i in 0..8 {
        g.add_vertex("person", &format!("p{i}"), ObjID::new(0))
            .expect("add");
        // Ten unindexed records per indexed one.
        for j in 0..10 {
            g.add_vertex("comment", &format!("c{i}_{j}"), ObjID::new(0))
                .expect("add");
        }
    }
    assert_eq!(
        g.indexed_root_count(),
        8,
        "the roots list must track indexed records only, not all 88"
    );
}

/// A deleted root does not come back: the roots list is append-only, so
/// liveness is checked against the `locs` mirror at rebuild.
#[test]
fn roots_rebuild_skips_deleted_roots() {
    let name = "t-a8-roots-dead";
    {
        let mut g = fresh_with(
            name,
            IndexSchema::new(IndexStrategy::LazyLabel).rebuild(RebuildSource::Roots),
        );
        g.set_label_indexed("n", true).expect("declare");
        let doomed = g.add_vertex("n", "doomed", ObjID::new(0)).expect("add");
        g.add_vertex("n", "kept", ObjID::new(0)).expect("add");
        g.delete_vertex(doomed).expect("delete");
        g.sync().expect("sync");
    }
    let g = Graph::open_or_create_arena(name, DEFAULT_ARENA_CAP).expect("reopen");
    assert_eq!(g.find_vertex("n", "doomed"), Lookup::NotFound);
    assert!(g.find_vertex("n", "kept").found().is_some());
}

/// Scanning is a policy, not a prohibition: under `Scan` an unindexed lookup
/// is answered authoritatively, positive or negative.
#[test]
fn scan_policy_answers_unindexed_lookups() {
    let mut g = fresh_with(
        "t-a8-scan",
        IndexSchema::new(IndexStrategy::LazyLabel).unindexed(UnindexedLookup::Scan),
    );
    let c = g.add_vertex("comment", "c1", ObjID::new(0)).expect("add");
    assert_eq!(g.find_vertex("comment", "c1"), Lookup::Found(c));
    assert_eq!(
        g.find_vertex("comment", "absent"),
        Lookup::NotFound,
        "a scan gives an authoritative negative — never NotIndexed"
    );
}

/// Scans are counted: their cost is invisible at the call site, so a caller
/// can check names were resolved from the index rather than by a walk.
#[test]
fn scans_are_counted() {
    let mut g = fresh_with(
        "t-a8-count",
        IndexSchema::new(IndexStrategy::LazyLabel).unindexed(UnindexedLookup::Scan),
    );
    g.set_label_indexed("person", true).expect("declare");
    g.add_vertex("person", "alice", ObjID::new(0)).expect("add");
    g.add_vertex("comment", "c1", ObjID::new(0)).expect("add");

    let _ = g.find_vertex("person", "alice");
    assert_eq!(g.scans_performed(), 0, "an indexed label must not scan");
    let _ = g.find_vertex("comment", "c1");
    assert_eq!(g.scans_performed(), 1, "an unindexed label under Scan does");
}

/// The rebuild source is injectable, and `Roots` reaches the same answers as
/// `Scan` while reading far less. Only parity is asserted here.
#[test]
fn roots_rebuild_matches_scan_rebuild() {
    for (name, source) in [
        ("t-a8-rb-scan", RebuildSource::Scan),
        ("t-a8-rb-roots", RebuildSource::Roots),
    ] {
        let a = {
            let mut g = fresh_with(
                name,
                IndexSchema::new(IndexStrategy::LazyLabel).rebuild(source),
            );
            g.set_label_indexed("n", true).expect("declare");
            let a = g.add_vertex("n", "alpha", ObjID::new(0)).expect("add");
            g.add_vertex("skip", "s1", ObjID::new(0)).expect("add");
            g.sync().expect("sync");
            a
        };
        let g = Graph::open_or_create_arena(name, DEFAULT_ARENA_CAP).expect("reopen");
        assert_eq!(g.find_vertex("n", "alpha"), Lookup::Found(a), "{source:?}");
        assert_eq!(g.find_vertex("n", "absent"), Lookup::NotFound, "{source:?}");
    }
}

/// The persistent strategy is retained as the comparison arm, and it is the
/// only strategy that owns an index object.
#[test]
fn persistent_strategy_still_owns_its_object() {
    let mut g = fresh("t-a8-compat", IndexStrategy::Persistent);
    g.set_label_indexed("n", true).expect("declare");
    g.add_vertex("n", "v0", ObjID::new(0)).expect("add");
    assert_eq!(
        g.index_object_ids().len(),
        1,
        "the persistent arm must still own exactly one index object"
    );
    assert!(g.find_vertex("n", "v0").found().is_some());
}

/// Declaring a label after its records are inserted works under `Roots`:
/// `set_label_indexed` backfills the roots list, and the backfill is durable.
#[test]
fn declare_after_insert_is_backfilled_under_roots() {
    let name = "t-a8-backfill";
    let (a, b) = {
        let mut g = fresh_with(
            name,
            IndexSchema::new(IndexStrategy::LazyLabel).rebuild(RebuildSource::Roots),
        );
        // Insert first — no RootEntry is written for either record.
        let a = g.add_vertex("late", "alpha", ObjID::new(0)).expect("add");
        let b = g.add_vertex("late", "beta", ObjID::new(0)).expect("add");
        // Declare after. The backfill walks records once and roots both.
        g.set_label_indexed("late", true).expect("declare");
        assert_eq!(
            g.find_vertex("late", "alpha"),
            Lookup::Found(a),
            "pre-declaration record invisible to the Roots rebuild"
        );
        // A post-declaration insert continues through index_on_insert.
        let c = g.add_vertex("late", "gamma", ObjID::new(0)).expect("add");
        assert_eq!(g.find_vertex("late", "gamma"), Lookup::Found(c));
        g.sync().expect("sync");
        (a, b)
    };
    // The backfilled entries are durable: a reopened handle rebuilds from the
    // persisted list alone.
    let g = Graph::open_or_create_arena(name, DEFAULT_ARENA_CAP).expect("reopen");
    assert_eq!(g.find_vertex("late", "alpha"), Lookup::Found(a));
    assert_eq!(g.find_vertex("late", "beta"), Lookup::Found(b));
    assert_eq!(g.find_vertex("late", "absent"), Lookup::NotFound);
}

/// `from_bits` refuses bits it does not understand, the reserved top byte
/// included, so a future build's schema fails loudly rather than being misread.
#[test]
fn unknown_index_schema_bits_are_refused() {
    // Every current schema round-trips.
    for strategy in [
        IndexStrategy::None,
        IndexStrategy::LazyLabel,
        IndexStrategy::Persistent,
    ] {
        for unindexed in [UnindexedLookup::Refuse, UnindexedLookup::Scan] {
            for rebuild in [RebuildSource::Scan, RebuildSource::Roots] {
                let s = IndexSchema::new(strategy).unindexed(unindexed).rebuild(rebuild);
                assert_eq!(
                    IndexSchema::from_bits(s.to_bits()),
                    Some(s),
                    "{strategy:?}/{unindexed:?}/{rebuild:?} must round-trip"
                );
            }
        }
    }
    // Unknown values in each decoded byte are refused, not defaulted.
    assert_eq!(IndexSchema::from_bits(3), None, "unknown strategy");
    assert_eq!(IndexSchema::from_bits(2 << 8), None, "unknown unindexed policy");
    assert_eq!(IndexSchema::from_bits(2 << 16), None, "unknown rebuild source");
    // And the extension byte: a future build's schema must fail loudly here,
    // not open as a misread of its low bytes.
    assert_eq!(IndexSchema::from_bits(1 << 24), None, "extension byte");
    assert_eq!(
        IndexSchema::from_bits((1 << 24) | 1),
        None,
        "a valid low encoding does not excuse unknown high bits"
    );
}

/// Deleting one of two vertices sharing `(label, name)` does not un-index the
/// other: index-entry removal matches on id, not on key alone.
#[test]
fn deleting_one_twin_keeps_the_other_findable() {
    let mut g = fresh("t-a8-twins", IndexStrategy::LazyLabel);
    g.set_label_indexed("n", true).expect("declare");
    let first = g.add_vertex("n", "dup", ObjID::new(0)).expect("add");
    let second = g.add_vertex("n", "dup", ObjID::new(0)).expect("add");

    // Build the map, and note which twin it resolves — the insert path's
    // last-writer-wins rule makes that the second.
    assert_eq!(g.find_vertex("n", "dup"), Lookup::Found(second));

    // Deleting the twin the map does NOT hold must leave the entry alone:
    // this is the id match doing its work on a built map.
    g.delete_vertex(first).expect("delete first twin");
    assert_eq!(
        g.find_vertex("n", "dup"),
        Lookup::Found(second),
        "deleting the unmapped twin un-indexed the survivor"
    );

    // Deleting the mapped twin removes the entry; with both twins gone the
    // name is authoritatively absent.
    g.delete_vertex(second).expect("delete second twin");
    assert_eq!(g.find_vertex("n", "dup"), Lookup::NotFound);
}
