//! Context: `gstress index 2000000 bulk` completed, so SF0.1 is viable, but 91%
//! of its 514 s was writeback and the single largest object in the write path
//! was the index — 196 MB for a structure that is write-only in this build.
//! `find_vertex` is its only reader and every caller in the tree is a test;
//! `add_vertex` never consults it, so it is not enforcing uniqueness either.
//!
//! The fix is not "make the index smaller". Indexing every record is the
//! KV-store assumption — IndraDB needs it because every traversal step is a
//! keyed lookup. Index-free adjacency only needs lookup at *query entry*, so the
//! index should cover roots, not records.

use twizzler::object::ObjID;

use super::super::*;

fn fresh(name: &str, strategy: IndexStrategy) -> Graph {
    fresh_with(name, IndexSchema::new(strategy))
}

fn fresh_with(name: &str, schema: IndexSchema) -> Graph {
    Graph::reset_arena_with_index(name, DEFAULT_ARENA_CAP, schema).expect("reset");
    Graph::open_or_create_arena_with_index(name, DEFAULT_ARENA_CAP, schema).expect("open")
}

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
        // Ten unindexed records per indexed one: the ratio that makes the
        // difference at SF0.1 (~1.5 k Person among ~2 M records).
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
