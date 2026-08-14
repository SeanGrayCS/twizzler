//! Engine integration tests, run on Twizzler via `cargo start-qemu --tests`.
//!
//! Organized by area in `src/tests/`: `crud` (create/read/delete), `adjacency`
//! (edges, neighbors, filters), `dsl` (traversal steps), `names` (NameKey and
//! label/name edge cases), `persistence` (reopen, reset, format guard), `arena`
//! and `arena_graph` (the arena storage layout), and `reclaim` (objects are
//! freed on reset and delete rather than orphaned). Module-local unit tests
//! (e.g. in `segvec.rs`) stay with their modules.
//!
//! The tests in one run share one boot's memory, and an object holding any
//! `InvPtr` costs far more than a plain one, so a test should create the
//! fewest entities its assertion needs.
//!
//! Each test creates a persistent graph registered under `data/<name>`. To
//! stay idempotent across runs every test resets its graph first (via the
//! helpers below) and graph names are unique per test.

use crate::Graph;

// Sliced by feature so a run can fit in one boot; see `Cargo.toml`. A slice
// is a memory partition, not a logical one — the groups are sized by what
// they cost, so a boot can finish. `test-all` runs everything at once.
#[cfg(feature = "test-core")]
mod adjacency;
#[cfg(feature = "test-storage")]
mod arena;
#[cfg(feature = "test-storage")]
mod arena_graph;
#[cfg(feature = "test-core")]
mod crud;
#[cfg(feature = "test-query")]
mod dsl;
#[cfg(feature = "test-storage")]
mod index_strategy;
#[cfg(feature = "test-core")]
mod names;
#[cfg(feature = "test-query")]
mod ordering;
#[cfg(feature = "test-core")]
mod persistence;
#[cfg(feature = "test-query")]
mod props;
#[cfg(feature = "test-query")]
mod repeat;
#[cfg(feature = "test-storage")]
mod text_blob;
#[cfg(feature = "test-storage")]
mod reclaim;

/// Vertices per arena for the general suite. Large on purpose: an arena
/// holding a cross-arena reference costs far more than a plain one, and a
/// small cap multiplies arenas, so the goal is to keep each whole test graph
/// in one arena and let the suite share one boot's frame budget. Cross-arena
/// links are exercised deliberately by `tests/arena.rs` and
/// `tests/arena_graph.rs`, which set their own small caps.
pub(crate) const TEST_ARENA_CAP: usize = 64;

/// Clear any existing graph of this name, then open a clean one with the
/// suite's indexed labels declared.
// Each slice uses only some of these helpers; the unused ones are not dead.
#[allow(dead_code)]
pub(crate) fn fresh(name: &str) -> Graph {
    Graph::reset_arena(name, TEST_ARENA_CAP).expect("reset arena graph");
    let mut g =
        Graph::open_or_create_arena(name, TEST_ARENA_CAP).expect("create arena graph");
    declare_test_labels(&mut g);
    g
}

/// Labels the suite resolves by name. The default schema indexes nothing
/// until asked, so a test that looks up by name gets `NotIndexed` unless its
/// label is declared.
///
/// Declared here rather than switching test graphs to `UnindexedLookup::Scan`:
/// scanning would answer every lookup, but it would stop the tests exercising
/// the volatile index at all. Declaring also makes `names.rs` check that a
/// rebuild handles truncated and multibyte names.
///
/// The empty label is deliberate — `names.rs` uses it.
pub(crate) const TEST_INDEXED_LABELS: &[&str] =
    &["", "n", "m", "c", "d", "d2", "tag", "file", "hub", "spoke"];

pub(crate) fn declare_test_labels(g: &mut Graph) {
    for l in TEST_INDEXED_LABELS {
        g.set_label_indexed(l, true).expect("declare test label");
    }
}
