//! Engine integration tests, run on Twizzler via `cargo start-qemu --tests`.
//!
//! Each test creates a persistent graph registered under `data/<name>`. To
//! stay idempotent across runs every test resets its graph first (via the
//! helpers below) and graph names are unique per test.

use crate::Graph;

#[cfg(feature = "test-core")]
mod adjacency;
#[cfg(feature = "test-storage")]
mod arena;
#[cfg(feature = "test-storage")]
mod arena_graph;
#[cfg(feature = "test-storage")]
mod bulk;
#[cfg(feature = "test-core")]
mod crud;
#[cfg(feature = "test-query")]
mod dsl;
#[cfg(feature = "test-core")]
mod names;
#[cfg(feature = "test-core")]
mod persistence;
#[cfg(feature = "test-query")]
mod props;
#[cfg(feature = "test-storage")]
mod reclaim;
#[cfg(feature = "test-storage")]
mod segmentation;

/// Vertices per arena for the general suite. Large on purpose.
///
/// It also bought coverage the general suite does not need: cross-arena links
/// are exercised deliberately, and cheaply, by `tests/arena.rs` and
/// `tests/arena_graph.rs`, which set their own small caps over a handful of
/// vertices. Here the goal is the opposite — keep whole test graphs in one
/// arena so ~90 tests can share one boot's frame budget.
pub(crate) const TEST_ARENA_CAP: usize = 64;

/// Clear any existing graph of this name, then open a clean one on the arena
/// layout (the current format).
///
/// Use [`fresh_v3`] only where the *old layout itself* is under test.
// Each slice uses only some of these helpers; the unused ones are not dead.
#[allow(dead_code)]
pub(crate) fn fresh(name: &str) -> Graph {
    Graph::reset_arena(name, TEST_ARENA_CAP).expect("reset arena graph");
    Graph::open_or_create_arena(name, TEST_ARENA_CAP).expect("create arena graph")
}

/// A clean graph on the legacy v3 layout, for tests of v3 internals that
/// have no v4 equivalent: registry segmentation (`segmentation`), per-vertex object
/// inventory (`reclaim`), and `BulkSession` (`bulk`), whose batching the arena
/// store does internally instead.
///
/// Equivalence between the layouts is covered by `tests/arena_graph.rs`, which
/// runs one workload through both and compares — not by having the whole suite
/// sit on the old format.
#[allow(dead_code)]
pub(crate) fn fresh_v3(name: &str) -> Graph {
    Graph::reset(name).expect("reset graph");
    Graph::open_or_create(name).expect("create graph")
}

/// Like `fresh`, but with a forced registry segment capacity, so segmentation
/// tests can trigger rollover with a handful of inserts. v3 — registry
/// segmentation is a v3 concern; v4 segments the location registry instead.
#[allow(dead_code)]
pub(crate) fn fresh_cap(name: &str, cap: usize) -> Graph {
    Graph::reset_with_capacity(name, cap).expect("reset graph");
    Graph::open_or_create_with_capacity(name, cap).expect("create graph")
}
