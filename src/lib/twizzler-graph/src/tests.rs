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
// Each slice uses only some of these helpers; the unused ones are not dead.
#[allow(dead_code)]
pub(crate) fn fresh(name: &str) -> Graph {
    Graph::reset_arena(name, TEST_ARENA_CAP).expect("reset arena graph");
    Graph::open_or_create_arena(name, TEST_ARENA_CAP).expect("create arena graph")
}
