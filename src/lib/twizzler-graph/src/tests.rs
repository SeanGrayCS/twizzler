//! Engine integration tests, run on Twizzler via `cargo start-qemu --tests`.
//!
//! Organized by area in `src/tests/`: `crud` (create/read/delete), `adjacency`
//! (edges, neighbors, filters), `dsl` (traversal steps), `names` (NameKey and
//! label/name edge cases), `persistence` (reopen, reset, format guard), and
//! `sharding` (segmented registries). Module-local unit tests (e.g. in
//! `segvec.rs`) stay with their modules.
//!
//! Each test creates a persistent graph registered under `data/<name>`. To
//! stay idempotent across runs every test resets its graph first (via the
//! helpers below) and graph names are unique per test.

use crate::Graph;

mod adjacency;
mod crud;
mod dsl;
mod names;
mod persistence;
mod sharding;

/// Clear any existing graph of this name, then open a clean one.
pub(crate) fn fresh(name: &str) -> Graph {
    Graph::reset(name).expect("reset graph");
    Graph::open_or_create(name).expect("create graph")
}

/// Like `fresh`, but with a forced registry segment capacity, so sharding
/// tests can trigger rollover with a handful of inserts.
pub(crate) fn fresh_cap(name: &str, cap: usize) -> Graph {
    Graph::reset_with_capacity(name, cap).expect("reset graph");
    Graph::open_or_create_with_capacity(name, cap).expect("create graph")
}
