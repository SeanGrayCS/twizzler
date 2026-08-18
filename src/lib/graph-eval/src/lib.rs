//! LDBC-SNB interactive short reads on both engines (board task E3 — the M3
//! milestone).
//!
//! M3 is "the same query set runs on the native engine and on
//! IndraDB-on-Twizzler". This crate is where that claim is made concrete and
//! checkable:
//!
//! - [`fixture`] defines one small social graph, engine-agnostically;
//! - [`results`] defines canonical result types phrased in **names**, never
//!   engine ids, so two engines' answers are directly comparable;
//! - [`native`] implements IS1–IS7 over `twizzler-graph`;
//! - [`baseline`] implements the same seven over
//!   `Database<TwizzlerDatastore>`, using IndraDB's public API only;
//! - `tests_equivalence` asserts the two engines return **identical** results
//!   for every query across the whole fixture — that assertion *is* M3.
//!
//! The queries live here rather than inside either engine so that neither can
//! quietly special-case them — the comparison is only meaningful if both are
//! driven through their public APIs.
//!
//! **Where the DSL falls short** (E3-AC4; updated 2026-08-18 — this note
//! described two gaps for a week after one of them closed):
//!
//! - **IS3** orders friends by a property of the *edge* (`knows.since`) and
//!   returns it alongside the friend. The DSL orders vertices by vertex
//!   properties only — wanted: `EdgeTraversal::order_by_prop{,_desc}` and an
//!   edge→endpoint step that keeps the edge's properties. **Still open**
//!   (board task B4); the fallback in [`native`] keeps its `DSL GAP` marker.
//! - **IS6**'s gap (an unbounded `replyOf` walk in a fixed-depth DSL)
//!   **closed when B3 landed** (2026-08-11): the native IS6 is a
//!   `repeat_out(replyOf).until_exhausted()` with `hit_depth_cap()` checked,
//!   and its `DSL GAP` marker is gone. Kept here as evidence of what a
//!   Gremlin-subset DSL actually needs — the gap list drove B3's design.

pub mod baseline;
pub mod fixture;
pub mod native;
pub mod results;

#[cfg(all(test, feature = "tests"))]
mod tests_equivalence;
#[cfg(all(test, feature = "tests"))]
mod tests_native;
