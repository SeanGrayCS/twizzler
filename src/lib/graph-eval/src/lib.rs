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
//! - the baseline implementation over `TwizzlerDatastore` and the
//!   cross-engine equivalence assertions follow in slice E3b.
//!
//! The queries live here rather than inside either engine so that neither can
//! quietly special-case them — the comparison is only meaningful if both are
//! driven through their public APIs.
//!
//! **Where the DSL falls short.** Two of the seven reads cannot be expressed
//! in the traversal DSL today, and both fallbacks are marked `DSL GAP` in
//! [`native`] with the missing step named (E3-AC4):
//!
//! - **IS3** orders friends by a property of the *edge* (`knows.since`) and
//!   returns it alongside the friend. The DSL orders vertices by vertex
//!   properties only — wanted: `EdgeTraversal::order_by_prop{,_desc}` and an
//!   edge→endpoint step that keeps the edge's properties.
//! - **IS6** walks a `replyOf` chain of unbounded depth to reach the root
//!   post. The DSL is fixed-depth — this is exactly board task **B3**
//!   (`repeat`/`until`).
//!
//! Both are implemented correctly in plain Rust over the engine API, so E3 is
//! not blocked; the gaps are recorded as evidence for what a Gremlin-subset
//! DSL actually needs, which is itself an RQ-relevant finding.

pub mod fixture;
pub mod native;
pub mod results;

#[cfg(test)]
mod tests_native;
