//! gstress — stress harness for the twizzler-graph engine.
//!
//! Deliberately not part of `cargo start-qemu --tests`, so the default
//! harness stays fast. Usage, from the Twizzler shell:
//!
//!   gstress            # = gstress small
//!   gstress tiny       # smoke run of every phase
//!   gstress small      # the default preset
//!   gstress medium     # larger; adjacency-ceiling probe
//!   gstress large      # go until something gives
//!   gstress scale:<N>  # arbitrary size with tiny's proportions, for size
//!                      # sweeps: `gstress scale:1000`, `gstress scale:2000
//!                      # indradb`
//!   gstress <preset> arena[:cap]
//!                      # explicit arena cap, for cap sweeps. Without it,
//!                      # DEFAULT_ARENA_CAP. Every native run is the arena
//!                      # layout; there is no layout argument.
//!   gstress <preset> indradb
//!                      # the same workload against the IndraDB baseline
//!   gstress seed <N>     # write a known graph, then reboot and
//!   gstress verify <N>   # check it survived. Two boots, by construction —
//!                        # a same-boot reopen maps resident pages and cannot
//!                        # tell durable from merely mapped. Records carry
//!                        # 0–3 inline traversal properties, interleaved, so
//!                        # variable stride is exercised across a real
//!                        # write-back and re-read.
//!   gstress indradb-seed <N>     # the same cross-reboot durability
//!   gstress indradb-verify <N>   # protocol against the IndraDB datastore
//!   gstress residency [cycle|hold|volatile|…|del] [N] [R]
//!                      # does dropping a handle return frames? Not a graph
//!                      # workload — a platform probe. See residency.rs; run
//!                      # each arm in its own boot.
//!   gstress index [N] [noindex|bulk|sync:K|throttle:K|rebuild]
//!                      # what limits a large load — insertion, the index,
//!                      # residency, or writeback? `noindex` is the control,
//!                      # `bulk` batches the index, `sync:K` syncs every K
//!                      # records (write-behind), `throttle:K` paces insertion
//!                      # without moving the sync. Own boot each; see
//!                      # index_probe.rs.
//!   gstress ldbc       # load LDBC-SNB SF0.1 from /initrd/*.csv and report
//!                      # vertices, edges, arenas, blob objects, and the
//!                      # node/edge/sync split. Needs `scripts/flatten_ldbc.py`
//!                      # run on the host first.
//!   gstress ldbc-query [N]
//!                      # LDBC short reads IS1-IS7 against the graph
//!                      # `gstress ldbc` left on disk. Own boot. Reports
//!                      # per-query latency percentiles.
//!   gstress is2 <arm> [N]
//!                      # which term of the IS2 ordering cost dominates.
//!                      # arm = walk | props | full | sort, one per boot —
//!                      # arms share a page cache otherwise. Prints a row per
//!                      # person so latency can be regressed on in-degree
//!                      # rather than averaged. See is2_probe.rs.
//!   gstress ldbc-indradb-load [edgeprops] [lean]
//!                      # baseline load pass. Slow: every property write is a
//!                      # transaction.
//!                      # `edgeprops` additionally stores the extra CSV columns
//!                      # (joinDate, workFrom, classYear, likes.creationDate)
//!                      # as edge properties, which the complex reads
//!                      # IC1/IC5/IC7/IC11 need. Off by default.
//!                      # `lean` additionally drops the five vertex columns no
//!                      # complex read touches (locationIP, browserUsed,
//!                      # length, language, url) plus knows.creationDate, for
//!                      # when the full load runs out of frames. Try without
//!                      # it first, and never run the short reads against a
//!                      # lean store — IS1 returns three of the dropped
//!                      # columns.
//!   gstress ldbc-query-equiv [N]
//!   gstress ldbc-indradb-equiv [N]
//!                      # same queries, digest output instead of latencies,
//!                      # to check the two arms agree. Run one per boot, keep
//!                      # both logs, diff them on the host.
//!   gstress ldbc-indradb [N]
//!                      # baseline query pass, in a separate boot so both
//!                      # engines are measured cold. Same seven queries, same
//!                      # data, same LDBC person ids as `ldbc-query`.
//!   gstress ldbc-indradb-indexed [N]
//!                      # baseline query pass with a key index created at
//!                      # open, making the two setups symmetric
//!   gstress ldbc-complex [N] [q:1,5,10] [detail:5] [budget:S] [digest]
//!   gstress ldbc-complex-indradb [same arguments]
//!                      # the LDBC interactive complex reads IC1-IC14,
//!                      # native and baseline. Own boot each, against the graph
//!                      # / store the matching load left on disk.
//!                      # `digest`  — one pass over the parameter list, emitting
//!                      #             result digests instead of a timing claim.
//!                      #             This is the correctness pass; run it
//!                      #             before quoting latency.
//!                      # `q:`      — which queries (default 1-14).
//!                      # `detail:` — also print per-parameter result rows for
//!                      #             these queries, for when a hash disagrees.
//!                      #             Ask for one query at a time: the guest's
//!                      #             stdout drops long output.
//!                      # `budget:` — per-query wall-clock seconds, 0 = off. A
//!                      #             query that cannot finish then costs one
//!                      #             query rather than the whole run.
//!                      # A host-side oracle computes the same answers from
//!                      # the CSVs, for a three-way diff of the results.
//!   gstress reclaim [N] [C] [keep] [cap:K]
//!                      # does Delete return frames? `keep` is the control
//!                      # (build and never destroy) and is not optional —
//!                      # `destroy` alone proves nothing. Report the pair.
//!                      # Own boot each; see reclaim_probe.rs.
//!   gstress destroy [N] [R] [reset]
//!                      # R cycles of build-then-destroy. The result is the
//!                      # disk image's size, measured on the host between
//!                      # boots. `reset` is the control arm.
//!   gstress props [N] [none]
//!                      # how many vertices can carry a property before the
//!                      # pager gives out? Properties are one persistent
//!                      # object each. `none` is the control arm — not
//!                      # optional, or a stall cannot be attributed to
//!                      # properties rather than vertex count. Own boot each.
//!
//! Clear `target/disk-<triple>.img` before any recorded measurement. It is
//! created only if absent and nothing ever deletes an object, so it carries
//! every graph every previous run made; a dirty image is not comparable to a
//! clean one.
//!
//! Every operation is direct. Batching is internal to the arena store — one
//! transaction per arena — so there is nothing for the harness to batch on
//! its behalf.
//!
//! Long loops print heartbeat lines so a stall is visible (and attributable)
//! immediately.
//!
//! Phases:
//!   A  vertex insertion rate, sampled lookups
//!   B  bulk random edges, sampled out-degree verification
//!   C  high-degree hub until failure or cap (adjacency ceiling);
//!      medium/large add a same-target variant (probes FOT dedup)
//!   D  add/delete churn, then a full verification scan
//!   E  drop + reopen by name, re-verify by sampling
//!   F  pathological shapes: deep chain walked end-to-end, dense clique
//!   H  read workloads (lookup, 1-hop, 2-hop, scan), cold and warm
//!   G  degradation probe: identical insert windows, first vs last rate
//!
//! Every phase reports ops and wall time. Any verification mismatch prints
//! `GSTRESS FAIL: ...` and the process exits nonzero at the end. Expected
//! capacity findings (e.g. the adjacency ceiling) print `GSTRESS FINDING:`
//! and do not fail the run.
//!
//! The graph is registered as `data/gstress` and reset at startup, so runs
//! are idempotent (old registries are orphaned, as with `Graph::reset`).

use std::time::Instant;

use twizzler::object::ObjID;
use twizzler_graph::{Graph, Labels, Lookup, VertexId};

const GRAPH: &str = "gstress";

/// The default schema indexes nothing until asked, so every label this
/// harness later resolves by name has to be declared — otherwise `find_vertex`
/// correctly answers `NotIndexed` and every verification reports a failure that
/// is really a missing declaration.
///
/// Declare before the inserts. `index_on_insert` records a `RootEntry` only
/// for labels indexed at insert time and `set_label_indexed` does no backfill,
/// so on a `Roots` graph a declare-after-insert label rebuilds an index that
/// cannot see the earlier records. Under `Scan` either order works.
pub(crate) fn declare_lookup_labels(g: &mut Graph) {
    // "vo" is the seed/verify vertices-only tranche.
    for l in ["n", "d", "d2", "c", "hub", "spoke", "tag", "vo"] {
        // Best-effort: a strategy of `None` refuses, and that is a legitimate
        // configuration for arms that never look up.
        let _ = g.set_label_indexed(l, true);
    }
}

/// Deterministic text payload for `seed`/`verify`: `verify` runs in a
/// fresh boot and cannot capture what `seed` wrote, so both sides derive the
/// value from `(i, width)` alone. Content varies with `i` so a swapped or
/// zeroed value cannot match by accident.
pub(crate) fn a9_text(i: usize, w: usize) -> String {
    (0..w)
        .map(|j| char::from(b'a' + ((i + j) % 26) as u8))
        .collect()
}

/// Deterministic blob payload, 2–3 KB, byte-varied per `i`.
pub(crate) fn a9_blob(i: usize) -> Vec<u8> {
    (0..(2048 + i % 1024))
        .map(|j| ((i.wrapping_mul(31) + j) % 251) as u8)
        .collect()
}

/// Harness revision, printed with every run, so a pasted result block can be
/// matched to the harness that produced it. Bump it on every change that can
/// move a number — workload, timing, or verification.
///
/// Edge records live in the vertex id space, so an id derived by arithmetic
/// is valid only under one invariant: a phase that creates vertices before
/// any edge gets contiguous ids from 0. `VertexId(i)` therefore works for
/// phase-A vertices and for `seed`'s `v*`; anything created after an edge is
/// not addressable that way, and a reordered phase turns such reads into
/// silent misreads rather than errors.
pub(crate) const HARNESS_REV: &str = "2026-08-19b";

mod index_probe;
mod is2_probe;
// The LDBC interactive complex reads (IC1–IC14), both arms.
// `ldbc_common` holds what the two must agree on byte for byte: the digest
// format, per-query parameter loading, latency accumulation and the budget.
mod ldbc_common;
mod ldbc_complex;
mod ldbc_complex_indradb;
mod ldbc_indradb;
mod ldbc_load;
mod ldbc_query;
mod reclaim_probe;
mod indradb_mode;
mod props_probe;
mod residency;

/// Print the provenance header. Every recorded result must carry this line;
/// results without one cannot be trusted after the engine changes.
pub(crate) fn stamp(mode: &str, preset: &Preset) {
    println!(
        "GSTRESS STAMP harness={} mode={} preset={} V={} E={} degCap={} chain={} clique={}",
        HARNESS_REV, mode, preset.name, preset.vertices, preset.bulk_edges,
        preset.degree_cap, preset.chain, preset.clique
    );
}

/// Workload sizes. Tunable constants.
pub(crate) struct Preset {
    pub(crate) name: &'static str,
    /// Phase A vertices.
    ///
    /// Phase A measures the vertex-insertion rate. No preset reaches the
    /// 262 144 segment cap, so registry rollover is not exercised here; the
    /// phase still prints as `A:rollover` so old logs stay comparable.
    pub(crate) vertices: usize,
    /// Phase B random edges.
    pub(crate) bulk_edges: usize,
    /// Phase C hub out-degree cap ("until failure or this").
    pub(crate) degree_cap: usize,
    /// Whether phase C also runs the same-target variant (FOT-dedup probe).
    same_target_variant: bool,
    /// Phase D vertices added after the deletes.
    pub(crate) churn_add: usize,
    /// Phase F chain length (walked end-to-end).
    pub(crate) chain: usize,
    /// Phase F clique size (k vertices, k*(k-1) directed edges).
    pub(crate) clique: usize,
    /// Phase G: number of equal windows in the degradation probe.
    pub(crate) degrade_windows: usize,
    /// Phase G: records per window (kept small and constant, so any change
    /// in window rate is the system degrading, not the workload changing).
    pub(crate) degrade_batch: usize,
    /// Phase H: how many times to repeat each read workload. Reads are fast
    /// enough that a single pass can land at or below timer resolution;
    /// repetition moves the measurement above the noise floor.
    pub(crate) read_reps: usize,
}

/// Build a preset of arbitrary size, with every sub-workload scaled in the
/// same proportions as `tiny` (which `scaled(200)` reproduces).
///
/// Lets a size sweep run without inventing a named preset per point:
/// `gstress scale:1000`, `gstress scale:2000 indradb`, and so on.
pub(crate) fn scaled(n: usize) -> Preset {
    let n = n.max(20);
    Preset {
        name: "scale",
        vertices: n,
        bulk_edges: n * 3 / 2,
        degree_cap: (n / 4).max(1),
        same_target_variant: false,
        churn_add: (n / 4).max(1),
        chain: (n / 2).max(2),
        clique: (((n / 2) as f64).sqrt() as usize).max(3),
        degrade_windows: 10,
        degrade_batch: (n / 10).max(2),
        read_reps: 20,
    }
}

/// Sampling stride for read phases — shared by both arms so they measure
/// the same vertices.
pub(crate) fn read_step(n: usize) -> usize {
    (n / 100).max(1)
}

/// Warm-phase target duration. A fixed rep count cannot serve both arms: a
/// count that gives the slower arm a sane runtime leaves the faster one at
/// timer resolution, and vice versa. Each arm therefore repeats until it
/// reaches this duration; comparing rates stays valid because the rate is
/// per-op.
const READ_TARGET_SECS: f64 = 1.0;

/// Measure a read workload in two regimes, reporting both.
///
/// Cold is the first pass — it pays for mapping objects and faulting them
/// in. Warm is steady-state, once the working set is resident. Averaging the
/// two into one number would hide the residency cost, which is the effect
/// worth reporting.
///
/// `f` performs one pass and returns the number of logical operations in it.
pub(crate) fn measure_read(phase: &str, max_reps: usize, mut f: impl FnMut() -> usize) {
    let t0 = Instant::now();
    let cold_ops = f();
    let cold = t0.elapsed().as_secs_f64();
    let cold_rate = if cold > 0.0 { cold_ops as f64 / cold } else { 0.0 };

    let t1 = Instant::now();
    let mut ops = 0usize;
    let mut reps = 0usize;
    while t1.elapsed().as_secs_f64() < READ_TARGET_SECS && reps < max_reps {
        ops += f();
        reps += 1;
    }
    let warm = t1.elapsed().as_secs_f64();
    let warm_rate = if warm > 0.0 { ops as f64 / warm } else { 0.0 };

    println!(
        "GSTRESS {phase:<10} cold {cold_ops:>7} ops {cold:>7.3}s {cold_rate:>10.0} ops/s | \
         warm {ops:>8} ops ({reps} reps) {warm:>6.3}s {warm_rate:>10.0} ops/s"
    );
    if cold_rate > 0.0 && warm_rate > 0.0 {
        println!(
            "GSTRESS RESIDENCY {phase}: warm/cold = {:.1}x",
            warm_rate / cold_rate
        );
    }
}

const TINY: Preset = Preset {
    name: "tiny",
    vertices: 200,
    bulk_edges: 300,
    degree_cap: 50,
    same_target_variant: false,
    churn_add: 50,
    chain: 100,
    clique: 10,
    degrade_windows: 10,
    degrade_batch: 20,
    read_reps: 20,
};
const SMALL: Preset = Preset {
    name: "small",
    vertices: 4_500,
    bulk_edges: 6_000,
    degree_cap: 1_000,
    same_target_variant: false,
    churn_add: 300,
    chain: 2_000,
    clique: 30,
    degrade_windows: 20,
    degrade_batch: 50,
    read_reps: 5,
};
const MEDIUM: Preset = Preset {
    name: "medium",
    vertices: 20_000,
    bulk_edges: 50_000,
    degree_cap: 50_000,
    same_target_variant: true,
    churn_add: 2_000,
    chain: 10_000,
    clique: 60,
    degrade_windows: 20,
    degrade_batch: 200,
    read_reps: 2,
};
const LARGE: Preset = Preset {
    name: "large",
    vertices: 50_000,
    bulk_edges: 100_000,
    degree_cap: 50_000,
    same_target_variant: true,
    churn_add: 5_000,
    chain: 10_000,
    clique: 80,
    degrade_windows: 20,
    degrade_batch: 500,
    read_reps: 1,
};

/// Deterministic xorshift64 so runs are reproducible.
pub(crate) struct Rng(pub(crate) u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Failure lines printed before suppression starts.
///
/// A systematic failure emits one line per checked item, and the checks run
/// over every vertex. Unbounded failure output can overwhelm the serial
/// console and kill the rest of the run, so the per-item lines are capped —
/// a dozen already show the pattern. The failure count is still reported in
/// full.
const MAX_FAIL_LINES: u64 = 20;

pub(crate) struct Stats {
    pub(crate) fails: u64,
    printed: u64,
    last: Option<String>,
}
impl Stats {
    pub(crate) fn new() -> Self {
        Stats {
            fails: 0,
            printed: 0,
            last: None,
        }
    }
    pub(crate) fn fail(&mut self, msg: String) {
        self.fails += 1;
        if self.printed < MAX_FAIL_LINES {
            println!("GSTRESS FAIL: {msg}");
            self.printed += 1;
            return;
        }
        if self.printed == MAX_FAIL_LINES {
            println!(
                "GSTRESS FAIL: further failure lines suppressed after {MAX_FAIL_LINES}; \
                 the last one and the total are reported at exit"
            );
            self.printed += 1;
        }
        self.last = Some(msg);
    }

    /// The final suppressed failure, printed at exit.
    ///
    /// A systematic failure fills the cap with a single shape, so anything
    /// different failing later would otherwise be invisible.
    pub(crate) fn report_suppressed(&self) {
        if let Some(msg) = &self.last {
            println!("GSTRESS FAIL (last suppressed): {msg}");
        }
    }
    pub(crate) fn ck(&mut self, cond: bool, msg: impl FnOnce() -> String) {
        if !cond {
            self.fail(msg());
        }
    }
}

/// Progress line inside long loops, so a stall is visible and attributable.
/// Windowed progress: reports the rate for this window alongside the
/// cumulative rate.
///
/// Cumulative rates hide decay — a rate that halves partway through shows up
/// as a gentle droop. Window rates show it directly.
pub(crate) struct Progress {
    phase: &'static str,
    start: Instant,
    last: Instant,
    last_done: usize,
    first_window_rate: Option<f64>,
    last_window_rate: f64,
}

impl Progress {
    pub(crate) fn new(phase: &'static str) -> Self {
        let now = Instant::now();
        Progress {
            phase,
            start: now,
            last: now,
            last_done: 0,
            first_window_rate: None,
            last_window_rate: 0.0,
        }
    }

    /// Record progress at `done` total records.
    pub(crate) fn tick(&mut self, done: usize, total: usize) {
        let now = Instant::now();
        let win_ops = done.saturating_sub(self.last_done);
        let win_secs = now.duration_since(self.last).as_secs_f64();
        let cum_secs = now.duration_since(self.start).as_secs_f64();
        let win_rate = if win_secs > 0.0 {
            win_ops as f64 / win_secs
        } else {
            0.0
        };
        let cum_rate = if cum_secs > 0.0 {
            done as f64 / cum_secs
        } else {
            0.0
        };
        if self.first_window_rate.is_none() {
            self.first_window_rate = Some(win_rate);
        }
        self.last_window_rate = win_rate;
        println!(
            "GSTRESS {}: {}/{}  window {:.1} ops/s  cumulative {:.1} ops/s",
            self.phase, done, total, win_rate, cum_rate
        );
        self.last = now;
        self.last_done = done;
    }

    /// Print first-vs-last window, the number that answers "does throughput
    /// decay as the run proceeds?".
    pub(crate) fn summarize(&self) {
        let first = self.first_window_rate.unwrap_or(0.0);
        if first <= 0.0 || self.last_window_rate <= 0.0 {
            return;
        }
        let ratio = first / self.last_window_rate;
        println!(
            "GSTRESS DEGRADE {}: first window {:.1} ops/s, last {:.1} ops/s, \
             slowdown {:.2}x",
            self.phase, first, self.last_window_rate, ratio
        );
    }
}

/// Progress line, with a projected time to finish the phase. An ETA that
/// moves is the difference between "slow" and "stuck".
///
/// It is a projection at the current cumulative rate, so a phase whose rate
/// decays will overshoot it. Read it as a floor, not a promise.
pub(crate) fn heartbeat(phase: &str, done: usize, total: usize, t: &Instant) {
    let secs = t.elapsed().as_secs_f64();
    let rate = if secs > 0.0 { done as f64 / secs } else { 0.0 };
    if rate > 0.0 && total > done {
        let eta = (total - done) as f64 / rate;
        println!("GSTRESS {phase}: {done}/{total} ({rate:.0} ops/s, ~{eta:.0}s left)");
    } else {
        println!("GSTRESS {phase}: {done}/{total} ({rate:.0} ops/s)");
    }
}

pub(crate) fn report(phase: &str, ops: usize, t: Instant) {
    let secs = t.elapsed().as_secs_f64();
    let rate = if secs > 0.0 { ops as f64 / secs } else { 0.0 };
    println!("GSTRESS {phase:<12} {ops:>8} ops  {secs:>8.2}s  {rate:>10.0} ops/s");
}

/// Churn (phase D) deletes every 7th of the phase-A vertices; hub targets and
/// verification anchors must avoid those indices.
fn survives_churn(i: usize) -> bool {
    i % 7 != 0
}

/// The `i`-th vertex index that survives churn — injective in `i`.
///
/// Distinct targets matter because the engines differ: ours is a multigraph
/// (parallel edges are distinct records — see the engine's `parallel_edges`
/// test), while IndraDB keys an edge on `(outbound, type, inbound)` and
/// silently rejects duplicates. Colliding targets would make the arms do
/// unequal work.
///
/// `i + i/6 + 1` skips every multiple of 7 and is strictly increasing, so
/// distinct `i` give distinct surviving targets.
pub(crate) fn safe_target(i: usize, n: usize) -> usize {
    let j = i + i / 6 + 1;
    debug_assert!(survives_churn(j), "target must survive churn");
    j % n
}

/// Largest hub degree for which [`safe_target`] stays injective; beyond it the
/// wrap reintroduces duplicates and the two engines diverge again.
pub(crate) fn max_distinct_degree(n: usize) -> usize {
    // invert j = i + i/6 + 1 < n
    (n.saturating_sub(1)) * 6 / 7
}

fn main() {
    let arg1 = std::env::args().nth(1);

    // Probe subcommands take their own arguments, so they dispatch before the
    // preset match.
    // index: is a vertex insert mostly the `(label, name)` index? See
    // index_probe.rs.
    if arg1.as_deref() == Some("index") {
        let n = std::env::args()
            .nth(2)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(20_000);
        let arm = std::env::args().nth(3).unwrap_or_else(|| "graph".into());
        index_probe::run(n, &arm);
        return;
    }

    // Load LDBC-SNB from /initrd and report load cost.
    if arg1.as_deref() == Some("ldbc") {
        ldbc_load::run();
        return;
    }

    // Run LDBC's short reads against the loaded graph and report per-query
    // latency percentiles.
    if arg1.as_deref() == Some("ldbc-query") {
        let iters = std::env::args()
            .nth(2)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1000);
        ldbc_query::run(iters);
        return;
    }

    // The IS2 ordering cost, decomposed. Separate from `ldbc-query` because it
    // runs one term per boot — see is2_probe.rs.
    if arg1.as_deref() == Some("is2") {
        let arm = std::env::args().nth(2).unwrap_or_else(|| "full".into());
        let iters = std::env::args()
            .nth(3)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(200);
        is2_probe::run(&arm, iters);
        return;
    }

    // Do the two arms compute the same thing? Emits a result digest per
    // person from the same code path the benchmark times, so the check cannot
    // drift from what is measured. Diff the two arms' logs on the host.
    if arg1.as_deref() == Some("ldbc-query-equiv") {
        let iters = std::env::args()
            .nth(2)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1000);
        ldbc_query::equiv(iters);
        return;
    }

    if arg1.as_deref() == Some("ldbc-indradb-equiv") {
        let iters = std::env::args()
            .nth(2)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1000);
        ldbc_indradb::equiv(iters);
        return;
    }

    // The LDBC interactive complex reads, IC1–IC14, native arm.
    //   gstress ldbc-complex [iters] [q:1,5,10] [detail:5] [budget:600] [digest]
    // Own boot, against the graph `gstress ldbc` left on disk — same protocol
    // as `ldbc-query`.
    if arg1.as_deref() == Some("ldbc-complex") {
        ldbc_complex::run();
        return;
    }

    // Baseline arm. Identical arguments, so the two runs differ only in
    // which engine answers.
    if arg1.as_deref() == Some("ldbc-complex-indradb") {
        ldbc_complex_indradb::run();
        return;
    }

    // Baseline load: same data, IndraDB.
    // `edgeprops` additionally stores the extra CSV columns as edge properties,
    // which the complex reads IC1/IC5/IC7/IC11 need. Off by default; see
    // `ldbc_indradb::load`.
    if arg1.as_deref() == Some("ldbc-indradb-load") {
        let edge_props = std::env::args().any(|a| a == "edgeprops");
        // `lean` drops columns no complex read touches, for when the full
        // load runs out of frames. Try without it first: a faithful store is
        // worth more.
        let lean = std::env::args().any(|a| a == "lean");
        ldbc_indradb::load(edge_props, lean);
        return;
    }

    // Same queries, baseline given a key index at open — the arm that
    // makes the two setups symmetric. See `ldbc_indradb::run_indexed`.
    if arg1.as_deref() == Some("ldbc-indradb-indexed") {
        let iters = std::env::args()
            .nth(2)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1000);
        ldbc_indradb::run_indexed(iters);
        return;
    }

    if arg1.as_deref() == Some("ldbc-indradb") {
        let iters = std::env::args()
            .nth(2)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1000);
        ldbc_indradb::run(iters);
        return;
    }

    if arg1.as_deref() == Some("reclaim") {
        let rest: Vec<String> = std::env::args().skip(2).collect();
        let nums: Vec<usize> = rest.iter().filter_map(|s| s.parse().ok()).collect();
        let n = nums.first().copied().unwrap_or(100_000);
        let cycles = nums.get(1).copied().unwrap_or(12);
        // `keep` is the control arm, not a variant: see reclaim_probe.rs.
        let keep = rest.iter().any(|s| s == "keep");
        let cap = rest
            .iter()
            .find_map(|s| s.strip_prefix("cap:").and_then(|v| v.parse().ok()))
            .unwrap_or(256);
        reclaim_probe::run(n, cycles, keep, cap);
        return;
    }

    if arg1.as_deref() == Some("props") {
        let n = std::env::args()
            .nth(2)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(30_000);
        let with_props = std::env::args().nth(3).as_deref() != Some("none");
        props_probe::run(n, with_props);
        return;
    }

    if arg1.as_deref() == Some("residency") {
        let arm = std::env::args().nth(2);
        let n = std::env::args()
            .nth(3)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(300);
        // Doubles as the per-object element count for the `write` arm, where
        // 10 rounds is meaningless but 64 elements is a sane default (every
        // push syncs, so this number is expensive).
        let rounds = std::env::args()
            .nth(4)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(if arm.as_deref() == Some("write") { 64 } else { 10 });
        residency::run(arm.as_deref(), n, rounds);
        return;
    }

    // The datastore's cross-reboot durability pair. Same protocol as the
    // native pair below: seed, reboot, verify.
    if matches!(arg1.as_deref(), Some("indradb-seed") | Some("indradb-verify")) {
        let n = std::env::args()
            .nth(2)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(5000);
        if arg1.as_deref() == Some("indradb-seed") {
            indradb_mode::seed_durable(n);
        } else {
            indradb_mode::verify_durable(n);
        }
        return;
    }

    // `gstress seed <N>` then, in a later boot, `gstress verify <N>`.
    //
    // A same-boot reopen maps objects whose pages are still resident — it
    // never reads the disk image, so it cannot distinguish durable from
    // merely mapped.
    //
    // The two phases are separate processes in separate boots by construction:
    // there is no way for `verify` to see anything `seed` left in memory.
    if matches!(arg1.as_deref(), Some("seed") | Some("verify")) {
        let verify = arg1.as_deref() == Some("verify");
        let n = std::env::args()
            .nth(2)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(5000);
        const DUR: &str = "gdurable";
        println!(
            "GSTRESS STAMP harness={} mode={} N={}",
            HARNESS_REV,
            if verify { "verify" } else { "seed" },
            n
        );
        let mut st = Stats::new();

        if !verify {
            Graph::reset_arena(DUR, twizzler_graph::DEFAULT_ARENA_CAP).expect("reset");
            let mut g = Graph::open_or_create_arena(DUR, twizzler_graph::DEFAULT_ARENA_CAP)
                .expect("open");
            crate::declare_lookup_labels(&mut g);
            let mut ids = Vec::with_capacity(n);
            for i in 0..n {
                // Deliberately varied inline widths, interleaved.
                //
                // A record's extent comes from its own `nprops`, so records of
                // different widths sit back-to-back at irregular offsets. A
                // stride error does not fail — it reads the neighbouring
                // record's bytes as this one's fields. Within-boot tests
                // cover the arithmetic; only a reboot covers it after the pager
                // has written back and re-read. Zero-width and widest records
                // are adjacent (i % 4 cycles 0,1,2,3) so a slip shows up as a
                // neighbour misread.
                let w = i % 4;
                let props: Vec<(&str, twizzler_graph::PropValue)> = (0..w)
                    .map(|j| {
                        (
                            ["t0", "t1", "t2"][j],
                            twizzler_graph::PropValue::I64((i * 10 + j) as i64),
                        )
                    })
                    .collect();
                ids.push(
                    g.add_vertex_with_props("d", &format!("v{i}"), ObjID::new(i as u128), &props)
                        .expect("add_vertex_with_props"),
                );
            }
            for i in 0..n.saturating_sub(1) {
                g.add_edge(ids[i], "e", ids[i + 1]).expect("add_edge");
            }
            // A property on every 5th, a tombstone on every 7th, so `verify`
            // checks live records, deleted ones, and properties.
            for i in (0..n).step_by(5) {
                g.set_vertex_prop(ids[i], "k", twizzler_graph::PropValue::I64(i as i64))
                    .expect("set prop");
            }
            for i in (0..n).step_by(7) {
                g.delete_vertex(ids[i]).expect("delete");
            }
            // Text and blob values at every tier width, written before the
            // sync so `verify` reads them from the disk image.
            // Widths cycle empty/31/32/255 so both sides of the Str/text
            // boundary and the text ceiling are on disk; every 8th carrier
            // also gets a multi-KB blob, which exercises the blob store's own
            // segments and sync path. Payloads are derived from `i` alone
            // (`a9_text`/`a9_blob`) so the verify boot can reconstruct them.
            for (t, i) in (0..n).step_by(11).enumerate() {
                if i % 7 == 0 {
                    continue; // tombstoned above; a dead vertex takes no writes
                }
                let w = [0usize, 31, 32, 255][t % 4];
                g.set_vertex_text(ids[i], "long", &a9_text(i, w))
                    .expect("set text");
                if t % 8 == 0 {
                    g.set_vertex_blob(ids[i], "content", &a9_blob(i))
                        .expect("set blob");
                }
            }
            g.sync().expect("sync");

            // Reopen mid-seed, then keep writing: a store is dropped and
            // reopened, and the reopened instance grows the graph. Writes
            // made through the second instance have to be durable even though
            // the first instance created the structures they live in. Without
            // this, the check passes on a store whose registries were only
            // ever written by one instance.
            drop(g);
            let mut g = Graph::open_or_create(DUR).expect("mid-seed reopen");
            let base = ids.len();
            for i in 0..n / 4 {
                let v = g
                    .add_vertex("d2", &format!("w{i}"), ObjID::new(i as u128))
                    .expect("add_vertex after reopen");
                ids.push(v);
            }
            for i in 0..(n / 4).saturating_sub(1) {
                g.add_edge(ids[base + i], "e2", ids[base + i + 1])
                    .expect("add_edge after reopen");
            }
            // Text and blob through the reopened handle too: the blob store
            // these land in was created by the first handle, so this covers
            // text/blob writes against reopened directories, the analogue of
            // the `w*` tranche one line up.
            if n >= 4 {
                g.set_vertex_text(ids[base], "long", &a9_text(base, 255))
                    .expect("set text after reopen");
                g.set_vertex_blob(ids[base], "content", &a9_blob(base))
                    .expect("set blob after reopen");
            }
            g.sync().expect("sync after reopen");

            // Vertices-only tranche.
            // sync → pure `add_vertex` → sync: between the two syncs nothing
            // touches the arenas except `add_record` itself — no edge,
            // property, text or tombstone write, each of which marks arenas
            // dirty through `record_ptr` incidentally. This isolates
            // `add_record`'s own dirty-marking.
            // `a_vertex_only_batch_syncs_its_arena` covers the same case
            // within-boot at the sync-count level; only this tranche proves
            // the bytes actually reach the disk image.
            let vonly = (n / 8).max(64);
            for i in 0..vonly {
                g.add_vertex("vo", &format!("q{i}"), ObjID::new(i as u128))
                    .expect("vertices-only add");
            }
            g.sync().expect("vertices-only sync");
            println!(
                "GSTRESS SEED: {} vertices ({} arenas), every 5th has a property, \
                 every 7th deleted, text/blob on every 11th survivor, {} more \
                 added through a reopened handle, then a vertices-only tranche \
                 of {}. Now reboot and run `gstress verify {n}`.",
                ids.len(),
                g.arena_count(),
                n / 4,
                vonly
            );
            return;
        }

        // --- verify, in a fresh boot -------------------------------------
        let g = match Graph::open_or_create(DUR) {
            Ok(g) => g,
            Err(e) => {
                println!("GSTRESS DURABILITY FAILED: cannot open the seeded graph: {e:?}");
                std::process::exit(1);
            }
        };
        // The `w*` vertices were written through a reopened handle; if the
        // registries created by the first handle did not reach disk, these are
        // the ones that vanish.
        let post = n / 4;
        for i in (0..post).step_by(23) {
            // By name, not by computed id: edge records share the vertex id
            // space, so after `n` vertices come `n-1` edge records and `w0`'s
            // id is nowhere near `n`. `verify` runs in a fresh boot and
            // cannot capture ids, so it must look them up the way a user
            // would.
            // Label `d2`, not `d` — the post-reopen vertices are written under
            // their own label, and `find_vertex` keys on (label, name).
            st.ck(
                g.find_vertex("d2", &format!("w{i}"))
                    .found()
                    .and_then(|v| g.vertex_info(v))
                    .map(|inf| inf.name)
                    == Some(format!("w{i}")),
                || format!("w{i} (written after a mid-seed reopen) did not survive"),
            );
        }
        // Resolve a sample of the `v*` tranche by name, and check the `e2`
        // edges written through the reopened handle.
        for i in (0..n).step_by(31) {
            if i % 7 == 0 {
                continue;
            }
            st.ck(
                g.find_vertex("d", &format!("v{i}")).found() == Some(VertexId(i as u64)),
                || format!("v{i} does not resolve by name to its own id"),
            );
        }
        for i in (0..post.saturating_sub(1)).step_by(17) {
            let a = g.find_vertex("d2", &format!("w{i}")).found();
            let b = g.find_vertex("d2", &format!("w{}", i + 1)).found();
            match (a, b) {
                (Some(a), Some(b)) => st.ck(
                    g.out_neighbors(a, Labels::these(&["e2"])).contains(&b),
                    || format!("w{i} lost its e2 edge to w{}", i + 1),
                ),
                _ => st.fail(format!("w{i}/w{} unresolvable for the e2 check", i + 1)),
            }
        }
        let vonly = (n / 8).max(64);
        let expect_live = (0..n).filter(|i| i % 7 != 0).count() + post + vonly;
        let live = g.vertices().len();
        st.ck(live == expect_live, || {
            format!("live vertices: got {live}, expected {expect_live}")
        });

        // The vertices-only tranche. These records' arenas were marked dirty
        // by nothing but `add_record` itself — if that mark is lost, the
        // names and targets below read back as zeros while the mirror still
        // counts them live (the count check above alone would pass).
        for i in (0..vonly).step_by(13) {
            st.ck(
                g.find_vertex("vo", &format!("q{i}"))
                    .found()
                    .and_then(|v| g.vertex_info(v))
                    .map(|inf| (inf.name, inf.target))
                    == Some((format!("q{i}"), ObjID::new(i as u128))),
                || {
                    format!(
                        "q{i} (vertices-only tranche) did not survive the reboot — \
                         its arena was never synced (add_record dirty-marking)"
                    )
                },
            );
        }
        println!("GSTRESS VERIFY: {} arenas recovered", g.arena_count());
        st.ck(g.arena_count() > 0, || {
            "arena directory came back empty — the store's arenas did not reach disk".into()
        });

        // Every text/blob tier width, read back after the reboot. Same
        // iteration as the seed loop, so `t` (and with it the width cycle and
        // the every-8th blob choice) lines up exactly; payloads are
        // re-derived from `i`.
        for (t, i) in (0..n).step_by(11).enumerate() {
            if i % 7 == 0 {
                continue;
            }
            let v = VertexId(i as u64);
            let w = [0usize, 31, 32, 255][t % 4];
            let want = a9_text(i, w);
            st.ck(
                g.get_vertex_text(v, "long").as_deref() == Some(want.as_str()),
                || {
                    format!(
                        "v{i}: {w}-byte text did not survive the reboot \
                         (got {:?}...)",
                        g.get_vertex_text(v, "long").map(|s| {
                            let mut s = s;
                            s.truncate(16);
                            s
                        })
                    )
                },
            );
            if t % 8 == 0 {
                let want = a9_blob(i);
                st.ck(
                    g.get_vertex_blob(v, "content").as_deref() == Some(&want[..]),
                    || {
                        format!(
                            "v{i}: {}-byte blob did not survive the reboot — \
                             blob segments have their own sync path",
                            want.len()
                        )
                    },
                );
            }
        }
        // The post-reopen text/blob pair, looked up by name like the rest of
        // the `w*` tranche.
        if n >= 4 {
            let w0 = g.find_vertex("d2", "w0").found();
            st.ck(
                w0.map(|v| g.get_vertex_text(v, "long"))
                    == Some(Some(a9_text(n, 255))),
                || "w0: text written through the reopened handle did not survive".into(),
            );
            st.ck(
                w0.map(|v| g.get_vertex_blob(v, "content"))
                    == Some(Some(a9_blob(n))),
                || "w0: blob written through the reopened handle did not survive".into(),
            );
        }

        // Every inline width, read back after the reboot. Stepping by a
        // number coprime to 4 so all four widths are sampled rather than one.
        for i in (0..n).step_by(97) {
            if i % 7 == 0 {
                continue; // tombstoned; checked below
            }
            let v = VertexId(i as u64);
            let w = i % 4;
            for j in 0..w {
                let key = ["t0", "t1", "t2"][j];
                let want = twizzler_graph::PropValue::I64((i * 10 + j) as i64);
                st.ck(g.get_vertex_prop(v, key) == Some(want), || {
                    format!(
                        "v{i} (nprops={w}) inline property {key} did not survive the \
                         reboot — a stride error reads the neighbouring record"
                    )
                });
            }
            // A record must not report slots it never had: reading past its own
            // extent is how a stride slip presents when it lands short.
            if w < 3 {
                st.ck(g.get_vertex_prop(v, ["t0", "t1", "t2"][w]).is_none(), || {
                    format!("v{i} (nprops={w}) returned a slot beyond its own width")
                });
            }
        }

        for i in (0..n).step_by(97) {
            let v = VertexId(i as u64);
            if i % 7 == 0 {
                st.ck(g.vertex_info(v).is_none(), || {
                    format!("v{i} was deleted before the reboot but came back")
                });
                continue;
            }
            match g.vertex_info(v) {
                Some(info) => {
                    st.ck(info.name == format!("v{i}"), || {
                        format!("v{i} name: got {}", info.name)
                    });
                    st.ck(info.target == ObjID::new(i as u128), || {
                        format!("v{i} target did not survive")
                    });
                }
                None => st.fail(format!("v{i} missing after reboot")),
            }
            if i % 5 == 0 {
                st.ck(
                    g.get_vertex_prop(v, "k") == Some(twizzler_graph::PropValue::I64(i as i64)),
                    || format!("v{i} property did not survive"),
                );
            }
            // Adjacency: i-1 -> i unless either end was deleted.
            if i > 0 && (i - 1) % 7 != 0 {
                let got = g.in_neighbors(v, Labels::any());
                st.ck(got.contains(&VertexId(i as u64 - 1)), || {
                    format!("v{i} lost its inbound edge from v{}", i - 1)
                });
            }
        }

        if st.fails == 0 {
            println!("GSTRESS DURABILITY OK: {n} vertices survived a reboot intact.");
        } else {
            st.report_suppressed();
            println!("GSTRESS DURABILITY FAILED: {} check(s)", st.fails);
            std::process::exit(1);
        }
        return;
    }

    // `gstress destroy <N> <R> [reset]` — R cycles of "build a graph of
    // N vertices, then tear it down". The point is what it does to the store,
    // which is measured on the host between boots (`du` on the disk image), not
    // from in here. `reset` runs the control arm: reset reclaims the outgoing
    // graph but leaves a fresh empty one, so it should grow the image where
    // `destroy` should not.
    if arg1.as_deref() == Some("destroy") {
        let n = std::env::args()
            .nth(2)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(200);
        let rounds = std::env::args()
            .nth(3)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(10);
        let control = std::env::args().nth(4).as_deref() == Some("reset");
        println!(
            "GSTRESS STAMP harness={} mode={} N={} R={}",
            HARNESS_REV,
            if control { "destroy-control-reset" } else { "destroy" },
            n,
            rounds
        );
        let t = Instant::now();
        let mut freed_total = 0usize;
        for r in 1..=rounds {
            Graph::reset_arena("gdestroy", twizzler_graph::DEFAULT_ARENA_CAP)
                .expect("reset arena graph");
            {
                let mut g =
                    Graph::open_or_create_arena("gdestroy", twizzler_graph::DEFAULT_ARENA_CAP)
                        .expect("open arena graph");
                let mut ids = Vec::with_capacity(n);
                for i in 0..n {
                    ids.push(
                        g.add_vertex("d", &format!("d{r}_{i}"), ObjID::new(0))
                            .expect("add_vertex"),
                    );
                }
                for i in 0..n.saturating_sub(1) {
                    g.add_edge(ids[i], "e", ids[i + 1]).expect("add_edge");
                }
                g.sync().expect("sync");
            }
            let freed = if control {
                0
            } else {
                Graph::destroy("gdestroy").expect("destroy")
            };
            freed_total += freed;
            println!(
                "GSTRESS DESTROY: round {r}/{rounds} freed {freed} objects, t={:.1}s",
                t.elapsed().as_secs_f64()
            );
        }
        println!(
            "GSTRESS DESTROY: {rounds} cycles x {n} vertices, {freed_total} objects freed \
             total in {:.1}s. Now measure the disk image on the host — that is the result.",
            t.elapsed().as_secs_f64()
        );
        return;
    }

    let scaled_preset;
    let preset: &Preset = match arg1.as_deref() {
        None | Some("small") => &SMALL,
        Some("tiny") => &TINY,
        Some("medium") => &MEDIUM,
        Some("large") => &LARGE,
        // `scale:N` — arbitrary size with tiny's proportions, for size
        // sweeps: `gstress scale:1000`, `gstress scale:2000 indradb`.
        Some(s) if s.starts_with("scale:") => {
            match s["scale:".len()..].parse::<usize>() {
                Ok(n) => {
                    scaled_preset = scaled(n);
                    &scaled_preset
                }
                Err(_) => {
                    println!("usage: gstress scale:<vertices> [arena:<cap>|indradb]");
                    std::process::exit(2);
                }
            }
        }
        Some(other) => {
            println!(
                "usage: gstress [tiny|small|medium|large|scale:<N>] [arena:<cap>|indradb]  \
                 (got '{other}')"
            );
            println!("       gstress residency [cycle|hold|volatile] [N] [R]   (A6 probe)");
            println!("       gstress props [N] [none]                          (A7-AC1 probe)");
            std::process::exit(2);
        }
    };
    // `arena:<cap>` sets an explicit cap for the sweep; `indradb` runs the
    // same workload against the IndraDB baseline instead. Anything else is
    // the native arena layout at the default cap.
    let mode = std::env::args().nth(2);

    // Explicit cap from `arena:<cap>`; anything else — including no mode
    // argument at all — is the default.
    let arena_cap: usize = mode
        .as_deref()
        .and_then(|m| m.strip_prefix("arena"))
        .and_then(|rest| rest.strip_prefix(':'))
        .and_then(|c| c.parse().ok())
        .unwrap_or(twizzler_graph::DEFAULT_ARENA_CAP);

    if matches!(mode.as_deref(), Some("indradb") | Some("baseline")) {
        stamp("indradb", preset);
        let mut st = Stats::new();
        indradb_mode::run(preset, &mut st);
        if st.fails > 0 {
            println!("GSTRESS: {} verification failure(s)", st.fails);
            std::process::exit(1);
        }
        return;
    }
    // The cap belongs in the stamp: a result that does not say which cap
    // produced it cannot be compared to anything.
    let mode_label = format!("native-arena:{arena_cap}");
    stamp(&mode_label, preset);
    println!(
        "gstress: preset {} ({}) (V={} bulkE={} degCap={} churn={} chain={} clique={})",
        preset.name,
        format!("v4 arena, cap={arena_cap}"),
        preset.vertices,
        preset.bulk_edges,
        preset.degree_cap,
        preset.churn_add,
        preset.chain,
        preset.clique
    );

    let mut st = Stats::new();
    // Two distinct quantities: `vertices` counts vertices created (what the
    // summary reports), `next_id` tracks the largest id handed out (what the
    // monotonicity check compares against). Edges share the id space, so the
    // id high-water runs well ahead of the vertex count.
    let mut counters: (u64, u64) = (0, 0);

    // Setup is timed separately and excluded from the run total, like the
    // final sync below. `reset` destroys the previous run's graph, so on a
    // dirty image it pays one object deletion per object that run created —
    // which at `arena:1` is one per vertex. A cleared image makes this ~0.
    let setup = Instant::now();
    Graph::reset_arena(GRAPH, arena_cap).expect("reset arena graph");
    let mut g = Graph::open_or_create_arena(GRAPH, arena_cap).expect("create arena graph");
            crate::declare_lookup_labels(&mut g);
    let setup_secs = setup.elapsed().as_secs_f64();
    println!("GSTRESS SETUP: reset + open {setup_secs:.2}s (excluded from the run total)");

    // Workload only, from here.
    let total = Instant::now();

    let add_v = |g: &mut Graph,
                 st: &mut Stats,
                 counters: &mut (u64, u64),
                 label: &str,
                 name: &str| {
        let (vertices, next_id) = counters;
        let v = g
            .add_vertex(label, name, ObjID::new(0))
            .expect("add_vertex");
        // Ids are append indices and never reused, but not dense per kind:
        // edge records share the id space, so a vertex created after `k`
        // edges gets `k` higher an id.
        //
        // What is asserted: ids strictly increase and are never handed out
        // twice. That is what callers depend on, and it is what would break
        // if slot reuse ever started recycling ids as well as space.
        if v.0 <= *next_id && *next_id != 0 {
            st.fail(format!(
                "vertex id did not advance: got {}, previous high-water {}",
                v.0, *next_id
            ));
        }
        *next_id = v.0;
        *vertices += 1;
        v
    };

    // --- Phase A: vertex insertion rate, sampled lookups --------------------
    let n = preset.vertices;
    let t = Instant::now();
    for i in 0..n {
        add_v(&mut g, &mut st, &mut counters, "n", &format!("v{i}"));
        if (i + 1) % 500 == 0 {
            heartbeat("A:rollover", i + 1, n, &t);
        }
    }
    // Boundary reads at 0, 4095/4096, and the tail (indices past the preset
    // are skipped).
    for i in [0usize, 4095, 4096, n - 1] {
        if i >= n {
            continue;
        }
        match g.vertex_info(VertexId(i as u64)) {
            Some(info) => st.ck(info.name == format!("v{i}"), || {
                format!("boundary read v{i}: got name {}", info.name)
            }),
            None => st.fail(format!("boundary read v{i}: missing")),
        }
    }
    // Sampled content-keyed lookups.
    let step = (n / 100).max(1);
    for i in (0..n).step_by(step) {
        st.ck(
            g.find_vertex("n", &format!("v{i}")) == Lookup::Found(VertexId(i as u64)),
            || format!("find_vertex v{i} failed"),
        );
    }
    report("A:rollover", n, t);

    // --- Phase B: bulk random edges -----------------------------------------
    let t = Instant::now();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(preset.bulk_edges);
    for k in 0..preset.bulk_edges {
        let u = rng.below(n);
        let v = rng.below(n);
        g.add_edge(VertexId(u as u64), "b", VertexId(v as u64))
            .expect("bulk add_edge");
        pairs.push((u as u32, v as u32));
        if (k + 1) % 500 == 0 {
            heartbeat("B:bulk", k + 1, preset.bulk_edges, &t);
        }
    }
    // Sampled out-degree verification (no deletes yet: expected = raw count).
    for s in 0..100 {
        let u = (s * step) % n;
        let expected = pairs.iter().filter(|(a, _)| *a as usize == u).count();
        let got = g.out_neighbors(VertexId(u as u64), Labels::any()).len();
        st.ck(got == expected, || {
            format!("bulk out-degree of v{u}: got {got}, expected {expected}")
        });
    }
    report("B:bulk", preset.bulk_edges, t);

    // --- Phase C: high-degree hub (adjacency ceiling probe) -----------------
    let t = Instant::now();
    let hub = add_v(&mut g, &mut st, &mut counters, "hub", "hub1");
    let mut hub_deg = 0usize;
    for i in 0..preset.degree_cap {
        let target = VertexId(safe_target(i, n) as u64);
        if i > 0 && i % 1000 == 0 {
            heartbeat("C:degree", i, preset.degree_cap, &t);
        }
        match g.add_edge(hub, "h", target) {
            Ok(_) => hub_deg += 1,
            Err(e) => {
                println!(
                    "GSTRESS FINDING: adjacency ceiling — hub add_edge #{} failed: {e} \
                     (A2 evidence; record on the board)",
                    i + 1
                );
                break;
            }
        }
    }
    let got = g.out_neighbors(hub, Labels::any()).len();
    st.ck(got == hub_deg, || {
        format!("hub out-degree: got {got}, expected {hub_deg}")
    });
    let mut hub2_deg = 0usize;
    if preset.same_target_variant {
        // Same-target parallel edges: if FOT entries dedup per target object,
        // this should reach a higher ceiling than the distinct-target hub.
        let hub2 = add_v(&mut g, &mut st, &mut counters, "hub", "hub2");
        let target = VertexId(1); // index 1 survives churn
        for i in 0..preset.degree_cap {
            if i > 0 && i % 1000 == 0 {
                heartbeat("C:degree2", i, preset.degree_cap, &t);
            }
            match g.add_edge(hub2, "h2", target) {
                Ok(_) => hub2_deg += 1,
                Err(e) => {
                    println!(
                        "GSTRESS FINDING: same-target ceiling — hub2 add_edge #{} failed: {e} \
                         (compare with hub1; informs FOT-dedup question, I0)",
                        i + 1
                    );
                    break;
                }
            }
        }
        let got2 = g.out_neighbors(hub2, Labels::any()).len();
        st.ck(got2 == hub2_deg, || {
            format!("hub2 out-degree: got {got2}, expected {hub2_deg}")
        });
    }
    report("C:degree", hub_deg + hub2_deg, t);

    // --- Phase D: churn + full verification scan ----------------------------
    let t = Instant::now();
    let mut deleted = vec![false; n];
    let mut ndel = 0usize;
    // One delete probed in isolation before the loop. Record and mirror
    // should agree.
    g.delete_vertex(VertexId(0)).expect("delete_vertex");
    if let Some(d) = g.debug_liveness(VertexId(0)) {
        println!("GSTRESS PROBE: immediately after deleting v0: {d}");
    }
    for i in (0..n).step_by(7) {
        g.delete_vertex(VertexId(i as u64)).expect("delete_vertex");
        deleted[i] = true;
        ndel += 1;
        if ndel % 200 == 0 {
            heartbeat("D:churn", ndel, n / 7 + 1, &t);
        }
    }
    // Liveness probes. Two controls first — v0 is deleted (0 % 7 == 0), v1 is
    // not — so the record-vs-mirror state is on the console even in a run
    // where the scan passes.
    for (v, expect) in [(0u64, "deleted"), (1u64, "live")] {
        if let Some(d) = g.debug_liveness(VertexId(v)) {
            println!("GSTRESS PROBE: control ({expect}) {d}");
        }
    }
    // Full scan: every phase-A id reads back consistent with the bookkeeping.
    let mut probes_left = 3;
    for i in 0..n {
        let alive = g.vertex_info(VertexId(i as u64)).is_some();
        if alive != !deleted[i] && probes_left > 0 {
            probes_left -= 1;
            if let Some(d) = g.debug_liveness(VertexId(i as u64)) {
                println!("GSTRESS PROBE: mismatch (deleted={}) {d}", deleted[i]);
            }
        }
        st.ck(alive == !deleted[i], || {
            format!("churn scan v{i}: alive={alive}, expected {}", !deleted[i])
        });
    }
    // Sampled degree re-verification: edges to/from deleted vertices hide.
    for s in 0..50 {
        let u = (s * step) % n;
        if deleted[u] {
            continue;
        }
        let expected = pairs
            .iter()
            .filter(|(a, b)| *a as usize == u && !deleted[*b as usize])
            .count();
        let got = g
            .out_neighbors(VertexId(u as u64), Labels::these(&["b"]))
            .len();
        st.ck(got == expected, || {
            format!("post-churn out-degree of v{u}: got {got}, expected {expected}")
        });
    }
    // Adds after deletes: ids continue, never reuse.
    for i in 0..preset.churn_add {
        add_v(&mut g, &mut st, &mut counters, "c", &format!("c{i}"));
    }
    report("D:churn", n + preset.churn_add, t);

    // --- Phase E: reopen by name, re-verify by sampling ---------------------
    let t = Instant::now();
    // Sync before dropping. Nothing is durable until `sync()`, so dropping an
    // unsynced graph throws writes away.
    g.sync().expect("sync before reopen");
    drop(g);
    // Reopen with the same cap. The cap is a runtime argument, not persisted
    // in `GraphRoot`, so a plain `open_or_create` would reconstruct the
    // placement policy as `FillTo { cap: DEFAULT_ARENA_CAP }` and a run
    // started at another cap would silently roll over at the default from
    // here on.
    let mut g = Graph::open_or_create_arena(GRAPH, arena_cap).expect("reopen arena graph");
            crate::declare_lookup_labels(&mut g);
    st.ck(g.vertex_info(VertexId(7)).is_none(), || {
        "reopen: deleted v7 came back".into()
    });
    st.ck(g.find_vertex("n", "v8") == Lookup::Found(VertexId(8)), || {
        "reopen: v8 lookup failed".into()
    });
    if n > 4096 {
        st.ck(
            g.vertex_info(VertexId(4096)).map(|i| i.name) == Some("v4096".into()),
            || "reopen: boundary v4096 failed".into(),
        );
    }
    let got = g.out_neighbors(hub, Labels::these(&["h"])).len();
    st.ck(got == hub_deg, || {
        format!("reopen: hub degree got {got}, expected {hub_deg}")
    });
    st.ck(
        g.find_vertex("c", &format!("c{}", preset.churn_add - 1))
            .is_found(),
        || "reopen: last churn vertex missing".into(),
    );
    report("E:reopen", 5, t);

    // --- Phase F: pathological shapes ----------------------------------------
    // Deep chain, walked end-to-end.
    let t = Instant::now();
    let head = add_v(&mut g, &mut st, &mut counters, "ch", "ch0");
    let mut prev = head;
    for i in 1..preset.chain {
        let v = add_v(&mut g, &mut st, &mut counters, "ch", &format!("ch{i}"));
        g.add_edge(prev, "next", v).expect("chain add_edge");
        prev = v;
        if (i + 1) % 500 == 0 {
            heartbeat("F:chain", i + 1, preset.chain, &t);
        }
    }
    let mut cur = head;
    let mut hops = 0usize;
    loop {
        // Safety bound: a cycle would otherwise walk forever.
        if hops > preset.chain {
            st.fail(format!(
                "chain walk exceeded {} hops — cycle?",
                preset.chain
            ));
            break;
        }
        let next = g.out_neighbors(cur, Labels::these(&["next"]));
        match next.len() {
            0 => break,
            1 => {
                cur = next[0];
                hops += 1;
            }
            k => {
                st.fail(format!("chain fan-out {k} at hop {hops}"));
                break;
            }
        }
    }
    st.ck(hops == preset.chain - 1, || {
        format!("chain walk: {hops} hops, expected {}", preset.chain - 1)
    });
    st.ck(cur == prev, || {
        "chain walk ended at the wrong vertex".into()
    });
    report("F:chain", preset.chain, t);

    // Dense clique: k vertices, k*(k-1) directed edges.
    let t = Instant::now();
    let k = preset.clique;
    let mut cl = Vec::with_capacity(k);
    for i in 0..k {
        cl.push(add_v(
            &mut g,
            &mut st,
            &mut counters,
            "cl",
            &format!("cl{i}"),
        ));
    }
    for i in 0..k {
        for j in 0..k {
            if i != j {
                g.add_edge(cl[i], "k", cl[j]).expect("clique add_edge");
            }
        }
        // Reports every 10 rows, so a long clique phase is distinguishable
        // from a hang.
        if (i + 1) % 10 == 0 {
            heartbeat("F:clique", (i + 1) * (k - 1), k * (k - 1), &t);
        }
    }
    for &i in &[0usize, k / 2, k - 1] {
        let out = g.out_neighbors(cl[i], Labels::these(&["k"])).len();
        let inn = g.in_neighbors(cl[i], Labels::these(&["k"])).len();
        let both = g.both_neighbors(cl[i], Labels::these(&["k"])).len();
        st.ck(out == k - 1, || {
            format!("clique cl{i} out {out} != {}", k - 1)
        });
        st.ck(inn == k - 1, || {
            format!("clique cl{i} in {inn} != {}", k - 1)
        });
        st.ck(both == 2 * (k - 1), || {
            format!("clique cl{i} both {both} != {}", 2 * (k - 1))
        });
    }
    report("F:clique", k * (k - 1), t);

    // --- Phase H: reads -----------------------------------------------------
    //
    // Everything above measures writes. This phase runs on the graph built by
    // A–F, before G adds more vertices.
    //
    // `lookup` uses each engine's native key path (ours built-in, the
    // baseline's a property index), while `1hop`/`2hop` start from ids
    // already in hand, isolating traversal from lookup.
    let rstep = read_step(n);
    let reps = preset.read_reps;
    let read_idx: Vec<usize> = (0..n).step_by(rstep).filter(|i| !deleted[*i]).collect();
    let read_sample: Vec<VertexId> = read_idx.iter().map(|i| VertexId(*i as u64)).collect();
    println!(
        "GSTRESS READS: {} sampled vertices x {} reps",
        read_sample.len(),
        reps
    );

    let max_reps = reps * 100;
    measure_read("H:lookup", max_reps, || {
        let mut found = 0usize;
        for i in &read_idx {
            if g.find_vertex("n", &format!("v{i}")).is_found() {
                found += 1;
            }
        }
        debug_assert_eq!(found, read_idx.len());
        read_idx.len()
    });
    measure_read("H:1hop", max_reps, || {
        let mut seen = 0usize;
        for v in &read_sample {
            seen += g.out_neighbors(*v, Labels::any()).len();
        }
        let _ = seen;
        read_sample.len()
    });
    measure_read("H:2hop", max_reps, || {
        let mut seen = 0usize;
        for v in &read_sample {
            for n1 in g.out_neighbors(*v, Labels::any()) {
                seen += g.out_neighbors(n1, Labels::any()).len();
            }
        }
        let _ = seen;
        read_sample.len()
    });
    measure_read("H:scan", max_reps, || g.vertices().len());

    // --- Phase G: degradation probe -----------------------------------------
    //
    // A deliberately flat workload: identical small batches of vertex
    // creations, repeated, reporting the rate for each window. The workload
    // does not change, so any decline across windows is the system degrading
    // as writes accumulate. Runs last so it measures the system at its most
    // loaded, and the first/last ratio is printed as `GSTRESS DEGRADE`.
    let t = Instant::now();
    let mut prog = Progress::new("G:degrade");
    let gtotal = preset.degrade_windows * preset.degrade_batch;
    let mut gdone = 0usize;
    for w in 0..preset.degrade_windows {
        let base = gdone;
        // The arena layout batches by construction, so the direct path here
        // is the like-for-like comparison.
        for j in 0..preset.degrade_batch {
            add_v(&mut g, &mut st, &mut counters, "g", &format!("g{}_{}", w, j));
        }
        gdone = base + preset.degrade_batch;
        prog.tick(gdone, gtotal);
    }
    prog.summarize();
    report("G:degrade", gtotal, t);

    // --- Summary --------------------------------------------------------------
    // Durability barrier, timed separately. Writes live in mapped memory
    // until `sync()`, so a run that skipped this would be timing an in-memory
    // workload and reporting it as a database. Timing it apart from the
    // workload keeps the deferred cost visible instead of hidden.
    let sync_t = Instant::now();
    if let Err(e) = g.sync() {
        println!("GSTRESS: final sync failed: {e:?}");
        st.fails += 1;
    }
    let sync_secs = sync_t.elapsed().as_secs_f64();

    let secs = total.elapsed().as_secs_f64();
    println!(
        "GSTRESS {}: preset {} — {} vertices, hub degree {} (+{} same-target), {:.1}s total \
         ({:.2}s of it the final sync)",
        if st.fails == 0 { "OK" } else { "FAILED" },
        preset.name,
        counters.0,
        hub_deg,
        hub2_deg,
        secs,
        sync_secs
    );
    // Object count is the packing argument, so a result without it cannot be
    // interpreted. The v3 figure in the line below is a stated baseline, not
    // a measurement of this run. `records` is what the arena holds — vertices
    // and edges — so objects-per-record is the honest ratio.
    let records = g.record_count() as u64;
    println!(
        "GSTRESS ARENA: {} arenas for {} records ({} vertices + {} edges), \
         {:.5} objects/record, {} syncs; v3 would have spent ~{} objects \
         (3/vertex + 1/edge)",
        g.arena_count(),
        records,
        counters.0,
        records.saturating_sub(counters.0),
        g.arena_count() as f64 / records.max(1) as f64,
        g.arena_sync_count(),
        counters.0 * 3 + records.saturating_sub(counters.0)
    );
    // `place` decides rollover from the policy view alone, so if the two rows
    // below disagree, cap is not controlling rollover.
    let (policy, actual) = g.arena_vertex_counts();
    println!("GSTRESS ARENA DIST policy: {policy:?}");
    println!("GSTRESS ARENA DIST locs:   {actual:?}");
    if policy != actual {
        let (ps, as_): (usize, usize) = (policy.iter().sum(), actual.iter().sum());
        println!(
            "GSTRESS ARENA DIST MISMATCH: the placement policy and the location \
             registry disagree (totals {ps} vs {as_}) — cap is not controlling rollover"
        );
    }
    if st.fails > 0 {
        st.report_suppressed();
        println!("GSTRESS: {} verification failure(s)", st.fails);
        std::process::exit(1);
    }
}
