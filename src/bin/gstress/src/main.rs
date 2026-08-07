//! Deliberately NOT part of `cargo start-qemu --tests`, so the default
//! harness stays fast. Usage, from the Twizzler shell:
//!
//! Clear `target/disk-<triple>.img` before any recorded measurement. It is
//! created only if absent and nothing ever deletes an object, so it carries
//! every graph every previous run made; a dirty image is not comparable to a
//! clean one.
//!
//! The graph is registered as `data/gstress` and reset at startup, so runs
//! are idempotent (old registries are orphaned, as with `Graph::reset`).

use std::time::Instant;

use twizzler::object::ObjID;
use twizzler_graph::{Graph, Labels, VertexId};

const GRAPH: &str = "gstress";

pub(crate) const HARNESS_REV: &str = "2026-08-04a";

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

pub(crate) struct Preset {
    pub(crate) name: &'static str,
    /// Phase A vertices (must exceed DEFAULT_SEG_CAP = 4096 to force rollover).
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
    /// Phase G: records per window (kept small and *constant*, so any change
    /// in window rate is the system degrading, not the workload changing).
    pub(crate) degrade_batch: usize,
    /// Phase H: how many times to repeat each read workload. Reads are fast
    /// enough that a single pass lands at or below timer resolution — the
    /// first H run reported 0.00 s for lookup and scan, making their rates
    /// meaningless. Repetition moves the measurement above the noise floor.
    pub(crate) read_reps: usize,
}

/// Build a preset of arbitrary size, with every sub-workload scaled in the
/// same proportions as `tiny` (which `scaled(200)` reproduces).
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
/// the same vertices. (They did not in the first H run: 85 samples natively
/// against 17 in the baseline, because each arm computed its own stride.)
pub(crate) fn read_step(n: usize) -> usize {
    (n / 100).max(1)
}

/// Warm-phase target duration. A *fixed* rep count cannot serve both arms:
/// the native engine's warm reads are ~24× the baseline's, so a count giving
/// the baseline a sane runtime leaves the native measurement at 0.02 s — at
/// timer resolution — while a count that measures the native arm properly
/// would run the baseline for minutes. Each arm therefore repeats until it
/// reaches this duration; comparing *rates* stays valid because the rate is
/// per-op.
const READ_TARGET_SECS: f64 = 1.0;

/// Measure a read workload in two regimes, reporting both.
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
/// A *systematic* failure emits one line per checked item, and the checks run
/// over every vertex. The `scale:20000` churn scan produced 2 858 of them and
/// the run died inside `println!` itself — "I/O error: data loss", the serial
/// console dropping writes — which took every phase after D with it. The
/// failure count is still reported in full; only the per-item lines are
/// capped, because a dozen of them already show the pattern and the rest costs
/// the remainder of the run.
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
/// Windowed progress: reports the rate for *this window* alongside the
/// cumulative rate.
///
/// Cumulative rates structurally hide decay — a rate that halves partway
/// through shows up as a gentle droop — which is why the phase heartbeats
/// could not answer whether throughput degrades *within* a run. Window rates
/// show it directly.
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

/// Progress line, with a projected time to finish the phase.
///
/// It is a projection at the *current cumulative* rate, so a phase whose rate
/// decays will overshoot it — the baseline's do, badly: `B:bulk` fell 26 → 13
/// ops/s within one phase. Read it as a floor, not a promise.
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
/// This must produce *distinct* targets, and the reason is a real semantic
/// difference between the engines rather than tidiness: our engine is a
/// multigraph (parallel edges are distinct records — see the engine's
/// `parallel_edges` test), while IndraDB keys an edge on
/// `(outbound, type, inbound)` and silently rejects duplicates. The previous
/// version mapped both `i=0` and `i=1` to vertex 1 (and six more collisions),
/// so a 50-edge hub phase stored 50 edges natively but only 43 in the
/// baseline — the arms were doing unequal work and the comparison was biased
/// against the native engine.
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
            let mut ids = Vec::with_capacity(n);
            for i in 0..n {
                ids.push(
                    g.add_vertex("d", &format!("v{i}"), ObjID::new(i as u128))
                        .expect("add_vertex"),
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
            g.sync().expect("sync");

            // Reopen mid-seed, then keep writing. This is the shape that
            // lost the arena directory: a store is dropped and reopened, and
            // the *reopened* instance grows the graph. Writes made through the
            // second instance have to be durable even though the first
            // instance created the structures they live in. Without this, the
            // check passes on a store whose registries were only ever written
            // by one instance — which is what let the bug through.
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
            g.sync().expect("sync after reopen");
            println!(
                "GSTRESS SEED: {} vertices ({} arenas), every 5th has a property, \
                 every 7th deleted, {} more added through a reopened handle. \
                 Now reboot and run `gstress verify {n}`.",
                ids.len(),
                g.arena_count(),
                n / 4
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
        st.ck(g.is_arena(), || "seeded graph did not come back as v4".into());
        // The `w*` vertices were written through a reopened handle; if the
        // registries created by the first handle did not reach disk, these are
        // the ones that vanish.
        let post = n / 4;
        for i in (0..post).step_by(23) {
            let v = VertexId((n + i) as u64);
            st.ck(
                g.vertex_info(v).map(|inf| inf.name) == Some(format!("w{i}")),
                || format!("w{i} (written after a mid-seed reopen) did not survive"),
            );
        }
        let expect_live = (0..n).filter(|i| i % 7 != 0).count() + post;
        let live = g.vertices().len();
        st.ck(live == expect_live, || {
            format!("live vertices: got {live}, expected {expect_live}")
        });
        println!("GSTRESS VERIFY: {} arenas recovered", g.arena_count());
        st.ck(g.arena_count() > 0, || {
            "arena directory came back empty — the store's arenas did not reach disk".into()
        });

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
        Some(s) if s.starts_with("scale:") => {
            match s["scale:".len()..].parse::<usize>() {
                Ok(n) => {
                    scaled_preset = scaled(n);
                    &scaled_preset
                }
                Err(_) => {
                    println!("usage: gstress scale:<vertices> [nobulk|indradb]");
                    std::process::exit(2);
                }
            }
        }
        Some(other) => {
            println!(
                "usage: gstress [tiny|small|medium|large|scale:<N>] [nobulk|indradb]  \
                 (got '{other}')"
            );
            println!("       gstress residency [cycle|hold|volatile] [N] [R]   (A6 probe)");
            std::process::exit(2);
        }
    };
    let mode = std::env::args().nth(2);

    let arena_cap: Option<usize> = mode.as_deref().and_then(|m| {
        let rest = m.strip_prefix("arena")?;
        Some(match rest.strip_prefix(':') {
            Some(c) => c.parse().unwrap_or(twizzler_graph::DEFAULT_ARENA_CAP),
            None => twizzler_graph::DEFAULT_ARENA_CAP,
        })
    });

    let use_bulk = mode.as_deref() != Some("nobulk") && arena_cap.is_none();
    const CHUNK: usize = 500;

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
    // The layout and the batching mode both belong in the stamp: a result that
    // does not say which layout produced it cannot be compared to anything.
    let mode_label = match arena_cap {
        Some(c) => format!("native-arena:{c}"),
        None if use_bulk => "native-bulk".to_string(),
        None => "native-nobulk".to_string(),
    };
    stamp(&mode_label, preset);
    println!(
        "gstress: preset {} ({}) (V={} bulkE={} degCap={} churn={} chain={} clique={})",
        preset.name,
        match arena_cap {
            Some(c) => format!("v4 arena, cap={c}"),
            None if use_bulk => "v3 bulk".to_string(),
            None => "v3 nobulk".to_string(),
        },
        preset.vertices,
        preset.bulk_edges,
        preset.degree_cap,
        preset.churn_add,
        preset.chain,
        preset.clique
    );

    let total = Instant::now();
    let mut st = Stats::new();
    let mut next_id: u64 = 0;

    let mut g = match arena_cap {
        Some(cap) => {
            Graph::reset_arena(GRAPH, cap).expect("reset arena graph");
            Graph::open_or_create_arena(GRAPH, cap).expect("create arena graph")
        }
        None => {
            Graph::reset(GRAPH).expect("reset gstress graph");
            Graph::open_or_create(GRAPH).expect("create gstress graph")
        }
    };

    let add_v = |g: &mut Graph, st: &mut Stats, next_id: &mut u64, label: &str, name: &str| {
        let v = g
            .add_vertex(label, name, ObjID::new(0))
            .expect("add_vertex");
        // Ids are append indices and never reused; any drift is a bug.
        if v.0 != *next_id {
            st.fail(format!(
                "vertex id drift: got {}, expected {}",
                v.0, *next_id
            ));
        }
        *next_id += 1;
        v
    };

    // --- Phase A: registry rollover at the real DEFAULT_SEG_CAP -------------
    let n = preset.vertices;
    let t = Instant::now();
    if use_bulk {
        let mut i = 0;
        while i < n {
            let hi = (i + CHUNK).min(n);
            g.bulk(|b| {
                for j in i..hi {
                    let v = b.add_vertex("n", &format!("v{j}"), ObjID::new(0))?;
                    if v.0 != next_id {
                        st.fail(format!(
                            "vertex id drift: got {}, expected {}",
                            v.0, next_id
                        ));
                    }
                    next_id += 1;
                }
                Ok(())
            })
            .expect("bulk add_vertex");
            heartbeat("A:rollover", hi, n, &t);
            i = hi;
        }
    } else {
        for i in 0..n {
            add_v(&mut g, &mut st, &mut next_id, "n", &format!("v{i}"));
            if (i + 1) % 500 == 0 {
                heartbeat("A:rollover", i + 1, n, &t);
            }
        }
    }
    // Boundary reads around the default segment capacity (4096) and the tail
    // (skipped when the preset is smaller than the boundary).
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
            g.find_vertex("n", &format!("v{i}")) == Some(VertexId(i as u64)),
            || format!("find_vertex v{i} failed"),
        );
    }
    report("A:rollover", n, t);

    // --- Phase B: bulk random edges -----------------------------------------
    let t = Instant::now();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(preset.bulk_edges);
    if use_bulk {
        let mut k = 0;
        while k < preset.bulk_edges {
            let hi = (k + CHUNK).min(preset.bulk_edges);
            g.bulk(|b| {
                for _ in k..hi {
                    let u = rng.below(n);
                    let v = rng.below(n);
                    b.add_edge(VertexId(u as u64), "b", VertexId(v as u64))?;
                    pairs.push((u as u32, v as u32));
                }
                Ok(())
            })
            .expect("bulk add_edge");
            heartbeat("B:bulk", hi, preset.bulk_edges, &t);
            k = hi;
        }
    } else {
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
    let hub = add_v(&mut g, &mut st, &mut next_id, "hub", "hub1");
    let mut hub_deg = 0usize;
    if use_bulk {
        let mut i = 0;
        while i < preset.degree_cap {
            let hi = (i + CHUNK).min(preset.degree_cap);
            let r = g.bulk(|b| {
                for j in i..hi {
                    let target = VertexId(safe_target(j, n) as u64);
                    b.add_edge(hub, "h", target)?;
                    hub_deg += 1;
                }
                Ok(())
            });
            if let Err(e) = r {
                println!(
                    "GSTRESS FINDING: adjacency ceiling — hub add_edge #{} failed: {e} \
                     (A2 evidence; record on the board)",
                    hub_deg + 1
                );
                break;
            }
            heartbeat("C:degree", hi, preset.degree_cap, &t);
            i = hi;
        }
    } else {
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
    }
    let got = g.out_neighbors(hub, Labels::any()).len();
    st.ck(got == hub_deg, || {
        format!("hub out-degree: got {got}, expected {hub_deg}")
    });
    let mut hub2_deg = 0usize;
    if preset.same_target_variant {
        // Same-target parallel edges: if FOT entries dedup per target object,
        // this should reach a higher ceiling than the distinct-target hub (I0).
        let hub2 = add_v(&mut g, &mut st, &mut next_id, "hub", "hub2");
        let target = VertexId(1); // index 1 survives churn
        if use_bulk {
            let mut i = 0;
            while i < preset.degree_cap {
                let hi = (i + CHUNK).min(preset.degree_cap);
                let r = g.bulk(|b| {
                    for _ in i..hi {
                        b.add_edge(hub2, "h2", target)?;
                        hub2_deg += 1;
                    }
                    Ok(())
                });
                if let Err(e) = r {
                    println!(
                        "GSTRESS FINDING: same-target ceiling — hub2 add_edge #{} failed: {e} \
                         (compare with hub1; informs FOT-dedup question, I0)",
                        hub2_deg + 1
                    );
                    break;
                }
                heartbeat("C:degree2", hi, preset.degree_cap, &t);
                i = hi;
            }
        } else {
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
    // One delete probed in isolation before the loop. Record and mirror should
    // now agree; a run where they don't is the mapping split returning.
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
    if use_bulk {
        let mut i = 0;
        while i < preset.churn_add {
            let hi = (i + CHUNK).min(preset.churn_add);
            g.bulk(|b| {
                for j in i..hi {
                    let v = b.add_vertex("c", &format!("c{j}"), ObjID::new(0))?;
                    if v.0 != next_id {
                        st.fail(format!(
                            "vertex id drift: got {}, expected {}",
                            v.0, next_id
                        ));
                    }
                    next_id += 1;
                }
                Ok(())
            })
            .expect("bulk churn add");
            i = hi;
        }
    } else {
        for i in 0..preset.churn_add {
            add_v(&mut g, &mut st, &mut next_id, "c", &format!("c{i}"));
        }
    }
    report("D:churn", n + preset.churn_add, t);

    // --- Phase E: reopen by name, re-verify by sampling ---------------------
    let t = Instant::now();
    // Sync before dropping. On v4 nothing is durable until `sync()`, so a bare
    // drop discards the batch — and worse, it used to strand the registries'
    // dirty flags, which is how the arena directory came back empty on the
    // next boot. The library no longer depends on that (see `SegVec::flush`),
    // but dropping an unsynced graph is still throwing writes away.
    g.sync().expect("sync before reopen");
    drop(g);
    let mut g = Graph::open_or_create(GRAPH).expect("reopen gstress graph");
    st.ck(g.vertex_info(VertexId(7)).is_none(), || {
        "reopen: deleted v7 came back".into()
    });
    st.ck(g.find_vertex("n", "v8") == Some(VertexId(8)), || {
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
            .is_some(),
        || "reopen: last churn vertex missing".into(),
    );
    report("E:reopen", 5, t);

    let t = Instant::now();
    let head = add_v(&mut g, &mut st, &mut next_id, "ch", "ch0");
    let mut prev = head;
    if use_bulk {
        let mut i = 1;
        while i < preset.chain {
            let hi = (i + CHUNK).min(preset.chain);
            g.bulk(|b| {
                for j in i..hi {
                    let v = b.add_vertex("ch", &format!("ch{j}"), ObjID::new(0))?;
                    if v.0 != next_id {
                        st.fail(format!(
                            "vertex id drift: got {}, expected {}",
                            v.0, next_id
                        ));
                    }
                    next_id += 1;
                    b.add_edge(prev, "next", v)?;
                    prev = v;
                }
                Ok(())
            })
            .expect("bulk chain");
            heartbeat("F:chain", hi, preset.chain, &t);
            i = hi;
        }
    } else {
        for i in 1..preset.chain {
            let v = add_v(&mut g, &mut st, &mut next_id, "ch", &format!("ch{i}"));
            g.add_edge(prev, "next", v).expect("chain add_edge");
            prev = v;
            if (i + 1) % 500 == 0 {
                heartbeat("F:chain", i + 1, preset.chain, &t);
            }
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
    if use_bulk {
        g.bulk(|b| {
            for i in 0..k {
                let v = b.add_vertex("cl", &format!("cl{i}"), ObjID::new(0))?;
                if v.0 != next_id {
                    st.fail(format!(
                        "vertex id drift: got {}, expected {}",
                        v.0, next_id
                    ));
                }
                next_id += 1;
                cl.push(v);
            }
            Ok(())
        })
        .expect("bulk clique vertices");
        // One batch per source row keeps chunks bounded (k-1 edges each).
        for i in 0..k {
            g.bulk(|b| {
                for j in 0..k {
                    if i != j {
                        b.add_edge(cl[i], "k", cl[j])?;
                    }
                }
                Ok(())
            })
            .expect("bulk clique edges");
            if (i + 1) % 10 == 0 {
                heartbeat("F:clique", (i + 1) * (k - 1), k * (k - 1), &t);
            }
        }
    } else {
        for i in 0..k {
            cl.push(add_v(
                &mut g,
                &mut st,
                &mut next_id,
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
            // The bulk path above reports every 10 rows; this one reported
            // nothing at all, which is how a 215 s phase looked like a hang.
            if (i + 1) % 10 == 0 {
                heartbeat("F:clique", (i + 1) * (k - 1), k * (k - 1), &t);
            }
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

    // Read on its own terms: `lookup` uses each engine's native key path
    // (ours built-in, the baseline's a property index), while `1hop`/`2hop`
    // start from ids already in hand, isolating traversal from lookup.
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
            if g.find_vertex("n", &format!("v{i}")).is_some() {
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

    // A deliberately *flat* workload: identical small batches of vertex
    // creations, repeated, reporting the rate for each window. The workload
    // does not change, so any decline across windows is the system degrading
    // as writes accumulate — the hypothesis that a run pays its own growing
    // pager-backlog tax. Runs last so it measures the system at its most
    // loaded, and the first/last ratio is printed as `GSTRESS DEGRADE`.
    let t = Instant::now();
    let mut prog = Progress::new("G:degrade");
    let gtotal = preset.degrade_windows * preset.degrade_batch;
    let mut gdone = 0usize;
    for w in 0..preset.degrade_windows {
        let base = gdone;
        // `bulk` is a v3 construct and now refuses on v4 (see `Graph::bulk`).
        // The arena layout batches by construction, so the direct path here is
        // the like-for-like comparison, not a slower one.
        if use_bulk {
            g.bulk(|b| {
                for j in 0..preset.degrade_batch {
                    let v = b.add_vertex("g", &format!("g{}_{}", w, j), ObjID::new(0))?;
                    if v.0 != next_id {
                        st.fail(format!("vertex id drift: got {}, expected {}", v.0, next_id));
                    }
                    next_id += 1;
                }
                Ok(())
            })
            .expect("degrade probe batch");
        } else {
            for j in 0..preset.degrade_batch {
                add_v(&mut g, &mut st, &mut next_id, "g", &format!("g{}_{}", w, j));
            }
        }
        gdone = base + preset.degrade_batch;
        prog.tick(gdone, gtotal);
    }
    prog.summarize();
    report("G:degrade", gtotal, t);

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
        next_id,
        hub_deg,
        hub2_deg,
        secs,
        sync_secs
    );
    if g.is_arena() {
        println!(
            "GSTRESS ARENA: {} arenas for {} vertices ({:.4} objects/vertex), {} syncs; \
             v3 would have spent ~{} objects (3/vertex + 1/edge)",
            g.arena_count(),
            next_id,
            g.arena_count() as f64 / next_id.max(1) as f64,
            g.arena_sync_count(),
            next_id * 3 + preset.bulk_edges as u64
        );
    }
    if st.fails > 0 {
        st.report_suppressed();
        println!("GSTRESS: {} verification failure(s)", st.fails);
        std::process::exit(1);
    }
}
