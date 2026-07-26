//! Deliberately NOT part of `cargo start-qemu --tests`, so the default
//! harness stays fast. Usage, from the Twizzler shell:
//!
//! The graph is registered as `data/gstress` and reset at startup, so runs
//! are idempotent (old registries are orphaned, as with `Graph::reset`).

use std::time::Instant;

use twizzler::object::ObjID;
use twizzler_graph::{Graph, Labels, VertexId};

const GRAPH: &str = "gstress";

mod indradb_mode;

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

pub(crate) struct Stats {
    pub(crate) fails: u64,
}
impl Stats {
    pub(crate) fn fail(&mut self, msg: String) {
        println!("GSTRESS FAIL: {msg}");
        self.fails += 1;
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

pub(crate) fn heartbeat(phase: &str, done: usize, total: usize, t: &Instant) {
    let secs = t.elapsed().as_secs_f64();
    let rate = if secs > 0.0 { done as f64 / secs } else { 0.0 };
    println!("GSTRESS {phase}: {done}/{total} ({rate:.0} ops/s)");
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
    let preset = match std::env::args().nth(1).as_deref() {
        None | Some("small") => &SMALL,
        Some("tiny") => &TINY,
        Some("medium") => &MEDIUM,
        Some("large") => &LARGE,
        Some(other) => {
            println!(
                "usage: gstress [tiny|small|medium|large] [nobulk|indradb]  (got '{other}')"
            );
            std::process::exit(2);
        }
    };
    let mode = std::env::args().nth(2);
    let use_bulk = mode.as_deref() != Some("nobulk");
    const CHUNK: usize = 500;

    if matches!(mode.as_deref(), Some("indradb") | Some("baseline")) {
        let mut st = Stats { fails: 0 };
        indradb_mode::run(preset, &mut st);
        if st.fails > 0 {
            println!("GSTRESS: {} verification failure(s)", st.fails);
            std::process::exit(1);
        }
        return;
    }
    println!(
        "gstress: preset {} ({}) (V={} bulkE={} degCap={} churn={} chain={} clique={})",
        preset.name,
        if use_bulk { "bulk" } else { "nobulk" },
        preset.vertices,
        preset.bulk_edges,
        preset.degree_cap,
        preset.churn_add,
        preset.chain,
        preset.clique
    );

    let total = Instant::now();
    let mut st = Stats { fails: 0 };
    let mut next_id: u64 = 0;

    Graph::reset(GRAPH).expect("reset gstress graph");
    let mut g = Graph::open_or_create(GRAPH).expect("create gstress graph");

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
    for i in (0..n).step_by(7) {
        g.delete_vertex(VertexId(i as u64)).expect("delete_vertex");
        deleted[i] = true;
        ndel += 1;
        if ndel % 200 == 0 {
            heartbeat("D:churn", ndel, n / 7 + 1, &t);
        }
    }
    // Full scan: every phase-A id reads back consistent with the bookkeeping.
    for i in 0..n {
        let alive = g.vertex_info(VertexId(i as u64)).is_some();
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
        gdone = base + preset.degrade_batch;
        prog.tick(gdone, gtotal);
    }
    prog.summarize();
    report("G:degrade", gtotal, t);

    // --- Summary --------------------------------------------------------------
    let secs = total.elapsed().as_secs_f64();
    println!(
        "GSTRESS {}: preset {} — {} vertices, hub degree {} (+{} same-target), {:.1}s total",
        if st.fails == 0 { "OK" } else { "FAILED" },
        preset.name,
        next_id,
        hub_deg,
        hub2_deg,
        secs
    );
    if st.fails > 0 {
        println!("GSTRESS: {} verification failure(s)", st.fails);
        std::process::exit(1);
    }
}
