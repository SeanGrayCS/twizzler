//! Deliberately NOT part of `cargo start-qemu --tests`, so the default
//! harness stays fast. Usage, from the Twizzler shell:
//!
//! The graph is registered as `data/gstress` and reset at startup, so runs
//! are idempotent (old registries are orphaned, as with `Graph::reset`).

use std::time::Instant;

use twizzler::object::ObjID;
use twizzler_graph::{Graph, Labels, VertexId};

const GRAPH: &str = "gstress";

struct Preset {
    name: &'static str,
    /// Phase A vertices (must exceed DEFAULT_SEG_CAP = 4096 to force rollover).
    vertices: usize,
    /// Phase B random edges.
    bulk_edges: usize,
    /// Phase C hub out-degree cap ("until failure or this").
    degree_cap: usize,
    /// Whether phase C also runs the same-target variant (FOT-dedup probe).
    same_target_variant: bool,
    /// Phase D vertices added after the deletes.
    churn_add: usize,
    /// Phase F chain length (walked end-to-end).
    chain: usize,
    /// Phase F clique size (k vertices, k*(k-1) directed edges).
    clique: usize,
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
};

/// Deterministic xorshift64 so runs are reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

struct Stats {
    fails: u64,
}
impl Stats {
    fn fail(&mut self, msg: String) {
        println!("GSTRESS FAIL: {msg}");
        self.fails += 1;
    }
    fn ck(&mut self, cond: bool, msg: impl FnOnce() -> String) {
        if !cond {
            self.fail(msg());
        }
    }
}

/// Progress line inside long loops, so a stall is visible and attributable.
fn heartbeat(phase: &str, done: usize, total: usize, t: &Instant) {
    let secs = t.elapsed().as_secs_f64();
    let rate = if secs > 0.0 { done as f64 / secs } else { 0.0 };
    println!("GSTRESS {phase}: {done}/{total} ({rate:.0} ops/s)");
}

fn report(phase: &str, ops: usize, t: Instant) {
    let secs = t.elapsed().as_secs_f64();
    let rate = if secs > 0.0 { ops as f64 / secs } else { 0.0 };
    println!("GSTRESS {phase:<12} {ops:>8} ops  {secs:>8.2}s  {rate:>10.0} ops/s");
}

/// Churn (phase D) deletes every 7th of the phase-A vertices; hub targets and
/// verification anchors must avoid those indices.
fn survives_churn(i: usize) -> bool {
    i % 7 != 0
}
fn safe_target(i: usize, n: usize) -> usize {
    let mut j = i % n;
    if !survives_churn(j) {
        j = if j + 1 >= n { 1 } else { j + 1 };
    }
    j
}

fn main() {
    let preset = match std::env::args().nth(1).as_deref() {
        None | Some("small") => &SMALL,
        Some("tiny") => &TINY,
        Some("medium") => &MEDIUM,
        Some("large") => &LARGE,
        Some(other) => {
            println!("usage: gstress [tiny|small|medium|large]  (got '{other}')");
            std::process::exit(2);
        }
    };
    println!(
        "gstress: preset {} (V={} bulkE={} degCap={} churn={} chain={} clique={})",
        preset.name,
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

    let mut add_v = |g: &mut Graph, st: &mut Stats, next_id: &mut u64, label: &str, name: &str| {
        let v = g.add_vertex(label, name, ObjID::new(0)).expect("add_vertex");
        // Ids are append indices and never reused; any drift is a bug.
        if v.0 != *next_id {
            st.fail(format!("vertex id drift: got {}, expected {}", v.0, *next_id));
        }
        *next_id += 1;
        v
    };

    // --- Phase A: registry rollover at the real DEFAULT_SEG_CAP -------------
    let n = preset.vertices;
    let t = Instant::now();
    for i in 0..n {
        add_v(&mut g, &mut st, &mut next_id, "n", &format!("v{i}"));
        if (i + 1) % 500 == 0 {
            heartbeat("A:rollover", i + 1, n, &t);
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
    let hub = add_v(&mut g, &mut st, &mut next_id, "hub", "hub1");
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
        // this should reach a higher ceiling than the distinct-target hub (I0).
        let hub2 = add_v(&mut g, &mut st, &mut next_id, "hub", "hub2");
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
        let got = g.out_neighbors(VertexId(u as u64), Labels::these(&["b"])).len();
        st.ck(got == expected, || {
            format!("post-churn out-degree of v{u}: got {got}, expected {expected}")
        });
    }
    // Adds after deletes: ids continue, never reuse.
    for i in 0..preset.churn_add {
        add_v(&mut g, &mut st, &mut next_id, "c", &format!("c{i}"));
    }
    report("D:churn", n + preset.churn_add, t);

    // --- Phase E: reopen by name, re-verify by sampling ---------------------
    let t = Instant::now();
    drop(g);
    let mut g = Graph::open_or_create(GRAPH).expect("reopen gstress graph");
    st.ck(g.vertex_info(VertexId(7)).is_none(), || {
        "reopen: deleted v7 came back".into()
    });
    st.ck(
        g.find_vertex("n", "v8") == Some(VertexId(8)),
        || "reopen: v8 lookup failed".into(),
    );
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
        g.find_vertex("c", &format!("c{}", preset.churn_add - 1)).is_some(),
        || "reopen: last churn vertex missing".into(),
    );
    report("E:reopen", 5, t);

    let t = Instant::now();
    let head = add_v(&mut g, &mut st, &mut next_id, "ch", "ch0");
    let mut prev = head;
    for i in 1..preset.chain {
        let v = add_v(&mut g, &mut st, &mut next_id, "ch", &format!("ch{i}"));
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
            st.fail(format!("chain walk exceeded {} hops — cycle?", preset.chain));
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
    st.ck(cur == prev, || "chain walk ended at the wrong vertex".into());
    report("F:chain", preset.chain, t);

    // Dense clique: k vertices, k*(k-1) directed edges.
    let t = Instant::now();
    let k = preset.clique;
    let mut cl = Vec::with_capacity(k);
    for i in 0..k {
        cl.push(add_v(&mut g, &mut st, &mut next_id, "cl", &format!("cl{i}")));
    }
    for i in 0..k {
        for j in 0..k {
            if i != j {
                g.add_edge(cl[i], "k", cl[j]).expect("clique add_edge");
            }
        }
    }
    for &i in &[0usize, k / 2, k - 1] {
        let out = g.out_neighbors(cl[i], Labels::these(&["k"])).len();
        let inn = g.in_neighbors(cl[i], Labels::these(&["k"])).len();
        let both = g.both_neighbors(cl[i], Labels::these(&["k"])).len();
        st.ck(out == k - 1, || format!("clique cl{i} out {out} != {}", k - 1));
        st.ck(inn == k - 1, || format!("clique cl{i} in {inn} != {}", k - 1));
        st.ck(both == 2 * (k - 1), || {
            format!("clique cl{i} both {both} != {}", 2 * (k - 1))
        });
    }
    report("F:clique", k * (k - 1), t);

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
