//! The IndraDB baseline arm of `gstress`.
//!
//! Runs the same workload shape as the native arm — same preset sizes, same
//! phases, same verification — against `Database<TwizzlerDatastore>`, so the
//! two load rates can be compared directly.
//!
//! Two deliberate asymmetries, so the numbers are read correctly:
//!
//! 1. Name resolution. Our engine gives a vertex a built-in `(label, name)`
//!    identity; IndraDB gives a UUID plus a type, so the name must be an
//!    ordinary property. Resolving names through property queries would
//!    measure our datastore's unindexed property scan (indexes are declared
//!    but not built), which is an implementation choice of ours, not a
//!    property of IndraDB. So the harness keeps a local `name -> Uuid` map for
//!    wiring, and phase A separately times name lookups through the real
//!    property path — reported apart from the write rate.
//! 2. Ops counted. One "vertex op" here is `create_vertex_from_type` plus
//!    one `set_properties` for the name — two IndraDB calls — because that is
//!    what it takes to reach parity with one native `add_vertex`. The
//!    comparison is per-record cost, not per-API-call cost.

use std::collections::HashMap;
use std::time::Instant;

use indradb::{
    Database, Edge, Identifier, Json, QueryExt, QueryOutputValue, SpecificVertexQuery,
};
use twizzler_indradb::TwizzlerDatastore;
use uuid::Uuid;

use crate::{heartbeat, report, safe_target, Preset, Rng, Stats};

const GRAPH: &str = "gstress-idb";
const P_NAME: &str = "name";

type Db = Database<TwizzlerDatastore>;

fn ident(s: &str) -> Identifier {
    Identifier::new(s).expect("valid identifier")
}

/// Name registry: the harness's own `name -> id` map (see the module note).
struct Names(HashMap<String, Uuid>);

impl Names {
    fn get(&self, n: &str) -> Uuid {
        *self.0.get(n).expect("known vertex name")
    }
}

fn add_vertex(db: &Db, names: &mut Names, t: &str, name: &str) -> Uuid {
    let id = db
        .create_vertex_from_type(ident(t))
        .expect("create_vertex_from_type");
    db.set_properties(
        SpecificVertexQuery::single(id),
        ident(P_NAME),
        &Json::new(name.into()),
    )
    .expect("set name");
    names.0.insert(name.to_string(), id);
    id
}

fn add_edge(db: &Db, from: Uuid, t: &str, to: Uuid) -> bool {
    db.create_edge(&Edge::new(from, ident(t), to))
        .expect("create_edge")
}

/// Outgoing edges of `id`, optionally filtered by type.
fn out_edges(db: &Db, id: Uuid, t: Option<&str>) -> Vec<Edge> {
    let Ok(q) = SpecificVertexQuery::single(id).outbound() else {
        return Vec::new();
    };
    let Ok(out) = db.get(q) else { return Vec::new() };
    match out.last() {
        Some(QueryOutputValue::Edges(es)) => match t {
            Some(t) => {
                let want = ident(t);
                es.iter().filter(|e| e.t == want).cloned().collect()
            }
            None => es.clone(),
        },
        _ => Vec::new(),
    }
}

fn vertex_exists(db: &Db, id: Uuid) -> bool {
    let Ok(out) = db.get(SpecificVertexQuery::single(id)) else {
        return false;
    };
    matches!(out.last(), Some(QueryOutputValue::Vertices(vs)) if !vs.is_empty())
}

/// Resolve a name through the real property path, to time it honestly.
fn lookup_by_name(db: &Db, name: &str) -> Option<Uuid> {
    let q = indradb::VertexWithPropertyValueQuery::new(ident(P_NAME), Json::new(name.into()));
    let out = db.get(q).ok()?;
    match out.last()? {
        QueryOutputValue::Vertices(vs) => vs.first().map(|v| v.id),
        _ => None,
    }
}

/// The baseline arm. Mirrors the native arm's phases.
pub fn run(preset: &Preset, st: &mut Stats) {
    println!(
        "gstress: preset {} (indradb baseline) (V={} bulkE={} degCap={} churn={} chain={} clique={})",
        preset.name,
        preset.vertices,
        preset.bulk_edges,
        preset.degree_cap,
        preset.churn_add,
        preset.chain,
        preset.clique
    );

    // Setup timed separately and excluded from the run total, matching the
    // native arm — `reset_db` tears down the previous run's store, which is
    // not this run's workload.
    let setup = Instant::now();
    TwizzlerDatastore::reset_db(GRAPH).expect("reset baseline datastore");
    let db = TwizzlerDatastore::open_db(GRAPH).expect("open baseline datastore");
    db.index_property(ident(P_NAME)).expect("index name");
    let mut names = Names(HashMap::new());
    let setup_secs = setup.elapsed().as_secs_f64();
    println!("GSTRESS SETUP: reset + open {setup_secs:.2}s (excluded from the run total)");

    // Workload only, from here.
    let total = Instant::now();

    // --- Phase A: bulk vertex creation ------------------------------------
    let n = preset.vertices;
    let t = Instant::now();
    let mut vids: Vec<Uuid> = Vec::with_capacity(n);
    for i in 0..n {
        vids.push(add_vertex(&db, &mut names, "n", &format!("v{i}")));
        if (i + 1) % 100 == 0 {
            heartbeat("A:rollover", i + 1, n, &t);
        }
    }
    report("A:rollover", n, t);

    // Name lookups through the real property path, timed separately: this is
    // our datastore's declared-not-built index (a scan), not IndraDB's design.
    let t = Instant::now();
    let step = (n / 20).max(1);
    let mut lookups = 0usize;
    for i in (0..n).step_by(step) {
        let want = names.get(&format!("v{i}"));
        st.ck(lookup_by_name(&db, &format!("v{i}")) == Some(want), || {
            format!("baseline name lookup v{i} failed")
        });
        lookups += 1;
    }
    report("A:namelookup", lookups, t);

    // --- Phase B: bulk random edges ---------------------------------------
    let t = Instant::now();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(preset.bulk_edges);
    for k in 0..preset.bulk_edges {
        let u = rng.below(n);
        let v = rng.below(n);
        add_edge(&db, vids[u], "b", vids[v]);
        pairs.push((u as u32, v as u32));
        if (k + 1) % 100 == 0 {
            heartbeat("B:bulk", k + 1, preset.bulk_edges, &t);
        }
    }
    for s in 0..20 {
        let u = (s * step) % n;
        let expected = pairs.iter().filter(|(a, _)| *a as usize == u).count();
        let got = out_edges(&db, vids[u], Some("b")).len();
        st.ck(got == expected, || {
            format!("baseline bulk out-degree of v{u}: got {got}, expected {expected}")
        });
    }
    report("B:bulk", preset.bulk_edges, t);

    // --- Phase C: high-degree hub -----------------------------------------
    let t = Instant::now();
    let hub = add_vertex(&db, &mut names, "hub", "hub1");
    let mut hub_deg = 0usize;
    // Beyond this, `safe_target` wraps and repeats targets — which our engine
    // stores as parallel edges but IndraDB rejects as duplicates, making the
    // arms incomparable. Say so rather than silently producing a skewed number.
    let cap = crate::max_distinct_degree(n);
    if preset.degree_cap > cap {
        println!(
            "GSTRESS NOTE: degree_cap {} exceeds the {} distinct surviving targets \
             for V={}; beyond that the engines store different graphs \
             (multigraph vs simple graph) and the arms are not comparable",
            preset.degree_cap, cap, n
        );
    }
    for i in 0..preset.degree_cap {
        let target = vids[safe_target(i, n)];
        if !add_edge(&db, hub, "h", target) {
            println!(
                "GSTRESS FINDING: baseline edge #{} refused (endpoint missing?)",
                i + 1
            );
            break;
        }
        hub_deg += 1;
        if hub_deg % 100 == 0 {
            heartbeat("C:degree", hub_deg, preset.degree_cap, &t);
        }
    }
    let got = out_edges(&db, hub, Some("h")).len();
    st.ck(got == hub_deg, || {
        format!("baseline hub out-degree: got {got}, expected {hub_deg}")
    });
    report("C:degree", hub_deg, t);

    // --- Phase D: churn ----------------------------------------------------
    let t = Instant::now();
    let mut deleted = vec![false; n];
    let mut ndel = 0usize;
    for i in (0..n).step_by(7) {
        db.delete(SpecificVertexQuery::single(vids[i]))
            .expect("delete vertex");
        deleted[i] = true;
        ndel += 1;
        if ndel % 50 == 0 {
            heartbeat("D:churn", ndel, n / 7 + 1, &t);
        }
    }
    for i in 0..n {
        let alive = vertex_exists(&db, vids[i]);
        st.ck(alive == !deleted[i], || {
            format!("baseline churn scan v{i}: alive={alive}, expected {}", !deleted[i])
        });
    }
    for i in 0..preset.churn_add {
        add_vertex(&db, &mut names, "c", &format!("c{i}"));
    }
    report("D:churn", n + preset.churn_add, t);

    // --- Phase E: reopen by name -------------------------------------------
    let t = Instant::now();
    let hub_id = names.get("hub1");
    let survivor = vids[1]; // index 1 is never deleted by the churn stride
    drop(db);
    let db = TwizzlerDatastore::open_db(GRAPH).expect("reopen baseline datastore");
    st.ck(!vertex_exists(&db, vids[0]), || {
        "baseline reopen: deleted v0 came back".into()
    });
    st.ck(vertex_exists(&db, survivor), || {
        "baseline reopen: survivor missing".into()
    });
    let got = out_edges(&db, hub_id, Some("h")).len();
    st.ck(got == hub_deg, || {
        format!("baseline reopen: hub degree got {got}, expected {hub_deg}")
    });
    report("E:reopen", 3, t);

    // --- Phase F: pathological shapes --------------------------------------
    let t = Instant::now();
    let head = add_vertex(&db, &mut names, "ch", "ch0");
    let mut prev = head;
    for i in 1..preset.chain {
        let v = add_vertex(&db, &mut names, "ch", &format!("ch{i}"));
        add_edge(&db, prev, "next", v);
        prev = v;
        if (i + 1) % 100 == 0 {
            heartbeat("F:chain", i + 1, preset.chain, &t);
        }
    }
    let mut cur = head;
    let mut hops = 0usize;
    loop {
        if hops > preset.chain {
            st.fail(format!("baseline chain walk exceeded {} hops", preset.chain));
            break;
        }
        let next = out_edges(&db, cur, Some("next"));
        match next.len() {
            0 => break,
            1 => {
                cur = next[0].inbound_id;
                hops += 1;
            }
            k => {
                st.fail(format!("baseline chain fan-out {k} at hop {hops}"));
                break;
            }
        }
    }
    st.ck(hops == preset.chain - 1, || {
        format!("baseline chain walk: {hops} hops, expected {}", preset.chain - 1)
    });
    report("F:chain", preset.chain, t);

    let t = Instant::now();
    let k = preset.clique;
    let mut cl = Vec::with_capacity(k);
    for i in 0..k {
        cl.push(add_vertex(&db, &mut names, "cl", &format!("cl{i}")));
    }
    for i in 0..k {
        for j in 0..k {
            if i != j {
                add_edge(&db, cl[i], "k", cl[j]);
            }
        }
        // The densest write phase in the run — k*(k-1) edges with no batching —
        // so on the baseline it is also the slowest; the heartbeat keeps it
        // from reading as a hang.
        if (i + 1) % 5 == 0 {
            heartbeat("F:clique", (i + 1) * (k - 1), k * (k - 1), &t);
        }
    }
    for &i in &[0usize, k / 2, k - 1] {
        let out = out_edges(&db, cl[i], Some("k")).len();
        st.ck(out == k - 1, || {
            format!("baseline clique cl{i} out {out} != {}", k - 1)
        });
    }
    report("F:clique", k * (k - 1), t);

    // --- Phase H: reads ------------------------------------------------------
    // Mirrors the native arm's read phase exactly. `lookup` goes through this
    // engine's native key path — a property index — which is the honest
    // comparison to our built-in `(label, name)` identity, and is where the
    // identity-model asymmetry shows up as a number. `1hop`/`2hop` start from
    // ids in hand, isolating traversal. The sample stride is shared with the
    // native arm so the two sample sets stay comparable.
    let rstep = crate::read_step(n);
    let reps = preset.read_reps;
    let read_sample: Vec<(usize, Uuid)> = (0..n)
        .step_by(rstep)
        .filter(|i| !deleted[*i])
        .map(|i| (i, vids[i]))
        .collect();
    println!(
        "GSTRESS READS: {} sampled vertices x {} reps",
        read_sample.len(),
        reps
    );

    let max_reps = reps * 100;
    crate::measure_read("H:lookup", max_reps, || {
        let mut found = 0usize;
        for (i, _) in &read_sample {
            if lookup_by_name(&db, &format!("v{i}")).is_some() {
                found += 1;
            }
        }
        debug_assert_eq!(found, read_sample.len());
        read_sample.len()
    });
    crate::measure_read("H:1hop", max_reps, || {
        let mut seen = 0usize;
        for (_, id) in &read_sample {
            seen += out_edges(&db, *id, None).len();
        }
        let _ = seen;
        read_sample.len()
    });
    crate::measure_read("H:2hop", max_reps, || {
        let mut seen = 0usize;
        for (_, id) in &read_sample {
            for e in out_edges(&db, *id, None) {
                seen += out_edges(&db, e.inbound_id, None).len();
            }
        }
        let _ = seen;
        read_sample.len()
    });
    crate::measure_read("H:scan", max_reps, || {
        match db.get(indradb::AllVertexQuery) {
            Ok(out) => match out.last() {
                Some(QueryOutputValue::Vertices(vs)) => vs.len(),
                _ => 0,
            },
            Err(_) => 0,
        }
    });

    // --- Phase G: degradation probe -----------------------------------------
    // Identical flat workload to the native arm's phase G, so the two engines'
    // self-inflicted degradation within a single run can be compared directly.
    let t = Instant::now();
    let mut prog = crate::Progress::new("G:degrade");
    let gtotal = preset.degrade_windows * preset.degrade_batch;
    let mut gdone = 0usize;
    for w in 0..preset.degrade_windows {
        for j in 0..preset.degrade_batch {
            add_vertex(&db, &mut names, "g", &format!("g{}_{}", w, j));
        }
        gdone += preset.degrade_batch;
        prog.tick(gdone, gtotal);
    }
    prog.summarize();
    report("G:degrade", gtotal, t);

    let secs = total.elapsed().as_secs_f64();
    println!(
        "GSTRESS {}: preset {} (indradb baseline) — hub degree {}, {:.1}s total",
        if st.fails == 0 { "OK" } else { "FAILED" },
        preset.name,
        hub_deg,
        secs
    );
}

// ---------------------------------------------------------------------------
// Cross-reboot durability pair
// ---------------------------------------------------------------------------
//
// Within-boot reopen tests map objects whose pages are still resident, so
// they evidence nothing about the disk image. This pair is the cross-reboot
// half: `gstress indradb-seed <N>`, then, in a later boot,
// `gstress indradb-verify <N>`.
//
// Verification resolves vertices through the real property-index path
// (`lookup_by_name`), not a captured id map: verify runs in a fresh process
// that cannot have one.

const DUR_DB: &str = "idur";

pub(crate) fn seed_durable(n: usize) {
    println!(
        "GSTRESS STAMP harness={} mode=indradb-seed N={n}",
        crate::HARNESS_REV
    );
    TwizzlerDatastore::reset_db(DUR_DB).expect("reset");
    let db = TwizzlerDatastore::open_db(DUR_DB).expect("open");
    db.index_property(ident(P_NAME)).expect("index name");

    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let id = db
            .create_vertex_from_type(ident("d"))
            .expect("create vertex");
        db.set_properties(
            SpecificVertexQuery::single(id),
            ident(P_NAME),
            &Json::new(format!("v{i}").into()),
        )
        .expect("set name");
        ids.push(id);
    }
    for i in 0..n.saturating_sub(1) {
        assert!(
            db.create_edge(&Edge::new(ids[i], ident("e"), ids[i + 1]))
                .expect("create edge"),
            "edge v{i}->v{} not created",
            i + 1
        );
    }
    // A property on every 5th, a delete on every 7th — verify then covers
    // live records, deleted ones, properties, and cascaded edges.
    for i in (0..n).step_by(5) {
        db.set_properties(
            SpecificVertexQuery::single(ids[i]),
            ident("k"),
            &Json::new(format!("k{i}").into()),
        )
        .expect("set k");
    }
    for i in (0..n).step_by(7) {
        db.delete(SpecificVertexQuery::single(ids[i])).expect("delete");
    }
    db.sync().expect("sync");

    // Mid-seed reopen, then keep writing. Writes through the second handle
    // must be durable even though the first handle created the structures
    // they land in.
    drop(db);
    let db = TwizzlerDatastore::open_db(DUR_DB).expect("mid-seed reopen");
    let mut w = Vec::with_capacity(n / 4);
    for i in 0..n / 4 {
        let id = db
            .create_vertex_from_type(ident("d2"))
            .expect("create vertex after reopen");
        db.set_properties(
            SpecificVertexQuery::single(id),
            ident(P_NAME),
            &Json::new(format!("w{i}").into()),
        )
        .expect("set name after reopen");
        w.push(id);
    }
    for i in 0..(n / 4).saturating_sub(1) {
        assert!(
            db.create_edge(&Edge::new(w[i], ident("e2"), w[i + 1]))
                .expect("create edge after reopen"),
        );
    }
    db.sync().expect("sync after reopen");
    println!(
        "GSTRESS IDB SEED: {n} vertices, every 5th with a property, every 7th \
         deleted (edges cascade), {} more with e2 edges through a reopened \
         handle. Now reboot and run `gstress indradb-verify {n}`.",
        n / 4
    );
}

pub(crate) fn verify_durable(n: usize) {
    println!(
        "GSTRESS STAMP harness={} mode=indradb-verify N={n}",
        crate::HARNESS_REV
    );
    let db = TwizzlerDatastore::open_db(DUR_DB).expect("open");
    let mut st = Stats::new();

    // `open_db` on a bare image creates an empty store, and an empty run must
    // be a loud failure, not a table.
    let total = match db.get(indradb::AllVertexQuery) {
        Ok(out) => match out.last() {
            Some(QueryOutputValue::Vertices(vs)) => vs.len(),
            _ => 0,
        },
        Err(_) => 0,
    };
    if total == 0 {
        println!(
            "GSTRESS IDB DURABILITY FAILED: `{DUR_DB}` is empty — run \
             `gstress indradb-seed {n}` in a previous boot first."
        );
        std::process::exit(1);
    }

    let expect = (0..n).filter(|i| i % 7 != 0).count() + n / 4;
    st.ck(total == expect, || {
        format!("live vertices: got {total}, expected {expect}")
    });

    for i in (0..n).step_by(97) {
        let name = format!("v{i}");
        let got = lookup_by_name(&db, &name);
        if i % 7 == 0 {
            st.ck(got.is_none(), || {
                format!("v{i} was deleted before the reboot but came back")
            });
            continue;
        }
        let Some(id) = got else {
            st.fail(format!("v{i} missing after reboot (property-index path)"));
            continue;
        };
        if i % 5 == 0 {
            let has_k = db
                .get(SpecificVertexQuery::single(id).properties().unwrap())
                .ok()
                .and_then(|out| match out.last() {
                    Some(QueryOutputValue::VertexProperties(ps)) => Some(
                        ps.iter().any(|vp| {
                            vp.props.iter().any(|p| {
                                p.name == ident("k")
                                    && p.value == Json::new(format!("k{i}").into())
                            })
                        }),
                    ),
                    _ => None,
                })
                .unwrap_or(false);
            st.ck(has_k, || format!("v{i}'s property did not survive"));
        }
        // The chain edge, when both ends survived the churn. (Cascade
        // correctness for edges incident to deleted vertices is pinned by
        // in-boot tests; here the live-vertex total above is what a failed
        // cascade would move.)
        if i + 1 < n && (i + 1) % 7 != 0 {
            let next = lookup_by_name(&db, &format!("v{}", i + 1));
            let outs = out_edges(&db, id, Some("e"));
            st.ck(
                next.is_some() && outs.iter().any(|e| Some(e.inbound_id) == next),
                || format!("v{i} lost its outgoing edge to v{}", i + 1),
            );
        }
    }

    // The post-reopen tranche, and its e2 edges — written through the second
    // handle.
    for i in (0..n / 4).step_by(23) {
        st.ck(lookup_by_name(&db, &format!("w{i}")).is_some(), || {
            format!("w{i} (written after a mid-seed reopen) did not survive")
        });
    }
    for i in (0..(n / 4).saturating_sub(1)).step_by(17) {
        let a = lookup_by_name(&db, &format!("w{i}"));
        let b = lookup_by_name(&db, &format!("w{}", i + 1));
        match (a, b) {
            (Some(a), Some(b)) => st.ck(
                out_edges(&db, a, Some("e2")).iter().any(|e| e.inbound_id == b),
                || format!("w{i} lost its e2 edge to w{}", i + 1),
            ),
            _ => st.fail(format!("w{i}/w{} unresolvable for the e2 check", i + 1)),
        }
    }

    if st.fails == 0 {
        println!(
            "GSTRESS IDB DURABILITY OK: {expect} datastore vertices survived a \
             reboot intact."
        );
    } else {
        st.report_suppressed();
        println!("GSTRESS IDB DURABILITY FAILED: {} check(s)", st.fails);
        std::process::exit(1);
    }
}
