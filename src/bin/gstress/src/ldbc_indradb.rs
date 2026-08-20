//! The seven LDBC short reads (IS1–IS7) against IndraDB, the comparison
//! baseline, plus the loader that builds its store from the CSVs in `/initrd`.
//! Both engines run the same seven queries, on the same SF0.1 data, with the
//! same LDBC person ids, reporting the same percentiles.
//!
//! The native engine gives a vertex a built-in `(label, name)` identity, so an
//! LDBC id is the name and `find_vertex` resolves it. IndraDB gives a vertex a
//! UUID and a type, so the id has to become an ordinary property and every
//! entry lookup is a property-index query.
//!
//! This is written against IndraDB's public API only, so it reads as a fair
//! account of what the baseline offers a query author rather than a
//! reach-through to internals.
//!
//! Loading is slow by construction: every vertex costs a create plus one
//! `set_properties` per column, and every property write is a transaction. The
//! heartbeat and projection lines exist so a long load can be judged while it
//! runs. A local `HashMap<(label, id), Uuid>` carries the id mapping during
//! load; resolving each edge endpoint through the property index instead would
//! measure the loader rather than the engine.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use indradb::{
    Database, Edge, Identifier, Json, QueryExt, QueryOutputValue, SpecificEdgeQuery,
    SpecificVertexQuery, VertexWithPropertyValueQuery,
};
use twizzler_indradb::TwizzlerDatastore;
use uuid::Uuid;

const DB: &str = "ldbc-idb";
const DIR: &str = "/initrd";
const P_ID: &str = "ldbcId";

/// Rows between flushes. A memory bound, not a tuning knob: `nosync` writes
/// stay dirty in mapped memory until `sync`, so the flush cadence is the
/// peak-dirty-set size, and flushing too rarely exhausts the frame pool
/// mid-load. `KvStore` has two monolithic objects, so it cannot flush
/// incrementally and must instead flush often.
const SYNC_EVERY: usize = 25_000;

const NODES: &[&str] = &[
    "comment",
    "forum",
    "organisation",
    "person",
    "place",
    "post",
    "tag",
    "tagclass",
];

type Db = Database<TwizzlerDatastore>;

fn ident(s: &str) -> Identifier {
    Identifier::new(s).expect("valid identifier")
}

fn edge_of(stem: &str) -> Option<(&'static str, &'static str)> {
    let p: Vec<&str> = stem.split('_').collect();
    if p.len() != 3 {
        return None;
    }
    Some((
        NODES.iter().find(|n| **n == p[0])?,
        NODES.iter().find(|n| **n == p[2])?,
    ))
}

fn open_csv(stem: &str) -> Option<(Vec<String>, BufReader<File>)> {
    let f = File::open(format!("{DIR}/{stem}.csv")).ok()?;
    let mut r = BufReader::new(f);
    let mut head = String::new();
    r.read_line(&mut head).ok()?;
    Some((
        head.trim_end().split('|').map(str::to_string).collect(),
        r,
    ))
}

fn by_id(db: &Db, t: &str, id: &str) -> Option<Uuid> {
    let q = VertexWithPropertyValueQuery::new(ident(P_ID), Json::new(id.into()));
    let out = db.get(q).ok()?;
    let want = ident(t);
    match out.last()? {
        QueryOutputValue::Vertices(vs) => vs.iter().find(|v| v.t == want).map(|v| v.id),
        _ => None,
    }
}

/// A message may be a Comment or a Post — same flattening as the native arm, so
/// the two are comparable.
fn find_message(db: &Db, id: &str) -> Option<Uuid> {
    by_id(db, "comment", id).or_else(|| by_id(db, "post", id))
}

fn out_edges(db: &Db, id: Uuid, t: &str) -> Vec<Edge> {
    let Ok(q) = SpecificVertexQuery::single(id).outbound() else {
        return Vec::new();
    };
    let Ok(out) = db.get(q) else {
        return Vec::new();
    };
    let want = ident(t);
    match out.last() {
        Some(QueryOutputValue::Edges(es)) => es.iter().filter(|e| e.t == want).cloned().collect(),
        _ => Vec::new(),
    }
}

fn in_edges(db: &Db, id: Uuid, t: &str) -> Vec<Edge> {
    let Ok(q) = SpecificVertexQuery::single(id).inbound() else {
        return Vec::new();
    };
    let Ok(out) = db.get(q) else {
        return Vec::new();
    };
    let want = ident(t);
    match out.last() {
        Some(QueryOutputValue::Edges(es)) => es.iter().filter(|e| e.t == want).cloned().collect(),
        _ => Vec::new(),
    }
}

fn vprop(db: &Db, id: Uuid, key: &str) -> String {
    let Ok(out) = db.get(SpecificVertexQuery::single(id).properties().unwrap()) else {
        return String::new();
    };
    let want = ident(key);
    match out.last() {
        Some(QueryOutputValue::VertexProperties(ps)) => ps
            .iter()
            .flat_map(|p| p.props.iter())
            .find(|p| p.name == want)
            .and_then(|p| p.value.0.as_str().map(str::to_string))
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// Time-based heartbeat. A row-count interval is a guess about the rate, and
/// this arm's rate is the thing under measurement; a slow file would mean long
/// silences indistinguishable from a hang.
///
/// Prints at most every `EVERY`, with a running rate and a projection for the
/// whole file, so "is this worth waiting for" is answerable in one interval
/// rather than one file.
struct Beat {
    last: Instant,
    start: Instant,
}

const EVERY: std::time::Duration = std::time::Duration::from_secs(15);

impl Beat {
    fn new() -> Self {
        Beat {
            last: Instant::now(),
            start: Instant::now(),
        }
    }
    fn tick(&mut self, what: &str, done: usize, total: usize) {
        if self.last.elapsed() < EVERY {
            return;
        }
        self.last = Instant::now();
        let el = self.start.elapsed().as_secs_f64();
        let rate = done as f64 / el.max(1e-9);
        let left = (total.saturating_sub(done)) as f64 / rate.max(1e-9);
        println!(
            "GSTRESS IDBQ   {what}: {done}/{total} ({rate:.1}/s, ~{:.0} min left \
             for this file)",
            left / 60.0
        );
    }
}

struct Lat {
    name: &'static str,
    us: Vec<u128>,
    /// First-touch versus repeat, mirroring `ldbc_query::Lat`, so cold-path
    /// effects are separable from steady-state behaviour in both arms.
    cold: Vec<u128>,
    warm: Vec<u128>,
    results: usize,
}

impl Lat {
    fn new(name: &'static str) -> Self {
        Lat {
            name,
            us: Vec::new(),
            cold: Vec::new(),
            warm: Vec::new(),
            results: 0,
        }
    }

    fn push(&mut self, us: u128, first_touch: bool) {
        self.us.push(us);
        if first_touch {
            self.cold.push(us);
        } else {
            self.warm.push(us);
        }
    }

    fn report_split(&mut self) {
        let pct = |v: &mut Vec<u128>, p: f64| -> u128 {
            if v.is_empty() {
                return 0;
            }
            v.sort_unstable();
            v[(((v.len() - 1) as f64) * p) as usize]
        };
        let (cn, wn) = (self.cold.len(), self.warm.len());
        let (c50, c99) = (pct(&mut self.cold, 0.50), pct(&mut self.cold, 0.99));
        let (w50, w99) = (pct(&mut self.warm, 0.50), pct(&mut self.warm, 0.99));
        println!(
            "GSTRESS IDBSPLIT {:<6} cold n={cn} p50 {c50}us p99 {c99}us | \
             warm n={wn} p50 {w50}us p99 {w99}us | cold/warm p50 {:.1}x | \
             warm p99/p50 {:.1}x",
            self.name,
            if w50 > 0 { c50 as f64 / w50 as f64 } else { 0.0 },
            if w50 > 0 { w99 as f64 / w50 as f64 } else { 0.0 }
        );
    }
    fn report(&mut self) {
        if self.us.is_empty() {
            println!("GSTRESS IDBQ {:<6} no samples", self.name);
            return;
        }
        self.us.sort_unstable();
        let n = self.us.len();
        let at = |p: f64| self.us[((n as f64 - 1.0) * p) as usize];
        let mean: u128 = self.us.iter().sum::<u128>() / n as u128;
        println!(
            "GSTRESS IDBQ {:<6} n={:<5} mean {:>8}us  p50 {:>8}us  p95 {:>8}us  \
             p99 {:>8}us  max {:>8}us  rows/q {:.1}",
            self.name,
            n,
            mean,
            at(0.50),
            at(0.95),
            at(0.99),
            self.us[n - 1],
            self.results as f64 / n as f64
        );
    }
}

/// `edge_props` on [`load`] stores the extra CSV columns — `knows.creationDate`,
/// `likes.creationDate`, `hasMember.joinDate`, `workAt.workFrom`,
/// `studyAt.classYear` — as edge properties. Off by default. The complex reads
/// IC1, IC5, IC7 and IC11 need them, so their store is built with
/// `gstress ldbc-indradb-load edgeprops`; a missing edge property is
/// indistinguishable from an unset one, so that arm probes for them at startup.
///
/// `LEAN_DROP` is the vertex columns no LDBC complex read touches, dropped
/// under `lean`. Chosen by reading every query in `ldbc_complex_indradb.rs`:
/// what survives is `ldbcId`, `firstName`, `lastName`, `gender`, `birthday`,
/// `creationDate`, `content`, `imageFile`, `title`, `name` and `type`. These
/// five are read by nothing, and they are the bulk of the store.
///
/// A lean store is not valid for the short reads: IS1 returns `locationIP`,
/// `browserUsed` and the person's `creationDate`, and against a lean store it
/// would report them as empty and look healthy doing it. The warning at the
/// end of `load` says so.
const LEAN_DROP: &[&str] = &["locationIP", "browserUsed", "length", "language", "url"];

/// The one edge column no complex read touches. `knows.creationDate` is real
/// LDBC data; it is dropped under `lean` only because nothing reads it.
const LEAN_DROP_EDGE: &[&str] = &["creationDate"];

pub(crate) fn load(edge_props: bool, lean: bool) {
    println!(
        "GSTRESS STAMP harness={} mode=ldbc-indradb-load edgeprops={} lean={}",
        crate::HARNESS_REV,
        edge_props,
        lean
    );
    if lean {
        println!(
            "GSTRESS IDBQ LEAN: dropping vertex columns [{}] and `knows.creationDate` — \
             ~1.02 M of ~3.3 M values, none of which any complex read touches. **This store \
             is not valid for `gstress ldbc-indradb` (the short reads): IS1 returns \
             locationIP, browserUsed and creationDate, and would report them empty.** \
             An E6-AC4 restriction to record with any number from this store.",
            LEAN_DROP.join(", ")
        );
    }

    let db = TwizzlerDatastore::open_db(DB).expect("open datastore");
    // Clear first: the datastore outlives the run, and a second load on top of
    // a first would compare different data while looking healthy.
    db.delete(indradb::AllVertexQuery).expect("clear");
    db.index_property(ident(P_ID)).expect("index ldbcId");

    let mut ids: HashMap<(String, String), Uuid> = HashMap::new();

    // Insert time and flush time are reported separately.
    let mut sync_s = 0.0f64;

    let t = Instant::now();
    let mut nverts = 0usize;
    let mut ndropped = 0usize;
    for label in NODES {
        let Some((cols, r)) = open_csv(label) else {
            println!("GSTRESS IDBQ: {label}.csv absent");
            continue;
        };
        let ty = ident(label);
        let mut n = 0;
        // Row count for the projection: cheap, and it turns "still going" into
        // "still going, N hours left", which is the number that decides whether
        // to wait.
        let total = std::fs::read_to_string(format!("{DIR}/{label}.csv"))
            .map(|s| s.lines().count().saturating_sub(1))
            .unwrap_or(0);
        let mut beat = Beat::new();
        let mut file_sync = 0.0f64;
        for line in r.lines().map_while(|l| l.ok()) {
            let f: Vec<&str> = line.trim_end().split('|').collect();
            if f.len() < cols.len() || f[0].is_empty() {
                continue;
            }
            let id = db.create_vertex_from_type(ty).expect("create vertex");
            db.set_properties(
                SpecificVertexQuery::single(id),
                ident(P_ID),
                &Json::new(f[0].into()),
            )
            .expect("set id");
            for (i, col) in cols.iter().enumerate().skip(1) {
                if f[i].is_empty() {
                    continue;
                }
                if lean && LEAN_DROP.contains(&col.as_str()) {
                    ndropped += 1;
                    continue;
                }
                db.set_properties(
                    SpecificVertexQuery::single(id),
                    ident(col),
                    &Json::new(f[i].into()),
                )
                .expect("set prop");
            }
            ids.insert(((*label).to_string(), f[0].to_string()), id);
            n += 1;
            if n % SYNC_EVERY == 0 {
                let ts = Instant::now();
                db.sync().expect("sync");
                file_sync += ts.elapsed().as_secs_f64();
            }
            beat.tick(label, n, total);
        }
        let ts = Instant::now();
        db.sync().expect("sync");
        file_sync += ts.elapsed().as_secs_f64();
        sync_s += file_sync;
        println!("GSTRESS IDBQ {label}: {n} vertices (flush {file_sync:.1}s)");
        nverts += n;
        if nverts > 0 {
            let el = t.elapsed().as_secs_f64();
            println!(
                "GSTRESS IDBQ PROJECTION: {nverts} vertices in {el:.0}s = \
                 {:.1}/s -> ~{:.1} h for all 327,588 vertices, and ~{:.1} h for \
                 the 1,477,965 edges at the same rate. **If that is not \
                 acceptable, stop here and record where it stopped — the load \
                 rate is itself the E6 finding about the baseline at this \
                 scale.**",
                nverts as f64 / el.max(1e-9),
                327_588.0 / (nverts as f64 / el.max(1e-9)) / 3600.0,
                1_477_965.0 / (nverts as f64 / el.max(1e-9)) / 3600.0
            );
        }
    }
    let nodes_s = t.elapsed().as_secs_f64();
    println!(
        "GSTRESS IDBQ nodes: {nverts} in {nodes_s:.1}s ({:.0}/s)",
        nverts as f64 / nodes_s.max(1e-9)
    );

    let t = Instant::now();
    let mut nedges = 0usize;
    let mut files: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(DIR) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if let Some(stem) = name.strip_suffix(".csv") {
                if edge_of(stem).is_some() {
                    files.push(stem.to_string());
                }
            }
        }
    }
    files.sort();
    let mut nprops = 0usize;
    for stem in &files {
        let (from, to) = edge_of(stem).expect("checked");
        let Some((cols, r)) = open_csv(stem) else {
            continue;
        };
        let rel_name = stem.split('_').nth(1).unwrap_or(stem);
        let rel_is_knows = rel_name == "knows";
        let rel = ident(rel_name);
        let mut n = 0;
        let total = std::fs::read_to_string(format!("{DIR}/{stem}.csv"))
            .map(|s| s.lines().count().saturating_sub(1))
            .unwrap_or(0);
        let mut beat = Beat::new();
        let mut file_sync = 0.0f64;
        for line in r.lines().map_while(|l| l.ok()) {
            let f: Vec<&str> = line.trim_end().split('|').collect();
            if f.len() < 2 {
                continue;
            }
            let (Some(a), Some(b)) = (
                ids.get(&(from.to_string(), f[0].to_string())),
                ids.get(&(to.to_string(), f[1].to_string())),
            ) else {
                continue;
            };
            let edge = Edge::new(*a, rel, *b);
            db.create_edge(&edge).expect("create edge");
            if edge_props {
                // Columns beyond the two endpoints are edge properties. Stored
                // as strings, as every vertex property here is, so the two
                // arms' values compare without a type-coercion difference
                // sitting between them.
                for (i, col) in cols.iter().enumerate().skip(2) {
                    if i < f.len() && !f[i].is_empty() {
                        // `knows.creationDate` is the only edge column no
                        // complex read touches; under `lean` it goes too.
                        if lean && rel_is_knows && LEAN_DROP_EDGE.contains(&col.as_str()) {
                            ndropped += 1;
                            continue;
                        }
                        db.set_properties(
                            SpecificEdgeQuery::single(edge.clone()),
                            ident(col),
                            &Json::new(f[i].into()),
                        )
                        .expect("set edge prop");
                        nprops += 1;
                    }
                }
            }
            n += 1;
            if n % SYNC_EVERY == 0 {
                let ts = Instant::now();
                db.sync().expect("sync");
                file_sync += ts.elapsed().as_secs_f64();
            }
            beat.tick(stem, n, total);
        }
        let ts = Instant::now();
        db.sync().expect("sync");
        file_sync += ts.elapsed().as_secs_f64();
        sync_s += file_sync;
        println!("GSTRESS IDBQ {stem}: {n} edges (flush {file_sync:.1}s)");
        nedges += n;
    }
    let edges_s = t.elapsed().as_secs_f64();
    println!(
        "GSTRESS IDBQ edges: {nedges} in {edges_s:.1}s ({:.0}/s)",
        nedges as f64 / edges_s.max(1e-9)
    );
    let total_s = nodes_s + edges_s;
    println!(
        "GSTRESS IDBQ LOAD TOTAL: {} records in {total_s:.1}s ({:.1}s insert, \
         {sync_s:.1}s flush = {:.0}% of load)",
        nverts + nedges,
        total_s - sync_s,
        100.0 * sync_s / total_s.max(1e-9)
    );
    if lean {
        println!(
            "GSTRESS IDBQ LOAD LEAN: {ndropped} values dropped. Complex reads only — see the \
             LEAN warning at the top of this run."
        );
    }
    if edge_props {
        println!(
            "GSTRESS IDBQ LOAD EDGEPROPS: {nprops} edge-property values written. **This load is \
             not the one RESULTS Table 17 measured** — that arm dropped these columns. Record the \
             two separately rather than quoting one against the other."
        );
    } else {
        println!(
            "GSTRESS IDBQ LOAD EDGEPROPS: none (default). IC1, IC5, IC7 and IC11 cannot be \
             answered from this store; re-run as `gstress ldbc-indradb-load edgeprops` for E10."
        );
    }

    println!(
        "GSTRESS IDBQ LOAD DONE — now reboot and run `gstress ldbc-indradb` to \
         query. **Loading and querying in one boot would leave every page warm \
         from the load**, while the native arm queries a cold graph in a fresh \
         boot. On this platform that gap is large — a cold index build measured \
         12.46 s against ~4.25 s warm — so a same-boot baseline would be \
         comparing a warm engine against a cold one and calling it a result."
    );
}

/// Query pass. Opens the datastore the load left behind. Run it in its own
/// boot, so both arms are measured cold; loading and querying in one boot
/// would leave every page warm from the load.
pub(crate) fn run(iters: usize) {
    run_inner(iters, false)
}

/// The same queries with the baseline given a key index at open, symmetric
/// with the native arm's query-entry index.
///
/// The default arm has no index at read time — the store builds its map on the
/// write path only — so every lookup binary-searches the sorted region. The
/// native arm builds an index at open and that cost is excluded from its
/// latencies; this arm builds one and excludes it the same way.
pub(crate) fn run_indexed(iters: usize) {
    run_inner_indexed(iters, false, true)
}

/// Same queries, same code path, emitting a per-iteration result digest
/// instead of latencies. Deliberately not a second implementation: an
/// equivalence check written alongside the benchmark can agree with itself
/// while both differ from the engine under test.
pub(crate) fn equiv(iters: usize) {
    run_inner(iters, true)
}

fn run_inner(iters: usize, digest: bool) {
    run_inner_indexed(iters, digest, false)
}

fn run_inner_indexed(iters: usize, digest: bool, build_index: bool) {
    println!(
        "GSTRESS STAMP harness={} mode=ldbc-indradb iters={} index={}",
        crate::HARNESS_REV,
        iters,
        build_index
    );
    // Open (and the optional index build) is excluded from the latencies
    // below, as the native arm's query-entry index build is.
    let t = Instant::now();
    let db = if build_index {
        TwizzlerDatastore::open_db_indexed(DB).expect("open datastore")
    } else {
        TwizzlerDatastore::open_db(DB).expect("open datastore")
    };
    println!(
        "GSTRESS IDBQ OPEN: {:.2}s, index={build_index} (excluded from latencies)",
        t.elapsed().as_secs_f64()
    );
    twizzler_indradb::kv_stats::reset();
    // No clear, and no reload: this must read what the load boot wrote, or the
    // comparison is against an empty store.
    let probe = crate::ldbc_query::load_person_params();
    if probe.is_empty() {
        println!("GSTRESS IDBQ FAILED: no interactive_*_param.txt in {DIR}");
        return;
    }
    if by_id(&db, "person", &probe[0]).is_none() {
        println!(
            "GSTRESS IDBQ FAILED: `{DB}` has no person {} — run \
             `gstress ldbc-indradb-load` first, in its own boot.",
            probe[0]
        );
        return;
    }

    let persons = crate::ldbc_query::load_person_params();
    if persons.is_empty() {
        println!("GSTRESS IDBQ FAILED: no interactive_*_param.txt in {DIR}");
        return;
    }

    // Digest mode does exactly one pass over the parameter list. The list
    // cycles, so iterations beyond it are exact repeats: no new information
    // for an equivalence check, and equiv is not timed, so there is nothing to
    // average over either.
    let iters = if digest { persons.len() } else { iters };

    let mut is1 = Lat::new("IS1");
    let mut is2 = Lat::new("IS2");
    let mut is3 = Lat::new("IS3");
    let mut is4 = Lat::new("IS4");
    let mut is5 = Lat::new("IS5");
    let mut is6 = Lat::new("IS6");
    let mut is7 = Lat::new("IS7");
    let mut no_msg = 0usize;

    for i in 0..iters {
        let pid = &persons[i % persons.len()];
        // The list cycles, so the first pass is every id's first touch.
        let first_touch = i < persons.len();

        let t = Instant::now();
        let mut rows = 0;
        if let Some(p) = by_id(&db, "person", pid) {
            for k in [
                "firstName",
                "lastName",
                "birthday",
                "locationIP",
                "browserUsed",
                "gender",
                "creationDate",
            ] {
                let _ = vprop(&db, p, k);
            }
            rows = out_edges(&db, p, "isLocatedIn").len();
        }
        is1.push(t.elapsed().as_micros(), first_touch);
        is1.results += rows;
        let r1 = rows;

        // IS2 — same shape as the native arm: gather the person's messages,
        // order by creationDate descending, take 10.
        let t = Instant::now();
        let mut rows = 0;
        let mut first: Option<Uuid> = None;
        if let Some(p) = by_id(&db, "person", pid) {
            let mut ms: Vec<(String, Uuid)> = in_edges(&db, p, "hasCreator")
                .into_iter()
                .map(|e| {
                    let d = vprop(&db, e.outbound_id, "creationDate");
                    (d, e.outbound_id)
                })
                .collect();
            ms.sort_by(|a, b| b.0.cmp(&a.0));
            ms.truncate(10);
            rows = ms.len();
            first = ms.first().map(|(_, u)| *u);
        }
        is2.push(t.elapsed().as_micros(), first_touch);
        is2.results += rows;
        let r2 = rows;

        let Some(m0) = first else {
            if digest && i < persons.len() {
                println!("EQUIV {pid} is1={r1} is2={r2} mid=- NOMSG");
            }
            no_msg += 1;
            continue;
        };
        let mid = vprop(&db, m0, P_ID);

        let t = Instant::now();
        let mut rows = 0;
        if let Some(p) = by_id(&db, "person", pid) {
            rows = out_edges(&db, p, "knows").len() + in_edges(&db, p, "knows").len();
        }
        is3.push(t.elapsed().as_micros(), first_touch);
        is3.results += rows;
        let r3 = rows;

        let t = Instant::now();
        let mut rows = 0;
        if let Some(m) = find_message(&db, &mid) {
            let _ = vprop(&db, m, "creationDate");
            let c = vprop(&db, m, "content");
            rows = usize::from(!c.is_empty());
        }
        is4.push(t.elapsed().as_micros(), first_touch);
        is4.results += rows;
        let r4 = rows;

        let t = Instant::now();
        let mut rows = 0;
        if let Some(m) = find_message(&db, &mid) {
            rows = out_edges(&db, m, "hasCreator").len();
        }
        is5.push(t.elapsed().as_micros(), first_touch);
        is5.results += rows;
        let r5 = rows;

        // IS6 — walk `replyOf` up to the root post, then its containing forum.
        // The hand-rolled loop is deliberate: the native arm uses `repeat_out`,
        // so the two are independent implementations of the same query and
        // each checks the other.
        let t = Instant::now();
        let mut rows = 0;
        if let Some(m) = find_message(&db, &mid) {
            let mut cur = m;
            let mut hops = 0;
            loop {
                let parents = out_edges(&db, cur, "replyOf");
                match parents.first() {
                    Some(e) => {
                        cur = e.inbound_id;
                        hops += 1;
                        if hops > 64 {
                            break;
                        }
                    }
                    None => break,
                }
            }
            rows = in_edges(&db, cur, "containerOf").len();
        }
        is6.push(t.elapsed().as_micros(), first_touch);
        is6.results += rows;
        let r6 = rows;

        let t = Instant::now();
        let mut rows = 0;
        if let Some(m) = find_message(&db, &mid) {
            rows = in_edges(&db, m, "replyOf").len();
        }
        is7.push(t.elapsed().as_micros(), first_touch);
        is7.results += rows;
        let r7 = rows;

        if digest && i < persons.len() {
            println!(
                "EQUIV {pid} is1={r1} is2={r2} mid={mid} is3={r3} is4={r4} \
                 is5={r5} is6={r6} is7={r7}"
            );
        }

        if (i + 1) % 100 == 0 {
            println!("GSTRESS IDBQ .. {}/{iters}", i + 1);
        }
    }

    println!("GSTRESS IDBQ RESULTS (IndraDB baseline, LDBC short reads, SF0.1):");
    for l in [
        &mut is1, &mut is2, &mut is3, &mut is4, &mut is5, &mut is6, &mut is7,
    ] {
        l.report();
    }
    println!("GSTRESS IDBQ SPLIT (first touch of each id vs repeats):");
    for l in [
        &mut is1, &mut is2, &mut is3, &mut is4, &mut is5, &mut is6, &mut is7,
    ] {
        l.report_split();
    }
    twizzler_indradb::kv_stats::report(if digest { "equiv" } else { "query" });
    println!(
        "GSTRESS IDBQ NOTE: entry lookups go through a property index because \
         IndraDB has no built-in (label, name) identity — that asymmetry is the \
         result, not an implementation detail. {no_msg} persons authored nothing."
    );
}
