//! # The asymmetry is the result, so it is stated rather than hidden
//!
//! Our engine gives a vertex a built-in `(label, name)` identity, so an LDBC id
//! *is* the name and `find_vertex` resolves it. IndraDB gives a vertex a UUID
//! and a type, so the id has to become an ordinary property and every entry
//! lookup is a property-index query. That is a genuine modelling difference
//! between a native property graph and a KV-backed one, not an implementation
//! detail — and it is precisely what index-free adjacency claims to avoid.
//!
//! This is deliberately written against IndraDB's public API only, exactly
//! as `graph-eval/src/baseline.rs` is, so it reads as a fair account of what the
//! baseline offers a query author rather than a reach-through to internals.
//!
//! # Loading is expected to be slow, and that is data too
//!
//! A local `HashMap<(label, id), Uuid>` carries the id mapping during load.
//! Resolving each edge endpoint through the property index instead would add
//! ~3 M index queries to a load that is already the slow half, and would measure
//! the loader rather than the engine.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use indradb::{
    Database, Edge, Identifier, Json, QueryExt, QueryOutputValue, SpecificVertexQuery,
    VertexWithPropertyValueQuery,
};
use twizzler_indradb::TwizzlerDatastore;
use uuid::Uuid;

const DB: &str = "ldbc-idb";
const DIR: &str = "/initrd";
const P_ID: &str = "ldbcId";

/// Rows between flushes. Not a tuning knob — a memory bound.
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

/// Time-based heartbeat.
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
    results: usize,
}

impl Lat {
    fn new(name: &'static str) -> Self {
        Lat {
            name,
            us: Vec::new(),
            results: 0,
        }
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

pub(crate) fn load() {
    println!(
        "GSTRESS STAMP harness={} mode=ldbc-indradb-load",
        crate::HARNESS_REV
    );

    let db = TwizzlerDatastore::open_db(DB).expect("open datastore");
    db.delete(indradb::AllVertexQuery).expect("clear");
    db.index_property(ident(P_ID)).expect("index ldbcId");

    let mut ids: HashMap<(String, String), Uuid> = HashMap::new();

    let mut sync_s = 0.0f64;

    let t = Instant::now();
    let mut nverts = 0usize;
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
    for stem in &files {
        let (from, to) = edge_of(stem).expect("checked");
        let Some((_cols, r)) = open_csv(stem) else {
            continue;
        };
        let rel = ident(stem.split('_').nth(1).unwrap_or(stem));
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
            db.create_edge(&Edge::new(*a, rel, *b)).expect("create edge");
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

    println!(
        "GSTRESS IDBQ LOAD DONE — now reboot and run `gstress ldbc-indradb` to \
         query. **Loading and querying in one boot would leave every page warm \
         from the load**, while the native arm queries a cold graph in a fresh \
         boot. On this platform that gap is large — a cold index build measured \
         12.46 s against ~4.25 s warm — so a same-boot baseline would be \
         comparing a warm engine against a cold one and calling it a result."
    );
}

/// Query pass. Opens the datastore the load left behind, in a fresh boot, so
/// both arms are measured cold. See the note at the end of `load`.
pub(crate) fn run(iters: usize) {
    println!(
        "GSTRESS STAMP harness={} mode=ldbc-indradb iters={}",
        crate::HARNESS_REV,
        iters
    );
    let db = TwizzlerDatastore::open_db(DB).expect("open datastore");
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
        is1.us.push(t.elapsed().as_micros());
        is1.results += rows;

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
        is2.us.push(t.elapsed().as_micros());
        is2.results += rows;

        let Some(m0) = first else {
            no_msg += 1;
            continue;
        };
        let mid = vprop(&db, m0, P_ID);

        let t = Instant::now();
        let mut rows = 0;
        if let Some(p) = by_id(&db, "person", pid) {
            rows = out_edges(&db, p, "knows").len() + in_edges(&db, p, "knows").len();
        }
        is3.us.push(t.elapsed().as_micros());
        is3.results += rows;

        let t = Instant::now();
        let mut rows = 0;
        if let Some(m) = find_message(&db, &mid) {
            let _ = vprop(&db, m, "creationDate");
            let c = vprop(&db, m, "content");
            rows = usize::from(!c.is_empty());
        }
        is4.us.push(t.elapsed().as_micros());
        is4.results += rows;

        let t = Instant::now();
        let mut rows = 0;
        if let Some(m) = find_message(&db, &mid) {
            rows = out_edges(&db, m, "hasCreator").len();
        }
        is5.us.push(t.elapsed().as_micros());
        is5.results += rows;

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
        is6.us.push(t.elapsed().as_micros());
        is6.results += rows;

        let t = Instant::now();
        let mut rows = 0;
        if let Some(m) = find_message(&db, &mid) {
            rows = in_edges(&db, m, "replyOf").len();
        }
        is7.us.push(t.elapsed().as_micros());
        is7.results += rows;

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
    println!(
        "GSTRESS IDBQ NOTE: entry lookups go through a property index because \
         IndraDB has no built-in (label, name) identity — that asymmetry is the \
         result, not an implementation detail. {no_msg} persons authored nothing."
    );
}
