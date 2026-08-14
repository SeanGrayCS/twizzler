//! Opens the graph `gstress ldbc` built and left on disk, so this runs in its
//! own boot. That is not merely tidy: it means the numbers are read from a graph
//! that survived a reboot, so durability at 1.8 M records is exercised by the
//! benchmark rather than asserted separately.
//!
//! # Message is a supertype, and our engine has no supertypes
//!
//! # Parameters
//!
//! Person ids come from LDBC's own `interactive_*_param.txt` substitution
//! parameters. There are deliberately no short-read parameter files: the
//! spec has IS1–IS7 parameterised from the driver's runtime state — ids seen in
//! earlier results — so message ids here are taken from *IS2's own output* for
//! the person under test, which is how the driver chains them.
//!
//! This is much closer to the benchmark's intent than sampling arbitrary
//! vertices, and it is still not an audited result: the official driver also
//! controls issue rate, mix, and dependency time. Stated plainly rather than
//! buried, because a latency number carries the benchmark's name whether or not
//! it earned it.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use twizzler_graph::{Graph, Labels, Lookup, PropValue, VertexId, DEFAULT_ARENA_CAP};

const NAME: &str = "ldbc";
const DIR: &str = "/initrd";
const MSG: Labels<'static> = Labels::These(&["comment", "post"]);

/// Latencies for one query, reported the way LDBC reports: percentiles, not a
/// mean. A mean hides the tail, and the tail is what a transactional workload is
/// judged on.
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
            println!("GSTRESS LDBCQ {:<6} no samples", self.name);
            return;
        }
        self.us.sort_unstable();
        let n = self.us.len();
        let at = |p: f64| self.us[((n as f64 - 1.0) * p) as usize];
        let mean: u128 = self.us.iter().sum::<u128>() / n as u128;
        println!(
            "GSTRESS LDBCQ {:<6} n={:<5} mean {:>8}us  p50 {:>8}us  p95 {:>8}us  \
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

/// Resolve a message id across both concrete labels.
fn find_message(g: &Graph, id: &str) -> Option<VertexId> {
    for l in ["comment", "post"] {
        if let Lookup::Found(v) = g.find_vertex(l, id) {
            return Some(v);
        }
    }
    None
}

fn str_prop(g: &Graph, v: VertexId, key: &str) -> String {
    match g.get_vertex_prop(v, key) {
        Some(PropValue::Str(s)) => s.as_str().to_string(),
        _ => g.get_vertex_text(v, key).unwrap_or_default(),
    }
}

pub(crate) fn run(iters: usize) {
    println!(
        "GSTRESS STAMP harness={} mode=ldbc-query iters={}",
        crate::HARNESS_REV,
        iters
    );

    let t = Instant::now();
    let g = match Graph::open_or_create_arena(NAME, DEFAULT_ARENA_CAP) {
        Ok(g) => g,
        Err(e) => {
            println!("GSTRESS LDBCQ FAILED: cannot open `{NAME}`: {e:?} — run `gstress ldbc` first");
            return;
        }
    };
    println!("GSTRESS LDBCQ open: {:.2}s, {} arenas", t.elapsed().as_secs_f64(), g.arena_count());

    // Force the index build before timing anything.
    let t = Instant::now();
    let _ = g.find_vertex("person", "warmup-nonexistent");
    println!(
        "GSTRESS LDBCQ index build: {:.2}s (excluded from query latencies)",
        t.elapsed().as_secs_f64()
    );

    let persons = load_person_params();
    if persons.is_empty() {
        println!(
            "GSTRESS LDBCQ FAILED: no interactive_*_param.txt in {DIR}. Re-run \
             scripts/flatten_ldbc.py with --params <substitution_parameters dir>."
        );
        return;
    }
    println!("GSTRESS LDBCQ params: {} LDBC person ids", persons.len());

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

        // IS1 — profile of a person.
        let t = Instant::now();
        let mut rows = 0;
        if let Lookup::Found(p) = g.find_vertex("person", pid) {
            let _ = str_prop(&g, p, "firstName");
            let _ = str_prop(&g, p, "lastName");
            let _ = str_prop(&g, p, "birthday");
            let _ = str_prop(&g, p, "locationIP");
            let _ = str_prop(&g, p, "browserUsed");
            let _ = str_prop(&g, p, "gender");
            let _ = str_prop(&g, p, "creationDate");
            rows = g
                .traversal()
                .v(p)
                .out(Labels::these(&["isLocatedIn"]))
                .count();
        }
        is1.us.push(t.elapsed().as_micros());
        is1.results += rows;

        // IS2 — a person's 10 most recent messages, newest first. Its result
        // supplies the message id for IS4-IS7, which is how the LDBC driver
        // parameterises the short reads.
        let t = Instant::now();
        let mut rows = 0;
        let mut msg: Option<VertexId> = None;
        if let Lookup::Found(p) = g.find_vertex("person", pid) {
            let ids = g
                .traversal()
                .v(p)
                .in_(Labels::these(&["hasCreator"]))
                .order_by_prop_desc("creationDate")
                .limit(10)
                .to_ids();
            rows = ids.len();
            msg = ids.first().copied();
        }
        is2.us.push(t.elapsed().as_micros());
        is2.results += rows;

        let Some(m0) = msg else {
            // This person authored nothing; IS4-IS7 have no parameter, and
            // timing them against a missing id would measure the miss path.
            no_msg += 1;
            continue;
        };
        let mid = g.vertex_info(m0).map(|i| i.name).unwrap_or_default();
        let mid = mid.as_str();

        // IS3 — friends, by knows.
        let t = Instant::now();
        let mut rows = 0;
        if let Lookup::Found(p) = g.find_vertex("person", pid) {
            rows = g.traversal().v(p).both(Labels::these(&["knows"])).count();
        }
        is3.us.push(t.elapsed().as_micros());
        is3.results += rows;

        // IS4 — a message's content and creation date.
        let t = Instant::now();
        let mut rows = 0;
        if let Some(m) = find_message(&g, mid) {
            let _ = str_prop(&g, m, "creationDate");
            let c = str_prop(&g, m, "content");
            rows = usize::from(!c.is_empty());
        }
        is4.us.push(t.elapsed().as_micros());
        is4.results += rows;

        // IS5 — a message's creator.
        let t = Instant::now();
        let mut rows = 0;
        if let Some(m) = find_message(&g, mid) {
            rows = g
                .traversal()
                .v(m)
                .out(Labels::these(&["hasCreator"]))
                .count();
        }
        is5.us.push(t.elapsed().as_micros());
        is5.results += rows;

        let t = Instant::now();
        let mut rows = 0;
        if let Some(m) = find_message(&g, mid) {
            let walk = g.traversal().v(m).repeat_out(Labels::these(&["replyOf"])).until_exhausted();
            if !walk.hit_depth_cap() {
                if let Some(root) = walk.first() {
                    rows = g
                        .traversal()
                        .v(root)
                        .in_(Labels::these(&["containerOf"]))
                        .count();
                }
            }
        }
        is6.us.push(t.elapsed().as_micros());
        is6.results += rows;

        // IS7 — direct replies to a message.
        let t = Instant::now();
        let mut rows = 0;
        if let Some(m) = find_message(&g, mid) {
            rows = g
                .traversal()
                .v(m)
                .in_(Labels::these(&["replyOf"]))
                .has_label("comment")
                .count();
        }
        is7.us.push(t.elapsed().as_micros());
        is7.results += rows;

        if (i + 1) % 200 == 0 {
            println!("GSTRESS LDBCQ .. {}/{iters}", i + 1);
        }
    }

    println!("GSTRESS LDBCQ RESULTS (LDBC-SNB Interactive short reads, SF0.1):");
    for l in [&mut is1, &mut is2, &mut is3, &mut is4, &mut is5, &mut is6, &mut is7] {
        l.report();
    }
    println!(
        "GSTRESS LDBCQ NOTE: person ids are LDBC's substitution parameters; \
         message ids come from IS2's own result, as the driver chains them. Not \
         an audited result — the official driver also controls issue rate, mix \
         and dependency time. `MSG` spans `comment`+`post` because the engine \
         has no type hierarchy. {no_msg} of {iters} persons authored nothing, so \
         IS4-IS7 have fewer samples than IS1-IS3."
    );
    let _ = MSG;
}

/// LDBC person ids from the substitution parameters.
///
/// Every `interactive_*_param.txt` whose header starts with `personId` — most of
/// the complex reads are person-rooted, so this is a large, LDBC-chosen pool
/// rather than the handful one file would give.
pub(crate) fn load_person_params() -> Vec<String> {
    use std::collections::BTreeSet;
    let mut ids: BTreeSet<String> = BTreeSet::new();
    let Ok(rd) = std::fs::read_dir(DIR) else {
        return Vec::new();
    };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if !name.starts_with("interactive_") || !name.ends_with("_param.txt") {
            continue;
        }
        let Ok(f) = File::open(format!("{DIR}/{name}")) else {
            continue;
        };
        let mut r = BufReader::new(f);
        let mut head = String::new();
        if r.read_line(&mut head).is_err() || !head.starts_with("personId") {
            continue;
        }
        for line in r.lines().map_while(|l| l.ok()) {
            if let Some(id) = line.split('|').next() {
                if !id.is_empty() {
                    ids.insert(id.to_string());
                }
            }
        }
    }
    ids.into_iter().collect()
}
