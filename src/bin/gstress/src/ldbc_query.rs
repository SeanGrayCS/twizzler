//! LDBC-SNB Interactive short reads (IS1–IS7) against the loaded SF0.1 graph,
//! reporting the benchmark's own metric — per-query latency.
//!
//! Opens the graph `gstress ldbc` built and left on disk: load and query run
//! in separate boots, so the numbers are read from a graph that survived a
//! reboot and durability is exercised by the benchmark itself.
//!
//! # Message is a supertype, and our engine has no supertypes
//!
//! LDBC models `Message` as the parent of `Comment` and `Post`. IS4–IS7 take a
//! message id, which may be either. A vertex here carries exactly one label,
//! so a message id is resolved by trying `comment` then `post`, and message
//! traversals name both labels. The engine's identity model is (label, name)
//! with no hierarchy, so the benchmark's type lattice has to be flattened
//! somewhere, and doing it in the query is more honest than inventing a
//! `message` label the data does not have.
//!
//! # Parameters
//!
//! Person ids come from LDBC's own `interactive_*_param.txt` substitution
//! parameters. There are no short-read parameter files: the spec has IS1–IS7
//! parameterised from the driver's runtime state — ids seen in earlier results
//! — so message ids here are taken from IS2's own output for the person under
//! test, which is how the driver chains them.
//!
//! Still not an audited result: the official driver also controls issue rate,
//! mix, and dependency time.

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
    /// Samples split by whether this iteration was the first use of its person
    /// id. The parameter list cycles, so first touches are a small fraction of
    /// the iterations and land in the top of the combined distribution —
    /// exactly where p95 and p99 are read.
    ///
    /// Splitting them tests whether the reported tail is a property of the
    /// engine or an artefact of parameter reuse.
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

    /// p50/p99 for first-touch and repeat samples separately.
    ///
    /// If the tail is parameter reuse, `cold` p50 lands near the combined p99
    /// and `warm` p99 collapses toward `warm` p50. If the tail is inherent, the
    /// two distributions have similar shape and both keep a long tail.
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
            "GSTRESS LDBCSPLIT {:<6} cold n={cn} p50 {c50}us p99 {c99}us | \
             warm n={wn} p50 {w50}us p99 {w99}us | cold/warm p50 {:.1}x | \
             warm p99/p50 {:.1}x",
            self.name,
            if w50 > 0 { c50 as f64 / w50 as f64 } else { 0.0 },
            if w50 > 0 { w99 as f64 / w50 as f64 } else { 0.0 }
        );
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
        // 32–255 B values live in the text tier, and >255 B in the blob tier;
        // without the blob fallback, long `content` values read back as "".
        // Blob bytes here are always flattened CSV text, so a lossy UTF-8 view
        // is exact in practice and safe otherwise.
        _ => match g.get_vertex_text(v, key) {
            Some(t) => t,
            None => g
                .get_vertex_blob(v, key)
                .map(|b| String::from_utf8_lossy(&b).into_owned())
                .unwrap_or_default(),
        },
    }
}

pub(crate) fn run(iters: usize) {
    run_inner(iters, false)
}

/// Same queries, same code path, emitting a per-iteration result digest
/// instead of latencies. Deliberately not a second implementation: an
/// equivalence check written alongside the benchmark can agree with itself
/// while both differ from the engine under test.
pub(crate) fn equiv(iters: usize) {
    run_inner(iters, true)
}

fn run_inner(iters: usize, digest: bool) {
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

    // Force the index build before timing anything: the first `find_vertex`
    // triggers the lazy rebuild over the whole graph, and without a warm-up
    // the first query of the run measures the index, not the query. Only
    // `find_vertex` forces it — `vertices_by_label` scans records and never
    // touches the index.
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

    // Digest mode does exactly one pass over the parameter list. The list
    // cycles, so iterations beyond it are exact repeats: no new information
    // for an equivalence check, at the cost of stdout lines the guest does not
    // have. Equiv is not timed, so there is nothing to average over either.
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
        // The list cycles, so the first pass over it is every id's first touch
        // and everything after is a repeat. No per-id tracking needed.
        let first_touch = i < persons.len();

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
        is1.push(t.elapsed().as_micros(), first_touch);
        is1.results += rows;
        let r1 = rows;

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
        is2.push(t.elapsed().as_micros(), first_touch);
        is2.results += rows;
        let r2 = rows;

        let Some(m0) = msg else {
            // This person authored nothing; IS4-IS7 have no parameter, and
            // timing them against a missing id would measure the miss path.
            //
            // IS3 is skipped here too even though it reads `both(knows)` and
            // never touches a message id — it is excluded only because it sits
            // below this guard, which biases its sample toward persons who
            // authored something. Moving it above the guard changes the sample
            // set, so do it in both arms together or not at all.
            if digest && i < persons.len() {
                println!("EQUIV {pid} is1={r1} is2={r2} mid=- NOMSG");
            }
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
        is3.push(t.elapsed().as_micros(), first_touch);
        is3.results += rows;
        let r3 = rows;

        // IS4 — a message's content and creation date.
        let t = Instant::now();
        let mut rows = 0;
        if let Some(m) = find_message(&g, mid) {
            let _ = str_prop(&g, m, "creationDate");
            let c = str_prop(&g, m, "content");
            rows = usize::from(!c.is_empty());
        }
        is4.push(t.elapsed().as_micros(), first_touch);
        is4.results += rows;
        let r4 = rows;

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
        is5.push(t.elapsed().as_micros(), first_touch);
        is5.results += rows;
        let r5 = rows;

        // IS6 — the forum a message belongs to: walk the replyOf chain to the
        // root post, then up to its forum.
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
        is6.push(t.elapsed().as_micros(), first_touch);
        is6.results += rows;
        let r6 = rows;

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
        is7.push(t.elapsed().as_micros(), first_touch);
        is7.results += rows;
        let r7 = rows;

        if digest && i < persons.len() {
            println!(
                "EQUIV {pid} is1={r1} is2={r2} mid={mid} is3={r3} is4={r4} \
                 is5={r5} is6={r6} is7={r7}"
            );
        }

        if (i + 1) % 200 == 0 {
            println!("GSTRESS LDBCQ .. {}/{iters}", i + 1);
        }
    }

    println!("GSTRESS LDBCQ RESULTS (LDBC-SNB Interactive short reads, SF0.1):");
    for l in [&mut is1, &mut is2, &mut is3, &mut is4, &mut is5, &mut is6, &mut is7] {
        l.report();
    }
    // First-touch versus repeat. See `Lat::report_split` — this is the test of
    // whether the reported tail is the engine's cold path or an artefact of
    // the parameter list cycling.
    println!("GSTRESS LDBCQ SPLIT (first touch of each id vs repeats):");
    for l in [&mut is1, &mut is2, &mut is3, &mut is4, &mut is5, &mut is6, &mut is7] {
        l.report_split();
    }
    println!(
        "GSTRESS LDBCQ NOTE: person ids are LDBC's substitution parameters; \
         message ids come from IS2's own result, as the driver chains them. Not \
         an audited result — the official driver also controls issue rate, mix \
         and dependency time. `MSG` spans `comment`+`post` because the engine \
         has no type hierarchy. {no_msg} of {iters} person *draws* resolved to a \
         person who authored nothing (a smaller number of distinct ids, each \
         drawn repeatedly), so IS3-IS7 have fewer samples than IS1-IS2 — those \
         iterations stop after IS2. IS3 needs no message id and is excluded only \
         because it sits below that guard; see the report's Appendix B."
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
