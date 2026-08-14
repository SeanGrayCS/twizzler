//! IS2: which term of the ordering cost dominates.
//!
//! That single number is compatible with at least four stories, and the project
//! has overturned a paper argument three times in one day by acting on one of
//! them:
//!
//! 1. walk — following ~n incoming `hasCreator` entries and resolving each
//!    neighbour record is what costs, in which case no index helps, because
//!    every candidate must be visited to be a candidate at all;
//! 2. property — the n property reads dominate, which is the case an
//!    ordering index would remove;
//! 3. amplification — the ordering step performs *more* than n reads;
//! 4. sort — the comparison sort itself.
//!
//! They are separated here by slope against degree, not by a mean. LDBC's
//! own substitution-parameter persons span in-degrees from 0 to 2 652 (p50 162,
//! mean 359 — counted host-side from `*_hasCreator_person.csv`), so a fit of
//! latency against `n`, `n·log₂n`, and a constant tells the four apart in one
//! run. A mean cannot: every story predicts "slow".
//!
//! # Arms, and why they are separate invocations
//!
//! Each arm runs the *whole* person set, and only one arm runs per invocation.
//! Running them back to back in one boot would let the first arm warm the pages
//! every later arm reads, which is the difference between the terms being
//! measured and the terms being ordered by whoever went first. Within-boot
//! variance on this workload reaches 3.3× and cross-boot ~1.2×, so the cheap
//! comparison is the wrong one.
//!
//! ```text
//! gstress is2 walk  [iters]   hop only: t = find + walk(n)
//! gstress is2 props [iters]   hop + one property read per candidate
//! gstress is2 full  [iters]   the real IS2, unchanged
//! gstress is2 sort  [iters]   sort pre-read keys; no graph reads in the timed region
//! ```
//!
//! Clear `target/disk-x86_64-unknown-twizzler.img` and boot fresh per arm.
//!
//! # What it prints
//!
//! A `GSTRESS IS2ROW` line per person — degree, microseconds, property-read
//! count — so the regression is done on the data rather than asserted from the
//! percentiles, plus a per-degree-bucket summary for reading by eye.
//!
//! `reads/deg` is the falsifiable number. An ordering step must read each
//! candidate's property once, so this ratio must be 1.0. If it is ~`2·log₂n`
//! instead — ~15 at the median person, ~23 at the largest — then story 3 holds
//! and the fix is a decorate–sort–undecorate in `sort_by_key_opt`, which costs
//! nothing on the write path and needs no index. Deciding between stories 1 and
//! 2 only matters *after* that, which is the point of measuring before building.

use std::time::Instant;

use twizzler_graph::{Graph, Labels, Lookup, PropValue, VertexId, DEFAULT_ARENA_CAP};

const NAME: &str = "ldbc";
const KEY: &str = "creationDate";
const CREATOR: Labels<'static> = Labels::These(&["hasCreator"]);

/// Percentiles, as `ldbc_query` reports them: a mean hides the tail and the
/// tail is where the degree distribution lives.
struct Lat {
    name: &'static str,
    us: Vec<u128>,
}

impl Lat {
    fn new(name: &'static str) -> Self {
        Lat { name, us: Vec::new() }
    }
    fn report(&mut self, reads: usize, cands: usize) {
        if self.us.is_empty() {
            println!("GSTRESS IS2 {:<6} no samples", self.name);
            return;
        }
        self.us.sort_unstable();
        let n = self.us.len();
        let at = |p: f64| self.us[((n as f64 - 1.0) * p) as usize];
        let mean: u128 = self.us.iter().sum::<u128>() / n as u128;
        println!(
            "GSTRESS IS2 {:<6} n={:<5} mean {:>8}us  p50 {:>8}us  p95 {:>8}us  \
             p99 {:>8}us  max {:>8}us | candidates {} reads {} reads/cand {:.2}",
            self.name,
            n,
            mean,
            at(0.50),
            at(0.95),
            at(0.99),
            self.us[n - 1],
            cands,
            reads,
            reads as f64 / cands.max(1) as f64
        );
    }
}

/// One measured person.
struct Row {
    deg: usize,
    us: u128,
    reads: usize,
}

pub(crate) fn run(arm: &str, iters: usize) {
    println!(
        "GSTRESS STAMP harness={} mode=is2 arm={} iters={}",
        crate::HARNESS_REV,
        arm,
        iters
    );
    if !matches!(arm, "walk" | "props" | "full" | "sort") {
        println!(
            "GSTRESS IS2 FAILED: unknown arm `{arm}` — expected walk | props | \
             full | sort, one per boot"
        );
        return;
    }

    let t = Instant::now();
    let g = match Graph::open_or_create_arena(NAME, DEFAULT_ARENA_CAP) {
        Ok(g) => g,
        Err(e) => {
            println!("GSTRESS IS2 FAILED: cannot open `{NAME}`: {e:?} — run `gstress ldbc` first");
            return;
        }
    };
    println!(
        "GSTRESS IS2 open: {:.2}s, {} arenas",
        t.elapsed().as_secs_f64(),
        g.arena_count()
    );

    // Force the lazy index build before timing, as `ldbc_query` does: without
    // it the first query of the run measures a rebuild over 1.8 M records and
    // supplies most of the arm's mean.
    let t = Instant::now();
    let _ = g.find_vertex("person", "warmup-nonexistent");
    println!(
        "GSTRESS IS2 index build: {:.2}s (excluded)",
        t.elapsed().as_secs_f64()
    );

    let persons = crate::ldbc_query::load_person_params();
    if persons.is_empty() {
        println!("GSTRESS IS2 FAILED: no interactive_*_param.txt in /initrd");
        return;
    }
    println!("GSTRESS IS2 params: {} LDBC person ids", persons.len());

    let mut lat = Lat::new(match arm {
        "walk" => "walk",
        "props" => "props",
        "sort" => "sort",
        _ => "full",
    });
    let mut rows: Vec<Row> = Vec::new();
    let mut total_reads = 0usize;
    let mut total_cands = 0usize;

    for i in 0..iters {
        let pid = &persons[i % persons.len()];
        let Lookup::Found(p) = g.find_vertex("person", pid) else {
            continue;
        };

        // Degree, outside the timed region: it is the independent variable, not
        // part of any arm's cost.
        let deg = g.traversal().v(p).in_(CREATOR).count();
        if deg == 0 {
            continue;
        }

        // The `sort` arm's input is read outside the timed region on purpose —
        // it exists to price the comparison sort alone, which is the term every
        // "we need an index" argument implicitly assumes is small.
        let mut keys: Vec<(Option<PropValue>, VertexId)> = if arm == "sort" {
            g.traversal()
                .v(p)
                .in_(CREATOR)
                .to_ids()
                .into_iter()
                .map(|v| (g.get_vertex_prop(v, KEY), v))
                .collect()
        } else {
            Vec::new()
        };

        g.reset_prop_reads();
        let t = Instant::now();
        match arm {
            // Story 1: the hop and the neighbour resolves, with no property
            // read at all. Whatever this costs, no index can remove it.
            "walk" => {
                let n = g.traversal().v(p).in_(CREATOR).count();
                std::hint::black_box(n);
            }
            // Story 2: the same hop plus exactly one property read per
            // candidate — the floor an ordering step could ever reach without
            // an index, and the thing an index would replace.
            "props" => {
                let vals = g.traversal().v(p).in_(CREATOR).values(KEY);
                std::hint::black_box(vals.len());
            }
            // Story 4: the sort, on keys already in hand.
            "sort" => {
                keys.sort_by(|(ka, a), (kb, b)| match (ka, kb) {
                    (Some(x), Some(y)) => y.cmp(x).then(a.0.cmp(&b.0)),
                    (Some(_), None) => core::cmp::Ordering::Less,
                    (None, Some(_)) => core::cmp::Ordering::Greater,
                    (None, None) => a.0.cmp(&b.0),
                });
                std::hint::black_box(keys.first());
            }
            // IS2 as `ldbc_query` runs it. The difference between this and
            // `props` is story 3, and `reads/cand` names it directly.
            _ => {
                let ids = g
                    .traversal()
                    .v(p)
                    .in_(CREATOR)
                    .order_by_prop_desc(KEY)
                    .limit(10)
                    .to_ids();
                std::hint::black_box(ids.len());
            }
        }
        let us = t.elapsed().as_micros();
        let reads = g.prop_reads();

        lat.us.push(us);
        total_reads += reads;
        total_cands += deg;
        rows.push(Row { deg, us, reads });

        println!("GSTRESS IS2ROW arm={arm} deg={deg} us={us} reads={reads}");
    }

    lat.report(total_reads, total_cands);
    buckets(arm, &rows);
    println!(
        "GSTRESS IS2 NOTE: one arm per boot — running arms back to back lets the \
         first warm the pages the rest read. `reads/cand` must be 1.00 for the \
         `full` arm; ~2·log2(n) means the ordering step is re-reading its key \
         inside the comparator, and the remedy is not an index. Degrees are \
         LDBC's own substitution-parameter persons (host-side census at SF0.1: \
         p50 162, mean 359, max 2652, 3 of 87 authored nothing)."
    );
}

/// Latency by degree bucket: the slope, without needing the host to fit
/// anything. A cost linear in `n` has a flat `us/cand`; a cost in `n·log₂n`
/// grows with the bucket; a fixed cost falls.
fn buckets(arm: &str, rows: &[Row]) {
    const EDGES: [usize; 6] = [16, 64, 128, 256, 1024, usize::MAX];
    println!("GSTRESS IS2 buckets (arm={arm}):");
    let mut lo = 0usize;
    for hi in EDGES {
        let label = if hi == usize::MAX {
            format!("[{lo:>5}..    +)")
        } else {
            format!("[{lo:>5}..{hi:<5})")
        };
        let mut us: Vec<u128> = Vec::new();
        let mut cands = 0usize;
        let mut reads = 0usize;
        for r in rows.iter().filter(|r| r.deg >= lo && r.deg < hi) {
            us.push(r.us);
            cands += r.deg;
            reads += r.reads;
        }
        if !us.is_empty() {
            us.sort_unstable();
            let n = us.len();
            let sum: u128 = us.iter().sum();
            println!(
                "GSTRESS IS2   deg {} n={:<4} p50 {:>8}us  mean/cand {:>8.2}us  \
                 reads/cand {:.2}",
                label,
                n,
                us[n / 2],
                sum as f64 / cands.max(1) as f64,
                reads as f64 / cands.max(1) as f64
            );
        }
        lo = hi;
    }
}
