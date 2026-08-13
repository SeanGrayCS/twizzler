//! Index probe: what limits a large load — insertion, the index, memory
//! residency, or writeback?
//!
//!   gstress index [N]                # Graph::add_vertex — record + index
//!   gstress index [N] noindex        # record only, sync at end (baseline)
//!   gstress index [N] bulk           # record + batched index
//!   gstress index [N] rebuild        # lazy rebuild cost, RebuildSource::Scan
//!   gstress index [N] rebuild:roots  # same, RebuildSource::Roots — the pair is
//!                                    # the measurement; either alone is not
//!   gstress index [N] sync:K         # record only, sync every K records
//!   gstress index [N] throttle:K     # record only, pause every K records
//!
//! Reading the result: `throttle` slows insertion without changing when pages
//! are written, so it isolates rate. `sync:K` bounds how much dirty state can
//! accumulate, so it isolates backlog depth.
//!
//! - `throttle` completes, `noindex` does not → backlog; pacing is the fix.
//! - `sync:K` completes, `throttle` does not → it is specifically unsynced
//!   pages, not rate; incremental sync is the fix and throttling is not.
//! - neither completes → residency, not writeback.
//! - both complete → either works; prefer `sync:K`, which costs no wall time
//!   that the final sync would not have cost anyway.
//!
//! `sync:K` also evaluates write-behind — sync each chunk as the next is
//! built, rather than one sync at the end. It should not change total bytes
//! written, but it bounds peak dirty state, and the per-chunk timings show
//! whether sync cost stays flat (clean arenas are free to re-sync) or grows.
//!
//! Run each arm in its own boot — ordering within a boot skews throughput.

use std::time::{Duration, Instant};

use twizzler::object::ObjID;
use twizzler_graph::{
    ArenaStore, FillTo, Graph, IndexSchema, IndexStrategy, RebuildSource, DEFAULT_ARENA_CAP,
    DEFAULT_SEG_CAP,
};

const NAME: &str = "gindex";

/// How the record-only arms pace themselves.
struct Pacing {
    /// Sync every this many records (0 = only at the end).
    sync_every: usize,
    /// Pause every this many records (0 = never).
    throttle_every: usize,
}

pub(crate) fn run(n: usize, arm: &str) {
    println!(
        "GSTRESS STAMP harness={} mode=index arm={} N={}",
        crate::HARNESS_REV,
        arm,
        n
    );

    let pacing = |prefix: &str| -> usize {
        arm.strip_prefix(prefix)
            .and_then(|k| k.parse::<usize>().ok())
            .unwrap_or(0)
    };

    let t = Instant::now();
    match arm {
        "bulk" => {
            Graph::reset_arena(NAME, DEFAULT_ARENA_CAP).expect("reset");
            let mut g = Graph::open_or_create_arena(NAME, DEFAULT_ARENA_CAP).expect("open");
            println!("GSTRESS SETUP: {:.2}s (excluded)", t.elapsed().as_secs_f64());
            let t = Instant::now();
            g.bulk_insert(|b| {
                for i in 0..n {
                    b.add_vertex("n", &format!("v{i}"), ObjID::new(0))?;
                    // Without a heartbeat a slow run and a hung one look
                    // identical at this scale.
                    heartbeat(i + 1, n, &t);
                }
                Ok(())
            })
            .expect("bulk_insert");
            report("bulk (record + batched index)", n, &t, g.arena_count());
            sync_and_report(|| g.sync().expect("sync"));
        }
        // What a lazy rebuild costs: load, sync, reopen in the same boot, then
        // time the first lookup (which pays for the rebuild) against the
        // second (which must not).
        //
        // Same-boot reopen, so this is a lower bound. The arenas are still
        // resident, so it measures the walk — reading every record's label and
        // name — and not the fault-in a cold reopen would pay.
        a if a.starts_with("rebuild") => {
            // `rebuild[:roots][:RATIO]` — RATIO indexes one record in every
            // RATIO, the rest under an undeclared label.
            //
            // RATIO is the variable. At 1 every record is indexed and both
            // sources read all N records, so the arms cannot meaningfully
            // differ. The case `Roots` exists for is a small indexed fraction:
            // there `Scan` still walks every record (`vertices_by_label`
            // filters all of `locs`) while `Roots` reads only the roots list.
            // Run the pair at the ratio you actually care about.
            let parts: Vec<&str> = a.split(':').collect();
            let source = if parts.contains(&"roots") {
                RebuildSource::Roots
            } else {
                RebuildSource::Scan
            };
            let ratio = parts
                .iter()
                .find_map(|p| p.parse::<usize>().ok())
                .unwrap_or(1)
                .max(1);
            let schema = IndexSchema::new(IndexStrategy::LazyLabel).rebuild(source);
            println!(
                "GSTRESS REBUILD: source={source:?} ratio=1:{ratio} \
                 ({} of {n} records indexed)",
                n.div_ceil(ratio)
            );
            Graph::reset_arena_with_index(NAME, DEFAULT_ARENA_CAP, schema).expect("reset");
            let mut g = Graph::open_or_create_arena_with_index(NAME, DEFAULT_ARENA_CAP, schema)
                .expect("open");
            g.set_label_indexed("n", true).expect("declare");
            println!("GSTRESS SETUP: {:.2}s (excluded)", t.elapsed().as_secs_f64());

            let t = Instant::now();
            g.bulk_insert(|b| {
                for i in 0..n {
                    // Only `n` is declared; `c` records are the unindexed bulk.
                    if i % ratio == 0 {
                        b.add_vertex("n", &format!("v{i}"), ObjID::new(0))?;
                    } else {
                        b.add_vertex("c", &format!("c{i}"), ObjID::new(0))?;
                    }
                    heartbeat(i + 1, n, &t);
                }
                Ok(())
            })
            .expect("bulk_insert");
            report("rebuild (load phase)", n, &t, g.arena_count());
            assert_eq!(
                g.index_builds(),
                0,
                "A8-AC4: a load that never looks up must not build the index"
            );
            sync_and_report(|| g.sync().expect("sync"));
            drop(g);

            let t = Instant::now();
            let g = Graph::open_or_create_arena(NAME, DEFAULT_ARENA_CAP).expect("reopen");
            let open_s = t.elapsed().as_secs_f64();

            // The first lookup pays for the build; the second must not.
            let t = Instant::now();
            let first = g.find_vertex("n", "v0");
            let cold_s = t.elapsed().as_secs_f64();
            let t = Instant::now();
            // Round to an indexed id, or at ratio>1 this looks up a record that
            // was never indexed and reports NotFound for the wrong reason.
            let mid = (n / 2) / ratio * ratio;
            let second = g.find_vertex("n", &format!("v{mid}"));
            let warm_s = t.elapsed().as_secs_f64();

            println!(
                "GSTRESS REBUILD: source={source:?} | reopen {open_s:.2}s | \
                 first lookup {cold_s:.2}s (builds={}) | second {warm_s:.4}s | \
                 {:.0} rec/s rebuilt | roots tracked {}",
                g.index_builds(),
                n as f64 / cold_s.max(1e-9),
                g.indexed_root_count()
            );
            println!(
                "GSTRESS REBUILD: first={first:?} second={second:?} — both must \
                 be Found, or the rebuild is not reconstructing what the load wrote"
            );
            println!(
                "GSTRESS REBUILD: **lower bound** — same-boot reopen, arenas \
                 still resident, so this is the walk cost without the cold \
                 fault-in. Compare against the 127 s the persistent index cost \
                 per sync, not against zero."
            );

            // Teardown is timed separately: it is the only way to tell
            // "dropping the graph is slow" from "the process will not exit".
            // A rebuild materialises a large in-heap map on top of the
            // resident arenas, so teardown under that pressure can be slow.
            let t = Instant::now();
            drop(g);
            println!(
                "GSTRESS REBUILD teardown: dropped graph in {:.2}s",
                t.elapsed().as_secs_f64()
            );
            println!("GSTRESS REBUILD DONE — anything after this is process exit, not the probe");
        }
        "graph" | "" => {
            Graph::reset_arena(NAME, DEFAULT_ARENA_CAP).expect("reset");
            let mut g = Graph::open_or_create_arena(NAME, DEFAULT_ARENA_CAP).expect("open");
            println!("GSTRESS SETUP: {:.2}s (excluded)", t.elapsed().as_secs_f64());
            let t = Instant::now();
            for i in 0..n {
                g.add_vertex("n", &format!("v{i}"), ObjID::new(0))
                    .expect("add_vertex");
                heartbeat(i + 1, n, &t);
            }
            report("graph (record + index)", n, &t, g.arena_count());
            sync_and_report(|| g.sync().expect("sync"));
        }
        _ => records_only(
            n,
            arm,
            Pacing {
                sync_every: pacing("sync:"),
                throttle_every: pacing("throttle:"),
            },
        ),
    }
}

/// The record-only arms: `noindex`, `sync:K`, `throttle:K`.
fn records_only(n: usize, arm: &str, p: Pacing) {
    let t = Instant::now();
    // `DEFAULT_SEG_CAP`, not a literal: a probe that hardcodes the constant
    // it measures is not a probe.
    let mut s = ArenaStore::create(
        Box::new(FillTo {
            cap: DEFAULT_ARENA_CAP,
        }),
        DEFAULT_SEG_CAP,
    )
    .expect("create store");
    println!("GSTRESS SETUP: {:.2}s (excluded)", t.elapsed().as_secs_f64());
    println!(
        "GSTRESS INDEX: seg_cap={DEFAULT_SEG_CAP} ({} locs segments), \
         sync_every={}, throttle_every={}",
        n.div_ceil(DEFAULT_SEG_CAP),
        p.sync_every,
        p.throttle_every
    );

    // Insert and sync time are tracked apart, because the whole question is
    // which of them dominates and whether moving sync earlier changes the total.
    let t = Instant::now();
    let mut sync_secs = 0.0f64;
    let mut syncs = 0usize;

    for i in 0..n {
        s.add_vertex(0, &format!("v{i}"), 0).expect("add_vertex");
        let done = i + 1;

        if p.sync_every != 0 && done % p.sync_every == 0 {
            let st = Instant::now();
            s.sync_all().expect("periodic sync");
            let el = st.elapsed().as_secs_f64();
            sync_secs += el;
            syncs += 1;
            // Per-chunk, so a growing cost is visible rather than averaged
            // away. Flat means re-syncing clean arenas is free and write-behind
            // works; growing means every sync rewrites the whole store.
            println!(
                "GSTRESS INDEX sync #{syncs} at {done}: {el:.2}s \
                 (cumulative sync {sync_secs:.1}s, insert {:.1}s)",
                t.elapsed().as_secs_f64() - sync_secs
            );
        }
        if p.throttle_every != 0 && done % p.throttle_every == 0 {
            // Pace insertion under writeback capacity without changing *when*
            // pages are written — that is what separates rate from backlog.
            std::thread::sleep(Duration::from_millis(20));
        }
        heartbeat(done, n, &t);
    }

    let wall = t.elapsed().as_secs_f64();
    let insert = wall - sync_secs;
    println!(
        "GSTRESS INDEX {arm}: {n} records — insert {insert:.2}s ({:.0} rec/s), \
         {syncs} periodic sync(s) {sync_secs:.2}s, {} arenas",
        n as f64 / insert.max(1e-9),
        s.arena_count()
    );
    sync_and_report(|| s.sync_all().expect("final sync"));
}

fn sync_and_report(f: impl FnOnce()) {
    let t = Instant::now();
    f();
    println!(
        "GSTRESS INDEX final sync: {:.2}s — **if this dominates, writeback is \
         the limit, not insertion**",
        t.elapsed().as_secs_f64()
    );
}

fn heartbeat(done: usize, total: usize, t: &Instant) {
    if done % 100_000 == 0 {
        let secs = t.elapsed().as_secs_f64();
        let rate = done as f64 / secs.max(1e-9);
        println!(
            "GSTRESS index: {done}/{total} ({rate:.0} ops/s, ~{:.0}s left)",
            (total - done) as f64 / rate.max(1e-9)
        );
    }
}

fn report(arm: &str, n: usize, t: &Instant, arenas: usize) {
    let secs = t.elapsed().as_secs_f64();
    println!(
        "GSTRESS INDEX {arm}: {n} vertices in {secs:.2}s = {:.0} ops/s, {arenas} arenas",
        n as f64 / secs.max(1e-9)
    );
}
