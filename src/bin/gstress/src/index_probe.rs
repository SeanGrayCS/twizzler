//! Findings so far, each from an arm below:
//!
//! - The index was 99.5% of a vertex insert (5.35 ms vs 26 µs), because
//!   `PersistentHashMap::insert` opens a transaction per call. `bulk` holds one
//!   open: 115× faster.
//! - `DEFAULT_SEG_CAP` was 64× too small for 16-byte `VertexLoc` records:
//!   733 registry objects at 3 M records against 184 arenas. Raising it gave
//!   2.5×, removed the throughput decay, and cleared a memory ceiling.
//! - Sync runs at 2–4 MB/s, ≈1.5 ms per 4 KB page, consistent across syncs
//!   from 4 MB to 196 MB. Insertion dirties pages ~2–5× faster than writeback
//!   retires them, which is what `overflowing pager queue, waiting...` is.
//!
//! The open question these last two arms settle. A 3 M-record load exhausted
//! memory. Two very different causes fit:
//!
//! They need opposite work, so guessing is expensive.
//!
//!   gstress index [N]                # Graph::add_vertex — record + index
//!   gstress index [N] noindex        # record only, sync at end (baseline)
//!   gstress index [N] bulk           # record + batched index
//!   gstress index [N] sync:K         # record only, sync every K records
//!   gstress index [N] throttle:K     # record only, pause every K records
//!
//! Reading the result. `throttle` slows insertion without changing when
//! pages are written, so it isolates *rate*. `sync:K` bounds how much dirty
//! state can accumulate, so it isolates *backlog depth*.
//!
//! `sync:K` also directly evaluates the write-behind design: sync each chunk as
//! the next is built, rather than one sync at the end. It should not change
//! *total* bytes written — the per-page cost is the same — but it bounds peak
//! dirty state, and the per-chunk timings below show whether sync cost stays
//! flat (clean arenas are free to re-sync) or grows.

use std::time::{Duration, Instant};

use twizzler::object::ObjID;
use twizzler_graph::{ArenaStore, FillTo, Graph, DEFAULT_ARENA_CAP, DEFAULT_SEG_CAP};

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
                }
                Ok(())
            })
            .expect("bulk_insert");
            report("bulk (record + batched index)", n, &t, g.arena_count());
            sync_and_report(|| g.sync().expect("sync"));
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
