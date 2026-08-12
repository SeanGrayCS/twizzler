//! Delete is the only operation on this platform that returns memory. The
//! kernel path is complete — `object_ctrl` marks for delete, calls
//! `pager::del_object` and runs `scan_deleted`; if the object has no contexts
//! and no pins the `Arc` drops, the `PageRangeTree` drops, `Page::drop` runs and
//! every frame goes back through `free_frame`. Everything else on this platform
//! retains: the object table is append-only, the per-object page cache has no
//! truncation, and `ObjectEvict` returns `NOT_SUPPORTED` for anything but
//! writeback.
//!
//! # Two arms, because one proves nothing
//!
//! Both arms insert `N` records per cycle for up to `C` cycles and differ in one
//! respect:
//!
//! - `destroy` (default) — each cycle destroys its graph before the next begins.
//! - `keep` — each cycle builds a differently named graph and never destroys
//!   it, so every arena ever created stays in the object table.
//!
//! `keep` is the control and it is not optional. There is no userspace frame
//! counter on this platform (`print_tracker_stats` is reachable only from
//! `MemoryTracker::wait`, i.e. once a thread is already blocked for memory), so
//! the ceiling cannot be read directly — it can only be located by running into
//! it. The first draft of this probe instead *assumed* a ~3 M-record ceiling and
//! sized itself under it; `noindex` had already reached 10 M records in an
//! earlier session, so that version would have printed a pass whether or not
//! `Delete` freed a single frame. A run of `destroy` alone still means nothing.
//! Report the pair or report neither.
//!
//! # Reading it: there is no single wall
//!
//! Three times corrected; read this before trusting any capacity number.
//! Draft 1: exhaustion prints a frame dump. Draft 2: no discrete failure, read
//! the rate *slope*, exhaustion unreachable. Draft 3: "the ceiling is objects,
//! not records — ~1450 arenas".
//!
//! Draft 3 was an inference the data could not support. It rested on two
//! `keep` runs at N=200 000 and N=20 000 that stalled at the same arena count —
//! but both used `cap=256`, so arenas and records were proportional and
//! "1450 arenas" and "371 k records" were the same statement. The 10× difference
//! in records per cycle varied only how fast each run *approached* the stall,
//! never the arenas-to-records ratio, so it discriminated nothing. Varying the
//! cap was the first run capable of telling them apart:
//!
//! What is settled:
//!
//! - A pure record wall is excluded (2.5 M ≫ 550 k).
//! - A fixed arena wall is excluded (625–650 clears cap=1024's 490–588).
//! - cap=4096 carries ~6.9× cap=256's records, so SF0.1 (~2 M) is reachable
//!   with ~25% headroom.
//!
//! What is not settled — and should not be guessed at again:
//!
//! Arenas at death run 1450 → 540 → 637, non-monotonic in cap. No linear
//! `C = A×arenas + B×records` fits: the pairwise solutions disagree by 100×
//! (A = 197B against A = 20619B), and a per-object-plus-data-pages model yields a
//! *negative* fixed cost. There is no capacity formula here. Two prior
//! attempts to state one — `wall × cap`, then "sublinear, ~700–900 k" — were both
//! wrong within a day.
//!
//! Slot exhaustion *is* ruled out: `SLOTS = (1<<47)/MAX_SIZE` = 131 072.
//!
//! Open: whether the cap=256/1024 hangs are the same failure as the cap=4096
//! panic. Only the panic has a cause attached.
//!
//! But it is not full reclaim, and passing the wall does not show that it is.
//! Full reclaim predicts a run that never stalls: the resident set should stay at
//! one cycle's 79 arenas and the rate at cycle 1's 4841 rec/s. Instead `destroy`
//! decayed on the control's own curve (0.44× vs 0.45× of cycle 1), showed the
//! same decay → plateau → stall shape, and stalled. Solving
//! `retained × created × frames_per_arena = ceiling` with the control fixing the
//! ceiling gives `1450/2330 ≈ 0.62`: ~38% of an arena's frames come back, ~62%
//! are retained. Order of magnitude only — n=1 per arm — but "partial" is
//! robust, since full reclaim cannot produce a stall.
//!
//! Confound on the record: `keep` registers a new graph name per cycle and
//! grows the image; `destroy` reuses one name and region, so part of the 1.6×
//! may be disk locality. A third arm — destroy *with* fresh names — separates
//! them.
//!
//! Per-stage timings (`reset open insert drop destroy`) and heartbeats every
//! `N/8` records stay, because they are what made the wall legible: without them
//! an over-long cycle is indistinguishable from a hang. They also ruled out
//! sync-on-drop, `drop` being flat at ~2.7 s while insert climbed.
//!
//! # Sizing
//!
//! Both arms must use the same cap or the comparison is void — the stall moves
//! with cap in both coordinates, so a cross-cap comparison measures the cap.
//!
//!   gstress reclaim [N] [C] [keep] [cap:K]  # default 100 000 × 12, cap 256

use std::time::Instant;

use twizzler::object::ObjID;
use twizzler_graph::Graph;

/// There is no single wall. A pure record wall is excluded (2 M ≫ 550 k). A
/// fixed *arena* wall is not yet excluded: the cap=4096 run ended at 500
/// arenas, inside cap=1024's 490–588 stall bracket, so it stopped just short of
/// the test. Capacity rises strongly with cap — 256→1024 gave 1.5× records,
/// 1024→4096 gave >3.6× — but the last figure is a lower bound.
///
/// A flat plateau does not imply survival: cap=256 held 0.45× for nine
/// cycles and then stalled anyway.
const KEEP_STALL: &[(usize, Option<usize>, Option<usize>)] = &[
    // (arena_cap, arenas at stall, records at stall)
    (256, Some(1450), Some(371_000)),
    (1024, Some(540), Some(550_000)),
    (4096, None, None), // reached 500 arenas / 2 M records, no stall
];

const CYCLE1_REF: &[(usize, f64)] = &[(256, 4081.0), (1024, 13612.0), (4096, 28251.0)];

fn keep_stall_for(cap: usize) -> Option<(usize, usize)> {
    KEEP_STALL
        .iter()
        .find(|(c, _, _)| *c == cap)
        .and_then(|(_, a, r)| Some(((*a)?, (*r)?)))
}

pub(crate) fn run(n: usize, cycles: usize, keep: bool, cap: usize) {
    let arm = if keep { "keep" } else { "destroy" };
    println!(
        "GSTRESS STAMP harness={} mode=reclaim arm={} N={} cycles={} cap={}",
        crate::HARNESS_REV,
        arm,
        n,
        cycles,
        cap
    );
    println!(
        "GSTRESS RECLAIM premise: arm={arm} inserts {n} records/cycle at \
         arena_cap={cap} (~{} arenas/cycle) for up to {cycles} cycles \
         ({} records total). `keep` retains every graph; `destroy` frees each \
         before the next. **Read the rec/s slope across cycles, not survival**; \
         neither arm means anything alone.",
        n.div_ceil(cap.max(1)),
        n * cycles
    );

    let mut total = 0usize;
    let mut total_arenas = 0usize;
    let mut first_insert_rate = 0.0f64;
    for c in 1..=cycles {
        // `keep` needs a fresh name each cycle: reusing one would let
        // `reset_arena` delete the outgoing graph and quietly turn the control
        // into a second copy of the treatment.
        let name = if keep {
            format!("gkeep{c}")
        } else {
            "greclaim".to_string()
        };

        let t = Instant::now();
        if !keep {
            Graph::reset_arena(&name, cap).expect("reset");
        }
        let reset_s = t.elapsed().as_secs_f64();

        // Every stage is timed separately. The first version of this probe
        // timed the cycle as one opaque block, so a slow cycle 2 could not be
        // attributed to open, insert, drop or destroy — and drop is a live
        // suspect, since a sync on drop would scale with resident arenas.
        let t_open = Instant::now();
        let mut g = Graph::open_or_create_arena(&name, cap).expect("open");
        let open_s = t_open.elapsed().as_secs_f64();

        let preexisting = g.record_count();
        if preexisting != 0 {
            println!(
                "GSTRESS RECLAIM VOID: `{name}` already holds {preexisting} \\
                 records. The image was not cleared, so this arm is inserting \\
                 duplicate names into an existing graph and the run is **not \\
                 comparable to any other**. Delete target/disk-<triple>.img and \\
                 start over."
            );
            return;
        }

        let t_ins = Instant::now();
        let hb = (n / 8).max(1);
        g.bulk_insert(|b| {
            for i in 0..n {
                b.add_vertex("n", &format!("v{i}"), ObjID::new(0))?;
                // Heartbeat, per the harness convention: a stall must be
                // visible and attributable while it is happening, not inferred
                // afterwards from a missing line.
                if i > 0 && i % hb == 0 {
                    let e = t_ins.elapsed().as_secs_f64();
                    println!(
                        "GSTRESS RECLAIM {arm} c{c} .. {i}/{n} records, \
                         {e:.1}s, {:.0} rec/s",
                        i as f64 / e.max(1e-9)
                    );
                }
            }
            Ok(())
        })
        .expect("bulk_insert");
        let ins_s = t_ins.elapsed().as_secs_f64();
        let arenas = g.arena_count();

        let t_drop = Instant::now();
        drop(g);
        let drop_s = t_drop.elapsed().as_secs_f64();

        let t_del = Instant::now();
        let freed = if keep {
            0
        } else {
            Graph::destroy(&name).expect("destroy")
        };
        let del_s = t_del.elapsed().as_secs_f64();

        total += n;
        total_arenas += arenas;
        let rate = n as f64 / ins_s.max(1e-9);
        if c == 1 {
            first_insert_rate = rate;
            // Reference is for N=100 000; other N shift the rate, so only check
            // the sizing the references were taken at.
            if n == 100_000 {
                if let Some((_, refr)) = CYCLE1_REF.iter().find(|(k, _)| *k == cap) {
                    let ratio = rate / refr;
                    if !(0.5..2.0).contains(&ratio) {
                        println!(
                            "GSTRESS RECLAIM SUSPECT: cycle 1 at {rate:.0} rec/s \
                             is {ratio:.3}× the clean-image reference {refr:.0} \
                             for cap={cap}. Cycle 1 precedes all residency, so \
                             this is not pressure — suspect a stale image, a \
                             changed workload, or host contention. **Do not \
                             compare this run to another.**"
                        );
                    }
                }
            }
        }
        println!(
            "GSTRESS RECLAIM {arm} cycle {c}/{cycles}: {arenas} arenas | \
             reset {reset_s:.1}s open {open_s:.1}s insert {ins_s:.1}s \
             drop {drop_s:.1}s destroy {del_s:.1}s ({freed} objs) | \
             {rate:.0} rec/s = {:.2}× cycle 1 | cumulative {total} records, \
             {total_arenas} arenas",
            rate / first_insert_rate.max(1e-9)
        );
        if let Some((wall_a, wall_r)) = keep_stall_for(cap) {
            if total_arenas >= wall_a || total >= wall_r {
                println!(
                    "GSTRESS RECLAIM {arm} PAST KEEP STALL: {total_arenas} \
                     arenas / {total} records, past the {wall_a} arenas / \
                     {wall_r} records where `keep` stalled at cap={cap}. For \
                     `destroy` that means frames ARE returned — but **not that \
                     reclaim is complete**: full reclaim never stalls at all, \
                     and the 2026-08-11 destroy arm stalled anyway at ~1.6×. \
                     Watch the rec/s ratio, not this line."
                );
            }
        }
    }

    println!(
        "GSTRESS RECLAIM {arm} COMPLETED {cycles} cycles, {total} records, \
         {total_arenas} arenas cumulative. **The readout is the slope of \
         `rec/s` across cycles, not survival** — compare against the other arm."
    );
}
