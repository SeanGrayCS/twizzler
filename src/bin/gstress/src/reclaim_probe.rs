//! Reclaim probe: does deleting Twizzler objects return frames?
//!
//! Delete is the only operation on this platform that returns memory. The
//! kernel marks the object for deletion and reaps it once nothing maps it;
//! the reaped object's frames go back to the allocator. Everything else
//! retains: the object table is append-only and eviction is unimplemented.
//!
//! # Two arms
//!
//! Both arms insert `N` records per cycle for up to `C` cycles and differ in
//! one respect:
//!
//! - `destroy` (default) — each cycle destroys its graph before the next
//!   begins.
//! - `keep` — each cycle builds a differently named graph and never destroys
//!   it, so every arena ever created stays in the object table.
//!
//! `keep` is the control. Run both arms and report the pair; either alone
//! means nothing.
//!
//! # What it prints
//!
//! Per cycle: per-stage timings (`reset open insert drop destroy`),
//! heartbeats every `N/8` records so a stall is visible and attributable
//! while it happens, a RESIDENT line (resident pages of this cycle's graph)
//! and a CUMULATIVE line (every id the run has ever created, re-read). The
//! `destroy` arm adds RECLAIM (pages before/after its destroy), SWEEP (the
//! same ids after a forced sweep) and CARRY (the previous cycle's ids,
//! re-read a cycle later). Reaping is deferred until nothing maps an object,
//! so the instantaneous RECLAIM figure can sit near zero while SWEEP and
//! CARRY show the frames coming back; those two lines and CUMULATIVE are the
//! measurement.
//!
//! Resident-page counts come from per-object stat calls and are lower bounds:
//! pager-held frames and kernel-side overhead sit outside any object's range
//! tree.
//!
//! # Rules
//!
//! Run each arm in its own boot, on a clean disk image; the probe voids the
//! run if its graph already holds records. Both arms must use the same cap or
//! the comparison is void — the stall point moves with cap.
//!
//! No sync, deliberately. Records occupy resident pages whether or not they
//! are synced, and syncing near the frame ceiling deadlocks: sync
//! write-protects pages, later writes then take CoW faults, and a CoW fault
//! must allocate.
//!
//!   gstress reclaim [N] [C] [keep] [cap:K]  # default 100 000 × 12, cap 256

use std::time::Instant;

use twizzler::object::ObjID;
use twizzler_graph::Graph;

/// Reference `keep` outcomes by arena cap, each taken in its own boot on a
/// clean image. `None` in the stall columns means the reference run finished
/// without stalling — a lower bound, not a stall point. Used to print a
/// marker when a run passes the corresponding stall.
const KEEP_STALL: &[(usize, Option<usize>, Option<usize>)] = &[
    // (arena_cap, arenas at stall, records at stall)
    (256, Some(1450), Some(371_000)),
    (1024, Some(540), Some(550_000)),
    (4096, None, None),
];

/// Reference cycle-1 insertion rates at N=100 000, by cap, on a clean image.
/// Cycle 1 runs before any residency has accumulated, so at fixed (N, cap) it
/// is the same work every time and makes a cross-boot reproducibility check:
/// if it does not reproduce, nothing later in the run is comparable to
/// another run, whatever the cause. The check band is wide (0.5–2.0×) to flag
/// structural breakage, not variance.
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
    // Cross-cycle carry: the line that separates a lag from a leak. Delete
    // marks, and the kernel reaps only objects mapped nowhere, so a cycle's
    // own destroy may reap none of its objects while a later delete sweeps
    // them up (`Graph::reset_arena` at the top of the next cycle calls
    // `delete_all`, adding one more sweep point). The previous cycle's ids are
    // re-statted at the end of this one: pages gone by then mean a one-cycle
    // lag; pages unchanged mean a retention.
    let mut prev: Option<(usize, Vec<u128>, usize)> = None; // (cycle, ids, pages_before)
    // Every id this run has ever created, for the CUMULATIVE line below.
    let mut all_ids: Vec<u128> = Vec::new();
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

        // Every stage is timed separately, so a slow cycle can be attributed
        // to open, insert, drop or destroy rather than guessed at.
        let t_open = Instant::now();
        let mut g = Graph::open_or_create_arena(&name, cap).expect("open");
        let open_s = t_open.elapsed().as_secs_f64();

        // The `keep` arm reuses fixed names across runs, so a stale
        // `target/disk-*.img` makes `open_or_create_arena` open last run's
        // graph rather than create one — every `add_vertex` then re-inserts an
        // already-indexed name into an image holding every graph the previous
        // run wrote. That is a different workload wearing this one's name, and
        // its symptom (uniform slowness) looks like memory pressure, so it is
        // detected here rather than assumed away.
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

        // Resident pages of this cycle's live graph, before it is dropped or
        // destroyed. Both arms report it. Lower bound: pager-held frames and
        // kernel-side per-object overhead sit outside any object's range tree,
        // so a large value establishes pressure and a small one does not rule
        // it out. 2 962 166 is the platform's total frame count.
        let (live_objs, live_pages) = g.resident_pages();
        let live_ids = g.owned_object_ids();
        println!(
            "GSTRESS RECLAIM RESIDENT {arm} c{c} {live_objs} objects \
             {live_pages} pages ({:.2}% of 2962166 frames, lower bound)",
            live_pages as f64 * 100.0 / 2_962_166.0
        );

        // CUMULATIVE: every id this run has ever created, re-read. RESIDENT
        // above covers only this cycle's graph; in `destroy` that is the whole
        // story, because nothing survives a cycle, but in `keep` a per-graph
        // figure would read flat and hide the accumulation the control exists
        // to show. Costs one stat syscall per id per cycle.
        all_ids.extend(live_ids.iter().copied());
        let (cum_objs, cum_pages) = twizzler_graph::pages_of_ids(&all_ids);
        println!(
            "GSTRESS RECLAIM CUMULATIVE {arm} c{c}: {cum_objs} of {} ids ever \
             created still resolve, {cum_pages} pages ({:.2}% of 2962166 frames, \
             lower bound)",
            all_ids.len(),
            cum_pages as f64 * 100.0 / 2_962_166.0
        );

        let t_drop = Instant::now();
        drop(g);
        let drop_s = t_drop.elapsed().as_secs_f64();

        // The `destroy` arm measures what its deletion returned. The control
        // never destroys, so it has nothing to measure here; the per-cycle
        // RESIDENT line above is the only figure the two arms share.
        let t_del = Instant::now();
        let freed = if keep {
            0
        } else {
            let ids_before = live_ids;
            let carry = prev.take();
            let rep = Graph::destroy_measured(&name).expect("destroy");
            println!(
                "GSTRESS RECLAIM RECLAIM {arm} c{c} pages {}->{} returned {} \
                 ({}) | ids {} accepted {} still-resolving {} | root {}->{}",
                rep.pages_before,
                rep.pages_after,
                rep.returned_pages(),
                match rep.returned_fraction() {
                    Some(f) => format!("{:.1}%", f * 100.0),
                    None => "undefined".to_string(),
                },
                rep.attempted,
                rep.accepted,
                rep.present_after,
                rep.root_pages_before,
                rep.root_pages_after,
            );
            // The eventual fraction. The kernel reaps a deleted object only
            // once nothing maps it; an object still mapped when its delete
            // lands stays in the table until some later delete sweeps it. So
            // the figure above is instantaneous and may be near zero for a
            // reason that has nothing to do with reclaim; this line re-reads
            // the same ids after a forced sweep.
            let swept = twizzler_graph::sweep_deleted_objects();
            let (still, pages_now) = twizzler_graph::pages_of_ids(&ids_before);
            println!(
                "GSTRESS RECLAIM SWEEP {arm} c{c} (ran: {swept}) {still} ids \
                 resolve, {pages_now} pages | eventual return {} of {} ({})",
                rep.pages_before.saturating_sub(pages_now),
                rep.pages_before,
                if rep.pages_before > 0 {
                    format!(
                        "{:.1}%",
                        rep.pages_before.saturating_sub(pages_now) as f64 * 100.0
                            / rep.pages_before as f64
                    )
                } else {
                    "undefined".to_string()
                }
            );
            // Per-structure object inventory, printed once on cycle 1, so the
            // census comes from the run itself rather than a hand count.
            if c == 1 {
                println!("GSTRESS RECLAIM INVENTORY {arm} c1 (per-structure):");
                for line in rep.report_lines() {
                    println!("GSTRESS RECLAIM INVENTORY {line}");
                }
            }
            // `present_after > 0` is expected, not an anomaly: delete marks,
            // and the kernel reaps only what is mapped nowhere. Counted
            // because the trend across cycles is informative.
            if rep.present_after > 0 {
                println!(
                    "GSTRESS RECLAIM PENDING {arm} c{c}: {} of {} ids still \
                     resolve immediately after their delete (expected — reaping \
                     is deferred until the mapping drops). The SWEEP and CARRY \
                     lines say whether they ever go.",
                    rep.present_after, rep.attempted
                );
            }
            if rep.grew() {
                println!(
                    "GSTRESS RECLAIM ANOMALY {arm} c{c}: resident pages ROSE \
                     across the destroy ({} -> {}). Deletion must not increase \
                     residency; treat this run as void until explained.",
                    rep.pages_before, rep.pages_after
                );
            }
            // CARRY: the previous cycle's objects, re-read now. Since they
            // were destroyed, three more sweep points have passed — this
            // cycle's `reset_arena` (which calls `delete_all`), this cycle's
            // destroy, and the sweep above. Pages gone by now mean a lag;
            // pages unchanged after a further cycle of deletions mean a
            // retention.
            if let Some((pc, pids, ppages)) = carry {
                let (still, now) = twizzler_graph::pages_of_ids(&pids);
                let returned = ppages.saturating_sub(now);
                println!(
                    "GSTRESS RECLAIM CARRY {arm} c{c}: cycle {pc}'s {} ids -> \
                     {still} still resolve, {ppages} -> {now} pages \
                     ({returned} returned, {})",
                    pids.len(),
                    if ppages > 0 {
                        format!("{:.1}%", returned as f64 * 100.0 / ppages as f64)
                    } else {
                        "undefined".to_string()
                    }
                );
            }
            prev = Some((c, ids_before, rep.pages_before));
            rep.accepted
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
        // Marker, not a conclusion: this run has passed the point where the
        // reference `keep` run stopped.
        if let Some((wall_a, wall_r)) = keep_stall_for(cap) {
            if total_arenas >= wall_a || total >= wall_r {
                println!(
                    "GSTRESS RECLAIM {arm} PAST KEEP STALL: {total_arenas} \
                     arenas / {total} records, past the {wall_a} arenas / \
                     {wall_r} records where `keep` stalled at cap={cap}. \
                     Reference point only — the CUMULATIVE and CARRY lines are \
                     the measurement."
                );
            }
        }
    }

    println!(
        "GSTRESS RECLAIM {arm} COMPLETED {cycles} cycles, {total} records, \
         {total_arenas} arenas cumulative. **Read CUMULATIVE for residency and \
         CARRY for reclaim.** Completing the configured cycle count is not a \
         ceiling measurement — the run stopped where it was told to, not where \
         the machine stopped it."
    );
}
