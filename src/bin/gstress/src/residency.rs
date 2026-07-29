//!   gstress residency cycle [N] [R]  # decisive: R rounds of N objects,
//!                                    # dropping every handle between rounds
//!   gstress residency hold [N]       # ceiling probe: allocate holding all
//!   gstress residency volatile [N]   # same as hold, non-persistent objects

use std::time::Instant;

use twizzler::{
    marker::{BaseType, Invariant},
    object::{Object, ObjectBuilder, TypedObject},
};

#[derive(Clone, Copy)]
#[repr(C)]
struct Cell {
    v: u64,
}
unsafe impl Invariant for Cell {}
impl BaseType for Cell {}

fn make(i: u64, persist: bool) -> Object<Cell> {
    ObjectBuilder::<Cell>::default()
        .persist(persist)
        .build(Cell { v: i })
        .expect("create object")
}

/// Heartbeat every this many objects. Small enough that a stall localises the
/// ceiling to within a few objects.
const STEP: usize = 25;

pub(crate) fn run(arm: Option<&str>, n: usize, rounds: usize) {
    println!(
        "GSTRESS STAMP harness={} mode=residency arm={} N={} R={}",
        crate::HARNESS_REV,
        arm.unwrap_or("cycle"),
        n,
        rounds
    );
    match arm {
        None | Some("cycle") => cycle(n, rounds),
        Some("hold") => hold(n, true),
        Some("volatile") => hold(n, false),
        // `rounds` doubles as the per-object element count here.
        Some("write") => write_hold(n, rounds),
        Some("ptr") => ptr_hold(n, rounds),
        Some("ptrcycle") => ptr_cycle(n, rounds),
        // `rounds` doubles as the per-link reference count here.
        Some("ptrwide") => ptr_wide(n, rounds),
        Some("ctrl") => ctrl_hold(n, rounds),
        Some("map") => map_churn(n, rounds),
        Some(other) => {
            println!(
                "usage: gstress residency \
                 [cycle|hold|volatile|write|ptr|ptrcycle|ptrwide|ctrl|map] [N] \
                 [R|elems|pool|width] (got '{other}')"
            );
            std::process::exit(2);
        }
    }
}

/// The arm that actually models the engine. `hold`/`cycle` create an object and
/// write 8 bytes to it; the engine creates an object, pushes many elements into
/// a `VecObject`, and syncs — which is what fills the pager's *page cache*, and
/// the page cache is what was at 75% when the suite died.
fn write_hold(n: usize, elems: usize) {
    use twizzler::collections::vec::{VecObject, VecObjectAlloc};

    println!(
        "residency write: {} persistent VecObjects, {} u64 pushed and synced into each",
        n, elems
    );
    println!("residency: compare the stall point here against `hold` -- if it is much lower, the");
    println!("residency: constraint is dirty pages, not object count (see A6 in docs/tasks.md)");

    let start = Instant::now();
    let mut held: Vec<VecObject<Cell, VecObjectAlloc>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut v: VecObject<Cell, VecObjectAlloc> =
            VecObject::new(ObjectBuilder::default().persist(true)).expect("create vecobject");
        for j in 0..elems {
            v.push(Cell { v: j as u64 }).expect("push");
            if i == 0 && (j + 1) % STEP == 0 {
                println!(
                    "residency write: object 1, push {}/{} t={:.1}s (per-push sync: this is \
                     the slow path)",
                    j + 1,
                    elems,
                    start.elapsed().as_secs_f64()
                );
            }
        }
        held.push(v);

        // After the first object, project the whole run and refuse to burn
        // hours on impossible parameters.
        if i == 0 {
            let per_obj = start.elapsed().as_secs_f64();
            let projected = per_obj * n as f64;
            println!(
                "residency write: object 1 took {:.1}s -> projected {:.1} min for {} objects",
                per_obj,
                projected / 60.0,
                n
            );
            if projected > 900.0 {
                let suggest = ((900.0 / per_obj) as usize).max(1);
                println!(
                    "residency write: ABORTING -- {:.1} min is not a usable experiment. \
                     Re-run with fewer elems (try `write {} 64`) or N<={}.",
                    projected / 60.0,
                    n,
                    suggest
                );
                return;
            }
        }
        if (i + 1) % STEP == 0 {
            println!(
                "residency write: {}/{} objects x {} elems t={:.1}s",
                i + 1,
                n,
                elems,
                start.elapsed().as_secs_f64()
            );
        }
    }
    println!(
        "residency write: reached {} live objects ({} elems each) in {:.1}s without stalling \
         -- ceiling is ABOVE {} objects at this write volume; raise N or elems until it stalls.",
        n,
        elems,
        start.elapsed().as_secs_f64(),
        n
    );
    drop(held);
}

/// The decisive arm. Allocate `n` objects holding every handle, drop them all,
/// repeat `rounds` times. Never more than `n` live at once, but `n * rounds`
/// created in total.
fn cycle(n: usize, rounds: usize) {
    println!(
        "residency cycle: {} rounds x {} objects = {} created, max {} live at once",
        rounds,
        n,
        n * rounds,
        n
    );
    println!("residency: if frames are NOT returned on drop, this stalls near round {}",
        (2900 / n.max(1)).max(1));

    let start = Instant::now();
    let mut first_round_secs = 0.0f64;
    for r in 1..=rounds {
        let t0 = Instant::now();
        let mut held: Vec<Object<Cell>> = Vec::with_capacity(n);
        for i in 0..n {
            held.push(make(i as u64, true));
            if (i + 1) % STEP == 0 {
                println!(
                    "residency cycle: round {}/{} obj {}/{} (total {}) t={:.1}s",
                    r,
                    rounds,
                    i + 1,
                    n,
                    (r - 1) * n + i + 1,
                    start.elapsed().as_secs_f64()
                );
            }
        }
        // The measurement: everything this round becomes unreachable here.
        drop(held);
        let secs = t0.elapsed().as_secs_f64();
        if r == 1 {
            first_round_secs = secs;
        }
        let ratio = if first_round_secs > 0.0 {
            secs / first_round_secs
        } else {
            1.0
        };
        println!(
            "residency cycle: round {}/{} done in {:.1}s ({:.2}x round 1), {} created so far",
            r,
            rounds,
            secs,
            ratio,
            r * n
        );
    }

    println!(
        "residency cycle: completed {} objects ({} rounds x {}) in {:.1}s without stalling.",
        n * rounds,
        rounds,
        n,
        start.elapsed().as_secs_f64()
    );
    println!(
        "residency cycle: INCONCLUSIVE ALONE -- now run `gstress residency hold {}` in a \
         fresh boot. If that stalls, drop returns frames. If it also completes, this run \
         applied no memory pressure and proves nothing; raise N*R and repeat.",
        n * rounds
    );
}

/// The prime suspect. `hold`, `write` and `cycle` all create objects that
/// reference nothing. The engine is nothing *but* cross-object references: an
/// edge object holds an `InvPtr` to each endpoint, and every adjacency entry
/// holds two more. Each `InvPtr` costs an FOT entry, and resolving one requires
/// the target object mapped — so the resident set may track *references*, not
/// objects. Nothing we have measured would have caught that.
fn ptr_hold(n: usize, pool: usize) {
    use twizzler::ptr::InvPtr;

    #[repr(C)]
    struct Link {
        a: InvPtr<Cell>,
        b: InvPtr<Cell>,
    }
    unsafe impl Invariant for Link {}
    impl BaseType for Link {}

    let pool = pool.max(2);
    println!(
        "residency ptr: {} target objects, then {} link objects with 2 InvPtrs each \
         ({} FOT entries total)",
        pool,
        n,
        n * 2
    );
    println!("residency: compare the stall point against `write` at the same N -- a much lower");
    println!("residency: ceiling here means references, not objects, are what we cannot afford");

    let start = Instant::now();
    let targets: Vec<Object<Cell>> = (0..pool).map(|i| make(i as u64, true)).collect();
    println!(
        "residency ptr: {} targets created in {:.1}s",
        pool,
        start.elapsed().as_secs_f64()
    );

    let mut links: Vec<Object<Link>> = Vec::with_capacity(n);
    for i in 0..n {
        // Spread the references across the pool the way a graph's edges do,
        // rather than hammering one object.
        let x = &targets[i % pool];
        let y = &targets[(i * 7 + 3) % pool];
        let l = ObjectBuilder::<Link>::default()
            .persist(true)
            .build_inplace(|tx| {
                let l = Link {
                    a: InvPtr::new(&tx, x.base_ref())?,
                    b: InvPtr::new(&tx, y.base_ref())?,
                };
                tx.write(l)
            })
            .expect("create link object");
        links.push(l);
        if (i + 1) % STEP == 0 {
            println!(
                "residency ptr: {}/{} links ({} InvPtrs) t={:.1}s",
                i + 1,
                n,
                (i + 1) * 2,
                start.elapsed().as_secs_f64()
            );
        }
    }
    println!(
        "residency ptr: reached {} links / {} InvPtrs over a {}-object pool in {:.1}s without \
         stalling -- ceiling is ABOVE this; raise N until it stalls.",
        n,
        n * 2,
        pool,
        start.elapsed().as_secs_f64()
    );
    drop(links);
    drop(targets);
}

/// Is the reference ceiling on *resident* references or on *created* ones?
///
/// `rounds` batches of `n/rounds` links each, dropped between batches: `n`
/// references created in total, at most `n/rounds` live. Run it with an `n`
/// well past the ~3 000 wall.
fn ptr_cycle(n: usize, rounds: usize) {
    use twizzler::ptr::InvPtr;

    #[repr(C)]
    struct Link {
        a: InvPtr<Cell>,
        b: InvPtr<Cell>,
    }
    unsafe impl Invariant for Link {}
    impl BaseType for Link {}

    let rounds = rounds.max(1);
    let per = (n / rounds).max(1);
    const POOL: usize = 64;
    println!(
        "residency ptrcycle: {} rounds x {} links = {} links / {} InvPtrs created, \
         max {} links live at once",
        rounds,
        per,
        rounds * per,
        rounds * per * 2,
        per
    );
    println!(
        "residency: `ptr` stalled at ~1 625 links / ~3 250 InvPtrs held. If this passes that \
         while dropping, the ceiling is on RESIDENT references and a working-set bound is \
         sufficient -- A6 returns. If it stalls at the same total, the ceiling is on created \
         references and no allocator can help."
    );

    let start = Instant::now();
    let targets: Vec<Object<Cell>> = (0..POOL).map(|i| make(i as u64, true)).collect();

    let mut first = 0.0f64;
    for r in 1..=rounds {
        let t0 = Instant::now();
        let mut links: Vec<Object<Link>> = Vec::with_capacity(per);
        for i in 0..per {
            let x = &targets[i % POOL];
            let y = &targets[(i * 7 + 3) % POOL];
            let l = ObjectBuilder::<Link>::default()
                .persist(true)
                .build_inplace(|tx| {
                    let l = Link {
                        a: InvPtr::new(&tx, x.base_ref())?,
                        b: InvPtr::new(&tx, y.base_ref())?,
                    };
                    tx.write(l)
                })
                .expect("create link object");
            links.push(l);
        }
        drop(links);
        let secs = t0.elapsed().as_secs_f64();
        if r == 1 {
            first = secs;
        }
        println!(
            "residency ptrcycle: round {}/{} done in {:.1}s ({:.2}x round 1) -- {} links / \
             {} InvPtrs created so far, t={:.1}s",
            r,
            rounds,
            secs,
            if first > 0.0 { secs / first } else { 1.0 },
            r * per,
            r * per * 2,
            start.elapsed().as_secs_f64()
        );
    }
    println!(
        "residency ptrcycle: created {} links / {} InvPtrs (max {} live) in {:.1}s without \
         stalling. Compare against `ptr`, which stalled holding ~3 250.",
        rounds * per,
        rounds * per * 2,
        per,
        start.elapsed().as_secs_f64()
    );
    drop(targets);
}

/// Is the FOT charge per object, or per entry?
///
/// Two design points that would otherwise flatten the curve being measured:
/// `insert_fot` deduplicates identical entries, so every slot in a link must point
/// at a *different* target or the object ends up with one FOT entry regardless of
/// width; and `POOL` is fixed rather than derived from the width, so the target
/// baseline is identical in every run and drops out of the comparison.
fn ptr_wide(n: usize, width: usize) {
    match width {
        1 => wide_hold::<1>(n),
        2 => wide_hold::<2>(n),
        4 => wide_hold::<4>(n),
        8 => wide_hold::<8>(n),
        16 => wide_hold::<16>(n),
        32 => wide_hold::<32>(n),
        other => {
            println!(
                "residency ptrwide: width must be one of 1, 2, 4, 8, 16, 32 (got {other}) -- \
                 widths are const-generic, so the set is fixed at compile time. Note the \
                 default R is 10, so pass the width explicitly."
            );
            std::process::exit(2);
        }
    }
}

/// Fixed target pool, shared by every width so the baseline cancels. Must be at
/// least the largest width [`ptr_wide`] dispatches.
const WIDE_POOL: usize = 32;

fn wide_hold<const W: usize>(n: usize) {
    use twizzler::ptr::InvPtr;

    // Named `N` rather than `W` so the struct's parameter is never confused with
    // the enclosing function's: an item declared inside a function cannot use that
    // function's generics, so these are genuinely distinct.
    #[repr(C)]
    struct WideLink<const N: usize> {
        refs: [InvPtr<Cell>; N],
    }
    unsafe impl<const N: usize> Invariant for WideLink<N> {}
    impl<const N: usize> BaseType for WideLink<N> {}

    println!(
        "residency ptrwide: {} link objects x {} InvPtrs = {} FOT entries, over a fixed \
         {}-object pool",
        n,
        W,
        n * W,
        WIDE_POOL
    );
    println!(
        "residency: compare the heartbeat `a:` delta per link against other widths at the \
         same N. Flat across W means the FOT charge is per object and packing suffices; \
         linear in W means it is per entry and A4b is required."
    );

    let start = Instant::now();
    let targets: Vec<Object<Cell>> = (0..WIDE_POOL).map(|i| make(i as u64, true)).collect();
    println!(
        "residency ptrwide: {} targets created in {:.1}s",
        WIDE_POOL,
        start.elapsed().as_secs_f64()
    );

    let mut links: Vec<Object<WideLink<W>>> = Vec::with_capacity(n);
    for i in 0..n {
        let l = ObjectBuilder::<WideLink<W>>::default()
            .persist(true)
            .build_inplace(|tx| {
                // `InvPtr` is not `Copy`, so array-repeat syntax is unavailable and
                // the fallible constructor cannot run inside `from_fn`. Build nulls,
                // then fill in place.
                let mut refs: [InvPtr<Cell>; W] = std::array::from_fn(|_| InvPtr::null());
                for (k, r) in refs.iter_mut().enumerate() {
                    // `(i + k) % WIDE_POOL` gives W distinct targets for any
                    // W <= WIDE_POOL, so no two slots dedup onto one FOT entry,
                    // while varying with `i` so links are not all identical.
                    *r = InvPtr::new(&tx, targets[(i + k) % WIDE_POOL].base_ref())?;
                }
                tx.write(WideLink { refs })
            })
            .expect("create wide link object");
        links.push(l);
        if (i + 1) % STEP == 0 {
            println!(
                "residency ptrwide: {}/{} links ({} InvPtrs) t={:.1}s",
                i + 1,
                n,
                (i + 1) * W,
                start.elapsed().as_secs_f64()
            );
        }
    }
    println!(
        "residency ptrwide: reached {} links / {} InvPtrs at width {} in {:.1}s without \
         stalling -- ceiling is ABOVE this at this width.",
        n,
        n * W,
        W,
        start.elapsed().as_secs_f64()
    );
    drop(links);
    drop(targets);
}

/// Control for [`ptr_hold`]. `ptr` differs from `write`/`hold` in *two*
/// ways, not one: it holds `InvPtr`s, and it builds through `build_inplace`
/// with a transaction instead of plain `build`. This arm is `ptr` with the
/// references removed and everything else identical — same transactional
/// creation path, same struct size, same pool walk, two `u64`s where the
/// `InvPtr`s were.
///
/// If `ctrl` stalls where `ptr` stalled, the cost is transactional object
/// creation and the `InvPtr` conclusion is wrong. If `ctrl` runs to 10 000 like
/// `write` did, references are confirmed as the binding constraint.
fn ctrl_hold(n: usize, pool: usize) {
    #[repr(C)]
    struct NoLink {
        a: u64,
        b: u64,
    }
    unsafe impl Invariant for NoLink {}
    impl BaseType for NoLink {}

    let pool = pool.max(2);
    println!(
        "residency ctrl: {} targets, then {} objects built the SAME way as `ptr` \
         (build_inplace + tx) but holding two u64 instead of two InvPtr",
        pool, n
    );
    println!("residency: this isolates transactional creation from the references themselves");

    let start = Instant::now();
    let targets: Vec<Object<Cell>> = (0..pool).map(|i| make(i as u64, true)).collect();
    println!(
        "residency ctrl: {} targets created in {:.1}s",
        pool,
        start.elapsed().as_secs_f64()
    );

    let mut links: Vec<Object<NoLink>> = Vec::with_capacity(n);
    for i in 0..n {
        let l = ObjectBuilder::<NoLink>::default()
            .persist(true)
            .build_inplace(|tx| {
                tx.write(NoLink {
                    a: (i % pool) as u64,
                    b: ((i * 7 + 3) % pool) as u64,
                })
            })
            .expect("create control object");
        links.push(l);
        if (i + 1) % STEP == 0 {
            println!(
                "residency ctrl: {}/{} objects t={:.1}s",
                i + 1,
                n,
                start.elapsed().as_secs_f64()
            );
        }
    }
    println!(
        "residency ctrl: reached {} transactionally-created objects in {:.1}s without \
         stalling -- creation path is NOT the cost; compare against `ptr` at the same N.",
        n,
        start.elapsed().as_secs_f64()
    );
    drop(links);
    drop(targets);
}

/// Creates `n` objects, drops every creation handle, then maps each by id
/// `rounds` times, holding nothing.
fn map_churn(n: usize, rounds: usize) {
    println!(
        "residency map: create {} objects, drop all handles, then map each by id {} times",
        n, rounds
    );
    println!("residency: if this stalls, dropped handles are not released and A6's LRU is dead");

    let start = Instant::now();
    let ids: Vec<twizzler::object::ObjID> = (0..n)
        .map(|i| {
            let o = make(i as u64, true);
            let id = o.id();
            drop(o);
            id
        })
        .collect();
    println!(
        "residency map: {} objects created and released in {:.1}s",
        n,
        start.elapsed().as_secs_f64()
    );

    for r in 1..=rounds {
        for (k, id) in ids.iter().enumerate() {
            let o = Object::<Cell>::map(
                *id,
                twizzler::object::MapFlags::READ | twizzler::object::MapFlags::PERSIST,
            )
            .expect("map by id");
            drop(o);
            if (k + 1) % (STEP * 4) == 0 {
                println!(
                    "residency map: round {}/{} mapped {}/{} t={:.1}s",
                    r,
                    rounds,
                    k + 1,
                    n,
                    start.elapsed().as_secs_f64()
                );
            }
        }
        println!("residency map: round {}/{} done t={:.1}s", r, rounds,
            start.elapsed().as_secs_f64());
    }
    println!(
        "residency map: {} map-and-drop operations over {} distinct ids in {:.1}s without \
         stalling -- handle drop does release, and the id cache is not an unbounded leak.",
        n * rounds,
        n,
        start.elapsed().as_secs_f64()
    );
}

fn hold(n: usize, persist: bool) {
    let kind = if persist { "persistent" } else { "volatile" };
    println!(
        "residency hold: allocating up to {} {} objects, holding every handle",
        n, kind
    );
    println!("residency: a stall here is the expected result -- the last line is the ceiling");

    let start = Instant::now();
    let mut held: Vec<Object<Cell>> = Vec::with_capacity(n);
    for i in 0..n {
        held.push(make(i as u64, persist));
        if (i + 1) % STEP == 0 {
            println!(
                "residency hold ({}): {}/{} objects t={:.1}s",
                kind,
                i + 1,
                n,
                start.elapsed().as_secs_f64()
            );
        }
    }
    println!(
        "residency hold ({}): reached {} live objects in {:.1}s without stalling -- the \
         ceiling is ABOVE {}. This is not yet a result: re-run with a larger N (double it) \
         until it stalls, because every other arm is interpreted relative to this number.",
        kind,
        n,
        start.elapsed().as_secs_f64(),
        n
    );
    drop(held);
}
