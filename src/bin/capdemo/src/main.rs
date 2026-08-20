//! capdemo — RQ3 / G1: what Twizzler's capability model actually enforces, and
//! whether a graph can hide behind it.
//!
//! **Read `docs/handoffs/G1-NOTES.md` first.** G1's acceptance criteria as
//! written are not achievable. Reasons 1 and 2 are why; 3 and 4 were found by
//! running this binary, which is what it exists for — to confirm or refute on
//! real hardware rather than argue on paper:
//!
//! 1. The engine creates every object it owns through `ObjectBuilder::default()`,
//!    whose `def_prot` is `Protections::all()` — so every graph object is
//!    readable and writable by every security context, and there is nothing to
//!    deny. Steps P2/P3 assert exactly this.
//! 2. A denial is not an error. `sys_object_map` never checks permissions
//!    (`kernel/src/syscall/object.rs:104`, the result is discarded); the check
//!    happens at first touch and arrives as `UpcallInfo::SecurityViolation`,
//!    which the monitor declines and turns into `sys_thread_exit(101)`. Steps
//!    Q3/Q5 assert that the map *succeeds* and the compartment then dies.
//!
//! 3. An unallocated object id does not fail cheaply. **Measured 2026-08-14:**
//!    it reaches the pager, which answers `uncategorized error: 0`, and the
//!    request retries without bound — the process hangs. So the "wrong id"
//!    control AC1 asked for does not exist here; P1 and Q2 are behind
//!    `--bogus-id` and off by default, with the evidence recorded at P1.
//!
//! 4. The masking half of the model cannot be reached at all. The kernel
//!    applies per-object `permmask`/`ovrmask` (`security.rs:169`) but nothing
//!    in `twizzler-security` can populate `SecCtxBase.masks` — there is no
//!    `insert_mask`. Step P7 asserts that against a live context, and explains
//!    why it is what blocks confirming the report's §6.3 defect from userspace.
//!
//! So the capability mechanism is exercised on hand-built objects (which can be
//! created with the right spec from outside the engine), and the engine's
//! objects are probed to show they cannot participate. That split is the
//! result, not a workaround for one.
//!
//! Layout: `capdemo` is compartment P, the driver. `capdemo child <step> ...`
//! is compartment Q, spawned once per step — steps Q3 and Q5 are expected to
//! kill Q, so they cannot share a process with anything that follows.
//!
//! Child exit codes are a protocol, and the unexpected ones are the
//! interesting ones:
//!
//! | code | meaning |
//! |------|---------|
//! | 0    | the step's own assertions passed |
//! | 2    | an assertion in the child failed |
//! | 3    | the access that was supposed to be denied **succeeded** |
//! | 4    | `map` returned an error where the model says it returns a slot |
//! | 101  | killed by the monitor after an upcall it declined (the denial) |
//!
//! A 3 or a 4 does not mean the demo is broken. It means `G1-NOTES.md` §1 is
//! wrong, which is worth more than a green run.

use std::process::Command;

use twizzler::object::{Object, ObjectBuilder, RawObject, TypedObject as _};
use twizzler_abi::{
    object::{ObjID, Protections},
    syscall::{sys_thread_active_sctx_id, ObjectCreate},
};
use twizzler_graph::{Graph, GraphError, Labels, PropValue, VertexId};
use twizzler_rt_abi::{
    error::{ObjectError, TwzError},
    object::MapFlags,
};
use twizzler_security::{
    Cap, SecCtx, SecCtxFlags, SecureBuilderExt as _, SigningKey, SigningScheme,
};

/// Harness revision, stamped into the first line of a run so a transcript can be
/// tied to the tree that produced it. Mirrors `gstress`'s `HARNESS_REV`
/// (`gstress/src/main.rs:210`) — same convention, same reason: without it a
/// recovered console capture cannot be attributed to a build.
///
/// `…a` (2026-08-19): first stamped revision. Runs before this one — including
/// the 2026-08-14 run whose graph model is reported in the write-up — emitted no
/// stamp, and are attributable only by their date.
const HARNESS_REV: &str = "2026-08-19a";

/// Child exit codes. See the table above.
const EXIT_ASSERT: i32 = 2;
const EXIT_NOT_DENIED: i32 = 3;
const EXIT_MAP_ERRED: i32 = 4;
/// What the monitor exits a thread with when it declines an upcall
/// (`rt/monitor/src/upcall.rs:24`). This is the observable form of a security
/// violation — see the note on `Q3` about how weak a signal it is.
const EXIT_KILLED: i32 = 101;

/// An object id nothing will have allocated. Intended as the "wrong id"
/// control, on the expectation that it would fail as a typed error at map time
/// and so be distinguishable from a denial.
///
/// **It does not.** Mapping it hangs in the pager — see the note at P1. Only
/// reachable behind `--bogus-id`.
const BOGUS_ID: u128 = 0xDEAD_BEEF_0000_0000_0000_0000_CAFE_F00D;

const PUBLIC_MAGIC: u64 = 0x5075_626C_6963_0001; // "Public"
const PRIVATE_MAGIC: u64 = 0x5072_6976_6174_0002; // "Privat"

/// The graph P opens for the §2 probe. Named distinctly so it cannot collide
/// with anything `gstress` leaves behind.
const PROBE_GRAPH: &str = "capdemo-probe";

// ---------------------------------------------------------------------------
// check harness
// ---------------------------------------------------------------------------

/// Records rather than panics, so one wrong assumption does not hide the rest
/// of the run. A hardware run of this binary is expensive to schedule; getting
/// all of the answers from one boot matters more than failing fast.
#[derive(Default)]
struct Checks {
    pass: usize,
    fail: usize,
}

impl Checks {
    fn check(&mut self, name: &str, ok: bool, detail: impl std::fmt::Display) {
        if ok {
            self.pass += 1;
            println!("  ok   {name}");
        } else {
            self.fail += 1;
            println!("  FAIL {name}: {detail}");
        }
    }

    fn eq<T: PartialEq + std::fmt::Debug>(&mut self, name: &str, got: T, want: T) {
        let ok = got == want;
        self.check(name, ok, format!("got {got:?}, want {want:?}"));
    }

    fn finish(self) -> ! {
        println!(
            "\ncapdemo: harness={} {} passed, {} failed",
            HARNESS_REV, self.pass, self.fail
        );
        std::process::exit(if self.fail == 0 { 0 } else { 1 });
    }
}

fn parse_id(s: &str) -> ObjID {
    ObjID::new(u128::from_str_radix(s, 16).expect("object id argument must be hex"))
}

// ---------------------------------------------------------------------------
// parent (compartment P)
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "child" {
        child(&args[2..])
    } else {
        parent(
            args.iter().any(|a| a == "--bogus-id"),
            args.iter().any(|a| a == "--graph"),
        )
    }
}

fn parent(bogus: bool, graph: bool) -> ! {
    // First line of the run, before anything can fail. Names the revision and the
    // arms that are actually enabled, so a transcript records what was run rather
    // than what the source happens to say today.
    println!(
        "CAPDEMO STAMP harness={} mode=parent graph={} bogus-id={}",
        HARNESS_REV, graph, bogus
    );

    let mut c = Checks::default();

    // --- P1: the "wrong object id" control (opt-in: `--bogus-id`) ---------
    //
    // AC1 warns against a test that would pass for the wrong reason — one that
    // accepts any failure, including a mistyped object id. That failure was
    // expected to have a *different shape*: an unallocated id failing inside
    // `sys_object_map`'s `lookup_object`, as a typed error, in-process, where a
    // denial (Q3) is fatal and deferred to first touch.
    //
    // **MEASURED 2026-08-14: it does not, and this step hangs the run.** An
    // unallocated id is not rejected by `lookup_object`. The kernel asks the
    // pager for it, the pager answers with `uncategorized error: 0` for
    // `ObjectInfoReq`, and the request is retried without bound — one log line
    // per attempt, forever. `map` never returns, so `NoSuchObject` is never
    // observed and the process cannot make progress.
    //
    // Two consequences, both findings rather than defects in this binary:
    //
    // 1. **The discrimination AC1 asked for is not available on this build.**
    //    There is no cheap "wrong id" failure to contrast a denial against,
    //    because the wrong-id path does not terminate. Q2 is disabled for the
    //    same reason.
    // 2. `G1-NOTES.md` §1.1 predicted a typed in-process error here. That
    //    prediction is refuted, and the notes were written from source without
    //    a run — which is exactly the class of claim this binary exists to
    //    settle.
    //
    // Left in place behind a flag rather than deleted: the hang is the
    // evidence, and re-deriving it later would cost another boot.
    if bogus {
        println!("P1: map an unallocated object id (WILL HANG — see the note in the source)");
        match unsafe { Object::<u64>::map_unchecked(ObjID::new(BOGUS_ID), MapFlags::READ) } {
            Err(e) => c.check(
                "P1 unallocated id yields ObjectError::NoSuchObject",
                matches!(e, TwzError::Object(ObjectError::NoSuchObject)),
                format!("got {e:?}"),
            ),
            Ok(_) => c.check(
                "P1 unallocated id yields ObjectError::NoSuchObject",
                false,
                "map succeeded on an id nothing allocated",
            ),
        }
    } else {
        println!("P1: skipped — unallocated ids hang the pager on this build (--bogus-id to run)");
    }

    // --- P2: what ObjectBuilder::default() actually creates ---------------
    //
    // `ObjectCreate::default()` is `def_prot = Protections::all(), kuid = None`
    // (rt-abi/src/object.rs:780). If this check fails, §2 of the notes is wrong
    // and G1 may be achievable after all.
    println!("P2: default-built object metadata");
    let public = ObjectBuilder::<u64>::default()
        .build(PUBLIC_MAGIC)
        .expect("build default object");
    let meta = unsafe { *public.meta_ptr() };
    c.eq("P2 default def_prot is all()", meta.default_prot, Protections::all());
    c.eq("P2 default kuid is 0", meta.kuid.raw(), 0u128);

    // --- P3: the same question, asked of a real graph ---------------------
    //
    // This is §2 of the notes as a test. The engine builds its root, arenas,
    // SegVec segments and blob segments through `ObjectBuilder::default()`
    // (graph.rs:670, arena_store.rs:772, segvec.rs:100/298/300,
    // blobstore.rs:156), so the root's metadata stands in for all of them.
    //
    // Limitation worth naming: only the root is checked, because
    // `owned_object_ids` is `#[cfg(test)] pub(crate)` and this binary must not
    // touch the engine to widen it. If the root is world-readable the arenas
    // are too — same constructor — but that last step is inference.
    println!("P3: graph root object metadata");
    let g = Graph::open_or_create(PROBE_GRAPH).expect("open probe graph");
    let root_id = g.root_id();
    // Mapped with the flags the engine itself uses for the root (graph.rs:556).
    // Metadata is written by the kernel at creation so the flags should not
    // matter, but this project has been bitten by assuming that about a
    // *different* mapping of the same object before.
    let root = unsafe { Object::<u64>::map_unchecked(root_id, MapFlags::READ | MapFlags::PERSIST) }
        .expect("map graph root");
    let rmeta = unsafe { *root.meta_ptr() };
    c.eq("P3 graph root def_prot is all()", rmeta.default_prot, Protections::all());
    c.eq("P3 graph root kuid is 0", rmeta.kuid.raw(), 0u128);
    drop(g);

    // --- P4: what a freshly loaded compartment starts with ----------------
    //
    // The monitor mints one SecCtx per compartment with global mask all() and
    // an empty cap map (dlengine.rs:82). Combined with P2/P3 that means a new
    // compartment can already reach every graph on the machine. Must run
    // before P5, which inserts a capability into this very context.
    println!("P4: this compartment's security context");
    // **The single most load-bearing check in this binary, as of the Q3/Q5
    // result on 2026-08-14.** Both enforcement points short-circuit to "allow"
    // when the thread's active security context id is 0:
    //
    //   * `check_security` (fault.rs:118) returns `Protections::all()`
    //   * `check_settings` (region.rs:80) returns `Ok(())`
    //
    // So an active sctx of 0 means *nothing is enforced anywhere*, and every
    // denial this binary expects would silently not happen. Q3 and Q5 observed
    // exactly that. If this check passes in P and the child prints 0, the
    // difference is where the explanation lives.
    let own_sctx = sys_thread_active_sctx_id();
    c.check(
        "P4 active sctx id is not the kernel context (0)",
        own_sctx.raw() != 0,
        format!("active sctx = {:x} — all enforcement is bypassed", own_sctx.raw()),
    );
    let ctx = SecCtx::active_ctx();
    let base = ctx.base();
    c.eq("P4 global mask is all()", base.global_mask, Protections::all());
    c.check(
        "P4 capability map starts empty",
        base.map.is_empty(),
        format!("{} entries", base.map.len()),
    );

    // --- P5: build something that *can* be protected ----------------------
    //
    // Everything above says the engine's objects cannot be. These are built
    // outside it, with `def_prot = empty` and a real kuid, which is the only
    // shape a capability can ever apply to (notes §1.3).
    //
    // `build_secure` resolves the chicken-and-egg — with def_prot empty the
    // creator cannot write its own object either, so it mints a cap into the
    // active context before writing the base (builder_ext.rs:31).
    println!("P5: build a capability-protected object");
    let (s_key, v_key) = SigningKey::new_keypair(&SigningScheme::Ecdsa, Default::default())
        .expect("keypair");
    let spec = ObjectCreate::new(
        Default::default(),
        Default::default(),
        Some(v_key.id()),
        Default::default(),
        Protections::empty(),
    );
    let private = ObjectBuilder::<u64>::new(spec)
        .build_secure(PRIVATE_MAGIC, s_key.base())
        .expect("build secure object");
    let pmeta = unsafe { *private.meta_ptr() };
    c.eq("P5 private def_prot is empty", pmeta.default_prot, Protections::empty());
    c.eq("P5 private kuid is the verifying key", pmeta.kuid.raw(), v_key.id().raw());
    // Positive control: the capability model works at all. If P cannot read its
    // own protected object then every denial below is uninformative, because a
    // denial would be the only outcome the mechanism can produce.
    let seen = unsafe { private.base_ptr::<u64>().read_volatile() };
    c.eq("P5 creator can read through its own capability", seen, PRIVATE_MAGIC);

    // --- P6: delegation ----------------------------------------------------
    //
    // `Del` is `unimplemented!()` in userspace (sec_ctx/user.rs:145), so this
    // is delegation by handing over a second *context* holding a read-only
    // capability — not delegation in the `Del` sense. The distinction matters
    // for the write-up; do not describe this as capability delegation proper.
    //
    // The cap is signed over its accessor (capability.rs:112), so it is bound
    // to this context and cannot be lifted into another one.
    println!("P6: mint a read-only capability in a second context");
    let mut deleg = SecCtx::new(
        ObjectCreate::default(),
        Protections::all(),
        SecCtxFlags::empty(),
    )
    .expect("delegation context");
    let cap = Cap::new(
        private.id(),
        deleg.id(),
        Protections::READ,
        s_key.base(),
        Default::default(),
        Default::default(),
        Default::default(),
    )
    .expect("mint read-only cap");
    deleg.insert_cap(cap).expect("insert cap");
    // Cheap, but it catches an `insert_cap` that returns Ok without landing an
    // entry — the map is bounded at SEC_CTX_MAP_LEN = 16 (base.rs:9) and
    // overflow is reported as `OutOfResources`, not silently.
    //
    // If this one fails on its own while Q4 still passes, suspect the mapping
    // coherence trap rather than `insert_cap`: it writes through a transaction
    // on a *clone* of the handle, and this reads back through the original.
    // That is the same shape as the bug that cost 2 858 vertices
    // (PROJECT_PLAN.md, 2026-08-04). The authoritative evidence that the cap
    // landed is Q4, which asks the kernel rather than the mapping.
    c.eq("P6 capability landed in the delegation context", deleg.base().map.len(), 1);

    // --- P7: the masking half of the model is unreachable ------------------
    //
    // The security paper's effective-permission formula is
    //
    //     P = (caps ∪ default_prot) ∩ permmask ∩ (global_mask ∪ ovrmask)
    //
    // and the kernel implements the mask terms: `security.rs:169` reads
    // `base.masks.get(&target)` and applies `permmask` and `ovrmask` to what
    // the capabilities and default permissions gave.
    //
    // **Nothing can put anything in that map.** `SecCtxBase.masks` is
    // initialised empty (`sec_ctx/base.rs:94`) and `SecCtx` exposes
    // `insert_cap`, `insert_del` (`unimplemented!()`), `remove_cap` and
    // `remove_del` (both `unimplemented!()`) — there is no `insert_mask`
    // anywhere in `twizzler-security`. So the kernel's mask branch is
    // unreachable from userspace and the per-object half of the formula is
    // inert. These checks are that claim asked of a live context rather than
    // of the source.
    //
    // **This is also why the global-mask defect in the report's §6.3 cannot be
    // demonstrated from here**, which is worth stating at the site rather than
    // only in the write-up. Isolating a restrictive context *is* possible —
    // `UNDETACHABLE` stops `search_access` falling back to this compartment's
    // own `all()` context (`security.rs:244`). But a *global* mask applies to
    // every object, so restricting any right also strips the child's own stack,
    // heap and text, and the compartment dies on its next stack write rather
    // than on the object under test: exit 101 either way, proving nothing. The
    // per-object mask is exactly the mechanism that would scope a restriction
    // to one object, and it cannot be set. Confirming that defect needs a
    // kernel-side test of `SecurityContext::lookup`, not a compartment.
    println!("P7: the mask half of the formula is unreachable");
    let own = SecCtx::active_ctx();
    c.check(
        "P7 this compartment's context has no per-object masks",
        own.base().masks.is_empty(),
        format!("{} entries", own.base().masks.len()),
    );
    // A context asked for an explicitly restrictive global mask: the global
    // mask lands as given, the per-object map stays empty. That asymmetry is
    // the finding — one half of the formula is settable and the other is not.
    let restricted = SecCtx::new(
        ObjectCreate::default(),
        Protections::READ,
        SecCtxFlags::empty(),
    )
    .expect("restricted context");
    c.eq(
        "P7 a restrictive global mask is stored as given",
        restricted.base().global_mask,
        Protections::READ,
    );
    c.check(
        "P7 a freshly created context has no per-object masks",
        restricted.base().masks.is_empty(),
        format!("{} entries", restricted.base().masks.len()),
    );
    // `insert_cap` is the only mutation the API offers. P6 used it on `deleg`;
    // the mask map is still empty, so the one available write path does not
    // reach it even incidentally.
    c.check(
        "P7 inserting a capability leaves the mask map empty",
        deleg.base().masks.is_empty(),
        format!("{} entries", deleg.base().masks.len()),
    );

    let pub_id = format!("{:x}", public.id().raw());
    let priv_id = format!("{:x}", private.id().raw());
    let deleg_id = format!("{:x}", deleg.id().raw());

    // --- Q1: scoped access, the baseline ----------------------------------
    println!("Q1: child reads a world-readable object");
    expect_exit(&mut c, "Q1 child reads the public object", 0, &["public-read", pub_id.as_str()]);

    // --- Q2: the discrimination AC1 wanted (opt-in: `--bogus-id`) ---------
    //
    // Disabled by default for the reason recorded at P1: an unallocated id
    // hangs in the pager rather than returning `NoSuchObject`, so this child
    // would never exit and the run would stop here. The contrast Q2/Q3 was
    // built to draw is therefore unavailable on this build.
    if bogus {
        println!("Q2: child maps an unallocated id (WILL HANG — see P1)");
        expect_exit(&mut c, "Q2 child sees NoSuchObject, not a denial", 0, &["missing"]);
    } else {
        println!("Q2: skipped — see P1");
    }

    // --- Q3: the denial ----------------------------------------------------
    //
    // Expected: the map succeeds, the read faults, the monitor declines the
    // upcall, the compartment dies with 101.
    //
    // 101 is a weak signal and this is the honest limit of AC1: a null
    // dereference and an out-of-bounds object access produce the same code.
    // What makes the pair Q2/Q3 mean something is the *contrast* — same child,
    // same binary, one failure typed and in-process, the other fatal and
    // deferred to first touch. Neither alone is evidence.
    println!("Q3: child reads a protected object with no capability");
    expect_exit(
        &mut c,
        "Q3 child is killed on first touch of the protected object",
        EXIT_KILLED,
        &["private-read", priv_id.as_str()],
    );

    // --- Q4: delegation lets it through -----------------------------------
    println!("Q4: child attaches the delegated context and reads");
    expect_exit(
        &mut c,
        "Q4 child reads the protected object under the delegated capability",
        0,
        &["deleg-read", priv_id.as_str(), deleg_id.as_str()],
    );

    // --- Q5: read-only means read-only ------------------------------------
    println!("Q5: child writes under a read-only capability");
    expect_exit(
        &mut c,
        "Q5 child is killed writing under a read-only capability",
        EXIT_KILLED,
        &["deleg-write", priv_id.as_str(), deleg_id.as_str()],
    );

    if graph {
        capability_graph(
            &mut c,
            own_sctx,
            &deleg,
            public.id(),
            private.id(),
            v_key.id(), // the VERIFYING key — see the note at Q(c)
            meta.default_prot,
            pmeta.default_prot,
        );
    } else {
        println!("P8: skipped — pass --graph to model this capability network as a graph");
    }

    c.finish()
}

/// Spawn Q for one step and assert its exit code.
///
/// `std::process::Command` is the right tool here despite appearances: on this
/// runtime it goes through `CompartmentLoader` (reference/src/runtime/exec.rs:70),
/// so the child really is a separate compartment with its own security context
/// (dlengine.rs:107), which is the whole premise of the test.
fn expect_exit(c: &mut Checks, name: &str, want: i32, args: &[&str]) {
    let exe = std::env::args().next().unwrap_or_else(|| "capdemo".to_string());
    let status = Command::new(&exe).arg("child").args(args).status();
    match status {
        Ok(st) => match st.code() {
            Some(code) => c.check(name, code == want, explain(code, want)),
            None => c.check(name, false, "child produced no exit code"),
        },
        Err(e) => c.check(name, false, format!("could not spawn child: {e}")),
    }
}

/// Turn the child's exit code into something that says what it means, so a
/// failing run does not need this file open beside it to be read.
fn explain(got: i32, want: i32) -> String {
    let meaning = match got {
        0 => "child's own assertions passed",
        EXIT_ASSERT => "an assertion inside the child failed",
        EXIT_NOT_DENIED => "the access that should have been denied SUCCEEDED — G1-NOTES §1 is wrong",
        EXIT_MAP_ERRED => "map returned an error where the model says it returns a slot — G1-NOTES §1.1 is wrong",
        EXIT_KILLED => "killed by the monitor after a declined upcall (the denial)",
        1 => "child exited 1 (a panic, or its own check harness failing)",
        _ => "unrecognised",
    };
    format!("got {got} ({meaning}), want {want}")
}

// ---------------------------------------------------------------------------
// child (compartment Q)
// ---------------------------------------------------------------------------

fn child(args: &[String]) -> ! {
    let step = args.first().map(String::as_str).unwrap_or("");
    // See the note at P4. Q3 and Q5 succeeded where they should have been
    // denied; if this prints 0, the enforcement points were short-circuited and
    // no capability was ever consulted. Printed rather than asserted because
    // the steps that matter are expected to die, and a dead child reports
    // nothing.
    println!(
        "  [child {step}] active sctx = {:x}",
        sys_thread_active_sctx_id().raw()
    );
    match step {
        // Denial is scoped, not global: Q reads a world-readable object in the
        // same run in which it is denied the protected one.
        "public-read" => {
            let id = parse_id(&args[1]);
            let obj = match unsafe { Object::<u64>::map_unchecked(id, MapFlags::READ) } {
                Ok(o) => o,
                Err(_) => std::process::exit(EXIT_MAP_ERRED),
            };
            let seen = unsafe { obj.base_ptr::<u64>().read_volatile() };
            std::process::exit(if seen == PUBLIC_MAGIC { 0 } else { EXIT_ASSERT });
        }

        // The control: a wrong id is a typed error, not a death.
        "missing" => {
            match unsafe { Object::<u64>::map_unchecked(ObjID::new(BOGUS_ID), MapFlags::READ) } {
                Err(TwzError::Object(ObjectError::NoSuchObject)) => std::process::exit(0),
                _ => std::process::exit(EXIT_ASSERT),
            }
        }

        // The denial. Both halves are load-bearing: the map is asserted to
        // *succeed*, which is the claim in §1.1 that mapping performs no
        // permission check at all. Only the read should be fatal.
        "private-read" => {
            let id = parse_id(&args[1]);
            let obj = match unsafe { Object::<u64>::map_unchecked(id, MapFlags::READ) } {
                Ok(o) => o,
                // The model says map performs no permission check at all. If it
                // errs here, that is the interesting result, not a failure.
                Err(_) => std::process::exit(EXIT_MAP_ERRED),
            };
            let seen = unsafe { obj.base_ptr::<u64>().read_volatile() };
            // Unreachable if the model holds. If it is reached, the object was
            // not protected and the parent needs to know that specifically.
            std::hint::black_box(seen);
            std::process::exit(EXIT_NOT_DENIED);
        }

        // Attach the delegated context and read.
        //
        // `attach` rather than `set_active`: the fault handler unions across
        // every attached context (`search_access`, security.rs:228), so
        // attaching is enough, and it leaves this compartment's own context
        // active rather than disturbing a running compartment mid-flight.
        //
        // Note that `sys_sctx_attach` authorises nothing (object.rs:250) — Q
        // needs only the id. That is the mechanism this step depends on and
        // also, separately, a finding.
        "deleg-read" => {
            let id = parse_id(&args[1]);
            let ctx_id = parse_id(&args[2]);
            let ctx = match SecCtx::try_from(ctx_id) {
                Ok(c) => c,
                Err(_) => std::process::exit(EXIT_ASSERT),
            };
            if ctx.attach().is_err() {
                std::process::exit(EXIT_ASSERT);
            }
            let obj = match unsafe { Object::<u64>::map_unchecked(id, MapFlags::READ) } {
                Ok(o) => o,
                Err(_) => std::process::exit(EXIT_MAP_ERRED),
            };
            let seen = unsafe { obj.base_ptr::<u64>().read_volatile() };
            std::process::exit(if seen == PRIVATE_MAGIC { 0 } else { EXIT_ASSERT });
        }

        // Read-only means read-only. The object is mapped READ | WRITE so that
        // the write reaches the security check rather than failing earlier on
        // the mapping's own protections — the denial under test is the
        // capability's `Protections::READ`, not the map flags.
        //
        // The read first is deliberate: it proves the capability is in force
        // before the write is attempted, so a death on the write cannot be
        // confused with the capability never having applied.
        "deleg-write" => {
            let id = parse_id(&args[1]);
            let ctx_id = parse_id(&args[2]);
            let ctx = match SecCtx::try_from(ctx_id) {
                Ok(c) => c,
                Err(_) => std::process::exit(EXIT_ASSERT),
            };
            if ctx.attach().is_err() {
                std::process::exit(EXIT_ASSERT);
            }
            let obj = match unsafe {
                Object::<u64>::map_unchecked(id, MapFlags::READ | MapFlags::WRITE)
            } {
                Ok(o) => o,
                Err(_) => std::process::exit(EXIT_MAP_ERRED),
            };
            let seen = unsafe { obj.base_ptr::<u64>().read_volatile() };
            if seen != PRIVATE_MAGIC {
                std::process::exit(EXIT_ASSERT);
            }
            unsafe { obj.base_mut_ptr::<u64>().write_volatile(!PRIVATE_MAGIC) };
            // Unreachable if the read-only mask holds.
            let after = unsafe { obj.base_ptr::<u64>().read_volatile() };
            std::hint::black_box(after);
            std::process::exit(EXIT_NOT_DENIED);
        }

        other => {
            eprintln!("capdemo child: unknown step {other:?}");
            std::process::exit(EXIT_ASSERT);
        }
    }
}

// ---------------------------------------------------------------------------
// P8 — the capability network as a graph (RQ3)
// ---------------------------------------------------------------------------

/// The graph P8 builds. Distinct from `PROBE_GRAPH` so the §2 probe and this
/// model cannot contaminate each other.
const CAP_GRAPH: &str = "capdemo-capgraph";

/// Emit one edge per granted right, rather than one edge carrying a bitfield.
///
/// This is the representation decision from `docs/capability-graph.md` §2(a),
/// and it is made for a reason the engine's layout supplies: `AdjRef` carries
/// the edge label, so a label-filtered walk decides membership *inside the
/// adjacency read*, while a `protections` property would cost a record-head and
/// a data-block read per candidate edge, mid-traversal. "Every context that can
/// write O" is then a structural query rather than an arithmetic one.
///
/// The full bitfield is kept as an edge property too, so a capability can be
/// reconstructed from the graph without consulting the source it came from.
fn grant_edges(
    g: &mut Graph,
    from: VertexId,
    to: VertexId,
    prots: Protections,
) -> Result<(), GraphError> {
    for (bit, label) in [
        (Protections::READ, "grants_r"),
        (Protections::WRITE, "grants_w"),
        (Protections::EXEC, "grants_x"),
    ] {
        if prots.contains(bit) {
            let e = g.add_edge(from, label, to)?;
            g.set_edge_prop(e, "protections", PropValue::U64(prots.bits() as u64))?;
        }
    }
    Ok(())
}

/// Model the capability network this run just created, then ask it the
/// questions §6.3 of the report says are worth asking.
///
/// **What this is.** The vertices and edges below are the *real* security
/// state of this process: the ids are the ones the kernel minted, the
/// protections are the ones on the signed capability P6 inserted, and the
/// default permissions are the ones read back from object metadata in P2/P5.
/// Nothing here is synthetic.
///
/// **What this is not.** It is one process's view, not the system's. Modelling
/// every context on the machine needs system-wide enumeration of security
/// contexts, which this project has never verified is possible — see G2-AC1 in
/// `docs/tasks.md`. That limitation is the honest boundary of the claim and is
/// stated in the report rather than papered over.
fn capability_graph(
    c: &mut Checks,
    own_ctx: ObjID,
    deleg: &SecCtx,
    public: ObjID,
    private: ObjID,
    key: ObjID,
    public_def_prot: Protections,
    private_def_prot: Protections,
) {
    println!("P8: model this capability network as a graph");
    match build_and_query(c, own_ctx, deleg, public, private, key, public_def_prot, private_def_prot) {
        Ok(()) => {}
        Err(e) => c.check("P8 graph model built and queried", false, format!("{e}")),
    }
}

fn build_and_query(
    c: &mut Checks,
    own_ctx: ObjID,
    deleg: &SecCtx,
    public: ObjID,
    private: ObjID,
    key: ObjID,
    public_def_prot: Protections,
    private_def_prot: Protections,
) -> Result<(), GraphError> {
    // Idempotent across runs, like every other graph this project builds.
    Graph::reset(CAP_GRAPH)?;
    let mut g = Graph::open_or_create(CAP_GRAPH)?;

    // --- vertices: contexts, objects, keys are all just objects -----------
    //
    // They share one label space *because they share one id space*. That is
    // the property the report leans on: on a data-centric OS every endpoint of
    // a security relation is an object with a 128-bit id, so the node space is
    // homogeneous without being made so.
    let v_own = g.add_vertex("context", "ctx:self", own_ctx)?;
    g.set_vertex_prop(v_own, "global_mask", PropValue::U64(Protections::all().bits() as u64))?;

    let v_deleg = g.add_vertex("context", "ctx:deleg", deleg.id())?;
    g.set_vertex_prop(
        v_deleg,
        "global_mask",
        PropValue::U64(deleg.base().global_mask.bits() as u64),
    )?;

    let v_public = g.add_vertex("object", "obj:public", public)?;
    g.set_vertex_prop(v_public, "default_prot", PropValue::U64(public_def_prot.bits() as u64))?;

    let v_private = g.add_vertex("object", "obj:private", private)?;
    g.set_vertex_prop(v_private, "default_prot", PropValue::U64(private_def_prot.bits() as u64))?;

    // Named for which key this is. The object's `kuid` records the *verifying*
    // key, not the signing key — see Q(c) for why the distinction decides what
    // the two-hop query below can and cannot answer.
    let v_key = g.add_vertex("key", "key:private-verifying", key)?;

    // --- edges -------------------------------------------------------------
    //
    // One `grants_*` edge per right on the capability P6 minted, and a
    // `keyed_by` edge to the key the object's `kuid` names. That key is the
    // *verifying* key — the one the kernel loads to check a signature
    // (`security.rs:123-134`) — not the signing key that mints. Q(c) below
    // depends on that distinction.
    grant_edges(&mut g, v_deleg, v_private, Protections::READ)?;
    g.add_edge(v_private, "keyed_by", v_key)?;

    c.eq("P8 graph holds 5 vertices", g.traversal().vertices().count(), 5);

    // --- Q(a): who can access the private object? --------------------------
    //
    // **This is the query the kernel cannot answer.** A security context
    // indexes its entries by target object id, so "what may this context do to
    // O" is fast; there is no inverse index anywhere in the system, so "who can
    // reach O" requires enumerating every context and scanning each. Under
    // index-free adjacency it is one label-filtered adjacency read.
    let readers = g
        .traversal()
        .v(v_private)
        .in_(Labels::these(&["grants_r"]))
        .to_ids();
    c.eq("P8 exactly one context can read the private object", readers.len(), 1);
    c.check(
        "P8 that context is the delegation context",
        readers.first() == Some(&v_deleg),
        "in-neighbour over grants_r was not ctx:deleg",
    );

    // Nobody can write it: the capability granted READ only, so no `grants_w`
    // edge was ever emitted. A bitfield property would need arithmetic here;
    // an absent label needs none.
    c.eq(
        "P8 no context can write the private object",
        g.traversal().v(v_private).in_(Labels::these(&["grants_w"])).count(),
        0,
    );

    // --- Q(b): the path, with the edge that authorised it ------------------
    //
    // B6 exists because of this query. A vertex-only path says ctx:deleg
    // reached obj:private; it cannot say *which* capability let it, and a
    // context may hold several for one target. The edge element is the answer.
    let paths = g.traversal().v(v_deleg).out(Labels::any()).path();
    c.eq("P8 one authorised path out of ctx:deleg", paths.len(), 1);
    let named_edge = paths
        .first()
        .and_then(|p| p.get(1).copied())
        .and_then(|e| e.as_edge())
        .and_then(|e| g.edge_info(e))
        .map(|i| i.label);
    c.eq(
        "P8 the path names the capability that authorised it",
        named_edge.as_deref(),
        Some("grants_r"),
    );

    // --- Q(c): rights over the object's key --------------------------------
    //
    // §5.3 of the security paper argues that "who may grant access to O" is
    // expressible, because issuing a capability for O requires O's signing key
    // and read access to a key object is ordinary access control. Two hops make
    // that shape *queryable* — but they cannot answer it on this build, and the
    // reason is worth more than the query.
    //
    // **`kuid` names the verifying key, not the signing key.** P5 asserts
    // exactly that (`pmeta.kuid == v_key.id()`), and the kernel loads it as a
    // `VerifyingKey` to check signatures. Read access to a verifying key confers
    // nothing — it is public. The key that can *mint* for obj:private is
    // `s_key`, and an object's recorded state does not name it anywhere, so it
    // is not reachable from the object by any traversal.
    //
    // So an empty answer here means "no capability has been granted over the
    // recorded verifying key", NOT "nobody can mint for obj:private". Do not
    // write it up as the latter. Modelling the issuing half needs a relation the
    // platform does not record.
    //
    // Adding `s_key.id()` as a second key vertex would model minting directly,
    // at the cost of changing the graph this run reports (five vertices, two
    // edge kinds). Left alone deliberately; recorded here as the next step.
    let granters = g
        .traversal()
        .v(v_private)
        .out(Labels::these(&["keyed_by"]))
        .in_(Labels::these(&["grants_r"]))
        .to_ids();
    c.eq(
        "P8 no context holds a capability over the private object's verifying key",
        granters.len(),
        0,
    );

    // --- Q(d): ambient authority -------------------------------------------
    //
    // Objects whose *default* permissions grant at least as much as any
    // capability does — where the capability system is decorative because the
    // defaults already permit everything.
    //
    // This reproduces P2/P3 as a query rather than as a hand-written
    // assertion, which is the point of the exercise: the finding that every
    // engine object is world-readable falls out of asking the graph a general
    // question, instead of being something a human had to think to check.
    let all_bits = Protections::all().bits() as u64;
    // `vertices().has_label(..)` rather than `with_label(..)`: A8 made the
    // index a schema decision, and this graph declares none, so the scan is the
    // dependable route in a demo whose point is not lookup performance.
    let wide: Vec<_> = g
        .traversal()
        .vertices()
        .has_label("object")
        .has("default_prot", PropValue::U64(all_bits))
        .to_ids();
    // The second half — "and no capability constrains it" — is a filter on a
    // sub-traversal, which the DSL cannot express: that is B5 (`where_`), still
    // unbuilt, and this is a second independent caller for it. Done in Rust
    // here, and recorded rather than worked around silently.
    let ambient: Vec<_> = wide
        .into_iter()
        .filter(|v| g.traversal().v(*v).in_(Labels::any()).count() == 0)
        .collect();
    c.eq("P8 exactly one object relies on ambient authority", ambient.len(), 1);
    c.check(
        "P8 the ambient-authority object is obj:public",
        ambient.first() == Some(&v_public),
        "expected the world-readable object",
    );

    println!("  (graph registered at data/{CAP_GRAPH}; survives reboot by name)");
    Ok(())
}
