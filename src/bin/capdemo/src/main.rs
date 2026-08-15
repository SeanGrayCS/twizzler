//! capdemo — RQ3 / G1: what Twizzler's capability model actually enforces, and
//! whether a graph can hide behind it.
//!
//! **Read `docs/handoffs/G1-NOTES.md` first.** G1's acceptance criteria as
//! written are not achievable, for two reasons this binary is built to confirm
//! or refute on real hardware rather than argue on paper:
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
    syscall::ObjectCreate,
};
use twizzler_graph::Graph;
use twizzler_rt_abi::{
    error::{ObjectError, TwzError},
    object::MapFlags,
};
use twizzler_security::{
    Cap, SecCtx, SecCtxFlags, SecureBuilderExt as _, SigningKey, SigningScheme,
};

/// Child exit codes. See the table above.
const EXIT_ASSERT: i32 = 2;
const EXIT_NOT_DENIED: i32 = 3;
const EXIT_MAP_ERRED: i32 = 4;
/// What the monitor exits a thread with when it declines an upcall
/// (`rt/monitor/src/upcall.rs:24`). This is the observable form of a security
/// violation — see the note on `Q3` about how weak a signal it is.
const EXIT_KILLED: i32 = 101;

/// An object id nothing will have allocated. Used as the "wrong id" control:
/// this failure mode is a typed error at map time, which is what makes it
/// distinguishable from a denial.
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
        println!("\ncapdemo: {} passed, {} failed", self.pass, self.fail);
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
        parent()
    }
}

fn parent() -> ! {
    let mut c = Checks::default();

    // --- P1: the "wrong object id" control -------------------------------
    //
    // AC1 warns against a test that would pass for the wrong reason — one that
    // accepts any failure, including a mistyped object id. That failure has a
    // *different shape*, and this pins it down: an unallocated id fails inside
    // `sys_object_map`'s `lookup_object`, as a typed error, in-process. A
    // denial (Q3) does not.
    println!("P1: map an unallocated object id");
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

    let pub_id = format!("{:x}", public.id().raw());
    let priv_id = format!("{:x}", private.id().raw());
    let deleg_id = format!("{:x}", deleg.id().raw());

    // --- Q1: scoped access, the baseline ----------------------------------
    println!("Q1: child reads a world-readable object");
    expect_exit(&mut c, "Q1 child reads the public object", 0, &["public-read", pub_id.as_str()]);

    // --- Q2: the discrimination AC1 wanted --------------------------------
    println!("Q2: child maps an unallocated id");
    expect_exit(&mut c, "Q2 child sees NoSuchObject, not a denial", 0, &["missing"]);

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
