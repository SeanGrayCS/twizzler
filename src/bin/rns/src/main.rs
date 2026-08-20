//! A relational namespace over Twizzler's own object structure.
//!
//! # What this is
//!
//! Twizzler stores several relations between objects, each in one direction
//! only:
//!
//! | relation | where it lives | forward | reverse |
//! |---|---|---|---|
//! | name → object | `naming` | `get` | — |
//! | object → object | the Foreign Object Table | read the FOT | — |
//! | namespace → member | `naming` | `enumerate_names` | — |
//!
//! An object's FOT names everything it points into. Nothing names the objects
//! that point at it. The namer maps a name to an id; nothing maps an id back
//! to its names. So the questions a system actually needs — what still
//! references this? is this safe to delete? what is this called? — are the
//! ones the platform cannot answer without scanning everything.
//!
//! `rns` builds the reverse index. It walks `data/`, maps each object, reads
//! its FOT, and records `references` and `contains` edges in a
//! `twizzler-graph` graph. After that, reverse questions are traversals.
//! Every vertex is a real object and every edge is a relation Twizzler
//! already maintains.
//!
//! Reading another object's metadata requires mapping it. That works here
//! because objects on this build are created world-readable; on a build that
//! enforced protections, the index could only cover objects the caller may
//! map.
//!
//! # Usage
//!
//!   rns make           create four real objects that reference each other
//!   rns corpus <n>     create n objects in a reference tree
//!   rns index          walk `data/`, read every FOT, build the graph
//!   rns bench          time every query both ways: graph vs the platform
//!   rns refs <name>    what does this object reference?      (forward)
//!   rns rrefs <name>   what references this object?          (reverse)
//!   rns names <name>   every name this object has            (reverse)
//!   rns safe           objects nothing references            (reclaim's question)
//!   rns unname <name>  try to unbind a `data/` name and report what happened
//!   rns reset          clear the graph
//!
//! The graph is registered at `data/rns`, so it re-opens by name after reboot.

use core::sync::atomic::Ordering;

use naming::{static_naming_factory, GetFlags, NsNodeKind};
use twizzler_rt_abi::object::FotFlags;
use twizzler::{
    marker::{BaseType, Invariant},
    object::{MapFlags, ObjID, Object, ObjectBuilder, RawObject, TypedObject},
    ptr::InvPtr,
};
use twizzler_graph::{Graph, GraphError, Labels, NameKey, PropValue, VertexId};

type Result<T> = core::result::Result<T, GraphError>;

const GRAPH: &str = "rns";
const ROOT: &str = "data";

/// Edge labels. Both name a relation Twizzler already stores.
const REFERENCES: &str = "references";
const CONTAINS: &str = "contains";

/// How far to scan an object's FOT before giving up.
///
/// A bound is needed because the FOT carries no recorded length (see
/// `fot_targets`) — the table ends at the first unallocated slot. This is
/// generous for the objects `rns` indexes; hit it and the count is reported
/// rather than silently truncated.
const MAX_FOT_SCAN: usize = 4096;

/// Probe: can the naming service unbind a persistent `data/` entry?
///
/// Reads the name, removes it, then reads it again, and prints which case
/// held: `remove` returns an error (unbinding is unsupported), the name no
/// longer resolves (unbinding works), or `remove` succeeds while the name
/// still resolves (a silent no-op callers cannot detect).
fn unname(name: &str) -> Result<()> {
    let mut namer = static_naming_factory().expect("naming service available");
    let path = format!("{ROOT}/{name}");

    match namer.get(&path, GetFlags::empty()) {
        Ok(n) => println!("rns: before — {path} resolves to {:x}", n.id.raw()),
        Err(e) => {
            println!("rns: {path} does not resolve ({e:?}); nothing to unbind");
            return Ok(());
        }
    }

    match namer.remove(&path) {
        Ok(()) => println!("rns: remove({path}) returned Ok"),
        Err(e) => {
            println!("rns: remove({path}) FAILED: {e:?}");
            println!("rns: VERDICT — unbinding is unsupported; the docs are right.");
            return Ok(());
        }
    }

    match namer.get(&path, GetFlags::empty()) {
        Ok(n) => println!(
            "rns: after — {path} STILL resolves to {:x}\n\
             rns: VERDICT — remove() is a silent no-op. Worse than an error: \
             callers cannot tell it failed.",
            n.id.raw()
        ),
        Err(e) => println!(
            "rns: after — {path} no longer resolves ({e:?})\n\
             rns: VERDICT — unbinding WORKS. `mvp.md`, the report's RQ4 section \
             and the A5 GraphRoot-leak justification in tasks.md are all wrong \
             and should be corrected; `Graph::destroy` could unbind instead of \
             marking MAGIC_DESTROYED."
        ),
    }
    Ok(())
}

/// A leaf file: data with no outbound references.
#[derive(Clone, Copy)]
#[repr(C)]
struct Blob {
    kind: u32,
    size: u64,
}
unsafe impl Invariant for Blob {}
impl BaseType for Blob {}

/// A file that uses another file. The `InvPtr` is what puts an entry in this
/// object's FOT, and therefore what the index can see. A raw `ObjID` field
/// would be just as functional and invisible to the index.
#[repr(C)]
struct Derived {
    kind: u32,
    size: u64,
    /// The file this one is built from.
    source: InvPtr<Blob>,
}
unsafe impl Invariant for Derived {}
impl BaseType for Derived {}

/// Create a small corpus of real objects with real references, and name them
/// under `data/`.
///
/// Shape, chosen so every interesting case appears:
///
/// ```text
///   sales.csv     (Blob)      referenced by two files
///   chart.png     (Derived) → sales.csv
///   report.txt    (Derived) → sales.csv
///   scratch.tmp   (Blob)      referenced by nothing
/// ```
///
/// After `rns index`: `sales.csv` has in-degree 2, `chart.png` and `report.txt`
/// have in-degree 0 but out-degree 1, `scratch.tmp` is isolated. So
/// `rns rrefs sales.csv` names the two files that would break if it were
/// deleted, and `rns safe` should list everything except `sales.csv`.
fn make() -> Result<()> {
    let mut namer = static_naming_factory().expect("naming service available");

    let sales = ObjectBuilder::<Blob>::default()
        .persist(true)
        .build(Blob { kind: 1, size: 8_192 })?;
    let scratch = ObjectBuilder::<Blob>::default()
        .persist(true)
        .build(Blob { kind: 1, size: 512 })?;

    let mut derived = Vec::new();
    for (name, size) in [("chart.png", 4_096u64), ("report.txt", 2_048)] {
        let obj = ObjectBuilder::<Derived>::default()
            .persist(true)
            .build_inplace(|tx| {
                let d = Derived {
                    kind: 2,
                    size,
                    // This is the FOT entry. Built against `tx` because a FOT
                    // belongs to the object that holds the pointer.
                    source: InvPtr::new(&tx, sales.base_ref())?,
                };
                tx.write(d)
            })?;
        derived.push((name, obj.id()));
    }

    let mut named = 0usize;
    for (name, id) in [("sales.csv", sales.id()), ("scratch.tmp", scratch.id())]
        .into_iter()
        .chain(derived.into_iter().map(|(n, i)| (n, i)))
    {
        let path = format!("{ROOT}/{name}");
        // `data/` supports create but not remove on this build, so a repeat
        // `make` hits an existing name. Report and continue rather than fail:
        // the objects are new, only the binding is stale.
        match namer.put(&path, id) {
            Ok(()) => {
                println!("rns: {name} -> {:x}", id.raw());
                named += 1;
            }
            Err(e) => println!("rns: could not name {name} ({e:?}); it already exists?"),
        }
    }
    println!(
        "rns: created 4 objects, named {named}. chart.png and report.txt each \
         hold an InvPtr to sales.csv, so each has one FOT entry."
    );
    println!("rns: now run `rns index`, then `rns rrefs sales.csv`.");
    Ok(())
}

/// One node of the `corpus` tree. `source` is a real `InvPtr`, so a real FOT
/// entry. Isolated nodes carry `InvPtr::null()`, which occupies no FOT slot,
/// keeping the type uniform.
#[repr(C)]
struct Node {
    kind: u32,
    size: u64,
    source: InvPtr<Node>,
}
unsafe impl Invariant for Node {}
impl BaseType for Node {}

/// Build a corpus: `n` objects named `data/f0 … f{n-1}`, wired as a binary
/// tree — node `i` references node `i/2` — except every 10th node, which
/// references nothing. So in-degrees vary, the safe-to-delete answer is
/// non-trivial, and ~90% of nodes are referencing objects.
///
/// Every object is persistent, so a large `n` can exhaust physical frames.
fn corpus(n: usize) -> Result<()> {
    let mut namer = static_naming_factory().expect("naming service available");
    let mut objs: Vec<Object<Node>> = Vec::with_capacity(n);
    let t = std::time::Instant::now();
    for i in 0..n {
        let isolated = i == 0 || i % 10 == 0;
        let parent = i / 2;
        let obj = ObjectBuilder::<Node>::default()
            .persist(true)
            .build_inplace(|tx| {
                let source = if isolated {
                    InvPtr::null()
                } else {
                    InvPtr::new(&tx, objs[parent].base_ref())?
                };
                tx.write(Node {
                    kind: if isolated { 0 } else { 1 },
                    size: 64,
                    source,
                })
            })?;
        let path = format!("{ROOT}/f{i}");
        if let Err(e) = namer.put(&path, obj.id()) {
            println!("rns: could not name f{i} ({e:?}) — fresh image needed?");
        }
        objs.push(obj);
        if (i + 1) % 100 == 0 {
            println!(
                "rns: corpus {}/{} ({:.0}/s)",
                i + 1,
                n,
                (i + 1) as f64 / t.elapsed().as_secs_f64()
            );
        }
    }
    println!(
        "rns: corpus of {n} objects in {:.1}s — ~{} referencing, {} isolated. \
         Now run `rns index`, then `rns bench`.",
        t.elapsed().as_secs_f64(),
        n - (n + 9) / 10,
        (n + 9) / 10
    );
    Ok(())
}

/// Every object an object's FOT points at.
///
/// `RawObject::fote_ptr` reads an entry; `FotEntry.values` is the target id in
/// `ObjID::parts` order, so `from_parts` inverts it. Resolver entries are
/// skipped: they name a resolver function rather than an object, so they are
/// not an edge.
fn fot_targets(id: ObjID) -> Vec<ObjID> {
    fot_targets_inner(id, true)
}

/// The same scan without the per-object report — for `bench`, whose platform
/// arm calls this in a loop over every object.
fn fot_targets_quiet(id: ObjID) -> Vec<ObjID> {
    fot_targets_inner(id, false)
}

fn fot_targets_inner(id: ObjID, verbose: bool) -> Vec<ObjID> {
    let Ok(obj) = Object::<()>::map(id, MapFlags::READ) else {
        // Unmappable is a finding, not an error — see the module docs on
        // protections. Report it rather than treating it as zero references.
        println!("rns: cannot map {id}, skipping (its references are unknown)");
        return Vec::new();
    };
    // The FOT has no recorded length; `MetaInfo.fotcount` cannot bound the
    // scan. The ABI declares it as the entry count, but this tree writes it
    // as 0 at every creation site and reads it nowhere; the runtime's
    // `insert_fot` allocates a slot by scanning `FotFlags`. So enumerate the
    // way the allocator does: from index 1 (index 0 is `InvPtr::new`'s
    // same-object case and never occupies a slot), stopping at the first slot
    // that was never allocated.
    let declared = unsafe { (*obj.meta_ptr()).fotcount } as usize;
    let mut out = Vec::new();
    let mut resolvers = 0usize;
    let mut nulls = 0usize;
    let mut deleted = 0usize;
    let mut scanned = 0usize;
    for i in 1..MAX_FOT_SCAN {
        let Some(p) = obj.fote_ptr(i) else { break };
        let e = unsafe { &*p };
        let flags = FotFlags::from_bits_truncate(e.flags.load(Ordering::SeqCst));
        if !flags.contains(FotFlags::ALLOCATED) {
            break; // first never-allocated slot ends the table
        }
        scanned += 1;
        if flags.contains(FotFlags::DELETED) {
            deleted += 1;
            continue;
        }
        // A resolver entry carries a function, not an id.
        if e.resolver != 0 {
            resolvers += 1;
            continue;
        }
        let target = ObjID::from_parts(e.values);
        if target.raw() == 0 || target == id {
            nulls += 1;
            continue;
        }
        out.push(target);
    }
    // Reported per object: a zero total is ambiguous between "this object
    // references nothing" and "the read is wrong". `declared` is printed
    // beside the scan so the reserved field's constant 0 shows up next to the
    // real count.
    if verbose {
        println!(
            "rns:   {id:x} scanned={scanned} refs={} resolver={resolvers} \
             deleted={deleted} null/self={nulls} (meta.fotcount={declared})",
            out.len()
        );
    }
    let _ = (scanned, declared, resolvers, deleted, nulls);
    out
}

/// Vertex for `id`, created on first sight. Named by hex id so the name is the
/// object's real identity; human names arrive as `contains` edges from their
/// namespace, which is what lets one object have several.
fn vertex_for(g: &mut Graph, id: ObjID) -> Result<VertexId> {
    let name = format!("{:x}", id.raw());
    if let Some(v) = g.find_vertex("object", &name).found() {
        return Ok(v);
    }
    g.add_vertex("object", &name, id)
}

fn index() -> Result<()> {
    let mut g = Graph::reset_arena(GRAPH, twizzler_graph::DEFAULT_ARENA_CAP)
        .and_then(|_| Graph::open_or_create_arena(GRAPH, twizzler_graph::DEFAULT_ARENA_CAP))?;
    // Resolving by object id is the hot path for the index build.
    g.set_label_indexed("object", true)?;

    let mut namer = static_naming_factory().expect("naming service available");
    let root = match namer.get(ROOT, GetFlags::empty()) {
        Ok(n) => n,
        Err(e) => {
            println!("rns: cannot open {ROOT}: {e:?}");
            return Ok(());
        }
    };
    let nodes = match namer.enumerate_names_nsid(root.id, 0, usize::MAX) {
        Ok(n) => n,
        Err(e) => {
            println!("rns: cannot enumerate {ROOT}: {e:?}");
            return Ok(());
        }
    };

    let ns = g.add_vertex("ns", ROOT, root.id)?;
    let mut objects = 0usize;
    let mut edges = 0usize;

    for node in nodes {
        // `NsNode::name` returns `Result<&str>` — a non-UTF-8 entry is skipped
        // rather than failing the index, since one bad name should not cost the
        // whole reverse index.
        let name: &str = match node.name() {
            Ok(n) => n,
            Err(_) => continue,
        };
        if name == "." || name == ".." || name == GRAPH {
            // Skip our own graph: indexing the index is legal but the
            // self-reference makes the `safe` output meaningless.
            continue;
        }
        if node.kind != NsNodeKind::Object {
            continue;
        }
        let v = vertex_for(&mut g, node.id)?;
        // The human name is a property of the binding, not the object — an
        // object under three names has three `contains` edges, each carrying
        // its own name. `PropValue::Str` is a `NameKey` — 31 bytes,
        // char-boundary truncated; namer entries are short, and the hex id on
        // the vertex is the authoritative identity.
        let e = g.add_edge(ns, CONTAINS, v)?;
        g.set_edge_prop(e, "name", PropValue::Str(NameKey::new(name)))?;
        // Display-convenience copy of the first name only.
        if g.get_vertex_prop(v, "name").is_none() {
            g.set_vertex_prop(v, "name", PropValue::Str(NameKey::new(name)))?;
        }
        objects += 1;

        for target in fot_targets(node.id) {
            let t = vertex_for(&mut g, target)?;
            g.add_edge(v, REFERENCES, t)?;
            edges += 1;
        }
    }

    g.sync()?;
    println!(
        "rns: indexed {objects} named objects, {edges} references, \
         {} vertices total",
        g.vertices().len()
    );
    if edges == 0 {
        println!(
            "rns: NO REFERENCES FOUND. Read the per-object lines before \
             concluding anything. `scanned=0` everywhere means no object here \
             holds an InvPtr — expected for `twizzler-graph` roots, which link \
             their arenas by raw u128 id (`arena_dir_raw`, `ArenaEntry.raw`) and \
             so leave no FOT entry at all. Run `rns make` first: it creates two \
             objects that genuinely do hold one."
        );
    } else {
        println!("rns: now try `rns rrefs sales.csv` — the direction the OS cannot go.");
    }
    Ok(())
}

/// How to print a vertex: its `data/` name when it has one, else its hex id.
///
/// Objects reached only through another object's FOT have no name — they are
/// real and referenced but never registered — so falling back to the id keeps
/// them visible instead of blank.
fn label_of(g: &Graph, v: VertexId) -> String {
    match g.get_vertex_prop(v, "name") {
        Some(PropValue::Str(s)) => s.as_str().to_string(),
        _ => g
            .vertex_info(v)
            .map(|i| format!("{} (unnamed)", i.name))
            .unwrap_or_else(|| "?".into()),
    }
}

/// Resolve a user-supplied name to a vertex: either a `data/` name we recorded
/// as a property, or a hex object id.
fn resolve(g: &Graph, name: &str) -> Option<VertexId> {
    if let Some(v) = g.find_vertex("object", name).found() {
        return Some(v);
    }
    g.vertices().into_iter().find(|v| {
        matches!(
            g.get_vertex_prop(*v, "name"),
            Some(PropValue::Str(s)) if s.as_str() == name
        )
    })
}

fn show(name: &str, out: bool) -> Result<()> {
    let g = Graph::open_or_create_arena(GRAPH, twizzler_graph::DEFAULT_ARENA_CAP)?;
    let Some(v) = resolve(&g, name) else {
        println!("rns: no object named {name}; run `rns index` first");
        return Ok(());
    };
    let ids = if out {
        g.out_neighbors(v, Labels::these(&[REFERENCES]))
    } else {
        g.in_neighbors(v, Labels::these(&[REFERENCES]))
    };
    let dir = if out { "references" } else { "is referenced by" };
    println!("{name} {dir} {} object(s):", ids.len());
    for id in ids {
        println!("  {}", label_of(&g, id));
    }
    if !out {
        println!(
            "(the platform has no index for this direction — answering it \
             without the graph means mapping every object and scanning its FOT)"
        );
    }
    Ok(())
}

fn names(name: &str) -> Result<()> {
    let g = Graph::open_or_create_arena(GRAPH, twizzler_graph::DEFAULT_ARENA_CAP)?;
    let Some(v) = resolve(&g, name) else {
        println!("rns: no object named {name}; run `rns index` first");
        return Ok(());
    };
    // One edge per binding, each carrying its own name — so this lists all
    // names, which the platform can only do by enumerating every namespace.
    let edges = g.traversal().v(v).in_e(Labels::these(&[CONTAINS])).to_ids();
    println!("{name} has {} name(s):", edges.len());
    for e in edges {
        if let Some(PropValue::Str(s)) = g.get_edge_prop(e, "name") {
            println!("  {}", s.as_str());
        }
    }
    Ok(())
}

/// Microseconds for `reps` runs of `f`, with the per-op figure.
fn time<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, f64, T) {
    let t = std::time::Instant::now();
    let mut last = f();
    for _ in 1..reps {
        last = f();
    }
    let us = t.elapsed().as_secs_f64() * 1e6;
    (us, us / reps as f64, last)
}

/// Every question timed both ways — through the graph, and through what the
/// platform offers (`naming` calls and raw FOT scans). Run after
/// `rns corpus <n>` and `rns index`, in the same boot.
///
/// Both arms run warm: `index` has already mapped every object once, so this
/// measures steady-state query cost, not first-touch. Each answer is checked
/// for equality across arms before its timing is trusted, because a fast
/// wrong answer is the failure mode that hides.
fn bench() -> Result<()> {
    let g = Graph::open_or_create_arena(GRAPH, twizzler_graph::DEFAULT_ARENA_CAP)?;
    let mut namer = static_naming_factory().expect("naming service available");
    let root = match namer.get(ROOT, GetFlags::empty()) {
        Ok(n) => n,
        Err(e) => {
            println!("rns: cannot open {ROOT}: {e:?}");
            return Ok(());
        }
    };
    let listing = namer
        .enumerate_names_nsid(root.id, 0, usize::MAX)
        .unwrap_or_default();
    let named: Vec<(String, ObjID)> = listing
        .iter()
        .filter(|n| n.kind == NsNodeKind::Object)
        .filter_map(|n| n.name().ok().map(|s| (s.to_string(), n.id)))
        .filter(|(s, _)| s != GRAPH)
        .collect();
    let n = named.len();
    if n < 4 {
        println!("rns: only {n} named objects — run `rns corpus <n>` and `rns index` first");
        return Ok(());
    }
    println!("RNS BENCH n={n} (both arms warm; index built this boot)");

    // Probe: f1 — a tree parent, so it has referrers; also present by name.
    let probe_name = "f1".to_string();
    let Some((_, probe_id)) = named.iter().find(|(s, _)| *s == probe_name) else {
        println!("rns: no f1 in {ROOT} — bench expects a `corpus` layout");
        return Ok(());
    };
    let probe_id = *probe_id;
    let probe_v = resolve(&g, &format!("{:x}", probe_id.raw())).expect("indexed");
    // The ns vertex has label "ns" (unindexed); find it by scan, once.
    let ns_v = g
        .vertices()
        .into_iter()
        .find(|v| g.vertex_info(*v).map(|i| i.label == "ns").unwrap_or(false))
        .expect("ns vertex — run `rns index` first");

    // 1. Name lookup. The graph arm is an O(N) property scan — `rns` has no
    //    human-name index — so the platform should win.
    let (_, plat, want) = time(200, || {
        namer.get(&format!("{ROOT}/{probe_name}"), GetFlags::empty()).ok().map(|x| x.id)
    });
    let (_, graph, got) = time(20, || resolve(&g, &probe_name));
    let ok = want == Some(probe_id) && got == Some(probe_v);
    println!("RNS BENCH lookup      platform={plat:.1}us graph={graph:.1}us agree={ok} (graph unindexed by design — see docs)");

    // 2. Namespace listing.
    let (_, plat, pl) = time(50, || {
        namer.enumerate_names_nsid(root.id, 0, usize::MAX).map(|v| v.len()).unwrap_or(0)
    });
    let (_, graph, gl) = time(50, || g.out_neighbors(ns_v, Labels::these(&[CONTAINS])).len());
    // Platform listing includes the graph's own entry; the index skips it.
    println!("RNS BENCH list        platform={plat:.1}us graph={graph:.1}us sizes={pl}/{gl}");

    // 3. Forward references of f1. The platform can do this — one map + scan.
    let (_, plat, pf) = time(200, || fot_targets_quiet(probe_id));
    let (_, graph, gf) = time(200, || g.out_neighbors(probe_v, Labels::these(&[REFERENCES])));
    let gf_ids: Vec<u128> = gf.iter().filter_map(|v| g.vertex_info(*v).map(|i| i.target.raw())).collect();
    let ok = pf.iter().map(|i| i.raw()).collect::<std::collections::BTreeSet<_>>()
        == gf_ids.iter().copied().collect();
    println!("RNS BENCH fwd-refs    platform={plat:.1}us graph={graph:.1}us agree={ok}");

    // 4. Reverse references of f1. The platform has no index: map every named
    //    object and scan its FOT for the probe id.
    let (tot, plat, pr) = time(3, || {
        let mut hits = Vec::new();
        for (_, id) in &named {
            if fot_targets_quiet(*id).contains(&probe_id) {
                hits.push(id.raw());
            }
        }
        hits
    });
    let (_, graph, gr) = time(200, || g.in_neighbors(probe_v, Labels::these(&[REFERENCES])));
    let gr_ids: std::collections::BTreeSet<u128> =
        gr.iter().filter_map(|v| g.vertex_info(*v).map(|i| i.target.raw())).collect();
    let ok = pr.iter().copied().collect::<std::collections::BTreeSet<_>>() == gr_ids;
    println!(
        "RNS BENCH rev-refs    platform={plat:.0}us graph={graph:.1}us agree={ok} \
         (platform scanned {n} objects; total {:.0}ms over 3 reps)",
        tot / 1e3
    );

    // 5. Safe-to-delete: nothing references it. Reclaim's discovery question.
    let (_, plat, ps) = time(3, || {
        let mut referenced = std::collections::BTreeSet::new();
        for (_, id) in &named {
            for t in fot_targets_quiet(*id) {
                referenced.insert(t.raw());
            }
        }
        named.iter().filter(|(_, id)| !referenced.contains(&id.raw())).count()
    });
    let (_, graph, gs) = time(20, || {
        g.vertices()
            .into_iter()
            .filter(|v| {
                g.vertex_info(*v).map(|i| i.label == "object").unwrap_or(false)
                    && g.in_neighbors(*v, Labels::these(&[REFERENCES])).is_empty()
            })
            .count()
    });
    println!("RNS BENCH safe        platform={plat:.0}us graph={graph:.0}us counts={ps}/{gs} (must agree)");
    println!(
        "RNS BENCH NOTE: reverse and safe are the claim — O(answer) against \
         O(everything). lookup and fwd-refs are the platform's home ground and \
         it should win them; report all five."
    );
    Ok(())
}

/// Objects nothing references — reclaim's discovery question.
///
/// The kernel needs this to decide what is collectable. Here it is an
/// in-degree test on `references`.
fn safe() -> Result<()> {
    let g = Graph::open_or_create_arena(GRAPH, twizzler_graph::DEFAULT_ARENA_CAP)?;
    // Only real objects: the `ns` vertex is the namespace itself, not
    // something that could be deleted.
    let objects: Vec<VertexId> = g
        .vertices()
        .into_iter()
        .filter(|v| {
            g.vertex_info(*v)
                .map(|i| i.label == "object")
                .unwrap_or(false)
        })
        .collect();
    let unreferenced: Vec<VertexId> = objects
        .iter()
        .copied()
        .filter(|v| g.in_neighbors(*v, Labels::these(&[REFERENCES])).is_empty())
        .collect();
    println!(
        "{} of {} objects have nothing referencing them:",
        unreferenced.len(),
        objects.len()
    );
    for v in unreferenced {
        println!("  {}", label_of(&g, v));
    }
    println!(
        "(in-degree zero on `references`. The platform keeps no reverse index, \
         so without this the same answer needs a scan of every object's FOT.)"
    );
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("index");
    let arg = args.get(2).map(|s| s.as_str());

    let r = match (cmd, arg) {
        ("make", _) => make(),
        ("corpus", Some(n)) => corpus(n.parse().unwrap_or(200)),
        ("bench", _) => bench(),
        ("unname", Some(n)) => unname(n),
        ("index", _) => index(),
        ("refs", Some(n)) => show(n, true),
        ("rrefs", Some(n)) => show(n, false),
        ("names", Some(n)) => names(n),
        ("safe", _) => safe(),
        ("reset", _) => {
            Graph::reset_arena(GRAPH, twizzler_graph::DEFAULT_ARENA_CAP).map(|_| ())
        }
        (c, _) => {
            println!(
                "usage: rns [make | corpus <n> | index | bench | refs <name> | \
                 rrefs <name> | names <name> | safe | unname <name> | reset]  (got {c})"
            );
            Ok(())
        }
    };
    if let Err(e) = r {
        println!("rns: {e:?}");
    }
}
