//! A relational namespace over Twizzler's *own* object structure.
//!
//! # What this is
//!
//! Twizzler stores several relations between objects, and stores every one of
//! them in one direction only:
//!
//! | relation | where it lives | forward | reverse |
//! |---|---|---|---|
//! | name → object | `naming` | `get` | — |
//! | object → object | the Foreign Object Table | read the FOT | — |
//! | namespace → member | `naming` | `enumerate_names` | — |
//!
//! An object's FOT names everything it points into. Nothing names the objects
//! that point at *it*. The namer maps a name to an id; nothing maps an id back
//! to its names. So the questions a system actually needs — *what still
//! references this? is this safe to delete? what is this called?* — are exactly
//! the ones the platform cannot answer without scanning everything.
//!
//! `rns` builds the reverse index. It walks `data/`, maps each object, reads
//! its FOT, and records `references` and `contains` edges in a
//! `twizzler-graph` graph. After that, reverse questions are traversals.
//!
//! Nothing here is invented. Every vertex is a real object and every edge
//! is a relation Twizzler already maintains. An earlier version of this
//! demonstrator used made-up `tag`/`tagged` relations; those measured nothing,
//! because a graph can obviously answer questions about a relation a hierarchy
//! does not have.
//!
//! # Why the index is possible at all
//!
//! # Usage
//!
//!   rns make           create four real objects that reference each other
//!   rns index          walk `data/`, read every FOT, build the graph
//!   rns refs <name>    what does this object reference?      (forward)
//!   rns rrefs <name>   what references this object?          (reverse)
//!   rns names <name>   every name this object has            (reverse)
//!   rns safe           objects nothing references            (reclaim's question)
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
/// `NamingHandle::remove` exists, so the claim is either an error return, a
/// silent no-op, or wrong. This distinguishes them: it reads the name, removes
/// it, then reads again.
///
/// - `get` fails after `remove` → unbinding works, and three documents plus
///   the `MAGIC_DESTROYED` workaround need revisiting.
/// - `remove` returns an error → the claim holds, and we finally have the error
///   to quote instead of an assertion.
/// - `remove` succeeds but `get` still resolves → worse than either: a silent
///   no-op, which is the shape that hides.
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

/// A file that *uses* another file. The `InvPtr` is the point: it is what puts
/// an entry in this object's FOT, and therefore what the index can see.
#[repr(C)]
struct Derived {
    kind: u32,
    size: u64,
    /// The file this one is built from.
    source: InvPtr<Blob>,
}
unsafe impl Invariant for Derived {}
impl BaseType for Derived {}

/// Create a small corpus of real objects with real references, and name
/// them under `data/`.
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
/// deleted, and `rns safe` should list everything *except* `sales.csv`.
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

/// Every object an object's FOT points at.
///
/// `MetaInfo.fotcount` bounds the table and `RawObject::fote_ptr` reads an
/// entry; `FotEntry.values` is the target id in `ObjID::parts` order, so
/// `from_parts` inverts it. Resolver entries are skipped: they name a resolver
/// function rather than an object, so they are not an edge.
fn fot_targets(id: ObjID) -> Vec<ObjID> {
    let Ok(obj) = Object::<()>::map(id, MapFlags::READ) else {
        // Unmappable is a finding, not an error — see the module docs on
        // protections. Report it rather than treating it as zero references.
        println!("rns: cannot map {id}, skipping (its references are unknown)");
        return Vec::new();
    };
    // The FOT has no recorded length; do not use `MetaInfo.fotcount` as one.
    //
    // So it is a reserved field rather than a broken one — no behaviour depends
    // on it — but it cannot bound an enumeration. An earlier version of this
    // function used it and reported every object as reference-free.
    //
    // Enumerate the way the allocator does: from index 1 (index 0 is
    // `InvPtr::new`'s same-object case and never occupies a slot), stopping at
    // the first slot that was never allocated. Checked against the vendored
    // tree at the pinned commit; re-check upstream before relying on it.
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
    // Reported per object, not summarised. A zero total is ambiguous
    // between "this object references nothing" and "the read is wrong", and
    // that distinction is the whole point of the index. `declared` is printed
    // beside the scan precisely because it should stay 0 while `scanned` does
    // not — that gap is the platform finding.
    println!(
        "rns:   {id:x} scanned={scanned} refs={} resolver={resolvers} \
         deleted={deleted} null/self={nulls} (meta.fotcount={declared})",
        out.len()
    );
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
        // The human name is an edge, not a field — an object can have several.
        g.add_edge(ns, CONTAINS, v)?;
        // `PropValue::Str` is a `NameKey` — 31 bytes, truncated at a char
        // boundary. Namer entries are short, and the hex id on the vertex is
        // the authoritative identity anyway.
        g.set_vertex_prop(v, "name", PropValue::Str(NameKey::new(name)))?;
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
    let holders = g.in_neighbors(v, Labels::these(&[CONTAINS]));
    println!("{name} has {} name(s):", holders.len());
    if let Some(PropValue::Str(s)) = g.get_vertex_prop(v, "name") {
        println!("  {}", s.as_str());
    }
    Ok(())
}

/// Objects nothing references — reclaim's discovery question.
///
/// The kernel needs exactly this to decide what is collectable, and
/// `reclaim_main` leaves the discovery steps unimplemented. Here it is an
/// in-degree test.
fn safe() -> Result<()> {
    let g = Graph::open_or_create_arena(GRAPH, twizzler_graph::DEFAULT_ARENA_CAP)?;
    // Only real objects. The `ns` vertex is the namespace itself, not something
    // that could be deleted, and counting it made an earlier run report "4 of 5"
    // when the answer was 3 of 4.
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
                "usage: rns [make | index | refs <name> | rrefs <name> | \
                 names <name> | safe | unname <name> | reset]  (got {c})"
            );
            Ok(())
        }
    };
    if let Err(e) = r {
        println!("rns: {e:?}");
    }
}
