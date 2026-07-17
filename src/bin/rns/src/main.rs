//! A relational namespace (files-by-relation) on `twizzler-graph`.
//!
//! Usage (in the Twizzler shell):
//!   rns build     create file/tag vertices and `tagged` edges (idempotent)
//!   rns lookup    list files tagged `thesis`
//!   rns all       build then lookup (default)
//!   rns reset     clear the graph

use twizzler::{
    marker::{BaseType, Invariant},
    object::{ObjID, ObjectBuilder},
};
use twizzler_graph::{Graph, GraphError, Labels, VertexId};

/// Result alias for the app (engine error type).
type Result<T> = core::result::Result<T, GraphError>;

/// Payload of a "file" object that the namespace names. Its `ObjID` becomes the
/// `target` property of the file vertex.
#[derive(Clone, Copy)]
#[repr(C)]
struct FileMeta {
    kind: u32,
    size: u64,
}
unsafe impl Invariant for FileMeta {}
impl BaseType for FileMeta {}

fn build() -> Result<()> {
    let mut g = Graph::open_or_create("rns")?;

    // Idempotent: if the thesis tag already exists, don't duplicate.
    if g.find_vertex("tag", "thesis").is_some() {
        println!("graph already built (data/rns); run `rns lookup`.");
        return Ok(());
    }

    // The underlying data objects being named.
    let doc_obj = ObjectBuilder::<FileMeta>::default()
        .persist(true)
        .build(FileMeta {
            kind: 0,
            size: 4_096,
        })?;
    let photo_obj = ObjectBuilder::<FileMeta>::default()
        .persist(true)
        .build(FileMeta {
            kind: 1,
            size: 220_000,
        })?;

    // Map files and a tag onto graph vertices, then relate them with edges.
    let doc = g.add_vertex("file", "doc", doc_obj.id())?;
    let photo = g.add_vertex("file", "photo", photo_obj.id())?;
    let thesis = g.add_vertex("tag", "thesis", ObjID::new(0))?;

    g.add_edge(doc, "tagged", thesis)?;
    g.add_edge(photo, "tagged", thesis)?;

    println!("built graph 'rns' (root {})", g.root_id());
    println!("  doc   -> {}", doc_obj.id());
    println!("  photo -> {}", photo_obj.id());
    Ok(())
}

fn lookup() -> Result<()> {
    let g = Graph::open_or_create("rns")?;

    let Some(thesis) = g.find_vertex("tag", "thesis") else {
        println!("no 'thesis' tag found; run `rns build` first.");
        return Ok(());
    };

    println!("files tagged 'thesis':");
    let mut count = 0;
    // Incoming `tagged` edges into the thesis vertex.
    let neighbors: Vec<VertexId> = g.in_neighbors(thesis, Labels::these(&["tagged"]));
    for v in neighbors {
        if let Some(info) = g.vertex_info(v) {
            println!("  - {} (target {})", info.name, info.target);
            count += 1;
        }
    }
    println!("\nlookup-by-relation returned {count} file(s).");
    Ok(())
}

/// Clear the `rns` graph (e.g. to replace a stale-format one).
fn reset() -> Result<()> {
    Graph::reset("rns")?;
    println!("reset: graph 'rns' cleared (run `rns build` to repopulate).");
    Ok(())
}

fn main() {
    println!("rns: relational namespace on the twizzler-graph engine\n");

    let cmd = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let result = match cmd.as_str() {
        "build" => build(),
        "lookup" => lookup(),
        "all" => build().and_then(|_| {
            println!();
            lookup()
        }),
        "reset" => reset(),
        other => {
            println!("usage: rns [build|lookup|all|reset]  (got '{other}')");
            Ok(())
        }
    };

    if let Err(e) = result {
        match e {
            GraphError::StaleVersion { found, expected } => {
                println!(
                    "rns: graph 'rns' has a stale on-disk format (version {found}, \
                     engine expects {expected})."
                );
                println!("     the existing graph was left intact.");
                println!("     run `rns reset` to discard it, then `rns build`.");
            }
            other => println!("rns error: {other}"),
        }
    }
}
