//! Files are baked into the initrd by `scripts/flatten_ldbc.py` (host side) and
//! appear flat as `<entity>.csv`, pipe-separated with a header line.
//!
//! # What is a node, an edge, and neither
//!
//! Classification is by filename, and the rule is exact rather than a list:
//! `X_rel_Y` is an edge iff both `X` and `Y` are node entities. That makes
//! `person_email_emailaddress` and `person_speaks_language` multi-valued
//! *properties* — there is no entity on the far side to point at — and it keeps
//! `params` and the three `updateStream*` files out entirely. Those last are the
//! benchmark's query parameters and update workload; `updateStream_0_0_forum`
//! alone is 287 k rows, so ingesting it as graph data would inflate every count
//! by 16% and look entirely plausible.
//!
//! # How a value is stored, and why it is three tiers rather than two
//!
//! # Ids
//!
//! An LDBC id becomes the vertex *name*, so `find_vertex(label, id)` resolves an
//! edge endpoint. All eight node labels are therefore declared indexed —
//! 327,588 of 1,805,553 records, ~1:5.5. That ratio is why `RebuildSource::Scan`
//! is used rather than `Roots`: at 1:1 `Roots` measured a net loss and at 1:1000
//! a 1.75 s win, and 1:5.5 is far nearer the losing end.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use twizzler::object::ObjID;
use twizzler_graph::{
    Graph, IndexSchema, IndexStrategy, Lookup, PropValue, VertexId, DEFAULT_ARENA_CAP,
    MAX_TEXT_LEN,
};

const NAME: &str = "ldbc";
const DIR: &str = "/initrd";

/// The eight node entities. Everything else is an edge, a property table, or not
/// graph data at all.
const NODES: &[&str] = &[
    "comment",
    "forum",
    "organisation",
    "person",
    "place",
    "post",
    "tag",
    "tagclass",
];

/// Not graph data: query parameters and the update workload.
fn is_not_graph(stem: &str) -> bool {
    stem == "params" || stem.starts_with("updateStream")
}

/// `(from_label, to_label)` if this filename names an edge.
fn edge_of(stem: &str) -> Option<(&'static str, &'static str)> {
    let parts: Vec<&str> = stem.split('_').collect();
    if parts.len() != 3 {
        return None;
    }
    let from = NODES.iter().find(|n| **n == parts[0])?;
    let to = NODES.iter().find(|n| **n == parts[2])?;
    Some((from, to))
}

struct Counts {
    vertices: usize,
    edges: usize,
    short: usize,
    text: usize,
    blobs: usize,
    skipped_missing: usize,
}

pub(crate) fn run() {
    println!(
        "GSTRESS STAMP harness={} mode=ldbc cap={}",
        crate::HARNESS_REV,
        DEFAULT_ARENA_CAP
    );

    // `Scan`, not `Roots`: see the module note on the 1:5.5 indexed ratio.
    let schema = IndexSchema::new(IndexStrategy::LazyLabel);
    Graph::reset_arena_with_index(NAME, DEFAULT_ARENA_CAP, schema).expect("reset");
    let mut g =
        Graph::open_or_create_arena_with_index(NAME, DEFAULT_ARENA_CAP, schema).expect("open");
    for l in NODES {
        g.set_label_indexed(l, true).expect("declare");
    }

    let mut c = Counts {
        vertices: 0,
        edges: 0,
        short: 0,
        text: 0,
        blobs: 0,
        skipped_missing: 0,
    };

    // Nodes before edges, necessarily: an edge row names both endpoints by
    // id, and an id cannot resolve before the vertex exists.
    let t_nodes = Instant::now();
    for n in NODES {
        load_nodes(&mut g, n, &mut c);
    }
    let nodes_s = t_nodes.elapsed().as_secs_f64();
    println!(
        "GSTRESS LDBC nodes: {} vertices in {nodes_s:.1}s ({:.0}/s), {} arenas",
        c.vertices,
        c.vertices as f64 / nodes_s.max(1e-9),
        g.arena_count()
    );

    let t_edges = Instant::now();
    let mut edge_files: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(DIR) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let Some(stem) = name.strip_suffix(".csv") else {
                continue;
            };
            if is_not_graph(stem) || NODES.contains(&stem) {
                continue;
            }
            edge_files.push(stem.to_string());
        }
    }
    edge_files.sort();
    for stem in &edge_files {
        match edge_of(stem) {
            Some((from, to)) => load_edges(&mut g, stem, from, to, &mut c),
            // Three parts but not two node labels, or not three parts at all:
            // a multi-valued property table (`person_email_emailaddress`).
            None => load_prop_table(&mut g, stem, &mut c),
        }
    }
    let edges_s = t_edges.elapsed().as_secs_f64();
    println!(
        "GSTRESS LDBC edges: {} edges in {edges_s:.1}s ({:.0}/s), {} arenas",
        c.edges,
        c.edges as f64 / edges_s.max(1e-9),
        g.arena_count()
    );

    let t_sync = Instant::now();
    g.sync().expect("sync");
    let sync_s = t_sync.elapsed().as_secs_f64();

    println!(
        "GSTRESS LDBC TOTALS: {} vertices + {} edges = {} records | {} arenas | \
         {} blob objects | props: {} short, {} text, {} blob | \
         nodes {nodes_s:.1}s edges {edges_s:.1}s sync {sync_s:.1}s = {:.1}s",
        c.vertices,
        c.edges,
        c.vertices + c.edges,
        g.arena_count(),
        g.blob_object_count(),
        c.short,
        c.text,
        c.blobs,
        nodes_s + edges_s + sync_s
    );
    if c.skipped_missing > 0 {
        println!(
            "GSTRESS LDBC WARNING: {} edge rows named an endpoint that did not \
             resolve. That is data loss, not a rounding error — investigate \
             before using any number from this run.",
            c.skipped_missing
        );
    }
}

fn open_csv(stem: &str) -> Option<(Vec<String>, BufReader<File>)> {
    let f = File::open(format!("{DIR}/{stem}.csv")).ok()?;
    let mut r = BufReader::new(f);
    let mut head = String::new();
    r.read_line(&mut head).ok()?;
    let cols: Vec<String> = head.trim_end().split('|').map(|s| s.to_string()).collect();
    Some((cols, r))
}

fn load_nodes(g: &mut Graph, label: &str, c: &mut Counts) {
    let Some((cols, r)) = open_csv(label) else {
        println!("GSTRESS LDBC: {label}.csv absent, skipping");
        return;
    };
    let mut n = 0usize;
    for line in r.lines().map_while(|l| l.ok()) {
        let f: Vec<&str> = line.trim_end().split('|').collect();
        if f.len() < cols.len() || f[0].is_empty() {
            continue;
        }
        let v = match g.add_vertex(label, f[0], ObjID::new(0)) {
            Ok(v) => v,
            Err(e) => {
                println!("GSTRESS LDBC FAIL: {label} row {n}: {e:?}");
                return;
            }
        };
        // Column 0 is the id, which is already the vertex name.
        for (i, col) in cols.iter().enumerate().skip(1) {
            set_value(g, v, col, f[i], c);
        }
        c.vertices += 1;
        n += 1;
        if n % 25_000 == 0 {
            println!("GSTRESS LDBC   {label}: {n} rows");
        }
    }
    println!("GSTRESS LDBC {label}: {n} vertices");
}

/// Store one value at the cheapest tier that preserves what queries need of it.
fn set_value(g: &mut Graph, v: VertexId, key: &str, raw: &str, c: &mut Counts) {
    if raw.is_empty() {
        return;
    }
    let n = raw.len();
    let r = if n <= 31 {
        // Stays a `PropValue` so it remains orderable — every LDBC sort is on
        // `creationDate` (28 B) or `id`.
        c.short += 1;
        g.set_vertex_prop(v, key, PropValue::str(raw))
    } else if n <= MAX_TEXT_LEN {
        c.text += 1;
        g.set_vertex_text(v, key, raw)
    } else {
        c.blobs += 1;
        g.set_vertex_blob(v, key, raw.as_bytes())
    };
    if let Err(e) = r {
        println!("GSTRESS LDBC FAIL: property {key} ({n} B): {e:?}");
    }
}

fn resolve(g: &Graph, label: &str, id: &str) -> Option<VertexId> {
    match g.find_vertex(label, id) {
        Lookup::Found(v) => Some(v),
        Lookup::NotFound => None,
        // Declared above, so this cannot happen — and if it does, the load is
        // producing an empty graph while looking healthy.
        Lookup::NotIndexed => panic!("{label} is not indexed; the load would be silently empty"),
    }
}

fn load_edges(g: &mut Graph, stem: &str, from: &str, to: &str, c: &mut Counts) {
    let Some((cols, r)) = open_csv(stem) else {
        return;
    };
    // The relation name is the middle segment; it becomes the edge label.
    let rel = stem.split('_').nth(1).unwrap_or(stem).to_string();
    let mut n = 0usize;
    for line in r.lines().map_while(|l| l.ok()) {
        let f: Vec<&str> = line.trim_end().split('|').collect();
        if f.len() < 2 {
            continue;
        }
        let (Some(a), Some(b)) = (resolve(g, from, f[0]), resolve(g, to, f[1])) else {
            c.skipped_missing += 1;
            continue;
        };
        match g.add_edge(a, &rel, b) {
            Ok(e) => {
                // Extra columns are edge properties (`knows.creationDate`,
                // `likes.creationDate`, `studyAt.classYear`).
                for (i, col) in cols.iter().enumerate().skip(2) {
                    if i < f.len() && !f[i].is_empty() && f[i].len() <= 31 {
                        let _ = g.set_edge_prop(e, col, PropValue::str(f[i]));
                    }
                }
            }
            Err(e) => {
                println!("GSTRESS LDBC FAIL: {stem} row {n}: {e:?}");
                return;
            }
        }
        c.edges += 1;
        n += 1;
        if n % 25_000 == 0 {
            println!("GSTRESS LDBC   {stem}: {n} rows");
        }
    }
    println!("GSTRESS LDBC {stem}: {n} edges");
}

/// A multi-valued property table: `<entity>.id | value`. Stored on the entity
/// under the file's own name, so `person_email_emailaddress` becomes a property
/// keyed `email` — one value survives per person, which is a restriction, not
/// a rounding: LDBC allows several and the record format holds one per key.
fn load_prop_table(g: &mut Graph, stem: &str, c: &mut Counts) {
    let Some((_cols, r)) = open_csv(stem) else {
        return;
    };
    let parts: Vec<&str> = stem.split('_').collect();
    let Some(owner) = NODES.iter().find(|n| **n == parts[0]) else {
        println!("GSTRESS LDBC: {stem} has no known owner entity, skipping");
        return;
    };
    let key = parts.get(1).copied().unwrap_or("value").to_string();
    let mut n = 0usize;
    let mut multi = 0usize;
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for line in r.lines().map_while(|l| l.ok()) {
        let f: Vec<&str> = line.trim_end().split('|').collect();
        if f.len() < 2 {
            continue;
        }
        let Some(v) = resolve(g, owner, f[0]) else {
            c.skipped_missing += 1;
            continue;
        };
        if !seen.insert(v.0) {
            multi += 1;
            continue;
        }
        set_value(g, v, &key, f[1], c);
        n += 1;
    }
    println!(
        "GSTRESS LDBC {stem}: {n} values as `{key}` ({multi} additional values \
         dropped — one per key is a format restriction, report it)"
    );
}
