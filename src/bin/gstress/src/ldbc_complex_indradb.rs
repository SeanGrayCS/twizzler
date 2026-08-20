//! The fourteen LDBC complex reads (IC1–IC14) against IndraDB, the comparison
//! baseline.
//!
//! Mirrors `ldbc_complex.rs` query for query. Where the native arm dereferences
//! adjacency, this issues keyed lookups. Written against IndraDB's public API
//! only, exactly as `ldbc_indradb.rs` is, so it reads as a fair account of what
//! the baseline offers a query author rather than a reach-through to internals.
//!
//! IndraDB gives a vertex a UUID and a type, so an LDBC id is an ordinary
//! property and every entry lookup is a property-index query, while the native
//! engine's `(label, name)` identity resolves one directly. That index is
//! affordable only because `load` declares it before inserting anything; one
//! cannot be built after the fact, so the queries parameterised by
//! human-readable names (IC3, IC6, IC11, IC12) resolve them with a startup
//! scan instead — see [`Ctx::build`]. Both arms pay a setup cost before the
//! timed loop, both exclude it, and both report it.
//!
//! IC1 (`classYear`, `workFrom`), IC5 (`joinDate`), IC7 (`likes.creationDate`)
//! and IC11 (`workFrom`) read edge properties, which exist only in a store
//! built with `gstress ldbc-indradb-load edgeprops`. This arm probes for them
//! at startup and refuses the affected queries rather than answering wrongly.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::time::Instant;

use indradb::{
    Database, Edge, Identifier, Json, QueryExt, QueryOutputValue, SpecificEdgeQuery,
    SpecificVertexQuery, VertexWithPropertyValueQuery,
};
use twizzler_indradb::TwizzlerDatastore;
use uuid::Uuid;

use crate::ldbc_common::{load_params, parse_args, san, self_check, Budget, Digest, Lat, Params};

const DB: &str = "ldbc-idb";
const P_ID: &str = "ldbcId";
const TAG: &str = "GSTRESS IDBC";

const MAX_HOPS: usize = 64;
const IC14_PATH_CAP: usize = 20_000;

type Db = Database<TwizzlerDatastore>;

fn ident(s: &str) -> Identifier {
    Identifier::new(s).expect("valid identifier")
}

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/// A property-index lookup. Only usable for `ldbcId`, which is indexed because
/// `load` declares it before inserting anything; every other property would
/// answer `NotIndexed`, and indexing one after the fact is not affordable
/// (see [`Ctx::build`]).
fn by_prop(db: &Db, t: &str, key: &str, value: &str) -> Option<Uuid> {
    let q = VertexWithPropertyValueQuery::new(ident(key), Json::new(value.into()));
    let out = db.get(q).ok()?;
    let want = ident(t);
    match out.last()? {
        QueryOutputValue::Vertices(vs) => vs.iter().find(|v| v.t == want).map(|v| v.id),
        _ => None,
    }
}

fn by_id(db: &Db, t: &str, id: &str) -> Option<Uuid> {
    by_prop(db, t, P_ID, id)
}

/// One property of one vertex.
///
/// The `.name(..)` filter matters: `PipePropertyQuery` carries an optional
/// name filter, and without it the adapter serves the query from
/// `all_vertex_properties_for_vertex`, which `scan_prefix`es and materialises
/// every property of the vertex in order to return one. With it, the query
/// resolves to `vertex_property` — a single keyed get.
fn vprop(db: &Db, id: Uuid, key: &str) -> String {
    let want = ident(key);
    let Ok(q) = SpecificVertexQuery::single(id).properties() else {
        return String::new();
    };
    let q = q.name(want);
    let Ok(out) = db.get(q) else {
        return String::new();
    };
    match out.last() {
        Some(QueryOutputValue::VertexProperties(ps)) => ps
            .iter()
            .flat_map(|p| p.props.iter())
            .find(|p| p.name == want)
            .and_then(|p| p.value.0.as_str().map(str::to_string))
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// An edge property. Absent unless the store was built with
/// `gstress ldbc-indradb-load edgeprops`.
fn eprop(db: &Db, e: &Edge, key: &str) -> Option<i64> {
    let want = ident(key);
    let q = SpecificEdgeQuery::single(e.clone()).properties().ok()?.name(want);
    let out = db.get(q).ok()?;
    match out.last()? {
        QueryOutputValue::EdgeProperties(ps) => ps
            .iter()
            .flat_map(|p| p.props.iter())
            .find(|p| p.name == want)
            .and_then(|p| p.value.0.as_str())
            .and_then(|s| s.parse().ok()),
        _ => None,
    }
}

/// Outgoing edges of one type.
///
/// The `.t(..)` filter matters for the same reason `vprop`'s `.name(..)` does:
/// `PipeQuery` carries an optional type filter, and without it the adapter
/// returns every incident edge and the filtering happens in this process.
fn out_edges(db: &Db, id: Uuid, t: &str) -> Vec<Edge> {
    let Ok(q) = SpecificVertexQuery::single(id).outbound() else {
        return Vec::new();
    };
    let Ok(out) = db.get(q.t(ident(t))) else {
        return Vec::new();
    };
    match out.last() {
        Some(QueryOutputValue::Edges(es)) => es.clone(),
        _ => Vec::new(),
    }
}

fn in_edges(db: &Db, id: Uuid, t: &str) -> Vec<Edge> {
    let Ok(q) = SpecificVertexQuery::single(id).inbound() else {
        return Vec::new();
    };
    let Ok(out) = db.get(q.t(ident(t))) else {
        return Vec::new();
    };
    match out.last() {
        Some(QueryOutputValue::Edges(es)) => es.clone(),
        _ => Vec::new(),
    }
}

/// Neighbours across an outgoing edge of the given type. Used for the small,
/// one-off reads (a single person's city, a single message's parent); the bulk
/// paths go through the batched helpers below.
fn out_n(db: &Db, id: Uuid, t: &str) -> Vec<Uuid> {
    out_edges(db, id, t).into_iter().map(|e| e.inbound_id).collect()
}

fn first_out(db: &Db, id: Uuid, t: &str) -> Option<Uuid> {
    out_n(db, id, t).into_iter().next()
}

fn lid(db: &Db, id: Uuid) -> i64 {
    vprop(db, id, P_ID).parse().unwrap_or(-1)
}

fn country_of(db: &Db, place: Uuid) -> Option<Uuid> {
    let mut cur = place;
    for _ in 0..8 {
        if vprop(db, cur, "type") == "country" {
            return Some(cur);
        }
        cur = first_out(db, cur, "isPartOf")?;
    }
    None
}

// ---------------------------------------------------------------------------
// Batched primitives
// ---------------------------------------------------------------------------
//
// IndraDB's query language is set-oriented: `SpecificVertexQuery::new(Vec<Uuid>)`
// takes many ids, and `.properties().name(k)` / `.outbound().t(t)` push the
// filters into the query. Asking for one vertex at a time is using it badly, so
// the bulk paths below fetch a whole frontier per `db.get` call. Batching
// changes only the round-trip count, not the answers.

/// Ids per query: large enough that the query count stops mattering, small
/// enough that one result set is not the whole store.
const BATCH: usize = 4096;

/// One named property of many vertices.
fn props_batch(db: &Db, ids: &[Uuid], key: &str) -> HashMap<Uuid, String> {
    let mut out = HashMap::with_capacity(ids.len());
    if ids.is_empty() {
        return out;
    }
    let want = ident(key);
    for chunk in ids.chunks(BATCH) {
        let Ok(q) = SpecificVertexQuery::new(chunk.to_vec()).properties() else {
            continue;
        };
        let Ok(res) = db.get(q.name(want)) else { continue };
        if let Some(QueryOutputValue::VertexProperties(ps)) = res.last() {
            for vp in ps {
                if let Some(p) = vp.props.iter().find(|p| p.name == want) {
                    if let Some(v) = p.value.0.as_str() {
                        out.insert(vp.vertex.id, v.to_string());
                    }
                }
            }
        }
    }
    out
}

/// As [`props_batch`], parsed. Absent and unparseable are both simply absent —
/// every caller treats a missing date or year as "does not qualify", which is
/// what the reference queries do with a null.
fn nums_batch(db: &Db, ids: &[Uuid], key: &str) -> HashMap<Uuid, i64> {
    props_batch(db, ids, key)
        .into_iter()
        .filter_map(|(k, v)| v.parse().ok().map(|n| (k, n)))
        .collect()
}

/// The type of many vertices, for the posts-only filters.
fn types_batch(db: &Db, ids: &[Uuid]) -> HashMap<Uuid, String> {
    let mut out = HashMap::with_capacity(ids.len());
    if ids.is_empty() {
        return out;
    }
    for chunk in ids.chunks(BATCH) {
        let Ok(res) = db.get(SpecificVertexQuery::new(chunk.to_vec())) else {
            continue;
        };
        if let Some(QueryOutputValue::Vertices(vs)) = res.last() {
            for v in vs {
                out.insert(v.id, v.t.as_str().to_string());
            }
        }
    }
    out
}

/// Edges of one type incident to many vertices, in one direction.
///
/// The source vertex is recoverable from the edge itself — `outbound_id` when
/// `outbound`, `inbound_id` otherwise — so the caller does not lose the
/// association by batching.
fn edges_batch(db: &Db, ids: &[Uuid], t: &str, outbound: bool) -> Vec<Edge> {
    let mut out = Vec::new();
    if ids.is_empty() {
        return out;
    }
    let want = ident(t);
    for chunk in ids.chunks(BATCH) {
        let q = SpecificVertexQuery::new(chunk.to_vec());
        let piped = if outbound { q.outbound() } else { q.inbound() };
        let Ok(piped) = piped else { continue };
        let Ok(res) = db.get(piped.t(want)) else { continue };
        if let Some(QueryOutputValue::Edges(es)) = res.last() {
            out.extend(es.iter().cloned());
        }
    }
    out
}

/// `(source, target)` pairs for a batched edge read, as a map. First wins,
/// which is exact here: every relation used this way in LDBC — `hasCreator`,
/// `isLocatedIn`, `replyOf`, `isPartOf` — is functional.
fn one_target(edges: &[Edge], outbound: bool) -> HashMap<Uuid, Uuid> {
    let mut m = HashMap::new();
    for e in edges {
        let (src, dst) = if outbound {
            (e.outbound_id, e.inbound_id)
        } else {
            (e.inbound_id, e.outbound_id)
        };
        m.entry(src).or_insert(dst);
    }
    m
}

/// All `(source, target)` pairs for a batched edge read, for the many-valued
/// relations (`hasTag`, `hasMember`, `containerOf`, `likes`, `knows`).
fn all_targets(edges: &[Edge], outbound: bool) -> Vec<(Uuid, Uuid)> {
    edges
        .iter()
        .map(|e| {
            if outbound {
                (e.outbound_id, e.inbound_id)
            } else {
                (e.inbound_id, e.outbound_id)
            }
        })
        .collect()
}

/// One named property of many edges, keyed by endpoint pair.
fn eprops_batch(db: &Db, edges: &[Edge], key: &str) -> HashMap<(Uuid, Uuid), i64> {
    let mut out = HashMap::with_capacity(edges.len());
    if edges.is_empty() {
        return out;
    }
    let want = ident(key);
    for chunk in edges.chunks(BATCH) {
        let Ok(q) = SpecificEdgeQuery::new(chunk.to_vec()).properties() else {
            continue;
        };
        let Ok(res) = db.get(q.name(want)) else { continue };
        if let Some(QueryOutputValue::EdgeProperties(ps)) = res.last() {
            for ep in ps {
                if let Some(p) = ep.props.iter().find(|p| p.name == want) {
                    if let Some(n) = p.value.0.as_str().and_then(|v| v.parse().ok()) {
                        out.insert((ep.edge.outbound_id, ep.edge.inbound_id), n);
                    }
                }
            }
        }
    }
    out
}

/// LDBC ids for many vertices at once — the batched `lid`.
fn lids(db: &Db, ids: &[Uuid]) -> HashMap<Uuid, i64> {
    nums_batch(db, ids, P_ID)
}

/// Breadth-first `knows` neighbourhood, two queries per level. `knows` is
/// undirected in LDBC and stored one-directionally, so both directions are
/// read.
fn hops(db: &Db, root: Uuid, max_depth: usize) -> Vec<(Uuid, usize)> {
    let mut seen: HashSet<Uuid> = HashSet::new();
    seen.insert(root);
    let mut frontier = vec![root];
    let mut out = Vec::new();
    for d in 1..=max_depth {
        let mut es = edges_batch(db, &frontier, "knows", true);
        es.extend(edges_batch(db, &frontier, "knows", false));
        let mut next = Vec::new();
        for e in &es {
            for w in [e.outbound_id, e.inbound_id] {
                if !seen.contains(&w) {
                    seen.insert(w);
                    out.push((w, d));
                    next.push(w);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    out
}

/// The country each place is in (itself, if it is one), walking `isPartOf`
/// upward one batched level at a time.
fn countries_of(db: &Db, places: &[Uuid]) -> HashMap<Uuid, Uuid> {
    let mut result: HashMap<Uuid, Uuid> = HashMap::new();
    let mut pending: Vec<(Uuid, Uuid)> = {
        let mut seen = HashSet::new();
        places
            .iter()
            .filter(|p| seen.insert(**p))
            .map(|p| (*p, *p))
            .collect()
    };
    for _ in 0..8 {
        if pending.is_empty() {
            break;
        }
        let mut cur: Vec<Uuid> = pending.iter().map(|(_, c)| *c).collect();
        cur.sort();
        cur.dedup();
        let types = props_batch(db, &cur, "type");
        let mut climb: Vec<(Uuid, Uuid)> = Vec::new();
        for (origin, c) in &pending {
            if types.get(c).map(String::as_str) == Some("country") {
                result.insert(*origin, *c);
            } else {
                climb.push((*origin, *c));
            }
        }
        if climb.is_empty() {
            break;
        }
        let mut ids: Vec<Uuid> = climb.iter().map(|(_, c)| *c).collect();
        ids.sort();
        ids.dedup();
        let parent = one_target(&edges_batch(db, &ids, "isPartOf", true), true);
        pending = climb
            .into_iter()
            .filter_map(|(o, c)| parent.get(&c).map(|p| (o, *p)))
            .collect();
    }
    result
}

/// Civil date from epoch milliseconds, UTC. Identical to the native arm's, so
/// IC10's window cannot be a source of disagreement between them.
fn month_day(millis: i64) -> (u32, u32) {
    let days = millis.div_euclid(86_400_000);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (m as u32, d as u32)
}

fn content_len(db: &Db, m: Uuid) -> usize {
    let c = vprop(db, m, "content");
    if c.is_empty() {
        vprop(db, m, "imageFile").len()
    } else {
        c.len()
    }
}

// ---------------------------------------------------------------------------
// Setup: the property index, and the edge-property probe
// ---------------------------------------------------------------------------

struct Ctx {
    /// Name → vertex for the three labels LDBC parameterises by name, built by
    /// a scan, exactly as the native arm's `Ctx` is.
    country: HashMap<String, Uuid>,
    tag: HashMap<String, Uuid>,
    class: HashMap<String, Uuid>,
    resolve_us: u128,
    /// False when the store was built without `edgeprops`; the four affected
    /// queries are then refused rather than answered wrongly.
    edge_props: bool,
}

impl Ctx {
    /// Resolve the parameter names by scanning, not by declaring an index.
    ///
    /// `TwizzlerDatastore::index_property` backfills by
    /// `scan_prefix(VERTEX_PROP_TAG)`, which materialises every vertex
    /// property in the store to find the few that carry a `name`. Declaring an
    /// index before a load is cheap because the backfill runs over an empty
    /// store, which is why `ldbcId` works; declaring one after is not
    /// affordable at this scale. So name resolution scans, in both arms.
    fn build(db: &Db, queries: &[usize], probe_person: Option<Uuid>) -> Ctx {
        let need_place = queries.iter().any(|q| matches!(q, 3 | 11));
        let need_tag = queries.contains(&6);
        let need_class = queries.contains(&12);
        let need_name = need_place || need_tag || need_class;

        let t = Instant::now();
        let (mut country, mut tag, mut class) = (HashMap::new(), HashMap::new(), HashMap::new());
        if need_name {
            // Collect the ids of interest first and let the full vertex list go
            // before reading any properties: `AllVertexQuery` materialises
            // every vertex, and only the places, tags and tag classes can
            // carry a name we want.
            let mut want: Vec<(Uuid, u8)> = Vec::new();
            {
                let out = db.get(indradb::AllVertexQuery);
                if let Ok(out) = out {
                    if let Some(QueryOutputValue::Vertices(vs)) = out.last() {
                        for v in vs {
                            let kind = match v.t.as_str() {
                                "place" if need_place => 0u8,
                                "tag" if need_tag => 1,
                                "tagclass" if need_class => 2,
                                _ => continue,
                            };
                            want.push((v.id, kind));
                        }
                    }
                }
            }
            for (id, kind) in want {
                match kind {
                    // Only countries: LDBC's `countryXName` names a Country,
                    // and city and country names are not disjoint.
                    0 if vprop(db, id, "type") == "country" => {
                        country.insert(vprop(db, id, "name"), id);
                    }
                    1 => {
                        tag.insert(vprop(db, id, "name"), id);
                    }
                    2 => {
                        class.insert(vprop(db, id, "name"), id);
                    }
                    _ => {}
                }
            }
        }
        let resolve_us = t.elapsed().as_micros();
        if need_name {
            println!(
                "{TAG} RESOLVE scanned name maps in {:.2}s ({} countries, {} tags, {} tag \
                 classes) — EXCLUDED from the latencies below. **Scanned, not indexed**: \
                 `index_property(name)` completed in 33.40s on 2026-08-19 and then exhausted \
                 the guest, because the adapter backfills by materialising every vertex \
                 property. Both arms therefore scan; see this function's doc comment.",
                resolve_us as f64 / 1e6,
                country.len(),
                tag.len(),
                class.len(),
            );
        }

        // Probe one `hasMember` edge for `joinDate`. A missing edge property is
        // indistinguishable from an unset one, so without this the four
        // affected queries would return empty or zero-weighted answers and look
        // healthy doing it.
        let edge_props = probe_person.map_or(false, |p| probe_edge_props(db, p));
        if !edge_props {
            println!(
                "{TAG} WARNING: this store has no edge properties — IC1, IC5, IC7 and IC11 \
                 cannot be answered and will be SKIPPED. Rebuild with \
                 `gstress ldbc-indradb-load edgeprops`. Note that is a *different, larger* load \
                 than the one RESULTS Table 17 measured; do not mix the two load timings."
            );
        }
        Ctx {
            country,
            tag,
            class,
            resolve_us,
            edge_props,
        }
    }

    fn charges(q: usize) -> bool {
        matches!(q, 3 | 6 | 11 | 12)
    }
    fn needs_edge_props(q: usize) -> bool {
        matches!(q, 1 | 5 | 7 | 11)
    }
}

/// True if this person's `hasMember` edges carry a `joinDate`.
///
/// Probed from one person rather than `AllEdgeQuery`, which would materialise
/// every edge in the store to answer a yes/no question. Every LDBC person
/// belongs to at least one forum, so a person with no `hasMember` edges at all
/// means something else is wrong and `false` is the right answer either way.
fn probe_edge_props(db: &Db, person: Uuid) -> bool {
    in_edges(db, person, "hasMember")
        .iter()
        .take(16)
        .any(|e| eprop(db, e, "joinDate").is_some())
}
// ---------------------------------------------------------------------------
// The fourteen queries
// ---------------------------------------------------------------------------
//
// Every bulk read below goes through the batched helpers.

/// One person's `knows` neighbours, both directions. Two queries.
fn knows(db: &Db, id: Uuid) -> Vec<Uuid> {
    let mut es = edges_batch(db, &[id], "knows", true);
    es.extend(edges_batch(db, &[id], "knows", false));
    es.iter()
        .map(|e| {
            if e.outbound_id == id {
                e.inbound_id
            } else {
                e.outbound_id
            }
        })
        .collect()
}

/// Every message authored by any of `people`, as `(message, author)`.
fn messages_of(db: &Db, people: &[Uuid]) -> Vec<(Uuid, Uuid)> {
    all_targets(&edges_batch(db, people, "hasCreator", false), false)
        .into_iter()
        .map(|(person, msg)| (msg, person))
        .collect()
}

fn ic1(db: &Db, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = by_id(db, "person", p.get(row, "personId")) else {
        return Vec::new();
    };
    let first = p.get(row, "firstName");
    let reached = hops(db, root, 3);
    let ids: Vec<Uuid> = reached.iter().map(|(v, _)| *v).collect();
    let firsts = props_batch(db, &ids, "firstName");
    let matched: Vec<(Uuid, usize)> = reached
        .into_iter()
        .filter(|(v, _)| firsts.get(v).map(String::as_str) == Some(first))
        .collect();
    let mids: Vec<Uuid> = matched.iter().map(|(v, _)| *v).collect();
    let lasts = props_batch(db, &mids, "lastName");
    let lid_of = lids(db, &mids);

    let mut cands: Vec<(usize, String, i64, Uuid)> = matched
        .into_iter()
        .map(|(v, d)| {
            (
                d,
                lasts.get(&v).cloned().unwrap_or_default(),
                lid_of.get(&v).copied().unwrap_or(-1),
                v,
            )
        })
        .collect();
    cands.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    cands.truncate(20);

    // At most 20 survivors, so the per-friend detail below is a bounded number
    // of small reads and batching it would buy nothing.
    let mut out = Vec::new();
    for (d, last, fid, f) in cands {
        let city = first_out(db, f, "isLocatedIn")
            .map(|c| vprop(db, c, "name"))
            .unwrap_or_default();
        let mut unis: Vec<String> = out_edges(db, f, "studyAt")
            .into_iter()
            .filter_map(|e| {
                let o = e.inbound_id;
                let city = first_out(db, o, "isLocatedIn")?;
                Some(format!(
                    "{}@{}@{}",
                    san(&vprop(db, o, "name")),
                    eprop(db, &e, "classYear")?,
                    san(&vprop(db, city, "name"))
                ))
            })
            .collect();
        unis.sort();
        let mut cos: Vec<String> = out_edges(db, f, "workAt")
            .into_iter()
            .filter_map(|e| {
                let o = e.inbound_id;
                let country = country_of(db, first_out(db, o, "isLocatedIn")?)?;
                Some(format!(
                    "{}@{}@{}",
                    san(&vprop(db, o, "name")),
                    eprop(db, &e, "workFrom")?,
                    san(&vprop(db, country, "name"))
                ))
            })
            .collect();
        cos.sort();
        out.push(format!(
            "{fid},{d},{},{},{},{}",
            san(&last),
            san(&city),
            unis.join("|"),
            cos.join("|")
        ));
    }
    out
}

fn ic2(db: &Db, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = by_id(db, "person", p.get(row, "personId")) else {
        return Vec::new();
    };
    let max_date = p.num(row, "maxDate");
    let friends = knows(db, root);
    let pairs = messages_of(db, &friends);
    let msgs: Vec<Uuid> = pairs.iter().map(|(m, _)| *m).collect();
    let dates = nums_batch(db, &msgs, "creationDate");
    let mids = lids(db, &msgs);
    let fids = lids(db, &friends);

    let mut rows: Vec<(i64, i64, i64, Uuid)> = pairs
        .into_iter()
        .filter_map(|(m, f)| {
            let cd = *dates.get(&m)?;
            if cd > max_date {
                return None;
            }
            Some((
                cd,
                mids.get(&m).copied().unwrap_or(-1),
                fids.get(&f).copied().unwrap_or(-1),
                m,
            ))
        })
        .collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(20);
    rows.into_iter()
        .map(|(cd, mid, fid, m)| format!("{fid},{mid},{cd},{}", content_len(db, m)))
        .collect()
}

fn ic3(db: &Db, ctx: &Ctx, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = by_id(db, "person", p.get(row, "personId")) else {
        return Vec::new();
    };
    let start = p.num(row, "startDate");
    let end = start + p.num(row, "durationDays") * 86_400_000;
    let (Some(cx), Some(cy)) = (
        ctx.country.get(p.get(row, "countryXName")).copied(),
        ctx.country.get(p.get(row, "countryYName")).copied(),
    ) else {
        return Vec::new();
    };

    // Friends and friends-of-friends whose own city is outside both countries.
    let fof: Vec<Uuid> = hops(db, root, 2).into_iter().map(|(v, _)| v).collect();
    let city_of = one_target(&edges_batch(db, &fof, "isLocatedIn", true), true);
    let cities: Vec<Uuid> = city_of.values().copied().collect();
    let city_country = countries_of(db, &cities);
    let eligible: Vec<Uuid> = fof
        .into_iter()
        .filter(|f| match city_of.get(f).and_then(|c| city_country.get(c)) {
            Some(c) => *c != cx && *c != cy,
            None => false,
        })
        .collect();

    // Their messages, inside the window, located in X or Y.
    let pairs = messages_of(db, &eligible);
    let msgs: Vec<Uuid> = pairs.iter().map(|(m, _)| *m).collect();
    let dates = nums_batch(db, &msgs, "creationDate");
    let in_window: Vec<(Uuid, Uuid)> = pairs
        .into_iter()
        .filter(|(m, _)| matches!(dates.get(m), Some(cd) if *cd >= start && *cd < end))
        .collect();
    let win_msgs: Vec<Uuid> = in_window.iter().map(|(m, _)| *m).collect();
    let msg_place = one_target(&edges_batch(db, &win_msgs, "isLocatedIn", true), true);
    let places: Vec<Uuid> = msg_place.values().copied().collect();
    let msg_country = countries_of(db, &places);

    let mut xy: HashMap<Uuid, (usize, usize)> = HashMap::new();
    for (m, f) in in_window {
        let Some(c) = msg_place.get(&m).and_then(|pl| msg_country.get(pl)) else {
            continue;
        };
        let e = xy.entry(f).or_insert((0, 0));
        if *c == cx {
            e.0 += 1;
        } else if *c == cy {
            e.1 += 1;
        }
    }
    let hits: Vec<Uuid> = xy
        .iter()
        .filter(|(_, (x, y))| *x > 0 && *y > 0)
        .map(|(f, _)| *f)
        .collect();
    let fids = lids(db, &hits);

    let mut rows: Vec<(usize, i64, usize, usize)> = hits
        .into_iter()
        .map(|f| {
            let (x, y) = xy[&f];
            (x + y, fids.get(&f).copied().unwrap_or(-1), x, y)
        })
        .collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(20);
    rows.into_iter()
        .map(|(_, fid, x, y)| format!("{fid},{x},{y}"))
        .collect()
}

fn ic4(db: &Db, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = by_id(db, "person", p.get(row, "personId")) else {
        return Vec::new();
    };
    let start = p.num(row, "startDate");
    let end = start + p.num(row, "durationDays") * 86_400_000;

    let friends = knows(db, root);
    let pairs = messages_of(db, &friends);
    let msgs: Vec<Uuid> = pairs.iter().map(|(m, _)| *m).collect();
    let types = types_batch(db, &msgs);
    let posts: Vec<Uuid> = msgs
        .into_iter()
        .filter(|m| types.get(m).map(String::as_str) == Some("post"))
        .collect();
    let dates = nums_batch(db, &posts, "creationDate");
    let tagged = all_targets(&edges_batch(db, &posts, "hasTag", true), true);

    let mut valid: HashMap<Uuid, usize> = HashMap::new();
    let mut invalid: HashMap<Uuid, usize> = HashMap::new();
    let mut seen: HashSet<(Uuid, Uuid)> = HashSet::new();
    for (post, t) in tagged {
        let Some(cd) = dates.get(&post).copied() else {
            continue;
        };
        // `WITH DISTINCT tag, post` — the spec's dedup.
        if !seen.insert((t, post)) {
            continue;
        }
        if cd >= start && cd < end {
            *valid.entry(t).or_insert(0) += 1;
        } else if cd < start {
            *invalid.entry(t).or_insert(0) += 1;
        }
    }
    let keep: Vec<Uuid> = valid
        .iter()
        .filter(|(t, c)| **c > 0 && invalid.get(*t).copied().unwrap_or(0) == 0)
        .map(|(t, _)| *t)
        .collect();
    let names = props_batch(db, &keep, "name");
    let mut rows: Vec<(usize, String)> = keep
        .into_iter()
        .map(|t| (valid[&t], san(names.get(&t).map(String::as_str).unwrap_or(""))))
        .collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(10);
    rows.into_iter().map(|(c, n)| format!("{n},{c}")).collect()
}

/// `friends` is scoped per forum: a post counts only when its creator joined
/// that forum after the cutoff, not any forum.
fn ic5(db: &Db, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = by_id(db, "person", p.get(row, "personId")) else {
        return Vec::new();
    };
    let min_date = p.num(row, "minDate");

    let fof: Vec<Uuid> = hops(db, root, 2).into_iter().map(|(v, _)| v).collect();
    let member_edges = edges_batch(db, &fof, "hasMember", false);
    let join = eprops_batch(db, &member_edges, "joinDate");
    let mut joined: HashMap<Uuid, HashSet<Uuid>> = HashMap::new();
    for e in &member_edges {
        if join
            .get(&(e.outbound_id, e.inbound_id))
            .map_or(false, |d| *d > min_date)
        {
            joined.entry(e.outbound_id).or_default().insert(e.inbound_id);
        }
    }

    let forums: Vec<Uuid> = joined.keys().copied().collect();
    let contained = all_targets(&edges_batch(db, &forums, "containerOf", true), true);
    let posts: Vec<Uuid> = contained.iter().map(|(_, p)| *p).collect();
    let creator = one_target(&edges_batch(db, &posts, "hasCreator", true), true);

    let mut counts: HashMap<Uuid, usize> = forums.iter().map(|f| (*f, 0)).collect();
    for (forum, post) in contained {
        if let (Some(c), Some(friends)) = (creator.get(&post), joined.get(&forum)) {
            if friends.contains(c) {
                *counts.entry(forum).or_insert(0) += 1;
            }
        }
    }
    let fids = lids(db, &forums);
    let titles = props_batch(db, &forums, "title");
    let mut rows: Vec<(usize, i64, String)> = forums
        .into_iter()
        .map(|f| {
            (
                counts.get(&f).copied().unwrap_or(0),
                fids.get(&f).copied().unwrap_or(-1),
                san(titles.get(&f).map(String::as_str).unwrap_or("")),
            )
        })
        .collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(20);
    rows.into_iter()
        .map(|(n, id, title)| format!("{id},{title},{n}"))
        .collect()
}

fn ic6(db: &Db, ctx: &Ctx, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = by_id(db, "person", p.get(row, "personId")) else {
        return Vec::new();
    };
    let Some(known) = ctx.tag.get(p.get(row, "tagName")).copied() else {
        return Vec::new();
    };

    let fof: Vec<Uuid> = hops(db, root, 2).into_iter().map(|(v, _)| v).collect();
    let pairs = messages_of(db, &fof);
    let msgs: Vec<Uuid> = pairs.iter().map(|(m, _)| *m).collect();
    let types = types_batch(db, &msgs);
    let posts: Vec<Uuid> = msgs
        .into_iter()
        .filter(|m| types.get(m).map(String::as_str) == Some("post"))
        .collect();

    let tagged = all_targets(&edges_batch(db, &posts, "hasTag", true), true);
    let mut by_post: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
    for (post, t) in tagged {
        by_post.entry(post).or_default().push(t);
    }
    let mut counts: HashMap<Uuid, usize> = HashMap::new();
    for tags in by_post.values() {
        if !tags.contains(&known) {
            continue;
        }
        for t in tags {
            if *t != known {
                *counts.entry(*t).or_insert(0) += 1;
            }
        }
    }
    let tids: Vec<Uuid> = counts.keys().copied().collect();
    let names = props_batch(db, &tids, "name");
    let mut rows: Vec<(usize, String)> = counts
        .into_iter()
        .map(|(t, c)| (c, san(names.get(&t).map(String::as_str).unwrap_or(""))))
        .collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(10);
    rows.into_iter().map(|(c, n)| format!("{n},{c}")).collect()
}

fn ic7(db: &Db, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = by_id(db, "person", p.get(row, "personId")) else {
        return Vec::new();
    };
    let own: Vec<Uuid> = messages_of(db, &[root]).into_iter().map(|(m, _)| m).collect();
    let like_edges = edges_batch(db, &own, "likes", false);
    let when_of = eprops_batch(db, &like_edges, "creationDate");
    let mids = lids(db, &own);

    let mut best: HashMap<Uuid, (i64, i64, Uuid)> = HashMap::new();
    for e in &like_edges {
        let Some(when) = when_of.get(&(e.outbound_id, e.inbound_id)).copied() else {
            continue;
        };
        let (liker, m) = (e.outbound_id, e.inbound_id);
        let mid = mids.get(&m).copied().unwrap_or(-1);
        // Latest like, ties broken by the smaller message id.
        let better = match best.get(&liker) {
            None => true,
            Some((w, i, _)) => when > *w || (when == *w && mid < *i),
        };
        if better {
            best.insert(liker, (when, mid, m));
        }
    }
    let likers: Vec<Uuid> = best.keys().copied().collect();
    let liker_ids = lids(db, &likers);
    let msg_dates = nums_batch(db, &own, "creationDate");
    let friends: HashSet<Uuid> = knows(db, root).into_iter().collect();

    let mut rows: Vec<(i64, i64, i64, i64, u8)> = best
        .into_iter()
        .filter_map(|(liker, (when, mid, m))| {
            let cd = msg_dates.get(&m).copied()?;
            Some((
                when,
                liker_ids.get(&liker).copied().unwrap_or(-1),
                mid,
                ((when - cd) / 1000) / 60,
                u8::from(!friends.contains(&liker)),
            ))
        })
        .collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(20);
    rows.into_iter()
        .map(|(w, l, m, lat, new)| format!("{l},{w},{m},{lat},{new}"))
        .collect()
}

/// No `DISTINCT` in the spec, and none here: a comment replies to exactly one
/// message, so it cannot double-count.
fn ic8(db: &Db, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = by_id(db, "person", p.get(row, "personId")) else {
        return Vec::new();
    };
    let own: Vec<Uuid> = messages_of(db, &[root]).into_iter().map(|(m, _)| m).collect();
    let replies: Vec<Uuid> = all_targets(&edges_batch(db, &own, "replyOf", false), false)
        .into_iter()
        .map(|(_, c)| c)
        .collect();
    let dates = nums_batch(db, &replies, "creationDate");
    let cids = lids(db, &replies);
    let author = one_target(&edges_batch(db, &replies, "hasCreator", true), true);
    let authors: Vec<Uuid> = author.values().copied().collect();
    let aids = lids(db, &authors);

    let mut rows: Vec<(i64, i64, i64)> = replies
        .into_iter()
        .filter_map(|c| {
            let cd = dates.get(&c).copied()?;
            let who = author.get(&c)?;
            Some((
                cd,
                cids.get(&c).copied().unwrap_or(-1),
                aids.get(who).copied().unwrap_or(-1),
            ))
        })
        .collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(20);
    rows.into_iter()
        .map(|(cd, c, who)| format!("{who},{c},{cd}"))
        .collect()
}

fn ic9(db: &Db, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = by_id(db, "person", p.get(row, "personId")) else {
        return Vec::new();
    };
    let max_date = p.num(row, "maxDate");
    let fof: Vec<Uuid> = hops(db, root, 2).into_iter().map(|(v, _)| v).collect();
    let pairs = messages_of(db, &fof);
    let msgs: Vec<Uuid> = pairs.iter().map(|(m, _)| *m).collect();
    let dates = nums_batch(db, &msgs, "creationDate");
    let mids = lids(db, &msgs);
    let fids = lids(db, &fof);

    let mut rows: Vec<(i64, i64, i64)> = pairs
        .into_iter()
        .filter_map(|(m, f)| {
            let cd = *dates.get(&m)?;
            if cd >= max_date {
                return None;
            }
            Some((
                cd,
                mids.get(&m).copied().unwrap_or(-1),
                fids.get(&f).copied().unwrap_or(-1),
            ))
        })
        .collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(20);
    rows.into_iter()
        .map(|(cd, m, f)| format!("{f},{m},{cd}"))
        .collect()
}

/// Score is `common − (total − common)`, and `postCount` counts posts only.
fn ic10(db: &Db, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = by_id(db, "person", p.get(row, "personId")) else {
        return Vec::new();
    };
    let month = p.num(row, "month") as u32;
    let next = month % 12 + 1;

    let direct: HashSet<Uuid> = knows(db, root).into_iter().collect();
    let dvec: Vec<Uuid> = direct.iter().copied().collect();
    let mut two = edges_batch(db, &dvec, "knows", true);
    two.extend(edges_batch(db, &dvec, "knows", false));
    let mut cands: Vec<Uuid> = Vec::new();
    let mut seen: HashSet<Uuid> = HashSet::new();
    for e in &two {
        for w in [e.outbound_id, e.inbound_id] {
            if w != root && !direct.contains(&w) && seen.insert(w) {
                cands.push(w);
            }
        }
    }

    let birthdays = nums_batch(db, &cands, "birthday");
    let cands: Vec<Uuid> = cands
        .into_iter()
        .filter(|f| match birthdays.get(f) {
            Some(bd) => {
                let (m, d) = month_day(*bd);
                (m == month && d >= 21) || (m == next && d < 22)
            }
            None => false,
        })
        .collect();

    let my_tags: HashSet<Uuid> = out_n(db, root, "hasInterest").into_iter().collect();
    let pairs = messages_of(db, &cands);
    let msgs: Vec<Uuid> = pairs.iter().map(|(m, _)| *m).collect();
    let types = types_batch(db, &msgs);
    let posts: Vec<(Uuid, Uuid)> = pairs
        .into_iter()
        .filter(|(m, _)| types.get(m).map(String::as_str) == Some("post"))
        .collect();
    let post_ids: Vec<Uuid> = posts.iter().map(|(m, _)| *m).collect();
    let tagged = all_targets(&edges_batch(db, &post_ids, "hasTag", true), true);
    let mut shares: HashSet<Uuid> = HashSet::new();
    for (post, t) in tagged {
        if my_tags.contains(&t) {
            shares.insert(post);
        }
    }
    let mut total: HashMap<Uuid, i64> = HashMap::new();
    let mut common: HashMap<Uuid, i64> = HashMap::new();
    for (m, f) in posts {
        *total.entry(f).or_insert(0) += 1;
        if shares.contains(&m) {
            *common.entry(f).or_insert(0) += 1;
        }
    }

    let city_of = one_target(&edges_batch(db, &cands, "isLocatedIn", true), true);
    let cities: Vec<Uuid> = city_of.values().copied().collect();
    let city_names = props_batch(db, &cities, "name");
    let fids = lids(db, &cands);

    let mut rows: Vec<(i64, i64, String)> = cands
        .into_iter()
        .map(|f| {
            let t = total.get(&f).copied().unwrap_or(0);
            let c = common.get(&f).copied().unwrap_or(0);
            let city = city_of
                .get(&f)
                .and_then(|c| city_names.get(c))
                .map(String::as_str)
                .unwrap_or("");
            (c - (t - c), fids.get(&f).copied().unwrap_or(-1), san(city))
        })
        .collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(10);
    rows.into_iter()
        .map(|(s, f, city)| format!("{f},{s},{city}"))
        .collect()
}

/// `ORDER BY workFrom ASC, personId ASC, organisationName DESC` — the third key
/// descends while the first two ascend.
fn ic11(db: &Db, ctx: &Ctx, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = by_id(db, "person", p.get(row, "personId")) else {
        return Vec::new();
    };
    let Some(cid) = ctx.country.get(p.get(row, "countryName")).copied() else {
        return Vec::new();
    };
    let year = p.num(row, "workFromYear");

    let fof: Vec<Uuid> = hops(db, root, 2).into_iter().map(|(v, _)| v).collect();
    let work = edges_batch(db, &fof, "workAt", true);
    let since = eprops_batch(db, &work, "workFrom");
    let orgs: Vec<Uuid> = work.iter().map(|e| e.inbound_id).collect();
    let org_types = props_batch(db, &orgs, "type");
    let org_names = props_batch(db, &orgs, "name");
    let org_place = one_target(&edges_batch(db, &orgs, "isLocatedIn", true), true);
    let places: Vec<Uuid> = org_place.values().copied().collect();
    let org_country = countries_of(db, &places);
    let fids = lids(db, &fof);

    let mut rows: Vec<(i64, i64, String)> = Vec::new();
    for e in &work {
        let (f, org) = (e.outbound_id, e.inbound_id);
        let Some(wf) = since.get(&(f, org)).copied() else {
            continue;
        };
        if wf >= year || org_types.get(&org).map(String::as_str) != Some("company") {
            continue;
        }
        let ok = org_place
            .get(&org)
            .and_then(|pl| org_country.get(pl))
            .map_or(false, |c| *c == cid);
        if ok {
            rows.push((
                wf,
                fids.get(&f).copied().unwrap_or(-1),
                san(org_names.get(&org).map(String::as_str).unwrap_or("")),
            ));
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(b.2.cmp(&a.2)));
    rows.truncate(10);
    rows.into_iter()
        .map(|(wf, f, name)| format!("{f},{name},{wf}"))
        .collect()
}

fn ic12(db: &Db, ctx: &Ctx, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = by_id(db, "person", p.get(row, "personId")) else {
        return Vec::new();
    };
    let Some(class) = ctx.class.get(p.get(row, "tagClassName")).copied() else {
        return Vec::new();
    };

    // Transitive subclass closure, walked downward.
    let mut closure: HashSet<Uuid> = HashSet::new();
    closure.insert(class);
    let mut frontier = vec![class];
    while !frontier.is_empty() {
        let subs = all_targets(&edges_batch(db, &frontier, "isSubclassOf", false), false);
        let mut next = Vec::new();
        for (_, sub) in subs {
            if closure.insert(sub) {
                next.push(sub);
            }
        }
        frontier = next;
    }
    let closure_v: Vec<Uuid> = closure.into_iter().collect();
    let tags: HashSet<Uuid> = all_targets(&edges_batch(db, &closure_v, "hasType", false), false)
        .into_iter()
        .map(|(_, t)| t)
        .collect();
    if tags.is_empty() {
        return Vec::new();
    }

    let friends = knows(db, root);
    let pairs = messages_of(db, &friends);
    let msgs: Vec<Uuid> = pairs.iter().map(|(m, _)| *m).collect();
    let types = types_batch(db, &msgs);
    let comments: Vec<(Uuid, Uuid)> = pairs
        .into_iter()
        .filter(|(m, _)| types.get(m).map(String::as_str) == Some("comment"))
        .collect();
    let cids: Vec<Uuid> = comments.iter().map(|(c, _)| *c).collect();
    // The reply must land on a post, not on another comment.
    let parent = one_target(&edges_batch(db, &cids, "replyOf", true), true);
    let parents: Vec<Uuid> = parent.values().copied().collect();
    let ptypes = types_batch(db, &parents);
    let post_parents: Vec<Uuid> = parents
        .into_iter()
        .filter(|m| ptypes.get(m).map(String::as_str) == Some("post"))
        .collect();
    let tagged = all_targets(&edges_batch(db, &post_parents, "hasTag", true), true);
    let mut hits_of: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
    for (post, t) in tagged {
        if tags.contains(&t) {
            hits_of.entry(post).or_default().push(t);
        }
    }

    let mut per_friend: HashMap<Uuid, (HashSet<Uuid>, BTreeSet<Uuid>)> = HashMap::new();
    for (c, f) in comments {
        let Some(par) = parent.get(&c) else { continue };
        let Some(hits) = hits_of.get(par) else { continue };
        let e = per_friend.entry(f).or_default();
        e.0.insert(c);
        for t in hits {
            e.1.insert(*t);
        }
    }
    let all_tags: Vec<Uuid> = per_friend
        .values()
        .flat_map(|(_, ts)| ts.iter().copied())
        .collect();
    let tag_names = props_batch(db, &all_tags, "name");
    let fids = lids(db, &friends);

    let mut rows: Vec<(usize, i64, String)> = per_friend
        .into_iter()
        .filter(|(_, (cs, _))| !cs.is_empty())
        .map(|(f, (cs, ts))| {
            let mut names: Vec<String> = ts
                .iter()
                .map(|t| san(tag_names.get(t).map(String::as_str).unwrap_or("")))
                .collect();
            names.sort();
            names.dedup();
            (
                cs.len(),
                fids.get(&f).copied().unwrap_or(-1),
                names.join("|"),
            )
        })
        .collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(20);
    rows.into_iter()
        .map(|(n, f, names)| format!("{f},{n},{names}"))
        .collect()
}

fn ic13(db: &Db, p: &Params, row: &[String]) -> Vec<String> {
    let (Some(a), Some(b)) = (
        by_id(db, "person", p.get(row, "person1Id")),
        by_id(db, "person", p.get(row, "person2Id")),
    ) else {
        return Vec::new();
    };
    if a == b {
        return vec!["0".to_string()];
    }
    let mut seen: HashSet<Uuid> = HashSet::new();
    seen.insert(a);
    let mut frontier = vec![a];
    for d in 1..=MAX_HOPS {
        let mut es = edges_batch(db, &frontier, "knows", true);
        es.extend(edges_batch(db, &frontier, "knows", false));
        let mut next = Vec::new();
        for e in &es {
            for w in [e.outbound_id, e.inbound_id] {
                if w == b {
                    return vec![d.to_string()];
                }
                if seen.insert(w) {
                    next.push(w);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    vec!["-1".to_string()]
}

/// Not expressible in the native arm's DSL, so both arms hand-roll
/// BFS-with-predecessor-lists. Returns `(rows, capped)`.
fn ic14(db: &Db, p: &Params, row: &[String]) -> (Vec<String>, bool) {
    let (Some(a), Some(b)) = (
        by_id(db, "person", p.get(row, "person1Id")),
        by_id(db, "person", p.get(row, "person2Id")),
    ) else {
        return (Vec::new(), false);
    };
    if a == b {
        return (vec![format!("{},0.0", lid(db, a))], false);
    }

    let mut dist: HashMap<Uuid, usize> = HashMap::new();
    let mut preds: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
    dist.insert(a, 0);
    let mut frontier = vec![a];
    let mut found = false;
    let mut depth = 0usize;
    while !frontier.is_empty() && !found && depth < MAX_HOPS {
        depth += 1;
        let mut es = edges_batch(db, &frontier, "knows", true);
        es.extend(edges_batch(db, &frontier, "knows", false));
        let cur: HashSet<Uuid> = frontier.iter().copied().collect();
        let mut next = Vec::new();
        for e in &es {
            // One endpoint is in the frontier; the other is the neighbour.
            let (v, w) = if cur.contains(&e.outbound_id) {
                (e.outbound_id, e.inbound_id)
            } else {
                (e.inbound_id, e.outbound_id)
            };
            match dist.get(&w) {
                None => {
                    dist.insert(w, depth);
                    preds.entry(w).or_default().push(v);
                    next.push(w);
                }
                Some(dw) if *dw == depth => {
                    preds.entry(w).or_default().push(v);
                }
                _ => {}
            }
            if w == b {
                found = true;
            }
        }
        frontier = next;
    }
    if !dist.contains_key(&b) {
        return (Vec::new(), false);
    }

    let mut paths: Vec<Vec<Uuid>> = Vec::new();
    let mut capped = false;
    let mut stack: Vec<(Uuid, Vec<Uuid>)> = vec![(b, vec![b])];
    while let Some((cur, acc)) = stack.pop() {
        if paths.len() >= IC14_PATH_CAP {
            capped = true;
            break;
        }
        if cur == a {
            let mut path = acc.clone();
            path.reverse();
            paths.push(path);
            continue;
        }
        for q in preds.get(&cur).map(Vec::as_slice).unwrap_or(&[]) {
            let mut nxt = acc.clone();
            nxt.push(*q);
            stack.push((*q, nxt));
        }
    }

    // Every person on any shortest path, so the weights and the rendered ids
    // are gathered in a bounded number of batched reads.
    let mut people: Vec<Uuid> = paths.iter().flatten().copied().collect();
    people.sort();
    people.dedup();
    let ids = lids(db, &people);
    let weights = interaction_weights(db, &people);

    let mut rows: Vec<(f64, String)> = Vec::new();
    for path in &paths {
        let mut w = 0.0f64;
        for pair in path.windows(2) {
            let (x, y) = (pair[0], pair[1]);
            let key = if x < y { (x, y) } else { (y, x) };
            w += weights.get(&key).copied().unwrap_or(0.0);
        }
        let rendered: Vec<String> = path
            .iter()
            .map(|v| ids.get(v).copied().unwrap_or(-1).to_string())
            .collect();
        rows.push((w, rendered.join("|")));
    }
    rows.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    (
        rows.into_iter()
            .map(|(w, ids)| format!("{ids},{w:.1}"))
            .collect(),
        capped,
    )
}

/// IC14's edge weights for every pair among `people`, in a handful of batched
/// reads: comment→post replies score 1.0 and comment→comment replies 0.5,
/// counted in both directions.
///
/// Gathers everyone's comments once, resolves each comment's parent and the
/// parent's author once, and then tallies. Pairs are keyed `(min, max)` so a
/// path traversing an edge in either direction finds it.
fn interaction_weights(db: &Db, people: &[Uuid]) -> HashMap<(Uuid, Uuid), f64> {
    let member: HashSet<Uuid> = people.iter().copied().collect();
    let pairs = messages_of(db, people);
    let msgs: Vec<Uuid> = pairs.iter().map(|(m, _)| *m).collect();
    let types = types_batch(db, &msgs);
    let comments: Vec<(Uuid, Uuid)> = pairs
        .into_iter()
        .filter(|(m, _)| types.get(m).map(String::as_str) == Some("comment"))
        .collect();
    let cids: Vec<Uuid> = comments.iter().map(|(c, _)| *c).collect();
    let parent = one_target(&edges_batch(db, &cids, "replyOf", true), true);
    let parents: Vec<Uuid> = parent.values().copied().collect();
    let ptypes = types_batch(db, &parents);
    let pauthor = one_target(&edges_batch(db, &parents, "hasCreator", true), true);

    let mut w: HashMap<(Uuid, Uuid), f64> = HashMap::new();
    for (c, author) in comments {
        let Some(par) = parent.get(&c) else { continue };
        let Some(other) = pauthor.get(par) else {
            continue;
        };
        if !member.contains(other) || *other == author {
            continue;
        }
        let key = if author < *other {
            (author, *other)
        } else {
            (*other, author)
        };
        let score = if ptypes.get(par).map(String::as_str) == Some("post") {
            1.0
        } else {
            0.5
        };
        *w.entry(key).or_insert(0.0) += score;
    }
    w
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

pub(crate) fn run() {
    let args = parse_args(2);
    println!(
        "GSTRESS STAMP harness={} mode=ldbc-complex-indradb iters={} queries={:?} budget={}",
        crate::HARNESS_REV,
        if args.digest_only { 0 } else { args.iters },
        args.queries,
        Budget::new(args.budget_s).describe(),
    );
    self_check(TAG);

    let t = Instant::now();
    let db = match TwizzlerDatastore::open_db(DB) {
        Ok(db) => db,
        Err(e) => {
            println!("{TAG} FAILED: cannot open `{DB}`: {e:?} — run `gstress ldbc-indradb-load edgeprops` first");
            return;
        }
    };
    println!(
        "{TAG} OPEN: {:.2}s (excluded from latencies)",
        t.elapsed().as_secs_f64()
    );
    twizzler_indradb::kv_stats::reset();

    // The store must hold what the load boot wrote, or the comparison is
    // against an empty store.
    let Some(p0) = load_params(1) else { return };
    let probe_id = p0.get(&p0.rows[0], "personId").to_string();
    let Some(probe_person) = by_id(&db, "person", &probe_id) else {
        println!(
            "{TAG} FAILED: `{DB}` has no person {probe_id} — run \
             `gstress ldbc-indradb-load edgeprops` first, in its own boot."
        );
        return;
    };

    let ctx = Ctx::build(&db, &args.queries, Some(probe_person));

    for q in &args.queries {
        if Ctx::needs_edge_props(*q) && !ctx.edge_props {
            println!("{TAG} IC{q}: SKIPPED — needs edge properties this store does not have");
            continue;
        }
        let Some(p) = load_params(*q) else { continue };
        let iters = if args.digest_only {
            p.rows.len()
        } else {
            args.iters
        };
        let mut lat = Lat::new(format!("IC{q}"));
        let mut dig = Digest::new(TAG, *q);
        let mut budget = Budget::new(args.budget_s);
        budget.restart();
        let t_query = Instant::now();

        for i in 0..iters {
            let row = &p.rows[i % p.rows.len()];
            let first_touch = i < p.rows.len();

            let t = Instant::now();
            let (rows, capped) = match q {
                1 => (ic1(&db, &p, row), false),
                2 => (ic2(&db, &p, row), false),
                3 => (ic3(&db, &ctx, &p, row), false),
                4 => (ic4(&db, &p, row), false),
                5 => (ic5(&db, &p, row), false),
                6 => (ic6(&db, &ctx, &p, row), false),
                7 => (ic7(&db, &p, row), false),
                8 => (ic8(&db, &p, row), false),
                9 => (ic9(&db, &p, row), false),
                10 => (ic10(&db, &p, row), false),
                11 => (ic11(&db, &ctx, &p, row), false),
                12 => (ic12(&db, &ctx, &p, row), false),
                13 => (ic13(&db, &p, row), false),
                14 => ic14(&db, &p, row),
                _ => {
                    println!("{TAG} FAILED: no such query IC{q}");
                    break;
                }
            };
            lat.push(t.elapsed().as_micros(), first_touch);
            lat.results += rows.len();
            if capped {
                dig.mark_capped();
            }
            if first_touch {
                dig.record(&p.key(row), &rows);
            }
            if budget.spent() {
                lat.budget_limited = true;
                println!(
                    "{TAG} IC{q}: stopped on budget after {}/{iters} iterations. That is an E10 \
                     finding about the baseline at this workload, not a failed run.",
                    i + 1
                );
                break;
            }
        }

        lat.report(&format!("{TAG} LAT"));
        lat.report_split(&format!("{TAG} SPLIT"));
        if Ctx::charges(*q) {
            lat.report_with_resolve(&format!("{TAG} RESOLVE-CHARGED"), ctx.resolve_us);
        }
        dig.report(args.detail.contains(q), t_query.elapsed().as_secs_f64());
    }

    twizzler_indradb::kv_stats::report("complex");
    println!(
        "{TAG} NOTE: entry lookups go through a property index because IndraDB has no built-in \
         (label, name) identity, and *name* lookups do too — which is the axis on which this arm \
         has the mechanism and the native arm does not. Both asymmetries are results, not \
         implementation details. Compare `n` per query before computing any ratio against the \
         native arm: a budget-limited query has fewer samples and its percentiles are not \
         comparable to a full run's."
    );
}
