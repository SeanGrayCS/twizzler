//! LDBC-SNB Interactive complex reads (IC1–IC14) against the native engine,
//! on the SF0.1 graph `gstress ldbc` left on disk.
//!
//! Companion to `ldbc_query.rs`, which runs the seven short reads. Runs in its
//! own boot, like `ldbc-query`, so the graph is opened cold and the run also
//! exercises durability of the loaded graph.
//!
//! # Where the work is done
//!
//! Candidate sets are gathered through the engine — `both`/`in_`/`out`,
//! `*_neighbors_with_edges`, `get_vertex_prop`. Grouping and ordering are done
//! in this file, not through `order_by_prop`: LDBC's sort keys are compound
//! (`creationDate DESC, toInteger(id) ASC`) and the DSL's tiebreak is the
//! record id, which is insertion order, not the LDBC id. Sorting here is the
//! only way to compute the specified answer. The property is still read once
//! per candidate, on this side of the API boundary rather than the other,
//! identically in both arms.
//!
//! Two costs sit inside the reported latencies:
//!
//! 1. `vertex_info` allocates. It is the only public way to test a neighbour's
//!    label, and it returns owned `String`s; the posts-only queries (IC4, IC6,
//!    IC10, IC12) pay it per candidate message. `VertexView`'s `*_where`
//!    variants expose an allocation-free `VertexHandle::label_id`, but the
//!    label ids are internal, so using them would need a calibration step this
//!    harness does not have. The overhead is bounded by the candidate set.
//! 2. Row rendering is inside the timer. Bounded by `LIMIT` — at most 20
//!    rows — so it is a constant, not a term that grows with the candidate set.
//!
//! # Name parameters
//!
//! IC3, IC6, IC11 and IC12 are parameterised by `tag`, `place` and `tagclass`
//! names, which are properties here, not identities, so the engine's only
//! means of resolving one is a scan. (The baseline is the mirror image:
//! IndraDB's UUID identity makes every id lookup a property-index query, our
//! `(label, name)` identity makes every name lookup a scan.)
//!
//! Resolution happens once, before the timed loop, and its cost is reported on
//! its own line, together with what each affected query's median would be if
//! it were charged per query instead. Neither figure is reported without the
//! other.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::time::Instant;

use twizzler_graph::{Graph, Labels, Lookup, PropValue, VertexId, DEFAULT_ARENA_CAP};

use crate::ldbc_common::{load_params, parse_args, san, self_check, Budget, Digest, Lat, Params};

const NAME: &str = "ldbc";
const TAG: &str = "GSTRESS LDBCC";

const KNOWS: Labels<'static> = Labels::These(&["knows"]);
const HAS_CREATOR: Labels<'static> = Labels::These(&["hasCreator"]);
const REPLY_OF: Labels<'static> = Labels::These(&["replyOf"]);
const HAS_TAG: Labels<'static> = Labels::These(&["hasTag"]);
const HAS_TYPE: Labels<'static> = Labels::These(&["hasType"]);
const IS_SUBCLASS_OF: Labels<'static> = Labels::These(&["isSubclassOf"]);
const IS_LOCATED_IN: Labels<'static> = Labels::These(&["isLocatedIn"]);
const IS_PART_OF: Labels<'static> = Labels::These(&["isPartOf"]);
const HAS_INTEREST: Labels<'static> = Labels::These(&["hasInterest"]);
const HAS_MEMBER: Labels<'static> = Labels::These(&["hasMember"]);
const CONTAINER_OF: Labels<'static> = Labels::These(&["containerOf"]);
const LIKES: Labels<'static> = Labels::These(&["likes"]);
const WORK_AT: Labels<'static> = Labels::These(&["workAt"]);
const STUDY_AT: Labels<'static> = Labels::These(&["studyAt"]);

/// Depth cap for the unbounded `knows` closures (IC13, IC14). The visited set
/// already guarantees termination; this is the second line, and matches the
/// DSL's own `DEFAULT_MAX_DEPTH`.
const MAX_HOPS: usize = 64;

/// IC14 path-enumeration bound. Reported when hit — a silent truncation would
/// read as "few trusted paths exist between these two people".
const IC14_PATH_CAP: usize = 20_000;

// ---------------------------------------------------------------------------
// Property and identity helpers
// ---------------------------------------------------------------------------

/// A vertex property as text, across the three storage tiers.
///
/// Same shape as `ldbc_query::str_prop`. The blob fallback matters: SF0.1 has
/// `content` values over 255 bytes, which live in the blob tier and read back
/// as `""` without it.
fn sprop(g: &Graph, v: VertexId, key: &str) -> String {
    match g.get_vertex_prop(v, key) {
        Some(PropValue::Str(s)) => s.as_str().to_string(),
        _ => match g.get_vertex_text(v, key) {
            Some(t) => t,
            None => g
                .get_vertex_blob(v, key)
                .map(|b| String::from_utf8_lossy(&b).into_owned())
                .unwrap_or_default(),
        },
    }
}

/// A numeric property, without allocating.
///
/// Every LDBC date in this dataset is epoch milliseconds as a decimal string
/// (`CsvBasic-LongDateFormatter`), fixed-width at SF0.1, so string order
/// happens to equal numeric order. This parses anyway: relying on a property
/// of one scale factor is how a harness silently produces a different answer
/// at SF1.
fn nprop(g: &Graph, v: VertexId, key: &str) -> Option<i64> {
    match g.get_vertex_prop(v, key) {
        Some(PropValue::Str(s)) => s.as_str().parse().ok(),
        Some(PropValue::I64(i)) => Some(i),
        Some(PropValue::U64(u)) => Some(u as i64),
        _ => None,
    }
}

/// An edge property as a number. The edge path has only the inline tier
/// (`ldbc_load.rs` counts anything longer as dropped), so there is no fallback
/// to make.
fn eprop(g: &Graph, e: VertexId, key: &str) -> Option<i64> {
    match g.get_edge_prop(e, key) {
        Some(PropValue::Str(s)) => s.as_str().parse().ok(),
        Some(PropValue::I64(i)) => Some(i),
        Some(PropValue::U64(u)) => Some(u as i64),
        _ => None,
    }
}

/// A vertex's LDBC id — which is its *name* here, since the loader makes an
/// LDBC id the vertex name so `find_vertex(label, id)` resolves an endpoint.
fn lid(g: &Graph, v: VertexId) -> i64 {
    g.vertex_info(v)
        .and_then(|i| i.name.parse().ok())
        .unwrap_or(-1)
}

fn has_label(g: &Graph, v: VertexId, label: &str) -> bool {
    g.vertex_info(v).map_or(false, |i| i.label == label)
}

fn first_out(g: &Graph, v: VertexId, labels: Labels) -> Option<VertexId> {
    g.out_neighbors(v, labels).into_iter().next()
}

/// The country a place is in, or the place itself if it is one.
fn country_of(g: &Graph, place: VertexId) -> Option<VertexId> {
    let mut cur = place;
    for _ in 0..8 {
        if sprop(g, cur, "type") == "country" {
            return Some(cur);
        }
        cur = first_out(g, cur, IS_PART_OF)?;
    }
    None
}

/// Breadth-first `knows` neighbourhood: `(person, distance)`, excluding the
/// root. `knows` is stored one-directionally by the loader (LDBC's CSV holds
/// one row per pair) and is undirected in the benchmark, so this walks `both`.
fn hops(g: &Graph, root: VertexId, max_depth: usize) -> Vec<(VertexId, usize)> {
    let mut seen: HashSet<u64> = HashSet::new();
    seen.insert(root.0);
    let mut frontier = vec![root];
    let mut out = Vec::new();
    for d in 1..=max_depth {
        let mut next = Vec::new();
        for v in &frontier {
            for w in g.both_neighbors(*v, KNOWS) {
                if seen.insert(w.0) {
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

/// Civil date from epoch milliseconds, UTC — `(month, day)`.
///
/// Howard Hinnant's `civil_from_days`. Written out rather than pulled in
/// because the guest has no date crate and IC10's window is the only place a
/// calendar is needed.
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

// ---------------------------------------------------------------------------
// Name resolution
// ---------------------------------------------------------------------------

struct Ctx {
    country: HashMap<String, VertexId>,
    tag: HashMap<String, VertexId>,
    class: HashMap<String, VertexId>,
    /// Wall-clock cost of building the maps, for the addendum showing what a
    /// query's median would be if resolution were charged per query.
    resolve_us: u128,
}

impl Ctx {
    /// One scan per label, reading each vertex's `name` property.
    ///
    /// Only the labels the selected queries actually need — running `q:1,2`
    /// should not pay for a tag scan.
    fn build(g: &Graph, queries: &[usize]) -> Ctx {
        let need_place = queries.iter().any(|q| matches!(q, 3 | 11));
        let need_tag = queries.contains(&6);
        let need_class = queries.contains(&12);

        let t = Instant::now();
        let mut country = HashMap::new();
        if need_place {
            for v in g.vertices_by_label("place") {
                if sprop(g, v, "type") == "country" {
                    country.insert(sprop(g, v, "name"), v);
                }
            }
        }
        let mut tag = HashMap::new();
        if need_tag {
            for v in g.vertices_by_label("tag") {
                tag.insert(sprop(g, v, "name"), v);
            }
        }
        let mut class = HashMap::new();
        if need_class {
            for v in g.vertices_by_label("tagclass") {
                class.insert(sprop(g, v, "name"), v);
            }
        }
        let resolve_us = t.elapsed().as_micros();
        println!(
            "{TAG} RESOLVE built name maps in {:.2}s ({} countries, {} tags, {} tag classes) \
             — EXCLUDED from the latencies below. LDBC parameterises IC3/IC6/IC11/IC12 by \
             human-readable names, which are properties here rather than identities, so the \
             engine's only means is a scan. See E10-AC6.",
            resolve_us as f64 / 1e6,
            country.len(),
            tag.len(),
            class.len(),
        );
        Ctx {
            country,
            tag,
            class,
            resolve_us,
        }
    }

    /// Whether a query pays the resolution cost, for the per-query addendum.
    fn charges(q: usize) -> bool {
        matches!(q, 3 | 6 | 11 | 12)
    }
}

// ---------------------------------------------------------------------------
// The fourteen queries. Each returns already-rendered digest rows.
// ---------------------------------------------------------------------------

fn person(g: &Graph, id: &str) -> Option<VertexId> {
    match g.find_vertex("person", id) {
        Lookup::Found(v) => Some(v),
        // `NotIndexed` is not `NotFound`. Collapsing them would report an empty
        // result for every parameter while looking healthy.
        Lookup::NotIndexed => {
            println!("{TAG} FAILED: label `person` is not indexed — the run would be silently empty");
            None
        }
        Lookup::NotFound => None,
    }
}

/// IC1. Transitive friends with a certain name.
///
/// The plan differs from the reference Cypher deliberately. Cypher anchors on
/// `(friend:Person {firstName})` because Neo4j has a secondary index; this
/// walks `knows*1..3` from the person and filters on `firstName`, which is the
/// same answer and is what a graph engine without a secondary index should do.
/// A plan choice, not a restriction.
fn ic1(g: &Graph, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = person(g, p.get(row, "personId")) else {
        return Vec::new();
    };
    let first = p.get(row, "firstName");
    let mut cands: Vec<(usize, String, i64, VertexId)> = hops(g, root, 3)
        .into_iter()
        .filter(|(v, _)| sprop(g, *v, "firstName") == first)
        .map(|(v, d)| (d, sprop(g, v, "lastName"), lid(g, v), v))
        .collect();
    cands.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    cands.truncate(20);

    let mut out = Vec::new();
    for (d, last, fid, f) in cands {
        let city = first_out(g, f, IS_LOCATED_IN)
            .map(|c| sprop(g, c, "name"))
            .unwrap_or_default();
        let mut unis: Vec<String> = g
            .out_neighbors_with_edges(f, STUDY_AT)
            .into_iter()
            .filter_map(|(e, o)| {
                let city = first_out(g, o, IS_LOCATED_IN)?;
                Some(format!(
                    "{}@{}@{}",
                    san(&sprop(g, o, "name")),
                    eprop(g, e, "classYear")?,
                    san(&sprop(g, city, "name"))
                ))
            })
            .collect();
        unis.sort();
        let mut cos: Vec<String> = g
            .out_neighbors_with_edges(f, WORK_AT)
            .into_iter()
            .filter_map(|(e, o)| {
                let place = first_out(g, o, IS_LOCATED_IN)?;
                let country = country_of(g, place)?;
                Some(format!(
                    "{}@{}@{}",
                    san(&sprop(g, o, "name")),
                    eprop(g, e, "workFrom")?,
                    san(&sprop(g, country, "name"))
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

/// IC2. Recent messages by friends.
fn ic2(g: &Graph, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = person(g, p.get(row, "personId")) else {
        return Vec::new();
    };
    let max_date = p.num(row, "maxDate");
    let mut rows: Vec<(i64, i64, i64, VertexId)> = Vec::new();
    for f in g.traversal().v(root).both(KNOWS).to_ids() {
        let fid = lid(g, f);
        for m in g.in_neighbors(f, HAS_CREATOR) {
            if let Some(cd) = nprop(g, m, "creationDate") {
                if cd <= max_date {
                    rows.push((cd, lid(g, m), fid, m));
                }
            }
        }
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(20);
    rows.into_iter()
        .map(|(cd, mid, fid, m)| format!("{fid},{mid},{cd},{}", content_len(g, m)))
        .collect()
}

/// `coalesce(content, imageFile)` measured in bytes.
///
/// The length, not the text: the read path differs by value size across the
/// three property tiers, and this digest should test which message was
/// returned, not re-test the content read that IS4 already covers.
fn content_len(g: &Graph, m: VertexId) -> usize {
    let c = sprop(g, m, "content");
    if c.is_empty() {
        sprop(g, m, "imageFile").len()
    } else {
        c.len()
    }
}

/// IC3. Friends and friends of friends who posted in two countries.
fn ic3(g: &Graph, ctx: &Ctx, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = person(g, p.get(row, "personId")) else {
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

    let mut rows: Vec<(usize, i64, usize, usize)> = Vec::new();
    for (f, _) in hops(g, root, 2) {
        let Some(city) = first_out(g, f, IS_LOCATED_IN) else {
            continue;
        };
        match country_of(g, city) {
            Some(c) if c.0 == cx.0 || c.0 == cy.0 => continue,
            None => continue,
            _ => {}
        }
        let (mut x, mut y) = (0usize, 0usize);
        for m in g.in_neighbors(f, HAS_CREATOR) {
            let Some(cd) = nprop(g, m, "creationDate") else {
                continue;
            };
            if cd < start || cd >= end {
                continue;
            }
            let Some(c) = first_out(g, m, IS_LOCATED_IN).and_then(|pl| country_of(g, pl)) else {
                continue;
            };
            if c.0 == cx.0 {
                x += 1;
            } else if c.0 == cy.0 {
                y += 1;
            }
        }
        if x > 0 && y > 0 {
            rows.push((x + y, lid(g, f), x, y));
        }
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(20);
    rows.into_iter()
        .map(|(_, fid, x, y)| format!("{fid},{x},{y}"))
        .collect()
}

/// IC4. New topics.
fn ic4(g: &Graph, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = person(g, p.get(row, "personId")) else {
        return Vec::new();
    };
    let start = p.num(row, "startDate");
    let end = start + p.num(row, "durationDays") * 86_400_000;

    let mut valid: HashMap<u64, usize> = HashMap::new();
    let mut invalid: HashMap<u64, usize> = HashMap::new();
    let mut seen: HashSet<(u64, u64)> = HashSet::new();
    for f in g.traversal().v(root).both(KNOWS).to_ids() {
        for m in g.in_neighbors(f, HAS_CREATOR) {
            if !has_label(g, m, "post") {
                continue;
            }
            let Some(cd) = nprop(g, m, "creationDate") else {
                continue;
            };
            for t in g.out_neighbors(m, HAS_TAG) {
                // `WITH DISTINCT tag, post` — the spec's dedup, kept even
                // though a post has exactly one creator at SF0.1.
                if !seen.insert((t.0, m.0)) {
                    continue;
                }
                if cd >= start && cd < end {
                    *valid.entry(t.0).or_insert(0) += 1;
                } else if cd < start {
                    *invalid.entry(t.0).or_insert(0) += 1;
                }
            }
        }
    }
    let mut rows: Vec<(usize, String)> = valid
        .iter()
        .filter(|(t, c)| **c > 0 && invalid.get(*t).copied().unwrap_or(0) == 0)
        .map(|(t, c)| (*c, san(&sprop(g, VertexId(*t), "name"))))
        .collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(10);
    rows.into_iter().map(|(c, n)| format!("{n},{c}")).collect()
}

/// IC5. New groups.
///
/// `friends` is scoped per forum: `WITH forum, collect(friend) AS friends`
/// groups by forum, so the post count is over the friends who joined that
/// forum after `minDate` — not over the whole friend-of-friend set. Getting
/// this wrong inflates every count while still looking plausible.
fn ic5(g: &Graph, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = person(g, p.get(row, "personId")) else {
        return Vec::new();
    };
    let min_date = p.num(row, "minDate");

    let mut joined: HashMap<u64, HashSet<u64>> = HashMap::new();
    let mut forum_of: HashMap<u64, VertexId> = HashMap::new();
    for (f, _) in hops(g, root, 2) {
        for (e, forum) in g.in_neighbors_with_edges(f, HAS_MEMBER) {
            if eprop(g, e, "joinDate").map_or(false, |d| d > min_date) {
                joined.entry(forum.0).or_default().insert(f.0);
                forum_of.insert(forum.0, forum);
            }
        }
    }

    let mut rows: Vec<(usize, i64, String)> = Vec::new();
    for (fid, friends) in &joined {
        let forum = forum_of[fid];
        let n = g
            .out_neighbors(forum, CONTAINER_OF)
            .into_iter()
            .filter(|post| {
                first_out(g, *post, HAS_CREATOR).map_or(false, |c| friends.contains(&c.0))
            })
            .count();
        rows.push((n, lid(g, forum), san(&sprop(g, forum, "title"))));
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(20);
    rows.into_iter()
        .map(|(n, id, title)| format!("{id},{title},{n}"))
        .collect()
}

/// IC6. Tag co-occurrence.
fn ic6(g: &Graph, ctx: &Ctx, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = person(g, p.get(row, "personId")) else {
        return Vec::new();
    };
    let Some(known) = ctx.tag.get(p.get(row, "tagName")).copied() else {
        return Vec::new();
    };

    let mut counts: HashMap<u64, usize> = HashMap::new();
    for (f, _) in hops(g, root, 2) {
        for m in g.in_neighbors(f, HAS_CREATOR) {
            if !has_label(g, m, "post") {
                continue;
            }
            let tags = g.out_neighbors(m, HAS_TAG);
            if !tags.iter().any(|t| t.0 == known.0) {
                continue;
            }
            for t in &tags {
                if t.0 != known.0 {
                    *counts.entry(t.0).or_insert(0) += 1;
                }
            }
        }
    }
    let mut rows: Vec<(usize, String)> = counts
        .into_iter()
        .map(|(t, c)| (c, san(&sprop(g, VertexId(t), "name"))))
        .collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(10);
    rows.into_iter().map(|(c, n)| format!("{n},{c}")).collect()
}

/// IC7. Recent likers — one row per liker, carrying their latest like.
fn ic7(g: &Graph, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = person(g, p.get(row, "personId")) else {
        return Vec::new();
    };
    // liker -> (likeTime, message LDBC id, message)
    let mut best: HashMap<u64, (i64, i64, VertexId)> = HashMap::new();
    for m in g.in_neighbors(root, HAS_CREATOR) {
        let mid = lid(g, m);
        for (e, liker) in g.in_neighbors_with_edges(m, LIKES) {
            let Some(when) = eprop(g, e, "creationDate") else {
                continue;
            };
            // `ORDER BY likeTime DESC, message.id ASC` then `head(collect(..))`:
            // latest like, ties broken by the smaller message id.
            let better = match best.get(&liker.0) {
                None => true,
                Some((w, i, _)) => when > *w || (when == *w && mid < *i),
            };
            if better {
                best.insert(liker.0, (when, mid, m));
            }
        }
    }
    let friends: HashSet<u64> = g
        .both_neighbors(root, KNOWS)
        .into_iter()
        .map(|v| v.0)
        .collect();

    let mut rows: Vec<(i64, i64, i64, i64, u8)> = best
        .into_iter()
        .filter_map(|(liker, (when, mid, m))| {
            let cd = nprop(g, m, "creationDate")?;
            let latency = ((when - cd) / 1000) / 60;
            Some((
                when,
                lid(g, VertexId(liker)),
                mid,
                latency,
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

/// IC8. Recent replies.
///
/// No `DISTINCT` in the spec, and none here: a comment replies to exactly one
/// message, so it cannot double-count. Adding a `dedup()` "for safety" would
/// be a silent deviation.
fn ic8(g: &Graph, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = person(g, p.get(row, "personId")) else {
        return Vec::new();
    };
    let mut rows: Vec<(i64, i64, i64)> = Vec::new();
    for m in g.in_neighbors(root, HAS_CREATOR) {
        for c in g.in_neighbors(m, REPLY_OF) {
            let Some(cd) = nprop(g, c, "creationDate") else {
                continue;
            };
            let Some(who) = first_out(g, c, HAS_CREATOR) else {
                continue;
            };
            rows.push((cd, lid(g, c), lid(g, who)));
        }
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(20);
    rows.into_iter()
        .map(|(cd, c, who)| format!("{who},{c},{cd}"))
        .collect()
}

/// IC9. Recent messages by friends and friends of friends.
///
/// The candidate set is every message of the two-hop neighbourhood, ordered to
/// return twenty — the largest candidate set of the fourteen.
fn ic9(g: &Graph, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = person(g, p.get(row, "personId")) else {
        return Vec::new();
    };
    let max_date = p.num(row, "maxDate");
    let mut rows: Vec<(i64, i64, i64)> = Vec::new();
    for (f, _) in hops(g, root, 2) {
        let fid = lid(g, f);
        for m in g.in_neighbors(f, HAS_CREATOR) {
            if let Some(cd) = nprop(g, m, "creationDate") {
                if cd < max_date {
                    rows.push((cd, lid(g, m), fid));
                }
            }
        }
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(20);
    rows.into_iter()
        .map(|(cd, m, f)| format!("{f},{m},{cd}"))
        .collect()
}

/// IC10. Friend recommendation.
///
/// Two details the Cypher's `WITH` chain leaves ambiguous, settled by LDBC's
/// reference SQL: the score is `common − (total − common)`, and `postCount`
/// counts posts only.
fn ic10(g: &Graph, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = person(g, p.get(row, "personId")) else {
        return Vec::new();
    };
    let month = p.num(row, "month") as u32;
    let next = month % 12 + 1;

    let direct: HashSet<u64> = g
        .both_neighbors(root, KNOWS)
        .into_iter()
        .map(|v| v.0)
        .collect();
    let mut cands: HashMap<u64, VertexId> = HashMap::new();
    for f in &direct {
        for w in g.both_neighbors(VertexId(*f), KNOWS) {
            if w.0 != root.0 && !direct.contains(&w.0) {
                cands.insert(w.0, w);
            }
        }
    }
    let my_tags: HashSet<u64> = g
        .out_neighbors(root, HAS_INTEREST)
        .into_iter()
        .map(|v| v.0)
        .collect();

    let mut rows: Vec<(i64, i64, String)> = Vec::new();
    for f in cands.values() {
        let Some(bd) = nprop(g, *f, "birthday") else {
            continue;
        };
        let (m, d) = month_day(bd);
        if !((m == month && d >= 21) || (m == next && d < 22)) {
            continue;
        }
        let (mut total, mut common) = (0i64, 0i64);
        for post in g.in_neighbors(*f, HAS_CREATOR) {
            if !has_label(g, post, "post") {
                continue;
            }
            total += 1;
            if g
                .out_neighbors(post, HAS_TAG)
                .iter()
                .any(|t| my_tags.contains(&t.0))
            {
                common += 1;
            }
        }
        let city = first_out(g, *f, IS_LOCATED_IN)
            .map(|c| sprop(g, c, "name"))
            .unwrap_or_default();
        rows.push((common - (total - common), lid(g, *f), san(&city)));
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(10);
    rows.into_iter()
        .map(|(s, f, city)| format!("{f},{s},{city}"))
        .collect()
}

/// IC11. Job referral.
///
/// `ORDER BY workFrom ASC, personId ASC, organisationName DESC` — the third key
/// descends while the first two ascend. Easy to mirror wrongly, so it is
/// spelled out rather than folded into a tuple sort.
fn ic11(g: &Graph, ctx: &Ctx, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = person(g, p.get(row, "personId")) else {
        return Vec::new();
    };
    let Some(cid) = ctx.country.get(p.get(row, "countryName")).copied() else {
        return Vec::new();
    };
    let year = p.num(row, "workFromYear");

    let mut rows: Vec<(i64, i64, String)> = Vec::new();
    for (f, _) in hops(g, root, 2) {
        let fid = lid(g, f);
        for (e, org) in g.out_neighbors_with_edges(f, WORK_AT) {
            let Some(wf) = eprop(g, e, "workFrom") else {
                continue;
            };
            if wf >= year || sprop(g, org, "type") != "company" {
                continue;
            }
            let ok = first_out(g, org, IS_LOCATED_IN)
                .and_then(|pl| country_of(g, pl))
                .map_or(false, |c| c.0 == cid.0);
            if ok {
                rows.push((wf, fid, san(&sprop(g, org, "name"))));
            }
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(b.2.cmp(&a.2)));
    rows.truncate(10);
    rows.into_iter()
        .map(|(wf, f, name)| format!("{f},{name},{wf}"))
        .collect()
}

/// IC12. Expert search.
fn ic12(g: &Graph, ctx: &Ctx, p: &Params, row: &[String]) -> Vec<String> {
    let Some(root) = person(g, p.get(row, "personId")) else {
        return Vec::new();
    };
    let Some(class) = ctx.class.get(p.get(row, "tagClassName")).copied() else {
        return Vec::new();
    };

    // The transitive subclass closure, walked downward from the named class.
    // The Cypher walks `IS_SUBCLASS_OF*0..` upward from each tag; this is the
    // same relation read the other way.
    let mut closure: HashSet<u64> = HashSet::new();
    closure.insert(class.0);
    let mut stack = vec![class];
    while let Some(c) = stack.pop() {
        for sub in g.in_neighbors(c, IS_SUBCLASS_OF) {
            if closure.insert(sub.0) {
                stack.push(sub);
            }
        }
    }
    let mut tags: HashSet<u64> = HashSet::new();
    for c in &closure {
        for t in g.in_neighbors(VertexId(*c), HAS_TYPE) {
            tags.insert(t.0);
        }
    }
    if tags.is_empty() {
        return Vec::new();
    }

    let mut rows: Vec<(usize, i64, String)> = Vec::new();
    for f in g.traversal().v(root).both(KNOWS).to_ids() {
        let mut names: BTreeSet<String> = BTreeSet::new();
        let mut comments: HashSet<u64> = HashSet::new();
        for c in g.in_neighbors(f, HAS_CREATOR) {
            if has_label(g, c, "post") {
                continue;
            }
            let Some(parent) = first_out(g, c, REPLY_OF) else {
                continue;
            };
            // The reply must land on a post, not on another comment — both the
            // reference Cypher and SQL require the parent to be a post.
            if !has_label(g, parent, "post") {
                continue;
            }
            let mut hit = false;
            for t in g.out_neighbors(parent, HAS_TAG) {
                if tags.contains(&t.0) {
                    hit = true;
                    names.insert(san(&sprop(g, t, "name")));
                }
            }
            if hit {
                comments.insert(c.0);
            }
        }
        if !comments.is_empty() {
            rows.push((
                comments.len(),
                lid(g, f),
                names.into_iter().collect::<Vec<_>>().join("|"),
            ));
        }
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(20);
    rows.into_iter()
        .map(|(n, f, names)| format!("{f},{n},{names}"))
        .collect()
}

/// IC13. Single shortest path over `knows`, `-1` if unreachable.
///
/// `repeat_both(knows).until(..)` would find the frontier, but the length is
/// not a DSL output, so the walk is hand-rolled. The baseline does the same, so
/// the two arms remain symmetric.
fn ic13(g: &Graph, p: &Params, row: &[String]) -> Vec<String> {
    let (Some(a), Some(b)) = (
        person(g, p.get(row, "person1Id")),
        person(g, p.get(row, "person2Id")),
    ) else {
        return Vec::new();
    };
    if a.0 == b.0 {
        return vec!["0".to_string()];
    }
    let mut seen: HashSet<u64> = HashSet::new();
    seen.insert(a.0);
    let mut frontier = vec![a];
    for d in 1..=MAX_HOPS {
        let mut next = Vec::new();
        for v in &frontier {
            for w in g.both_neighbors(*v, KNOWS) {
                if w.0 == b.0 {
                    return vec![d.to_string()];
                }
                if seen.insert(w.0) {
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

/// IC14. Trusted connection paths — all shortest paths, weighted.
///
/// Not expressible in the DSL: `Repeat`'s visited set is keyed by vertex, so
/// alternative routes of equal length collapse to one. Both arms hand-roll the
/// standard BFS-with-predecessor-lists construction instead.
///
/// Returns `(rows, capped)`.
fn ic14(g: &Graph, p: &Params, row: &[String]) -> (Vec<String>, bool) {
    let (Some(a), Some(b)) = (
        person(g, p.get(row, "person1Id")),
        person(g, p.get(row, "person2Id")),
    ) else {
        return (Vec::new(), false);
    };
    if a.0 == b.0 {
        return (vec![format!("{},0.0", lid(g, a))], false);
    }

    let mut dist: HashMap<u64, usize> = HashMap::new();
    let mut preds: HashMap<u64, Vec<VertexId>> = HashMap::new();
    dist.insert(a.0, 0);
    let mut frontier = vec![a];
    let mut found = false;
    let mut depth = 0usize;
    while !frontier.is_empty() && !found && depth < MAX_HOPS {
        depth += 1;
        let mut next = Vec::new();
        for v in &frontier {
            for w in g.both_neighbors(*v, KNOWS) {
                match dist.get(&w.0) {
                    None => {
                        dist.insert(w.0, depth);
                        preds.entry(w.0).or_default().push(*v);
                        next.push(w);
                    }
                    Some(dw) if *dw == depth => {
                        preds.entry(w.0).or_default().push(*v);
                    }
                    _ => {}
                }
                if w.0 == b.0 {
                    found = true;
                }
            }
        }
        frontier = next;
    }
    if !dist.contains_key(&b.0) {
        return (Vec::new(), false);
    }

    // Enumerate every shortest path, back to front.
    let mut paths: Vec<Vec<VertexId>> = Vec::new();
    let mut capped = false;
    let mut stack: Vec<(VertexId, Vec<VertexId>)> = vec![(b, vec![b])];
    while let Some((cur, acc)) = stack.pop() {
        if paths.len() >= IC14_PATH_CAP {
            capped = true;
            break;
        }
        if cur.0 == a.0 {
            let mut path = acc.clone();
            path.reverse();
            paths.push(path);
            continue;
        }
        for q in preds.get(&cur.0).map(Vec::as_slice).unwrap_or(&[]) {
            let mut nxt = acc.clone();
            nxt.push(*q);
            stack.push((*q, nxt));
        }
    }

    let mut cache: HashMap<(u64, u64), f64> = HashMap::new();
    let mut rows: Vec<(f64, String)> = Vec::new();
    for path in &paths {
        let mut w = 0.0f64;
        for pair in path.windows(2) {
            let (x, y) = (pair[0], pair[1]);
            let key = if x.0 < y.0 { (x.0, y.0) } else { (y.0, x.0) };
            let ew = match cache.get(&key) {
                Some(v) => *v,
                None => {
                    let v = interaction_weight(g, x, y);
                    cache.insert(key, v);
                    v
                }
            };
            w += ew;
        }
        let ids: Vec<String> = path.iter().map(|v| lid(g, *v).to_string()).collect();
        rows.push((w, ids.join("|")));
    }
    // Weight descending; LDBC leaves ties unspecified, so the path string
    // breaks them and all three implementations produce the same bytes.
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

/// IC14's edge weight: comment→post replies score 1.0 and comment→comment
/// replies 0.5, counted in both directions between the pair.
fn interaction_weight(g: &Graph, x: VertexId, y: VertexId) -> f64 {
    let mut w = 0.0f64;
    for (src, dst) in [(x, y), (y, x)] {
        for c in g.in_neighbors(src, HAS_CREATOR) {
            if has_label(g, c, "post") {
                continue;
            }
            let Some(parent) = first_out(g, c, REPLY_OF) else {
                continue;
            };
            let Some(creator) = first_out(g, parent, HAS_CREATOR) else {
                continue;
            };
            if creator.0 != dst.0 {
                continue;
            }
            w += if has_label(g, parent, "post") { 1.0 } else { 0.5 };
        }
    }
    w
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

pub(crate) fn run() {
    let args = parse_args(2);
    println!(
        "GSTRESS STAMP harness={} mode=ldbc-complex iters={} queries={:?} budget={}",
        crate::HARNESS_REV,
        if args.digest_only { 0 } else { args.iters },
        args.queries,
        Budget::new(args.budget_s).describe(),
    );

    self_check(TAG);

    let t = Instant::now();
    let g = match Graph::open_or_create_arena(NAME, DEFAULT_ARENA_CAP) {
        Ok(g) => g,
        Err(e) => {
            println!("{TAG} FAILED: cannot open `{NAME}`: {e:?} — run `gstress ldbc` first");
            return;
        }
    };
    println!(
        "{TAG} open: {:.2}s, {} arenas",
        t.elapsed().as_secs_f64(),
        g.arena_count()
    );

    // Force the lazy index build before timing anything: without a warm-up the
    // first query of the run measures the rebuild, not the query.
    let t = Instant::now();
    let _ = g.find_vertex("person", "warmup-nonexistent");
    println!(
        "{TAG} index build: {:.2}s (excluded from query latencies)",
        t.elapsed().as_secs_f64()
    );

    let ctx = Ctx::build(&g, &args.queries);

    for q in &args.queries {
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
                1 => (ic1(&g, &p, row), false),
                2 => (ic2(&g, &p, row), false),
                3 => (ic3(&g, &ctx, &p, row), false),
                4 => (ic4(&g, &p, row), false),
                5 => (ic5(&g, &p, row), false),
                6 => (ic6(&g, &ctx, &p, row), false),
                7 => (ic7(&g, &p, row), false),
                8 => (ic8(&g, &p, row), false),
                9 => (ic9(&g, &p, row), false),
                10 => (ic10(&g, &p, row), false),
                11 => (ic11(&g, &ctx, &p, row), false),
                12 => (ic12(&g, &ctx, &p, row), false),
                13 => (ic13(&g, &p, row), false),
                14 => ic14(&g, &p, row),
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
            // The digest comes from the timed code path, so an equivalence
            // check cannot drift from what was measured.
            if first_touch {
                dig.record(&p.key(row), &rows);
            }
            if budget.spent() {
                lat.budget_limited = true;
                println!(
                    "{TAG} IC{q}: stopped on budget after {}/{iters} iterations",
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

    println!(
        "{TAG} NOTE: LDBC's own substitution parameters, one file per query, 15 rows each at \
         SF0.1 — a far smaller pool than the official driver's, so a p99 over many iterations \
         is mostly repeat-visit behaviour (the cold/warm split above separates them). Not an \
         audited LDBC result: the driver also controls issue rate, query mix and dependency \
         time. `Message` spans `comment`+`post` because the engine has no type hierarchy. \
         Grouping and ordering are done in the harness, identically in both arms — see this \
         module's header for why, and `docs/handoffs/E10-COMPLEX-READS.md` for the rest."
    );
}
