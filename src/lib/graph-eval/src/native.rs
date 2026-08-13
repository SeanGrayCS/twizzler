//! The LDBC short reads against the native engine (`twizzler-graph`).
//!
//! Written in the DSL wherever the DSL can express the query, and in plain
//! Rust over the engine API where it cannot — every such fallback is marked
//! `DSL GAP` with the missing step named, which is E3-AC4's deliverable and
//! feeds the B workstream.

use twizzler::object::ObjID;
use twizzler_graph::{Graph, GraphError, Labels, Lookup, PropValue, VertexId};

use crate::fixture::*;
use crate::results::*;

type Result<T> = core::result::Result<T, GraphError>;

/// Load [`FIXTURE`] into a freshly reset graph registered at `data/<name>`.
///
/// **Batched through `Graph::bulk_insert`.**
///
/// *This note previously read: "the arena store batches internally — one
/// transaction per arena — so the direct path **is** the batched path. Nothing
/// here needs to change to stay fast." That was true of the arena and false of
/// the index.* `PersistentHashMap::insert` opens a transaction per call and
/// syncs on drop, so the direct path paid one sync per vertex — measured at
/// **5.35 ms against 46.3 µs batched**, a 115× difference that was invisible
/// because it hid inside "vertex creation is slow on this platform".
///
/// The fixture is tiny, so this is not about *this* loader's speed; it is that
/// the loader is the template an LDBC load will be built from, and it should
/// not teach the wrong pattern.
pub fn load(name: &str, f: &Fixture) -> Result<Graph> {
    Graph::reset_arena(name, twizzler_graph::DEFAULT_ARENA_CAP)?;
    let mut g = Graph::open_or_create_arena(name, twizzler_graph::DEFAULT_ARENA_CAP)?;

    // **A8: declare before anything looks up.** The default schema indexes
    // nothing until asked, and this loader resolves every vertex by name to wire
    // its edges — so the declaration has to precede the *loading* lookups below,
    // not just the queries. Declaring after the load would leave `find` hitting
    // `NotIndexed` on the first edge.
    //
    // These are exactly the labels resolved by name: `find` uses all three while
    // loading, and the queries use PERSON and MESSAGE. Nothing else is indexed,
    // which is the point — the index covers roots, not records.
    for l in [PERSON, MESSAGE, FORUM] {
        g.set_label_indexed(l, true)?;
    }

    // One index transaction for every vertex, rather than one per vertex.
    g.bulk_insert(|b| {
        for p in f.people {
            b.add_vertex(PERSON, p.name, ObjID::new(0))?;
        }
        for m in f.messages {
            b.add_vertex(MESSAGE, m.name, ObjID::new(0))?;
        }
        for fo in f.forums {
            b.add_vertex(FORUM, fo.name, ObjID::new(0))?;
        }
        Ok(())
    })?;

    for p in f.people {
        let v = find(&g, PERSON, p.name)?;
        g.set_vertex_prop(v, P_FIRST, PropValue::str(p.first))?;
        g.set_vertex_prop(v, P_LAST, PropValue::str(p.last))?;
        g.set_vertex_prop(v, P_BIRTHDAY, PropValue::U64(p.birthday))?;
        g.set_vertex_prop(v, P_CREATED, PropValue::U64(p.created))?;
    }
    for m in f.messages {
        let v = find(&g, MESSAGE, m.name)?;
        g.set_vertex_prop(v, P_CONTENT, PropValue::str(m.content))?;
        g.set_vertex_prop(v, P_CREATED, PropValue::U64(m.created))?;
    }
    for fo in f.forums {
        let v = find(&g, FORUM, fo.name)?;
        g.set_vertex_prop(v, P_TITLE, PropValue::str(fo.title))?;
    }

    // Edges. `knows` carries a property, so it is created then annotated.
    for m in f.messages {
        let mv = find(&g, MESSAGE, m.name)?;
        let cv = find(&g, PERSON, m.creator)?;
        g.add_edge(mv, HAS_CREATOR, cv)?;
        if let Some(parent) = m.reply_to {
            let pv = find(&g, MESSAGE, parent)?;
            g.add_edge(mv, REPLY_OF, pv)?;
        }
    }
    for fo in f.forums {
        let fv = find(&g, FORUM, fo.name)?;
        let mv = find(&g, PERSON, fo.moderator)?;
        g.add_edge(fv, HAS_MODERATOR, mv)?;
        for msg in fo.contains {
            let target = find(&g, MESSAGE, msg)?;
            g.add_edge(fv, CONTAINER_OF, target)?;
        }
    }
    for (a, b, since) in f.knows {
        let av = find(&g, PERSON, a)?;
        let bv = find(&g, PERSON, b)?;
        let e = g.add_edge(av, KNOWS, bv)?;
        g.set_edge_prop(e, P_SINCE, PropValue::U64(*since))?;
    }
    Ok(g)
}

/// **A8: `NotIndexed` must not become "not found" here.** Under the default
/// schema a label is only indexed if the loader declared it, and collapsing the
/// two would make every query return empty results while looking like a graph
/// that simply had no matches — a benchmark reporting zeros instead of failing.
/// So it panics: a loud crash beats a plausible wrong number.
fn resolve(g: &Graph, label: &str, name: &str) -> Option<VertexId> {
    match g.find_vertex(label, name) {
        Lookup::Found(v) => Some(v),
        Lookup::NotFound => None,
        Lookup::NotIndexed => panic!(
            "label `{label}` is not indexed, so this query cannot resolve its root. \
             Declare it with `set_label_indexed` when loading, or set \
             `UnindexedLookup::Scan` in the schema. Returning \"no results\" here \
             would report a benchmark number that is silently meaningless."
        ),
    }
}

fn find(g: &Graph, label: &str, name: &str) -> Result<VertexId> {
    resolve(g, label, name)
        .ok_or_else(|| GraphError::Twz(twizzler_rt_abi::error::ArgumentError::InvalidArgument.into()))
}

fn name_of(g: &Graph, v: VertexId) -> String {
    g.vertex_info(v).map(|i| i.name).unwrap_or_default()
}

fn str_prop(g: &Graph, v: VertexId, key: &str) -> String {
    match g.get_vertex_prop(v, key) {
        Some(PropValue::Str(s)) => s.as_str().to_string(),
        _ => String::new(),
    }
}

fn u64_prop(g: &Graph, v: VertexId, key: &str) -> u64 {
    match g.get_vertex_prop(v, key) {
        Some(PropValue::U64(n)) => n,
        _ => 0,
    }
}

// --- mutation and scan, for the pre-v5 equivalence surface -----------------
//
// Neither of these is an LDBC read. They exist because A7 (the v5 record
// format) changes exactly this surface and nothing was checking it against the
// baseline: edges become records living in `locs` beside vertices, so `vertices()`
// has to start excluding them, and a tombstoned edge-record begins sharing a
// code path with a tombstoned vertex. Both are cheap to pin *now*, against an
// oracle that does not change, and expensive to reconstruct afterwards.

/// Every person's name, sorted. The sort is the point: neither engine promises
/// scan order, so comparing unsorted would assert something neither guarantees
/// and would break on an unrelated placement change.
pub fn all_people(g: &Graph) -> Vec<String> {
    let mut names: Vec<String> = g
        .vertices_by_label(PERSON)
        .into_iter()
        .filter_map(|v| g.vertex_info(v).map(|i| i.name))
        .collect();
    names.sort();
    names
}

/// Delete a person. Incident edges become unreachable rather than being
/// removed — an edge is alive only while both endpoints are.
pub fn delete_person(g: &mut Graph, name: &str) -> Result<()> {
    let v = find(g, PERSON, name)?;
    g.delete_vertex(v)
}

/// Delete the `knows` edge between two people, in whichever direction it was
/// stored. Returns whether one was found.
pub fn delete_knows(g: &mut Graph, a: &str, b: &str) -> Result<bool> {
    let av = find(g, PERSON, a)?;
    let bv = find(g, PERSON, b)?;
    let target = g
        .vertex_view(av)
        .map(|view| view.both_edges(Labels::these(&[KNOWS])))
        .unwrap_or_default()
        .into_iter()
        .find(|e| {
            g.edge_info(*e)
                .map(|i| i.from == bv || i.to == bv)
                .unwrap_or(false)
        });
    match target {
        Some(e) => {
            g.delete_edge(e)?;
            Ok(true)
        }
        None => Ok(false),
    }
}

/// IS1 — profile of a person.
pub fn is1_profile(g: &Graph, person: &str) -> Option<Profile> {
    let v = resolve(g, PERSON, person)?;
    Some(Profile {
        name: person.to_string(),
        first: str_prop(g, v, P_FIRST),
        last: str_prop(g, v, P_LAST),
        birthday: u64_prop(g, v, P_BIRTHDAY),
        created: u64_prop(g, v, P_CREATED),
    })
}

/// IS2 — a person's most recent messages, newest first, capped at `limit`.
/// Pure DSL: incoming `hasCreator` edges, ordered by the message's date.
pub fn is2_recent_messages(g: &Graph, person: &str, limit: usize) -> Vec<MessageRow> {
    let Some(v) = resolve(g, PERSON, person) else {
        return Vec::new();
    };
    g.traversal()
        .v(v)
        .in_(Labels::these(&[HAS_CREATOR]))
        .order_by_prop_desc(P_CREATED)
        .limit(limit)
        .to_ids()
        .into_iter()
        .map(|m| MessageRow {
            name: name_of(g, m),
            content: str_prop(g, m, P_CONTENT),
            created: u64_prop(g, m, P_CREATED),
        })
        .collect()
}

/// IS3 — a person's friends, ordered by friendship date descending then name.
///
/// **DSL GAP.** LDBC orders by a property of the *edge* (`knows.since`), and
/// carries that property into the result. Our DSL can order vertices by a
/// vertex property (`order_by_prop`) but has no `EdgeTraversal::order_by_prop`,
/// and no step that projects an edge's property alongside its endpoint. So the
/// traversal collects incident edges and the sort happens here in Rust.
/// Missing steps, for the B workstream: `EdgeTraversal::order_by_prop{,_desc}`
/// and an edge→endpoint step that retains the edge's properties.
pub fn is3_friends(g: &Graph, person: &str) -> Vec<FriendRow> {
    let Some(v) = resolve(g, PERSON, person) else {
        return Vec::new();
    };
    // `knows` is stored one way; friendship is symmetric, so read both.
    let edges = g
        .traversal()
        .v(v)
        .both_e(Labels::these(&[KNOWS]))
        .to_ids();

    let mut rows: Vec<FriendRow> = edges
        .into_iter()
        .filter_map(|e| {
            let info = g.edge_info(e)?;
            let other = if info.from == v { info.to } else { info.from };
            let since = match g.get_edge_prop(e, P_SINCE) {
                Some(PropValue::U64(n)) => n,
                _ => 0,
            };
            Some(FriendRow {
                name: name_of(g, other),
                since,
            })
        })
        .collect();
    // LDBC: date descending, then a stable ascending tiebreak.
    rows.sort_by(|a, b| b.since.cmp(&a.since).then(a.name.cmp(&b.name)));
    rows
}

/// IS4 — content and date of a message.
pub fn is4_message(g: &Graph, message: &str) -> Option<MessageRow> {
    let m = resolve(g, MESSAGE, message)?;
    Some(MessageRow {
        name: message.to_string(),
        content: str_prop(g, m, P_CONTENT),
        created: u64_prop(g, m, P_CREATED),
    })
}

/// IS5 — the person who created a message. Pure DSL.
pub fn is5_creator(g: &Graph, message: &str) -> Option<Profile> {
    let m = resolve(g, MESSAGE, message)?;
    let creator = g
        .traversal()
        .v(m)
        .out(Labels::these(&[HAS_CREATOR]))
        .first()?;
    is1_profile(g, &name_of(g, creator))
}

/// IS6 — the forum a message belongs to, and its moderator. Replies are not
/// contained by a forum directly, so this walks the `replyOf` chain to the
/// root post first.
///
/// **DSL GAP closed by B3 (2026-08-11).** This was a hand-rolled loop with its
/// own hop counter, because the DSL was fixed-depth and a reply chain is not.
/// It is now the one step B3 was built for. The counter, the cycle guard and the
/// `parents.first()` bookkeeping are all gone: the visited set is always on, and
/// the depth cap is the engine's.
///
/// **The IndraDB baseline keeps its hand-rolled walk**, deliberately. It cannot
/// use this DSL, so the equivalence suite now checks `repeat_out` against an
/// independent implementation of the same query on every run — a stronger check
/// than freezing both, which is what B3-AC9 originally asked for.
///
/// Truncation is not silently accepted: a chain deeper than the cap would return
/// a wrong forum rather than none, so it is reported as no result.
pub fn is6_forum(g: &Graph, message: &str) -> Option<ForumRow> {
    let start = resolve(g, MESSAGE, message)?;
    let walk = g
        .traversal()
        .v(start)
        .repeat_out(Labels::these(&[REPLY_OF]))
        .until_exhausted();
    if walk.hit_depth_cap() {
        return None;
    }
    // The last non-empty frontier is the root post — or `start` itself when the
    // message is already a root and the walk took no hops.
    let cur = walk.first().unwrap_or(start);
    let forum = g
        .traversal()
        .v(cur)
        .in_(Labels::these(&[CONTAINER_OF]))
        .first()?;
    let moderator = g
        .traversal()
        .v(forum)
        .out(Labels::these(&[HAS_MODERATOR]))
        .first()?;
    Some(ForumRow {
        title: str_prop(g, forum, P_TITLE),
        moderator: name_of(g, moderator),
    })
}

/// IS7 — direct replies to a message, their authors, and whether each author
/// knows the author of the message replied to. Newest first, then author name.
pub fn is7_replies(g: &Graph, message: &str) -> Vec<ReplyRow> {
    let Some(m) = resolve(g, MESSAGE, message) else {
        return Vec::new();
    };
    let parent_author = g
        .traversal()
        .v(m)
        .out(Labels::these(&[HAS_CREATOR]))
        .first();

    let replies = g
        .traversal()
        .v(m)
        .in_(Labels::these(&[REPLY_OF]))
        .order_by_prop_desc(P_CREATED)
        .to_ids();

    let mut rows: Vec<ReplyRow> = replies
        .into_iter()
        .filter_map(|r| {
            let author = g
                .traversal()
                .v(r)
                .out(Labels::these(&[HAS_CREATOR]))
                .first()?;
            let knows = match parent_author {
                // `knows` is symmetric in LDBC and stored one way here.
                Some(pa) => g
                    .traversal()
                    .v(author)
                    .both(Labels::these(&[KNOWS]))
                    .to_ids()
                    .contains(&pa),
                None => false,
            };
            Some(ReplyRow {
                message: name_of(g, r),
                content: str_prop(g, r, P_CONTENT),
                created: u64_prop(g, r, P_CREATED),
                author: name_of(g, author),
                author_knows_parent_author: knows,
            })
        })
        .collect();
    rows.sort_by(|a, b| b.created.cmp(&a.created).then(a.author.cmp(&b.author)));
    rows
}
