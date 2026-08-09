//! The same LDBC short reads against the IndraDB baseline
//! (`Database<TwizzlerDatastore>`).
//!
//! Deliberately written against IndraDB's *public* API only — no reach-through
//! to `TwizzlerDatastore` internals — so this is a fair reading of what the
//! baseline engine offers a query author.
//!
//! **Modelling difference worth noting for the report.** Our engine gives a
//! vertex a built-in `(label, name)` identity; IndraDB gives it a UUID and a
//! type, so the fixture's stable name has to become an ordinary property here.
//! Entry-point lookups therefore go through a property index
//! (`index_property("name")`), where the native engine uses its built-in
//! `find_vertex`. That is a real design asymmetry, not an implementation
//! detail: the baseline pays an index lookup for something the native engine
//! gets from its identity model.
//!
//! Type filtering on edges is done in Rust after materialising, rather than
//! through query-builder combinators, to keep this code on the API surface
//! already exercised by the datastore's own tests.

use indradb::{
    Database, Edge, Identifier, Json, QueryExt, QueryOutputValue, SpecificEdgeQuery,
    SpecificVertexQuery, VertexWithPropertyValueQuery,
};
use twizzler_indradb::TwizzlerDatastore;
use uuid::Uuid;

use crate::fixture::*;
use crate::results::*;

/// The fixture's stable key, which IndraDB must carry as a property.
pub const P_NAME: &str = "name";

type Db = Database<TwizzlerDatastore>;

fn ident(s: &str) -> Identifier {
    Identifier::new(s).expect("valid identifier")
}

/// Load [`FIXTURE`] into a datastore registered at `data/<name>`.
pub fn load(name: &str, f: &Fixture) -> Db {
    let db = TwizzlerDatastore::open_db(name).expect("open datastore");
    // Name lookups are property queries here, so the index must be declared.
    db.index_property(ident(P_NAME)).expect("index name");

    let mk = |t: &str, n: &str| -> Uuid {
        let id = db.create_vertex_from_type(ident(t)).expect("create vertex");
        set_str(&db, id, P_NAME, n);
        id
    };

    for p in f.people {
        let id = mk(PERSON, p.name);
        set_str(&db, id, P_FIRST, p.first);
        set_str(&db, id, P_LAST, p.last);
        set_u64(&db, id, P_BIRTHDAY, p.birthday);
        set_u64(&db, id, P_CREATED, p.created);
    }
    for m in f.messages {
        let id = mk(MESSAGE, m.name);
        set_str(&db, id, P_CONTENT, m.content);
        set_u64(&db, id, P_CREATED, m.created);
    }
    for fo in f.forums {
        let id = mk(FORUM, fo.name);
        set_str(&db, id, P_TITLE, fo.title);
    }

    for m in f.messages {
        let mv = by_name(&db, MESSAGE, m.name).expect("message");
        let cv = by_name(&db, PERSON, m.creator).expect("creator");
        edge(&db, mv, HAS_CREATOR, cv);
        if let Some(parent) = m.reply_to {
            let pv = by_name(&db, MESSAGE, parent).expect("parent");
            edge(&db, mv, REPLY_OF, pv);
        }
    }
    for fo in f.forums {
        let fv = by_name(&db, FORUM, fo.name).expect("forum");
        let mv = by_name(&db, PERSON, fo.moderator).expect("moderator");
        edge(&db, fv, HAS_MODERATOR, mv);
        for msg in fo.contains {
            let target = by_name(&db, MESSAGE, msg).expect("contained message");
            edge(&db, fv, CONTAINER_OF, target);
        }
    }
    for (a, b, since) in f.knows {
        let av = by_name(&db, PERSON, a).expect("person a");
        let bv = by_name(&db, PERSON, b).expect("person b");
        let e = edge(&db, av, KNOWS, bv);
        db.set_properties(
            SpecificEdgeQuery::single(e),
            ident(P_SINCE),
            &Json::new((*since).into()),
        )
        .expect("set since");
    }
    db
}

fn edge(db: &Db, from: Uuid, t: &str, to: Uuid) -> Edge {
    let e = Edge::new(from, ident(t), to);
    assert!(db.create_edge(&e).expect("create edge"), "endpoints exist");
    e
}

fn set_str(db: &Db, id: Uuid, key: &str, value: &str) {
    db.set_properties(
        SpecificVertexQuery::single(id),
        ident(key),
        &Json::new(value.into()),
    )
    .expect("set string property");
}

fn set_u64(db: &Db, id: Uuid, key: &str, value: u64) {
    db.set_properties(
        SpecificVertexQuery::single(id),
        ident(key),
        &Json::new(value.into()),
    )
    .expect("set u64 property");
}

// --- mutation and scan (see the matching section in `native.rs`) -----------

/// Every person's name, sorted — the oracle for `native::all_people`.
///
/// IndraDB has no label registry, so "all of type X" is a full vertex scan
/// filtered by `v.t`. That asymmetry is the point of having both: whatever the
/// native engine does to make its scan cheap must not change the answer.
pub fn all_people(db: &Db) -> Vec<String> {
    let want = ident(PERSON);
    let Ok(out) = db.get(indradb::AllVertexQuery) else {
        return Vec::new();
    };
    let mut names: Vec<String> = match out.last() {
        Some(QueryOutputValue::Vertices(vs)) => vs
            .iter()
            .filter(|v| v.t == want)
            .map(|v| vstr(db, v.id, P_NAME))
            .collect(),
        _ => Vec::new(),
    };
    names.sort();
    names
}

/// Delete a person.
///
/// **IndraDB cascades; our engine tombstones.** IndraDB removes the incident
/// edges outright (`delete_vertex_cascades_to_edges`), while `twizzler-graph`
/// leaves them in place and hides them, since an edge is alive only while both
/// endpoints are. The two disagree about what is *stored* and must still agree
/// about every answer — which is exactly the equivalence worth asserting.
pub fn delete_person(db: &Db, name: &str) -> bool {
    match by_name(db, PERSON, name) {
        Some(id) => {
            db.delete(SpecificVertexQuery::single(id))
                .expect("delete vertex");
            true
        }
        None => false,
    }
}

/// Delete the `knows` edge between two people, in whichever direction it was
/// stored. Returns whether one was found.
pub fn delete_knows(db: &Db, a: &str, b: &str) -> bool {
    let (Some(av), Some(bv)) = (by_name(db, PERSON, a), by_name(db, PERSON, b)) else {
        return false;
    };
    let want = ident(KNOWS);
    let found = out_edges(db, av, KNOWS)
        .into_iter()
        .chain(out_edges(db, bv, KNOWS))
        .find(|e| {
            e.t == want && ((e.outbound_id == av && e.inbound_id == bv)
                || (e.outbound_id == bv && e.inbound_id == av))
        });
    match found {
        Some(e) => {
            db.delete(SpecificEdgeQuery::single(e))
                .expect("delete edge");
            true
        }
        None => false,
    }
}

// --- reads -----------------------------------------------------------------

/// Find a vertex by the fixture's stable name, restricted to a type.
pub fn by_name(db: &Db, t: &str, name: &str) -> Option<Uuid> {
    let q = VertexWithPropertyValueQuery::new(ident(P_NAME), Json::new(name.into()));
    let out = db.get(q).ok()?;
    let want = ident(t);
    match out.last()? {
        QueryOutputValue::Vertices(vs) => {
            vs.iter().find(|v| v.t == want).map(|v| v.id)
        }
        _ => None,
    }
}

fn vprops(db: &Db, id: Uuid) -> Vec<(Identifier, Json)> {
    let Ok(q) = SpecificVertexQuery::single(id).properties() else {
        return Vec::new();
    };
    let Ok(out) = db.get(q) else { return Vec::new() };
    match out.last() {
        Some(QueryOutputValue::VertexProperties(props)) => props
            .first()
            .map(|vp| {
                vp.props
                    .iter()
                    .map(|p| (p.name, p.value.clone()))
                    .collect()
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn vstr(db: &Db, id: Uuid, key: &str) -> String {
    let want = ident(key);
    vprops(db, id)
        .into_iter()
        .find(|(n, _)| *n == want)
        .and_then(|(_, v)| v.0.as_str().map(str::to_string))
        .unwrap_or_default()
}

fn vu64(db: &Db, id: Uuid, key: &str) -> u64 {
    let want = ident(key);
    vprops(db, id)
        .into_iter()
        .find(|(n, _)| *n == want)
        .and_then(|(_, v)| v.0.as_u64())
        .unwrap_or(0)
}

fn eu64(db: &Db, e: &Edge, key: &str) -> u64 {
    let Ok(q) = SpecificEdgeQuery::single(e.clone()).properties() else {
        return 0;
    };
    let Ok(out) = db.get(q) else { return 0 };
    let want = ident(key);
    match out.last() {
        Some(QueryOutputValue::EdgeProperties(props)) => props
            .first()
            .and_then(|ep| {
                ep.props
                    .iter()
                    .find(|p| p.name == want)
                    .and_then(|p| p.value.0.as_u64())
            })
            .unwrap_or(0),
        _ => 0,
    }
}

/// Outgoing edges of `id` with the given type.
fn out_edges(db: &Db, id: Uuid, t: &str) -> Vec<Edge> {
    let Ok(q) = SpecificVertexQuery::single(id).outbound() else {
        return Vec::new();
    };
    let Ok(out) = db.get(q) else { return Vec::new() };
    let want = ident(t);
    match out.last() {
        Some(QueryOutputValue::Edges(es)) => {
            es.iter().filter(|e| e.t == want).cloned().collect()
        }
        _ => Vec::new(),
    }
}

/// Incoming edges of `id` with the given type.
fn in_edges(db: &Db, id: Uuid, t: &str) -> Vec<Edge> {
    let Ok(q) = SpecificVertexQuery::single(id).inbound() else {
        return Vec::new();
    };
    let Ok(out) = db.get(q) else { return Vec::new() };
    let want = ident(t);
    match out.last() {
        Some(QueryOutputValue::Edges(es)) => {
            es.iter().filter(|e| e.t == want).cloned().collect()
        }
        _ => Vec::new(),
    }
}

/// Everyone `id` knows, in either direction (`knows` is stored one way).
fn knows_set(db: &Db, id: Uuid) -> Vec<Uuid> {
    let mut out: Vec<Uuid> = out_edges(db, id, KNOWS).iter().map(|e| e.inbound_id).collect();
    out.extend(in_edges(db, id, KNOWS).iter().map(|e| e.outbound_id));
    out
}

// --- the short reads -------------------------------------------------------

pub fn is1_profile(db: &Db, person: &str) -> Option<Profile> {
    let v = by_name(db, PERSON, person)?;
    Some(Profile {
        name: person.to_string(),
        first: vstr(db, v, P_FIRST),
        last: vstr(db, v, P_LAST),
        birthday: vu64(db, v, P_BIRTHDAY),
        created: vu64(db, v, P_CREATED),
    })
}

pub fn is2_recent_messages(db: &Db, person: &str, limit: usize) -> Vec<MessageRow> {
    let Some(v) = by_name(db, PERSON, person) else {
        return Vec::new();
    };
    let mut rows: Vec<MessageRow> = in_edges(db, v, HAS_CREATOR)
        .into_iter()
        .map(|e| {
            let m = e.outbound_id;
            MessageRow {
                name: vstr(db, m, P_NAME),
                content: vstr(db, m, P_CONTENT),
                created: vu64(db, m, P_CREATED),
            }
        })
        .collect();
    rows.sort_by(|a, b| b.created.cmp(&a.created).then(a.name.cmp(&b.name)));
    rows.truncate(limit);
    rows
}

pub fn is3_friends(db: &Db, person: &str) -> Vec<FriendRow> {
    let Some(v) = by_name(db, PERSON, person) else {
        return Vec::new();
    };
    let mut rows: Vec<FriendRow> = Vec::new();
    for e in out_edges(db, v, KNOWS) {
        rows.push(FriendRow {
            name: vstr(db, e.inbound_id, P_NAME),
            since: eu64(db, &e, P_SINCE),
        });
    }
    for e in in_edges(db, v, KNOWS) {
        rows.push(FriendRow {
            name: vstr(db, e.outbound_id, P_NAME),
            since: eu64(db, &e, P_SINCE),
        });
    }
    rows.sort_by(|a, b| b.since.cmp(&a.since).then(a.name.cmp(&b.name)));
    rows
}

pub fn is4_message(db: &Db, message: &str) -> Option<MessageRow> {
    let m = by_name(db, MESSAGE, message)?;
    Some(MessageRow {
        name: message.to_string(),
        content: vstr(db, m, P_CONTENT),
        created: vu64(db, m, P_CREATED),
    })
}

pub fn is5_creator(db: &Db, message: &str) -> Option<Profile> {
    let m = by_name(db, MESSAGE, message)?;
    let e = out_edges(db, m, HAS_CREATOR).into_iter().next()?;
    let name = vstr(db, e.inbound_id, P_NAME);
    is1_profile(db, &name)
}

pub fn is6_forum(db: &Db, message: &str) -> Option<ForumRow> {
    let mut cur = by_name(db, MESSAGE, message)?;
    let mut hops = 0usize;
    while let Some(e) = out_edges(db, cur, REPLY_OF).into_iter().next() {
        cur = e.inbound_id;
        hops += 1;
        if hops > 64 {
            return None;
        }
    }
    let forum = in_edges(db, cur, CONTAINER_OF).into_iter().next()?.outbound_id;
    let moderator = out_edges(db, forum, HAS_MODERATOR)
        .into_iter()
        .next()?
        .inbound_id;
    Some(ForumRow {
        title: vstr(db, forum, P_TITLE),
        moderator: vstr(db, moderator, P_NAME),
    })
}

pub fn is7_replies(db: &Db, message: &str) -> Vec<ReplyRow> {
    let Some(m) = by_name(db, MESSAGE, message) else {
        return Vec::new();
    };
    let parent_author = out_edges(db, m, HAS_CREATOR)
        .into_iter()
        .next()
        .map(|e| e.inbound_id);

    let mut rows: Vec<ReplyRow> = in_edges(db, m, REPLY_OF)
        .into_iter()
        .filter_map(|e| {
            let r = e.outbound_id;
            let author = out_edges(db, r, HAS_CREATOR).into_iter().next()?.inbound_id;
            let knows = match parent_author {
                Some(pa) => knows_set(db, author).contains(&pa),
                None => false,
            };
            Some(ReplyRow {
                message: vstr(db, r, P_NAME),
                content: vstr(db, r, P_CONTENT),
                created: vu64(db, r, P_CREATED),
                author: vstr(db, author, P_NAME),
                author_knows_parent_author: knows,
            })
        })
        .collect();
    rows.sort_by(|a, b| b.created.cmp(&a.created).then(a.author.cmp(&b.author)));
    rows
}
