//! **E3-AC2 — the M3 milestone assertion.**
//!
//! The same seven queries, over the same fixture, on both engines, must return
//! *identical* results. Combined with the hand-computed goldens in
//! `tests_native`, this pins both engines to the same correct answers rather
//! than merely to each other.
//!
//! Why this matters beyond correctness: every Phase-4 benchmark number is a
//! comparison of two systems answering the same question. Without this test,
//! a performance difference could just as easily be a semantic difference —
//! one engine quietly returning fewer rows, or a different order, and looking
//! faster for it.

use crate::fixture::FIXTURE;
use crate::{baseline, native};

/// Load both engines from the same fixture. Distinct registration names keep
/// the two stores independent.
fn both(tag: &str) -> (twizzler_graph::Graph, indradb::Database<twizzler_indradb::TwizzlerDatastore>) {
    let g = native::load(&format!("e3n-{tag}"), &FIXTURE).expect("load native");
    let db = baseline::load(&format!("e3b-{tag}"), &FIXTURE);
    (g, db)
}

/// Every person and message in the fixture, so the equivalence checks cover
/// the whole graph rather than a lucky sample — including `dave`, who has no
/// friends and no messages, and `m5`, a leaf reply.
const PEOPLE: &[&str] = &["alice", "bob", "carol", "dave"];
const MESSAGES: &[&str] = &["m1", "m2", "m3", "m4", "m5"];

#[test]
fn is1_agrees() {
    let (g, db) = both("is1");
    for p in PEOPLE.iter().chain(["nobody"].iter()) {
        assert_eq!(
            native::is1_profile(&g, p),
            baseline::is1_profile(&db, p),
            "IS1 disagreed for {p}"
        );
    }
}

#[test]
fn is2_agrees() {
    let (g, db) = both("is2");
    for p in PEOPLE {
        for limit in [1usize, 10] {
            assert_eq!(
                native::is2_recent_messages(&g, p, limit),
                baseline::is2_recent_messages(&db, p, limit),
                "IS2 disagreed for {p} (limit {limit})"
            );
        }
    }
}

#[test]
fn is3_agrees() {
    let (g, db) = both("is3");
    for p in PEOPLE {
        assert_eq!(
            native::is3_friends(&g, p),
            baseline::is3_friends(&db, p),
            "IS3 disagreed for {p}"
        );
    }
}

#[test]
fn is4_agrees() {
    let (g, db) = both("is4");
    for m in MESSAGES.iter().chain(["m99"].iter()) {
        assert_eq!(
            native::is4_message(&g, m),
            baseline::is4_message(&db, m),
            "IS4 disagreed for {m}"
        );
    }
}

#[test]
fn is5_agrees() {
    let (g, db) = both("is5");
    for m in MESSAGES.iter().chain(["m99"].iter()) {
        assert_eq!(
            native::is5_creator(&g, m),
            baseline::is5_creator(&db, m),
            "IS5 disagreed for {m}"
        );
    }
}

#[test]
fn is6_agrees() {
    let (g, db) = both("is6");
    for m in MESSAGES {
        assert_eq!(
            native::is6_forum(&g, m),
            baseline::is6_forum(&db, m),
            "IS6 disagreed for {m}"
        );
    }
}

#[test]
fn is7_agrees() {
    let (g, db) = both("is7");
    for m in MESSAGES {
        assert_eq!(
            native::is7_replies(&g, m),
            baseline::is7_replies(&db, m),
            "IS7 disagreed for {m}"
        );
    }
}

// --- the surface A7 is about to change -------------------------------------
//
// Added 2026-08-04, deliberately *before* the v5 record format. A7 puts edges
// in `locs` beside vertices, so `vertices()` must begin excluding them and a
// tombstoned edge-record starts sharing a code path with a tombstoned vertex.
// None of that was checked against the baseline. Pinning it now costs three
// tests; reconstructing the expected answers afterwards, from an engine that
// has already changed, costs much more and is worth much less.

/// Label scan agrees. Both engines reach the same set by different means — the
/// native one through its label registry, IndraDB through a full vertex scan
/// filtered by type — so a change to how the native scan is made cheap cannot
/// quietly change *what* it finds.
#[test]
fn label_scan_agrees() {
    let (g, db) = both("scan");
    assert_eq!(native::all_people(&g), baseline::all_people(&db));
    assert_eq!(
        native::all_people(&g),
        vec!["alice", "bob", "carol", "dave"],
        "and both agree with the fixture, not merely with each other"
    );
}

/// Deleting a vertex agrees, across every read.
///
/// **The engines disagree about storage and must agree about answers.** IndraDB
/// cascades the delete to incident edges; `twizzler-graph` tombstones the vertex
/// and leaves the edges in place, hidden because an edge is alive only while
/// both endpoints are. Two different representations of "gone" — every query has
/// to be unable to tell.
///
/// `bob` is chosen for having the most structure to disturb: he authors m2,
/// moderates f1, and is known by both alice and carol.
#[test]
fn deleting_a_vertex_agrees() {
    let (mut g, db) = both("delv");
    native::delete_person(&mut g, "bob").expect("native delete");
    assert!(baseline::delete_person(&db, "bob"), "baseline delete");

    assert_eq!(native::all_people(&g), baseline::all_people(&db));
    assert!(!native::all_people(&g).contains(&"bob".to_string()));

    for p in PEOPLE {
        assert_eq!(
            native::is1_profile(&g, p),
            baseline::is1_profile(&db, p),
            "IS1 disagreed for {p} after deleting bob"
        );
        assert_eq!(
            native::is2_recent_messages(&g, p, 10),
            baseline::is2_recent_messages(&db, p, 10),
            "IS2 disagreed for {p} after deleting bob"
        );
        assert_eq!(
            native::is3_friends(&g, p),
            baseline::is3_friends(&db, p),
            "IS3 disagreed for {p} after deleting bob"
        );
    }
    // The reads that route *through* the deleted vertex, not just past it.
    for m in MESSAGES {
        assert_eq!(
            native::is5_creator(&g, m),
            baseline::is5_creator(&db, m),
            "IS5 disagreed for {m} after deleting its creator"
        );
        assert_eq!(
            native::is6_forum(&g, m),
            baseline::is6_forum(&db, m),
            "IS6 disagreed for {m} after deleting the moderator"
        );
    }
}

/// Deleting an edge agrees. Friendship is symmetric and stored one way, so this
/// also pins that both engines drop it from *both* endpoints' answers.
#[test]
fn deleting_an_edge_agrees() {
    let (mut g, db) = both("dele");
    assert!(
        native::delete_knows(&mut g, "alice", "bob").expect("native delete"),
        "the fixture has an alice-bob knows edge"
    );
    assert!(baseline::delete_knows(&db, "alice", "bob"), "baseline delete");

    for p in PEOPLE {
        assert_eq!(
            native::is3_friends(&g, p),
            baseline::is3_friends(&db, p),
            "IS3 disagreed for {p} after deleting alice-bob"
        );
    }
    // Both endpoints, and only the intended edge.
    assert!(!native::is3_friends(&g, "alice").iter().any(|f| f.name == "bob"));
    assert!(!native::is3_friends(&g, "bob").iter().any(|f| f.name == "alice"));
    assert!(native::is3_friends(&g, "alice").iter().any(|f| f.name == "carol"));

    // Vertices are untouched by an edge delete.
    assert_eq!(native::all_people(&g), baseline::all_people(&db));
}

/// The baseline reproduces the hand-computed goldens too — so "the engines
/// agree" cannot be satisfied by both being wrong in the same way.
#[test]
fn baseline_matches_goldens() {
    use crate::results::*;
    let db = baseline::load("e3b-golden", &FIXTURE);

    assert_eq!(
        baseline::is1_profile(&db, "alice"),
        Some(Profile {
            name: "alice".into(),
            first: "Alice".into(),
            last: "Ng".into(),
            birthday: 19900101,
            created: 100,
        })
    );
    assert_eq!(
        baseline::is2_recent_messages(&db, "alice", 10),
        vec![
            MessageRow {
                name: "m3".into(),
                content: "again".into(),
                created: 3000
            },
            MessageRow {
                name: "m1".into(),
                content: "hello".into(),
                created: 1000
            },
        ]
    );
    assert_eq!(
        baseline::is3_friends(&db, "alice"),
        vec![
            FriendRow {
                name: "carol".into(),
                since: 700
            },
            FriendRow {
                name: "bob".into(),
                since: 500
            },
        ]
    );
    // The reply-chain walk to a root post's forum.
    assert_eq!(
        baseline::is6_forum(&db, "m5"),
        Some(ForumRow {
            title: "General".into(),
            moderator: "bob".into(),
        })
    );
    assert_eq!(
        baseline::is7_replies(&db, "m1"),
        vec![ReplyRow {
            message: "m4".into(),
            content: "re: hello".into(),
            created: 1500,
            author: "carol".into(),
            author_knows_parent_author: true,
        }]
    );
}
