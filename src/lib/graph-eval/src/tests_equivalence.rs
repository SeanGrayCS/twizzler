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
