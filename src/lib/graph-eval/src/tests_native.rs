//! E3a: golden results for IS1–IS7 on the native engine.
//!
//! Expected values are written out literally, computed by hand from
//! [`FIXTURE`], not derived from the implementation — otherwise the test would
//! only prove the code agrees with itself. In E3b the same goldens are
//! asserted against the IndraDB baseline, which is what makes M3's
//! "same query set, same answers" claim checkable.

use crate::fixture::FIXTURE;
use crate::native::*;
use crate::results::*;

/// Each test loads its own graph name so runs stay independent and idempotent.
fn g(name: &str) -> twizzler_graph::Graph {
    load(name, &FIXTURE).expect("load fixture")
}

#[test]
fn is1_profile_golden() {
    let g = g("e3-is1");
    assert_eq!(
        is1_profile(&g, "alice"),
        Some(Profile {
            name: "alice".into(),
            first: "Alice".into(),
            last: "Ng".into(),
            birthday: 19900101,
            created: 100,
        })
    );
    // A person who does not exist.
    assert_eq!(is1_profile(&g, "nobody"), None);
}

#[test]
fn is2_recent_messages_golden() {
    let g = g("e3-is2");
    // alice wrote m1 (1000) and m3 (3000); newest first.
    assert_eq!(
        is2_recent_messages(&g, "alice", 10),
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
    // The limit is the "top N" the LDBC read specifies.
    assert_eq!(
        is2_recent_messages(&g, "alice", 1),
        vec![MessageRow {
            name: "m3".into(),
            content: "again".into(),
            created: 3000
        }]
    );
    // dave wrote nothing.
    assert!(is2_recent_messages(&g, "dave", 10).is_empty());
}

#[test]
fn is3_friends_golden() {
    let g = g("e3-is3");
    // alice knows bob (since 500) and carol (since 700): date desc.
    assert_eq!(
        is3_friends(&g, "alice"),
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
    // bob's friendships are stored in both directions relative to him
    // (alice→bob, bob→carol) — `both_e` must find both.
    assert_eq!(
        is3_friends(&g, "bob"),
        vec![
            FriendRow {
                name: "carol".into(),
                since: 600
            },
            FriendRow {
                name: "alice".into(),
                since: 500
            },
        ]
    );
    assert!(is3_friends(&g, "dave").is_empty());
}

#[test]
fn is4_message_golden() {
    let g = g("e3-is4");
    assert_eq!(
        is4_message(&g, "m4"),
        Some(MessageRow {
            name: "m4".into(),
            content: "re: hello".into(),
            created: 1500,
        })
    );
    assert_eq!(is4_message(&g, "m99"), None);
}

#[test]
fn is5_creator_golden() {
    let g = g("e3-is5");
    let creator = is5_creator(&g, "m4").expect("m4 has a creator");
    assert_eq!(creator.name, "carol");
    assert_eq!(creator.first, "Carol");
    assert_eq!(is5_creator(&g, "m99"), None);
}

#[test]
fn is6_forum_golden() {
    let g = g("e3-is6");
    let expected = ForumRow {
        title: "General".into(),
        moderator: "bob".into(),
    };
    // A root post sits in the forum directly.
    assert_eq!(is6_forum(&g, "m1"), Some(expected.clone()));
    // A reply reaches it by walking one `replyOf` hop...
    assert_eq!(is6_forum(&g, "m4"), Some(expected.clone()));
    // ...and a reply-to-a-reply by walking two. This is the chain walk that
    // B3's `repeat`/`until` would express in the DSL.
    assert_eq!(is6_forum(&g, "m5"), Some(expected));
}

#[test]
fn is7_replies_golden() {
    let g = g("e3-is7");
    // m1 (by alice) has one direct reply, m4 by carol. carol knows alice.
    assert_eq!(
        is7_replies(&g, "m1"),
        vec![ReplyRow {
            message: "m4".into(),
            content: "re: hello".into(),
            created: 1500,
            author: "carol".into(),
            author_knows_parent_author: true,
        }]
    );
    // m4 (by carol) has one reply, m5 by bob. bob knows carol.
    assert_eq!(
        is7_replies(&g, "m4"),
        vec![ReplyRow {
            message: "m5".into(),
            content: "re: re: hello".into(),
            created: 1600,
            author: "bob".into(),
            author_knows_parent_author: true,
        }]
    );
    // Leaf message: no replies.
    assert!(is7_replies(&g, "m5").is_empty());
}

/// The fixture loads to the shape the goldens assume — a guard so a fixture
/// edit that silently changes the graph fails here rather than confusing
/// every query test at once.
#[test]
fn fixture_loads_expected_shape() {
    let g = g("e3-shape");
    assert_eq!(g.vertices_by_label(crate::fixture::PERSON).len(), 4);
    assert_eq!(g.vertices_by_label(crate::fixture::MESSAGE).len(), 5);
    assert_eq!(g.vertices_by_label(crate::fixture::FORUM).len(), 1);
    // 5 hasCreator + 2 replyOf + 1 hasModerator + 3 containerOf + 3 knows.
    assert_eq!(g.vertices().len(), 10);
}
