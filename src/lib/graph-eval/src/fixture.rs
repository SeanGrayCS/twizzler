//! The shared fixture: one small social graph, defined once, loaded into
//! either engine.
//!
//! Engine-agnostic on purpose. Elements are identified by **name** (a stable
//! string key), never by engine id — our engine assigns append indices while
//! IndraDB assigns UUIDs, so names are the only currency in which the two
//! engines' answers can be compared. Every golden result in the tests is
//! phrased in names for the same reason.
//!
//! Schema (LDBC-SNB shaped, trimmed to what the short reads touch):
//!
//! ```text
//! person  --knows{since}-->  person      (stored one way; queries read both)
//! message --hasCreator-->    person
//! message --replyOf-->       message     (chains, arbitrarily deep)
//! forum   --containerOf-->   message     (root posts only, as in LDBC)
//! forum   --hasModerator-->  person
//! ```

/// Vertex labels.
pub const PERSON: &str = "person";
pub const MESSAGE: &str = "message";
pub const FORUM: &str = "forum";

/// Edge labels.
pub const KNOWS: &str = "knows";
pub const HAS_CREATOR: &str = "hasCreator";
pub const REPLY_OF: &str = "replyOf";
pub const CONTAINER_OF: &str = "containerOf";
pub const HAS_MODERATOR: &str = "hasModerator";

/// Property keys. Kept short: our engine's `NameKey` truncates at 31 bytes,
/// and LDBC's own names (`creationDate`) fit comfortably.
pub const P_FIRST: &str = "first";
pub const P_LAST: &str = "last";
pub const P_BIRTHDAY: &str = "birthday";
pub const P_CREATED: &str = "created";
pub const P_CONTENT: &str = "content";
pub const P_TITLE: &str = "title";
/// On `knows` edges: when the friendship was formed.
pub const P_SINCE: &str = "since";

/// A person to load.
pub struct PersonSpec {
    pub name: &'static str,
    pub first: &'static str,
    pub last: &'static str,
    pub birthday: u64,
    pub created: u64,
}

/// A message to load. `reply_to` is `None` for a root post.
pub struct MessageSpec {
    pub name: &'static str,
    pub content: &'static str,
    pub created: u64,
    pub creator: &'static str,
    pub reply_to: Option<&'static str>,
}

/// A forum to load, with the root posts it contains.
pub struct ForumSpec {
    pub name: &'static str,
    pub title: &'static str,
    pub moderator: &'static str,
    pub contains: &'static [&'static str],
}

/// A `knows` edge: `(a, b, since)`, stored a → b.
pub type KnowsSpec = (&'static str, &'static str, u64);

/// The whole fixture.
pub struct Fixture {
    pub people: &'static [PersonSpec],
    pub messages: &'static [MessageSpec],
    pub forums: &'static [ForumSpec],
    pub knows: &'static [KnowsSpec],
}

/// The canonical fixture. Deliberately small enough to reason about by hand
/// (golden results are written out literally) but shaped to exercise every
/// short read: a reply chain three deep, a person with several messages at
/// distinct timestamps, friendships with distinct `since` values, and a
/// message whose forum is only reachable by walking the reply chain to its
/// root.
pub const FIXTURE: Fixture = Fixture {
    people: &[
        PersonSpec {
            name: "alice",
            first: "Alice",
            last: "Ng",
            birthday: 19900101,
            created: 100,
        },
        PersonSpec {
            name: "bob",
            first: "Bob",
            last: "Ito",
            birthday: 19851212,
            created: 200,
        },
        PersonSpec {
            name: "carol",
            first: "Carol",
            last: "Yu",
            birthday: 19920630,
            created: 300,
        },
        PersonSpec {
            name: "dave",
            first: "Dave",
            last: "Oh",
            birthday: 19880505,
            created: 400,
        },
    ],
    messages: &[
        // alice posts m1 and m3; bob posts m2. m4 replies to m1, m5 to m4.
        MessageSpec {
            name: "m1",
            content: "hello",
            created: 1000,
            creator: "alice",
            reply_to: None,
        },
        MessageSpec {
            name: "m2",
            content: "world",
            created: 2000,
            creator: "bob",
            reply_to: None,
        },
        MessageSpec {
            name: "m3",
            content: "again",
            created: 3000,
            creator: "alice",
            reply_to: None,
        },
        MessageSpec {
            name: "m4",
            content: "re: hello",
            created: 1500,
            creator: "carol",
            reply_to: Some("m1"),
        },
        MessageSpec {
            name: "m5",
            content: "re: re: hello",
            created: 1600,
            creator: "bob",
            reply_to: Some("m4"),
        },
    ],
    forums: &[ForumSpec {
        name: "f1",
        title: "General",
        moderator: "bob",
        contains: &["m1", "m2", "m3"],
    }],
    // alice–bob, alice–carol, bob–carol. dave knows nobody (empty-result case).
    knows: &[
        ("alice", "bob", 500),
        ("alice", "carol", 700),
        ("bob", "carol", 600),
    ],
};
