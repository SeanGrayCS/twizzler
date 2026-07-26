//! Canonical result types for the short reads.
//!
//! Both engines' implementations return *these* types, phrased entirely in
//! names and property values — no engine ids. That is what makes the
//! cross-engine equivalence assertion in E3-AC2 a real comparison rather than
//! a shape check: if the two engines disagree about anything observable, these
//! values differ and the test fails.

/// IS1 — profile of a person.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Profile {
    pub name: String,
    pub first: String,
    pub last: String,
    pub birthday: u64,
    pub created: u64,
}

/// IS2 — one of a person's recent messages. Also IS4's shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageRow {
    pub name: String,
    pub content: String,
    pub created: u64,
}

/// IS3 — a friend, with when the friendship formed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FriendRow {
    pub name: String,
    pub since: u64,
}

/// IS6 — the forum a message ultimately sits in, and its moderator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForumRow {
    pub title: String,
    pub moderator: String,
}

/// IS7 — a reply, its author, and whether that author knows the author of the
/// message being replied to (LDBC's `knows` flag).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplyRow {
    pub message: String,
    pub content: String,
    pub created: u64,
    pub author: String,
    pub author_knows_parent_author: bool,
}
