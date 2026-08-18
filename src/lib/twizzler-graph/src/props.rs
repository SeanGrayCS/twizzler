//! Typed property values ([`PropValue`]).
//!
//! Properties live in the arena, in the record itself:
//!
//! - traversal properties are inline slots, fixed at insert, contiguous with
//!   the record head so a filtering walk never leaves its cache lines;
//! - data properties live in a block elsewhere in the same arena, reached
//!   through one `u64` in the record.
//!
//! Both are `PropSlot`s in `arena_store.rs`. Nothing in this module allocates,
//! maps or fails — it defines a value type and nothing else.
//!
//! Values are a fixed-size tagged union and keys are interned ids, which is
//! what keeps a record `Invariant` (no heap pointers). Strings inherit
//! [`NameKey`] semantics: byte-wise truncation at 31.
use twizzler::marker::Invariant;

use crate::name::NameKey;

/// A property value: fixed-size, invariant, and comparable — equality backs
/// the DSL's `has(key, value)`, ordering backs `order_by_prop`.
///
/// Ordering *within* a variant is the natural one (`Str` compares as text,
/// not as its raw record — see `NameKey`'s manual `Ord`). Ordering *across*
/// variants follows declaration order, which is arbitrary but deterministic;
/// mixed-type properties are a schema smell, and a stable answer beats an
/// unpredictable one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(C, u32)]
pub enum PropValue {
    I64(i64),
    U64(u64),
    Bool(bool),
    /// An object reference by raw id (relocatable, like the registries).
    ObjId(u128),
    /// A short string; truncates byte-wise at 31 like all `NameKey`s.
    Str(NameKey),
    /// A variable-width string (≤255 B) in the shared byte store.
    ///
    /// Never handed to a caller: `get_vertex_prop` filters these out and
    /// `get_vertex_text` resolves them instead. The derives above are
    /// structural, so two identical strings stored separately compare unequal,
    /// and ordering would be by insertion position rather than by content — a
    /// filter or sort that saw one of these would be silently wrong rather
    /// than merely unsupported.
    TextRef { seg: u32, off: u64, len: u32 },
    /// Arbitrary-length bytes in the shared byte store, not queryable. Same
    /// non-exposure rule as `TextRef`, and for the same reason.
    BlobRef { seg: u32, off: u64, len: u32 },
}
unsafe impl Invariant for PropValue {}

impl PropValue {
    /// Convenience constructor for string values (truncates like `NameKey`).
    pub fn str(s: &str) -> Self {
        PropValue::Str(NameKey::new(s))
    }
}
