//! Typed property values ([`PropValue`]).
//!
//! Properties now live in the arena, in the record itself:
//!
//! - traversal properties are inline slots, fixed at insert, contiguous with
//!   the record head so a filtering walk never leaves its cache lines;
//! - data properties live in a block elsewhere in the same arena, reached
//!   through one `u64` in the record.
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
    TextRef { seg: u32, off: u64, len: u32 },
    BlobRef { seg: u32, off: u64, len: u32 },
}
unsafe impl Invariant for PropValue {}

impl PropValue {
    /// Convenience constructor for string values (truncates like `NameKey`).
    pub fn str(s: &str) -> Self {
        PropValue::Str(NameKey::new(s))
    }
}
