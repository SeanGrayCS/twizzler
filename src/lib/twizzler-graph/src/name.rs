//! Inline, fixed-size names (`NameKey`) for labels and vertices.
//!
//! Object-resident data must be `Invariant` (no heap pointers), so names are a
//! fixed buffer rather than a `String`. Names longer than 31 bytes are
//! truncated.

use twizzler::marker::Invariant;

// `new` zero-fills the unused tail, so byte-wise Eq/Hash are well-defined.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct NameKey {
    len: u8,
    bytes: [u8; 31],
}
unsafe impl Invariant for NameKey {}

// Ordering is by *string* content, not by the raw record. Deriving `Ord`
// would compare `len` first — making "z" < "aa" — which would silently
// mis-sort every `order_by_name` / `order_by_prop` result.
impl Ord for NameKey {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}
impl PartialOrd for NameKey {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl NameKey {
    pub fn new(s: &str) -> Self {
        let b = s.as_bytes();
        let n = b.len().min(31);
        let mut bytes = [0u8; 31];
        bytes[..n].copy_from_slice(&b[..n]);
        NameKey {
            len: n as u8,
            bytes,
        }
    }
    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.bytes[..self.len as usize]).unwrap_or("")
    }
    pub fn eq_str(&self, s: &str) -> bool {
        self.as_str() == s
    }
}

/// Debug prints the string form, not the raw byte buffer — that's what you
/// want in a test-failure message (needed since `PropValue::Str` derives Debug).
impl core::fmt::Debug for NameKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("NameKey").field(&self.as_str()).finish()
    }
}
