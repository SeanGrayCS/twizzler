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
