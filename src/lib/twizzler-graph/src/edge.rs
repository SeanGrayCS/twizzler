//! Edges.
//!
//! An edge is its own persistent object holding an `InvPtr` to each endpoint
//! vertex, so per-vertex adjacency lists can point at it and following an edge
//! is a pointer dereference. `EdgeRef` is the registry mirror for enumeration
//! and id lookup. `EdgeHandle` is the element passed to user filter predicates.
//!
//! `props_raw` and `flags` are reserved (0 now): an `ObjID` of a side property
//! object, and a tombstone bit for deletes.

use twizzler::{
    marker::{BaseType, Invariant},
    ptr::InvPtr,
};

use crate::vertex::{Vertex, VertexId};

/// Public edge id (an append index).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeId(pub u64);

/// An edge object: both endpoints are invariant pointers.
#[repr(C)]
pub(crate) struct Edge {
    pub(crate) id: u64,
    pub(crate) label: u32,
    pub(crate) from_id: u64,
    pub(crate) to_id: u64,
    pub(crate) from: InvPtr<Vertex>,
    pub(crate) to: InvPtr<Vertex>,
    pub(crate) props_raw: u128, // reserved (0 = none)
    pub(crate) flags: u32,      // reserved (bit 0 = tombstone)
}
unsafe impl Invariant for Edge {}
impl BaseType for Edge {}

/// Registry mirror of an edge (enumeration / id lookup; not used for traversal).
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct EdgeRef {
    pub(crate) id: u64,
    pub(crate) label: u32,
    pub(crate) from_id: u64,
    pub(crate) to_id: u64,
    pub(crate) eobj_raw: u128,
    pub(crate) flags: u32,
}
unsafe impl Invariant for EdgeRef {}

/// Edge element passed to user filter predicates.
#[derive(Clone, Copy)]
pub struct EdgeHandle {
    id: EdgeId,
    label: u32,
    from: VertexId,
    to: VertexId,
}
impl EdgeHandle {
    pub(crate) fn new(id: EdgeId, label: u32, from: VertexId, to: VertexId) -> Self {
        EdgeHandle { id, label, from, to }
    }
    pub fn id(&self) -> EdgeId {
        self.id
    }
    pub fn label_id(&self) -> u32 {
        self.label
    }
    pub fn from(&self) -> VertexId {
        self.from
    }
    pub fn to(&self) -> VertexId {
        self.to
    }
}
