#![allow(dead_code)]

//! Mechanism. [`ArenaObject`] is a bump allocator *inside one object*. The
//! current layout is forced into three objects only because `VecObjectAlloc`
//! hardcodes a single data region per object; an arena hands out many disjoint
//! allocations, so a vertex record and both its adjacency chains can live in one
//! object — or a thousand vertices can.
//!
//! References are arena offsets, not `InvPtr`s. Within an arena a link is a
//! `u64` offset applied to the arena's own mapping, so intra-arena traversal
//! needs no FOT entry and touches no second object. Neighbours in *other*
//! arenas are named by `VertexId` and resolved through the location registry —
//! the "A4b" reference form.

use core::mem::size_of;

use twizzler::{
    alloc::arena::{ArenaBase, ArenaObject},
    marker::Invariant,
    object::{ObjID, ObjectBuilder, RawObject, TxObject},
    ptr::{GlobalPtr, InvPtr},
};

use crate::name::NameKey;
use crate::props::PropValue;
use crate::segvec::SegVec;

type Result<T> = core::result::Result<T, twizzler::error::TwzError>;

/// Adjacency entries per chunk. Small enough that tests exercise chunk
/// rollover cheaply; large enough that low-degree vertices need one chunk.
pub const ADJ_CHUNK: usize = 8;

/// `flags` bit 0: record is deleted.
const TOMBSTONE: u32 = 1;

/// Mirrored into `VertexLoc.flags` beside `TOMBSTONE`, and that is the whole
/// point. With edges as records, `locs` holds both, so `vertices()` has to
/// exclude edges — and if the only place that fact lived were the record, the
/// scan would have to resolve every record to answer. That is precisely the
/// 3.2× cold regression the liveness mirror was built to remove, traded against
/// its measured 5.2× warm gain. Keeping the bit in the mirror makes scan cost
/// independent of how wide records get, which matters more once records carry
/// inline properties.
///
/// It follows that is-edge must not also be expressible as a property, or
/// there are two sources of truth for it.
#[allow(dead_code)] // wired up with the record format
const IS_EDGE: u32 = 2;

/// `flags` bit 2/3: the record's inline out-/in-adjacency slot is occupied.
///
/// One slot each way, not two. Two would cover degree-2 vertices as well, but
/// the case that matters is the edge record, which is exactly degree-1, and
/// every extra slot widens *every* record including the vertices that spill to
/// chunks anyway.
const HAS_INLINE_OUT: u32 = 4;
const HAS_INLINE_IN: u32 = 8;

/// 48 bytes, of which 8 are padding. `PropValue` contains a `u128` variant,
/// so it aligns to 16 and the `u32` key cannot share its first word. The waste
/// is recorded rather than optimised: shrinking it means either dropping
/// `ObjId(u128)` from `PropValue` or splitting keys into a parallel array, and
/// both are format decisions that should be made against a measurement rather
/// than against this comment.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(crate) struct PropSlot {
    pub(crate) key_id: u32,
    pub(crate) _pad: u32,
    pub(crate) val: PropValue,
}
unsafe impl Invariant for PropSlot {}

/// Hand-written rather than derived, to exclude `_pad`.
///
/// Two slots are equal when they *mean* the same thing. A derived `PartialEq`
/// would compare the padding word, making equality depend on bytes nothing
/// reads — and the only thing keeping that word zero is the allocation-time
/// zeroing, which `add_record` deliberately does not rely on for correctness
/// (see the note there about the transaction's mapping). Deriving would quietly
/// promote padding from "never read" to "load-bearing in tests".
impl PartialEq for PropSlot {
    fn eq(&self, other: &Self) -> bool {
        self.key_id == other.key_id && self.val == other.val
    }
}

/// Header of a record's data-property block, followed by `cap` [`PropSlot`]s
/// of which `len` are live.
///
/// This block is the thing that may move, and it is the only one. It has
/// exactly one referent — `ArenaRecordHead::data_props`, a single `u64` — so
/// growing it is: allocate a bigger block, copy, write one field. No inbound
/// `InvPtr` names it, which is precisely why data properties can be added after
/// insert while inline traversal slots cannot.
///
/// 16 bytes, which is both `PropSlot`'s alignment and the arena's minimum, so
/// the slots that follow need no padding.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct DataBlockHead {
    pub(crate) len: u16,
    pub(crate) cap: u16,
    pub(crate) _pad: [u8; 12],
}
unsafe impl Invariant for DataBlockHead {}

/// Bytes occupied by a data-property block with room for `cap` entries.
pub(crate) const fn data_block_size(cap: u16) -> usize {
    size_of::<DataBlockHead>() + cap as usize * size_of::<PropSlot>()
}

/// Bytes occupied by a record carrying `nprops` inline traversal properties.
#[allow(dead_code)] // wired up with the record format
pub(crate) const fn record_size(nprops: u16) -> usize {
    size_of::<ArenaRecordHead>() + nprops as usize * size_of::<PropSlot>()
}


/// Self-describing on purpose: there is no schema. The record carries
/// `nprops`, so its extent is derivable from its own bytes and nothing external
/// has to be consulted to read it. That is what let the earlier schema-typed
/// draft (declared types, a `SegVec<TypeDef>`, type ids in `flags`) be dropped:
/// it fixed size per *declared type* where this fixes it per *record*, needing
/// no declaration step and wasting no slots.
///
/// Size is fixed at insert and the record never moves. This is forced by
/// pointer topology, not chosen: a record has O(in-degree) inbound
/// `AdjRef.neighbor` `InvPtr`s, each holding its offset, so relocating it means
/// rewriting all of them with durability ordering to get right. Data properties
/// escape this because their block has exactly one referent — `data_props`,
/// a single `u64` that can be repointed in place. *Many referents ⇒ immovable;
/// one referent ⇒ freely movable.*
// Not `Copy`, unlike every other record type here: it embeds `AdjRef`s,
// whose `InvPtr` carries a FOT index that means something only inside its own
// object. Copying a head between arenas would silently mis-resolve, which is
// the same reason `AdjRef` and `AdjChunk` are not `Copy`.
#[repr(C)]
#[allow(dead_code)] // wired up with the record format
pub(crate) struct ArenaRecordHead {
    pub(crate) id: u64,
    pub(crate) flags: u32,
    pub(crate) label: u32,
    /// Inline traversal slots. `u16` bounds a record at 65 535 of them, far
    /// above the ~11 000-vertex property ceiling this format exists to remove.
    pub(crate) nprops: u16,
    pub(crate) _pad: u16,
    /// A tombstoned record cannot simply be overwritten: inbound
    /// `AdjRef.neighbor` `InvPtr`s still hold its offset, and `walk_adj`
    /// resolves the pointer *before* checking liveness. Reusing the bytes with
    /// no invalidation would make a stale pointer resolve to a live record with
    /// a valid id — the liveness check passes and traversal returns a neighbour
    /// that was never connected. Silent, and worse than the `resolve`/
    /// `resolve_mut` incoherence because nothing faults.
    ///
    /// Every `AdjRef` records the generation it expects, so a stale entry
    /// mismatches and is skipped. Belongs to the slot, not the record: a
    /// reused slot's new record has a different id *and* a higher generation.
    ///
    /// Free: it lives in padding the head already had, so the record stays 128
    /// bytes and `AdjRef` stays 24.
    pub(crate) generation: u32,
    pub(crate) name: NameKey,
    pub(crate) target_raw: u128,
    pub(crate) data_props: u64,
    /// Chunk-chain heads. 0 means "no chunk", which for a degree-≤1 record is
    /// now the normal case — see the inline slots below.
    pub(crate) out_head: u64,
    pub(crate) in_head: u64,
    /// These name the neighbour by *id*, not by `InvPtr` — see [`InlineAdj`].
    pub(crate) inline_out: InlineAdj,
    pub(crate) inline_in: InlineAdj,
}
unsafe impl Invariant for ArenaRecordHead {}

/// One adjacency entry (VERSION 4).
///
/// The neighbour is an [`InvPtr`], which handles both cases in one field:
/// a target in *this* arena gets FOT index 0 — no FOT entry, and
/// `resolve()` takes an inlined base+offset path — while a target elsewhere
/// costs one FOT entry, deduped per target arena by the runtime's
/// `insert_fot`. Traversal therefore never consults the location registry.
///
/// This started as an `AdjRef` and had to change. An `InvPtr`'s FOT index is
/// meaningful only inside its containing object, and a chunk earns one by being
/// allocated through the arena's transaction. The inline slot is written
/// straight into the record with a raw pointer, and cross-arena resolution from
/// there returned garbage: `gstress scale:20000` lost every cross-arena
/// inline entry — clique out-degrees of 0 where 99 were due — while the chunk
/// entries beside them resolved correctly. Same-arena resolution worked, which
/// is why a single-arena test suite never saw it.
///
/// Ids sidestep the question rather than answering it. They also cost less here
/// than pointers did: no FOT entry per degree-1 record, and no generation
/// check — an id is never reused, so a stale one finds a tombstone in `locs`
/// and `is_alive` rejects it. Generations exist to catch a stale *pointer* into
/// a reused slot; an id cannot go stale that way.
///
/// The high-degree path keeps `InvPtr`s in chunks, so index-free adjacency —
/// the mechanism the project is about — is untouched where it matters.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct InlineAdj {
    pub(crate) edge_id: u64,
    pub(crate) neighbor_id: u64,
    pub(crate) label: u32,
    pub(crate) _pad: u32,
}
unsafe impl Invariant for InlineAdj {}

#[repr(C)]
pub(crate) struct AdjRef {
    pub(crate) edge_id: u64,
    pub(crate) neighbor: InvPtr<ArenaRecordHead>,
    pub(crate) label: u32,
    pub(crate) neighbor_gen: u32,
}
unsafe impl Invariant for AdjRef {}

impl AdjRef {
    /// Filler for the unused tail of a freshly allocated chunk. `len` bounds
    /// every read, so these are never resolved.
    fn null() -> Self {
        AdjRef {
            edge_id: 0,
            neighbor: InvPtr::null(),
            label: 0,
            neighbor_gen: 0,
        }
    }
}

/// A chunk of adjacency entries. Chunks are prepended, so walking from the
/// head yields newest-first; readers reverse the chunk order to recover
/// insertion order (entries within a chunk are already in order).
#[repr(C)]
pub(crate) struct AdjChunk {
    pub(crate) len: u32,
    pub(crate) _pad: u32,
    pub(crate) next: u64,
    pub(crate) entries: [AdjRef; ADJ_CHUNK],
}
unsafe impl Invariant for AdjChunk {}

/// Where a vertex lives: which arena, at what offset, and whether it is alive.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct VertexLoc {
    pub(crate) arena: u32,
    pub(crate) flags: u32,
    pub(crate) off: u64,
}
unsafe impl Invariant for VertexLoc {}

/// Directory entry naming one arena object.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct ArenaEntry {
    raw: u128,
}
unsafe impl Invariant for ArenaEntry {}

/// What a placement policy is told about an existing arena.
#[derive(Clone, Copy, Debug)]
pub struct ArenaStat {
    /// Vertices already placed in this arena.
    pub vertices: usize,
}

pub trait Placement {
    /// `Some(i)` to place in existing arena `i`, `None` to open a new one.
    fn place(&mut self, arenas: &[ArenaStat]) -> Option<usize>;
    fn name(&self) -> &'static str;
}

pub struct OnePerArena;
impl Placement for OnePerArena {
    fn place(&mut self, _arenas: &[ArenaStat]) -> Option<usize> {
        None
    }
    fn name(&self) -> &'static str {
        "one-per-arena"
    }
}

pub struct FillTo {
    pub cap: usize,
}
impl Placement for FillTo {
    fn place(&mut self, arenas: &[ArenaStat]) -> Option<usize> {
        match arenas.last() {
            Some(a) if a.vertices < self.cap => Some(arenas.len() - 1),
            _ => None,
        }
    }
    fn name(&self) -> &'static str {
        "fill-to"
    }
}

/// Arena-backed store: a directory of arenas, a vertex-location registry, and
/// a placement policy.
pub struct ArenaStore {
    dir: SegVec<ArenaEntry>,
    locs: SegVec<VertexLoc>,
    open: Vec<ArenaObject>,
    txs: Vec<Option<TxObject<ArenaBase>>>,
    stats: Vec<ArenaStat>,
    /// Which arenas have been touched since the last `sync_all`.
    ///
    /// Conservative on purpose: set by `record_ptr`/`chunk_ptr`, which are the
    /// only ways to obtain a writable pointer into an arena. They serve reads
    /// too, so a read-heavy workload marks arenas it only read and syncs them
    /// needlessly — no worse than the old unconditional behaviour, and never
    /// *unsafe*. Missing a write path would be silent data loss, which is not a
    /// trade worth making for a sync we can afford. A precise version needs
    /// separate read/write pointer accessors; noted, not done.
    dirty: Vec<bool>,
    /// Reclaimed record slots per arena, as `(offset, stride)`.
    ///
    /// In-memory, rebuilt on `open`. Persisting it would mean a second
    /// structure to keep coherent with the records themselves — exactly the
    /// record/mirror split that produced the `resolve`/`resolve_mut` bug — and
    /// it is derivable: a slot is free iff some tombstoned `locs` entry names it
    /// and no live entry does. Deriving costs one pass at open and cannot drift.
    ///
    /// Exact-stride reuse only. A freed 128-byte slot is not offered to a
    /// record wanting 176, and is not split for one wanting 80. Both would work
    /// but both need a size-class scheme, and today nearly every record is
    /// `record_size(0)` — vertices default to no inline slots and edges always
    /// have none — so exact match covers the common case at no complexity. If a
    /// profile ever shows churn on mixed widths, that is when to generalise.
    free: Vec<Vec<(u64, usize)>>,
    policy: Box<dyn Placement>,
    /// Syncs issued by `sync_all`, so a test can assert the batching property
    /// directly rather than inferring it from wall time.
    syncs: usize,
    /// Both criteria were written as claims about *mechanism* ("performs no
    /// second resolution", "assert the scan's cost"), which nothing could
    /// check: they would have passed by inspection, which is how the
    /// `resolve`/`resolve_mut` incoherence survived five days of a green suite.
    /// A counter makes them falsifiable.
    ///
    /// `Cell` because `record_ptr` takes `&self`; the store is single-threaded
    /// per handle, which is the same contract the raw pointers already rely on.
    #[cfg(test)]
    record_touches: core::cell::Cell<usize>,
    #[cfg(test)]
    data_block_reads: core::cell::Cell<usize>,
    /// Adjacency counters. Test-only, like `record_touches`: they sit on the
    /// hottest path in the engine.
    #[cfg(test)]
    pub(crate) diag: core::cell::Cell<AdjDiag>,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct AdjDiag {
    /// Entries visited by `walk_adj`, before any filtering.
    pub seen: usize,
    /// …of which came from a record's inline slot rather than a chunk.
    pub inline: usize,
    /// …whose neighbour `InvPtr` is not FOT-index 0, i.e. resolves through
    /// `slow_resolve` into a different mapping of another arena.
    pub cross_arena: usize,
    /// …dropped because the neighbour's slot generation did not match.
    pub skipped_gen: usize,
    /// …dropped because the neighbour was tombstoned.
    pub skipped_dead: usize,
    /// Of the generation-skipped entries, how many were cross-arena. If this
    /// equals `skipped_gen`, every skip is explained by the mapping split.
    pub skipped_gen_cross: usize,
}

impl ArenaStore {
    pub fn create(policy: Box<dyn Placement>, seg_cap: usize) -> Result<Self> {
        Ok(ArenaStore {
            dir: SegVec::create(seg_cap)?,
            locs: SegVec::create(seg_cap)?,
            open: Vec::new(),
            txs: Vec::new(),
            stats: Vec::new(),
            dirty: Vec::new(),
            free: Vec::new(),
            policy,
            syncs: 0,
            #[cfg(test)]
            record_touches: core::cell::Cell::new(0),
            #[cfg(test)]
            data_block_reads: core::cell::Cell::new(0),
            #[cfg(test)]
            diag: core::cell::Cell::new(AdjDiag::default()),
        })
    }

    /// Object ids of the directory and location registry — persist these to
    /// reopen the store.
    pub fn ids(&self) -> (u128, u128) {
        (self.dir.dir_raw(), self.locs.dir_raw())
    }

    pub fn open(
        dir_raw: u128,
        locs_raw: u128,
        policy: Box<dyn Placement>,
        seg_cap: usize,
    ) -> Result<Self> {
        let dir: SegVec<ArenaEntry> = SegVec::open(dir_raw, seg_cap)?;
        let locs: SegVec<VertexLoc> = SegVec::open(locs_raw, seg_cap)?;
        let mut open = Vec::with_capacity(dir.len());
        for i in 0..dir.len() {
            let raw = dir.get_ref(i).map(|e| e.raw).unwrap_or(0);
            open.push(ArenaObject::from_objid(ObjID::new(raw))?);
        }
        // Consistency check on reopen. `locs` names arenas by position, so a
        // registry describing vertices in arenas the directory does not list is
        // unusable — every lookup would index past the end. Refuse here, where
        // the cause is visible, rather than panicking later in whatever path
        // happens to touch a vertex first.
        if locs.len() > 0 && open.is_empty() {
            eprintln!(
                "twizzler-graph: arena store inconsistent — {} vertices in registry, \
                 0 arenas in directory (dir={dir_raw:#x} locs={locs_raw:#x}); refusing to open",
                locs.len()
            );
            return Err(twizzler_rt_abi::error::ArgumentError::InvalidArgument.into());
        }

        // Rebuild per-arena counts from the location registry.
        let mut stats = vec![ArenaStat { vertices: 0 }; open.len()];
        let mut stray = 0usize;
        for i in 0..locs.len() {
            if let Some(l) = locs.get_ref(i) {
                match stats.get_mut(l.arena as usize) {
                    Some(s) => s.vertices += 1,
                    None => stray += 1,
                }
            }
        }
        if stray > 0 {
            eprintln!(
                "twizzler-graph: arena store has {stray} vertices naming arenas beyond the \
                 directory's {} entries (dir={dir_raw:#x})",
                open.len()
            );
        }
        // Rebuild the free list (see the field's doc for why it is derived
        // rather than persisted). A tombstoned entry's offset is free unless a
        // live entry also names it — which happens exactly when the slot was
        // already reused, in which case the live record owns it.
        let mut free: Vec<Vec<(u64, usize)>> = (0..open.len()).map(|_| Vec::new()).collect();
        {
            let mut taken: Vec<(u32, u64)> = Vec::new();
            for i in 0..locs.len() {
                if let Some(l) = locs.get_ref(i) {
                    if l.flags & TOMBSTONE == 0 {
                        taken.push((l.arena, l.off));
                    }
                }
            }
            for i in 0..locs.len() {
                let Some(l) = locs.get_ref(i).map(|l| *l) else {
                    continue;
                };
                if l.flags & TOMBSTONE == 0 || taken.contains(&(l.arena, l.off)) {
                    continue;
                }
                let Some(a) = open.get(l.arena as usize) else {
                    continue;
                };
                // The dead record still describes its own extent.
                let Some(p) = a
                    .object()
                    .lea(l.off as usize, size_of::<ArenaRecordHead>())
                else {
                    continue;
                };
                let nprops = unsafe { (*(p as *const ArenaRecordHead)).nprops };
                if let Some(slots) = free.get_mut(l.arena as usize) {
                    slots.push((l.off, record_size(nprops)));
                }
            }
        }
        let open_len = open.len();
        let txs = (0..open.len()).map(|_| None).collect();
        Ok(ArenaStore {
            dir,
            locs,
            open,
            txs,
            stats,
            dirty: vec![false; open_len],
            free,
            policy,
            syncs: 0,
            #[cfg(test)]
            record_touches: core::cell::Cell::new(0),
            #[cfg(test)]
            data_block_reads: core::cell::Cell::new(0),
            #[cfg(test)]
            diag: core::cell::Cell::new(AdjDiag::default()),
        })
    }

    /// Every object this store owns: the arena directory, the location
    /// registry, each arena, and each vertex's property object.
    ///
    /// Walking the vertices to collect `props_raw` pages the arenas in, which
    /// is wasted work if the caller is not about to free them — so this is for
    /// teardown only. It is cheap in the way that matters: arena count is
    /// `vertices/cap`, not `vertices`.
    pub fn owned_object_ids(&self) -> Vec<u128> {
        let mut ids = self.dir.object_ids();
        ids.extend(self.locs.object_ids());
        ids.extend(self.open.iter().map(|a| a.object().id().raw()));
        ids
    }

    pub fn arena_count(&self) -> usize {
        self.open.len()
    }

    pub fn record_count(&self) -> usize {
        self.locs.len()
    }

    pub fn policy_name(&self) -> &'static str {
        self.policy.name()
    }

    /// Diagnostic: vertices per arena as the placement policy sees them
    /// (`stats`), beside the same counts recomputed from the location
    /// registry (ground truth). Returns `(policy, actual)`.
    ///
    /// `place` decides rollover from `stats` alone, so if the two columns
    /// disagree the cap is not doing what it says. `ArenaStore::open` rebuilds
    /// `stats` from `locs` and counts tombstoned vertices as live, so the
    /// harness's mid-run `E:reopen` is the first place to look.
    pub fn arena_vertex_counts(&self) -> (Vec<usize>, Vec<usize>) {
        let policy: Vec<usize> = self.stats.iter().map(|s| s.vertices).collect();
        let mut actual = vec![0usize; self.open.len()];
        for i in 0..self.locs.len() {
            if let Some(l) = self.locs.get_ref(i) {
                if let Some(c) = actual.get_mut(l.arena as usize) {
                    *c += 1;
                }
            }
        }
        (policy, actual)
    }

    fn new_arena(&mut self) -> Result<usize> {
        let arena = ArenaObject::new(ObjectBuilder::default().persist(true))?;
        let raw = arena.object().id().raw();
        self.dir.push_nosync(ArenaEntry { raw })?;
        self.open.push(arena);
        self.txs.push(None);
        self.stats.push(ArenaStat { vertices: 0 });
        self.dirty.push(true); // a fresh arena has a base to write out
        self.free.push(Vec::new());
        Ok(self.open.len() - 1)
    }

    fn tx_for(&mut self, idx: usize) -> Result<&mut TxObject<ArenaBase>> {
        if self.txs[idx].is_none() {
            let tx = self.open[idx].as_tx()?;
            self.txs[idx] = Some(tx);
        }
        Ok(self.txs[idx].as_mut().expect("just opened"))
    }

    pub fn sync_count(&self) -> usize {
        self.syncs
    }

    /// Bounds-checked arena lookup, for paths reading a *persisted* index.
    fn try_arena_id(&self, idx: usize) -> Option<ObjID> {
        self.open.get(idx).map(|a| a.object().id())
    }

    /// A `GlobalPtr` naming a vertex record — an `(ObjID, offset)` pair, used
    /// where one is *required* rather than resolved: `InvPtr::new` needs a
    /// global address to build an adjacency entry against. Never resolve one
    /// of these to touch a record; go through [`Self::record_ptr`].
    fn vertex_ptr(&self, id: u64) -> Option<GlobalPtr<ArenaRecordHead>> {
        let loc = self.locs.get_ref(id as usize)?;
        let aid = self.try_arena_id(loc.arena as usize)?;
        Some(GlobalPtr::new(aid, loc.off))
    }

    // --- record access: one mapping per arena --------------------------------
    //
    // Every read and write of an arena record goes through the two helpers
    // below, and none through `GlobalPtr::resolve`/`resolve_mut`.
    //
    // `resolve` maps its object `READ`; `resolve_mut` maps it
    // `READ | WRITE | PERSIST` (`ptr/global.rs`). Different flags, so
    // `twz_rt_map_object` returns different mappings, and a write through one
    // is not visible through the other. That cost us 2 858 silently-undeleted
    // vertices on `scale:20000` — see `delete_vertex`.
    //
    // `ArenaObject::from_objid` already maps `READ | WRITE | PERSIST`, so the
    // handle in `self.open` *is* the write mapping, and `lea`/`lea_mut` are
    // plain `handle().start() + offset` against it. Reads and writes therefore
    // land in the same pages by construction. It is also cheaper: `resolve()`
    // calls `twz_rt_map_object` on every single access, and these do not.
    //
    // Both return `*mut` regardless of intent so that callers share one path;
    // the borrow discipline is the same single-threaded-per-handle contract
    // the rest of this file runs on.

    /// Raw pointer to a vertex record, inside its arena's own mapping.
    fn record_ptr(&self, loc: &VertexLoc) -> Option<*mut ArenaRecordHead> {
        #[cfg(test)]
        self.record_touches.set(self.record_touches.get() + 1);
        self.mark_dirty(loc.arena as usize);
        let obj = self.open.get(loc.arena as usize)?.object();
        obj.lea_mut(loc.off as usize, size_of::<ArenaRecordHead>())
            .map(|p| p as *mut ArenaRecordHead)
    }

    /// Records touched since the last [`Self::reset_record_touches`].
    ///
    /// Counts *attempts*, incremented before the bounds check, so a lookup that
    /// fails still registers. A path that "does not touch records" must not be
    /// reaching this function at all — counting only successes would let a
    /// miss-heavy path look clean.
    #[cfg(test)]
    pub(crate) fn record_touches(&self) -> usize {
        self.record_touches.get()
    }

    #[cfg(test)]
    pub(crate) fn reset_record_touches(&self) {
        self.record_touches.set(0);
        self.data_block_reads.set(0);
    }

    #[cfg(test)]
    pub(crate) fn data_block_reads(&self) -> usize {
        self.data_block_reads.get()
    }

    /// Raw pointer to an adjacency chunk, inside its arena's own mapping.
    fn chunk_ptr(&self, arena: u32, off: u64) -> Option<*mut AdjChunk> {
        self.mark_dirty(arena as usize);
        let obj = self.open.get(arena as usize)?.object();
        obj.lea_mut(off as usize, size_of::<AdjChunk>())
            .map(|p| p as *mut AdjChunk)
    }

    /// The location entry for a vertex that is live, in one `locs` read.
    ///
    /// Liveness is the mirror's call, not the record's. That was forced when
    /// record writes were invisible; it stays now that they are not, because
    /// a *cross-arena* neighbour is still reached through `InvPtr::resolve`,
    /// which maps `READ | INDIRECT` and so has the original problem. Reading
    /// liveness from the mirror keeps every path on one answer.
    fn live_loc(&self, vertex: u64) -> Option<VertexLoc> {
        let loc = self.locs.get_ref(vertex as usize).map(|l| *l)?;
        if loc.flags & TOMBSTONE != 0 {
            return None;
        }
        Some(loc)
    }

    /// Add a vertex with no inline traversal properties.
    pub fn add_vertex(&mut self, label: u32, name: &str, target_raw: u128) -> Result<u64> {
        self.add_record(label, name, target_raw, &[], false)
    }

    /// Add a record — vertex or edge — with `props` inline traversal slots.
    ///
    /// `props.len()` is fixed here and for the record's lifetime. Not a
    /// policy choice: a record has O(in-degree) inbound `AdjRef.neighbor`
    /// `InvPtr`s, each holding its arena offset, so growing one would mean
    /// rewriting every inbound pointer with durability ordering to get right.
    /// Anything added later becomes a *data* property, whose block has exactly
    /// one referent and can therefore move freely.
    /// `pub(crate)`, unlike [`Self::add_vertex`]: it takes [`PropSlot`], which is
    /// a *persisted layout* type. Exposing it would publish the record format as
    /// API and commit us to its field order. The public route to inline
    /// properties is `Graph::add_vertex_with_props`, which speaks `&str` and
    /// `PropValue`.
    pub(crate) fn add_record(
        &mut self,
        label: u32,
        name: &str,
        target_raw: u128,
        props: &[PropSlot],
        is_edge: bool,
    ) -> Result<u64> {
        let nprops = u16::try_from(props.len())
            .map_err(|_| twizzler_rt_abi::error::ArgumentError::InvalidArgument)?;
        // Reuse before placement. The policy counts records *ever* allocated,
        // so it considers an arena full even when a delete has freed a slot in
        // it — ask it first and it opens a new arena while reclaimed space sits
        // unused. This overrides the policy only where the alternative is
        // growth, and only for an exact stride match.
        //
        // Deliberate tension worth naming: placement stops being purely
        // policy-driven, so a future locality-aware policy will sometimes be
        // overruled by "wherever a slot happened to free up". That is the right
        // default while the alternative is unbounded arena growth, but a policy
        // that cares about locality should be able to decline a reclaimed slot.
        // `Placement` has no way to express that yet.
        let want = record_size(nprops);
        let reusable = self
            .free
            .iter()
            .position(|slots| slots.iter().any(|(_, sz)| *sz == want));
        let idx = match reusable {
            Some(i) => i,
            None => match self.policy.place(&self.stats) {
                Some(i) if i < self.open.len() => i,
                _ => self.new_arena()?,
            },
        };
        let id = self.locs.len() as u64;
        // Not `tx.alloc(head)`, which reserves `Layout::new::<T>()` and so can
        // only ever place a record with zero inline slots. Routing the ordinary
        // vertex path through the variable-stride allocator now — at `nprops =
        // 0`, where it is equivalent — means the whole test suite and `gstress`
        // exercise it, rather than it sitting untested beside a working path
        // until the day it is switched on.
        let (off, generation) = self.alloc_record_bytes(idx, nprops)?;
        {
            let obj = self.open[idx].object();
            let base = obj
                .lea_mut(off as usize, record_size(nprops))
                .ok_or(twizzler_rt_abi::error::ArgumentError::InvalidArgument)?;
            let p = base as *mut ArenaRecordHead;
            unsafe {
                (*p).id = id;
                (*p).flags = 0;
                (*p).label = label;
                (*p).nprops = nprops;
                (*p).generation = generation;
                (*p).name = NameKey::new(name);
                (*p).target_raw = target_raw;
                (*p).data_props = 0;
                (*p).out_head = 0;
                (*p).in_head = 0;
                if is_edge {
                    (*p).flags |= IS_EDGE;
                }
                // Slots follow the head contiguously. `record_stride_has_no_
                // hidden_padding` is what makes this pointer arithmetic sound.
                let slots = base.add(size_of::<ArenaRecordHead>()) as *mut PropSlot;
                for (i, s) in props.iter().enumerate() {
                    slots.add(i).write(*s);
                }
            }
        }
        self.locs.push_nosync(VertexLoc {
            arena: idx as u32,
            flags: if is_edge { IS_EDGE } else { 0 },
            off,
        })?;
        self.stats[idx].vertices += 1;
        Ok(id)
    }

    /// A live record's inline traversal slots.
    pub(crate) fn traversal_props(&self, vertex: u64) -> Option<Vec<PropSlot>> {
        let loc = self.live_loc(vertex)?;
        let p = self.record_ptr(&loc)?;
        let n = unsafe { (*p).nprops } as usize;
        if n == 0 {
            return Some(Vec::new());
        }
        let obj = self.open.get(loc.arena as usize)?.object();
        let base = obj.lea(loc.off as usize, record_size(n as u16))?;
        let slots = unsafe { base.add(size_of::<ArenaRecordHead>()) } as *const PropSlot;
        Some((0..n).map(|i| unsafe { *slots.add(i) }).collect())
    }

    /// One inline traversal property by interned key.
    pub(crate) fn traversal_prop(&self, vertex: u64, key_id: u32) -> Option<PropValue> {
        self.traversal_props(vertex)?
            .into_iter()
            .find(|s| s.key_id == key_id)
            .map(|s| s.val)
    }

    /// Update an existing inline slot in place. Returns whether the key was
    /// found; `false` means the caller should store it as a data property.
    ///
    /// Updating is safe where adding is not. A slot is fixed-size, so
    /// overwriting its value moves nothing and no inbound `AdjRef.neighbor`
    /// offset changes. It is *adding* a key that would grow the record, which is
    /// the thing the format forbids.
    ///
    /// This exists to keep one source of truth. Without it, `set_vertex_prop` on
    /// a key that happens to be inline would write the side object while readers
    /// still saw the inline slot — a silent divergence, and indistinguishable
    /// from an unset property from the outside.
    pub(crate) fn set_traversal_prop(
        &mut self,
        vertex: u64,
        key_id: u32,
        val: PropValue,
    ) -> Option<bool> {
        let loc = self.live_loc(vertex)?;
        let p = self.record_ptr(&loc)?;
        let n = unsafe { (*p).nprops } as usize;
        if n == 0 {
            return Some(false);
        }
        let obj = self.open.get(loc.arena as usize)?.object();
        let base = obj.lea_mut(loc.off as usize, record_size(n as u16))?;
        let slots = unsafe { base.add(size_of::<ArenaRecordHead>()) } as *mut PropSlot;
        for i in 0..n {
            unsafe {
                if (*slots.add(i)).key_id == key_id {
                    (*slots.add(i)).val = val;
                    return Some(true);
                }
            }
        }
        Some(false)
    }

    pub(crate) fn data_props(&self, vertex: u64) -> Option<Vec<PropSlot>> {
        let loc = self.live_loc(vertex)?;
        let p = self.record_ptr(&loc)?;
        let off = unsafe { (*p).data_props };
        if off == 0 {
            return Some(Vec::new());
        }
        #[cfg(test)]
        self.data_block_reads.set(self.data_block_reads.get() + 1);
        let obj = self.open.get(loc.arena as usize)?.object();
        let hp = obj.lea(off as usize, size_of::<DataBlockHead>())? as *const DataBlockHead;
        let (len, cap) = unsafe { ((*hp).len, (*hp).cap) };
        let base = obj.lea(off as usize, data_block_size(cap))?;
        let slots = unsafe { base.add(size_of::<DataBlockHead>()) } as *const PropSlot;
        Some((0..len as usize).map(|i| unsafe { *slots.add(i) }).collect())
    }

    /// Set a data property, allocating or growing the block as needed.
    pub(crate) fn set_data_prop(&mut self, vertex: u64, key_id: u32, val: PropValue) -> Result<()> {
        let loc = self
            .live_loc(vertex)
            .ok_or(twizzler_rt_abi::error::ArgumentError::InvalidArgument)?;
        let arena = loc.arena as usize;
        let off = {
            let p = self
                .record_ptr(&loc)
                .ok_or(twizzler_rt_abi::error::ArgumentError::InvalidArgument)?;
            unsafe { (*p).data_props }
        };

        // Existing block: update in place, or append if there is room.
        if off != 0 {
            let obj = self.open[arena].object();
            let hp = obj
                .lea_mut(off as usize, size_of::<DataBlockHead>())
                .ok_or(twizzler_rt_abi::error::ArgumentError::InvalidArgument)?
                as *mut DataBlockHead;
            let (len, cap) = unsafe { ((*hp).len, (*hp).cap) };
            let base = obj
                .lea_mut(off as usize, data_block_size(cap))
                .ok_or(twizzler_rt_abi::error::ArgumentError::InvalidArgument)?;
            let slots = unsafe { base.add(size_of::<DataBlockHead>()) } as *mut PropSlot;
            for i in 0..len as usize {
                unsafe {
                    if (*slots.add(i)).key_id == key_id {
                        (*slots.add(i)).val = val;
                        return Ok(());
                    }
                }
            }
            if len < cap {
                unsafe {
                    slots.add(len as usize).write(PropSlot {
                        key_id,
                        _pad: 0,
                        val,
                    });
                    (*hp).len = len + 1;
                }
                return Ok(());
            }
        }

        // No block, or it is full: allocate a bigger one and copy.
        let existing = self.data_props(vertex).unwrap_or_default();
        let new_cap = if existing.is_empty() {
            4
        } else {
            (existing.len() as u16).saturating_mul(2)
        };
        let zeros = vec![0u8; data_block_size(new_cap)];
        let new_off = {
            let tx = self.tx_for(arena)?;
            tx.alloc_with_slice::<u8>(&zeros)?.1.offset()
        };
        {
            let obj = self.open[arena].object();
            let base = obj
                .lea_mut(new_off as usize, data_block_size(new_cap))
                .ok_or(twizzler_rt_abi::error::ArgumentError::InvalidArgument)?;
            let hp = base as *mut DataBlockHead;
            let slots = unsafe { base.add(size_of::<DataBlockHead>()) } as *mut PropSlot;
            unsafe {
                for (i, s) in existing.iter().enumerate() {
                    slots.add(i).write(*s);
                }
                slots.add(existing.len()).write(PropSlot {
                    key_id,
                    _pad: 0,
                    val,
                });
                (*hp).len = existing.len() as u16 + 1;
                (*hp).cap = new_cap;
            }
        }
        // Repoint the record last: until this write lands, the record still
        // names the old block, so an interruption leaks a block rather than
        // leaving the record pointing at a half-built one.
        let p = self
            .record_ptr(&loc)
            .ok_or(twizzler_rt_abi::error::ArgumentError::InvalidArgument)?;
        unsafe { (*p).data_props = new_off };
        Ok(())
    }

    pub(crate) fn live_record(&self, vertex: u64) -> Option<()> {
        self.live_loc(vertex).map(|_| ())
    }

    /// Read and clear the adjacency diagnostic counters.
    #[cfg(test)]
    pub fn take_diag(&self) -> AdjDiag {
        let d = self.diag.get();
        self.diag.set(AdjDiag::default());
        d
    }

    /// Note an arena as needing a sync. `&self` because the pointer accessors
    /// that call it are `&self`; `Cell` would be tidier but `Vec<bool>` is
    /// indexed hot and this stays a plain write behind an existing borrow.
    fn mark_dirty(&self, idx: usize) {
        // Interior mutability via raw pointer is deliberate and local: `dirty`
        // is engine bookkeeping, never persisted, and is only ever set to
        // `true` here and cleared under `&mut self` in `sync_all`.
        if idx < self.dirty.len() {
            unsafe {
                let p = self.dirty.as_ptr() as *mut bool;
                *p.add(idx) = true;
            }
        }
    }

    /// A slot's current generation, for stamping into an `AdjRef`.
    fn generation_of(&self, vertex: u64) -> Option<u32> {
        let loc = self.live_loc(vertex)?;
        let p = self.record_ptr(&loc)?;
        Some(unsafe { (*p).generation })
    }

    /// Whether a record is an edge. Reads the mirror, not the record — the
    /// point of duplicating the bit into `VertexLoc.flags`.
    pub(crate) fn is_edge(&self, vertex: u64) -> bool {
        self.locs
            .get_ref(vertex as usize)
            .map_or(false, |l| l.flags & IS_EDGE != 0)
    }

    /// # Why a byte slice and not `alloc_inplace`
    ///
    /// A record's size depends on `nprops`, which is a runtime value, and the
    /// arena's typed path cannot express that:
    ///
    /// Alignment is safe but incidental, and worth knowing: `Layout::array
    /// ::<u8>` has align 1, yet `ArenaBase::reserve` raises every allocation to
    /// `MIN_ALIGN = 16`, which is what `ArenaRecordHead` and `PropSlot` need.
    /// We depend on that floor. If upstream ever lowers `MIN_ALIGN`, records
    /// become misaligned — hence `record_head_alignment_holds` in the layout
    /// tests, which fails loudly rather than letting reads go sideways.
    ///
    /// The bytes are zeroed, so a record is fully initialised before any field
    /// is written and padding never carries stale arena contents to disk.
    fn alloc_record_bytes(&mut self, idx: usize, nprops: u16) -> Result<(u64, u32)> {
        let want = record_size(nprops);
        // Reclaimed slot first — this is what stops churn growing arenas
        // without bound. The caller bumps `generation` on the reused slot, which
        // is what invalidates any `AdjRef` still pointing here.
        if let Some(slots) = self.free.get_mut(idx) {
            if let Some(pos) = slots.iter().position(|(_, sz)| *sz == want) {
                let (off, _) = slots.swap_remove(pos);
                // Read the old generation before zeroing, and hand back its
                // successor. Zeroing would otherwise reset it to 0 and every
                // stale `AdjRef` pointing here — which expects 0 for a slot that
                // has never been reused — would match again. That single
                // ordering mistake would silently undo the whole mechanism.
                let mut next_gen = 1;
                if let Some(p) = self.open[idx].object().lea_mut(off as usize, want) {
                    unsafe {
                        next_gen = (*(p as *const ArenaRecordHead)).generation.wrapping_add(1);
                        core::ptr::write_bytes(p, 0, want);
                    }
                }
                return Ok((off, next_gen));
            }
        }
        let zeros = vec![0u8; want];
        let tx = self.tx_for(idx)?;
        // Generation 0: a slot handed out for the first time has never been
        // reused, so nothing can hold a stale reference to it.
        Ok((tx.alloc_with_slice::<u8>(&zeros)?.1.offset(), 0))
    }

    /// Append an adjacency entry to `vertex`'s out (or in) chain. The chunk is
    /// allocated in the vertex's own arena, so a low-degree vertex adds no
    /// objects at all.
    ///
    /// The neighbour is given as a location rather than a built [`AdjRef`]:
    /// the entry's `InvPtr` can only be constructed against the transaction of
    /// the arena that will *hold* it, since a FOT index is object-relative.
    /// Building it here is what makes a same-arena neighbour cost no FOT entry.
    fn append_adj(
        &mut self,
        vertex: u64,
        out: bool,
        edge_id: u64,
        label: u32,
        neighbor: GlobalPtr<ArenaRecordHead>,
        neighbor_id: u64,
        neighbor_gen: u32,
    ) -> Result<()> {
        let Some(loc) = self.locs.get_ref(vertex as usize).map(|l| *l) else {
            return Err(twizzler_rt_abi::error::ArgumentError::InvalidArgument.into());
        };
        let arena_idx = loc.arena as usize;
        let Some(vp) = self.record_ptr(&loc) else {
            return Err(twizzler_rt_abi::error::ArgumentError::InvalidArgument.into());
        };

        let head = unsafe {
            if out {
                (*vp).out_head
            } else {
                (*vp).in_head
            }
        };

        // Note this runs before the `AdjRef` is built, so a degree-1 record
        // also costs no FOT entry — `InvPtr::new` is never called for it.
        {
            let bit = if out { HAS_INLINE_OUT } else { HAS_INLINE_IN };
            let occupied = unsafe { (*vp).flags & bit != 0 };
            if !occupied {
                unsafe {
                    let dst = if out {
                        &raw mut (*vp).inline_out
                    } else {
                        &raw mut (*vp).inline_in
                    };
                    dst.write(InlineAdj {
                        edge_id,
                        neighbor_id,
                        label,
                        _pad: 0,
                    });
                    (*vp).flags |= bit;
                }
                return Ok(());
            }
        }

        // Build the entry against this arena's transaction. `InvPtr::new`
        // returns FOT index 0 when `neighbor` lives in this same arena.
        let entry = {
            let tx = self.tx_for(arena_idx)?;
            AdjRef {
                edge_id,
                neighbor: InvPtr::new(&*tx, neighbor)?,
                label,
                neighbor_gen,
            }
        };

        // Room in the head chunk? Append there and we are done.
        if head != 0 {
            if let Some(cp) = self.chunk_ptr(loc.arena, head) {
                let c = unsafe { &mut *cp };
                if (c.len as usize) < ADJ_CHUNK {
                    let n = c.len as usize;
                    c.entries[n] = entry;
                    c.len += 1;
                    return Ok(());
                }
            }
        }

        // Otherwise allocate a fresh chunk in the same arena and link it in.
        // `from_fn` rather than an array-repeat literal because `AdjRef` is not
        // `Copy` (see its docs); indices are visited in order, so `take` at 0 is
        // the single move of `entry`.
        let mut slot = Some(entry);
        let entries: [AdjRef; ADJ_CHUNK] = core::array::from_fn(|i| {
            if i == 0 {
                slot.take().expect("index 0 is visited exactly once")
            } else {
                AdjRef::null()
            }
        });
        let new_off = {
            let tx = self.tx_for(arena_idx)?;
            tx.alloc(AdjChunk {
                len: 1,
                _pad: 0,
                next: head,
                entries,
            })?
            .offset()
        };
        // `vp` is a raw pointer into the arena's mapping, so it survives the
        // `&mut self` borrows `tx_for` needs above, and the mapping base does
        // not move when the arena grows.
        unsafe {
            if out {
                (*vp).out_head = new_off;
            } else {
                (*vp).in_head = new_off;
            }
        }
        Ok(())
    }

    /// Record an edge on both endpoints.
    pub fn add_edge(&mut self, from: u64, to: u64, edge_id: u64, label: u32) -> Result<()> {
        let Some(from_gp) = self.vertex_ptr(from) else {
            return Err(twizzler_rt_abi::error::ArgumentError::InvalidArgument.into());
        };
        let Some(to_gp) = self.vertex_ptr(to) else {
            return Err(twizzler_rt_abi::error::ArgumentError::InvalidArgument.into());
        };
        let (Some(fg), Some(tg)) = (self.generation_of(from), self.generation_of(to)) else {
            return Err(twizzler_rt_abi::error::ArgumentError::InvalidArgument.into());
        };
        self.append_adj(from, true, edge_id, label, to_gp, to, tg)?;
        self.append_adj(to, false, edge_id, label, from_gp, from, fg)
    }

    /// The topology becomes `from → edge → to`, with the edge record carrying
    /// its own adjacency, rather than `from → to` with the edge id riding along
    /// in the entry. That is what makes edge properties identical to vertex
    /// properties and hyperedges need no new machinery — an edge record with
    /// several out-links simply *is* a hyperedge.
    ///
    /// Four links, not two, and each is one direction of one hop:
    ///
    /// | chain | entry points at |
    /// |---|---|
    /// | `from` out | the edge record |
    /// | edge out | `to` |
    /// | `to` in | the edge record |
    /// | edge in | `from` |
    pub(crate) fn add_edge_record(&mut self, label: u32, from: u64, to: u64) -> Result<u64> {
        if self.live_loc(from).is_none() || self.live_loc(to).is_none() {
            return Err(twizzler_rt_abi::error::ArgumentError::InvalidArgument.into());
        }
        // The record first: the links below need its location to exist.
        let edge = self.add_record(label, "", 0, &[], true)?;

        let (Some(from_gp), Some(to_gp), Some(edge_gp)) = (
            self.vertex_ptr(from),
            self.vertex_ptr(to),
            self.vertex_ptr(edge),
        ) else {
            return Err(twizzler_rt_abi::error::ArgumentError::InvalidArgument.into());
        };

        let (Some(fg), Some(tg), Some(eg)) = (
            self.generation_of(from),
            self.generation_of(to),
            self.generation_of(edge),
        ) else {
            return Err(twizzler_rt_abi::error::ArgumentError::InvalidArgument.into());
        };
        self.append_adj(from, true, edge, label, edge_gp, edge, eg)?;
        self.append_adj(to, false, edge, label, edge_gp, edge, eg)?;
        self.append_adj(edge, true, edge, label, to_gp, to, tg)?;
        self.append_adj(edge, false, edge, label, from_gp, from, fg)?;
        Ok(edge)
    }

    /// Attach a further participant to an existing edge record, making it a
    /// hyperedge.
    ///
    /// The label is taken from the edge record so participants cannot disagree
    /// about what edge they are on.
    pub(crate) fn add_edge_endpoint(&mut self, edge: u64, vertex: u64, out: bool) -> Result<()> {
        if !self.is_edge(edge) || self.live_loc(vertex).is_none() {
            return Err(twizzler_rt_abi::error::ArgumentError::InvalidArgument.into());
        }
        let label = self
            .vertex_label(edge)
            .ok_or(twizzler_rt_abi::error::ArgumentError::InvalidArgument)?;
        let (Some(v_gp), Some(e_gp)) = (self.vertex_ptr(vertex), self.vertex_ptr(edge)) else {
            return Err(twizzler_rt_abi::error::ArgumentError::InvalidArgument.into());
        };
        let (Some(vg), Some(eg)) = (self.generation_of(vertex), self.generation_of(edge)) else {
            return Err(twizzler_rt_abi::error::ArgumentError::InvalidArgument.into());
        };
        self.append_adj(edge, out, edge, label, v_gp, vertex, vg)?;
        self.append_adj(vertex, !out, edge, label, e_gp, edge, eg)
    }

    /// An edge record's `(from, to)`, or `None` if `edge` is not a live edge.
    ///
    /// Reads the edge's own chains: its in-chain names the source, its out-chain
    /// the target. A hyperedge has several of each; this returns the first of
    /// each, so callers wanting the general shape must walk instead.
    pub(crate) fn edge_endpoints(&self, edge: u64) -> Option<(u64, u64)> {
        if !self.is_edge(edge) {
            return None;
        }
        self.live_loc(edge)?;
        let mut from = None;
        let mut to = None;
        self.walk_adj(edge, false, |_, _, nb| {
            if from.is_none() {
                from = Some(nb)
            }
        });
        self.walk_adj(edge, true, |_, _, nb| {
            if to.is_none() {
                to = Some(nb)
            }
        });
        Some((from?, to?))
    }

    /// `(edge record id, label, far vertex id)` for a vertex's incident edges —
    /// the two-hop walk that replaces the old one-hop entry.
    ///
    /// Hyperedges yield one tuple per far endpoint, which is why this is a
    /// nested walk rather than a lookup: an edge record with three out-links is
    /// three neighbours through one edge, and nothing special has to know that.
    pub(crate) fn neighbors_via_edges(&self, vertex: u64, out: bool, inc: bool) -> Vec<(u64, u32, u64)> {
        let mut edges = Vec::new();
        if out {
            self.walk_adj(vertex, true, |e, l, _| edges.push((e, l, true)));
        }
        if inc {
            self.walk_adj(vertex, false, |e, l, _| edges.push((e, l, false)));
        }
        let mut out_v = Vec::new();
        for (e, l, forward) in edges {
            // Follow the edge record onward: an out-edge continues down the
            // edge's out-chain, an in-edge back down its in-chain. Walking the
            // *matching* direction is what makes this correct — no filtering of
            // the source is needed, and none is done, because a self-loop's far
            // endpoint legitimately *is* the source and must be returned.
            self.walk_adj(e, forward, |_, _, far| out_v.push((e, l, far)));
        }
        out_v
    }

    /// Walk `vertex`'s out (or in) chain in insertion order, calling
    /// `f(edge_id, label, neighbour_id)` for each live entry.
    fn walk_adj(&self, vertex: u64, out: bool, mut f: impl FnMut(u64, u32, u64)) {
        let Some(loc) = self.live_loc(vertex) else {
            return;
        };
        let Some(vp) = self.record_ptr(&loc) else {
            return;
        };
        let inline_bit = if out { HAS_INLINE_OUT } else { HAS_INLINE_IN };
        let mut inline: Option<(u64, u32, u64)> = None;
        unsafe {
            if (*vp).flags & inline_bit != 0 {
                let e = if out { &(*vp).inline_out } else { &(*vp).inline_in };
                #[cfg(test)]
                let mut d = self.diag.get();
                #[cfg(test)]
                {
                    d.seen += 1;
                    d.inline += 1;
                }
                // No resolve and no generation check: the neighbour is named by
                // id, and `is_alive` reads the flat `locs` array through its own
                // coherent mapping. Nothing here depends on a cross-arena view
                // of another record, which is what the `InvPtr` version got
                // wrong.
                if !self.is_alive(e.neighbor_id) {
                    #[cfg(test)]
                    {
                        d.skipped_dead += 1;
                    }
                } else {
                    inline = Some((e.edge_id, e.label, e.neighbor_id));
                }
                #[cfg(test)]
                self.diag.set(d);
            }
        }
        if let Some((eid, lbl, nb)) = inline {
            f(eid, lbl, nb);
        }

        let mut off = unsafe {
            if out {
                (*vp).out_head
            } else {
                (*vp).in_head
            }
        };

        // Bounded walk. A malformed `next` — a cycle, or an offset misread from
        // a stale on-disk layout — would otherwise spin here forever with no
        // output, which is the worst failure mode we have: no panic, no log,
        // nothing to attribute it to. The bound converts that into a loud,
        // located failure. `vertex_count` is a true upper bound because a chunk
        // is only ever allocated by `append_adj`, at most one per entry.
        let max_chunks = self.locs.len() + 2;
        let mut chunks = Vec::new();
        while off != 0 {
            let Some(cp) = self.chunk_ptr(loc.arena, off) else {
                break;
            };
            let c = unsafe { &*cp };
            chunks.push((off, c.len as usize));
            off = c.next;
            if chunks.len() > max_chunks {
                panic!(
                    "arena_store: adjacency chain for vertex {vertex} in arena \
                     {} exceeded {max_chunks} chunks — cycle or corrupt `next` \
                     (last offset {off:#x})",
                    loc.arena
                );
            }
        }
        chunks.reverse();

        for (coff, len) in chunks {
            let Some(cp) = self.chunk_ptr(loc.arena, coff) else {
                continue;
            };
            let c = unsafe { &*cp };
            for e in c.entries.iter().take(len) {
                // Same-arena neighbours take `InvPtr`'s inlined local path
                // (FOT index 0): `local_resolve` masks the entry's *own*
                // address to its object base, so because the chunk was reached
                // through the arena's mapping, so is the neighbour. No registry
                // read, no FOT lookup, and no second mapping.
                //
                // A cross-arena neighbour instead goes through
                // `slow_resolve(READ | INDIRECT)` — a different mapping of the
                // target arena, with the incoherence described on
                // `delete_vertex`. Only `nb.id` is read from it, which is fixed
                // at allocation and never written again. Do not read a
                // mutable field of a neighbour record here.
                #[cfg(test)]
                let mut d = self.diag.get();
                #[cfg(test)]
                let cross = !e.neighbor.is_local();
                #[cfg(test)]
                {
                    d.seen += 1;
                    if cross {
                        d.cross_arena += 1;
                    }
                }
                let nb = unsafe { e.neighbor.resolve() };
                // Generation check before anything else. The slot this
                // entry points at may have been reclaimed and handed to a
                // different record; if so `nb.id` is a valid, live id belonging
                // to something that was never our neighbour, and every check
                // below would pass. This is the one guard that makes space
                // reuse safe, and it has to come first.
                //
                // `generation` is written once when the slot is (re)allocated
                // and not touched again while the record lives, so reading it
                // through a cross-arena mapping is safe for the same reason
                // `nb.id` is. The liveness bit beside them is not.
                if nb.generation != e.neighbor_gen {
                    #[cfg(test)]
                    {
                        d.skipped_gen += 1;
                        if cross {
                            d.skipped_gen_cross += 1;
                        }
                        self.diag.set(d);
                    }
                    continue;
                }
                #[cfg(test)]
                {
                    if !self.is_alive(nb.id) {
                        d.skipped_dead += 1;
                    }
                    self.diag.set(d);
                }
                if self.is_alive(nb.id) {
                    f(e.edge_id, e.label, nb.id);
                }
            }
        }
    }

    /// Neighbours in insertion order.
    pub fn neighbors(&self, vertex: u64, out: bool) -> Vec<u64> {
        let mut ids = Vec::new();
        self.walk_adj(vertex, out, |_, _, nb| ids.push(nb));
        ids
    }

    /// Neighbours whose incident edge carries one of `labels`; `None` means any.
    /// Filtering happens inside the walk so a selective query does not
    /// materialise the whole neighbourhood first.
    pub fn neighbors_labeled(&self, vertex: u64, out: bool, labels: Option<&[u32]>) -> Vec<u64> {
        let mut ids = Vec::new();
        self.walk_adj(vertex, out, |_, l, nb| {
            if labels.map_or(true, |ls| ls.contains(&l)) {
                ids.push(nb);
            }
        });
        ids
    }

    /// Incident edge ids, same filter semantics as [`Self::neighbors_labeled`].
    pub fn edge_ids(&self, vertex: u64, out: bool, labels: Option<&[u32]>) -> Vec<u64> {
        let mut ids = Vec::new();
        self.walk_adj(vertex, out, |e, l, _| {
            if labels.map_or(true, |ls| ls.contains(&l)) {
                ids.push(e);
            }
        });
        ids
    }

    // --- record accessors (VERSION 4) ---------------------------------------
    //
    // These exist so `Graph` can retire the `verts` SegVec: everything the old
    // `VertexRef` mirror carried now lives in the arena record, and the store
    // is the single place that knows how to reach it. Each returns `None` for a
    // missing or tombstoned vertex, so callers get `Graph`'s liveness semantics
    // without repeating the check.

    /// Run `f` on a live vertex's record, or `None` if missing/tombstoned.
    ///
    /// One `locs` read serves both the liveness check and the record address.
    /// Splitting them cost a second lookup on every call — measurably, on the
    /// `scale:20000` churn scan.
    fn with_vertex<R>(&self, vertex: u64, f: impl FnOnce(&ArenaRecordHead) -> R) -> Option<R> {
        let loc = self.live_loc(vertex)?;
        let p = self.record_ptr(&loc)?;
        Some(f(unsafe { &*p }))
    }

    /// Whether the vertex exists and is not tombstoned.
    pub fn is_alive(&self, vertex: u64) -> bool {
        self.locs
            .get_ref(vertex as usize)
            .map_or(false, |l| l.flags & TOMBSTONE == 0)
    }

    /// Diagnostic: the mirror's view of a vertex against the record's, and
    /// where the record sits.
    pub fn debug_liveness(&self, vertex: u64) -> String {
        let Some(loc) = self.locs.get_ref(vertex as usize).map(|l| *l) else {
            return format!("v{vertex}: no loc entry (locs.len={})", self.locs.len());
        };
        let Some(p) = self.record_ptr(&loc) else {
            return format!(
                "v{vertex}: loc{{arena={}, off={}, flags={:#x}}} — arena index out of range \
                 (arenas_open={})",
                loc.arena,
                loc.off,
                loc.flags,
                self.open.len()
            );
        };
        let v = unsafe { &*p };
        format!(
            "v{vertex}: mirror_flags={:#x} (alive={}), record_flags={:#x} (alive={}), \
             record_id={} (expected {vertex}), loc{{arena={}, off={}}}, arenas_open={}",
            loc.flags,
            loc.flags & TOMBSTONE == 0,
            v.flags,
            v.flags & TOMBSTONE == 0,
            v.id,
            loc.arena,
            loc.off,
            self.open.len()
        )
    }

    /// `(label, name, target)` — the old `VertexInfo` triple.
    pub fn vertex_info(&self, vertex: u64) -> Option<(u32, String, u128)> {
        self.with_vertex(vertex, |v| {
            (v.label, v.name.as_str().to_string(), v.target_raw)
        })
    }

    pub fn vertex_label(&self, vertex: u64) -> Option<u32> {
        self.with_vertex(vertex, |v| v.label)
    }

    /// `(label, name)` without allocating — what a traversal predicate needs
    /// per candidate, so filtering a large neighbourhood does not build a
    /// `String` per vertex just to throw it away.
    pub(crate) fn vertex_key(&self, vertex: u64) -> Option<(u32, NameKey)> {
        self.with_vertex(vertex, |v| (v.label, v.name))
    }

    /// `(edge_id, edge_label, neighbour_id)` in traversal order, for the DSL.
    /// `out`/`inc` select the direction(s); both selected yields out then in,
    /// matching `VertexView`'s `Which::Both`.
    pub(crate) fn adjacency(&self, vertex: u64, out: bool, inc: bool) -> Vec<(u64, u32, u64)> {
        let mut res = Vec::new();
        if out {
            self.walk_adj(vertex, true, |e, l, nb| res.push((e, l, nb)));
        }
        if inc {
            self.walk_adj(vertex, false, |e, l, nb| res.push((e, l, nb)));
        }
        res
    }

    /// The vertex's property-object id (0 = none).

    /// All live vertex ids — a linear walk of `locs`, touching no arena.
    ///
    /// Two bit tests on one already-loaded `u32` — no record is resolved, which
    /// is why this stays cheap as records get wider. See [`IS_EDGE`].
    pub fn vertices(&self) -> Vec<u64> {
        let mut out = Vec::new();
        for id in 0..self.locs.len() {
            if self
                .locs
                .get_ref(id)
                .map_or(false, |l| l.flags & (TOMBSTONE | IS_EDGE) == 0)
            {
                out.push(id as u64);
            }
        }
        out
    }

    /// Live vertices carrying `label`. Linear scan, matching `Graph`.
    pub fn vertices_by_label(&self, label: u32) -> Vec<u64> {
        (0..self.locs.len() as u64)
            .filter(|id| self.vertex_label(*id) == Some(label))
            .collect()
    }

    pub fn vertex_name(&self, vertex: u64) -> Option<String> {
        self.with_vertex(vertex, |v| v.name.as_str().to_string())
    }

    /// Tombstone a vertex, in the record and in the `locs` mirror.
    ///
    /// `GlobalPtr::resolve` maps its object `READ`, while `GlobalPtr::resolve_mut`
    /// maps it `READ | WRITE | PERSIST` (`ptr/global.rs`). Different flags mean
    /// `twz_rt_map_object` hands back *different mappings*, and a write through
    /// one is not visible through the other. Measured on `scale:20000`: after
    /// `v.flags |= TOMBSTONE`, a fresh `resolve_mut` read the bit set and a
    /// fresh `resolve` read it clear, identically in a stale arena and the
    /// current one. Every reader here uses `resolve`, so the record bit was
    /// invisible and 2 858 deleted vertices stayed live.
    ///
    /// So the record's bit is written and kept consistent, but `is_alive` and
    /// [`Self::live_loc`] are what decide. Do not gate a read on the record's
    /// `flags` — it is right today only for arenas you reached locally.
    pub fn delete_vertex(&mut self, vertex: u64) -> Result<()> {
        let Some(loc) = self.locs.get_ref(vertex as usize).map(|l| *l) else {
            return Ok(());
        };
        // Guard on the mirror, before touching anything. A second delete
        // must not return the slot twice — two records would then be handed the
        // same bytes, which is not a stale-reference problem that generations
        // can catch but straightforward corruption. `deletes_stay_idempotent`
        // covers the double-delete path.
        if loc.flags & TOMBSTONE != 0 {
            return Ok(());
        }
        let mut stride = None;
        if let Some(p) = self.record_ptr(&loc) {
            unsafe {
                (*p).flags |= TOMBSTONE;
                stride = Some(record_size((*p).nprops));
            }
        }
        // The slot is immediately reusable: inbound `AdjRef`s still name it, but
        // they carry the generation it had, and reuse bumps it.
        if let (Some(sz), Some(slots)) = (stride, self.free.get_mut(loc.arena as usize)) {
            slots.push((loc.off, sz));
        }
        // nosync: `with_mut_at` would sync the registry on every delete, which
        // measured 2.5× on the churn phase. Drained by `sync_all`'s
        // `locs.flush()`, like every other write here.
        self.locs.with_mut_at_nosync(vertex as usize, |l| {
            l.flags |= TOMBSTONE;
            Ok(())
        })?;
        Ok(())
    }

    /// Test-only: tombstone the mirror and leave the record alive, forcing
    /// the disagreement that `resolve`/`resolve_mut` incoherence produced at
    /// scale but that a small in-boot test cannot provoke on its own.
    #[cfg(test)]
    pub(crate) fn tombstone_mirror_only(&mut self, vertex: u64) -> Result<()> {
        self.locs.with_mut_at_nosync(vertex as usize, |l| {
            l.flags |= TOMBSTONE;
            Ok(())
        })
    }

    /// Closes each arena's batching transaction first. `abort()` is what makes
    /// the batching worth anything: it suppresses the transaction's sync-on-drop
    /// so the arena is synced exactly once here, rather than once per open
    /// transaction plus once again below. Upstream transactions have no
    /// rollback, so the writes stand — the assumption is pinned by
    /// `tx_abort_does_not_roll_back` in `tests/bulk.rs`, which fails loudly if
    /// upstream ever implements one.
    pub fn sync_all(&mut self) -> Result<()> {
        for slot in self.txs.iter_mut() {
            if let Some(mut tx) = slot.take() {
                tx.abort();
            }
        }
        for (i, a) in self.open.iter().enumerate() {
            if !self.dirty.get(i).copied().unwrap_or(true) {
                continue;
            }
            unsafe { a.object().as_mut()?.sync()? };
            self.syncs += 1;
        }
        for d in self.dirty.iter_mut() {
            *d = false;
        }
        self.dir.flush()?;
        self.locs.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod layout_tests {

    // Imported explicitly, as the module already does for `size_of`: these are
    // prelude items only on newer toolchains.
    use core::mem::{align_of, size_of};

    use super::*;

    /// The stride must be head + slots with no inter-element padding, which
    /// holds only while `PropSlot`'s size is a multiple of the head's alignment.
    /// Asserted rather than assumed — this is the arithmetic that goes wrong.
    #[test]
    fn record_stride_has_no_hidden_padding() {
        assert_eq!(
            size_of::<PropSlot>() % align_of::<ArenaRecordHead>(),
            0,
            "PropSlot ({}) must be a multiple of the head's alignment ({}), or \
             records do not pack back-to-back and `record_size` is a lie",
            size_of::<PropSlot>(),
            align_of::<ArenaRecordHead>()
        );
        assert_eq!(
            size_of::<ArenaRecordHead>() % align_of::<PropSlot>(),
            0,
            "the head must end on a PropSlot boundary, or slot 0 is misaligned"
        );
    }

    #[test]
    fn record_size_matches_the_types() {
        assert_eq!(record_size(0), size_of::<ArenaRecordHead>());
        for n in [1u16, 2, 7, 64, 1000, u16::MAX] {
            assert_eq!(
                record_size(n),
                size_of::<ArenaRecordHead>() + n as usize * size_of::<PropSlot>()
            );
        }
        // Strictly increasing, so two records can never be given the same
        // extent by different `nprops` — a walk over an arena depends on it.
        assert!(record_size(0) < record_size(1));
        assert!(record_size(1) < record_size(2));
    }

    /// The mirror must not grow. `VertexLoc` is what makes a scan cheap:
    /// 16 B means 256 entries per 4 KB page against ~42 records, and that 6.1×
    /// density ratio is the model behind the measured 5.2× warm-scan gain.
    /// `IS_EDGE` is a *bit*, so it costs nothing here — a test because the
    /// tempting fix when the scan needs more information is to widen this.
    #[test]
    fn the_liveness_mirror_stays_sixteen_bytes() {
        assert_eq!(
            size_of::<VertexLoc>(),
            16,
            "widening VertexLoc trades away the property that makes scans cheap"
        );
    }

    /// `TOMBSTONE` and `IS_EDGE` are distinct bits in the same word, and both
    /// fit where the mirror already carries flags.
    #[test]
    fn record_flags_are_disjoint_bits() {
        assert_eq!(TOMBSTONE & IS_EDGE, 0, "flags must not overlap");
        assert_eq!(TOMBSTONE.count_ones(), 1);
        assert_eq!(IS_EDGE.count_ones(), 1);
    }

    /// `alloc_record_bytes` depends on an alignment floor it does not set.
    ///
    /// Records are reserved as a `[u8]`, whose `Layout` alignment is 1;
    /// `ArenaBase::reserve` raises every allocation to `MIN_ALIGN = 16`, and
    /// that is the only reason a record lands 16-aligned. The constant is
    /// upstream and private, so this asserts the requirement rather than the
    /// mechanism: if the head or slots ever need more than 16, the byte-slice
    /// allocation is silently wrong and this is where it shows.
    #[test]
    fn record_head_alignment_holds() {
        const UPSTREAM_ARENA_MIN_ALIGN: usize = 16;
        assert!(
            align_of::<ArenaRecordHead>() <= UPSTREAM_ARENA_MIN_ALIGN,
            "record head needs {}-byte alignment but arena allocations only \
             guarantee {}; `alloc_record_bytes` can no longer use a byte slice",
            align_of::<ArenaRecordHead>(),
            UPSTREAM_ARENA_MIN_ALIGN
        );
        assert!(
            align_of::<PropSlot>() <= UPSTREAM_ARENA_MIN_ALIGN,
            "PropSlot needs {}-byte alignment, above the arena's guarantee",
            align_of::<PropSlot>()
        );
    }

    /// A record head must stay comfortably inside a page: a head split across
    /// pages costs a second fault on every touch, and the traversal argument is
    /// built on a hop being one.
    #[test]
    fn a_record_head_fits_well_within_a_page() {
        assert!(
            size_of::<ArenaRecordHead>() <= 4096 / 8,
            "record head is {} bytes; at this size page density collapses",
            size_of::<ArenaRecordHead>()
        );
    }
}
