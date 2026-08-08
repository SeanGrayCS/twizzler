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
use crate::segvec::SegVec;

type Result<T> = core::result::Result<T, twizzler::error::TwzError>;

/// Adjacency entries per chunk. Small enough that tests exercise chunk
/// rollover cheaply; large enough that low-degree vertices need one chunk.
pub const ADJ_CHUNK: usize = 8;

/// `flags` bit 0: record is deleted.
const TOMBSTONE: u32 = 1;

/// A vertex, allocated inside an arena. `out_head`/`in_head` are arena
/// offsets (0 = empty), not pointers.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct ArenaVertex {
    pub(crate) id: u64,
    pub(crate) label: u32,
    pub(crate) flags: u32,
    pub(crate) name: NameKey,
    /// VERSION 4: absorbed from `VertexRef`, which is retiring. Holding these
    /// here is what lets the `verts` registry go away entirely rather than
    /// shadowing the arena with a second structure that assigns ids in
    /// lockstep by convention.
    pub(crate) target_raw: u128,
    pub(crate) props_raw: u128,
    pub(crate) out_head: u64,
    pub(crate) in_head: u64,
}
unsafe impl Invariant for ArenaVertex {}

/// One adjacency entry (VERSION 4).
///
/// The neighbour is an [`InvPtr`], which handles both cases in one field:
/// a target in *this* arena gets FOT index 0 — no FOT entry, and
/// `resolve()` takes an inlined base+offset path — while a target elsewhere
/// costs one FOT entry, deduped per target arena by the runtime's
/// `insert_fot`. Traversal therefore never consults the location registry.
///
/// Neither this nor [`AdjChunk`] is `Copy`, and that is load-bearing: a FOT
/// index means something only inside its containing object, so copying an entry
/// between arenas would silently mis-resolve. `InvPtr` is not `Copy` for
/// exactly this reason; do not "fix" it by storing the raw `u64`.
#[repr(C)]
pub(crate) struct AdjRef {
    pub(crate) edge_id: u64,
    pub(crate) neighbor: InvPtr<ArenaVertex>,
    pub(crate) label: u32,
    pub(crate) _pad: u32,
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
            _pad: 0,
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
    policy: Box<dyn Placement>,
    /// Syncs issued by `sync_all`, so a test can assert the batching property
    /// directly rather than inferring it from wall time.
    syncs: usize,
}

impl ArenaStore {
    pub fn create(policy: Box<dyn Placement>, seg_cap: usize) -> Result<Self> {
        Ok(ArenaStore {
            dir: SegVec::create(seg_cap)?,
            locs: SegVec::create(seg_cap)?,
            open: Vec::new(),
            txs: Vec::new(),
            stats: Vec::new(),
            policy,
            syncs: 0,
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
        let txs = (0..open.len()).map(|_| None).collect();
        Ok(ArenaStore {
            dir,
            locs,
            open,
            txs,
            stats,
            policy,
            syncs: 0,
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
        for id in 0..self.locs.len() as u64 {
            // Tombstoned vertices included: their property objects are exactly
            // the ones nothing else will ever free.
            if let Some(loc) = self.locs.get_ref(id as usize).map(|l| *l) {
                if let Some(rp) = self.record_ptr(&loc) {
                    let p = unsafe { (*rp).props_raw };
                    if p != 0 {
                        ids.push(p);
                    }
                }
            }
        }
        ids
    }

    pub fn arena_count(&self) -> usize {
        self.open.len()
    }

    pub fn vertex_count(&self) -> usize {
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
    fn vertex_ptr(&self, id: u64) -> Option<GlobalPtr<ArenaVertex>> {
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
    fn record_ptr(&self, loc: &VertexLoc) -> Option<*mut ArenaVertex> {
        let obj = self.open.get(loc.arena as usize)?.object();
        obj.lea_mut(loc.off as usize, size_of::<ArenaVertex>())
            .map(|p| p as *mut ArenaVertex)
    }

    /// Raw pointer to an adjacency chunk, inside its arena's own mapping.
    fn chunk_ptr(&self, arena: u32, off: u64) -> Option<*mut AdjChunk> {
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

    /// Add a vertex, letting the policy choose its arena.
    pub fn add_vertex(&mut self, label: u32, name: &str, target_raw: u128) -> Result<u64> {
        let idx = match self.policy.place(&self.stats) {
            Some(i) if i < self.open.len() => i,
            _ => self.new_arena()?,
        };
        let id = self.locs.len() as u64;
        let off = {
            let tx = self.tx_for(idx)?;
            tx.alloc(ArenaVertex {
                id,
                label,
                flags: 0,
                name: NameKey::new(name),
                target_raw,
                props_raw: 0,
                out_head: 0,
                in_head: 0,
            })?
            .offset()
        };
        self.locs.push_nosync(VertexLoc {
            arena: idx as u32,
            flags: 0,
            off,
        })?;
        self.stats[idx].vertices += 1;
        Ok(id)
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
        neighbor: GlobalPtr<ArenaVertex>,
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

        // Build the entry against this arena's transaction. `InvPtr::new`
        // returns FOT index 0 when `neighbor` lives in this same arena.
        let entry = {
            let tx = self.tx_for(arena_idx)?;
            AdjRef {
                edge_id,
                neighbor: InvPtr::new(&*tx, neighbor)?,
                label,
                _pad: 0,
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
        self.append_adj(from, true, edge_id, label, to_gp)?;
        self.append_adj(to, false, edge_id, label, from_gp)
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
                let nb = unsafe { e.neighbor.resolve() };
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
    fn with_vertex<R>(&self, vertex: u64, f: impl FnOnce(&ArenaVertex) -> R) -> Option<R> {
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
    pub fn props_raw(&self, vertex: u64) -> Option<u128> {
        self.with_vertex(vertex, |v| v.props_raw)
    }

    /// Point the vertex at a (possibly new) property object. Writes mapped
    /// memory; durable at [`Self::sync_all`] like every other mutation here.
    pub fn set_props_raw(&mut self, vertex: u64, raw: u128) -> Result<()> {
        let Some(loc) = self.live_loc(vertex) else {
            return Err(twizzler_rt_abi::error::ArgumentError::InvalidArgument.into());
        };
        let Some(p) = self.record_ptr(&loc) else {
            return Err(twizzler_rt_abi::error::ArgumentError::InvalidArgument.into());
        };
        unsafe { (*p).props_raw = raw };
        Ok(())
    }

    /// All live vertex ids — a linear walk of `locs`, touching no arena.
    pub fn vertices(&self) -> Vec<u64> {
        let mut out = Vec::new();
        for id in 0..self.locs.len() {
            if self
                .locs
                .get_ref(id)
                .map_or(false, |l| l.flags & TOMBSTONE == 0)
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
        if let Some(p) = self.record_ptr(&loc) {
            unsafe { (*p).flags |= TOMBSTONE };
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
        for a in &self.open {
            unsafe { a.object().as_mut()?.sync()? };
            self.syncs += 1;
        }
        self.dir.flush()?;
        self.locs.flush()?;
        Ok(())
    }
}
