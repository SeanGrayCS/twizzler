#![allow(dead_code)]

//! Mechanism. [`ArenaObject`] is a bump allocator *inside one object*. The
//! current layout is forced into three objects only because `VecObjectAlloc`
//! hardcodes a single data region per object; an arena hands out many disjoint
//! allocations, so a vertex record and both its adjacency chains can live in one
//! object — or a thousand vertices can.
//!
//! References are arena offsets, not `InvPtr`s. Within an arena a link is a
//! `u64` offset resolved through [`GlobalPtr`], so intra-arena traversal needs
//! no FOT entry and touches no second object. Neighbours in *other* arenas are
//! named by `VertexId` and resolved through the location registry — the "A4b"
//! reference form. This was chosen deliberately: the cold/warm measurement
//! showed cold `InvPtr` traversal is no better than the KV baseline (1.1×),
//! while warm speed comes from data being *mapped*, which ids into a mapped
//! registry also enjoy. A4a (`InvPtr`s) remains the comparison variant.

use twizzler::{
    alloc::arena::ArenaObject,
    marker::Invariant,
    object::{ObjID, ObjectBuilder},
    ptr::GlobalPtr,
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
    pub(crate) out_head: u64,
    pub(crate) in_head: u64,
}
unsafe impl Invariant for ArenaVertex {}

/// One adjacency entry. The neighbour is named by id, not by pointer — see the
/// module docs on reference form.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct AdjRef {
    pub(crate) edge_id: u64,
    pub(crate) neighbor: u64,
    pub(crate) label: u32,
    pub(crate) _pad: u32,
}
unsafe impl Invariant for AdjRef {}

/// A chunk of adjacency entries. Chunks are prepended, so walking from the
/// head yields newest-first; readers reverse the chunk order to recover
/// insertion order (entries within a chunk are already in order).
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct AdjChunk {
    pub(crate) len: u32,
    pub(crate) _pad: u32,
    pub(crate) next: u64,
    pub(crate) entries: [AdjRef; ADJ_CHUNK],
}
unsafe impl Invariant for AdjChunk {}

/// Where a vertex lives: which arena, and at what offset within it.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct VertexLoc {
    pub(crate) arena: u32,
    pub(crate) _pad: u32,
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
    stats: Vec<ArenaStat>,
    policy: Box<dyn Placement>,
}

impl ArenaStore {
    pub fn create(policy: Box<dyn Placement>, seg_cap: usize) -> Result<Self> {
        Ok(ArenaStore {
            dir: SegVec::create(seg_cap)?,
            locs: SegVec::create(seg_cap)?,
            open: Vec::new(),
            stats: Vec::new(),
            policy,
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
        // Rebuild per-arena counts from the location registry.
        let mut stats = vec![ArenaStat { vertices: 0 }; open.len()];
        for i in 0..locs.len() {
            if let Some(l) = locs.get_ref(i) {
                if let Some(s) = stats.get_mut(l.arena as usize) {
                    s.vertices += 1;
                }
            }
        }
        Ok(ArenaStore {
            dir,
            locs,
            open,
            stats,
            policy,
        })
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

    fn new_arena(&mut self) -> Result<usize> {
        let arena = ArenaObject::new(ObjectBuilder::default().persist(true))?;
        let raw = arena.object().id().raw();
        self.dir.push(ArenaEntry { raw })?;
        self.open.push(arena);
        self.stats.push(ArenaStat { vertices: 0 });
        Ok(self.open.len() - 1)
    }

    fn arena_id(&self, idx: usize) -> ObjID {
        self.open[idx].object().id()
    }

    fn vertex_ptr(&self, id: u64) -> Option<GlobalPtr<ArenaVertex>> {
        let loc = self.locs.get_ref(id as usize)?;
        Some(GlobalPtr::new(self.arena_id(loc.arena as usize), loc.off))
    }

    /// Add a vertex, letting the policy choose its arena.
    pub fn add_vertex(&mut self, label: u32, name: &str) -> Result<u64> {
        let idx = match self.policy.place(&self.stats) {
            Some(i) if i < self.open.len() => i,
            _ => self.new_arena()?,
        };
        let id = self.locs.len() as u64;
        let gp = self.open[idx].alloc(ArenaVertex {
            id,
            label,
            flags: 0,
            name: NameKey::new(name),
            out_head: 0,
            in_head: 0,
        })?;
        let off = gp.offset();
        self.locs.push(VertexLoc {
            arena: idx as u32,
            _pad: 0,
            off,
        })?;
        self.stats[idx].vertices += 1;
        Ok(id)
    }

    /// Append an adjacency entry to `vertex`'s out (or in) chain. The chunk is
    /// allocated in the vertex's own arena, so a low-degree vertex adds no
    /// objects at all.
    fn append_adj(&mut self, vertex: u64, out: bool, entry: AdjRef) -> Result<()> {
        let Some(loc) = self.locs.get_ref(vertex as usize).map(|l| *l) else {
            return Err(twizzler_rt_abi::error::ArgumentError::InvalidArgument.into());
        };
        let arena_idx = loc.arena as usize;
        let aid = self.arena_id(arena_idx);
        let vgp: GlobalPtr<ArenaVertex> = GlobalPtr::new(aid, loc.off);

        let head = {
            let v = unsafe { vgp.resolve() };
            if out {
                v.out_head
            } else {
                v.in_head
            }
        };

        // Room in the head chunk? Append there and we are done.
        if head != 0 {
            let cgp: GlobalPtr<AdjChunk> = GlobalPtr::new(aid, head);
            let mut c = unsafe { cgp.resolve_mut() };
            if (c.len as usize) < ADJ_CHUNK {
                let n = c.len as usize;
                c.entries[n] = entry;
                c.len += 1;
                return Ok(());
            }
        }

        // Otherwise allocate a fresh chunk in the same arena and link it in.
        let mut entries = [AdjRef {
            edge_id: 0,
            neighbor: 0,
            label: 0,
            _pad: 0,
        }; ADJ_CHUNK];
        entries[0] = entry;
        let cgp = self.open[arena_idx].alloc(AdjChunk {
            len: 1,
            _pad: 0,
            next: head,
            entries,
        })?;
        let new_off = cgp.offset();
        let mut v = unsafe { vgp.resolve_mut() };
        if out {
            v.out_head = new_off;
        } else {
            v.in_head = new_off;
        }
        Ok(())
    }

    /// Record an edge on both endpoints.
    pub fn add_edge(&mut self, from: u64, to: u64, edge_id: u64, label: u32) -> Result<()> {
        self.append_adj(
            from,
            true,
            AdjRef {
                edge_id,
                neighbor: to,
                label,
                _pad: 0,
            },
        )?;
        self.append_adj(
            to,
            false,
            AdjRef {
                edge_id,
                neighbor: from,
                label,
                _pad: 0,
            },
        )
    }

    /// Neighbours in insertion order. Chunks are prepended, so the chunk list
    /// is walked then reversed; entries within a chunk are already ordered.
    pub fn neighbors(&self, vertex: u64, out: bool) -> Vec<u64> {
        let Some(loc) = self.locs.get_ref(vertex as usize).map(|l| *l) else {
            return Vec::new();
        };
        let aid = self.arena_id(loc.arena as usize);
        let vgp: GlobalPtr<ArenaVertex> = GlobalPtr::new(aid, loc.off);
        let v = unsafe { vgp.resolve() };
        if v.flags & TOMBSTONE != 0 {
            return Vec::new();
        }
        let mut off = if out { v.out_head } else { v.in_head };
        drop(v);

        let mut chunks = Vec::new();
        while off != 0 {
            let cgp: GlobalPtr<AdjChunk> = GlobalPtr::new(aid, off);
            let c = unsafe { cgp.resolve() };
            chunks.push((off, c.len as usize));
            off = c.next;
        }
        chunks.reverse();

        let mut out_ids = Vec::new();
        for (coff, len) in chunks {
            let cgp: GlobalPtr<AdjChunk> = GlobalPtr::new(aid, coff);
            let c = unsafe { cgp.resolve() };
            for e in c.entries.iter().take(len) {
                out_ids.push(e.neighbor);
            }
        }
        out_ids
    }

    pub fn vertex_name(&self, vertex: u64) -> Option<String> {
        let loc = self.locs.get_ref(vertex as usize)?;
        let gp: GlobalPtr<ArenaVertex> =
            GlobalPtr::new(self.arena_id(loc.arena as usize), loc.off);
        let v = unsafe { gp.resolve() };
        if v.flags & TOMBSTONE != 0 {
            return None;
        }
        Some(v.name.as_str().to_string())
    }

    pub fn delete_vertex(&mut self, vertex: u64) -> Result<()> {
        let Some(loc) = self.locs.get_ref(vertex as usize).map(|l| *l) else {
            return Ok(());
        };
        let gp: GlobalPtr<ArenaVertex> =
            GlobalPtr::new(self.arena_id(loc.arena as usize), loc.off);
        let mut v = unsafe { gp.resolve_mut() };
        v.flags |= TOMBSTONE;
        Ok(())
    }

    pub fn sync_all(&mut self) -> Result<()> {
        for a in &self.open {
            unsafe { a.object().as_mut()?.sync()? };
        }
        self.dir.flush()?;
        self.locs.flush()?;
        Ok(())
    }
}
