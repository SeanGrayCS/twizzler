//! Shared byte storage for long property values.
//!
//! Backs both `PropValue::TextRef` (≤255 B, queryable) and `PropValue::BlobRef`
//! (arbitrary length, not queryable). They share this store because they
//! differ in the API — a length limit and whether a filter exists — not in how
//! bytes are kept.
//!
//! Long values live out-of-line because records sit back-to-back at a stride
//! derived from `nprops`. Variable-length bytes among them would make that
//! stride per-record in a way nothing validates — a reader would take a
//! neighbouring record's bytes as this one's fields. Keeping the bytes here
//! leaves `record_size(nprops)` untouched.
//!
//! Values are bump-allocated whole into a few large segments. `SegVec` would
//! cost one synced push per value, and an object per value would spend the
//! platform's scarce resource: objects, not bytes.

use twizzler::{
    alloc::arena::{ArenaBase, ArenaObject},
    marker::Invariant,
    object::{ObjID, ObjectBuilder, RawObject, TxObject},
};

use crate::error::Result;
use crate::segvec::SegVec;

/// Directory entry naming one backing object.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct BlobEntry {
    pub(crate) raw: u128,
}
unsafe impl Invariant for BlobEntry {}

/// Roll to a fresh object past this many bytes.
///
/// Well under the 1 GB object maximum: the point is not to fill an object but
/// to keep any single one small enough that syncing it is not a long stall.
const SEGMENT_BYTES: usize = 64 * 1024 * 1024;

/// Append-only byte store, segmented across persistent objects.
pub(crate) struct BlobStore {
    dir: SegVec<BlobEntry>,
    open: Vec<ArenaObject>,
    txs: Vec<Option<TxObject<ArenaBase>>>,
    /// Bytes handed out per segment, to decide when to roll.
    used: Vec<usize>,
    /// Which segments have been written since the last `sync_all`.
    dirty: Vec<bool>,
}

impl BlobStore {
    pub(crate) fn create(cap: usize) -> Result<Self> {
        Ok(BlobStore {
            dir: SegVec::create(cap)?,
            open: Vec::new(),
            txs: Vec::new(),
            used: Vec::new(),
            dirty: Vec::new(),
        })
    }

    pub(crate) fn open(dir_raw: u128, cap: usize) -> Result<Self> {
        let dir = SegVec::<BlobEntry>::open(dir_raw, cap)?;
        let mut open = Vec::with_capacity(dir.len());
        for i in 0..dir.len() {
            let raw = dir.get_ref(i).map(|e| e.raw).unwrap_or(0);
            open.push(ArenaObject::from_objid(ObjID::new(raw))?);
        }
        let n = open.len();
        Ok(BlobStore {
            dir,
            open,
            txs: (0..n).map(|_| None).collect(),
            // Reopened segments are treated as full. Their bump offset lives
            // in the object's own arena base, not here, so `used` cannot be
            // recovered — and guessing low would hand out offsets the allocator
            // has already given away. Appending after a reopen therefore starts
            // a new segment, which wastes tail space and never corrupts.
            used: vec![SEGMENT_BYTES; n],
            dirty: vec![false; n],
        })
    }

    pub(crate) fn dir_raw(&self) -> u128 {
        self.dir.dir_raw()
    }

    /// Objects this store owns, for teardown and accounting.
    pub(crate) fn object_ids(&self) -> Vec<u128> {
        let mut ids = self.dir.object_ids();
        ids.extend(self.open.iter().map(|a| a.object().id().raw()));
        ids
    }

    pub(crate) fn object_count(&self) -> usize {
        self.open.len()
    }

    /// Store `bytes`, returning `(segment, offset, len)`.
    ///
    /// Empty input still gets a real location rather than a sentinel: a stored
    /// empty string must be distinguishable from an absent property, and
    /// encoding "absent" as offset 0 would make the two identical.
    pub(crate) fn append(&mut self, bytes: &[u8]) -> Result<(u32, u64, u32)> {
        let idx = self.segment_for(bytes.len())?;
        let tx = self.tx_for(idx)?;
        let off = tx.alloc_with_slice::<u8>(bytes)?.1.offset();
        self.used[idx] += bytes.len();
        self.dirty[idx] = true;
        Ok((idx as u32, off, bytes.len() as u32))
    }

    /// Read back bytes written by [`Self::append`].
    ///
    /// Returns `None` rather than partial data if the location is out of range:
    /// a truncated read here would surface as a mangled property value.
    pub(crate) fn read(&self, seg: u32, off: u64, len: u32) -> Option<Vec<u8>> {
        let obj = self.open.get(seg as usize)?.object();
        let p = obj.lea(off as usize, len as usize)?;
        // Safety: `lea` bounds-checks against the mapping, and the store is
        // append-only, so bytes handed out are never rewritten or freed.
        Some(unsafe { core::slice::from_raw_parts(p, len as usize) }.to_vec())
    }

    fn segment_for(&mut self, want: usize) -> Result<usize> {
        if let Some(i) = self
            .used
            .iter()
            .position(|u| u.saturating_add(want) <= SEGMENT_BYTES)
        {
            return Ok(i);
        }
        self.new_segment()
    }

    fn new_segment(&mut self) -> Result<usize> {
        let seg = ArenaObject::new(ObjectBuilder::default().persist(true))?;
        let raw = seg.object().id().raw();
        // nosync: drained by `sync_all`'s `dir.flush()`. A plain `push` would
        // sync the directory object once per segment.
        self.dir.push_nosync(BlobEntry { raw })?;
        self.open.push(seg);
        self.txs.push(None);
        self.used.push(0);
        self.dirty.push(true);
        Ok(self.open.len() - 1)
    }

    fn tx_for(&mut self, idx: usize) -> Result<&mut TxObject<ArenaBase>> {
        if self.txs[idx].is_none() {
            self.txs[idx] = Some(self.open[idx].as_tx()?);
        }
        Ok(self.txs[idx].as_mut().expect("just opened"))
    }

    /// Close transactions and sync touched segments.
    ///
    /// `abort()` rather than commit, matching `ArenaStore::sync_all`: it
    /// suppresses sync-on-drop without rolling back, which is what makes the
    /// `nosync` write path durable exactly once, here.
    pub(crate) fn sync_all(&mut self) -> Result<()> {
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
        }
        for d in self.dirty.iter_mut() {
            *d = false;
        }
        self.dir.flush()?;
        Ok(())
    }
}
