//! `SegVec`: a persistent, segmented vector of fixed-capacity `VecObject`s.
//!
//! A single `VecObject` lives inside one Twizzler object, so its growth is
//! bounded by the object size (and, for element types holding `InvPtr`s, by
//! the per-object FOT limit). `SegVec` lifts those bounds by chaining
//! segments: a small persistent *directory* object records the `ObjID` of
//! each segment in order, and every segment except the last is kept exactly
//! full (`cap` elements), so an element index maps to
//! `(idx / cap, idx % cap)` in O(1) — preserving the engine's ids-are-append-
//! indices lookup scheme.
//!
//! The capacity is chosen when the containing graph is created, persisted in
//! the graph root, and always used on open, keeping segment geometry uniform
//! for the life of the graph. Directory entries are raw `ObjID`s (no
//! `InvPtr`s), so the directory itself has no FOT pressure.

use twizzler::{
    collections::vec::{Vec as TwzVec, VecObject, VecObjectAlloc},
    error::TwzError,
    marker::{Invariant, StoreCopy},
    object::{MapFlags, ObjID, Object, ObjectBuilder},
    ptr::Ref,
};
use twizzler_rt_abi::error::ArgumentError;

type Result<T> = core::result::Result<T, TwzError>;

// --- public-API nosync primitives (confined to this crate) ------------------
//
// Batching needs writes whose durability is deferred to one explicit sync per
// object at batch close. The twizzler crate exposes no nosync API, and we
// deliberately change nothing outside our section of the tree. Instead we use
// the public `TxObject::abort()`: in the current upstream implementation,
// transactions have no rollback, so `abort` only suppresses the sync-on-drop
// and the writes remain. That assumption is load-bearing and pinned by
// `tx_abort_does_not_roll_back` in `tests/arena.rs` — if upstream ever
// implements real rollback, that canary fails loudly and these helpers must
// switch to a proper upstream nosync API.

/// `VecObject::push` with the sync-on-drop suppressed; durability deferred
/// to a later explicit sync of the object.
pub(crate) fn vec_push_nosync<T>(v: &mut VecObject<T, VecObjectAlloc>, val: T) -> Result<()>
where
    T: Invariant + StoreCopy,
{
    let mut tx = v.object().as_tx()?;
    tx.base_mut().push(val)?;
    tx.abort();
    Ok(())
}

/// `VecObject::new` without the new object's initial sync.
pub(crate) fn vec_new_nosync<T: Invariant>(
    builder: ObjectBuilder<TwzVec<T, VecObjectAlloc>>,
) -> Result<VecObject<T, VecObjectAlloc>> {
    Ok(VecObject::from(builder.build_inplace(|tx| {
        let mut done = tx.write(TwzVec::new_in(VecObjectAlloc))?;
        done.abort();
        Ok(done)
    })?))
}

/// Read/write/persist map flags for reopening mutable segments.
fn rw() -> MapFlags {
    MapFlags::READ | MapFlags::WRITE | MapFlags::PERSIST
}

/// Directory entry: the ObjID of one segment (raw, so the on-disk format is
/// backend-agnostic and relocatable, matching the graph root).
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct SegEntry {
    raw: u128,
}
unsafe impl Invariant for SegEntry {}

/// A segmented persistent vector; see the module docs.
pub(crate) struct SegVec<T: Invariant> {
    /// The directory object: segment ObjIDs, in order.
    dir: VecObject<SegEntry, VecObjectAlloc>,
    /// Open handles for every segment, same order as `dir`.
    segs: Vec<VecObject<T, VecObjectAlloc>>,
    /// Per-segment capacity; every non-last segment holds exactly `cap`.
    cap: usize,
    /// Segment indices with writes whose durability was deferred
    /// (`push_nosync`); drained by [`SegVec::flush`].
    dirty: std::collections::HashSet<usize>,
}

impl<T: Invariant> SegVec<T> {
    /// Create a new, empty segmented vector with the given segment capacity.
    pub(crate) fn create(cap: usize) -> Result<Self> {
        if cap == 0 {
            return Err(ArgumentError::InvalidArgument.into());
        }
        let dir = VecObject::new(ObjectBuilder::default().persist(true))?;
        Ok(SegVec {
            dir,
            segs: Vec::new(),
            cap,
            dirty: std::collections::HashSet::new(),
        })
    }

    /// Open an existing segmented vector from its directory ObjID, mapping
    /// every segment listed in the directory.
    pub(crate) fn open(dir_raw: u128, cap: usize) -> Result<Self> {
        if cap == 0 {
            return Err(ArgumentError::InvalidArgument.into());
        }
        let dir: VecObject<SegEntry, VecObjectAlloc> =
            VecObject::from(Object::<TwzVec<SegEntry, VecObjectAlloc>>::map(
                ObjID::new(dir_raw),
                rw(),
            )?);
        let mut segs = Vec::with_capacity(dir.len());
        for i in 0..dir.len() {
            let raw = dir
                .get_ref(i)
                .ok_or(TwzError::from(ArgumentError::InvalidArgument))?
                .raw;
            segs.push(VecObject::from(Object::<TwzVec<T, VecObjectAlloc>>::map(
                ObjID::new(raw),
                rw(),
            )?));
        }
        // Validate the stored geometry against `cap`: every non-last segment
        // must be exactly full and the last at most full, or the index
        // arithmetic (idx / cap, idx % cap) would misresolve elements.
        for (i, seg) in segs.iter().enumerate() {
            let ok = if i + 1 == segs.len() {
                seg.len() <= cap
            } else {
                seg.len() == cap
            };
            if !ok {
                return Err(ArgumentError::InvalidArgument.into());
            }
        }
        Ok(SegVec {
            dir,
            segs,
            cap,
            dirty: std::collections::HashSet::new(),
        })
    }

    /// The directory object's ObjID — what the graph root records.
    pub(crate) fn dir_raw(&self) -> u128 {
        self.dir.object().id().raw()
    }

    /// Every object this vector owns: the directory plus each segment, in that
    /// order. Used by reclaim to free a whole registry, and by the reclaim
    /// tests to assert the objects really went away.
    pub(crate) fn object_ids(&self) -> Vec<u128> {
        let mut ids = Vec::with_capacity(self.segs.len() + 1);
        ids.push(self.dir_raw());
        ids.extend(self.segs.iter().map(|s| s.object().id().raw()));
        ids
    }

    /// Total number of elements across all segments. Non-last segments are
    /// exactly full, so only the last one needs its length read.
    pub(crate) fn len(&self) -> usize {
        match self.segs.last() {
            None => 0,
            Some(last) => (self.segs.len() - 1) * self.cap + last.len(),
        }
    }

    /// Number of segments (diagnostic/test seam).
    #[cfg(test)]
    pub(crate) fn segments(&self) -> usize {
        self.segs.len()
    }

    /// A reference to the element at `idx`, or `None` if out of bounds.
    pub(crate) fn get_ref(&self, idx: usize) -> Option<Ref<'_, T>> {
        self.segs.get(idx / self.cap)?.get_ref(idx % self.cap)
    }

    /// Mutate the entry at `idx`, deferring durability to [`SegVec::flush`].
    ///
    /// This is the only in-place mutation `SegVec` offers, and it does not
    /// sync. A variant that opened a syncing transaction would cost a sync
    /// per element on a hot path; defer instead, and let `flush` sync each
    /// dirty object once.
    pub(crate) fn with_mut_at_nosync<R>(
        &mut self,
        idx: usize,
        f: impl FnOnce(&mut T) -> Result<R>,
    ) -> Result<R> {
        let (si, off) = (idx / self.cap, idx % self.cap);
        let seg = self
            .segs
            .get_mut(si)
            .ok_or(TwzError::from(ArgumentError::InvalidArgument))?;
        let mut tx = seg.object().as_tx()?;
        let r = {
            let mut base = tx.base_mut();
            // Safety: single-threaded per graph handle, and the index is
            // bounds-checked below — same contract as `with_mut_slice`.
            let mut item = unsafe { base.get_mut(off) }
                .ok_or(TwzError::from(ArgumentError::InvalidArgument))?;
            f(&mut item)?
        };
        // Suppress sync-on-drop; the write stands (see the module docs on
        // `abort` and the `tx_abort_does_not_roll_back` canary).
        tx.abort();
        self.dirty.insert(si);
        Ok(r)
    }

    /// Append an element, rolling over to a fresh segment when the last one
    /// is full.
    pub(crate) fn push(&mut self, item: T) -> Result<()>
    where
        T: StoreCopy,
    {
        self.grow_for_push(false)?;
        self.segs
            .last_mut()
            .expect("segment exists after grow_for_push")
            .push(item)
    }

    /// Like [`SegVec::push`], but defers durability: the touched segment (and
    /// the directory, on rollover) is only marked dirty. Call
    /// [`SegVec::flush`] to issue one sync per dirty object.
    pub(crate) fn push_nosync(&mut self, item: T) -> Result<()>
    where
        T: StoreCopy,
    {
        self.grow_for_push(true)?;
        let idx = self.segs.len() - 1;
        let seg = self
            .segs
            .last_mut()
            .expect("segment exists after grow_for_push");
        vec_push_nosync(seg, item)?;
        self.dirty.insert(idx);
        Ok(())
    }

    /// Sync every dirty segment, once each, then the directory. The directory
    /// is always synced: deferred directory writes may predate this instance,
    /// so no flag can prove it clean.
    pub(crate) fn flush(&mut self) -> Result<()> {
        // Iterate without draining, and clear only after every sync
        // succeeded: an error mid-loop must leave the dirty set intact so a
        // retried `flush` still syncs every segment with deferred writes.
        for &i in self.dirty.iter() {
            if let Some(seg) = self.segs.get(i) {
                // Safety: the engine is single-threaded per graph handle; no
                // other mapping mutates these objects concurrently.
                unsafe { seg.object().as_mut()?.sync()? };
            }
        }
        self.dirty.clear();
        // Always sync the directory, not just when this instance grew it. A
        // grew-it flag would live on the instance: drop one without flushing
        // and the flag is lost while the writes stay visible in mapped
        // memory, so a later instance would sync segments but never the
        // directory. One extra object sync per `flush` is the cheap side of
        // that trade.
        unsafe { self.dir.object().as_mut()?.sync()? };
        Ok(())
    }

    /// Ensure the last segment has room: create and register a new segment if
    /// the vector is empty or the last segment is full. In `nosync` mode the
    /// new segment and the directory write defer durability until `flush`.
    fn grow_for_push(&mut self, nosync: bool) -> Result<()> {
        if self.segs.last().map_or(false, |s| s.len() < self.cap) {
            return Ok(());
        }
        let seg: VecObject<T, VecObjectAlloc> = if nosync {
            vec_new_nosync(ObjectBuilder::default().persist(true))?
        } else {
            VecObject::new(ObjectBuilder::default().persist(true))?
        };
        let entry = SegEntry {
            raw: seg.object().id().raw(),
        };
        if nosync {
            vec_push_nosync(&mut self.dir, entry)?;
            self.dirty.insert(self.segs.len());
        } else {
            self.dir.push(entry)?;
        }
        self.segs.push(seg);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for `SegVec` in isolation (no graph, no naming). Each test
    //! creates fresh, unregistered objects, so runs are naturally idempotent
    //! (orphaned objects are the same accepted cost as `Graph::reset`).

    use twizzler::marker::Invariant;

    use super::SegVec;

    /// Minimal invariant record for exercising `SegVec` alone.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    #[repr(C)]
    struct TestRec {
        v: u64,
    }
    unsafe impl Invariant for TestRec {}

    /// A SegVec of capacity `cap` holding `0..n`.
    fn filled(cap: usize, n: u64) -> SegVec<TestRec> {
        let mut sv = SegVec::create(cap).expect("create segvec");
        for i in 0..n {
            sv.push(TestRec { v: i }).expect("push");
        }
        sv
    }

    #[test]
    fn segvec_create_empty() {
        let sv: SegVec<TestRec> = SegVec::create(4).unwrap();
        assert_eq!(sv.len(), 0);
        assert_eq!(sv.segments(), 0);
        assert!(sv.get_ref(0).is_none());
    }

    #[test]
    fn segvec_zero_capacity_rejected() {
        assert!(SegVec::<TestRec>::create(0).is_err());
        let dir_raw = {
            let sv = filled(4, 1);
            sv.dir_raw()
        };
        assert!(SegVec::<TestRec>::open(dir_raw, 0).is_err());
    }

    #[test]
    fn segvec_rollover_boundaries() {
        let mut sv = SegVec::create(4).unwrap();
        for i in 0..4 {
            sv.push(TestRec { v: i }).unwrap();
        }
        // Exactly full: still one segment.
        assert_eq!((sv.len(), sv.segments()), (4, 1));
        sv.push(TestRec { v: 4 }).unwrap();
        assert_eq!((sv.len(), sv.segments()), (5, 2));
        for i in 5..8 {
            sv.push(TestRec { v: i }).unwrap();
        }
        assert_eq!((sv.len(), sv.segments()), (8, 2));
        sv.push(TestRec { v: 8 }).unwrap();
        assert_eq!((sv.len(), sv.segments()), (9, 3));
    }

    #[test]
    fn segvec_get_ref_and_bounds() {
        let sv = filled(4, 9);
        for i in [0u64, 3, 4, 7, 8] {
            assert_eq!(sv.get_ref(i as usize).expect("in bounds").v, i);
        }
        assert!(sv.get_ref(9).is_none()); // one past the end (in last segment)
        assert!(sv.get_ref(100).is_none()); // past all segments
    }

    /// In-place mutation, and the durability contract that comes with it.
    ///
    /// `with_mut_at_nosync` is the only in-place mutation `SegVec` has, and
    /// it is not durable until `flush`. That is the property most likely to
    /// be assumed away by a future caller, so it is asserted here rather
    /// than only documented.
    #[test]
    fn segvec_with_mut_at_nosync_defers_durability() {
        let mut sv = filled(4, 9);
        // Mutate inside the second segment; read back.
        sv.with_mut_at_nosync(5, |r| {
            r.v = 55;
            Ok(())
        })
        .unwrap();
        assert_eq!(sv.get_ref(5).unwrap().v, 55, "visible immediately in memory");
        assert_eq!(sv.get_ref(4).unwrap().v, 4, "neighbour untouched");

        // Durability is the caller's job. `flush` is what makes it stick; a
        // within-boot read cannot tell the difference, so this asserts that
        // flush succeeds and leaves the value intact rather than claiming to
        // have proved persistence.
        sv.flush().expect("flush");
        assert_eq!(sv.get_ref(5).unwrap().v, 55);

        // Out of bounds is an error, not a panic.
        assert!(sv.with_mut_at_nosync(100, |_| Ok(())).is_err());
    }

    #[test]
    fn segvec_reopen_preserves_and_appends() {
        let (dir_raw, n) = {
            let sv = filled(4, 9);
            (sv.dir_raw(), sv.len())
        };
        let mut sv = SegVec::<TestRec>::open(dir_raw, 4).expect("open");
        assert_eq!(sv.len(), n);
        assert_eq!(sv.segments(), 3);
        for i in 0..n as u64 {
            assert_eq!(sv.get_ref(i as usize).unwrap().v, i);
        }
        // Appends continue in the last segment, then roll over.
        for i in 9..13 {
            sv.push(TestRec { v: i }).unwrap();
        }
        assert_eq!((sv.len(), sv.segments()), (13, 4));
        assert_eq!(sv.get_ref(12).unwrap().v, 12);
    }

    /// Nosync pushes are immediately visible (mapped memory), rollover
    /// works, and `flush` makes the batch durable for reopen.
    #[test]
    fn segvec_push_nosync_and_flush() {
        let mut sv = SegVec::create(4).unwrap();
        for i in 0..10 {
            sv.push_nosync(TestRec { v: i }).unwrap();
        }
        assert_eq!((sv.len(), sv.segments()), (10, 3));
        assert_eq!(sv.get_ref(9).unwrap().v, 9);
        sv.flush().unwrap();
        let dir = sv.dir_raw();
        drop(sv);
        let sv = SegVec::<TestRec>::open(dir, 4).unwrap();
        assert_eq!(sv.len(), 10);
        assert_eq!(sv.get_ref(4).unwrap().v, 4);
        assert_eq!(sv.get_ref(9).unwrap().v, 9);
    }

    #[test]
    fn segvec_open_with_wrong_capacity_rejected() {
        // 9 elements at cap 4 → segments of 4, 4, 1.
        let dir_raw = {
            let sv = filled(4, 9);
            sv.dir_raw()
        };
        // cap 8 disagrees with the stored geometry (a non-last segment holds 4,
        // not 8): opening must refuse rather than misindex.
        assert!(SegVec::<TestRec>::open(dir_raw, 8).is_err());
        assert!(SegVec::<TestRec>::open(dir_raw, 2).is_err());
        // The recorded capacity still opens fine.
        assert!(SegVec::<TestRec>::open(dir_raw, 4).is_ok());
    }
}
