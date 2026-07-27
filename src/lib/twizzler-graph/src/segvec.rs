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

use std::mem::MaybeUninit;

use twizzler::{
    collections::vec::{Vec as TwzVec, VecObject, VecObjectAlloc},
    error::TwzError,
    marker::{Invariant, StoreCopy},
    object::{MapFlags, ObjID, Object, ObjectBuilder},
    ptr::{Ref, RefMut},
};
use twizzler_rt_abi::error::ArgumentError;

type Result<T> = core::result::Result<T, TwzError>;

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

/// `VecObject::push_ctor` with the sync-on-drop suppressed.
pub(crate) fn vec_push_ctor_nosync<T, F>(
    v: &mut VecObject<T, VecObjectAlloc>,
    ctor: F,
) -> Result<()>
where
    T: Invariant,
    F: FnOnce(RefMut<MaybeUninit<T>>) -> Result<RefMut<T>>,
{
    let mut tx = v.object().as_tx()?;
    tx.base_mut().push_ctor(ctor)?;
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
    /// Whether the directory itself has deferred writes.
    dir_dirty: bool,
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
            dir_dirty: false,
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
            dir_dirty: false,
        })
    }

    /// The directory object's ObjID — what the graph root records.
    pub(crate) fn dir_raw(&self) -> u128 {
        self.dir.object().id().raw()
    }

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

    /// Run `f` on a mutable reference to the element at `idx` (transactional,
    /// so the mutation is synced to the backing store).
    pub(crate) fn with_mut_at<R>(
        &mut self,
        idx: usize,
        f: impl FnOnce(&mut T) -> Result<R>,
    ) -> Result<R> {
        let (si, off) = (idx / self.cap, idx % self.cap);
        let seg = self
            .segs
            .get_mut(si)
            .ok_or(TwzError::from(ArgumentError::InvalidArgument))?;
        seg.with_mut_slice(off..off + 1, |s| f(&mut s[0]))
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

    /// Sync every object with deferred writes, once each: dirty segments and,
    /// if it grew during a batch, the directory.
    pub(crate) fn flush(&mut self) -> Result<()> {
        for i in self.dirty.drain() {
            if let Some(seg) = self.segs.get(i) {
                // Safety: the engine is single-threaded per graph handle; no
                // other mapping mutates these objects concurrently.
                unsafe { seg.object().as_mut()?.sync()? };
            }
        }
        if self.dir_dirty {
            unsafe { self.dir.object().as_mut()?.sync()? };
            self.dir_dirty = false;
        }
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
            self.dir_dirty = true;
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

    #[test]
    fn segvec_with_mut_at() {
        let mut sv = filled(4, 9);
        // Mutate inside the second segment; read back.
        sv.with_mut_at(5, |r| {
            r.v = 55;
            Ok(())
        })
        .unwrap();
        assert_eq!(sv.get_ref(5).unwrap().v, 55);
        assert_eq!(sv.get_ref(4).unwrap().v, 4); // neighbor untouched
                                                 // Out of bounds is an error, not a panic.
        assert!(sv.with_mut_at(100, |_| Ok(())).is_err());
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
