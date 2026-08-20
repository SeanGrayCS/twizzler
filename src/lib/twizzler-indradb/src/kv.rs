//! A sorted key-value store over Twizzler objects (board task D2a).
//!
//! # Read-path instrumentation (E7)
//!
//! [`stats`] counts what a read actually costs, because "the baseline is
//! slower" is not a result and the mechanism is. Two things it exists to
//! separate: **key search** — how many binary-search comparisons, each one a
//! dereference into a 226 MB arena at an unpredictable offset — and **range
//! amplification**, entries materialised against entries the caller consumed.
//! `FIND_MAP_HITS` being zero over a whole query run is itself a finding: the
//! volatile map is built on the write path only, so a read-only boot has no
//! index at all. See [`Kv::build_read_index`].
//!
//! IndraDB's `Transaction` trait is written against sorted KV backends (its
//! upstream one is RocksDB; "and sled" stood here until 2026-08-19, but
//! indradb-lib 5.0.0 ships no sled datastore): it needs ordered iteration
//! (`range_vertices` by UUID, `range_edges` by edge order) and prefix scans.
//! This module provides that substrate directly on Twizzler persistent
//! objects, so the RQ2 comparison contrasts our graph-native `InvPtr`-linked
//! engine against KV-on-objects — the honest framing — rather than lending
//! the baseline our engine's data structures. Nothing here depends on
//! `twizzler-graph`.
//!
//! Layout (two objects, ids recorded by the caller):
//!
//! ```text
//! data:  VecObject<u8>    append-only byte arena: keys and values, never moved
//! index: VecObject<Slot>  slots sorted by key bytes; binary search + scan
//! ```
//!
//! A `Slot` points at the arena rather than inlining bytes, so keys and values
//! are variable-length while every object-resident record stays fixed-size
//! (`Invariant`, no heap pointers). Overwrites append the new value and
//! repoint the slot, orphaning the old bytes; deletes tombstone the slot.
//! Both leave garbage in the arena — compaction is future work, noted rather
//! than hidden, and harmless for the benchmark-sized graphs D2 targets.
//!
//! # Insertion was O(n) per key until 2026-08-11
//!
//! `put` used to keep the index sorted by pushing a slot and then
//! `rotate_right(1)` over everything after the insertion point — an O(n)
//! memmove *over persistent memory*, so a load of n keys was O(n²) and every
//! rotate dirtied index pages that then went through writeback at ~1.2–1.7 MB/s.
//!
//! Measured: loading LDBC SF0.1 started at ~8 vertices/s and decayed to ~2.5/s
//! by 1,265 vertices (~8,800 keys), projecting to **36+ hours for the vertices
//! alone**. The native engine loads the same data in 655 s. That gap was **our
//! adapter's defect, not a property of IndraDB or of KV-on-objects**, and
//! reporting it as a baseline result would have been a comparison quietly wrong
//! in our own favour.
//!
//! Now: slots `[0, sorted)` are key-sorted and `[sorted, len)` is an unsorted
//! unsorted region, with an in-memory hash index over the whole store. `put`
//! appends — no memmove, no sync — and that region
//! is merged into the sorted region when it grows past a fraction of it, so
//! total merge work across a load is linear-ish rather than quadratic. Ordered
//! reads still see one sorted sequence: `collect` reads the prefix in order,
//! scans the (bounded) unsorted region, and sorts the *result*, which is small.
//!
//! Cost note (see board A3): each `put` still issues a small number of object
//! syncs, and IndraDB's write path calls it per record. That is inherent to
//! IndraDB's per-call API and is a fair part of the comparison; the quadratic
//! memmove was not.

use twizzler::{
    collections::vec::{Vec as TwzVec, VecObject, VecObjectAlloc},
    error::TwzError,
    marker::{Invariant, StoreCopy},
    object::{MapFlags, ObjID, Object, ObjectBuilder},
};

type Result<T> = core::result::Result<T, TwzError>;

/// `flags` bit 0: the entry is deleted.
const TOMBSTONE: u32 = 1;

use core::sync::atomic::Ordering::Relaxed;

/// Read-path counters for E7. Process-global rather than per-store: the
/// datastore hands the `KvStore` out through a `RefCell` and re-borrows it once
/// per chunk, so a field would have to cross every borrow boundary for no gain.
///
/// Relaxed ordering throughout — these are read once at the end of a
/// single-threaded run, not used for synchronisation.
pub mod stats {
    use core::sync::atomic::{AtomicU64, Ordering::Relaxed};

    /// Calls to `find` — one per point lookup.
    pub static FINDS: AtomicU64 = AtomicU64::new(0);
    /// Of those, how many the volatile map served. **Zero on a read-only boot**
    /// unless `build_read_index` was called: that is the asymmetry E7 exists to
    /// measure, since the native arm always has its query-entry index by then.
    pub static FIND_MAP_HITS: AtomicU64 = AtomicU64::new(0);
    /// Binary-search key comparisons. Each one dereferences a slot into the
    /// arena at an unpredictable offset, so this is the page-touch proxy.
    pub static SEARCH_CMPS: AtomicU64 = AtomicU64::new(0);
    /// Calls to the range collector.
    pub static SCAN_CALLS: AtomicU64 = AtomicU64::new(0);
    /// Entries copied out of the store by those calls.
    pub static SCAN_MATERIALIZED: AtomicU64 = AtomicU64::new(0);
    /// Entries the cursor actually yielded. The gap against
    /// `SCAN_MATERIALIZED` is the range amplification — work done for a caller
    /// that walked away.
    pub static SCAN_YIELDED: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {
        for c in [
            &FINDS,
            &FIND_MAP_HITS,
            &SEARCH_CMPS,
            &SCAN_CALLS,
            &SCAN_MATERIALIZED,
            &SCAN_YIELDED,
        ] {
            c.store(0, Relaxed);
        }
    }

    /// Three lines, prefixed so a run can be grepped out of a serial log.
    pub fn report(label: &str) {
        let finds = FINDS.load(Relaxed);
        let hits = FIND_MAP_HITS.load(Relaxed);
        let cmps = SEARCH_CMPS.load(Relaxed);
        let calls = SCAN_CALLS.load(Relaxed);
        let mat = SCAN_MATERIALIZED.load(Relaxed);
        let yld = SCAN_YIELDED.load(Relaxed);
        println!(
            "GSTRESS KVSTATS {label} finds={finds} map_hits={hits} indexed={}",
            finds > 0 && hits == finds
        );
        println!(
            "GSTRESS KVSTATS {label} search_cmps={cmps} per_find={:.1}",
            if finds > 0 {
                cmps as f64 / finds as f64
            } else {
                0.0
            }
        );
        println!(
            "GSTRESS KVSTATS {label} scan_calls={calls} materialized={mat} yielded={yld} amplification={:.1}x",
            if yld > 0 { mat as f64 / yld as f64 } else { 0.0 }
        );
    }
}

/// `VecObject::push` **without the per-call object sync.**
///
/// `VecObject::push`/`::append`/`::with_mut_slice` all route through
/// `Object::with_tx`, whose `TxObject` has `sync_on_drop = true` — so the
/// *whole object* is synced when the closure ends. One `put` of a new key cost
/// three such syncs, and at ~7 keys per LDBC row that is ~21 object syncs per
/// row against a writeback path measured at 1.2-1.7 MB/s (`tasks.md`, A8-AC8).
///
/// `abort()` suppresses the sync **without rolling back** — the guarantee
/// `twizzler-graph`'s `tx_abort_does_not_roll_back` test pins upstream, and the
/// same mechanism `SegVec::push_nosync` uses. [`KvStore::flush`] is the
/// durability point instead, reached via `Datastore::sync`.
fn push_nosync<T: Invariant + StoreCopy>(
    v: &mut VecObject<T, VecObjectAlloc>,
    val: T,
) -> Result<()> {
    let mut tx = v.object().as_tx()?;
    tx.base_mut().push(val)?;
    tx.abort();
    Ok(())
}

/// Append bytes to a byte vector object under a single aborted transaction.
fn extend_nosync(v: &mut VecObject<u8, VecObjectAlloc>, bytes: &[u8]) -> Result<()> {
    let mut tx = v.object().as_tx()?;
    for b in bytes {
        tx.base_mut().push(*b)?;
    }
    tx.abort();
    Ok(())
}

/// In-place slot mutation without the object sync.
fn with_mut_slice_nosync<T: Invariant, R>(
    v: &mut VecObject<T, VecObjectAlloc>,
    range: core::ops::Range<usize>,
    f: impl FnOnce(&mut [T]) -> Result<R>,
) -> Result<R> {
    let mut tx = v.object().as_tx()?;
    let r = tx.base_mut().with_mut_slice(range, f)?;
    tx.abort();
    Ok(r)
}

fn rw() -> MapFlags {
    MapFlags::READ | MapFlags::WRITE | MapFlags::PERSIST
}

/// One index entry: where the key and value live in the arena.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct Slot {
    key_off: u64,
    val_off: u64,
    key_len: u32,
    val_len: u32,
    flags: u32,
}
unsafe impl Invariant for Slot {}

/// A sorted key-value store; see the module docs.
pub(crate) struct KvStore {
    data: VecObject<u8, VecObjectAlloc>,
    index: VecObject<Slot, VecObjectAlloc>,
    /// Slots `[0, sorted)` are sorted by key bytes; `[sorted, len)` is an
    /// unsorted region. Not persisted — [`Self::open`] merges, so a store on disk
    /// is always fully sorted and the invariant cannot be broken by a crash.
    sorted: usize,
    /// `key -> slot index`, covering **every** entry, live or tombstoned.
    ///
    /// *Was tail-only.* That made a lookup O(1) while the tail still held
    /// everything, and a persistent binary search the moment it merged — one
    /// random page touch per probe, into an arena where keys sit in insertion
    /// order rather than key order. The LDBC load's rate cliff is at ~8,200
    /// keys, which was the old floor: the first merge.
    ///
    /// Volatile, and rebuilt on `open` — exactly as the engine's own vertex
    /// index has been since A8. That symmetry is the point; otherwise the two
    /// arms are compared with their indexes in different places.
    /// **Lazy.** `None` means "not built"; a read-only boot never builds it.
    ///
    /// The 2026-08-10 query boot died inside `open` before running a query,
    /// having rebuilt this over ~5.6 M keys (~600 MB at ~110 B/entry) for a
    /// workload that barely needs it: `by_id`, `out_edges`, `in_edges` and
    /// `vprop` are prefix scans, and the one point lookup (`vertex_type`) is
    /// answerable by binary search over the sorted region. The map is for the
    /// *write* path, so it is now built on the first write and not before.
    map: Option<std::collections::HashMap<Vec<u8>, usize>>,
}

/// Merge the unsorted region once it exceeds this fraction of the sorted one,
/// or the floor below — whichever is larger.
///
/// Proportional rather than fixed: a fixed threshold T makes merges happen n/T
/// times at O(n) each, which is quadratic again with a smaller constant.
///
/// **Doubling, not eighths, and a floor of 1M.** A merge rewrites *every* slot,
/// so the whole index object is dirtied and written back at ~1 MB/s. The
/// 2026-08-10 run shows exactly that: the index sync grew 36 -> 74 -> 86 -> 88
/// -> 97 -> 114 MB while the data object stayed at 5-14 MB. At the old
/// threshold that is ~37 merges over SF0.1, ~1.4 GB of index writeback, for
/// work a bulk load never uses — point lookups go through `map`, and ordering
/// matters only to scans. Doubling makes it ~3 merges, and `open` restores full
/// sortedness anyway, so a load-then-reboot never pays for the rest.
/// Slots per write-back transaction in [`KvStore::merge`].
///
/// 8,192 slots = 256 KB. Small enough that no transaction spans a large range,
/// large enough that the per-transaction overhead is amortised over thousands of
/// elements rather than paid per slot as the load's `push_nosync` does.
const MERGE_CHUNK: usize = 8192;
const MERGE_FRACTION: usize = 1;
const MERGE_FLOOR: usize = 1 << 20;

impl KvStore {
    /// Create a new, empty store (two fresh persistent objects).
    pub(crate) fn create() -> Result<Self> {
        Ok(KvStore {
            data: VecObject::new(ObjectBuilder::default().persist(true))?,
            index: VecObject::new(ObjectBuilder::default().persist(true))?,
            sorted: 0,
            // A fresh store is trivially mapped: empty and correct.
            map: Some(std::collections::HashMap::new()),
        })
    }

    /// Reopen a store from the object ids returned by [`KvStore::ids`], with
    /// the `sorted` boundary the last `sync` persisted.
    ///
    /// *Was `sorted = 0` plus a full merge.* Recovering sortedness from nothing
    /// meant re-sorting the entire store on every open — 5.6 M slots with a
    /// comparator that random-reads the whole data arena. Persisting the
    /// boundary leaves the store as two sorted runs, which `sort_by` (a
    /// run-detecting merge sort) folds in one pass.
    ///
    /// `sorted` is still *repaired* rather than trusted blindly: a crash
    /// mid-load leaves a tail, and the merge below absorbs it. An out-of-range
    /// value is treated as 0 — see the comment on the assignment.
    pub(crate) fn open(data_raw: u128, index_raw: u128, sorted: usize) -> Result<Self> {
        let mut kv = KvStore {
            data: VecObject::from(Object::<TwzVec<u8, VecObjectAlloc>>::map(
                ObjID::new(data_raw),
                rw(),
            )?),
            index: VecObject::from(Object::<TwzVec<Slot, VecObjectAlloc>>::map(
                ObjID::new(index_raw),
                rw(),
            )?),
            sorted: 0,
            map: None,
        };
        // **An out-of-range boundary falls back to 0, not to `len`.**
        //
        // The two directions are not symmetric. Understating is always safe —
        // the merge below absorbs whatever tail is left. Overstating never is:
        // `find` would binary-search slots that are not in key order and miss
        // keys that are present, and `collect` would short-circuit early. So a
        // value that cannot be vouched for is repaired to "nothing is sorted",
        // which costs one merge and is always correct.
        //
        // Clamping to `len` instead — the first version of this — turned a
        // garbage value into a *plausible* one, which is worse than either:
        // `kv_open_repairs_an_understated_sorted_boundary` caught it returning
        // `None` for a key that was present.
        kv.sorted = if sorted <= kv.index.len() { sorted } else { 0 };

        // **Diagnostic, deliberately in the library and deliberately before the
        // merge.** The 2026-08-10 query boot exhausted the frame pool inside
        // this function, so no harness code ever ran and the store's real size
        // was left to inference — which was out by ~7x. Printing after the
        // merge would report nothing on the run that matters.
        let (dbytes, ibytes) = kv.sizes();
        println!(
            "KV open: {:.0}MB data + {:.0}MB index ({} slots), sorted={} of {}, \
             tail={}",
            dbytes as f64 / 1e6,
            ibytes as f64 / 1e6,
            kv.index.len(),
            kv.sorted,
            kv.index.len(),
            kv.index.len() - kv.sorted
        );

        kv.merge()?;
        Ok(kv)
    }

    /// The sorted-prefix boundary, for persisting in the datastore root.
    pub(crate) fn sorted_len(&self) -> usize {
        self.sorted
    }

    /// Live byte counts of the two objects, for progress reporting. The
    /// arena is append-only, so `data` includes garbage left by overwrites.
    pub(crate) fn sizes(&self) -> (usize, usize) {
        (self.data.len(), self.index.len() * core::mem::size_of::<Slot>())
    }

    /// The (data, index) object ids — persist these to find the store again.
    pub(crate) fn ids(&self) -> (u128, u128) {
        (
            self.data.object().id().raw(),
            self.index.object().id().raw(),
        )
    }

    /// Number of live (non-tombstoned) entries.
    ///
    /// Kept though the datastore counts via namespace scans: a sorted KV
    /// store without a live count is an odd thing to hand to the next caller,
    /// and D2's bulk-insert work (AC6) wants it for progress reporting.
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        let slots = self.index.as_slice();
        slots
            .as_slice()
            .iter()
            .filter(|s| s.flags & TOMBSTONE == 0)
            .count()
    }

    /// Slot index for `key`.
    ///
    /// O(1) once the map is built. Without it — a read-only boot — this is a
    /// binary search over the sorted region plus a scan of whatever tail
    /// remains, which costs page touches but allocates nothing.
    fn find(&self, key: &[u8]) -> Option<usize> {
        stats::FINDS.fetch_add(1, Relaxed);
        if let Some(m) = &self.map {
            stats::FIND_MAP_HITS.fetch_add(1, Relaxed);
            return m.get(key).copied();
        }
        let data = self.data.as_slice();
        let arena = data.as_slice();
        let idx = self.index.as_slice();
        let slots = idx.as_slice();
        if let Ok(i) = Self::search(&slots[..self.sorted], arena, key) {
            return Some(i);
        }
        slots[self.sorted..]
            .iter()
            .position(|sl| Self::key_of(arena, sl) == key)
            .map(|i| i + self.sorted)
    }

    /// Build the volatile map if absent. Called on the **write** path only.
    fn ensure_map(&mut self) -> Result<()> {
        if self.map.is_some() {
            return Ok(());
        }
        self.rebuild_map();
        Ok(())
    }

    /// Build the volatile map for a **read-only** boot (E7).
    ///
    /// Without this a query boot never builds the map — `ensure_map` is on the
    /// write path — so every lookup binary-searches the whole sorted region,
    /// touching the arena at ~log2(n) unpredictable offsets. The native arm
    /// meanwhile builds its query-entry index at open and has that cost
    /// excluded from the reported latencies. **Excluding both setup costs is
    /// only symmetric if both arms actually get an index**, which is what this
    /// makes possible; call it at open and exclude it exactly as the native
    /// index build is excluded.
    ///
    /// It is not free and the cost is itself a result: the map holds one entry
    /// per key — 5.49 M at SF0.1 against the native index's 327 588 roots —
    /// because a KV store must index every record while index-free adjacency
    /// indexes only query entry points.
    pub fn build_read_index(&mut self) {
        if self.map.is_none() {
            self.rebuild_map();
        }
    }

    /// Rebuild the volatile map from the current slot order.
    ///
    /// **Required after any `merge`**, which permutes slots: a map holding slot
    /// indices is stale the instant they move, and a stale index is silent —
    /// it still returns *a* slot, just the wrong one.
    fn rebuild_map(&mut self) {
        let map = {
            let data = self.data.as_slice();
            let arena = data.as_slice();
            let idx = self.index.as_slice();
            let slots = idx.as_slice();
            let mut m = std::collections::HashMap::with_capacity(slots.len());
            for (i, sl) in slots.iter().enumerate() {
                m.insert(Self::key_of(arena, sl).to_vec(), i);
            }
            m
        };
        self.map = Some(map);
    }

    /// The value for `key`, or `None` if absent or deleted.
    pub(crate) fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let i = self.find(key)?;
        let data = self.data.as_slice();
        let arena = data.as_slice();
        let idx = self.index.as_slice();
        let slots = idx.as_slice();
        let s = &slots[i];
        if s.flags & TOMBSTONE != 0 {
            None
        } else {
            Some(Self::val_of(arena, s).to_vec())
        }
    }

    /// Sort the unsorted region into the sorted one, so ordered reads can
    /// binary-search.
    ///
    /// **The sort happens in plain memory, not inside the transaction.**
    ///
    /// Sorting the persistent slice in place — `with_mut_slice(0..end, sort_by)`
    /// — performs O(n log n) element *moves* under one `TxObject`: ~120 M writes
    /// for 5.49 M slots. That is what made SF0.1 unopenable, twice: ~9.1 GB of
    /// frames on a 402 MB store, in a fresh boot with no volatile map. The
    /// load's 4.19 M-slot merge survived and the query's 5.49 M-slot merge did
    /// not — a 31% larger input, which fits a superlinear cost.
    ///
    /// Copying out, sorting, and writing back once costs ~5.5 M writes plus one
    /// bulk copy. **This is justified by doing less work inside the transaction,
    /// not by any claim about what the transaction charges per write** — that is
    /// still unknown, and worth a dedicated probe (see `ADAPTER-FIX.md`).
    ///
    /// Costs a `Vec<Slot>` of the whole index (~176 MB at SF0.1), which is the
    /// trade: bounded, predictable heap in exchange for an unbounded and
    /// unexplained transaction cost.
    fn merge(&mut self) -> Result<()> {
        let end = self.index.len();
        if self.sorted == end {
            return Ok(());
        }
        // Phase markers for large merges only (silent in tests). Two boots
        // have now been spent inferring which part of this function costs
        // ~9.1 GB; printing is cheaper than a third.
        let loud = end > (1 << 20);
        if loud {
            println!("KV merge: {end} slots, tail {}, copying out", end - self.sorted);
        }
        let mut slots: Vec<Slot> = {
            let idx = self.index.as_slice();
            idx.as_slice().to_vec()
        };
        if loud {
            println!("KV merge: copied out, sorting");
        }
        {
            let data = self.data.as_slice();
            let arena = data.as_slice();
            // The arena is append-only and `merge` appends nothing, so keys do
            // not move while the slots are permuted.
            slots.sort_by(|a, b| Self::key_of(arena, a).cmp(Self::key_of(arena, b)));
        }
        if loud {
            println!(
                "KV merge: sorted, writing back in {} chunks of {MERGE_CHUNK}",
                end.div_ceil(MERGE_CHUNK)
            );
        }
        // **Write back in bounded chunks, one transaction each.**
        //
        // Sorting outside the transaction was not enough — the boot still died
        // at ~9.1 GB with a single `0..5_486_603` range. What distinguishes the
        // path that *works*: the load calls `push_nosync` ~5.49 M times on this
        // same object as it grows to 176 MB, and completes. So neither `as_tx`
        // nor object size is the cost; the **range handed to `with_mut_slice`**
        // is. `TxRefSlice::from_ref(r, len).slice(range).as_slice_mut()` over a
        // 5.49 M-element range is the one thing the working path never does.
        for start in (0..end).step_by(MERGE_CHUNK) {
            let stop = (start + MERGE_CHUNK).min(end);
            with_mut_slice_nosync(&mut self.index, start..stop, |s| {
                s.copy_from_slice(&slots[start..stop]);
                Ok(())
            })?;
        }
        if loud {
            println!("KV merge: done");
        }
        self.sorted = end;
        // **Invalidate, don't rebuild.** Merge permutes slots, so the map's
        // indices are stale; but a read-only boot has no map and must not be
        // made to build one. The next write rebuilds it.
        self.map = None;
        Ok(())
    }

    /// **The durability point.** Syncs each object once, in place of the
    /// per-operation syncs that `nosync` suppressed. Reached via
    /// `Datastore::sync`, which is what IndraDB's API provides it for.
    pub(crate) fn flush(&mut self) -> Result<()> {
        // Safety: single-threaded per datastore handle, and no other mapping
        // mutates these objects concurrently - the same argument `SegVec::flush`
        // makes.
        unsafe {
            self.data.object().as_mut()?.sync()?;
            self.index.object().as_mut()?.sync()?;
        }
        Ok(())
    }

    /// Insert or overwrite `key`.
    pub(crate) fn put(&mut self, key: &[u8], val: &[u8]) -> Result<()> {
        // The write path is what the map exists for; build it here rather than
        // at `open`, so a query-only boot never pays for it.
        self.ensure_map()?;
        // Append the value (and the key, if this is a new entry) to the arena
        // first: slots must never point at bytes that do not exist yet.
        let val_off = self.data.len() as u64;
        extend_nosync(&mut self.data, val)?;

        match self.find(key) {
            // Existing key (live or tombstoned): repoint at the new value and
            // clear the tombstone. The old value's bytes become garbage.
            Some(i) => {
                with_mut_slice_nosync(&mut self.index, i..i + 1, |s| {
                    s[0].val_off = val_off;
                    s[0].val_len = val.len() as u32;
                    s[0].flags &= !TOMBSTONE;
                    Ok(())
                })?;
            }
            // New key: append its bytes and push the slot onto the unsorted
            // region. No memmove — that was the O(n) step that made a load
            // quadratic — and no sync, which was the next one.
            None => {
                let key_off = self.data.len() as u64;
                extend_nosync(&mut self.data, key)?;
                let slot = Slot {
                    key_off,
                    val_off,
                    key_len: key.len() as u32,
                    val_len: val.len() as u32,
                    flags: 0,
                };
                push_nosync(&mut self.index, slot)?;
                if let Some(m) = &mut self.map {
                    m.insert(key.to_vec(), self.index.len() - 1);
                }
                if self.index.len() - self.sorted
                    > MERGE_FLOOR.max(self.sorted / MERGE_FRACTION)
                {
                    self.merge()?;
                    // **Flush immediately**: the merge just dirtied every slot,
                    // and letting that ride on top of the deferred writes is
                    // what exhausted the frame pool on 2026-08-10. Deliberately
                    // here rather than inside `merge`, so `open`'s merge stays
                    // a read-only operation.
                    self.flush()?;
                }
            }
        }
        Ok(())
    }

    /// Tombstone `key`. Returns whether a live entry was removed.
    pub(crate) fn delete(&mut self, key: &[u8]) -> Result<bool> {
        let found = self.find(key).filter(|i| {
            let idx = self.index.as_slice();
            idx.as_slice()[*i].flags & TOMBSTONE == 0
        });
        let Some(i) = found else { return Ok(false) };
        with_mut_slice_nosync(&mut self.index, i..i + 1, |s| {
            s[0].flags |= TOMBSTONE;
            Ok(())
        })?;
        Ok(true)
    }

    /// All live entries with `key >= start`, in key order.
    ///
    /// The datastore always bounds its scans to one namespace
    /// ([`Self::scan_range_limited`]), so this open-ended form is currently only
    /// exercised by tests — kept because "everything from here on" is the
    /// primitive the bounded variants are built from, and removing it would
    /// leave them looking arbitrary.
    #[allow(dead_code)]
    pub(crate) fn scan_from(&self, start: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.collect(start, None)
    }

    /// All live entries whose key begins with `prefix`, in key order.
    pub(crate) fn scan_prefix(&self, prefix: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.collect(prefix, Some(prefix))
    }

    /// Entries `>= start` that also begin with `prefix`, in key order, stopping
    /// after `limit` live entries.
    ///
    /// This is the shape IndraDB's `range_*` methods need — "everything at or
    /// after this value", without spilling into the next key namespace — and
    /// **the limit is about peak memory, not speed.** `range_edges` asks for
    /// "every edge from this offset on", which at SF0.1 is up to 1.48 M entries,
    /// each an allocated key `Vec`, materialized so IndraDB's executor can
    /// take-while a handful off the front. A real KV backend hands back a lazy
    /// range cursor; this lets the datastore fake one in chunks.
    ///
    /// *An unlimited `scan_range` used to sit alongside this.* It became
    /// unreachable once every caller went through the chunked cursor, and two
    /// spellings of the same call is how `resolve`/`resolve_mut` went wrong
    /// (§7 of `docs/HANDOFF.md`). Pass `usize::MAX` for the whole range.
    pub(crate) fn scan_range_limited(
        &self,
        start: &[u8],
        prefix: &[u8],
        limit: usize,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.collect_limited(start, Some(prefix), limit)
    }


    /// Every live entry, in key order.
    #[cfg(test)]
    pub(crate) fn scan_all(&self) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.collect(&[], None)
    }

    // --- internals ---------------------------------------------------------

    fn collect(&self, start: &[u8], prefix: Option<&[u8]>) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.collect_limited(start, prefix, usize::MAX)
    }

    /// `limit` is honoured **only when the store is fully sorted.** With an
    /// unsorted region present, ordering the result requires seeing all of it,
    /// so a limit would silently drop entries that belong in the first `limit`.
    /// Counting wrapper — every materialisation goes through here, so the
    /// entries-produced side of the E7 ratio has one place to be measured.
    fn collect_limited(
        &self,
        start: &[u8],
        prefix: Option<&[u8]>,
        limit: usize,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        stats::SCAN_CALLS.fetch_add(1, Relaxed);
        let out = self.collect_limited_inner(start, prefix, limit);
        stats::SCAN_MATERIALIZED.fetch_add(out.len() as u64, Relaxed);
        out
    }

    fn collect_limited_inner(
        &self,
        start: &[u8],
        prefix: Option<&[u8]>,
        limit: usize,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        let data = self.data.as_slice();
        let arena = data.as_slice();
        let idx = self.index.as_slice();
        let slots = idx.as_slice();

        let from = match Self::search(&slots[..self.sorted], arena, start) {
            Ok(i) => i,
            Err(i) => i,
        };
        let mut out = Vec::new();
        for s in &slots[from..self.sorted] {
            let k = Self::key_of(arena, s);
            if let Some(p) = prefix {
                if !k.starts_with(p) {
                    break; // sorted: past the prefix range
                }
            }
            if s.flags & TOMBSTONE == 0 {
                out.push((k.to_vec(), Self::val_of(arena, s).to_vec()));
                if out.len() >= limit && self.sorted == slots.len() {
                    return out;
                }
            }
        }
        // **The unsorted region cannot be short-circuited** — every entry is
        // examined. It is bounded by `MERGE_FLOOR.max(sorted/8)`, and it is
        // empty after `open` and after any merge, so ordered reads on a loaded
        // store pay nothing for it.
        for s in &slots[self.sorted..] {
            let k = Self::key_of(arena, s);
            if k < start {
                continue;
            }
            if let Some(p) = prefix {
                if !k.starts_with(p) {
                    continue;
                }
            }
            if s.flags & TOMBSTONE == 0 {
                out.push((k.to_vec(), Self::val_of(arena, s).to_vec()));
            }
        }
        if self.sorted != slots.len() {
            // Sorting the *result* — small — rather than the store.
            out.sort_by(|a, b| a.0.cmp(&b.0));
        }
        out
    }

    /// Binary search by key bytes. `Ok(i)` = exact slot, `Err(i)` = insertion
    /// point. Tombstoned slots participate: they keep their sorted position
    /// so a later `put` of the same key revives the entry in place.
    fn search(slots: &[Slot], arena: &[u8], key: &[u8]) -> core::result::Result<usize, usize> {
        slots.binary_search_by(|s| {
            stats::SEARCH_CMPS.fetch_add(1, Relaxed);
            Self::key_of(arena, s).cmp(key)
        })
    }

    fn key_of<'a>(arena: &'a [u8], s: &Slot) -> &'a [u8] {
        let off = s.key_off as usize;
        &arena[off..off + s.key_len as usize]
    }

    fn val_of<'a>(arena: &'a [u8], s: &Slot) -> &'a [u8] {
        let off = s.val_off as usize;
        &arena[off..off + s.val_len as usize]
    }
}

#[cfg(test)]
mod tests {
    //! D2a acceptance tests. These run on Twizzler under `--tests`; each
    //! creates fresh unregistered objects, so runs are naturally idempotent.

    use super::KvStore;

    fn kv() -> KvStore {
        KvStore::create().expect("create kv store")
    }

    /// **D2c-AC1: deferral is about durability, not visibility.**
    ///
    /// Writes no longer sync per operation, so the risk this pins is a store
    /// that defers *the write itself* rather than just its flush. Every read
    /// path must see an unflushed write.
    #[test]
    fn kv_writes_are_visible_before_flush() {
        let mut s = kv();
        for i in 0..40u32 {
            s.put(format!("k{i:03}").as_bytes(), format!("v{i}").as_bytes())
                .unwrap();
        }
        assert_eq!(s.get(b"k017").as_deref(), Some(&b"v17"[..]));
        assert_eq!(s.len(), 40);
        assert_eq!(s.scan_all().len(), 40);
        assert!(s.delete(b"k017").unwrap());
        assert_eq!(s.get(b"k017"), None);
        s.flush().unwrap();
        assert_eq!(s.get(b"k017"), None, "flush must not resurrect");
        assert_eq!(s.len(), 39);
    }

    /// **D2c-AC2: a flushed store reopens intact.** Without this, AC1 could be
    /// satisfied by a store that never writes anything at all.
    #[test]
    fn kv_flush_then_reopen_roundtrips() {
        let (data, index) = {
            let mut s = kv();
            for i in 0..40u32 {
                s.put(format!("k{i:03}").as_bytes(), format!("v{i}").as_bytes())
                    .unwrap();
            }
            s.delete(b"k000").unwrap();
            s.flush().unwrap();
            s.ids()
        };
        // `sorted = 0`: nothing merged before the flush, which is exactly the
        // crash-mid-load shape. `open` must repair it.
        let s = KvStore::open(data, index, 0).expect("reopen");
        assert_eq!(s.len(), 39);
        assert_eq!(s.get(b"k000"), None, "tombstone survived");
        assert_eq!(s.get(b"k039").as_deref(), Some(&b"v39"[..]));
        let keys: Vec<Vec<u8>> = s.scan_all().into_iter().map(|(k, _)| k).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }

    /// **D2c-AC3: the volatile map covers every live key, not just a tail.**
    ///
    /// `merge` permutes slots, so a map holding slot indices is stale the
    /// instant it runs. This drives a merge directly and then checks that
    /// lookups still resolve — the bug it guards against is silent, since a
    /// stale index still returns *a* slot, just the wrong one.
    #[test]
    fn kv_lookup_map_survives_merge() {
        let mut s = kv();
        for i in (0..64u32).rev() {
            s.put(format!("k{i:03}").as_bytes(), format!("v{i}").as_bytes())
                .unwrap();
        }
        s.merge().unwrap();
        for i in 0..64u32 {
            assert_eq!(
                s.get(format!("k{i:03}").as_bytes()).as_deref(),
                Some(format!("v{i}").as_bytes()),
                "key {i} resolved to the wrong slot after merge"
            );
        }
        // An overwrite after a merge must still hit the existing slot rather
        // than appending a duplicate.
        s.put(b"k007", b"new").unwrap();
        assert_eq!(s.get(b"k007").as_deref(), Some(&b"new"[..]));
        assert_eq!(s.len(), 64);
    }

    /// D2a-AC1: point ops round-trip; overwrite replaces without growing the
    /// live count; deleting a missing key is `false`, not an error.
    #[test]
    fn kv_put_get_overwrite_delete() {
        let mut s = kv();
        assert_eq!(s.len(), 0);
        assert_eq!(s.get(b"missing"), None);

        s.put(b"k", b"v1").unwrap();
        assert_eq!(s.get(b"k").as_deref(), Some(&b"v1"[..]));
        assert_eq!(s.len(), 1);

        s.put(b"k", b"v2-longer").unwrap();
        assert_eq!(s.get(b"k").as_deref(), Some(&b"v2-longer"[..]));
        assert_eq!(s.len(), 1, "overwrite must not add an entry");

        assert!(!s.delete(b"nope").unwrap(), "deleting absent key is false");
        assert!(s.delete(b"k").unwrap());
        assert_eq!(s.get(b"k"), None);
        assert_eq!(s.len(), 0);
        assert!(!s.delete(b"k").unwrap(), "double delete is false");
    }

    /// D2a-AC2: byte-lexicographic ordering, including the prefix-vs-longer
    /// case and keys containing zero bytes.
    /// **The tail must be invisible.** Insertion now leaves an unsorted region,
    /// so every read path has to merge it back conceptually: `scan_all` must be
    /// sorted and complete regardless of how many entries are unmerged.
    ///
    /// Enough keys to cross `MERGE_FLOOR` would be slow in a unit test, so
    /// `kv_lookup_map_survives_merge` drives the boundary directly instead.
    #[test]
    fn kv_tail_is_invisible_to_ordered_reads() {
        let mut kv = kv();
        // Reverse insertion order: with the old rotate this was the worst case,
        // and with a tail it is the case where sortedness is purely the read
        // path's doing.
        for i in (0..64u32).rev() {
            kv.put(format!("k{i:04}").as_bytes(), format!("v{i}").as_bytes())
                .unwrap();
        }
        let all = kv.scan_all();
        assert_eq!(all.len(), 64);
        let keys: Vec<Vec<u8>> = all.iter().map(|(k, _)| k.clone()).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "ordered reads must not expose the tail's order");

        // Prefix scans short-circuit on the sorted region but cannot on the
        // tail; both halves must still be found. `k000` selects k0000-k0009 —
        // note `k00` would select all 64, which is what this assertion first
        // claimed was 10.
        kv.put(b"other", b"x").unwrap();
        assert_eq!(kv.scan_prefix(b"k000").len(), 10);
        assert_eq!(kv.scan_prefix(b"k0").len(), 64, "`other` must not match");
    }

    /// Reads must find a key whether it is in the sorted region or the tail,
    /// and an overwrite of a tail key must not create a second entry.
    #[test]
    fn kv_get_and_overwrite_span_sorted_and_tail() {
        let mut kv = kv();
        kv.put(b"a", b"1").unwrap();
        kv.merge().unwrap(); // `a` is now in the sorted region
        kv.put(b"b", b"2").unwrap(); // `b` is in the tail

        assert_eq!(kv.get(b"a").as_deref(), Some(&b"1"[..]));
        assert_eq!(kv.get(b"b").as_deref(), Some(&b"2"[..]));

        kv.put(b"b", b"22").unwrap();
        assert_eq!(kv.get(b"b").as_deref(), Some(&b"22"[..]));
        assert_eq!(kv.scan_all().len(), 2, "overwrite must not duplicate");

        kv.put(b"a", b"11").unwrap();
        assert_eq!(kv.get(b"a").as_deref(), Some(&b"11"[..]));
        assert_eq!(kv.scan_all().len(), 2);
    }

    /// A delete must work on a tail entry, and survive the merge that follows.
    #[test]
    fn kv_delete_in_tail_survives_merge() {
        let mut kv = kv();
        kv.put(b"keep", b"1").unwrap();
        kv.put(b"gone", b"2").unwrap();
        assert!(kv.delete(b"gone").unwrap());
        kv.merge().unwrap();
        assert_eq!(kv.get(b"gone"), None);
        assert_eq!(kv.get(b"keep").as_deref(), Some(&b"1"[..]));
        assert_eq!(kv.scan_all().len(), 1);
    }

    /// `merge` is idempotent and leaves the store fully sorted, which is the
    /// invariant `open` relies on to avoid persisting `sorted`.
    #[test]
    fn kv_merge_is_idempotent_and_total() {
        let mut kv = kv();
        for i in 0..32u32 {
            kv.put(format!("k{i:04}").as_bytes(), b"v").unwrap();
        }
        kv.merge().unwrap();
        let after_one = kv.scan_all();
        kv.merge().unwrap();
        assert_eq!(kv.scan_all(), after_one);
        assert_eq!(kv.sorted, kv.index.len(), "merge must leave nothing unsorted");
        // The map is *rebuilt* by merge, not cleared: it covers the whole store.
        // Merge invalidates the map; the next write rebuilds it. What must
        // hold is that lookups still resolve, which the loop above checked.
        assert!(kv.map.is_none(), "merge must invalidate the stale map");
        kv.put(b"k0031", b"again").unwrap();
        assert_eq!(kv.map.as_ref().map(|m| m.len()), Some(32));
    }

    /// **D2c: an under-stated `sorted` boundary is repaired by `open`.**
    ///
    /// This is the shape a crash between a write and a `sync` leaves behind:
    /// the root claims less sortedness than the index has, and the merge
    /// absorbs the difference. An out-of-range claim falls back to 0 and is
    /// re-sorted, so it must read back identically too.
    ///
    /// **An in-range *over*-claim is trusted and is not tested here**, because
    /// it is not defended: detecting one needs a full ordering pass, which
    /// costs what the merge costs, so the boundary is a performance hint whose
    /// only writer is `sync`. `VERSION` guards the format that carries it. This
    /// is why out-of-range repairs to 0 rather than to `len` — the latter would
    /// manufacture exactly the in-range over-claim that nothing can detect.
    #[test]
    fn kv_open_repairs_an_understated_sorted_boundary() {
        for claim in [0usize, 7, 999_999] {
            let (data, index) = {
                // **A fresh store per claim.** Reusing one would let the first
                // `open` sort the index and make later claims pass on already
                // -repaired data — true, but for the wrong reason.
                let mut s = kv();
                for i in (0..40u32).rev() {
                    s.put(format!("k{i:03}").as_bytes(), b"v").unwrap();
                }
                s.flush().unwrap();
                s.ids()
            };
            let s = KvStore::open(data, index, claim).expect("reopen");
            assert_eq!(s.len(), 40, "claim {claim}");
            assert_eq!(
                s.get(b"k017").as_deref(),
                Some(&b"v"[..]),
                "claim {claim} lost a key"
            );
            let keys: Vec<Vec<u8>> = s.scan_all().into_iter().map(|(k, _)| k).collect();
            let mut sorted = keys.clone();
            sorted.sort();
            assert_eq!(keys, sorted, "claim {claim} broke scan order");
        }
    }

    #[test]
    fn kv_sorted_order() {
        let mut s = kv();
        // Inserted deliberately out of order.
        for k in [
            &b"b"[..],
            &b"a"[..],
            &b"ab"[..],
            &b"a\x00b"[..],
            &b"\x00"[..],
        ] {
            s.put(k, b"x").unwrap();
        }
        let keys: Vec<Vec<u8>> = s.scan_all().into_iter().map(|(k, _)| k).collect();
        assert_eq!(
            keys,
            vec![
                b"\x00".to_vec(),
                b"a".to_vec(),
                b"a\x00b".to_vec(),
                b"ab".to_vec(),
                b"b".to_vec(),
            ]
        );
    }

    /// D2a-AC3: `scan_from` is inclusive of the start key; `scan_prefix`
    /// stops at the prefix boundary.
    #[test]
    fn kv_scan_from_and_prefix() {
        let mut s = kv();
        for k in [
            &b"V:1"[..],
            &b"V:2"[..],
            &b"V:10"[..],
            &b"E:1"[..],
            &b"W:1"[..],
        ] {
            s.put(k, b"x").unwrap();
        }

        // scan_from is >= start, in order.
        let from: Vec<Vec<u8>> = s.scan_from(b"V:").into_iter().map(|(k, _)| k).collect();
        assert_eq!(
            from,
            vec![
                b"V:1".to_vec(),
                b"V:10".to_vec(),
                b"V:2".to_vec(),
                b"W:1".to_vec(),
            ],
            "everything at or after 'V:', including the later 'W:' namespace"
        );

        // scan_prefix stops at the boundary — no 'W:' key.
        let pre: Vec<Vec<u8>> = s.scan_prefix(b"V:").into_iter().map(|(k, _)| k).collect();
        assert_eq!(
            pre,
            vec![b"V:1".to_vec(), b"V:10".to_vec(), b"V:2".to_vec()]
        );
        assert_eq!(s.scan_prefix(b"E:").len(), 1);
        assert!(s.scan_prefix(b"ZZ").is_empty());
    }

    /// D2a-AC4: tombstones are invisible to every read path.
    #[test]
    fn kv_tombstones_hidden_everywhere() {
        let mut s = kv();
        s.put(b"p:1", b"a").unwrap();
        s.put(b"p:2", b"b").unwrap();
        s.put(b"p:3", b"c").unwrap();
        assert!(s.delete(b"p:2").unwrap());

        assert_eq!(s.get(b"p:2"), None);
        assert_eq!(s.len(), 2);
        let keys: Vec<Vec<u8>> = s.scan_prefix(b"p:").into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![b"p:1".to_vec(), b"p:3".to_vec()]);
        let keys: Vec<Vec<u8>> = s.scan_from(b"p:").into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![b"p:1".to_vec(), b"p:3".to_vec()]);

        // Re-putting a deleted key revives it in sorted position.
        s.put(b"p:2", b"b2").unwrap();
        assert_eq!(s.get(b"p:2").as_deref(), Some(&b"b2"[..]));
        let keys: Vec<Vec<u8>> = s.scan_prefix(b"p:").into_iter().map(|(k, _)| k).collect();
        assert_eq!(
            keys,
            vec![b"p:1".to_vec(), b"p:2".to_vec(), b"p:3".to_vec()]
        );
    }

    /// D2a-AC5: reopen by object ids; reads agree and further puts still sort.
    #[test]
    fn kv_reopen_by_ids() {
        let (data, index) = {
            let mut s = kv();
            s.put(b"b", b"2").unwrap();
            s.put(b"a", b"1").unwrap();
            s.delete(b"b").unwrap();
            s.ids()
        };

        let mut s = KvStore::open(data, index, 0).expect("reopen");
        assert_eq!(s.get(b"a").as_deref(), Some(&b"1"[..]));
        assert_eq!(s.get(b"b"), None, "tombstone survives reopen");
        assert_eq!(s.len(), 1);

        // Insert before the existing key: ordering still holds after reopen.
        s.put(b"A", b"0").unwrap();
        let keys: Vec<Vec<u8>> = s.scan_all().into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![b"A".to_vec(), b"a".to_vec()]);
    }

    /// D2a-AC6: empty keys and empty values are legal and distinct from
    /// absence (IndraDB uses valueless entries for edge existence).
    #[test]
    fn kv_empty_key_and_value() {
        let mut s = kv();
        s.put(b"", b"empty-key").unwrap();
        s.put(b"has-empty-val", b"").unwrap();

        assert_eq!(s.get(b"").as_deref(), Some(&b"empty-key"[..]));
        let v = s.get(b"has-empty-val");
        assert_eq!(v.as_deref(), Some(&b""[..]), "present but empty");
        assert!(v.is_some(), "empty value is not absence");
        assert_eq!(s.get(b"absent"), None);
        assert_eq!(s.len(), 2);
    }
}
