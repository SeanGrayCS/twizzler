//! A sorted key-value store over Twizzler objects (board task D2a).
//!
//! IndraDB's `Transaction` trait is written against sorted KV backends (its
//! upstream ones are RocksDB and sled): it needs ordered iteration
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
//! Cost note (see board A3): each `put` issues a small number of object syncs,
//! and IndraDB's write path calls it per record. Batching a whole
//! `bulk_insert` into one sync is the obvious follow-up if D2b's bulk AC
//! proves impractical.

use twizzler::{
    collections::vec::{Vec as TwzVec, VecObject, VecObjectAlloc},
    error::TwzError,
    marker::Invariant,
    object::{MapFlags, ObjID, Object, ObjectBuilder},
};

type Result<T> = core::result::Result<T, TwzError>;

/// `flags` bit 0: the entry is deleted.
const TOMBSTONE: u32 = 1;

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
}

impl KvStore {
    /// Create a new, empty store (two fresh persistent objects).
    pub(crate) fn create() -> Result<Self> {
        Ok(KvStore {
            data: VecObject::new(ObjectBuilder::default().persist(true))?,
            index: VecObject::new(ObjectBuilder::default().persist(true))?,
        })
    }

    /// Reopen a store from the object ids returned by [`KvStore::ids`].
    pub(crate) fn open(data_raw: u128, index_raw: u128) -> Result<Self> {
        Ok(KvStore {
            data: VecObject::from(Object::<TwzVec<u8, VecObjectAlloc>>::map(
                ObjID::new(data_raw),
                rw(),
            )?),
            index: VecObject::from(Object::<TwzVec<Slot, VecObjectAlloc>>::map(
                ObjID::new(index_raw),
                rw(),
            )?),
        })
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

    /// The value for `key`, or `None` if absent or deleted.
    pub(crate) fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let data = self.data.as_slice();
        let arena = data.as_slice();
        let idx = self.index.as_slice();
        let slots = idx.as_slice();
        match Self::search(slots, arena, key) {
            Ok(i) => {
                let s = &slots[i];
                if s.flags & TOMBSTONE != 0 {
                    None
                } else {
                    Some(Self::val_of(arena, s).to_vec())
                }
            }
            Err(_) => None,
        }
    }

    /// Insert or overwrite `key`.
    pub(crate) fn put(&mut self, key: &[u8], val: &[u8]) -> Result<()> {
        // Append the value (and the key, if this is a new entry) to the arena
        // first: slots must never point at bytes that do not exist yet.
        let val_off = self.data.len() as u64;
        self.data.append(val.iter().copied())?;

        let found = {
            let data = self.data.as_slice();
            let arena = data.as_slice();
            let idx = self.index.as_slice();
            let slots = idx.as_slice();
            Self::search(slots, arena, key)
        };

        match found {
            // Existing key (live or tombstoned): repoint at the new value and
            // clear the tombstone. The old value's bytes become garbage.
            Ok(i) => {
                self.index.with_mut_slice(i..i + 1, |s| {
                    s[0].val_off = val_off;
                    s[0].val_len = val.len() as u32;
                    s[0].flags &= !TOMBSTONE;
                    Ok(())
                })?;
            }
            // New key: append its bytes, then insert the slot in sorted
            // position (push to the end, then rotate it into place —
            // `VecObject` has no `insert`).
            Err(pos) => {
                let key_off = self.data.len() as u64;
                self.data.append(key.iter().copied())?;
                let slot = Slot {
                    key_off,
                    val_off,
                    key_len: key.len() as u32,
                    val_len: val.len() as u32,
                    flags: 0,
                };
                self.index.push(slot)?;
                let end = self.index.len();
                if pos + 1 < end {
                    self.index.with_mut_slice(pos..end, |s| {
                        s.rotate_right(1);
                        Ok(())
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Tombstone `key`. Returns whether a live entry was removed.
    pub(crate) fn delete(&mut self, key: &[u8]) -> Result<bool> {
        let found = {
            let data = self.data.as_slice();
            let arena = data.as_slice();
            let idx = self.index.as_slice();
            let slots = idx.as_slice();
            match Self::search(slots, arena, key) {
                Ok(i) if slots[i].flags & TOMBSTONE == 0 => Some(i),
                _ => None,
            }
        };
        let Some(i) = found else { return Ok(false) };
        self.index.with_mut_slice(i..i + 1, |s| {
            s[0].flags |= TOMBSTONE;
            Ok(())
        })?;
        Ok(true)
    }

    /// All live entries with `key >= start`, in key order.
    ///
    /// The datastore always bounds its scans to one namespace
    /// ([`Self::scan_range`]), so this open-ended form is currently only
    /// exercised by tests — kept because "everything from here on" is the
    /// primitive `scan_range` is built from, and removing it would leave the
    /// bounded variant looking arbitrary.
    #[allow(dead_code)]
    pub(crate) fn scan_from(&self, start: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.collect(start, None)
    }

    /// All live entries whose key begins with `prefix`, in key order.
    pub(crate) fn scan_prefix(&self, prefix: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.collect(prefix, Some(prefix))
    }

    /// Entries `>= start` that also begin with `prefix`, in key order. This is
    /// the shape IndraDB's `range_*` methods need: "everything at or after
    /// this value" without spilling into the next key namespace.
    pub(crate) fn scan_range(&self, start: &[u8], prefix: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.collect(start, Some(prefix))
    }

    /// Every live entry, in key order.
    #[cfg(test)]
    pub(crate) fn scan_all(&self) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.collect(&[], None)
    }

    // --- internals ---------------------------------------------------------

    fn collect(&self, start: &[u8], prefix: Option<&[u8]>) -> Vec<(Vec<u8>, Vec<u8>)> {
        let data = self.data.as_slice();
        let arena = data.as_slice();
        let idx = self.index.as_slice();
        let slots = idx.as_slice();

        let from = match Self::search(slots, arena, start) {
            Ok(i) => i,
            Err(i) => i,
        };
        let mut out = Vec::new();
        for s in &slots[from..] {
            let k = Self::key_of(arena, s);
            if let Some(p) = prefix {
                if !k.starts_with(p) {
                    break; // sorted: past the prefix range
                }
            }
            if s.flags & TOMBSTONE == 0 {
                out.push((k.to_vec(), Self::val_of(arena, s).to_vec()));
            }
        }
        out
    }

    /// Binary search by key bytes. `Ok(i)` = exact slot, `Err(i)` = insertion
    /// point. Tombstoned slots participate: they keep their sorted position
    /// so a later `put` of the same key revives the entry in place.
    fn search(slots: &[Slot], arena: &[u8], key: &[u8]) -> core::result::Result<usize, usize> {
        slots.binary_search_by(|s| Self::key_of(arena, s).cmp(key))
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

        let mut s = KvStore::open(data, index).expect("reopen");
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
