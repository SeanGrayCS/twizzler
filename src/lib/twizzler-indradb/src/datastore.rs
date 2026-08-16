//! `TwizzlerDatastore`: IndraDB persisted in Twizzler objects (board D2b-2).
//!
//! `Datastore` in 5.0 is only a transaction factory; all storage behaviour is
//! [`Transaction`]'s 26 required methods, implemented here over the sorted KV
//! store ([`crate::kv`]) using the key encoding in [`crate::keys`].
//!
//! **Interior mutability.** `Datastore::transaction(&self)` hands out
//! transactions from `&self`, but `Transaction`'s write methods take
//! `&mut self` — so the store sits behind a `RefCell` and each call borrows it
//! briefly. Reads *materialize* into owned `Vec`s before boxing them as
//! `DynIter`, which both drops the borrow before returning and sidesteps the
//! iterator-lifetime problem entirely. That costs a copy per query; for a
//! comparison baseline that is the right trade against fighting lifetimes.
//!
//! **Transaction semantics (declare, don't pretend).** These "transactions"
//! are not atomic and not isolated: every write lands immediately, and a
//! failure part-way leaves earlier writes in place. Upstream's own
//! `MemoryDatastore` is likewise a lock around shared state rather than a
//! rollback log, so this matches the baseline it replaces — but it must be
//! stated, not assumed. Crash atomicity for *our* engine is board task F2;
//! the same gap here is recorded in the capability matrix.

use std::cell::{Cell, RefCell};

use indradb::{
    Datastore, DynIter, Edge, Error, Identifier, Json, Result, Transaction, Vertex,
};
use naming::{static_naming_factory, GetFlags};
use twizzler::{
    marker::{BaseType, Invariant},
    object::{MapFlags, Object, ObjectBuilder, TypedObject},
};
use uuid::Uuid;

use crate::{keys, kv::KvStore};

const MAGIC: u64 = 0x4931_4E44_5241_4442; // "I1NDRADB"
const VERSION: u32 = 2; // 2: `sorted` persisted (D2c)

/// Root record: finds the two KV objects after a reboot.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct KvRoot {
    magic: u64,
    version: u32,
    data_raw: u128,
    index_raw: u128,
    /// How much of the index was sorted at the last `sync`.
    ///
    /// Without this, `open` had to assume nothing and re-sort the whole store,
    /// which is what made the SF0.1 query boot unopenable. Treated as a hint —
    /// `KvStore::open` clamps it and merges any tail — so a crash between a
    /// write and a `sync` costs performance, not correctness.
    sorted: u64,
}
unsafe impl Invariant for KvRoot {}
impl BaseType for KvRoot {}

/// Read/write/persist flags for reopening the root mutably.
fn rw() -> MapFlags {
    MapFlags::READ | MapFlags::WRITE | MapFlags::PERSIST
}

/// Map any Twizzler-side failure into IndraDB's error type.
fn twz_err(e: impl core::fmt::Debug) -> Error {
    Error::Datastore(Box::new(std::io::Error::new(
        std::io::ErrorKind::Other,
        format!("twizzler: {e:?}"),
    )))
}

fn ident_from_bytes(b: &[u8]) -> Result<Identifier> {
    let s = core::str::from_utf8(b).map_err(twz_err)?;
    Identifier::new(s).map_err(Error::from)
}

fn json_to_bytes(j: &Json) -> Result<Vec<u8>> {
    serde_json::to_vec(j).map_err(Error::from)
}

fn json_from_bytes(b: &[u8]) -> Result<Json> {
    serde_json::from_slice(b).map_err(Error::from)
}

/// Entries fetched per chunk by [`ChunkScan`].
const SCAN_CHUNK: usize = 1024;

/// A **lazy** range cursor, materialized `SCAN_CHUNK` entries at a time.
///
/// # Why this exists
///
/// `range_edges` means "every edge at or after this offset". IndraDB's executor
/// uses it to get one vertex's edges: it asks from `(v, ..)` and take-whiles off
/// the front. Returning that range eagerly materializes the rest of the
/// namespace — up to 1.48 M entries at SF0.1, each an allocated key `Vec` plus a
/// decoded `Edge` — so the executor can keep a handful. Across 1000 query
/// iterations that is what exhausted the frame pool *long after* the merge had
/// finished, which is the detail that pointed here.
///
/// A real KV backend (sled, RocksDB) hands back a lazy range cursor and stops
/// after a few reads. This fakes one: each chunk takes the `RefCell` borrow,
/// copies at most `SCAN_CHUNK` entries, and drops the borrow before yielding —
/// so laziness costs a re-borrow per chunk rather than a held borrow.
struct ChunkScan<'a, T> {
    ds: &'a TwizzlerDatastore,
    /// Inclusive start of the next chunk; `None` once the range is exhausted.
    next: Option<Vec<u8>>,
    prefix: Vec<u8>,
    buf: std::vec::IntoIter<T>,
    decode: fn(&[u8], &[u8]) -> Option<T>,
}

impl<T> Iterator for ChunkScan<'_, T> {
    type Item = Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(v) = self.buf.next() {
                // Counted here rather than at materialisation: the point of the
                // E7 ratio is entries the caller took against entries the store
                // produced, and IndraDB's executor take-whiles a handful off a
                // chunk of `SCAN_CHUNK`.
                crate::kv::stats::SCAN_YIELDED
                    .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                return Some(Ok(v));
            }
            let start = self.next.take()?;
            let entries = {
                let kv = self.ds.kv.borrow();
                kv.scan_range_limited(&start, &self.prefix, SCAN_CHUNK)
            };
            if entries.is_empty() {
                return None;
            }
            // Resume strictly after the last key. Appending 0x00 gives the
            // immediate successor under byte-lexicographic order, so no entry is
            // seen twice and none is skipped.
            self.next = if entries.len() < SCAN_CHUNK {
                None
            } else {
                let mut k = entries[entries.len() - 1].0.clone();
                k.push(0);
                Some(k)
            };
            let items: Vec<T> = entries
                .iter()
                .filter_map(|(k, v)| (self.decode)(k, v))
                .collect();
            self.buf = items.into_iter();
        }
    }
}

fn chunk_scan<'a, T: 'a>(
    ds: &'a TwizzlerDatastore,
    start: Vec<u8>,
    prefix: Vec<u8>,
    decode: fn(&[u8], &[u8]) -> Option<T>,
) -> DynIter<'a, T> {
    Box::new(ChunkScan {
        ds,
        next: Some(start),
        prefix,
        buf: Vec::new().into_iter(),
        decode,
    })
}

fn decode_edge(k: &[u8], _v: &[u8]) -> Option<Edge> {
    keys::decode_edge_key(k)
}

fn decode_rev_edge(k: &[u8], _v: &[u8]) -> Option<Edge> {
    keys::decode_rev_edge_key(k)
}

fn decode_vertex(k: &[u8], v: &[u8]) -> Option<Vertex> {
    let id = keys::decode_vertex_key(k)?;
    let t = ident_from_bytes(v).ok()?;
    Some(Vertex::with_id(id, t))
}

/// Box an owned collection as a `DynIter`. Still used where the result set is
/// bounded by the query itself (specific ids, index hits) rather than by a
/// namespace range.
fn iter_of<T: 'static>(items: Vec<T>) -> DynIter<'static, T> {
    Box::new(items.into_iter().map(Ok))
}

/// An IndraDB datastore backed by Twizzler persistent objects.
pub struct TwizzlerDatastore {
    kv: RefCell<KvStore>,
    /// Retained so `sync` can persist the sorted boundary. `None` for an
    /// unregistered store (`new_db`), which has no root to write.
    root: RefCell<Option<Object<KvRoot>>>,
    /// Last value written to `root.sorted`, so `sync` can skip the write when
    /// nothing changed.
    ///
    /// **This is on the hot path.** `sync` runs every `SYNC_EVERY` rows, but
    /// `sorted` only moves at a merge — 3 times in an SF0.1 load — so ~97 of
    /// ~100 root writes stored a value that was already there. Each one is an
    /// `Object::sync`, and the 2026-08-10 run 3 showed flushes of 900+ s on
    /// files far too small to have dirtied anything (`forum_hasModerator_person`:
    /// 13,750 edges, 960 s flush), which says those syncs were *waiting*, not
    /// writing.
    persisted_sorted: Cell<u64>,
}

impl TwizzlerDatastore {
    /// A fresh, unregistered datastore (its objects are reachable only while
    /// the handle lives — useful for tests).
    pub fn new_db() -> Result<indradb::Database<Self>> {
        let kv = KvStore::create().map_err(twz_err)?;
        Ok(indradb::Database::new(TwizzlerDatastore {
            kv: RefCell::new(kv),
            root: RefCell::new(None),
            persisted_sorted: Cell::new(0),
        }))
    }

    /// Clear the datastore registered at `data/<name>`, reusing its
    /// registration; no-op if absent. Mirrors `Graph::reset`, and for the same
    /// reason: `data/` supports create but not remove on this build, so the
    /// root is rewritten in place to point at fresh, empty KV objects (the old
    /// ones are orphaned). Needed for repeatable benchmark runs — without it a
    /// re-run measures a datastore that still holds the previous run's data.
    pub fn reset_db(name: &str) -> Result<()> {
        let mut namer = static_naming_factory().expect("naming service available");
        let path = format!("data/{name}");
        let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) else {
            return Ok(());
        };
        let mut root = Object::<KvRoot>::map(node.id.into(), rw()).map_err(twz_err)?;
        if root.base().magic != MAGIC {
            return Err(twz_err(format!("data/{name} is not a TwizzlerDatastore")));
        }
        let kv = KvStore::create().map_err(twz_err)?;
        let (data_raw, index_raw) = kv.ids();
        root.with_tx(|tx| {
            let mut b = tx.base_mut();
            b.magic = MAGIC;
            b.version = VERSION;
            b.data_raw = data_raw;
            b.index_raw = index_raw;
            b.sorted = 0;
            Ok(())
        })
        .map_err(twz_err)?;
        Ok(())
    }

    /// Open the datastore registered at `data/<name>`, creating and
    /// registering it if absent. Survives reboot, like the engine's
    /// `Graph::open_or_create`.
    /// Open, and build the volatile key index before handing the store over
    /// (E7).
    ///
    /// The store builds its map on the **write path only**, so a query boot
    /// otherwise binary-searches every lookup over the whole sorted region.
    /// The native arm has its query-entry index by the time latencies are
    /// measured, and that build is excluded from them; excluding both setups is
    /// only symmetric if both arms have an index to show for it.
    ///
    /// Built before the store is handed to `indradb::Database`, so this does
    /// not depend on the wrapper's accessor shape.
    pub fn open_db_indexed(name: &str) -> Result<indradb::Database<Self>> {
        Self::open_db_inner(name, true)
    }

    pub fn open_db(name: &str) -> Result<indradb::Database<Self>> {
        Self::open_db_inner(name, false)
    }

    fn open_db_inner(name: &str, index: bool) -> Result<indradb::Database<Self>> {
        let mut namer = static_naming_factory().expect("naming service available");
        let path = format!("data/{name}");

        if let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) {
            // **Mapped rw**, not read-only: `sync` writes the sorted boundary
            // back here. `MapFlags::READ` and `READ|WRITE|PERSIST` are
            // different mappings on this platform, so this cannot be upgraded
            // later without remapping.
            let root = Object::<KvRoot>::map(node.id.into(), rw()).map_err(twz_err)?;
            let (magic, version, data_raw, index_raw, sorted) = {
                let r = root.base();
                (r.magic, r.version, r.data_raw, r.index_raw, r.sorted)
            };
            if magic != MAGIC || version != VERSION {
                return Err(twz_err(format!(
                    "stale datastore format at data/{name}: magic/version mismatch \
                     (found version {version}, expected {VERSION})"
                )));
            }
            let mut kv = KvStore::open(data_raw, index_raw, sorted as usize).map_err(twz_err)?;
            if index {
                kv.build_read_index();
            }
            return Ok(indradb::Database::new(TwizzlerDatastore {
                kv: RefCell::new(kv),
                root: RefCell::new(Some(root)),
                persisted_sorted: Cell::new(sorted),
            }));
        }

        let kv = KvStore::create().map_err(twz_err)?;
        let (data_raw, index_raw) = kv.ids();
        let root = ObjectBuilder::<KvRoot>::default()
            .persist(true)
            .build(KvRoot {
                magic: MAGIC,
                version: VERSION,
                data_raw,
                index_raw,
                sorted: 0,
            })
            .map_err(twz_err)?;
        let _ = namer.remove(&path);
        namer.put(&path, root.id()).map_err(twz_err)?;
        Ok(indradb::Database::new(TwizzlerDatastore {
            kv: RefCell::new(kv),
            root: RefCell::new(Some(root)),
            persisted_sorted: Cell::new(0),
        }))
    }
}

impl TwizzlerDatastore {
    /// (data bytes, index bytes) in the backing store.
    ///
    /// Exposed for the LDBC harness: the 2026-08-10 OOM was diagnosed from
    /// pager log lines that only report the *dirty* delta, which left the
    /// store's actual size a matter of inference. A measurement beats an
    /// inference, and this one costs two field reads.
    pub fn store_bytes(&self) -> (usize, usize) {
        self.kv.borrow().sizes()
    }
}

impl Datastore for TwizzlerDatastore {
    type Transaction<'a>
        = TwizzlerTransaction<'a>
    where
        Self: 'a;

    fn transaction(&self) -> Self::Transaction<'_> {
        TwizzlerTransaction { ds: self }
    }
}

/// A transaction over [`TwizzlerDatastore`]; see the module docs for the
/// (deliberately modest) semantics.
pub struct TwizzlerTransaction<'a> {
    ds: &'a TwizzlerDatastore,
}

impl TwizzlerTransaction<'_> {
    fn vertex_type(&self, id: Uuid) -> Result<Option<Identifier>> {
        let kv = self.ds.kv.borrow();
        match kv.get(&keys::vertex_key(id)) {
            Some(bytes) => Ok(Some(ident_from_bytes(&bytes)?)),
            None => Ok(None),
        }
    }

    fn is_indexed(&self, name: &Identifier) -> bool {
        self.ds.kv.borrow().get(&keys::indexed_key(name)).is_some()
    }

    /// Every edge touching `id`, in real orientation (outbound scan + inbound
    /// scan through the reverse index).
    fn incident_edges(&self, id: Uuid) -> Vec<Edge> {
        let kv = self.ds.kv.borrow();
        let mut out = Vec::new();
        let mut pfx = keys::edge_prefix();
        pfx.extend_from_slice(id.as_bytes());
        for (k, _) in kv.scan_prefix(&pfx) {
            if let Some(e) = keys::decode_edge_key(&k) {
                out.push(e);
            }
        }
        let mut rpfx = keys::rev_edge_prefix();
        rpfx.extend_from_slice(id.as_bytes());
        for (k, _) in kv.scan_prefix(&rpfx) {
            if let Some(rev) = keys::decode_rev_edge_key(&k) {
                out.push(keys::flip(&rev));
            }
        }
        out
    }

    /// Remove an edge's forward key, reverse key, and every property.
    fn purge_edge(kv: &mut KvStore, e: &Edge) -> Result<()> {
        kv.delete(&keys::edge_key(e)).map_err(twz_err)?;
        kv.delete(&keys::rev_edge_key(e)).map_err(twz_err)?;
        let prop_keys: Vec<Vec<u8>> = kv
            .scan_prefix(&keys::edge_props_prefix(e))
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        for k in prop_keys {
            kv.delete(&k).map_err(twz_err)?;
        }
        Ok(())
    }
}

impl<'a> Transaction<'a> for TwizzlerTransaction<'a> {
    // --- vertices ----------------------------------------------------------

    fn vertex_count(&self) -> u64 {
        self.ds.kv.borrow().scan_prefix(&keys::vertex_prefix()).len() as u64
    }

    fn all_vertices(&'a self) -> Result<DynIter<'a, Vertex>> {
        Ok(chunk_scan(
            self.ds,
            keys::vertex_prefix(),
            keys::vertex_prefix(),
            decode_vertex,
        ))
    }

    fn range_vertices(&'a self, offset: Uuid) -> Result<DynIter<'a, Vertex>> {
        Ok(chunk_scan(
            self.ds,
            keys::vertex_key(offset),
            keys::vertex_prefix(),
            decode_vertex,
        ))
    }

    fn specific_vertices(&'a self, ids: Vec<Uuid>) -> Result<DynIter<'a, Vertex>> {
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(t) = self.vertex_type(id)? {
                out.push(Vertex::with_id(id, t));
            }
        }
        Ok(iter_of(out))
    }

    fn vertex_ids_with_property(
        &'a self,
        name: Identifier,
    ) -> Result<Option<DynIter<'a, Uuid>>> {
        if !self.is_indexed(&name) {
            return Ok(None); // "not indexed" is a value here, not an error
        }
        // D2c: a prefix scan of the index, not a sweep of every vertex
        // property. This is the operation IS1-IS7 use to resolve an LDBC id.
        let entries = self
            .ds
            .kv
            .borrow()
            .scan_prefix(&keys::prop_value_name_prefix(&name));
        let out: Vec<Uuid> = entries
            .iter()
            .filter_map(|(k, _)| keys::decode_prop_value_id(k))
            .collect();
        Ok(Some(iter_of(out)))
    }

    fn vertex_ids_with_property_value(
        &'a self,
        name: Identifier,
        value: &Json,
    ) -> Result<Option<DynIter<'a, Uuid>>> {
        if !self.is_indexed(&name) {
            return Ok(None);
        }
        let want = json_to_bytes(value)?;
        let entries = self
            .ds
            .kv
            .borrow()
            .scan_prefix(&keys::prop_value_exact_prefix(&name, &want));
        let out: Vec<Uuid> = entries
            .iter()
            .filter_map(|(k, _)| keys::decode_prop_value_id(k))
            .collect();
        Ok(Some(iter_of(out)))
    }

    // --- edges -------------------------------------------------------------

    fn edge_count(&self) -> u64 {
        self.ds.kv.borrow().scan_prefix(&keys::edge_prefix()).len() as u64
    }

    fn all_edges(&'a self) -> Result<DynIter<'a, Edge>> {
        Ok(chunk_scan(
            self.ds,
            keys::edge_prefix(),
            keys::edge_prefix(),
            decode_edge,
        ))
    }

    fn range_edges(&'a self, offset: Edge) -> Result<DynIter<'a, Edge>> {
        Ok(chunk_scan(
            self.ds,
            keys::edge_key(&offset),
            keys::edge_prefix(),
            decode_edge,
        ))
    }

    fn range_reversed_edges(&'a self, offset: Edge) -> Result<DynIter<'a, Edge>> {
        // `offset` arrives in reversed form, which is how the reverse index is
        // keyed; flip it to build the start key, and yield reversed form back.
        let start = keys::rev_edge_key(&keys::flip(&offset));
        Ok(chunk_scan(
            self.ds,
            start,
            keys::rev_edge_prefix(),
            decode_rev_edge,
        ))
    }

    fn specific_edges(&'a self, edges: Vec<Edge>) -> Result<DynIter<'a, Edge>> {
        let kv = self.ds.kv.borrow();
        let out = edges
            .into_iter()
            .filter(|e| kv.get(&keys::edge_key(e)).is_some())
            .collect();
        Ok(iter_of(out))
    }

    fn edges_with_property(&'a self, name: Identifier) -> Result<Option<DynIter<'a, Edge>>> {
        if !self.is_indexed(&name) {
            return Ok(None);
        }
        let entries = self.ds.kv.borrow().scan_prefix(&[keys::EDGE_PROP_TAG]);
        let mut out = Vec::new();
        for (k, _) in entries {
            if let Some((e, n)) = keys::decode_edge_prop_key(&k) {
                if n == name {
                    out.push(e);
                }
            }
        }
        Ok(Some(iter_of(out)))
    }

    fn edges_with_property_value(
        &'a self,
        name: Identifier,
        value: &Json,
    ) -> Result<Option<DynIter<'a, Edge>>> {
        if !self.is_indexed(&name) {
            return Ok(None);
        }
        let want = json_to_bytes(value)?;
        let entries = self.ds.kv.borrow().scan_prefix(&[keys::EDGE_PROP_TAG]);
        let mut out = Vec::new();
        for (k, v) in entries {
            if let Some((e, n)) = keys::decode_edge_prop_key(&k) {
                if n == name && v == want {
                    out.push(e);
                }
            }
        }
        Ok(Some(iter_of(out)))
    }

    // --- properties --------------------------------------------------------

    fn vertex_property(&self, vertex: &Vertex, name: Identifier) -> Result<Option<Json>> {
        let kv = self.ds.kv.borrow();
        match kv.get(&keys::vertex_prop_key(vertex.id, &name)) {
            Some(b) => Ok(Some(json_from_bytes(&b)?)),
            None => Ok(None),
        }
    }

    fn all_vertex_properties_for_vertex(
        &'a self,
        vertex: &Vertex,
    ) -> Result<DynIter<'a, (Identifier, Json)>> {
        let entries = self
            .ds
            .kv
            .borrow()
            .scan_prefix(&keys::vertex_props_prefix(vertex.id));
        let mut out = Vec::with_capacity(entries.len());
        for (k, v) in entries {
            if let Some((_, name)) = keys::decode_vertex_prop_key(&k) {
                out.push((name, json_from_bytes(&v)?));
            }
        }
        Ok(iter_of(out))
    }

    fn edge_property(&self, edge: &Edge, name: Identifier) -> Result<Option<Json>> {
        let kv = self.ds.kv.borrow();
        match kv.get(&keys::edge_prop_key(edge, &name)) {
            Some(b) => Ok(Some(json_from_bytes(&b)?)),
            None => Ok(None),
        }
    }

    fn all_edge_properties_for_edge(
        &'a self,
        edge: &Edge,
    ) -> Result<DynIter<'a, (Identifier, Json)>> {
        let entries = self
            .ds
            .kv
            .borrow()
            .scan_prefix(&keys::edge_props_prefix(edge));
        let mut out = Vec::with_capacity(entries.len());
        for (k, v) in entries {
            if let Some((_, name)) = keys::decode_edge_prop_key(&k) {
                out.push((name, json_from_bytes(&v)?));
            }
        }
        Ok(iter_of(out))
    }

    // --- writes ------------------------------------------------------------

    fn create_vertex(&mut self, vertex: &Vertex) -> Result<bool> {
        let key = keys::vertex_key(vertex.id);
        let mut kv = self.ds.kv.borrow_mut();
        if kv.get(&key).is_some() {
            return Ok(false); // uuid already taken
        }
        kv.put(&key, vertex.t.as_str().as_bytes()).map_err(twz_err)?;
        Ok(true)
    }

    fn create_edge(&mut self, edge: &Edge) -> Result<bool> {
        // Both endpoints must exist (IndraDB returns false, not an error).
        {
            let kv = self.ds.kv.borrow();
            if kv.get(&keys::vertex_key(edge.outbound_id)).is_none()
                || kv.get(&keys::vertex_key(edge.inbound_id)).is_none()
            {
                return Ok(false);
            }
        }
        let mut kv = self.ds.kv.borrow_mut();
        kv.put(&keys::edge_key(edge), &[]).map_err(twz_err)?;
        kv.put(&keys::rev_edge_key(edge), &[]).map_err(twz_err)?;
        Ok(true)
    }

    fn delete_vertices(&mut self, vertices: Vec<Vertex>) -> Result<()> {
        for v in vertices {
            // Incident edges go with the vertex (IndraDB semantics).
            let incident = self.incident_edges(v.id);
            let mut kv = self.ds.kv.borrow_mut();
            for e in &incident {
                Self::purge_edge(&mut kv, e)?;
            }
            // **Values are kept, not just keys**: retiring an index entry
            // needs the value it was filed under. A stale entry is worse than
            // no index — it resurrects a deleted vertex in query results.
            let props = kv.scan_prefix(&keys::vertex_props_prefix(v.id));
            for (k, val) in props {
                if let Some((pid, n)) = keys::decode_vertex_prop_key(&k) {
                    if kv.get(&keys::indexed_key(&n)).is_some() {
                        kv.delete(&keys::prop_value_key(&n, &val, pid))
                            .map_err(twz_err)?;
                    }
                }
                kv.delete(&k).map_err(twz_err)?;
            }
            kv.delete(&keys::vertex_key(v.id)).map_err(twz_err)?;
        }
        Ok(())
    }

    fn delete_edges(&mut self, edges: Vec<Edge>) -> Result<()> {
        let mut kv = self.ds.kv.borrow_mut();
        for e in &edges {
            Self::purge_edge(&mut kv, e)?;
        }
        Ok(())
    }

    fn delete_vertex_properties(&mut self, props: Vec<(Uuid, Identifier)>) -> Result<()> {
        let mut kv = self.ds.kv.borrow_mut();
        for (id, name) in props {
            let key = keys::vertex_prop_key(id, &name);
            if kv.get(&keys::indexed_key(&name)).is_some() {
                if let Some(old) = kv.get(&key) {
                    kv.delete(&keys::prop_value_key(&name, &old, id))
                        .map_err(twz_err)?;
                }
            }
            kv.delete(&key).map_err(twz_err)?;
        }
        Ok(())
    }

    fn delete_edge_properties(&mut self, props: Vec<(Edge, Identifier)>) -> Result<()> {
        let mut kv = self.ds.kv.borrow_mut();
        for (e, name) in props {
            kv.delete(&keys::edge_prop_key(&e, &name)).map_err(twz_err)?;
        }
        Ok(())
    }

    /// Declare **and build**. *Was recorded-only*, which left
    /// `vertex_ids_with_property_value` sweeping every vertex property — the
    /// exact operation the LDBC short reads depend on (D2c).
    fn index_property(&mut self, name: Identifier) -> Result<()> {
        let mut kv = self.ds.kv.borrow_mut();
        if kv.get(&keys::indexed_key(&name)).is_some() {
            return Ok(()); // idempotent: backfilling a second time is wasted work
        }
        kv.put(&keys::indexed_key(&name), &[]).map_err(twz_err)?;
        // IndraDB allows declaring an index after the data exists, so the
        // declaration has to backfill. The LDBC loader declares first, which is
        // precisely why this path would otherwise go untested.
        let existing = kv.scan_prefix(&[keys::VERTEX_PROP_TAG]);
        for (k, val) in existing {
            if let Some((id, n)) = keys::decode_vertex_prop_key(&k) {
                if n == name {
                    kv.put(&keys::prop_value_key(&name, &val, id), &[])
                        .map_err(twz_err)?;
                }
            }
        }
        Ok(())
    }

    fn set_vertex_properties(
        &mut self,
        vertices: Vec<Uuid>,
        name: Identifier,
        value: &Json,
    ) -> Result<()> {
        let bytes = json_to_bytes(value)?;
        // **Resolved before the mutable borrow**: `is_indexed` reads the same
        // `RefCell`, and taking it while `borrow_mut` is held would panic.
        let indexed = self.is_indexed(&name);
        let mut kv = self.ds.kv.borrow_mut();
        for id in vertices {
            let key = keys::vertex_prop_key(id, &name);
            if indexed {
                // Retire the previous value's entry first — otherwise an
                // overwrite leaves the vertex findable under both values.
                if let Some(old) = kv.get(&key) {
                    kv.delete(&keys::prop_value_key(&name, &old, id))
                        .map_err(twz_err)?;
                }
                kv.put(&keys::prop_value_key(&name, &bytes, id), &[])
                    .map_err(twz_err)?;
            }
            kv.put(&key, &bytes).map_err(twz_err)?;
        }
        Ok(())
    }

    fn set_edge_properties(
        &mut self,
        edges: Vec<Edge>,
        name: Identifier,
        value: &Json,
    ) -> Result<()> {
        let bytes = json_to_bytes(value)?;
        let mut kv = self.ds.kv.borrow_mut();
        for e in edges {
            kv.put(&keys::edge_prop_key(&e, &name), &bytes)
                .map_err(twz_err)?;
        }
        Ok(())
    }

    /// **The durability point** (D2c).
    ///
    /// *Was a no-op*, on the grounds that every `put` already synced its
    /// objects. That was true and was the defect: it made each write
    /// individually durable at ~1.2-1.7 MB/s writeback, ~21 object syncs per
    /// LDBC row, while our own engine batched via `push_nosync` +
    /// `Graph::sync`. IndraDB exposes `sync` so a datastore *can* defer —
    /// RocksDB buffers in a memtable and flushes on demand — and the port now
    /// uses it as intended.
    fn sync(&self) -> Result<()> {
        let sorted = {
            let mut kv = self.ds.kv.borrow_mut();
            kv.flush().map_err(twz_err)?;
            kv.sorted_len() as u64
        };
        // Persist the boundary *after* the data is durable: a root claiming
        // more sortedness than the index has would be worse than one claiming
        // less, since `open` repairs a short claim but cannot detect a long one
        // beyond clamping.
        if sorted != self.ds.persisted_sorted.get() {
            if let Some(root) = self.ds.root.borrow_mut().as_mut() {
                root.with_tx(|tx| {
                    tx.base_mut().sorted = sorted;
                    Ok(())
                })
                .map_err(twz_err)?;
            }
            self.ds.persisted_sorted.set(sorted);
        }
        Ok(())
    }
}
