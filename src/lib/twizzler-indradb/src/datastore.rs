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

use std::cell::RefCell;

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
const VERSION: u32 = 1;

/// Root record: finds the two KV objects after a reboot.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct KvRoot {
    magic: u64,
    version: u32,
    data_raw: u128,
    index_raw: u128,
}
unsafe impl Invariant for KvRoot {}
impl BaseType for KvRoot {}

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

/// Box an owned collection as a `DynIter`. Materializing is what lets reads
/// return without holding the `RefCell` borrow.
fn iter_of<T: 'static>(items: Vec<T>) -> DynIter<'static, T> {
    Box::new(items.into_iter().map(Ok))
}

/// An IndraDB datastore backed by Twizzler persistent objects.
pub struct TwizzlerDatastore {
    kv: RefCell<KvStore>,
}

impl TwizzlerDatastore {
    /// A fresh, unregistered datastore (its objects are reachable only while
    /// the handle lives — useful for tests).
    pub fn new_db() -> Result<indradb::Database<Self>> {
        let kv = KvStore::create().map_err(twz_err)?;
        Ok(indradb::Database::new(TwizzlerDatastore {
            kv: RefCell::new(kv),
        }))
    }

    /// Open the datastore registered at `data/<name>`, creating and
    /// registering it if absent. Survives reboot, like the engine's
    /// `Graph::open_or_create`.
    pub fn open_db(name: &str) -> Result<indradb::Database<Self>> {
        let mut namer = static_naming_factory().expect("naming service available");
        let path = format!("data/{name}");

        if let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) {
            let root = Object::<KvRoot>::map(
                node.id.into(),
                MapFlags::READ | MapFlags::PERSIST,
            )
            .map_err(twz_err)?;
            let (magic, version, data_raw, index_raw) = {
                let r = root.base();
                (r.magic, r.version, r.data_raw, r.index_raw)
            };
            if magic != MAGIC || version != VERSION {
                return Err(twz_err(format!(
                    "stale datastore format at data/{name}: magic/version mismatch \
                     (found version {version}, expected {VERSION})"
                )));
            }
            let kv = KvStore::open(data_raw, index_raw).map_err(twz_err)?;
            return Ok(indradb::Database::new(TwizzlerDatastore {
                kv: RefCell::new(kv),
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
            })
            .map_err(twz_err)?;
        let _ = namer.remove(&path);
        namer.put(&path, root.id()).map_err(twz_err)?;
        Ok(indradb::Database::new(TwizzlerDatastore {
            kv: RefCell::new(kv),
        }))
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

    fn decode_vertices(entries: Vec<(Vec<u8>, Vec<u8>)>) -> Result<Vec<Vertex>> {
        let mut out = Vec::with_capacity(entries.len());
        for (k, v) in entries {
            let Some(id) = keys::decode_vertex_key(&k) else {
                continue;
            };
            out.push(Vertex::with_id(id, ident_from_bytes(&v)?));
        }
        Ok(out)
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
        let entries = self.ds.kv.borrow().scan_prefix(&keys::vertex_prefix());
        Ok(iter_of(Self::decode_vertices(entries)?))
    }

    fn range_vertices(&'a self, offset: Uuid) -> Result<DynIter<'a, Vertex>> {
        let entries = self
            .ds
            .kv
            .borrow()
            .scan_range(&keys::vertex_key(offset), &keys::vertex_prefix());
        Ok(iter_of(Self::decode_vertices(entries)?))
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
        let entries = self
            .ds
            .kv
            .borrow()
            .scan_prefix(&[keys::VERTEX_PROP_TAG]);
        let mut out = Vec::new();
        for (k, _) in entries {
            if let Some((id, n)) = keys::decode_vertex_prop_key(&k) {
                if n == name {
                    out.push(id);
                }
            }
        }
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
            .scan_prefix(&[keys::VERTEX_PROP_TAG]);
        let mut out = Vec::new();
        for (k, v) in entries {
            if let Some((id, n)) = keys::decode_vertex_prop_key(&k) {
                if n == name && v == want {
                    out.push(id);
                }
            }
        }
        Ok(Some(iter_of(out)))
    }

    // --- edges -------------------------------------------------------------

    fn edge_count(&self) -> u64 {
        self.ds.kv.borrow().scan_prefix(&keys::edge_prefix()).len() as u64
    }

    fn all_edges(&'a self) -> Result<DynIter<'a, Edge>> {
        let entries = self.ds.kv.borrow().scan_prefix(&keys::edge_prefix());
        let out = entries
            .into_iter()
            .filter_map(|(k, _)| keys::decode_edge_key(&k))
            .collect();
        Ok(iter_of(out))
    }

    fn range_edges(&'a self, offset: Edge) -> Result<DynIter<'a, Edge>> {
        let entries = self
            .ds
            .kv
            .borrow()
            .scan_range(&keys::edge_key(&offset), &keys::edge_prefix());
        let out = entries
            .into_iter()
            .filter_map(|(k, _)| keys::decode_edge_key(&k))
            .collect();
        Ok(iter_of(out))
    }

    fn range_reversed_edges(&'a self, offset: Edge) -> Result<DynIter<'a, Edge>> {
        // `offset` arrives in reversed form, which is how the reverse index is
        // keyed; flip it to build the start key, and yield reversed form back.
        let start = keys::rev_edge_key(&keys::flip(&offset));
        let entries = self
            .ds
            .kv
            .borrow()
            .scan_range(&start, &keys::rev_edge_prefix());
        let out = entries
            .into_iter()
            .filter_map(|(k, _)| keys::decode_rev_edge_key(&k))
            .collect();
        Ok(iter_of(out))
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
            let prop_keys: Vec<Vec<u8>> = kv
                .scan_prefix(&keys::vertex_props_prefix(v.id))
                .into_iter()
                .map(|(k, _)| k)
                .collect();
            for k in prop_keys {
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
            kv.delete(&keys::vertex_prop_key(id, &name))
                .map_err(twz_err)?;
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

    fn index_property(&mut self, name: Identifier) -> Result<()> {
        // Recorded, not built: queries below scan and filter. Correct, not
        // fast — declared as such in the capability matrix (board D2b AC7).
        let mut kv = self.ds.kv.borrow_mut();
        kv.put(&keys::indexed_key(&name), &[]).map_err(twz_err)?;
        Ok(())
    }

    fn set_vertex_properties(
        &mut self,
        vertices: Vec<Uuid>,
        name: Identifier,
        value: &Json,
    ) -> Result<()> {
        let bytes = json_to_bytes(value)?;
        let mut kv = self.ds.kv.borrow_mut();
        for id in vertices {
            kv.put(&keys::vertex_prop_key(id, &name), &bytes)
                .map_err(twz_err)?;
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

    /// Every `put` already syncs its objects (see `kv`), so there is no
    /// deferred state to flush. Overridden because the default errors out.
    fn sync(&self) -> Result<()> {
        Ok(())
    }
}
