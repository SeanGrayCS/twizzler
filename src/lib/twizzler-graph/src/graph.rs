//! The `Graph` engine: open/create a graph and run vertex/edge operations.
//!
//! `Graph` owns the cross-cutting orchestration and the registries (labels
//! and the index logs). Vertex-centric traversal lives on [`VertexView`] in
//! `vertex.rs`; edge/vertex record types live in their own modules.
//!
//! The registries are segmented vectors ([`SegVec`]) so they outgrow a single
//! object; lookups by id index them directly (ids are append indices, and
//! segments are uniformly sized, so id -> (segment, offset) is O(1)). The
//! `(label, name) -> vertex` point lookup goes through the [`IndexSchema`]:
//! a persistent `hachage` index, a lazily built in-memory map, or a scan.
//! `vertices_by_label` is a scan.
//!
//! Vertices are not in a registry at all: they live in [`ArenaStore`]'s
//! packed arenas, with adjacency as a chunk chain inside the same arena.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use naming::{static_naming_factory, GetFlags};
use twizzler::{
    collections::hachage::{PHMsession, PersistentHashMap, PersistentHashMapBase},
    marker::{BaseType, Invariant},
    object::{MapFlags, ObjID, Object, ObjectBuilder, TypedObject},
};
use twizzler_rt_abi::error::ArgumentError;

use crate::{
    arena_store::{ArenaStore, FillTo, PropSlot, MAX_TEXT_LEN},
    edge::{EdgeId, EdgeInfo},
    error::{GraphError, Result},
    blobstore::BlobStore,
    index::{
        IndexSchema, IndexStrategy, Lookup, RebuildSource, UnindexedLookup, VolatileIndex,
    },
    name::NameKey,
    props::PropValue,
    reclaim,
    segvec::SegVec,
    vertex::{Labels, VertexId, VertexInfo, VertexView},
};

pub(crate) const MAGIC: u64 = 0x4731_5457_5A47_5248; // "G1TWZGRH"

/// Magic of a root whose graph has been [`destroyed`](Graph::destroy).
///
/// A destroyed root cannot simply be zeroed: `data/` names cannot be unbound
/// on this build, so the root outlives the graph, and a zeroed one would be
/// indistinguishable from an object that was never ours. The distinct marker
/// keeps three states apart: a live graph, our destroyed root (rebuildable in
/// place), and a foreign object (must not be touched).
pub(crate) const MAGIC_DESTROYED: u64 = MAGIC ^ 0xFFFF_FFFF_FFFF_FFFF;

/// On-disk format 5: segmented registries, one vertex object plus two
/// adjacency objects per vertex, one object per edge.
///
/// Nothing reads or writes this layout. The constant survives because version
/// numbers are never reused — a future layout reusing 5 would be misread as
/// this one by any build still carrying the guard — and so `version_supported`
/// can name what it rejected when an old image turns up.
///
/// Any change to a persisted layout — size, field order, alignment, or field
/// meaning — must bump the version. A registry read at the wrong stride comes
/// back as garbage rather than an error, and the disk image survives between
/// QEMU runs, so the guard has to reject an old graph loudly instead of
/// misreading it.
#[allow(dead_code)] // reserved, not obsolete
pub(crate) const VERSION: u32 = 5;

/// The current on-disk format, the arena layout: vertices, edges, and
/// adjacency live in packed arenas ([`ArenaStore`]) instead of one object per
/// entity. The only format this build reads.
///
/// Any change to a persisted layout — size, field order, alignment, or field
/// meaning, in a record or in `GraphRoot` itself — must bump this, even when
/// no struct grows: same bytes with a different meaning read as garbage
/// rather than failing, and the disk image survives between QEMU runs.
/// Version numbers are never reused. See [`version_reclaimable`] for the
/// deliberately weaker guard on freeing.
pub(crate) const VERSION_ARENA: u32 = 15;

/// The arena format before `GraphRoot` gained `arena_cap`. Neither readable
/// nor reclaimable; kept for the same reason as [`VERSION`]: the number must
/// never be reused.
#[allow(dead_code)] // reserved, not obsolete
pub(crate) const VERSION_ARENA_NOCAP: u32 = 7;

/// Whether this build can operate on a graph in the given on-disk format —
/// read it, write it, hand it to a caller.
///
/// Strict on purpose: misreading a layout yields garbage rather than an error.
fn version_supported(v: u32) -> bool {
    v == VERSION_ARENA
}

/// Whether this build can free a graph in the given format — deliberately
/// broader than [`version_supported`].
///
/// Reading and reclaiming are different questions. Reading needs every field
/// to mean what the code thinks it means; reclaiming only needs to find the
/// object ids. A predecessor format qualifies whenever its object graph is
/// unchanged, whatever happened to the interpretation of individual records.
///
/// Extend this, not `version_supported`, whenever a bump leaves object
/// placement untouched. Doing the reverse turns a version bump into a silent
/// storage leak.
fn version_reclaimable(v: u32) -> bool {
    // No predecessor qualifies: every earlier format changed a persisted
    // layout the inventory walk depends on. Walking one at the current
    // offsets is the misread this guard exists to prevent, so an old image
    // is leaked rather than mis-freed.
    v == VERSION_ARENA
}

/// Default per-segment registry capacity.
///
/// Registry entries are small — tens of bytes — and a segment costs a whole
/// object whatever it holds, so the cap is large enough to keep a segment's
/// content in the right order against the object's own overhead.
///
/// This costs a small graph nothing: `cap` is a rollover threshold, not a
/// preallocation. `SegVec` maps element `i` to `(i / cap, i % cap)` and only
/// creates the next segment when the last one fills, and the underlying
/// `VecObject` grows on demand.
///
/// `seg_cap` is persisted per graph in `GraphRoot` and honoured on open, so an
/// existing graph keeps the geometry it was built with. Segment geometry must
/// stay uniform for a graph's lifetime — the O(1) index arithmetic depends on
/// it.
pub const DEFAULT_SEG_CAP: usize = 262_144;

/// Default records packed per arena.
///
/// The cap governs records, not just vertices — edges are records too. An
/// arena's fixed object cost is amortised over `cap` records, and locality
/// favours a larger cap: `InvPtr::new` assigns FOT index 0 to a same-arena
/// target, so operations that construct pointers get cheaper when neighbours
/// share an arena. Against that, deleted slots are reused at exact stride, so
/// churn over records of mixed inline widths strands space, and a larger cap
/// strands more of it.
pub const DEFAULT_ARENA_CAP: usize = 16384;

/// Read/write/persist map flags for reopening mutable registries.
fn rw() -> MapFlags {
    MapFlags::READ | MapFlags::WRITE | MapFlags::PERSIST
}

/// Base of the graph root object: format guard, the registry segment
/// capacity, and the registry ObjIDs (raw, so the on-disk format is
/// backend-agnostic and relocatable). The registry ids point at `SegVec`
/// directory objects.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct GraphRoot {
    pub(crate) magic: u64,
    pub(crate) version: u32,
    pub(crate) seg_cap: u32,
    pub(crate) labels_raw: u128,
    pub(crate) vindex_raw: u128,
    /// The [`ArenaStore`]'s arena directory and location registry.
    pub(crate) arena_dir_raw: u128,
    pub(crate) arena_locs_raw: u128,
    /// Records per arena, as the graph was created.
    ///
    /// Placement is a property of the graph, not of the call that opened it,
    /// so this is read on open in preference to the caller's argument.
    /// `seg_cap` above is the precedent: geometry that must stay uniform for
    /// the graph's lifetime belongs in the root.
    pub(crate) arena_cap: u32,
    /// Packed [`IndexSchema`] — strategy, unindexed-lookup policy, rebuild
    /// source. Same argument as `arena_cap` above: how the graph indexes and
    /// answers lookups must be uniform for its lifetime, so it belongs here
    /// rather than in whichever call happened to open the graph. Packed into
    /// one `u32` so a fourth policy does not need another bump.
    pub(crate) index_bits: u32,
    /// Directory of the [`IndexedLabel`] log.
    pub(crate) index_labels_raw: u128,
    /// Directory of the [`RootEntry`] list backing [`RebuildSource::Roots`].
    /// Always a real directory id — the SegVec is created unconditionally at
    /// graph creation; under [`RebuildSource::Scan`] it is the list length
    /// that stays zero, not this field.
    pub(crate) index_roots_raw: u128,
    /// Directory of the shared byte store backing long text and blobs.
    pub(crate) blob_dir_raw: u128,
}
unsafe impl Invariant for GraphRoot {}
impl BaseType for GraphRoot {}

/// An interned label (string ↔ small id).
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct LabelEntry {
    pub(crate) id: u32,
    pub(crate) name: NameKey,
}

/// One entry per `set_label_indexed` call — an append-only log, last entry
/// wins, folded into a set at open.
///
/// A flag on `LabelEntry` would be the obvious shape, but `SegVec` has only
/// `push`: flipping a flag in place would mean adding `set` to the type every
/// registry in the engine is built on, to save a structure that holds one
/// word per label (tens of entries, not millions). The log is the cheaper
/// shape.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct IndexedLabel {
    pub(crate) label: u32,
    /// Non-zero to index, zero to stop indexing. Recording the negative rather
    /// than deleting keeps the log append-only.
    pub(crate) indexed: u32,
}
unsafe impl Invariant for IndexedLabel {}

/// One entry per inserted record of an indexed label.
///
/// A single list for every indexed label rather than one per label: only
/// indexed records appear, so it stays small, and a rebuild filtering it by
/// label is trivial next to the alternative — `vertices_by_label` walks all
/// of `locs` and touches every record, at a cost that is the same however
/// little is indexed. One list also means one `u128` in the root instead of a
/// directory of per-label objects.
///
/// Append-only, and not authoritative for liveness: a deleted record's id
/// stays here, and the rebuild checks `locs` and skips it. So the list is
/// bounded by indexed records ever created, not live ones; there is no
/// compaction.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct RootEntry {
    pub(crate) id: u64,
    pub(crate) label: u32,
    pub(crate) _pad: u32,
}
unsafe impl Invariant for RootEntry {}
unsafe impl Invariant for LabelEntry {}

/// Key for the vertex index: a (label id, name) pair.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
struct VKey {
    label: u32,
    name: NameKey,
}
unsafe impl Invariant for VKey {}

/// Persistent index from (label, name) to vertex id.
type VIndex = PersistentHashMap<VKey, u64>;

/// An open graph: the root id, the registries, the vertex index, and the store.
pub struct Graph {
    root_id: ObjID,
    labels: SegVec<LabelEntry>,
    /// How this graph indexes. Read from the root on open.
    schema: IndexSchema,
    /// The append-only record of which labels are indexed.
    index_labels: SegVec<IndexedLabel>,
    /// Ids of indexed records, for `RebuildSource::Roots`. Empty and unused
    /// under `Scan`.
    index_roots: SegVec<RootEntry>,
    /// Shared byte store for values too long for a `PropSlot`.
    blobs: BlobStore,
    /// `index_labels` folded to its current state. Derived, never authoritative
    /// — the log on disk is.
    indexed_set: RefCell<HashSet<u32>>,
    /// `Some` only under [`IndexStrategy::Persistent`]. Under the other
    /// strategies this is `None` and no index object exists at all.
    vindex: Option<VIndex>,
    /// The in-memory index for [`IndexStrategy::LazyLabel`]. `RefCell` because
    /// `find_vertex` takes `&self` and must be able to build on first use —
    /// the laziness is the point, so it cannot need `&mut`.
    volatile: RefCell<VolatileIndex>,
    /// Record scans performed by this handle. A scan answers correctly but its
    /// cost is invisible at the call site, so it has to be countable.
    scans: Cell<usize>,
    /// Property lookups performed by this handle.
    ///
    /// Same reasoning as `scans`, one level down: a property read is the unit
    /// of work an ordering step spends, and its count is invisible at the
    /// call site.
    prop_reads: Cell<usize>,
    /// Vertices and adjacency — the whole graph, in packed arenas.
    store: ArenaStore,
}

/// Per-structure page accounting for one [`Graph::destroy_measured`] — one
/// row per structure `destroy` walks.
#[derive(Debug, Clone, Default)]
pub struct StructPages {
    /// `arenas+locs`, `labels`, `index_labels`, `index_roots`, `blobs`, `vindex`.
    pub label: &'static str,
    /// Ids charged to this structure, after the cross-group dedup.
    pub ids: usize,
    /// Of those, how many the kernel still knew about before deletion. A gap
    /// against `ids` means `destroy` is walking ids that are already gone.
    pub present_before: usize,
    /// Resident pages held before deletion.
    pub pages_before: usize,
    /// Deletes the kernel accepted.
    pub accepted: usize,
    /// Ids that still resolve after deletion. Zero is expected for every
    /// accepted id.
    pub present_after: usize,
    /// Resident pages still held after deletion.
    pub pages_after: usize,
}

/// What a [`Graph::destroy_measured`] measured. The quantity of interest is
/// [`returned_fraction`](Self::returned_fraction).
///
/// `measured` distinguishes "nothing came back" from "nobody looked": a plain
/// `destroy` returns this struct with `measured: false` and every page field
/// zero, and those zeros must not be read as a reclaim result.
#[derive(Debug, Clone, Default)]
pub struct DestroyReport {
    /// False when produced by [`Graph::destroy`], which skips the stat calls.
    pub measured: bool,
    /// Deletes the kernel accepted — what `destroy` returns.
    pub accepted: usize,
    /// Ids offered for deletion.
    pub attempted: usize,
    /// Resident pages over every owned id, before deletion.
    pub pages_before: usize,
    /// Resident pages over the same ids, after deletion.
    pub pages_after: usize,
    /// Owned ids that still resolve after deletion.
    pub present_after: usize,
    /// The root is deliberately never deleted — the naming service cannot
    /// unbind on this build, so deleting it would leave the registered name
    /// pointing at a deleted object. These two fields read its residency.
    pub root_pages_before: usize,
    pub root_pages_after: usize,
    /// One row per structure walked.
    pub by_struct: Vec<StructPages>,
}

impl DestroyReport {
    /// Pages that went back. Saturating, because a rise across a deletion is a
    /// real possible observation and must not wrap into a huge fake return —
    /// use [`grew`](Self::grew) to test for it.
    pub fn returned_pages(&self) -> usize {
        self.pages_before.saturating_sub(self.pages_after)
    }

    /// `None` when nothing was measured or nothing was resident, so a caller
    /// cannot mistake "no denominator" for 0%.
    pub fn returned_fraction(&self) -> Option<f64> {
        (self.measured && self.pages_before > 0)
            .then(|| self.returned_pages() as f64 / self.pages_before as f64)
    }

    /// Did any structure end up holding more pages than it started with?
    /// Destroying an object should never increase its residency.
    pub fn grew(&self) -> bool {
        self.pages_after > self.pages_before
    }

    /// One line per structure plus a total, for the harness log.
    pub fn report_lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        for s in &self.by_struct {
            out.push(format!(
                "  {:<13} ids {:>5} present {:>5}->{:>5} pages {:>8}->{:>8} accepted {:>5}",
                s.label, s.ids, s.present_before, s.present_after, s.pages_before, s.pages_after,
                s.accepted
            ));
        }
        out.push(format!(
            "  {:<13} ids {:>5} present {:>5}->{:>5} pages {:>8}->{:>8} accepted {:>5}",
            "TOTAL",
            self.attempted,
            self.by_struct.iter().map(|s| s.present_before).sum::<usize>(),
            self.present_after,
            self.pages_before,
            self.pages_after,
            self.accepted
        ));
        out.push(format!(
            "  {:<13} pages {:>8}->{:>8} (never deleted, by design)",
            "root", self.root_pages_before, self.root_pages_after
        ));
        out
    }
}

impl Graph {
    /// Total records the store holds — vertices and edges both, since an edge
    /// is a record.
    pub fn record_count(&self) -> usize {
        self.store.record_count()
    }

    /// Arena objects backing this graph.
    pub fn arena_count(&self) -> usize {
        self.store.arena_count()
    }

    /// Syncs issued by the arena store.
    pub fn arena_sync_count(&self) -> usize {
        self.store.sync_count()
    }

    /// Diagnostic pass-through to [`ArenaStore::arena_vertex_counts`]:
    /// `(policy view, ground truth)` vertices per arena.
    pub fn arena_vertex_counts(&self) -> (Vec<usize>, Vec<usize>) {
        self.store.arena_vertex_counts()
    }
}

impl Graph {
    /// Open the graph registered at `data/<name>`, or create and register a
    /// fresh one at [`DEFAULT_ARENA_CAP`]. If an existing graph has an
    /// incompatible format (magic/version mismatch) this returns
    /// [`GraphError::StaleVersion`] and leaves the existing graph intact; use
    /// [`Graph::reset`] to discard it.
    pub fn open_or_create(name: &str) -> Result<Graph> {
        Self::open_or_create_with_capacity(name, DEFAULT_SEG_CAP)
    }

    /// Like [`Graph::open_or_create`], with an explicit registry segment
    /// capacity. The capacity is used only when creating a graph; an
    /// existing graph always keeps the capacity recorded in its root, since
    /// segment geometry must stay uniform for the graph's lifetime. (Small
    /// capacities let tests force segment rollover cheaply.)
    pub fn open_or_create_with_capacity(name: &str, cap: usize) -> Result<Graph> {
        Self::open_inner(name, cap, DEFAULT_ARENA_CAP)
    }

    /// Open or create a graph packing `arena_cap` records per arena object.
    ///
    /// `arena_cap` applies only at creation. An existing graph reuses the
    /// placement its arenas were built with: the cap is persisted in the root
    /// and governs on open, so the caller's value is ignored.
    ///
    /// Every graph uses the arena layout, so the `_arena` suffix is
    /// redundant; it stays to avoid churning call sites.
    pub fn open_or_create_arena(name: &str, arena_cap: usize) -> Result<Graph> {
        Self::open_inner(name, DEFAULT_SEG_CAP, arena_cap)
    }

    /// [`Graph::open_or_create_arena`] with an explicit registry segment
    /// capacity, for tests that force segment rollover cheaply.
    pub fn open_or_create_arena_with_capacity(
        name: &str,
        cap: usize,
        arena_cap: usize,
    ) -> Result<Graph> {
        Self::open_inner(name, cap, arena_cap)
    }

    /// Every graph created here is the arena layout at `VERSION_ARENA`;
    /// `arena_cap` sets placement at creation and is ignored (see above) when
    /// opening an existing graph.
    fn open_inner(name: &str, cap: usize, arena_cap: usize) -> Result<Graph> {
        Self::open_inner_schema(name, cap, arena_cap, IndexSchema::default())
    }

    fn open_inner_schema(
        name: &str,
        cap: usize,
        arena_cap: usize,
        schema: IndexSchema,
    ) -> Result<Graph> {
        if cap == 0 || cap > u32::MAX as usize {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        // `arena_cap` is persisted as a `u32`, so an out-of-range value would
        // truncate on write and come back as a different, silently-wrong
        // packing rather than an error.
        if arena_cap == 0 || arena_cap > u32::MAX as usize {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        let mut namer = static_naming_factory().expect("naming service available");
        let path = format!("data/{name}");

        if let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) {
            let root =
                Object::<GraphRoot>::map(node.id.into(), MapFlags::READ | MapFlags::PERSIST)?;
            let (
                magic,
                version,
                seg_cap,
                labels_raw,
                vindex_raw,
                index_bits,
                index_labels_raw,
                index_roots_raw,
                blob_dir_raw,
            ) = {
                let r = root.base();
                (
                    r.magic,
                    r.version,
                    r.seg_cap,
                    r.labels_raw,
                    r.vindex_raw,
                    r.index_bits,
                    r.index_labels_raw,
                    r.index_roots_raw,
                    r.blob_dir_raw,
                )
            };
            if magic == MAGIC && version_supported(version) {
                // The persisted capacity governs, not the caller's.
                let cap = seg_cap as usize;
                // The stored schema governs, not the argument. Same rule as
                // `seg_cap` and `arena_cap`, for the same reason: a graph
                // whose indexing depended on how it was last opened would
                // answer the same lookup differently between runs. Unknown
                // bits are refused rather than defaulted: guessing would
                // answer name lookups wrongly instead of not at all.
                let schema = IndexSchema::from_bits(index_bits)
                    .ok_or(GraphError::UnknownIndexSchema { bits: index_bits })?;
                // Only the persistent strategy has an object to map. Mapping
                // `ObjID::new(0)` under the others would fault.
                let vindex = if schema.strategy == IndexStrategy::Persistent {
                    let vbacking: Object<PersistentHashMapBase<VKey, u64>> =
                        Object::map(ObjID::new(vindex_raw), rw())?;
                    Some(PersistentHashMap::from(vbacking))
                } else {
                    None
                };
                let index_labels = SegVec::<IndexedLabel>::open(index_labels_raw, cap)?;
                let index_roots = SegVec::<RootEntry>::open(index_roots_raw, cap)?;
                let blobs = BlobStore::open(blob_dir_raw, cap)?;
                let indexed_set = Self::fold_indexed(&index_labels);
                // `version_supported` above already rejected anything but
                // VERSION_ARENA, so there is exactly one layout to open.
                let store = {
                    let (dir, locs, stored_cap) = {
                        let r = root.base();
                        (r.arena_dir_raw, r.arena_locs_raw, r.arena_cap)
                    };
                    // The persisted cap governs, not the caller's: placement
                    // is a property of the graph, not of whichever call
                    // reopened it. A stored 0 is impossible past the version
                    // guard, but falling back keeps that failure a
                    // wrong-but-working cap rather than an arena cap of 0,
                    // which `FillTo` would treat as "always roll over" and
                    // turn into one arena per record.
                    let cap_for_placement = if stored_cap == 0 {
                        DEFAULT_ARENA_CAP
                    } else {
                        stored_cap as usize
                    };
                    ArenaStore::open(
                        dir,
                        locs,
                        Box::new(FillTo {
                            cap: cap_for_placement,
                        }),
                        cap,
                    )?
                };
                return Ok(Graph {
                    root_id: node.id.into(),
                    labels: SegVec::open(labels_raw, cap)?,
                    schema,
                    index_labels,
                    index_roots,
                    blobs,
                    indexed_set: RefCell::new(indexed_set),
                    vindex,
                    volatile: RefCell::new(VolatileIndex::default()),
                    scans: Cell::new(0),
                    prop_reads: Cell::new(0),
                    store,
                });
            }
            // Incompatible/stale format: do NOT touch the existing graph.
            return Err(GraphError::StaleVersion {
                found: version,
                expected: VERSION_ARENA,
            });
        }

        let labels = SegVec::create(cap)?;
        let index_labels = SegVec::<IndexedLabel>::create(cap)?;
        let index_roots = SegVec::<RootEntry>::create(cap)?;
        let blobs = BlobStore::create(cap)?;
        // No index object at all unless the schema asks for one.
        let vindex = if schema.strategy == IndexStrategy::Persistent {
            Some(VIndex::new_persist()?)
        } else {
            None
        };
        let store = ArenaStore::create(Box::new(FillTo { cap: arena_cap }), cap)?;
        let (arena_dir_raw, arena_locs_raw) = store.ids();

        let root = ObjectBuilder::<GraphRoot>::default()
            .persist(true)
            .build(GraphRoot {
                magic: MAGIC,
                version: VERSION_ARENA,
                seg_cap: cap as u32,
                labels_raw: labels.dir_raw(),
                vindex_raw: vindex.as_ref().map_or(0, |v| v.object().id().raw()),
                index_bits: schema.to_bits(),
                index_labels_raw: index_labels.dir_raw(),
                index_roots_raw: index_roots.dir_raw(),
                blob_dir_raw: blobs.dir_raw(),
                arena_dir_raw,
                arena_locs_raw,
                arena_cap: arena_cap as u32,
            })?;

        let _ = namer.remove(&path);
        namer.put(&path, root.id())?;

        Ok(Graph {
            root_id: root.id(),
            labels,
            schema,
            index_labels,
            index_roots,
            blobs,
            indexed_set: RefCell::new(HashSet::new()),
            vindex,
            volatile: RefCell::new(VolatileIndex::default()),
            scans: Cell::new(0),
            prop_reads: Cell::new(0),
            store,
        })
    }

    /// The graph root's ObjID.
    pub fn root_id(&self) -> ObjID {
        self.root_id
    }

    /// Reset a graph to empty, reusing its registration. No-op if no such graph
    /// is registered.
    ///
    /// This does not remove the `data/<name>` entry: removing a name under the
    /// persistent `data/` namespace is unsupported on the current Twizzler
    /// build (the pager's external unlink is unimplemented). Instead it
    /// rewrites the existing root object in place to point at fresh, empty
    /// registries. The outgoing graph's objects are deleted, not orphaned.
    pub fn reset(name: &str) -> Result<()> {
        Self::reset_inner(name, None)
    }

    /// Like [`Graph::reset`], but the rebuilt graph uses the given registry
    /// segment capacity instead of keeping the existing one. No-op if no such
    /// graph is registered — pair it with
    /// [`Graph::open_or_create_with_capacity`] so both paths agree on `cap`.
    pub fn reset_with_capacity(name: &str, cap: usize) -> Result<()> {
        if cap == 0 || cap > u32::MAX as usize {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        Self::reset_inner(name, Some(cap))
    }

    /// Open or create under an explicit [`IndexSchema`], packing `arena_cap`
    /// records per arena. The schema is recorded in the root, so subsequent
    /// opens by any other constructor honour it.
    pub fn open_or_create_arena_with_index(
        name: &str,
        arena_cap: usize,
        schema: IndexSchema,
    ) -> Result<Graph> {
        Self::open_inner_schema(name, DEFAULT_SEG_CAP, arena_cap, schema)
    }

    /// Reset to empty under an explicit [`IndexSchema`].
    pub fn reset_arena_with_index(
        name: &str,
        arena_cap: usize,
        schema: IndexSchema,
    ) -> Result<()> {
        if arena_cap == 0 || arena_cap > u32::MAX as usize {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        Self::reset_inner_schema(name, None, arena_cap, schema)
    }

    pub fn reset_arena(name: &str, arena_cap: usize) -> Result<()> {
        // Bounded as well as non-zero: the cap is persisted as a `u32`.
        if arena_cap == 0 || arena_cap > u32::MAX as usize {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        Self::reset_inner_fmt(name, None, arena_cap)
    }

    fn reset_inner(name: &str, cap: Option<usize>) -> Result<()> {
        Self::reset_inner_fmt(name, cap, DEFAULT_ARENA_CAP)
    }

    /// Free everything a graph owns. Returns how many objects the kernel
    /// accepted deletes for.
    ///
    /// `reset` also reclaims the outgoing graph, but leaves a fresh empty one
    /// in its place — right for "start over", wrong for "this graph is
    /// finished". `destroy` leaves no replacement, so a create/destroy cycle
    /// is flat on disk rather than costing a graph's worth of objects each
    /// time.
    ///
    /// The name is not unbound, because `data/` entries cannot be removed on
    /// this build. The root object is retained with its magic set to
    /// [`MAGIC_DESTROYED`], which makes it (a) unmistakably not a graph, so a
    /// later `open_or_create` refuses rather than reading freed ids, and (b)
    /// one leaked object per destroyed name instead of a whole graph.
    /// Re-using the name needs an explicit `reset`/`reset_arena`, which
    /// rebuilds in place.
    pub fn destroy(name: &str) -> Result<usize> {
        Self::destroy_inner(name, false).map(|r| r.accepted)
    }

    /// [`destroy`](Self::destroy), with per-object page accounting around the
    /// deletion.
    ///
    /// Same code path as `destroy`, parameterised rather than duplicated, so
    /// the id-collection walk cannot drift between the two. `destroy` pays no
    /// syscalls for this; `destroy_measured` pays two per object.
    pub fn destroy_measured(name: &str) -> Result<DestroyReport> {
        Self::destroy_inner(name, true)
    }

    fn destroy_inner(name: &str, measure: bool) -> Result<DestroyReport> {
        let mut namer = static_naming_factory().expect("naming service available");
        let path = format!("data/{name}");
        let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) else {
            return Ok(DestroyReport::default()); // nothing registered
        };
        let mut root = Object::<GraphRoot>::map(node.id.into(), rw())?;
        let (is_graph, version, cap, l, x, ad, al, il, ir, bd) = {
            let r = root.base();
            (
                r.magic == MAGIC,
                r.version,
                r.seg_cap as usize,
                r.labels_raw,
                r.vindex_raw,
                r.arena_dir_raw,
                r.arena_locs_raw,
                r.index_labels_raw,
                r.index_roots_raw,
                r.blob_dir_raw,
            )
        };
        if !is_graph {
            // Already destroyed (magic is `MAGIC_DESTROYED`), or never ours.
            // Idempotent either way, and safe: a destroyed root's registry ids
            // are zeroed, so there is nothing left to chase.
            return Ok(DestroyReport::default());
        }
        // `version_reclaimable`, not `version_supported`: destroy's job is to
        // free what the graph owns, and that only needs the object graph to be
        // walkable. Refusing here on a merely-outdated format would strand the
        // graph forever — the name cannot be unbound on this build, so nothing
        // would ever come back to free it.
        if !version_reclaimable(version) || cap == 0 {
            // Still refuse a format whose objects we cannot locate: deleting ids
            // read out of a layout we do not understand is how a live graph gets
            // destroyed.
            return Err(GraphError::StaleVersion {
                found: version,
                expected: VERSION_ARENA,
            });
        }

        // `version_reclaimable` above admits only `VERSION_ARENA`, so there is
        // one layout to walk. Grouped by structure, not flattened, so the
        // report can charge pages to each structure and every run re-derives
        // the inventory.
        let mut groups: Vec<(&'static str, Vec<u128>)> = Vec::new();
        {
            if let Ok(store) = ArenaStore::open(
                ad,
                al,
                Box::new(FillTo {
                    cap: DEFAULT_ARENA_CAP,
                }),
                cap,
            ) {
                groups.push(("arenas+locs", store.owned_object_ids()));
            }
            if let Ok(sv) = SegVec::<LabelEntry>::open(l, cap) {
                groups.push(("labels", sv.object_ids()));
            }
            if let Ok(sv) = SegVec::<IndexedLabel>::open(il, cap) {
                groups.push(("index_labels", sv.object_ids()));
            }
            if let Ok(sv) = SegVec::<RootEntry>::open(ir, cap) {
                groups.push(("index_roots", sv.object_ids()));
            }
            if let Ok(bs) = BlobStore::open(bd, cap) {
                groups.push(("blobs", bs.object_ids()));
            }
            // Zero under every strategy but `Persistent`, and `delete_raw`
            // already ignores 0 — but skipping it here keeps the reported
            // freed count honest rather than counting a no-op.
            if x != 0 {
                groups.push(("vindex", vec![x]));
            }
        }
        // Guard against a double-free: an id reachable two ways (a shared
        // property object, say) would otherwise be deleted twice. Deduping
        // across groups rather than over a flat list keeps each id charged to
        // exactly one structure, so the per-group pages sum to the total.
        {
            let mut seen = std::collections::BTreeSet::new();
            for (_, v) in groups.iter_mut() {
                v.retain(|r| *r != 0 && seen.insert(*r));
            }
            groups.retain(|(_, v)| !v.is_empty());
        }

        // Sample before the root is marked, so the `before` figure describes a
        // live graph. The root is statted alongside the owned ids even though
        // it is never deleted: it is the one deliberate leak.
        let root_raw = root.id().raw();
        let mut rep = DestroyReport::default();
        if measure {
            rep.by_struct = groups
                .iter()
                .map(|(label, v)| {
                    let (present, pages) = reclaim::pages_of(v.iter().copied());
                    StructPages {
                        label: *label,
                        ids: v.len(),
                        present_before: present,
                        pages_before: pages,
                        ..StructPages::default()
                    }
                })
                .collect();
            rep.pages_before = rep.by_struct.iter().map(|s| s.pages_before).sum();
            rep.root_pages_before = reclaim::object_pages(root_raw).unwrap_or(0);
        }

        // Mark the root dead before freeing, so an interruption leaves a root
        // that refuses to open rather than one naming freed objects.
        root.with_tx(|tx| {
            let mut b = tx.base_mut();
            b.magic = MAGIC_DESTROYED;
            b.labels_raw = 0;
            b.vindex_raw = 0;
            b.arena_dir_raw = 0;
            b.arena_locs_raw = 0;
            b.arena_cap = 0;
            b.index_labels_raw = 0;
            b.index_roots_raw = 0;
            b.blob_dir_raw = 0;
            Ok(())
        })?;

        for (i, (_, v)) in groups.iter().enumerate() {
            rep.attempted += v.len();
            let accepted = reclaim::delete_all(v.iter().copied());
            rep.accepted += accepted;
            if measure {
                let (present, pages) = reclaim::pages_of(v.iter().copied());
                let s = &mut rep.by_struct[i];
                s.accepted = accepted;
                s.present_after = present;
                s.pages_after = pages;
            }
        }
        if measure {
            rep.pages_after = rep.by_struct.iter().map(|s| s.pages_after).sum();
            rep.present_after = rep.by_struct.iter().map(|s| s.present_after).sum();
            rep.root_pages_after = reclaim::object_pages(root_raw).unwrap_or(0);
            rep.measured = true;
        }
        Ok(rep)
    }

    /// Rebuild empty at `VERSION_ARENA`, packing `arena_cap` records per
    /// arena. The outgoing graph's objects are reclaimed when its format is
    /// reclaimable; otherwise they are leaked rather than guessed at — a disk
    /// image outlives the code that wrote it.
    fn reset_inner_schema(
        name: &str,
        cap: Option<usize>,
        arena_cap: usize,
        schema: IndexSchema,
    ) -> Result<()> {
        Self::reset_inner_fmt_schema(name, cap, arena_cap, schema).map(|_| ())
    }

    fn reset_inner_fmt(name: &str, cap: Option<usize>, arena_cap: usize) -> Result<()> {
        Self::reset_inner_fmt_schema(name, cap, arena_cap, IndexSchema::default()).map(|_| ())
    }

    /// Test seam: [`Graph::reset_arena`], reporting how many outgoing-graph
    /// objects the kernel accepted deletes for. A `Delete` is accepted at the
    /// mark, so the count measures the inventory; actual reaping additionally
    /// waits on mappings dropping and a sweep, which is platform timing a
    /// unit test reports rather than asserts.
    #[cfg(test)]
    pub(crate) fn reset_arena_measured(name: &str, arena_cap: usize) -> Result<usize> {
        if arena_cap == 0 || arena_cap > u32::MAX as usize {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        Self::reset_inner_fmt_schema(name, None, arena_cap, IndexSchema::default())
    }

    /// Returns how many outgoing-graph objects the kernel accepted deletes
    /// for (0 when no graph was registered, or when the outgoing format is
    /// not reclaimable). Public wrappers discard the count.
    fn reset_inner_fmt_schema(
        name: &str,
        cap: Option<usize>,
        arena_cap: usize,
        schema: IndexSchema,
    ) -> Result<usize> {
        let mut namer = static_naming_factory().expect("naming service available");
        let path = format!("data/{name}");
        let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) else {
            return Ok(0); // nothing registered
        };

        let mut root = Object::<GraphRoot>::map(node.id.into(), rw())?;
        // Only clobber something that is actually one of our graphs. A
        // destroyed root is ours and rebuildable in place; only a root that
        // was never ours is refused. Without this, `destroy` would burn the
        // name permanently — the root survives in the disk image, so the next
        // boot inherits the refusal too.
        let (is_graph, old_version, old_cap) = {
            let r = root.base();
            (
                r.magic == MAGIC || r.magic == MAGIC_DESTROYED,
                r.version,
                r.seg_cap,
            )
        };
        let was_destroyed = root.base().magic == MAGIC_DESTROYED;
        if !is_graph {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        // Segment capacity is not inherited from the outgoing graph: absent an
        // explicit `cap`, the rebuild uses `DEFAULT_SEG_CAP`.
        let cap = cap.unwrap_or(DEFAULT_SEG_CAP);

        // Take an inventory of the outgoing graph before we repoint the root,
        // so its objects can be freed instead of orphaned. Only a reclaimable
        // format is walked; an older layout is left alone (leaked) rather
        // than guessed at.
        let old_ids = {
            let (l, x, ad, al, il, ir, bd) = {
                let r = root.base();
                (
                    r.labels_raw,
                    r.vindex_raw,
                    r.arena_dir_raw,
                    r.arena_locs_raw,
                    r.index_labels_raw,
                    r.index_roots_raw,
                    r.blob_dir_raw,
                )
            };
            match old_version {
                // A destroyed root already freed everything and zeroed its
                // registry ids; walking them would chase freed objects.
                _ if was_destroyed => Vec::new(),
                // Arenas, the registries, and the index. Matches any
                // reclaimable format, not just the current one — see
                // `version_reclaimable`.
                ver if version_reclaimable(ver) && old_cap != 0 => {
                    let cap = old_cap as usize;
                    let mut ids = Vec::new();
                    if let Ok(store) = ArenaStore::open(
                        ad,
                        al,
                        Box::new(FillTo {
                            cap: DEFAULT_ARENA_CAP,
                        }),
                        cap,
                    ) {
                        ids.extend(store.owned_object_ids());
                    }
                    if let Ok(sv) = SegVec::<LabelEntry>::open(l, cap) {
                        ids.extend(sv.object_ids());
                    }
                    if let Ok(sv) = SegVec::<IndexedLabel>::open(il, cap) {
                        ids.extend(sv.object_ids());
                    }
                    if let Ok(sv) = SegVec::<RootEntry>::open(ir, cap) {
                        ids.extend(sv.object_ids());
                    }
                    if let Ok(bs) = BlobStore::open(bd, cap) {
                        ids.extend(bs.object_ids());
                    }
                    ids.push(x);
                    ids
                }
                // An unrecognised format is left alone rather than guessed at:
                // deleting objects named by a layout we cannot read is how a
                // live graph gets destroyed.
                _ => Vec::new(),
            }
        };

        // Fresh, empty registries.
        let labels = SegVec::<LabelEntry>::create(cap)?;
        let index_labels = SegVec::<IndexedLabel>::create(cap)?;
        let index_roots = SegVec::<RootEntry>::create(cap)?;
        let blobs = BlobStore::create(cap)?;
        // No index object unless the schema asks for one.
        let vindex = if schema.strategy == IndexStrategy::Persistent {
            Some(VIndex::new_persist()?)
        } else {
            None
        };
        let (labels_raw, vindex_raw) = (
            labels.dir_raw(),
            vindex.as_ref().map_or(0, |v| v.object().id().raw()),
        );
        // A fresh store. The outgoing graph's arenas were collected into
        // `old_ids` above and are deleted at the end of this function.
        let store = ArenaStore::create(Box::new(FillTo { cap: arena_cap }), cap)?;
        let (arena_dir_raw, arena_locs_raw) = store.ids();

        // Rewrite the root transactionally so the change is synced to the
        // backing store; a raw write would be lost on reboot.
        root.with_tx(|tx| {
            let mut b = tx.base_mut();
            b.magic = MAGIC;
            b.version = VERSION_ARENA;
            b.seg_cap = cap as u32;
            b.labels_raw = labels_raw;
            b.vindex_raw = vindex_raw;
            b.arena_dir_raw = arena_dir_raw;
            b.arena_locs_raw = arena_locs_raw;
            // The rebuilt graph's placement, so a later reopen without a cap
            // keeps packing the way this reset chose.
            b.arena_cap = arena_cap as u32;
            // The reset graph's schema, so a later open honours it.
            b.index_bits = schema.to_bits();
            b.index_labels_raw = index_labels.dir_raw();
            b.index_roots_raw = index_roots.dir_raw();
            b.blob_dir_raw = blobs.dir_raw();
            Ok(())
        })?;

        // Reclaim strictly after the root commits to the new, empty
        // registries. The graph is valid and openable at this point, so an
        // interruption here leaks objects but can never leave the root naming
        // a deleted one. Best-effort: a failed delete must not fail the reset.
        Ok(reclaim::delete_all(old_ids))
    }

    /// Add a vertex carrying inline traversal properties.
    ///
    /// The set supplied here is fixed for the record's lifetime, because the
    /// record's size is: every inbound `AdjRef.neighbor` holds its arena
    /// offset, so a record that grew would have to have all of them rewritten.
    /// Anything added later via [`Graph::set_vertex_prop`] becomes a data
    /// property, which lives behind one indirection and can move freely.
    ///
    /// Choosing is the caller's job, and the criterion is access pattern: put
    /// a property here if traversals filter on it, since inline slots sit in
    /// cache lines a walk has already paid for. Put everything else in data
    /// properties — inline slots widen every record, and record width is what
    /// sets page density.
    ///
    /// Keys are interned through the same table as labels. They cannot
    /// collide: a label id is read from `record.label` and a key id from
    /// `slot.key_id`, which are different fields consulted in different
    /// contexts.
    pub fn add_vertex_with_props(
        &mut self,
        label: &str,
        name: &str,
        target: ObjID,
        props: &[(&str, PropValue)],
    ) -> Result<VertexId> {
        let lbl = self.intern_label(label)?;
        let mut slots = Vec::with_capacity(props.len());
        for (k, v) in props {
            slots.push(PropSlot {
                key_id: self.intern_label(k)?,
                _pad: 0,
                val: *v,
            });
        }
        let id = self
            .store
            .add_record(lbl, name, target.raw(), &slots, false)?;
        self.index_on_insert(lbl, name, id)?;
        Ok(VertexId(id))
    }

    pub fn add_vertex(&mut self, label: &str, name: &str, target: ObjID) -> Result<VertexId> {
        let lbl = self.intern_label(label)?;

        // One slot in a shared arena; no per-vertex objects.
        let id = self.store.add_vertex(lbl, name, target.raw())?;
        self.index_on_insert(lbl, name, id)?;
        Ok(VertexId(id))
    }

    /// Add a typed edge `from -> to`, linked into `from`'s outgoing chain and
    /// `to`'s incoming chain.
    pub fn add_edge(&mut self, from: VertexId, label: &str, to: VertexId) -> Result<EdgeId> {
        let lbl = self.intern_label(label)?;

        // The edge is a record — `from → edge → to` — so the returned id is a
        // record id from the same sequence as vertex ids; there is no
        // separate edge id space. Capture what this returns: an `EdgeId(n)`
        // built by value names whatever record n happens to be.
        let id = self.store.add_edge_record(lbl, from.0, to.0)?;
        Ok(EdgeId(id))
    }

    /// Attach a further participant to an existing edge, making it a hyperedge.
    ///
    /// `out = true` adds `vertex` as another target of `edge`, `false` as
    /// another source. Edges are records, so this needs no machinery beyond one
    /// more link in each direction.
    pub fn add_edge_endpoint(&mut self, edge: EdgeId, vertex: VertexId, out: bool) -> Result<()> {
        self.store.add_edge_endpoint(edge.0, vertex.0, out)?;
        Ok(())
    }

    /// The single place an insert touches the index, so no path can bypass the
    /// schema.
    ///
    /// Under the lazy strategy this only updates an already-built map. An
    /// unbuilt index stays unbuilt, which is what keeps a pure bulk load free
    /// — building here would reintroduce per-record index cost.
    fn index_on_insert(&mut self, lbl: u32, name: &str, id: u64) -> Result<()> {
        if !self.indexes_on_insert(lbl) {
            return Ok(());
        }
        match self.schema.strategy {
            IndexStrategy::Persistent => {
                if let Some(v) = self.vindex.as_mut() {
                    v.insert(
                        VKey {
                            label: lbl,
                            name: NameKey::new(name),
                        },
                        id,
                    )?;
                }
            }
            IndexStrategy::LazyLabel => {
                self.volatile
                    .borrow_mut()
                    .insert_if_built(lbl, NameKey::new(name), id);
                // Persist the id so a later rebuild need not find it by
                // walking every record.
                if self.schema.rebuild == RebuildSource::Roots {
                    // `push_nosync`, not `push`: `push` opens a transaction
                    // and syncs on drop — one writeback per record. Drained
                    // by `Graph::sync`.
                    self.index_roots.push_nosync(RootEntry {
                        id,
                        label: lbl,
                        _pad: 0,
                    })?;
                }
            }
            IndexStrategy::None => {}
        }
        Ok(())
    }

    /// How many indexed records the roots list tracks.
    pub fn indexed_root_count(&self) -> usize {
        self.index_roots.len()
    }

    /// The strategy this graph was created with, read from its root.
    pub fn index_strategy(&self) -> IndexStrategy {
        self.schema.strategy
    }

    pub fn index_schema(&self) -> IndexSchema {
        self.schema
    }

    /// How many record scans this handle has performed.
    pub fn scans_performed(&self) -> usize {
        self.scans.get()
    }

    /// How many property lookups this handle has performed — the unit an
    /// ordering step spends. Read it around a traversal with
    /// [`Graph::reset_prop_reads`].
    pub fn prop_reads(&self) -> usize {
        self.prop_reads.get()
    }

    /// Zero the property-lookup counter, so a measurement can bracket one
    /// traversal rather than a whole session.
    pub fn reset_prop_reads(&self) {
        self.prop_reads.set(0);
    }

    /// How many times the volatile index has been built. Inserting must leave
    /// this at zero.
    pub fn index_builds(&self) -> usize {
        self.volatile.borrow().builds()
    }

    /// The objects the index owns. Empty under every strategy but
    /// [`IndexStrategy::Persistent`].
    pub fn index_object_ids(&self) -> Vec<u128> {
        match &self.vindex {
            Some(v) => vec![v.object().id().raw()],
            None => Vec::new(),
        }
    }

    /// Live index entries. Forces a build under the lazy strategy, so it is a
    /// diagnostic rather than something to call on a hot path.
    pub fn indexed_entry_count(&self) -> usize {
        match self.schema.strategy {
            IndexStrategy::None => 0,
            IndexStrategy::Persistent => self.vindex.as_ref().map_or(0, |v| v.len()),
            IndexStrategy::LazyLabel => {
                self.ensure_volatile_built();
                self.volatile.borrow().len()
            }
        }
    }

    /// Declare a label indexed. Errors under [`IndexStrategy::None`] rather
    /// than silently doing nothing — a workload must not be able to believe
    /// it declared an index it did not get.
    pub fn set_label_indexed(&mut self, label: &str, indexed: bool) -> Result<()> {
        if self.schema.strategy == IndexStrategy::None {
            return Err(GraphError::IndexingDisabled);
        }
        let lbl = self.intern_label(label)?;
        // Idempotent, and it has to be: the log is append-only and callers
        // declare at every open, so re-appending an unchanged state would
        // grow it without bound across reopens.
        if self.indexed_set.borrow().contains(&lbl) == indexed {
            return Ok(());
        }
        // `Roots` backfill. `index_on_insert` records a `RootEntry` only when
        // the label is indexed at insert time, so a label declared after its
        // records were inserted needs them collected here — without this, a
        // rebuild could not see any pre-declaration record and `find_vertex`
        // would answer an authoritative `NotFound` for live vertices. The
        // backfill walks records once (counted in `scans_performed`) and runs
        // before the declaration is appended, so a crash between the two
        // leaves harmless orphan entries (the rebuild filters by declared
        // label) rather than a durable declaration with missing roots. A
        // re-declared label (off → on) can duplicate entries already in the
        // list; the rebuild's `map.insert` absorbs those.
        if indexed
            && self.schema.strategy == IndexStrategy::LazyLabel
            && self.schema.rebuild == RebuildSource::Roots
            && self.store.record_count() > 0
        {
            self.scans.set(self.scans.get() + 1);
            for id in self.store.vertices_by_label(lbl) {
                self.index_roots.push_nosync(RootEntry {
                    id,
                    label: lbl,
                    _pad: 0,
                })?;
            }
            self.index_roots.flush()?;
        }
        self.index_labels.push(IndexedLabel {
            label: lbl,
            indexed: u32::from(indexed),
        })?;
        if indexed {
            self.indexed_set.borrow_mut().insert(lbl);
        } else {
            self.indexed_set.borrow_mut().remove(&lbl);
        }
        // Membership changed, so anything already built is stale. Cheaper and
        // safer than patching the map: a rebuild is lazy anyway, so this costs
        // nothing unless someone actually looks up afterwards.
        self.volatile.borrow_mut().clear();
        Ok(())
    }

    /// Fold the append-only log into its current state. Later entries win.
    fn fold_indexed(log: &SegVec<IndexedLabel>) -> HashSet<u32> {
        let mut set = HashSet::new();
        for i in 0..log.len() {
            let Some(e) = log.get_ref(i).map(|r| *r) else {
                continue;
            };
            if e.indexed != 0 {
                set.insert(e.label);
            } else {
                set.remove(&e.label);
            }
        }
        set
    }

    pub fn is_label_indexed(&self, label: &str) -> bool {
        self.find_label(label)
            .is_some_and(|l| self.is_label_indexed_id(l))
    }

    fn is_label_indexed_id(&self, lbl: u32) -> bool {
        match self.schema.strategy {
            // Every record is indexed under the persistent strategy.
            IndexStrategy::Persistent => true,
            IndexStrategy::None => false,
            IndexStrategy::LazyLabel => self.indexed_set.borrow().contains(&lbl),
        }
    }

    fn indexed_label_ids(&self) -> Vec<u32> {
        self.indexed_set.borrow().iter().copied().collect()
    }

    /// Whether an insert of `lbl` should touch the index at all. Under the
    /// lazy strategy an unbuilt index stays unbuilt — see
    /// [`VolatileIndex::insert_if_built`].
    fn indexes_on_insert(&self, lbl: u32) -> bool {
        self.is_label_indexed_id(lbl)
    }

    /// Find a vertex by (label, name).
    ///
    /// Returns [`Lookup`], not `Option`: with per-label opt-in there are two
    /// distinct negatives, and `None` for an unindexed label would read as
    /// "no such vertex" when the truth is "I did not look".
    pub fn find_vertex(&self, label: &str, name: &str) -> Lookup {
        let Some(lbl) = self.find_label(label) else {
            // An un-interned label is an authoritative negative, whatever the
            // policy: `intern_label` runs on every insert, so a label with no
            // registry entry cannot be carried by any record. `NotIndexed`
            // here would report "I did not look" about a question that needs
            // no looking.
            return Lookup::NotFound;
        };
        let key = NameKey::new(name);

        let hit = match self.schema.strategy {
            IndexStrategy::Persistent => {
                let idx = self.vindex.as_ref().expect("persistent index present");
                idx.get(&VKey {
                    label: lbl,
                    name: key,
                })
                .copied()
            }
            IndexStrategy::LazyLabel if self.is_label_indexed_id(lbl) => {
                self.ensure_volatile_built();
                self.volatile.borrow().get(lbl, key)
            }
            // Not indexed: answer per policy rather than pretending.
            _ => {
                return match self.schema.unindexed {
                    UnindexedLookup::Refuse => Lookup::NotIndexed,
                    UnindexedLookup::Scan => self.scan_lookup(lbl, name),
                };
            }
        };

        match hit.map(VertexId) {
            // The index can outlive the record it names, so liveness is still
            // checked against the `locs` mirror, which is authoritative.
            Some(id) if self.is_vertex_alive(id) => Lookup::Found(id),
            _ => Lookup::NotFound,
        }
    }

    /// Walk records for a name. Public so a workload can ask for the scan
    /// explicitly even under `Refuse` — the policy governs what `find_vertex`
    /// does implicitly, not what the caller may request.
    pub fn scan_for_vertex(&self, label: &str, name: &str) -> Option<VertexId> {
        let lbl = self.find_label(label)?;
        self.scan_lookup(lbl, name).found()
    }

    fn scan_lookup(&self, lbl: u32, name: &str) -> Lookup {
        self.scans.set(self.scans.get() + 1);
        match self
            .store
            .vertices_by_label(lbl)
            .into_iter()
            .find(|id| self.store.vertex_name(*id).as_deref() == Some(name))
        {
            Some(id) => Lookup::Found(VertexId(id)),
            None => Lookup::NotFound,
        }
    }

    /// Build the volatile index if it has not been built. Insertions
    /// deliberately do not trigger this: a bulk load that never looks up must
    /// stay free, which is where the saving comes from.
    fn ensure_volatile_built(&self) {
        if self.volatile.borrow().is_built() {
            return;
        }
        let indexed: Vec<u32> = self.indexed_label_ids();
        let mut map = HashMap::new();
        match self.schema.rebuild {
            // Walks every record of each indexed label, paging in every
            // arena.
            RebuildSource::Scan => {
                for lbl in indexed {
                    for id in self.store.vertices_by_label(lbl) {
                        // The stored `NameKey`, not `NameKey::new(vertex_name)`:
                        // truncation can split a multibyte char and leave bytes
                        // that do not round-trip through `String`.
                        if let Some(k) = self.store.vertex_name_key(id) {
                            map.insert((lbl, k), id);
                        }
                    }
                }
            }
            // Reads only what was indexed. `Scan` above filters all of `locs`
            // and touches every record, at a cost that is the same whether
            // one label is indexed or all of them.
            RebuildSource::Roots => {
                for i in 0..self.index_roots.len() {
                    let Some(e) = self.index_roots.get_ref(i).map(|r| *r) else {
                        continue;
                    };
                    if !indexed.contains(&e.label) {
                        // The label was un-declared since the entry was written.
                        continue;
                    }
                    // The list is append-only and keeps ids of deleted records,
                    // so `locs` — which is authoritative for liveness — decides.
                    if !self.is_vertex_alive(VertexId(e.id)) {
                        continue;
                    }
                    if let Some(k) = self.store.vertex_name_key(e.id) {
                        map.insert((e.label, k), e.id);
                    }
                }
            }
        }
        self.volatile.borrow_mut().install(map);
    }

    /// A traversal handle for a vertex, or `None` if it is deleted.
    pub fn vertex_view(&self, id: VertexId) -> Option<VertexView<'_>> {
        if !self.is_vertex_alive(id) {
            return None;
        }
        Some(VertexView { graph: self, id })
    }

    /// Convenience: outgoing/incoming/both neighbors of `id` (no predicate).
    pub fn out_neighbors(&self, id: VertexId, labels: Labels) -> Vec<VertexId> {
        self.arena_neighbors(id, labels, true, false)
    }
    pub fn in_neighbors(&self, id: VertexId, labels: Labels) -> Vec<VertexId> {
        self.arena_neighbors(id, labels, false, true)
    }
    pub fn both_neighbors(&self, id: VertexId, labels: Labels) -> Vec<VertexId> {
        self.arena_neighbors(id, labels, true, true)
    }

    /// As the `*_neighbors` trio, but paired with the edge crossed to reach
    /// each neighbour.
    ///
    /// This costs nothing extra: `arena_adjacency` already yields
    /// `(edge_id, label, neighbour_id)`, because `AdjRef` stores the edge id
    /// beside the neighbour.
    pub fn out_neighbors_with_edges(
        &self,
        id: VertexId,
        labels: Labels,
    ) -> Vec<(EdgeId, VertexId)> {
        self.arena_neighbors_with_edges(id, labels, true, false)
    }
    pub fn in_neighbors_with_edges(
        &self,
        id: VertexId,
        labels: Labels,
    ) -> Vec<(EdgeId, VertexId)> {
        self.arena_neighbors_with_edges(id, labels, false, true)
    }
    pub fn both_neighbors_with_edges(
        &self,
        id: VertexId,
        labels: Labels,
    ) -> Vec<(EdgeId, VertexId)> {
        self.arena_neighbors_with_edges(id, labels, true, true)
    }

    // --- arena seams for the DSL (`vertex.rs`) ------------------------------

    /// `(edge_id, edge_label, neighbour_id)` in traversal order. `VertexView`
    /// uses this instead of mapping adjacency objects.
    pub(crate) fn arena_adjacency(
        &self,
        id: VertexId,
        out: bool,
        inc: bool,
    ) -> Vec<(u64, u32, u64)> {
        // No liveness filter here, and none is needed: a deleted edge is a
        // tombstoned record, and `walk_adj` skips it like any other dead
        // neighbour — on both hops, which also drops a dead far endpoint.
        self.store.neighbors_via_edges(id.0, out, inc)
    }

    /// A neighbour's `(label, name)` for predicate evaluation — no `String`
    /// allocation, since most candidates are filtered out.
    pub(crate) fn arena_vertex_key(&self, id: u64) -> Option<(u32, NameKey)> {
        self.store.vertex_key(id)
    }

    /// An edge's label and endpoints, read from its record. Used to build
    /// `EdgeHandle`s.
    ///
    /// A hyperedge has several of each endpoint; this returns the first, which
    /// is what an `EdgeHandle` can represent. Callers needing the general shape
    /// walk the edge's chains instead.
    pub(crate) fn edge_endpoints(&self, e: EdgeId) -> Option<(u32, VertexId, VertexId)> {
        let label = self.store.vertex_label(e.0)?;
        let (from, to) = self.store.edge_endpoints(e.0)?;
        Some((label, VertexId(from), VertexId(to)))
    }

    /// Neighbour query. Out then in, matching `VertexView`'s `Which::Both`
    /// ordering.
    fn arena_neighbors(
        &self,
        id: VertexId,
        labels: Labels,
        out: bool,
        inc: bool,
    ) -> Vec<VertexId> {
        // A projection of the edge-carrying form rather than a second walk, so
        // the two cannot drift apart on which entries count or in what order —
        // the same reason that one is routed through `arena_adjacency` rather
        // than the store's own `neighbors_labeled`.
        self.arena_neighbors_with_edges(id, labels, out, inc)
            .into_iter()
            .map(|(_, nb)| nb)
            .collect()
    }

    /// Label-filtered adjacency as `(edge, neighbour)` pairs, in traversal
    /// order. Dead-edge hiding happens on the `arena_adjacency` path, so it
    /// applies here and to every projection of this.
    fn arena_neighbors_with_edges(
        &self,
        id: VertexId,
        labels: Labels,
        out: bool,
        inc: bool,
    ) -> Vec<(EdgeId, VertexId)> {
        let filter = self.resolve_labels(labels);
        self.arena_adjacency(id, out, inc)
            .into_iter()
            .filter(|(_, l, _)| filter.as_ref().map_or(true, |ls| ls.contains(l)))
            .map(|(e, _, nb)| (EdgeId(e), VertexId(nb)))
            .collect()
    }

    /// All live vertex ids in the graph. Linear scan.
    pub fn vertices(&self) -> Vec<VertexId> {
        self.store.vertices().into_iter().map(VertexId).collect()
    }

    /// An edge's label and endpoints by id, or `None` if it is deleted or is
    /// not an edge.
    ///
    /// The second case is a runtime check standing in for the type system:
    /// edges and vertices share one id space, so "edge passed where a vertex
    /// belongs" cannot be rejected at compile time. `IS_EDGE` does the
    /// rejecting instead, and `edge_info` on a vertex record returns `None`.
    pub fn edge_info(&self, id: EdgeId) -> Option<EdgeInfo> {
        if !self.is_edge_alive(id) {
            return None;
        }
        let (label, from, to) = self.edge_endpoints(id)?;
        Some(EdgeInfo {
            label: self.label_name(label).unwrap_or_default(),
            from,
            to,
        })
    }

    /// Delete a vertex (tombstone). Its incident edges become hidden too,
    /// since an edge is alive only while both endpoints are. No-op if already
    /// gone.
    ///
    /// Tombstone plus slot reuse: the record's slot returns to a per-arena
    /// free list and is reused at exact stride. Frames return only with the
    /// arena object, so slot reuse caps growth under churn without shrinking
    /// residency.
    pub fn delete_vertex(&mut self, id: VertexId) -> Result<()> {
        // Drop the name from an already-built volatile map before the record
        // goes, or a lookup could resolve a tombstone. The `locs` liveness
        // check in `find_vertex` would catch it anyway; this keeps the map
        // honest rather than relying on that second line of defence.
        //
        // By the record's stored key, and only when the mapped id is this
        // record: a `NameKey::new(&String)` round-trip re-truncates and can
        // split a multibyte character, yielding a key the insert never used,
        // and key-only removal would un-index the surviving twin under
        // duplicate `(label, name)` pairs, which are permitted.
        if self.schema.strategy == IndexStrategy::LazyLabel {
            if let (Some(lbl), Some(key)) =
                (self.vertex_label_id(id), self.store.vertex_name_key(id.0))
            {
                self.volatile.borrow_mut().remove_if_built(lbl, key, id.0);
            }
        }
        // `?` rather than a bare tail: the store speaks `TwzError`, the graph
        // API speaks `GraphError`.
        self.store.delete_vertex(id.0)?;
        Ok(())
    }

    fn vertex_label_id(&self, id: VertexId) -> Option<u32> {
        self.store.vertex_label(id.0)
    }

    /// Delete an edge (tombstone). No-op if already gone.
    ///
    /// An edge is a record, so this is record deletion — and every read path
    /// hides it for free, because `walk_adj` already skips tombstoned
    /// neighbours. There is no separate edge-deletion path to keep in sync
    /// with vertex deletion.
    pub fn delete_edge(&mut self, id: EdgeId) -> Result<()> {
        if !self.store.is_edge(id.0) {
            return Ok(()); // not an edge record: no-op, as for an unknown id
        }
        self.store.delete_vertex(id.0)?;
        Ok(())
    }

    /// Every object this graph owns, in the same form `reset` reclaims, so
    /// tests assert against the real code path.
    ///
    /// Mirrors the id walk in [`Graph::destroy`] and the reset path — store
    /// ids, then the registries, then the index — so "everything the graph
    /// owns" means the same thing whichever path asks. Public because
    /// [`resident_pages`](Self::resident_pages) is read over a live graph
    /// outside test builds.
    pub fn owned_object_ids(&self) -> Vec<u128> {
        let mut ids = Vec::new();
        ids.extend(self.store.owned_object_ids());
        ids.extend(self.labels.object_ids());
        ids.extend(self.index_labels.object_ids());
        ids.extend(self.index_roots.object_ids());
        ids.extend(self.blobs.object_ids());
        // Empty under every strategy but `Persistent`.
        ids.extend(self.index_object_ids());
        ids
    }

    /// Test seam: the object ids of the `index_labels`, `index_roots`, and
    /// blob-store families alone. Split out from
    /// [`Self::owned_object_ids`] so a test can ask "did these survive a
    /// reset" without the answer being diluted by arenas.
    #[cfg(test)]
    pub(crate) fn index_family_object_ids(&self) -> Vec<u128> {
        let mut ids = self.index_labels.object_ids();
        ids.extend(self.index_roots.object_ids());
        ids.extend(self.blobs.object_ids());
        ids.retain(|r| *r != 0);
        ids
    }

    /// Resident pages this graph holds right now: `(objects, pages)` over
    /// [`owned_object_ids`](Self::owned_object_ids) plus the root.
    ///
    /// A lower bound, for the reason given on [`reclaim::object_pages`]:
    /// pager-held frames and kernel-side per-object overhead are outside any
    /// object's range tree. A large value establishes memory pressure; a
    /// small one does not by itself establish its absence.
    pub fn resident_pages(&self) -> (usize, usize) {
        let (mut objects, mut pages) = reclaim::pages_of(self.owned_object_ids());
        if let Some(p) = reclaim::object_pages(self.root_id.raw()) {
            objects += 1;
            pages += p;
        }
        (objects, pages)
    }

    /// Test seam: records reached through the arena's record pointer since
    /// the last reset. See [`ArenaStore::record_touches`].
    #[cfg(test)]
    pub(crate) fn record_touches(&self) -> usize {
        self.store.record_touches()
    }

    #[cfg(test)]
    pub(crate) fn reset_record_touches(&self) {
        self.store.reset_record_touches();
    }

    /// Test seam: a vertex's property-object id, or `None` if the vertex is
    /// dead.
    #[cfg(test)]
    pub(crate) fn vertex_props_raw(&self, v: VertexId) -> Option<u128> {
        // Always `Some(0)` for a live record: properties are arena bytes and
        // no object id exists. Kept so tests can assert exactly that.
        self.store.live_record(v.0).map(|_| 0)
    }

    /// Test seam: data-property blocks dereferenced since the last reset.
    #[cfg(test)]
    pub(crate) fn data_block_reads(&self) -> usize {
        self.store.data_block_reads()
    }

    /// Test seam: an edge's property-object id — always `Some(0)` for a live
    /// edge, as for [`Self::vertex_props_raw`].
    #[cfg(test)]
    pub(crate) fn edge_props_raw(&self, e: EdgeId) -> Option<u128> {
        self.store.live_record(e.0).map(|_| 0)
    }

    // --- properties ---------------------------------------------------------

    /// Set a property on a vertex; errors if it is missing or tombstoned.
    /// The value lands in the record's inline slot when the key already has
    /// one, else in the record's data-property block — arena bytes either
    /// way; there are no property objects.
    pub fn set_vertex_prop(&mut self, v: VertexId, key: &str, val: PropValue) -> Result<()> {
        if !self.is_vertex_alive(v) {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        // If the key is already an inline traversal slot, update it there.
        // Writing the data block instead would leave two values for one key,
        // with readers preferring the stale inline one — a divergence
        // invisible from outside, since a wrong property reads exactly like a
        // right one. Updating in place is sound because a slot is fixed-size;
        // it is adding a key that the format forbids, not changing one.
        let key_id = self.intern_label(key)?;
        if self.store.set_traversal_prop(v.0, key_id, val) == Some(true) {
            return Ok(());
        }
        // A data property is arena bytes, not an object.
        self.store.set_data_prop(v.0, key_id, val)?;
        Ok(())
    }

    /// A vertex property, or `None` if unset or the vertex is dead.
    ///
    /// Inline slots are checked first, then the data-property block. Inline
    /// is the cheaper of the two, and a key can only be in one place: the
    /// traversal set is fixed at insert, and `set_vertex_prop` updates an
    /// inline key in place rather than shadowing it in the data block.
    pub fn get_vertex_prop(&self, v: VertexId, key: &str) -> Option<PropValue> {
        // A long value is not a `PropValue` a caller may see. `PropValue`'s
        // `PartialEq`/`Ord` are structural, so a `TextRef` would compare by
        // (seg, off, len): two identical strings stored separately would test
        // unequal, and a sort would order by insertion position. A filter
        // built on that is silently wrong, so long values resolve through
        // `get_vertex_text`/`get_vertex_blob` only.
        match self.raw_prop(v, key) {
            Some(PropValue::TextRef { .. }) | Some(PropValue::BlobRef { .. }) => None,
            other => other,
        }
    }

    fn raw_prop(&self, v: VertexId, key: &str) -> Option<PropValue> {
        // Counted here rather than in `get_vertex_prop`, so the text and blob
        // paths are counted too: they pay the same lookup, and a decomposition
        // that omitted them would flatter whichever query reads long values.
        // Counted before the early returns, because a lookup that finds nothing
        // still spent the liveness check and the label scan.
        self.prop_reads.set(self.prop_reads.get() + 1);
        if !self.is_vertex_alive(v) {
            return None;
        }
        let key_id = self.find_label(key)?;
        if let Some(val) = self.store.traversal_prop(v.0, key_id) {
            return Some(val);
        }
        self.store
            .data_props(v.0)?
            .into_iter()
            .find(|s| s.key_id == key_id)
            .map(|s| s.val)
    }

    /// Set a text property: a variable-width string, up to [`MAX_TEXT_LEN`]
    /// bytes. Longer input is refused, not truncated — the API cannot quietly
    /// shorten a value.
    pub fn set_vertex_text(&mut self, v: VertexId, key: &str, text: &str) -> Result<()> {
        if text.len() > MAX_TEXT_LEN {
            return Err(GraphError::TextTooLong {
                len: text.len(),
                max: MAX_TEXT_LEN,
            });
        }
        self.set_long(v, key, text.as_bytes(), true)
    }

    pub fn get_vertex_text(&self, v: VertexId, key: &str) -> Option<String> {
        match self.raw_prop(v, key)? {
            PropValue::TextRef { seg, off, len } => {
                String::from_utf8(self.blobs.read(seg, off, len)?).ok()
            }
            // A blob is not text: its bytes are not readable through the
            // text API.
            _ => None,
        }
    }

    /// Set a blob property: arbitrary-length bytes, not queryable. There is
    /// deliberately no `has_blob`; filtering on a blob is a compile error
    /// rather than a runtime one, which is the loudest failure available.
    pub fn set_vertex_blob(&mut self, v: VertexId, key: &str, bytes: &[u8]) -> Result<()> {
        self.set_long(v, key, bytes, false)
    }

    pub fn get_vertex_blob(&self, v: VertexId, key: &str) -> Option<Vec<u8>> {
        match self.raw_prop(v, key)? {
            PropValue::BlobRef { seg, off, len } => self.blobs.read(seg, off, len),
            _ => None,
        }
    }

    fn set_long(&mut self, v: VertexId, key: &str, bytes: &[u8], text: bool) -> Result<()> {
        if !self.is_vertex_alive(v) {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        let (seg, off, len) = self.blobs.append(bytes)?;
        let val = if text {
            PropValue::TextRef { seg, off, len }
        } else {
            PropValue::BlobRef { seg, off, len }
        };
        let key_id = self.intern_label(key)?;
        // Same precedence as `set_vertex_prop`: update an inline slot in place
        // if the key already has one, or two values would exist for one key.
        if self.store.set_traversal_prop(v.0, key_id, val) == Some(true) {
            return Ok(());
        }
        self.store.set_data_prop(v.0, key_id, val)?;
        Ok(())
    }

    /// How many objects the byte store occupies. Must not scale with the
    /// number of values stored.
    pub fn blob_object_count(&self) -> usize {
        self.blobs.object_count()
    }

    /// Content-exact comparison of a long text property. Used by
    /// `VertexTraversal::has_text`; a prefix compare here would match the
    /// wrong records.
    pub(crate) fn text_eq(&self, v: VertexId, key: &str, want: &str) -> bool {
        if self.get_vertex_text(v, key).as_deref() == Some(want) {
            return true;
        }
        // `Str`-tier fallback: loaders tier values by length, so one column
        // can legally mix `Str` and text-tier values. Compare against the
        // short tier by content too — but only when `want` round-trips
        // through `NameKey` exactly, so a long probe can never false-match a
        // truncated stored key. The other direction — `has` with
        // `PropValue::str` matching text-tier values — stays closed by
        // design.
        let k = NameKey::new(want);
        k.as_str() == want && self.get_vertex_prop(v, key) == Some(PropValue::Str(k))
    }

    /// All of a vertex's properties (empty if dead/unset).
    ///
    /// Inline traversal properties first, in slot order, then data properties in
    /// insertion order. The two sets are disjoint by construction.
    pub fn vertex_props(&self, v: VertexId) -> Vec<(String, PropValue)> {
        if !self.is_vertex_alive(v) {
            return Vec::new();
        }
        let mut slots = self.store.traversal_props(v.0).unwrap_or_default();
        slots.extend(self.store.data_props(v.0).unwrap_or_default());
        slots
            .into_iter()
            // The reference variants never leave the crate (`props.rs` states
            // the invariant): their `Eq`/`Ord` are structural, so enumerating
            // them here would hand callers the silently-wrong comparisons
            // `get_vertex_prop` filters against. Long values remain readable
            // by key via `get_vertex_text`/`get_vertex_blob`.
            .filter(|s| {
                !matches!(
                    s.val,
                    PropValue::TextRef { .. } | PropValue::BlobRef { .. }
                )
            })
            .filter_map(|s| self.label_name(s.key_id).map(|k| (k, s.val)))
            .collect()
    }

    /// Set a property on an edge; errors if it is missing, tombstoned, or has
    /// a dead endpoint.
    pub fn set_edge_prop(&mut self, e: EdgeId, key: &str, val: PropValue) -> Result<()> {
        if !self.is_edge_alive(e) {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        // Identical to the vertex path, because an edge is a record.
        self.set_vertex_prop(VertexId(e.0), key, val)
    }

    /// An edge property, or `None` if unset or the edge is dead.
    pub fn get_edge_prop(&self, e: EdgeId, key: &str) -> Option<PropValue> {
        if !self.is_edge_alive(e) {
            return None;
        }
        self.get_vertex_prop(VertexId(e.0), key)
    }

    /// All of an edge's properties in insertion order (empty if dead/unset).
    pub fn edge_props(&self, e: EdgeId) -> Vec<(String, PropValue)> {
        if !self.is_edge_alive(e) {
            return Vec::new();
        }
        self.vertex_props(VertexId(e.0))
    }

    /// Whether a vertex exists and is not tombstoned.
    pub(crate) fn is_vertex_alive(&self, id: VertexId) -> bool {
        self.store.is_alive(id.0)
    }

    /// Make the graph durable — one sync per arena plus the registries.
    pub fn sync(&mut self) -> Result<()> {
        self.store.sync_all()?;
        self.labels.flush()?;
        // The roots list is written with `push_nosync`, so this is where it
        // becomes durable. Without this flush a `Roots` graph would rebuild
        // from a truncated list after a reboot and silently fail to resolve
        // records that were written but never drained.
        self.index_roots.flush()?;
        self.index_labels.flush()?;
        // Long values are written through the byte store's own nosync path,
        // so this is where they become durable.
        self.blobs.sync_all()?;
        Ok(())
    }

    /// Whether `id` names a live edge record whose endpoints are both live.
    ///
    /// The `is_edge` test is what keeps the unified id space honest: without
    /// it every live vertex would answer "yes" and `edge_info`/`get_edge_prop`
    /// would happily treat a vertex as an edge.
    pub(crate) fn is_edge_alive(&self, id: EdgeId) -> bool {
        if !self.store.is_edge(id.0) || !self.store.is_alive(id.0) {
            return false;
        }
        // An edge is alive only while both endpoints are, read from the
        // edge's own chains.
        match self.store.edge_endpoints(id.0) {
            Some((f, t)) => self.store.is_alive(f) && self.store.is_alive(t),
            None => false,
        }
    }

    /// All vertices with the given label. Linear scan.
    pub fn vertices_by_label(&self, label: &str) -> Vec<VertexId> {
        let Some(lbl) = self.find_label(label) else {
            return Vec::new();
        };
        self.store
            .vertices_by_label(lbl)
            .into_iter()
            .map(VertexId)
            .collect()
    }

    /// Read back a vertex's data, or `None` if it is deleted.
    /// O(1): ids are append indices, so the record is at position `id`.
    pub fn vertex_info(&self, id: VertexId) -> Option<VertexInfo> {
        // An edge record is not a vertex — the mirror of the `IS_EDGE` check
        // in `edge_info`. With one id space the type system does not separate
        // the two, so every accessor has to reject the wrong kind at runtime
        // or it will happily describe an edge as a vertex.
        if self.store.is_edge(id.0) {
            return None;
        }
        let (lbl, name, target) = self.store.vertex_info(id.0)?;
        Some(VertexInfo {
            label: self.label_name(lbl).unwrap_or_default(),
            name,
            target: ObjID::new(target),
        })
    }

    /// Diagnostic, temporary: pass-through to [`ArenaStore::debug_liveness`].
    pub fn debug_liveness(&self, id: VertexId) -> Option<String> {
        Some(self.store.debug_liveness(id.0))
    }

    /// Resolve a [`Labels`] filter to label ids. `None` means "any".
    pub(crate) fn resolve_labels(&self, labels: Labels) -> Option<Vec<u32>> {
        match labels {
            Labels::Any => None,
            Labels::These(names) => {
                let mut ids = Vec::new();
                for n in names {
                    if let Some(id) = self.find_label(n) {
                        ids.push(id);
                    }
                }
                Some(ids)
            }
        }
    }

    // --- private lookup helpers ---

    // The label lookups below are linear scans over the label registry.

    fn find_label(&self, name: &str) -> Option<u32> {
        find_label_in(&self.labels, name)
    }

    fn label_name(&self, id: u32) -> Option<String> {
        for i in 0..self.labels.len() {
            let e = self.labels.get_ref(i)?;
            if e.id == id {
                return Some(e.name.as_str().to_string());
            }
        }
        None
    }

    fn intern_label(&mut self, name: &str) -> Result<u32> {
        if let Some(id) = self.find_label(name) {
            return Ok(id);
        }
        let id = self.labels.len() as u32;
        self.labels.push(LabelEntry {
            id,
            name: NameKey::new(name),
        })?;
        Ok(id)
    }

    /// Insert many vertices under one index transaction.
    ///
    /// `PersistentHashMap::insert` opens a `TxObject` per call and syncs it on
    /// drop — one sync per vertex. `write_session` opens one transaction and
    /// holds it for the whole batch.
    ///
    /// Why a closure rather than a field: `PHMsession<'a>` borrows the map,
    /// so it cannot be stored beside `vindex` in `Graph` — that is a
    /// self-referential borrow. (`ArenaStore` gets away with holding its
    /// transactions because `TxObject<ArenaBase>` is owned.) Scoping the
    /// session to a closure is what the borrow checker leaves available, and
    /// it also makes the durability boundary explicit: the index is durable
    /// when the closure returns, not before.
    ///
    /// Records are still batched per arena as usual, so a bulk load pays one
    /// index sync plus one sync per arena rather than one per vertex.
    pub fn bulk_insert<R>(&mut self, f: impl FnOnce(&mut BulkInsert<'_>) -> Result<R>) -> Result<R> {
        // Disjoint field borrows: the session borrows `vindex`, the handle
        // borrows `store` and `labels`.
        let Graph {
            labels,
            vindex,
            store,
            schema,
            indexed_set,
            volatile,
            index_roots,
            ..
        } = self;
        // Only the persistent strategy has a transaction to open. Under the
        // others there is no index object, so there is nothing to batch.
        let session = match vindex.as_mut() {
            Some(v) => Some(v.write_session()?),
            None => None,
        };
        let mut b = BulkInsert {
            store,
            labels,
            session,
            schema: *schema,
            indexed: indexed_set,
            volatile,
            roots: index_roots,
        };
        f(&mut b)
    }
}

/// A batching handle from [`Graph::bulk_insert`]. Holds one index transaction
/// open for its lifetime; the index becomes durable when it is dropped.
pub struct BulkInsert<'a> {
    store: &'a mut ArenaStore,
    labels: &'a mut SegVec<LabelEntry>,
    /// `Some` only under [`IndexStrategy::Persistent`].
    session: Option<PHMsession<'a, VKey, u64>>,
    schema: IndexSchema,
    indexed: &'a RefCell<HashSet<u32>>,
    volatile: &'a RefCell<VolatileIndex>,
    /// Bulk inserts record roots too, or a bulk-loaded graph would rebuild
    /// from an empty list and find nothing.
    roots: &'a mut SegVec<RootEntry>,
}

impl BulkInsert<'_> {
    /// As [`Graph::add_vertex`], but the index write joins the open transaction.
    pub fn add_vertex(&mut self, label: &str, name: &str, target: ObjID) -> Result<VertexId> {
        let lbl = self.intern_label(label)?;
        let id = self.store.add_record(lbl, name, target.raw(), &[], false)?;
        // Mirrors `Graph::index_on_insert`: honour the schema, and never
        // build a lazy index from an insert — a bulk load that never looks up
        // stays free.
        let indexed = match self.schema.strategy {
            IndexStrategy::Persistent => true,
            IndexStrategy::None => false,
            IndexStrategy::LazyLabel => self.indexed.borrow().contains(&lbl),
        };
        if indexed {
            match (&mut self.session, self.schema.strategy) {
                (Some(sess), _) => {
                    // `insert` hands back the previous value; discarded, since
                    // a duplicate (label, name) is not an error here —
                    // `add_vertex` does not enforce uniqueness.
                    sess.insert(
                        VKey {
                            label: lbl,
                            name: NameKey::new(name),
                        },
                        id,
                    )?;
                }
                (None, IndexStrategy::LazyLabel) => {
                    self.volatile
                        .borrow_mut()
                        .insert_if_built(lbl, NameKey::new(name), id);
                }
                (None, _) => {}
            }
            if self.schema.strategy == IndexStrategy::LazyLabel
                && self.schema.rebuild == RebuildSource::Roots
            {
                // `push_nosync` for the same reason as in
                // `Graph::index_on_insert`; drained by `Graph::sync`.
                self.roots.push_nosync(RootEntry {
                    id,
                    label: lbl,
                    _pad: 0,
                })?;
            }
        }
        Ok(VertexId(id))
    }

    /// As [`Graph::add_edge`]. Edges touch no index, so this is here only so a
    /// bulk load need not drop out of the session to add them — leaving and
    /// re-entering would close and reopen the index transaction per edge, which
    /// is the cost the session exists to avoid.
    pub fn add_edge(&mut self, from: VertexId, label: &str, to: VertexId) -> Result<EdgeId> {
        let lbl = self.intern_label(label)?;
        Ok(EdgeId(self.store.add_edge_record(lbl, from.0, to.0)?))
    }

    fn intern_label(&mut self, name: &str) -> Result<u32> {
        if let Some(id) = find_label_in(self.labels, name) {
            return Ok(id);
        }
        let id = self.labels.len() as u32;
        self.labels.push(LabelEntry {
            id,
            name: NameKey::new(name),
        })?;
        Ok(id)
    }
}

// --- shared lookup helpers -------------------------------------------------

/// Label lookup by name over the label registry.
fn find_label_in(labels: &SegVec<LabelEntry>, name: &str) -> Option<u32> {
    for i in 0..labels.len() {
        let e = labels.get_ref(i)?;
        if e.name.eq_str(name) {
            return Some(e.id);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    //! Unit tests for the format guards and the geometry validation.
    //!
    //! Pure functions and early-return argument checks only — nothing here
    //! creates an object, so these cost the shared boot nothing. The
    //! behavioural counterparts live in `tests/arena_graph.rs`.

    use super::*;

    /// Reading is gated strictly: exactly one format, and no predecessor.
    #[test]
    fn only_the_current_format_is_readable() {
        assert!(version_supported(VERSION_ARENA));
        assert!(!version_supported(VERSION_ARENA_NOCAP), "v7 is not readable");
        assert!(!version_supported(VERSION), "v3 is not readable");
        // Every number except the current one, written as a filter over a
        // range rather than a literal list so the current version can never
        // end up in the rejected set.
        for v in (0..=32u32).filter(|v| *v != VERSION_ARENA) {
            assert!(!version_supported(v), "version {v} must not be readable");
        }
    }

    /// Freeing is gated on whether the object graph is walkable — a weaker
    /// condition than readability, but not a free pass. Currently no
    /// predecessor qualifies.
    #[test]
    fn reclaimability_tracks_whether_records_are_still_walkable() {
        assert!(version_reclaimable(VERSION_ARENA));
        assert!(
            !version_reclaimable(VERSION_ARENA_NOCAP),
            "a format whose record layout we can no longer read must not be \
             walked for object ids: leaking beats mis-freeing"
        );
    }

    /// Format 5 (`VERSION`) is not reclaimable and must not quietly become
    /// so: its object graph is genuinely different (three objects per vertex,
    /// one per edge), and no walker for it exists.
    #[test]
    fn v3_is_not_reclaimable() {
        assert!(!version_reclaimable(VERSION));
        for v in (0..=32u32).filter(|v| *v != VERSION_ARENA) {
            assert!(!version_reclaimable(v), "version {v} must not be freed");
        }
    }

    /// Anything this build can read, it must also be able to free — otherwise
    /// opening a graph and then resetting it leaks the very objects it was
    /// just using. Asserting the relationship rather than two enumerations is
    /// what makes this survive the next bump: a new format added to
    /// `version_supported` alone fails here.
    #[test]
    fn everything_readable_is_also_reclaimable() {
        for v in 0..=32u32 {
            assert!(
                !version_supported(v) || version_reclaimable(v),
                "version {v} is readable but not reclaimable — a reset would \
                 leak a graph this build had open"
            );
        }
    }

    /// `arena_cap` is persisted as a `u32`, so an out-of-range value must be
    /// refused rather than truncated into a different, silently-wrong
    /// packing. Zero is refused separately: `FillTo { cap: 0 }` never reuses
    /// an arena, so it would degenerate to one arena per record.
    #[test]
    fn arena_cap_is_rejected_outside_the_persisted_range() {
        let too_big = u32::MAX as usize + 1;
        assert!(Graph::open_or_create_arena("t-cap-guard", 0).is_err());
        assert!(Graph::open_or_create_arena("t-cap-guard", too_big).is_err());
        assert!(Graph::reset_arena("t-cap-guard", 0).is_err());
        assert!(Graph::reset_arena("t-cap-guard", too_big).is_err());
        assert!(Graph::open_or_create_arena_with_capacity("t-cap-guard", 4, 0).is_err());
    }

    /// The registry segment capacity has the same persisted-as-`u32` constraint.
    #[test]
    fn seg_cap_is_rejected_outside_the_persisted_range() {
        assert!(Graph::open_or_create_with_capacity("t-seg-guard", 0).is_err());
        assert!(
            Graph::open_or_create_with_capacity("t-seg-guard", u32::MAX as usize + 1).is_err()
        );
    }
}
