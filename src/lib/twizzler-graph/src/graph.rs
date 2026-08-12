//! The `Graph` engine: open/create a graph and run vertex/edge operations.
//!
//! `Graph` owns the cross-cutting orchestration and the registries (vertices,
//! edges, labels). Vertex-centric traversal lives on [`VertexView`] in
//! `vertex.rs`; edge/vertex record types live in their own modules.
//!
//! The registries are segmented vectors ([`SegVec`]) so they outgrow a single
//! object; lookups by id index them directly (ids are append indices, and
//! segments are uniformly sized, so id -> (segment, offset) is O(1)). The
//! `(label, name) -> vertex` point lookup uses a persistent `hachage` index.
//! `vertices_by_label` is still a scan.

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
    arena_store::{ArenaStore, FillTo, PropSlot},
    edge::{EdgeId, EdgeInfo},
    error::{GraphError, Result},
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
/// A destroyed root cannot simply be zeroed: `data/` names cannot be unbound on
/// this build, so the root outlives the graph, and a zeroed one is
/// indistinguishable from "not our object" — which made `reset` refuse it and
/// the name unusable forever, across boots, since the root persists in the
/// disk image. A distinct marker keeps three states apart: a live graph, our
/// destroyed root (rebuildable in place), and something that was never ours
/// (must not be touched).
pub(crate) const MAGIC_DESTROYED: u64 = MAGIC ^ 0xFFFF_FFFF_FFFF_FFFF;

/// On-disk format 5 ("v3"): segmented registries, one vertex object plus two
/// adjacency objects per vertex, one object per edge.
///
/// 1. The number 5 is burned. Version numbers must never be recycled — a
///    future layout reusing 5 would be read as v3 by any build still carrying
///    this guard, which is the misread-as-garbage failure the whole scheme
///    exists to prevent.
/// 2. It documents what `version_supported` is rejecting when an old image
///    turns up.
///
/// The disk image survives between QEMU runs, which is what turns "I changed a
/// struct" into "the next boot hangs". Any change to a persisted record's
/// layout — size, field order, or alignment — must bump this, so the guard
/// rejects the old graph loudly instead of misreading it.
#[allow(dead_code)] // see above: reserved, not obsolete
pub(crate) const VERSION: u32 = 5;

/// On-disk format 8 (the arena layout): vertices and adjacency live in
/// packed arenas ([`ArenaStore`]) instead of three objects per vertex plus one
/// per edge.
///
/// Reading a format-8 graph with this build would interpret padding as a
/// generation, mismatch every adjacency entry, and silently return a graph with
/// no edges. Clear the disk image.
pub(crate) const VERSION_ARENA: u32 = 13;

/// No longer reclaimable as of format 9. It was, while 8 differed from 7
/// only in a trailing root field; format 9 moved the *record* layout, and the
/// inventory walk reads records. Kept for the same reason as [`VERSION`]: the
/// number is burned and must never be recycled.
#[allow(dead_code)] // reserved, not obsolete — see above
pub(crate) const VERSION_ARENA_NOCAP: u32 = 7;

/// Whether this build can *operate* on a graph in the given on-disk format —
/// read it, write it, hand it to a caller.
///
/// Strict on purpose: misreading a layout yields garbage rather than an error.
fn version_supported(v: u32) -> bool {
    v == VERSION_ARENA
}

/// Whether this build can *free* a graph in the given format — deliberately
/// broader than [`version_supported`].
///
/// Reading and reclaiming are different questions. Reading needs every field
/// to mean what the code thinks it means. Reclaiming only needs to find the
/// object ids, so a predecessor format qualifies whenever its *object graph* is
/// unchanged, whatever happened to the interpretation of individual records.
fn version_reclaimable(v: u32) -> bool {
    // This is the rule from the 7 → 8 mistake applied in the other direction:
    // extend `version_reclaimable` when placement is untouched, and *don't*
    // when it isn't.
    v == VERSION_ARENA
}

/// Default per-segment registry capacity.
///
/// This costs a small graph nothing. `cap` is a rollover threshold, not a
/// preallocation — `SegVec` maps element `i` to `(i / cap, i % cap)` and only
/// creates the next segment when the last one fills, and the underlying
/// `VecObject` grows on demand. A five-vertex graph has one segment object
/// either way.
///
/// No format bump. `seg_cap` is persisted per graph in `GraphRoot` and
/// honoured on open, so existing graphs keep the geometry they were built with;
/// only newly created (and `reset`) graphs take the new default. That is also
/// why segment geometry must stay uniform for a graph's lifetime — the O(1)
/// index arithmetic depends on it.
pub const DEFAULT_SEG_CAP: usize = 262_144;

/// Default vertices packed per arena on the VERSION 4 layout.
///
/// The benefit saturates once the interconnected set fits in one arena:
/// 16384 and 65536 were indistinguishable on every metric, because both held
/// all 6 001 phase-A vertices in a single arena. Cap only matters up to the
/// working set being connected.
///
/// Memory is not the constraint — per-vertex overhead is `1454/cap` frames,
/// so this raise takes it from ~0.36 to ~0.089.
///
/// # This value is PROVISIONAL
pub const DEFAULT_ARENA_CAP: usize = 16384;

/// Read/write/persist map flags for reopening mutable registries.
fn rw() -> MapFlags {
    MapFlags::READ | MapFlags::WRITE | MapFlags::PERSIST
}

/// Base of the graph root object: format guard, the registry segment
/// capacity, and the registry ObjIDs (raw, so the on-disk format is
/// backend-agnostic and relocatable). The registry ids point at `SegVec`
/// directory objects as of version 3.
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
    /// Read on open in preference to the caller's argument. `seg_cap` above is
    /// the precedent: geometry that must stay uniform for the graph's lifetime
    /// belongs in the root.
    pub(crate) arena_cap: u32,
    pub(crate) index_bits: u32,
    pub(crate) index_labels_raw: u128,
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

/// A flag on `LabelEntry` would have been the obvious shape, but `SegVec` has
/// only `push`: flipping a flag in place would mean adding `set` to the type
/// every registry in the engine is built on, to save a structure that holds one
/// word per *label* (tens of entries, not millions). The log is the cheaper
/// risk, and its sync cost is nil at this size.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct IndexedLabel {
    pub(crate) label: u32,
    /// Non-zero to index, zero to stop indexing. Recording the negative rather
    /// than deleting keeps the log append-only.
    pub(crate) indexed: u32,
}
unsafe impl Invariant for IndexedLabel {}
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
    schema: IndexSchema,
    /// The append-only record of which labels are indexed.
    index_labels: SegVec<IndexedLabel>,
    /// `index_labels` folded to its current state. Derived, never authoritative
    /// — the log on disk is.
    indexed_set: RefCell<HashSet<u32>>,
    vindex: Option<VIndex>,
    volatile: RefCell<VolatileIndex>,
    scans: Cell<usize>,
    /// Vertices and adjacency — the whole graph, in packed arenas.
    store: ArenaStore,
}

impl Graph {

    pub fn record_count(&self) -> usize {
        self.store.record_count()
    }

    pub fn arena_count(&self) -> usize {
        self.store.arena_count()
    }

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
    /// incompatible format (magic/version mismatch — which now includes every
    /// v3 graph) this returns [`GraphError::StaleVersion`] and leaves the
    /// existing graph intact; use [`Graph::reset`] to discard it.
    pub fn open_or_create(name: &str) -> Result<Graph> {
        Self::open_or_create_with_capacity(name, DEFAULT_SEG_CAP)
    }

    /// Like [`Graph::open_or_create`], with an explicit registry segment
    /// capacity. The capacity is used only when *creating* a graph; an
    /// existing graph always keeps the capacity recorded in its root, since
    /// segment geometry must stay uniform for the graph's lifetime. (Small
    /// capacities let tests force segment rollover cheaply.)
    pub fn open_or_create_with_capacity(name: &str, cap: usize) -> Result<Graph> {
        Self::open_inner(name, cap, DEFAULT_ARENA_CAP)
    }

    /// Open or create a graph packing `arena_cap` vertices per arena object.
    ///
    /// The name is now redundant (every graph is an arena graph); it stays to
    /// avoid churning ~40 call sites, and should become
    /// `open_or_create_with_arena_cap` when something else touches them.
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

    /// Every graph created here is VERSION 4; `arena_cap` sets placement at
    /// creation and is ignored (see above) when opening an existing graph.
    fn open_inner(name: &str, cap: usize, arena_cap: usize) -> Result<Graph> {
        Self::open_inner_schema(name, cap, arena_cap, IndexSchema::default())
    }

    fn open_inner_schema(
        name: &str,
        cap: usize,
        arena_cap: usize,
        schema: IndexSchema,
    ) -> Result<Graph> {
        Self::validate_schema(schema)?;
        if cap == 0 || cap > u32::MAX as usize {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        if arena_cap == 0 || arena_cap > u32::MAX as usize {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        let mut namer = static_naming_factory().expect("naming service available");
        let path = format!("data/{name}");

        if let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) {
            let root =
                Object::<GraphRoot>::map(node.id.into(), MapFlags::READ | MapFlags::PERSIST)?;
            let (magic, version, seg_cap, labels_raw, vindex_raw, index_bits, index_labels_raw) = {
                let r = root.base();
                (
                    r.magic,
                    r.version,
                    r.seg_cap,
                    r.labels_raw,
                    r.vindex_raw,
                    r.index_bits,
                    r.index_labels_raw,
                )
            };
            if magic == MAGIC && version_supported(version) {
                // The persisted capacity governs, not the caller's.
                let cap = seg_cap as usize;
                let schema = IndexSchema::from_bits(index_bits)
                    .ok_or(GraphError::UnknownIndexSchema { bits: index_bits })?;
                Self::validate_schema(schema)?;
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
                let indexed_set = Self::fold_indexed(&index_labels);
                // `version_supported` above already rejected anything but
                // VERSION_ARENA, so there is exactly one layout to open.
                let store = {
                    let (dir, locs, stored_cap) = {
                        let r = root.base();
                        (r.arena_dir_raw, r.arena_locs_raw, r.arena_cap)
                    };
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
                    indexed_set: RefCell::new(indexed_set),
                    vindex,
                    volatile: RefCell::new(VolatileIndex::default()),
                    scans: Cell::new(0),
                    store,
                });
            }
            // Incompatible/stale format: do NOT touch the existing graph.
            // A v3 graph lands here now — deliberately, see `version_supported`.
            return Err(GraphError::StaleVersion {
                found: version,
                expected: VERSION_ARENA,
            });
        }

        let labels = SegVec::create(cap)?;
        let index_labels = SegVec::<IndexedLabel>::create(cap)?;
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
            indexed_set: RefCell::new(HashSet::new()),
            vindex,
            volatile: RefCell::new(VolatileIndex::default()),
            scans: Cell::new(0),
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
    /// persistent `data/` namespace is unsupported on the current Twizzler build
    /// (the pager's external unlink is unimplemented). Instead it rewrites the
    /// existing root object in place to point at fresh, empty registries. Old
    /// registry objects are orphaned; reclamation and true unregistration are
    /// future work.
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

    /// Discard a graph and rebuild it empty, packing `arena_cap` vertices per
    /// arena object.
    ///
    /// Tests need this to be idempotent across runs: the disk image survives
    /// between QEMU invocations, so a graph left behind by an earlier run would
    /// otherwise be re-opened in whatever format it was written in. Unbinding
    /// the name is not an option — `data/` names cannot be removed on this
    /// build — so the root is rewritten in place, exactly as `reset` does.
    pub fn open_or_create_arena_with_index(
        name: &str,
        arena_cap: usize,
        schema: IndexSchema,
    ) -> Result<Graph> {
        Self::open_inner_schema(name, DEFAULT_SEG_CAP, arena_cap, schema)
    }

    pub fn reset_arena_with_index(
        name: &str,
        arena_cap: usize,
        schema: IndexSchema,
    ) -> Result<()> {
        Self::validate_schema(schema)?;
        if arena_cap == 0 || arena_cap > u32::MAX as usize {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        Self::reset_inner_schema(name, None, arena_cap, schema)
    }

    pub fn reset_arena(name: &str, arena_cap: usize) -> Result<()> {
        if arena_cap == 0 || arena_cap > u32::MAX as usize {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        Self::reset_inner_fmt(name, None, arena_cap)
    }

    fn reset_inner(name: &str, cap: Option<usize>) -> Result<()> {
        Self::reset_inner_fmt(name, cap, DEFAULT_ARENA_CAP)
    }

    /// The name is not unbound, because `data/` entries cannot be removed
    /// on this build. The root object is retained and its magic cleared, which
    /// makes it (a) unmistakably not a graph, so a later `open_or_create`
    /// refuses rather than reading freed ids, and (b) one leaked object per
    /// destroyed name instead of a whole graph. Re-using the name needs an
    /// explicit `reset`/`reset_arena`, which rebuilds in place.
    pub fn destroy(name: &str) -> Result<usize> {
        let mut namer = static_naming_factory().expect("naming service available");
        let path = format!("data/{name}");
        let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) else {
            return Ok(0); // nothing registered
        };
        let mut root = Object::<GraphRoot>::map(node.id.into(), rw())?;
        let (is_graph, version, cap, l, x, ad, al, il) = {
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
            )
        };
        if !is_graph {
            // Already destroyed (magic is `MAGIC_DESTROYED`), or never ours.
            // Idempotent either way, and safe: a destroyed root's registry ids
            // are zeroed, so there is nothing left to chase.
            return Ok(0);
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

        // `version_supported` above admits only `VERSION_ARENA`, so there is one
        // layout to walk.
        let mut ids = {
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
            if x != 0 {
                ids.push(x);
            }
            ids
        };
        // Guard against a double-free: an id reachable two ways (a shared
        // property object, say) would otherwise be deleted twice.
        ids.sort_unstable();
        ids.dedup();

        // Mark the root dead *before* freeing, so an interruption leaves a root
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
            Ok(())
        })?;

        Ok(reclaim::delete_all(ids))
    }

    /// Rebuild empty on VERSION 4, packing `arena_cap` vertices per arena.
    ///
    /// The *outgoing* graph may still be v3 — a disk image outlives the code
    /// that wrote it — so the inventory below keeps its v3 arm even though
    /// nothing creates v3 any more. That arm is what stops a stale image from
    /// leaking a graph's worth of objects on the first reset after the upgrade.
    fn reset_inner_schema(
        name: &str,
        cap: Option<usize>,
        arena_cap: usize,
        schema: IndexSchema,
    ) -> Result<()> {
        Self::reset_inner_fmt_schema(name, cap, arena_cap, schema)
    }

    fn reset_inner_fmt(name: &str, cap: Option<usize>, arena_cap: usize) -> Result<()> {
        Self::reset_inner_fmt_schema(name, cap, arena_cap, IndexSchema::default())
    }

    fn reset_inner_fmt_schema(
        name: &str,
        cap: Option<usize>,
        arena_cap: usize,
        schema: IndexSchema,
    ) -> Result<()> {
        let mut namer = static_naming_factory().expect("naming service available");
        let path = format!("data/{name}");
        let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) else {
            return Ok(()); // nothing registered
        };

        let mut root = Object::<GraphRoot>::map(node.id.into(), rw())?;
        // Only clobber something that is actually one of our graphs; trust the
        // stored capacity only if the root has the current layout.
        // A destroyed root is ours and rebuildable in place; only a root that
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
        let cap = cap.unwrap_or(DEFAULT_SEG_CAP);

        let old_ids = {
            let (l, x, ad, al) = {
                let r = root.base();
                (
                    r.labels_raw,
                    r.vindex_raw,
                    r.arena_dir_raw,
                    r.arena_locs_raw,
                )
            };
            match old_version {
                // A destroyed root already freed everything and zeroed its
                // registry ids; walking them would chase freed objects.
                _ if was_destroyed => Vec::new(),
                // Arenas, the label registry and the index. Matches any
                // *reclaimable* format, not just the current one — see
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
        let vindex = if schema.strategy == IndexStrategy::Persistent {
            Some(VIndex::new_persist()?)
        } else {
            None
        };
        let (labels_raw, vindex_raw) = (
            labels.dir_raw(),
            vindex.as_ref().map_or(0, |v| v.object().id().raw()),
        );
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
            b.arena_cap = arena_cap as u32;
            b.index_bits = schema.to_bits();
            b.index_labels_raw = index_labels.dir_raw();
            Ok(())
        })?;

        reclaim::delete_all(old_ids);
        Ok(())
    }

    /// The set supplied here is fixed for the record's lifetime, because the
    /// record's size is: every inbound `AdjRef.neighbor` holds its arena offset,
    /// so a record that grew would have to have all of them rewritten. Anything
    /// added later via [`Graph::set_vertex_prop`] becomes a *data* property,
    /// which lives behind one indirection and can move freely.
    ///
    /// Choosing is the user's job, and the criterion is access pattern: put
    /// a property here if traversals *filter* on it, since inline slots sit in
    /// cache lines a walk has already paid for. Put everything else in data
    /// properties — inline slots widen every record, and record width is what
    /// sets page density.
    ///
    /// Keys are interned through the same table as labels. They cannot collide:
    /// a label id is read from `record.label` and a key id from `slot.key_id`,
    /// which are different fields consulted in different contexts.
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

        // One arena allocation, shared with `cap-1` other vertices, instead of
        // the three whole objects v3 used (vertex + two adjacency `VecObject`s).
        let id = self.store.add_vertex(lbl, name, target.raw())?;
        self.index_on_insert(lbl, name, id)?;
        Ok(VertexId(id))
    }

    /// Add a typed edge `from -> to`: append an adjacency entry to `from`'s
    /// outgoing chain and `to`'s incoming chain. The edge itself is only a
    /// registry row — v3's separate edge object is gone.
    pub fn add_edge(&mut self, from: VertexId, label: &str, to: VertexId) -> Result<EdgeId> {
        let lbl = self.intern_label(label)?;

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
            }
            IndexStrategy::None => {}
        }
        Ok(())
    }

    fn validate_schema(schema: IndexSchema) -> Result<()> {
        if schema.rebuild == RebuildSource::Roots {
            return Err(GraphError::RebuildSourceUnimplemented);
        }
        Ok(())
    }

    pub fn index_strategy(&self) -> IndexStrategy {
        self.schema.strategy
    }

    pub fn index_schema(&self) -> IndexSchema {
        self.schema
    }

    pub fn scans_performed(&self) -> usize {
        self.scans.get()
    }

    pub fn index_builds(&self) -> usize {
        self.volatile.borrow().builds()
    }

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

    pub fn set_label_indexed(&mut self, label: &str, indexed: bool) -> Result<()> {
        if self.schema.strategy == IndexStrategy::None {
            return Err(GraphError::IndexingDisabled);
        }
        let lbl = self.intern_label(label)?;
        // Idempotent, and it has to be. The log is append-only, and callers
        // declare at every open (`gstress::declare_lookup_labels`, `rns`), so
        // re-appending an unchanged state would grow it without bound across
        // reopens — a slow leak in the structure introduced to avoid a leak.
        if self.indexed_set.borrow().contains(&lbl) == indexed {
            return Ok(());
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
            IndexStrategy::Persistent => true,
            IndexStrategy::None => false,
            IndexStrategy::LazyLabel => self.indexed_set.borrow().contains(&lbl),
        }
    }

    fn indexed_label_ids(&self) -> Vec<u32> {
        self.indexed_set.borrow().iter().copied().collect()
    }

    /// Whether an insert of `lbl` should touch the index at all. Under the lazy
    /// strategy an *unbuilt* index stays unbuilt — see
    /// [`VolatileIndex::insert_if_built`].
    fn indexes_on_insert(&self, lbl: u32) -> bool {
        self.is_label_indexed_id(lbl)
    }

    /// Find a vertex by (label, name).
    pub fn find_vertex(&self, label: &str, name: &str) -> Lookup {
        let Some(lbl) = self.find_label(label) else {
            // An un-interned label is an authoritative negative, whatever
            // the policy: `intern_label` runs on every insert, so a label with
            // no registry entry cannot be carried by any record. Returning
            // `NotIndexed` here — as this did at first — would report "I did not
            // look" about a question that needs no looking, and would make
            // `find_vertex` on a fresh graph indistinguishable from one on a
            // misconfigured schema. Caught by
            // `arena_graph::arena_read_paths_return_the_expected_shape`
            // ("label is part of the key") and by the post-reset lookup in
            // `reclaim::reset_leaves_a_working_empty_graph`.
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

    fn ensure_volatile_built(&self) {
        if self.volatile.borrow().is_built() {
            return;
        }
        let indexed: Vec<u32> = self.indexed_label_ids();
        let mut map = HashMap::new();
        match self.schema.rebuild {
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
            // Unreachable: `validate_schema` refuses `Roots` at open/create, so
            // a graph configured this way never gets far enough to rebuild.
            // Deliberately not falling through to `Scan` — quietly running a
            // different strategy than the schema asks for is how a measurement
            // ends up describing something other than what it claims.
            RebuildSource::Roots => unreachable!("Roots is refused by validate_schema"),
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

    // --- arena seams for the DSL (`vertex.rs`) ------------------------------

    /// `(edge_id, edge_label, neighbour_id)` in traversal order. `VertexView`
    /// uses this instead of mapping adjacency objects.
    pub(crate) fn arena_adjacency(
        &self,
        id: VertexId,
        out: bool,
        inc: bool,
    ) -> Vec<(u64, u32, u64)> {
        // Worth stating because deleting a filter is exactly the change that
        // looks like a regression later: the guarantee moved rather than went
        // away, and `deleting_an_edge_agrees` in `graph-eval` is what would
        // notice if it had gone away.
        //
        // It is also a small win. The old filter ran `is_edge_alive` per entry,
        // which now resolves records; keeping it would have put that on the hot
        // path for no benefit.
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

    fn arena_neighbors(
        &self,
        id: VertexId,
        labels: Labels,
        out: bool,
        inc: bool,
    ) -> Vec<VertexId> {
        // Routed through `arena_adjacency` rather than the store's own
        // `neighbors_labeled` so the dead-edge filter applies here too — the
        // two must not have separate notions of which entries count.
        let filter = self.resolve_labels(labels);
        self.arena_adjacency(id, out, inc)
            .into_iter()
            .filter(|(_, l, _)| filter.as_ref().map_or(true, |ls| ls.contains(l)))
            .map(|(_, _, nb)| VertexId(nb))
            .collect()
    }

    /// All live vertex ids in the graph. Linear scan.
    pub fn vertices(&self) -> Vec<VertexId> {
        self.store.vertices().into_iter().map(VertexId).collect()
    }

    /// An edge's label and endpoints by id, or `None` if it is deleted or is
    /// not an edge.
    ///
    /// That second case is new with the unified id space and is the runtime
    /// check replacing what the type system used to give us: `EdgeId` and
    /// `VertexId` are the same type now, so "edge passed where a vertex belongs"
    /// cannot be rejected at compile time. `IS_EDGE` does the rejecting instead,
    /// and `edge_info` on a vertex record must return `None`.
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

    /// Delete a vertex (tombstone). Its incident edges become hidden too, since
    /// an edge is alive only while both endpoints are. No-op if already gone.
    ///
    /// v3 additionally freed the vertex's two adjacency objects and its property
    /// object here. v4 has no per-vertex adjacency objects at all, and the
    /// property object is deliberately left named by the tombstoned record: it
    /// is unreachable through the graph but still allocated, so
    /// `owned_object_ids` must keep reporting it or nothing will ever free it.
    pub fn delete_vertex(&mut self, id: VertexId) -> Result<()> {
        // Drop the name from an already-built volatile map before the record
        // goes, or a lookup could resolve a tombstone. The `locs` liveness check
        // in `find_vertex` would catch it anyway; this keeps the map honest
        // rather than relying on that second line of defence.
        if self.schema.strategy == IndexStrategy::LazyLabel {
            if let (Some(lbl), Some(nm)) = (self.vertex_label_id(id), self.store.vertex_name(id.0))
            {
                self.volatile
                    .borrow_mut()
                    .remove_if_built(lbl, NameKey::new(&nm));
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
    pub fn delete_edge(&mut self, id: EdgeId) -> Result<()> {
        if !self.store.is_edge(id.0) {
            return Ok(()); // not an edge record: no-op, as for an unknown id
        }
        self.store.delete_vertex(id.0)?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn owned_object_ids(&self) -> Vec<u128> {
        let mut ids = Vec::new();
        ids.extend(self.store.owned_object_ids());
        // Edges: no object to map, and the property id lives in the mirror.
        // `verts` is vestigial on v4 but still allocated, and still freed.
        ids.extend(self.labels.object_ids());
        ids.extend(self.index_labels.object_ids());
        ids.extend(self.index_object_ids());
        ids
    }

    #[cfg(test)]
    pub(crate) fn record_touches(&self) -> usize {
        self.store.record_touches()
    }

    #[cfg(test)]
    pub(crate) fn reset_record_touches(&self) {
        self.store.reset_record_touches();
    }

    #[cfg(test)]
    pub(crate) fn vertex_props_raw(&self, v: VertexId) -> Option<u128> {
        self.store.live_record(v.0).map(|_| 0)
    }

    #[cfg(test)]
    pub(crate) fn data_block_reads(&self) -> usize {
        self.store.data_block_reads()
    }

    #[cfg(test)]
    pub(crate) fn edge_props_raw(&self, e: EdgeId) -> Option<u128> {
        self.store.live_record(e.0).map(|_| 0)
    }

    /// Set a property on a vertex; errors if it is missing or tombstoned.
    /// Creates the vertex's property object on first use and records its id in
    /// the vertex's arena record.
    pub fn set_vertex_prop(&mut self, v: VertexId, key: &str, val: PropValue) -> Result<()> {
        if !self.is_vertex_alive(v) {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
        // If the key is already an inline traversal slot, update it there.
        // Writing the side object instead would leave two values for one key,
        // with readers preferring the stale inline one — a divergence invisible
        // from outside, since a wrong property reads exactly like a right one.
        // Updating in place is sound because a slot is fixed-size; it is
        // *adding* a key that the format forbids, not changing one.
        let key_id = self.intern_label(key)?;
        if self.store.set_traversal_prop(v.0, key_id, val) == Some(true) {
            return Ok(());
        }
        self.store.set_data_prop(v.0, key_id, val)?;
        Ok(())
    }

    /// A vertex property, or `None` if unset or the vertex is dead.
    pub fn get_vertex_prop(&self, v: VertexId, key: &str) -> Option<PropValue> {
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
            .filter_map(|s| self.label_name(s.key_id).map(|k| (k, s.val)))
            .collect()
    }

    /// Set a property on an edge; errors if it is missing, tombstoned, or has
    /// a dead endpoint. The property-object id lives in the edge registry,
    /// since there is no edge object to hold it.
    pub fn set_edge_prop(&mut self, e: EdgeId, key: &str, val: PropValue) -> Result<()> {
        if !self.is_edge_alive(e) {
            return Err(GraphError::Twz(ArgumentError::InvalidArgument.into()));
        }
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
        Ok(())
    }

    /// Whether `id` names a live edge record whose endpoints are both live.
    ///
    /// The `is_edge` test is what keeps the unified id space honest: without it
    /// every live vertex would answer "yes" and `edge_info`/`get_edge_prop`
    /// would happily treat a vertex as an edge.
    pub(crate) fn is_edge_alive(&self, id: EdgeId) -> bool {
        if !self.store.is_edge(id.0) || !self.store.is_alive(id.0) {
            return false;
        }
        // An edge is alive only while both endpoints are — unchanged semantics,
        // now read from the edge's own chains rather than a registry mirror.
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

    /// Read back a vertex's data from the registry, or `None` if it is deleted.
    /// O(1): ids are append indices, so the record is at position `id`.
    pub fn vertex_info(&self, id: VertexId) -> Option<VertexInfo> {
        // An edge record is not a vertex. The mirror of the `IS_EDGE` check
        // in `edge_info`, and it was missing: with one id space the type system
        // no longer separates the two, so every accessor has to reject the
        // wrong kind at runtime or it will happily describe an edge as a vertex.
        // `gstress verify` found this by reading an id that had silently become
        // an edge's.
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

    /// Diagnostic, temporary — pass-through to
    /// [`ArenaStore::debug_liveness`].
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

    /// O(1): ids are append indices, so the record is at position `id`.

    // The content-keyed lookups below are linear scans (where a hachage index
    // would later go): find_label by name, find_vertex by (label, name), and
    // vertices_by_label.

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
    /// Why a closure rather than a field. `PHMsession<'a>` borrows the map,
    /// so it cannot be stored beside `vindex` in `Graph` — that is a
    /// self-referential borrow. `ArenaStore` gets away with holding its
    /// transactions because `TxObject<ArenaBase>` is owned. Scoping the session
    /// to a closure is what the borrow checker leaves available, and it also
    /// makes the durability boundary explicit: the index is durable when the
    /// closure returns, not before.
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
            ..
        } = self;
        // Only the persistent strategy has a transaction to open. Under the
        // default there is no index object, so there is nothing to batch — the
        // 99.5% of insert cost this session existed to amortise is simply not
        // paid.
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
}

impl BulkInsert<'_> {
    /// As [`Graph::add_vertex`], but the index write joins the open transaction.
    pub fn add_vertex(&mut self, label: &str, name: &str, target: ObjID) -> Result<VertexId> {
        let lbl = self.intern_label(label)?;
        let id = self.store.add_record(lbl, name, target.raw(), &[], false)?;
        let indexed = match self.schema.strategy {
            IndexStrategy::Persistent => true,
            IndexStrategy::None => false,
            IndexStrategy::LazyLabel => self.indexed.borrow().contains(&lbl),
        };
        if indexed {
            match (&mut self.session, self.schema.strategy) {
                (Some(sess), _) => {
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
        // Every number *except* the current one. Written as a filter over a
        // range rather than a literal list, which is how `9` ended up in a
        // "must not be readable" list on the very build that made 9 current.
        for v in (0..=32u32).filter(|v| *v != VERSION_ARENA) {
            assert!(!version_supported(v), "version {v} must not be readable");
        }
    }

    /// Freeing is gated on whether the *object graph* is walkable — a weaker
    /// condition than readability, but not a free pass.
    ///
    /// Currently no predecessor qualifies, and that is a statement about
    /// format 9 rather than a permanent one: 7 qualified while 8 was current,
    /// because 7 → 8 moved only a trailing root field. The rule is what to
    /// assert, not the membership.
    #[test]
    fn reclaimability_tracks_whether_records_are_still_walkable() {
        assert!(version_reclaimable(VERSION_ARENA));
        // Format 7 was reclaimable while 8 was current, because 7 → 8 touched
        // only a trailing root field. Format 9 moved the record layout, and the
        // inventory walk reads records — so 7 dropped out, deliberately.
        assert!(
            !version_reclaimable(VERSION_ARENA_NOCAP),
            "a format whose record layout we can no longer read must not be \
             walked for object ids: leaking beats mis-freeing"
        );
    }

    #[test]
    fn v3_is_not_reclaimable() {
        assert!(!version_reclaimable(VERSION));
        for v in (0..=32u32).filter(|v| *v != VERSION_ARENA) {
            assert!(!version_reclaimable(v), "version {v} must not be freed");
        }
    }

    /// The invariant that the 7 → 8 bump broke. Anything this build can
    /// read, it must also be able to free — otherwise opening a graph and then
    /// resetting it leaks the very objects it was just using. Asserting the
    /// relationship rather than two enumerations is what makes this survive the
    /// next bump: a new format added to `version_supported` alone fails here.
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
