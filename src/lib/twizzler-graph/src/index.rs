//! The index as a schema decision.
//!
//! Indexing every record is the KV-store assumption, where every traversal
//! step is a keyed lookup. Index-free adjacency follows ids inside arenas and
//! never consults an index — only query entry does: a query resolves a
//! handful of roots by name and then walks. So the index covers roots, not
//! records, and which roots is a property of the workload. That makes it a
//! schema decision rather than a constant, which is what this module encodes.
//!
//! The seam mirrors [`crate::Placement`], which already does this for arena
//! layout: the engine asks the schema rather than hardcoding a policy, so a new
//! strategy costs an enum arm and no call-site changes.

use std::collections::HashMap;

use crate::name::NameKey;
use crate::vertex::VertexId;

/// What the graph indexes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexStrategy {
    /// Nothing. The cheapest graph: no index entries, no rebuild, and
    /// [`Lookup::NotIndexed`] for every name lookup. For loads that address
    /// vertices by id and never by name.
    None,
    /// Default. Only labels declared with `set_label_indexed`, held in
    /// memory and built on first lookup. Nothing is persisted or synced, so the
    /// index leaves the write path entirely.
    LazyLabel,
    /// Every record, in a persistent `hachage` map. Retained as the
    /// comparison arm so its size and sync cost can be compared against the
    /// default rather than assumed.
    Persistent,
}

/// What happens when a lookup names a label that is not indexed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnindexedLookup {
    /// Answer [`Lookup::NotIndexed`] and touch nothing. Default.
    ///
    /// Not because scanning is wrong — some workloads genuinely need name
    /// lookup on labels they chose not to index — but because a scan's cost is
    /// invisible at the call site: a lookup meant to resolve one root would
    /// silently become a full scan. Opting in is one builder call, and
    /// [`crate::Graph::scans_performed`] makes it auditable.
    Refuse,
    /// Walk records and answer authoritatively. Never returns `NotIndexed`.
    Scan,
}

/// Where a lazy index gets its entries when it is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RebuildSource {
    /// Walk every record. Persists nothing, but pages in every arena.
    /// Default, because it is the version with no extra persistent structure
    /// to justify.
    Scan,
    /// Keep one `u64` id per indexed record and build from that: rebuild
    /// reads only the recorded roots for declared labels instead of every
    /// record, at the cost of a small persistent id list.
    Roots,
}

/// The graph's indexing schema. Stored in `GraphRoot`, following `arena_cap`:
/// geometry that must stay uniform for the graph's lifetime belongs in the root,
/// not in the call that happened to open it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexSchema {
    pub strategy: IndexStrategy,
    pub unindexed: UnindexedLookup,
    pub rebuild: RebuildSource,
}

impl Default for IndexSchema {
    fn default() -> Self {
        Self::new(IndexStrategy::LazyLabel)
    }
}

impl IndexSchema {
    pub fn new(strategy: IndexStrategy) -> Self {
        Self {
            strategy,
            unindexed: UnindexedLookup::Refuse,
            rebuild: RebuildSource::Scan,
        }
    }

    pub fn unindexed(mut self, u: UnindexedLookup) -> Self {
        self.unindexed = u;
        self
    }

    pub fn rebuild(mut self, r: RebuildSource) -> Self {
        self.rebuild = r;
        self
    }

    /// Pack for `GraphRoot`. Three small enums in one `u32` rather than three
    /// fields, so a fourth policy does not need another format bump.
    pub(crate) fn to_bits(self) -> u32 {
        let s = match self.strategy {
            IndexStrategy::None => 0,
            IndexStrategy::LazyLabel => 1,
            IndexStrategy::Persistent => 2,
        };
        let u = match self.unindexed {
            UnindexedLookup::Refuse => 0,
            UnindexedLookup::Scan => 1,
        };
        let r = match self.rebuild {
            RebuildSource::Scan => 0,
            RebuildSource::Roots => 1,
        };
        s | (u << 8) | (r << 16)
    }

    /// `None` for bits this build does not understand — a graph written by a
    /// newer build must not be silently reinterpreted as the default, which
    /// would index the wrong labels and answer lookups wrongly rather than
    /// refusing to open.
    pub(crate) fn from_bits(b: u32) -> Option<Self> {
        // The top byte is the room the u32 packing reserved for a fourth
        // policy. A nonzero value there comes from a newer build, so it must
        // refuse like any other unknown bit pattern.
        if b >> 24 != 0 {
            return None;
        }
        Some(Self {
            strategy: match b & 0xff {
                0 => IndexStrategy::None,
                1 => IndexStrategy::LazyLabel,
                2 => IndexStrategy::Persistent,
                _ => return None,
            },
            unindexed: match (b >> 8) & 0xff {
                0 => UnindexedLookup::Refuse,
                1 => UnindexedLookup::Scan,
                _ => return None,
            },
            rebuild: match (b >> 16) & 0xff {
                0 => RebuildSource::Scan,
                1 => RebuildSource::Roots,
                _ => return None,
            },
        })
    }
}

/// The result of a name lookup.
///
/// `NotFound` and `NotIndexed` are different answers and must never be
/// collapsed. `Option` cannot carry the distinction, and returning `None` for
/// an unindexed label would read as "no such vertex" when the truth is "I did
/// not look".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lookup {
    Found(VertexId),
    /// Searched, and authoritatively absent.
    NotFound,
    /// This label is not indexed and [`UnindexedLookup::Refuse`] is in force, so
    /// no search was performed. This is not "absent".
    NotIndexed,
}

impl Lookup {
    /// For call sites that only want the happy path. This discards the
    /// `NotFound`/`NotIndexed` distinction — only use it where the label is
    /// known to be indexed.
    pub fn found(self) -> Option<VertexId> {
        match self {
            Lookup::Found(v) => Some(v),
            _ => None,
        }
    }

    pub fn is_found(self) -> bool {
        matches!(self, Lookup::Found(_))
    }
}

/// The in-memory index for [`IndexStrategy::LazyLabel`].
///
/// `map` is `None` until the first lookup builds it, so a pure bulk load
/// never pays for it. `builds` is a test seam for that.
#[derive(Default)]
pub(crate) struct VolatileIndex {
    map: Option<HashMap<(u32, NameKey), u64>>,
    builds: usize,
}

impl VolatileIndex {
    pub(crate) fn is_built(&self) -> bool {
        self.map.is_some()
    }

    pub(crate) fn builds(&self) -> usize {
        self.builds
    }

    pub(crate) fn len(&self) -> usize {
        self.map.as_ref().map_or(0, |m| m.len())
    }

    /// Install a freshly built map. The caller supplies the entries because
    /// walking records belongs to `ArenaStore`, not here.
    pub(crate) fn install(&mut self, entries: HashMap<(u32, NameKey), u64>) {
        self.map = Some(entries);
        self.builds += 1;
    }

    pub(crate) fn get(&self, label: u32, name: NameKey) -> Option<u64> {
        self.map.as_ref()?.get(&(label, name)).copied()
    }

    /// Keep an already-built map current. Insertions must not build it — a
    /// bulk load into an unbuilt index has to stay free, or the lazy saving
    /// is lost.
    pub(crate) fn insert_if_built(&mut self, label: u32, name: NameKey, id: u64) {
        if let Some(m) = self.map.as_mut() {
            m.insert((label, name), id);
        }
    }

    /// Drop a key from an already-built map, so a delete cannot leave a name
    /// resolving to a tombstoned record — but only when the entry is the
    /// record being deleted. Duplicate `(label, name)` pairs are permitted,
    /// so a key-only removal could un-index a live twin, and `find_vertex`
    /// would then answer an authoritative `NotFound` for it until the next
    /// rebuild. The id match also keeps this arm answer-identical to
    /// `Persistent` under duplicates.
    pub(crate) fn remove_if_built(&mut self, label: u32, name: NameKey, id: u64) {
        if let Some(m) = self.map.as_mut() {
            if m.get(&(label, name)) == Some(&id) {
                m.remove(&(label, name));
            }
        }
    }

    /// Invalidate, forcing a rebuild on next lookup.
    pub(crate) fn clear(&mut self) {
        self.map = None;
    }
}
