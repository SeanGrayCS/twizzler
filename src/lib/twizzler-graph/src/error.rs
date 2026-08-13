//! Engine error type.
//!
//! Wraps the underlying Twizzler error and adds engine-specific cases. The
//! `StaleVersion` case is returned when an on-disk graph's format does not match
//! the engine; the existing graph is left intact and the caller decides what to
//! do.

use core::fmt;

use twizzler::error::TwzError;

#[derive(Debug)]
pub enum GraphError {
    /// An underlying Twizzler error (object, naming, etc.).
    Twz(TwzError),
    /// The on-disk graph has an incompatible format. The graph is left intact;
    /// the caller must explicitly reset it to discard it.
    StaleVersion { found: u32, expected: u32 },
    /// The graph's index schema bits are not ones this build understands —
    /// written by a newer engine. Refusing is deliberate: reinterpreting
    /// unknown bits as the default would index the wrong labels and answer
    /// lookups wrongly rather than failing.
    UnknownIndexSchema { bits: u32 },
    /// `set_label_indexed` under [`crate::IndexStrategy::None`], which
    /// indexes nothing by definition. Erroring rather than silently accepting
    /// the call, so a workload cannot believe it declared an index it did not.
    IndexingDisabled,
    /// A text property exceeded the inline limit. Refused rather than
    /// truncated: the API must not be able to silently shorten a value. Use a
    /// blob for values with no length bound; blobs are not queryable, which is
    /// the trade.
    TextTooLong { len: usize, max: usize },
    /// Strict mode: a `repeat` stopped at its depth cap, so the result would
    /// be short rather than complete. Returned only by `StrictRepeat`'s
    /// terminators; the lax path reports the same condition through
    /// `VertexTraversal::hit_depth_cap`.
    WalkTruncated,
}

impl From<TwzError> for GraphError {
    fn from(e: TwzError) -> Self {
        GraphError::Twz(e)
    }
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::Twz(e) => write!(f, "{e:?}"),
            GraphError::UnknownIndexSchema { bits } => write!(
                f,
                "unknown index schema bits {bits:#x}: this graph was written by a \
                 newer engine, and guessing a policy would answer name lookups \
                 wrongly rather than not at all"
            ),
            GraphError::WalkTruncated => write!(
                f,
                "recursive traversal stopped at its depth cap: the result is \
                 truncated, not complete (use `max_depth` and check \
                 `hit_depth_cap()` to accept a partial answer)"
            ),
            GraphError::TextTooLong { len, max } => write!(
                f,
                "text property is {len} bytes, over the {max}-byte limit for a \
                 queryable value; store it as a blob (not filterable) or shorten it"
            ),
            GraphError::IndexingDisabled => write!(
                f,
                "cannot declare an indexed label: this graph's index strategy is \
                 `None`, which indexes nothing"
            ),
            GraphError::StaleVersion { found, expected } => write!(
                f,
                "stale graph format: on-disk version {found}, engine expects {expected}; \
                 the existing graph was left intact (reset it explicitly to discard)"
            ),
        }
    }
}

impl std::error::Error for GraphError {}

/// Engine result type.
pub type Result<T> = core::result::Result<T, GraphError>;
