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
    UnknownIndexSchema { bits: u32 },
    IndexingDisabled,
    RebuildSourceUnimplemented,
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
            GraphError::RebuildSourceUnimplemented => write!(
                f,
                "index rebuild source `Roots` is not implemented yet; use `Scan` \
                 (it is refused rather than downgraded so a measurement cannot \
                 silently describe the wrong strategy)"
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
