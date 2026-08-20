//! Object reclaim: delete Twizzler objects and read their resident-page
//! counts. Used by `Graph::destroy` and `Graph::reset_inner_fmt`.
//!
//! Deletion is best-effort. A failed delete leaves a leak, not a corruption,
//! so teardown never fails because of one; callers that care about the count
//! use [`delete_all`]'s return value.
//!
//! Only pass objects that nothing can still reach. Traversal resolves chunked
//! invariant pointers before it checks liveness (`arena_store::walk_adj`), so
//! a record's arena must outlive its tombstones.

use twizzler::object::ObjID;
use twizzler_abi::syscall::{sys_object_ctrl, sys_object_stat, DeleteFlags, ObjectControlCmd};

/// Resident pages of one object, or `None` if the kernel has no such object.
///
/// Counts only pages in this object's own range tree. Page-table pages, the
/// tree itself, and pager-held frames not yet attached are excluded, so a sum
/// over objects is a lower bound on frames retained.
pub(crate) fn object_pages(raw: u128) -> Option<usize> {
    if raw == 0 {
        return None;
    }
    sys_object_stat(ObjID::new(raw)).ok().map(|info| info.pages)
}

/// Sum [`object_pages`] over `raws`, returning `(objects_still_present, pages)`.
/// Ids the kernel no longer knows contribute to neither figure, so the same
/// helper works before and after a deletion.
pub(crate) fn pages_of(raws: impl IntoIterator<Item = u128>) -> (usize, usize) {
    raws.into_iter()
        .filter_map(object_pages)
        .fold((0, 0), |(n, p), pages| (n + 1, p + pages))
}

/// Smallest thing that can own an object, and volatile on purpose — see
/// [`sweep_deleted`].
#[derive(Clone, Copy)]
#[repr(C)]
struct SweepCell {
    v: u64,
}
unsafe impl twizzler::marker::Invariant for SweepCell {}
impl twizzler::marker::BaseType for SweepCell {}

/// Force the kernel to re-run `scan_deleted`, and report whether it ran.
///
/// The `Delete` syscall marks the object, tells the pager, then calls
/// `scan_deleted` — and it reports success after the mark, whatever the scan
/// did. The scan frees an object only if it is pending-delete and mapped into
/// no address space, and nothing re-runs it on a timer, so an object whose
/// mapping drops after its delete sits pending until an unrelated `Delete`
/// sweeps it up.
///
/// That gives userspace a probe, because `scan_deleted` scans the whole map,
/// not just the id being deleted: deleting any object re-examines every
/// pending one. This creates a volatile throwaway and deletes it purely for
/// that side effect. Volatile matters: the kernel skips the pager entirely for
/// it, so the sweep cannot perturb the pager.
pub(crate) fn sweep_deleted() -> bool {
    use twizzler::object::ObjectBuilder;
    let Ok(obj) = ObjectBuilder::<SweepCell>::default().build(SweepCell { v: 0 }) else {
        return false;
    };
    let id = obj.id().raw();
    drop(obj);
    delete_raw(id)
}

/// Delete one object by raw id. Returns whether the kernel accepted it.
/// A zero id means "no such object" (our records use 0 as the null id) and is
/// reported as not-deleted without a syscall.
pub(crate) fn delete_raw(raw: u128) -> bool {
    if raw == 0 {
        return false;
    }
    sys_object_ctrl(
        ObjID::new(raw),
        ObjectControlCmd::Delete(DeleteFlags::empty()),
    )
    .is_ok()
}

/// Delete every id in `raws`, ignoring failures. Returns how many the kernel
/// accepted.
pub(crate) fn delete_all(raws: impl IntoIterator<Item = u128>) -> usize {
    raws.into_iter().filter(|r| delete_raw(*r)).count()
}
