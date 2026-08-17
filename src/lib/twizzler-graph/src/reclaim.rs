// `delete_all` is live at `Graph::destroy` and `Graph::reset_inner_fmt`.

//! Reclaim is best-effort by design. A delete that fails (the object is
//! already gone, or the kernel refuses it) must not fail the teardown that
//! asked for it: the caller's job is to leave the *graph* consistent, and a
//! surviving object is a leak, not a corruption. Callers that care about the
//! count use [`delete_all`]'s return value.

use twizzler::object::ObjID;
use twizzler_abi::syscall::{sys_object_ctrl, sys_object_stat, DeleteFlags, ObjectControlCmd};

/// Resident pages of one object, or `None` if the kernel has no such object.
///
/// - `None` — `lookup_object` failed, i.e. the id is out of the object table.
/// - `Some(n)` — still resident, holding `n` pages.
///
/// Known bias, and it points the safe way. `pages` counts pages in *this
/// object's* range tree only: page-table pages, the range tree itself and
/// pager-held frames not yet attached are all excluded. A sum over our objects
/// is therefore a lower bound on frames retained — so a positive finding
/// (pages survive a `Delete`) is strong, while a zero needs a system-level
/// cross-check before it is believed.
pub(crate) fn object_pages(raw: u128) -> Option<usize> {
    if raw == 0 {
        return None;
    }
    sys_object_stat(ObjID::new(raw)).ok().map(|info| info.pages)
}

/// Sum [`object_pages`] over `raws`, returning `(objects_still_present, pages)`.
/// Ids the kernel no longer knows contribute to neither figure, which is what
/// makes one helper usable on both sides of a deletion.
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
/// accepted, which is what the reclaim tests assert on.
pub(crate) fn delete_all(raws: impl IntoIterator<Item = u128>) -> usize {
    raws.into_iter().filter(|r| delete_raw(*r)).count()
}
