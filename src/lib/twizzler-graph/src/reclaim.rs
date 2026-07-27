#![allow(dead_code)]

//! Reclaim is best-effort by design. A delete that fails (the object is
//! already gone, or the kernel refuses it) must not fail the teardown that
//! asked for it: the caller's job is to leave the *graph* consistent, and a
//! surviving object is a leak, not a corruption. Callers that care about the
//! count use [`delete_all`]'s return value.

use twizzler::object::ObjID;
use twizzler_abi::syscall::{sys_object_ctrl, DeleteFlags, ObjectControlCmd};

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
