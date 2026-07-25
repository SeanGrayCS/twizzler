use twizzler::{
    collections::vec::{Vec as TwzVec, VecObject, VecObjectAlloc},
    error::TwzError,
    marker::Invariant,
    object::{MapFlags, ObjID, Object, ObjectBuilder},
};

use crate::name::NameKey;

type Result<T> = core::result::Result<T, TwzError>;

fn rw() -> MapFlags {
    MapFlags::READ | MapFlags::WRITE | MapFlags::PERSIST
}

/// A property value: fixed-size, invariant, and comparable — equality backs
/// the DSL's `has(key, value)`, ordering backs `order_by_prop`.
///
/// Ordering *within* a variant is the natural one (`Str` compares as text,
/// not as its raw record — see `NameKey`'s manual `Ord`). Ordering *across*
/// variants follows declaration order, which is arbitrary but deterministic;
/// mixed-type properties are a schema smell, and a stable answer beats an
/// unpredictable one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(C, u32)]
pub enum PropValue {
    I64(i64),
    U64(u64),
    Bool(bool),
    /// An object reference by raw id (relocatable, like the registries).
    ObjId(u128),
    /// A short string; truncates byte-wise at 31 like all `NameKey`s.
    Str(NameKey),
}
unsafe impl Invariant for PropValue {}

impl PropValue {
    /// Convenience constructor for string values (truncates like `NameKey`).
    pub fn str(s: &str) -> Self {
        PropValue::Str(NameKey::new(s))
    }
}

/// One (key, value) record in an element's property object.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct PropEntry {
    pub(crate) key: NameKey,
    pub(crate) val: PropValue,
}
unsafe impl Invariant for PropEntry {}

/// Set `key` in the property object at `raw` (0 = none yet): creates the
/// object on first use, overwrites in place if the key exists (preserving its
/// position), appends otherwise. Returns the property object's id — the
/// caller stores it back into the registry record iff it changed.
pub(crate) fn set_in(raw: u128, key: &str, val: PropValue) -> Result<u128> {
    let k = NameKey::new(key);
    let mut props: VecObject<PropEntry, VecObjectAlloc> = if raw == 0 {
        VecObject::new(ObjectBuilder::default().persist(true))?
    } else {
        VecObject::from(Object::<TwzVec<PropEntry, VecObjectAlloc>>::map(
            ObjID::new(raw),
            rw(),
        )?)
    };

    let mut found = None;
    for i in 0..props.len() {
        if let Some(e) = props.get_ref(i) {
            if e.key == k {
                found = Some(i);
                break;
            }
        }
    }
    match found {
        Some(i) => props.with_mut_slice(i..i + 1, |s| {
            s[0].val = val;
            Ok(())
        })?,
        None => props.push(PropEntry { key: k, val })?,
    }
    Ok(props.object().id().raw())
}

/// Read `key` from the property object at `raw` (0 = none).
pub(crate) fn get_in(raw: u128, key: &str) -> Option<PropValue> {
    if raw == 0 {
        return None;
    }
    let props = VecObject::from(
        Object::<TwzVec<PropEntry, VecObjectAlloc>>::map(
            ObjID::new(raw),
            MapFlags::READ | MapFlags::PERSIST,
        )
        .ok()?,
    );
    let k = NameKey::new(key);
    for i in 0..props.len() {
        if let Some(e) = props.get_ref(i) {
            if e.key == k {
                return Some(e.val);
            }
        }
    }
    None
}

/// All (key, value) pairs at `raw`, in insertion order.
pub(crate) fn list_in(raw: u128) -> Vec<(String, PropValue)> {
    if raw == 0 {
        return Vec::new();
    }
    let Ok(obj) = Object::<TwzVec<PropEntry, VecObjectAlloc>>::map(
        ObjID::new(raw),
        MapFlags::READ | MapFlags::PERSIST,
    ) else {
        return Vec::new();
    };
    let props = VecObject::from(obj);
    let mut out = Vec::with_capacity(props.len());
    for i in 0..props.len() {
        if let Some(e) = props.get_ref(i) {
            out.push((e.key.as_str().to_string(), e.val));
        }
    }
    out
}
