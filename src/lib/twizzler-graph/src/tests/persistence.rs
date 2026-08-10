//! Lifecycle: reopen-by-name, reset semantics, the format guard, index
//! persistence, and graph isolation.
//!
//! These tests do not demonstrate durability across a reboot, and nothing
//! in this suite can: every test runs inside a single QEMU session, so the
//! strongest claim available here is that a graph survives dropping its
//! handles and being re-opened *within one boot* — which exercises remapping,
//! not the write-back path to the disk image. A graph could in principle live
//! entirely in mapped memory and pass every test in this file.

use twizzler::object::ObjID;

use super::fresh;
use crate::{Graph, GraphError};

#[test]
fn reopen_within_boot_persists() {
    let name = "t-reopen";
    let _ = Graph::reset(name);
    let v = {
        let mut g = Graph::open_or_create(name).unwrap();
        g.add_vertex("file", "persisted", ObjID::new(0)).unwrap()
    };
    // Reopen the same graph by name and confirm the vertex is still there.
    let g2 = Graph::open_or_create(name).unwrap();
    assert_eq!(g2.vertex_info(v).unwrap().name, "persisted");
    let _ = Graph::reset(name);
}

#[test]
fn reset_clears_and_is_idempotent() {
    let name = "t-reset";
    let _ = Graph::reset(name);
    {
        let mut g = Graph::open_or_create(name).unwrap();
        g.add_vertex("file", "x", ObjID::new(0)).unwrap();
    }
    Graph::reset(name).unwrap();
    // Resetting again is a no-op (not an error).
    Graph::reset(name).unwrap();
    // After reset, the registration is reused and the graph is empty.
    let g = Graph::open_or_create(name).unwrap();
    assert_eq!(g.vertices_by_label("file").len(), 0);
    let _ = Graph::reset(name);
}

#[test]
fn vertex_index_find_delete_and_persist() {
    let name = "t-vindex";
    let _ = Graph::reset(name);
    let (t, deleted) = {
        let mut g = Graph::open_or_create(name).unwrap();
        let t = g.add_vertex("tag", "thesis", ObjID::new(0)).unwrap();
        let d = g.add_vertex("tag", "gone", ObjID::new(0)).unwrap();
        // Index lookup.
        assert_eq!(g.find_vertex("tag", "thesis"), Some(t));
        // Deleted vertices are not returned by the index lookup.
        g.delete_vertex(d).unwrap();
        assert_eq!(g.find_vertex("tag", "gone"), None);
        (t, d)
    };
    // The index persists: reopen and look up again.
    let g = Graph::open_or_create(name).unwrap();
    assert_eq!(g.find_vertex("tag", "thesis"), Some(t));
    assert_eq!(g.find_vertex("tag", "gone"), None);
    assert!(g.vertex_info(deleted).is_none());
    let _ = Graph::reset(name);
}

#[test]
fn delete_persists_on_reopen() {
    let name = "t-delp";
    let _ = Graph::reset(name);
    let a = {
        let mut g = Graph::open_or_create(name).unwrap();
        let a = g.add_vertex("n", "a", ObjID::new(0)).unwrap();
        g.delete_vertex(a).unwrap();
        a
    };
    let g = Graph::open_or_create(name).unwrap();
    assert!(g.vertex_info(a).is_none());
    let _ = Graph::reset(name);
}

#[test]
fn stale_v2_root_detected_and_resettable() {
    use naming::{static_naming_factory, GetFlags};
    use twizzler::object::{MapFlags, Object, ObjectBuilder};

    use crate::graph::{GraphRoot, MAGIC, VERSION_ARENA};

    let name = "t-stalev2";
    let path = format!("data/{name}");
    let mut namer = static_naming_factory().expect("naming service available");
    let rw = MapFlags::READ | MapFlags::WRITE | MapFlags::PERSIST;

    // Plant a version-2 root at data/<name>. If a previous run left a graph
    // registered here, rewrite it in place (data/ names cannot be removed).
    if let Ok(node) = namer.get(&path, GetFlags::FOLLOW_SYMLINK) {
        let mut root = Object::<GraphRoot>::map(node.id.into(), rw).unwrap();
        root.with_tx(|tx| {
            let mut b = tx.base_mut();
            b.magic = MAGIC;
            b.version = 2;
            Ok(())
        })
        .unwrap();
    } else {
        let root = ObjectBuilder::<GraphRoot>::default()
            .persist(true)
            .build(GraphRoot {
                magic: MAGIC,
                version: 2,
                seg_cap: 0,
                labels_raw: 0,
                vindex_raw: 0,
                arena_dir_raw: 0,
                arena_locs_raw: 0,
                arena_cap: 0,
            })
            .unwrap();
        namer.put(&path, root.id()).unwrap();
    }

    // The guard refuses and reports both versions; the graph is not touched.
    match Graph::open_or_create(name) {
        Err(GraphError::StaleVersion { found, expected }) => {
            assert_eq!(found, 2);
            assert_eq!(expected, VERSION_ARENA);
        }
        Ok(_) => panic!("expected StaleVersion, but the stale graph opened"),
        Err(e) => panic!("expected StaleVersion, got {e:?}"),
    }

    // Discarding is explicit — and works on the stale root.
    Graph::reset(name).expect("reset stale graph");
    let g = Graph::open_or_create(name).expect("open after reset");
    assert!(g.vertices().is_empty());
}

#[test]
fn multiple_graphs_coexist() {
    let mut g1 = fresh("t-multi-a");
    let mut g2 = fresh("t-multi-b");
    let a = g1.add_vertex("n", "only-in-a", ObjID::new(0)).unwrap();
    let b = g2.add_vertex("m", "only-in-b", ObjID::new(0)).unwrap();

    assert_ne!(g1.root_id(), g2.root_id());
    assert_eq!(g1.find_vertex("n", "only-in-a"), Some(a));
    assert_eq!(g1.find_vertex("m", "only-in-b"), None);
    assert_eq!(g2.find_vertex("m", "only-in-b"), Some(b));
    assert_eq!(g2.find_vertex("n", "only-in-a"), None);
    assert_eq!(g1.vertices(), vec![a]);
    assert_eq!(g2.vertices(), vec![b]);
}
