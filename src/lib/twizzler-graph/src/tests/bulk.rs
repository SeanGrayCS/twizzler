use twizzler::object::ObjID;

use super::{fresh, fresh_cap};
use crate::{Graph, Labels, VertexId};

#[test]
fn bulk_visible_after_close() {
    let mut g = fresh("t-bulk1");
    let (vs, es) = g
        .bulk(|b| {
            let mut vs = Vec::new();
            for i in 0..10 {
                vs.push(b.add_vertex("n", &format!("v{i}"), ObjID::new(0))?);
            }
            let mut es = Vec::new();
            for i in 0..9 {
                es.push(b.add_edge(vs[i], "e", vs[i + 1])?);
            }
            Ok((vs, es))
        })
        .expect("bulk");

    assert_eq!(g.vertices().len(), 10);
    for (i, v) in vs.iter().enumerate() {
        assert_eq!(g.vertex_info(*v).unwrap().name, format!("v{i}"));
    }
    assert_eq!(g.find_vertex("n", "v7"), Some(vs[7]));
    assert_eq!(g.out_neighbors(vs[0], Labels::any()), vec![vs[1]]);
    assert_eq!(g.in_neighbors(vs[9], Labels::any()), vec![vs[8]]);
    assert_eq!(g.edge_info(es[8]).unwrap().to, vs[9]);
    // A chain walk through the DSL sees the bulk data.
    assert_eq!(
        g.traversal()
            .v(vs[0])
            .out(Labels::any())
            .out(Labels::any())
            .to_ids(),
        vec![vs[2]]
    );
}

#[test]
fn bulk_persists_on_reopen() {
    let name = "t-bulk2";
    let _ = Graph::reset(name);
    let (a, b_) = {
        let mut g = Graph::open_or_create(name).unwrap();
        g.bulk(|b| {
            let a = b.add_vertex("f", "a", ObjID::new(0))?;
            let c = b.add_vertex("f", "b", ObjID::new(0))?;
            b.add_edge(a, "e", c)?;
            Ok((a, c))
        })
        .unwrap()
    };
    let g = Graph::open_or_create(name).unwrap();
    assert_eq!(g.find_vertex("f", "a"), Some(a));
    assert_eq!(g.find_vertex("f", "b"), Some(b_));
    assert_eq!(g.out_neighbors(a, Labels::any()), vec![b_]);
    assert_eq!(g.vertex_info(b_).unwrap().name, "b");
    let _ = Graph::reset(name);
}

#[test]
fn bulk_spans_segment_rollover() {
    let name = "t-bulk3";
    let vs = {
        let mut g = fresh_cap(name, 4);
        let vs = g
            .bulk(|b| {
                let mut vs = Vec::new();
                for i in 0..10 {
                    vs.push(b.add_vertex("n", &format!("v{i}"), ObjID::new(0))?);
                }
                Ok(vs)
            })
            .unwrap();
        assert_eq!(g.registry_segments().0, 3);
        for i in [0usize, 3, 4, 9] {
            assert_eq!(g.vertex_info(vs[i]).unwrap().name, format!("v{i}"));
        }
        vs
    };
    // The rollover performed inside the batch survives reopen.
    let g = Graph::open_or_create(name).unwrap();
    assert_eq!(g.registry_segments().0, 3);
    assert_eq!(g.find_vertex("n", "v9"), Some(vs[9]));
    let _ = Graph::reset(name);
}

#[test]
fn bulk_mixes_with_direct_ops() {
    let mut g = fresh("t-bulk4");
    let pre = g.add_vertex("n", "pre", ObjID::new(0)).unwrap();
    let t = g
        .bulk(|b| {
            let t = b.add_vertex("n", "t", ObjID::new(0))?;
            // Edge from a pre-existing vertex, added inside the batch.
            b.add_edge(pre, "e", t)?;
            Ok(t)
        })
        .unwrap();
    let post = g.add_vertex("n", "post", ObjID::new(0)).unwrap();
    g.add_edge(t, "e", post).unwrap();

    assert_eq!(g.out_neighbors(pre, Labels::any()), vec![t]);
    assert_eq!(g.out_neighbors(t, Labels::any()), vec![post]);
    // One interned label across both paths.
    assert_eq!(g.vertices_by_label("n").len(), 3);
    assert_eq!(g.vertices().len(), 3);
}

/// CANARY — the load-bearing assumption of the crate-local bulk path.
#[test]
fn tx_abort_does_not_roll_back() {
    use twizzler::object::{ObjectBuilder, TypedObject};

    let obj = ObjectBuilder::default().build(1u32).expect("build");
    let mut tx = obj.as_tx().expect("as_tx");
    let mut base = tx.base_mut();
    *base = 7;
    drop(base);
    tx.abort();
    drop(tx);
    assert_eq!(
        *obj.base(),
        7,
        "TxObject::abort rolled back a write — upstream tx semantics changed; \
         the A3 bulk path is now unsound. See the A3 note in docs/tasks.md."
    );
}

#[test]
fn bulk_error_propagates_nonatomically() {
    let mut g = fresh("t-bulk5");
    let r = g.bulk(|b| {
        b.add_vertex("n", "kept", ObjID::new(0))?;
        // Missing endpoint: this op fails and aborts the closure.
        b.add_edge(VertexId(0), "e", VertexId(9999))?;
        Ok(())
    });
    assert!(r.is_err(), "bulk should propagate the closure error");

    assert_eq!(g.find_vertex("n", "kept"), Some(VertexId(0)));
    // The graph stays usable after a failed batch.
    let v = g.add_vertex("n", "after", ObjID::new(0)).unwrap();
    assert_eq!(g.vertex_info(v).unwrap().name, "after");
}
