//! Property ceiling probe: how many vertices can carry a property before the
//! pager gives out?
//!
//! Vertex records are packed into arenas, but properties are not: the
//! property store keeps a lazily-created side object per element, so every
//! vertex carrying a property costs one persistent Twizzler object — the
//! thing the arena layout exists to avoid.
//!
//! Not a unit test, because frame exhaustion blocks the allocating thread
//! rather than returning an error — there is nothing to assert on. The
//! measurement is the console output, and the last count printed before a
//! stall is the ceiling.
//!
//! Arms — run each in its own boot:
//!
//!   gstress props [N]        # N vertices, one property on every one
//!   gstress props [N] none   # control: the same N vertices, no properties
//!
//! The control is not optional: without it a stall cannot be attributed to
//! property objects rather than to vertex count.

use std::time::Instant;

use twizzler::object::ObjID;
use twizzler_graph::{Graph, PropValue};

use crate::{heartbeat, HARNESS_REV};

const GRAPH: &str = "gstress-props";

/// Predicted ceiling, printed so the operator can watch it being crossed (or
/// not): ~2.37 M usable frames / ~149 frames per reference-free persistent
/// object. The number under test, not an input.
const PREDICTED_CEILING: usize = 15_900;

pub fn run(n: usize, with_props: bool) {
    println!(
        "GSTRESS STAMP harness={} mode=props N={} arm={}",
        HARNESS_REV,
        n,
        if with_props { "with-props" } else { "control" }
    );
    println!(
        "gstress: property ceiling probe — {} vertices, {}. Arithmetic predicts \
         a stall near {} in the with-props arm and none in the control.",
        n,
        if with_props {
            "one property object per vertex"
        } else {
            "no properties (control)"
        },
        PREDICTED_CEILING
    );

    // Setup sits outside the timer and is reported: on a dirty image the
    // reset must reclaim every property object a previous with-props run left
    // behind, which can take a long time. Clear the image before this probe;
    // if this line is not ~0, that is why.
    let setup = Instant::now();
    Graph::reset_arena(GRAPH, twizzler_graph::DEFAULT_ARENA_CAP).expect("reset_arena");
    let mut g =
        Graph::open_or_create_arena(GRAPH, twizzler_graph::DEFAULT_ARENA_CAP).expect("open");
    println!(
        "GSTRESS SETUP: reset + open {:.2}s (excluded from the run total)",
        setup.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let mut made = 0usize;
    for i in 0..n {
        let v = match g.add_vertex("p", &format!("p{i}"), ObjID::new(0)) {
            Ok(v) => v,
            Err(e) => {
                println!("GSTRESS PROPS CEILING: add_vertex failed at {i}: {e:?}");
                break;
            }
        };
        if with_props {
            // One property is enough: the property *object* is created on first
            // use regardless of how many keys it later holds, so the object
            // count tracks vertices-with-properties, not properties-per-vertex.
            if let Err(e) = g.set_vertex_prop(v, "k", PropValue::I64(i as i64)) {
                println!("GSTRESS PROPS CEILING: set_vertex_prop failed at vertex {i}: {e:?}");
                break;
            }
        }
        made += 1;
        if made % 500 == 0 {
            heartbeat("props", made, n, &t);
        }
    }
    let build_secs = t.elapsed().as_secs_f64();

    // Timed separately: on the arena layout writes live in mapped memory until
    // sync, so folding it into the build time would hide the deferred cost.
    let st = Instant::now();
    if let Err(e) = g.sync() {
        println!("GSTRESS PROPS: final sync failed: {e:?}");
    }
    let sync_secs = st.elapsed().as_secs_f64();

    println!(
        "GSTRESS PROPS {}: {} of {} vertices built in {:.1}s (+{:.2}s sync), {} arenas",
        if with_props { "with-props" } else { "control" },
        made,
        n,
        build_secs,
        sync_secs,
        g.arena_count()
    );
    if with_props {
        println!(
            "GSTRESS PROPS: {} property objects — one per vertex. Predicted ceiling {}.",
            made, PREDICTED_CEILING
        );
    }
    if made < n {
        println!(
            "GSTRESS PROPS CEILING: stopped at {} of {} — this is the measurement.",
            made, n
        );
    }
}
