use std::cell::RefCell;
use std::hint::black_box;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::{atomic::AtomicU16, Mutex};
use std::time::Instant;

use knyst_core::resources::{BufferId, WavetableId};

use crate::controller::KnystCommands;
use crate::graph::{Graph, GraphSettings, NodeId};
use crate::handles::{GraphHandle, Handle};
use crate::prelude::{CallbackHandle, MultiThreadedKnystCommands};
use crate::sphere::KnystSphere;

/// A unique id for a KnystSphere.
#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Debug, Hash)]
pub struct SphereId(u16);

// #[derive(Clone, Debug)]
// pub struct MultiThreadedKnystCommands(Arc<String>);
// #[derive(Clone, Debug)]
// pub struct SingleThreadedKnystCommands;
// #[derive(Clone, Debug)]
pub struct DummyKnystCommands;
impl DummyKnystCommands {
    fn report_dummy(&self) {
        eprintln!("No KnystCommands set, command ignored.")
    }
}
pub enum UnifiedKnystCommands {
    Real(Rc<RefCell<SelectedKnystCommands>>),
    Dummy(DummyKnystCommands),
}
type SelectedKnystCommands = MultiThreadedKnystCommands;
impl KnystCommands for UnifiedKnystCommands {
    fn push_without_inputs(
        &mut self,
        gen_or_graph: impl crate::graph::GenOrGraph,
    ) -> crate::graph::NodeId {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().push_without_inputs(gen_or_graph),
            UnifiedKnystCommands::Dummy(kc) => {
                kc.report_dummy();
                NodeId::new()
            }
        }
    }

    fn push(
        &mut self,
        gen_or_graph: impl crate::graph::GenOrGraph,
        inputs: impl Into<crate::graph::connection::InputBundle>,
    ) -> crate::graph::NodeId {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().push(gen_or_graph, inputs),
            UnifiedKnystCommands::Dummy(kc) => {
                kc.report_dummy();
                NodeId::new()
            }
        }
    }

    fn push_to_graph_without_inputs(
        &mut self,
        gen_or_graph: impl crate::graph::GenOrGraph,
        graph_id: crate::graph::GraphId,
    ) -> crate::graph::NodeId {
        match self {
            UnifiedKnystCommands::Real(kc) => kc
                .borrow_mut()
                .push_to_graph_without_inputs(gen_or_graph, graph_id),
            UnifiedKnystCommands::Dummy(kc) => {
                kc.report_dummy();
                NodeId::new()
            }
        }
    }

    fn push_to_graph(
        &mut self,
        gen_or_graph: impl crate::graph::GenOrGraph,
        graph_id: crate::graph::GraphId,
        inputs: impl Into<crate::graph::connection::InputBundle>,
    ) -> crate::graph::NodeId {
        match self {
            UnifiedKnystCommands::Real(kc) => {
                kc.borrow_mut()
                    .push_to_graph(gen_or_graph, graph_id, inputs)
            }
            UnifiedKnystCommands::Dummy(kc) => {
                kc.report_dummy();
                NodeId::new()
            }
        }
    }

    fn connect(&mut self, connection: crate::graph::Connection) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().connect(connection),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn connect_bundle(&mut self, bundle: impl Into<crate::graph::connection::ConnectionBundle>) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().connect_bundle(bundle),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn schedule_beat_callback(
        &mut self,
        callback: impl FnMut(
                crate::prelude::Superbeats,
                &mut MultiThreadedKnystCommands,
            ) -> Option<crate::prelude::Superbeats>
            + Send
            + 'static,
        start_time: crate::controller::StartBeat,
    ) -> crate::prelude::CallbackHandle {
        match self {
            UnifiedKnystCommands::Real(kc) => {
                kc.borrow_mut().schedule_beat_callback(callback, start_time)
            }
            UnifiedKnystCommands::Dummy(kc) => {
                kc.report_dummy();
                CallbackHandle::dummy_new()
            }
        }
    }

    fn disconnect(&mut self, connection: crate::graph::Connection) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().disconnect(connection),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn free_disconnected_nodes(&mut self) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().free_disconnected_nodes(),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn free_node_mend_connections(&mut self, node: crate::graph::NodeId) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().free_node_mend_connections(node),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn free_node(&mut self, node: crate::graph::NodeId) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().free_node(node),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn schedule_change(&mut self, change: crate::graph::ParameterChange) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().schedule_change(change),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn schedule_changes(&mut self, changes: crate::graph::SimultaneousChanges) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().schedule_changes(changes),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn insert_buffer(
        &mut self,
        buffer: knyst_core::buffer::Buffer,
    ) -> knyst_core::resources::BufferId {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().insert_buffer(buffer),
            UnifiedKnystCommands::Dummy(kc) => {
                kc.report_dummy();
                BufferId::new(&buffer)
            }
        }
    }

    fn remove_buffer(&mut self, buffer_id: knyst_core::resources::BufferId) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().remove_buffer(buffer_id),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn replace_buffer(
        &mut self,
        buffer_id: knyst_core::resources::BufferId,
        buffer: knyst_core::buffer::Buffer,
    ) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().replace_buffer(buffer_id, buffer),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn insert_wavetable(
        &mut self,
        wavetable: knyst_core::wavetable::Wavetable,
    ) -> knyst_core::resources::WavetableId {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().insert_wavetable(wavetable),
            UnifiedKnystCommands::Dummy(kc) => {
                kc.report_dummy();
                WavetableId::new()
            }
        }
    }

    fn remove_wavetable(&mut self, wavetable_id: knyst_core::resources::WavetableId) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().remove_wavetable(wavetable_id),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn replace_wavetable(
        &mut self,
        id: knyst_core::resources::WavetableId,
        wavetable: knyst_core::wavetable::Wavetable,
    ) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().replace_wavetable(id, wavetable),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn change_musical_time_map(
        &mut self,
        change_fn: impl FnOnce(&mut crate::scheduling::MusicalTimeMap) + Send + 'static,
    ) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().change_musical_time_map(change_fn),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn default_graph_settings(&self) -> crate::graph::GraphSettings {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().default_graph_settings(),
            UnifiedKnystCommands::Dummy(kc) => {
                kc.report_dummy();
                GraphSettings::default()
            }
        }
    }

    fn init_local_graph(&mut self, settings: GraphSettings) -> crate::graph::GraphId {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().init_local_graph(settings),
            UnifiedKnystCommands::Dummy(kc) => {
                kc.report_dummy();
                0
            }
        }
    }

    fn upload_local_graph(&mut self) -> Handle<GraphHandle> {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().upload_local_graph(),
            UnifiedKnystCommands::Dummy(kc) => {
                kc.report_dummy();
                Handle::new(GraphHandle::new(NodeId::new(), 0, 0))
            }
        }
    }

    // fn to_graph(&self, graph_id: crate::graph::GraphId) -> Self {
    //     match self {
    //         UnifiedKnystCommands::Real(kc) => kc.borrow_mut().to_graph(),
    //         UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
    //     }
    // }

    // fn to_top_level_graph(&self) -> Self {
    //     match self {
    //         UnifiedKnystCommands::Real(kc) => kc.borrow_mut().to_top_level_graph(),
    //         UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
    //     }
    // }
    // fn push(&mut self) {
    //     match self {
    //         UnifiedKnystCommands::Real(kc) => kc.borrow_mut().push(),
    //         UnifiedKnystCommands::Dummy(kc) => kc.push(),
    //     }
    // }
}

static DEFAULT_KNYST_SPHERE: AtomicU16 = AtomicU16::new(0);
static ALL_KNYST_SPHERES: Mutex<Vec<(KnystSphere, SphereId)>> = Mutex::new(vec![]);

thread_local! {
    static ACTIVE_KNYST_SPHERE: RefCell<SphereId> = RefCell::new(SphereId(0));
    // The inner Rc<Refcell<>> cuts execution time to 1/3
    static ACTIVE_KNYST_SPHERE_COMMANDS: RefCell<Option<Rc<RefCell<SelectedKnystCommands>>>> = RefCell::new(None);
}

pub(crate) fn register_sphere(sphere: KnystSphere) -> Result<SphereId, Box<dyn std::error::Error>> {
    let mut spheres = ALL_KNYST_SPHERES.lock()?;
    // Get first unused sphereid
    let mut new_id = None;
    for i in 0..u16::MAX {
        let mut id_found = false;
        for (_, id) in (*spheres).iter() {
            if id.0 == i {
                id_found = true;
                break;
            }
        }
        if !id_found {
            new_id = Some(SphereId(i));
            break;
        }
    }
    if let Some(id) = new_id {
        spheres.push((sphere, id));
        Ok(id)
    } else {
        Err("No available SphereId found".into())
    }
}

fn get_sphere_commands(
    sphere_id: SphereId,
) -> Result<MultiThreadedKnystCommands, Box<dyn std::error::Error>> {
    let spheres = ALL_KNYST_SPHERES.lock()?;
    if let Some((sphere, id)) = spheres.iter().find(|(_s, id)| *id == sphere_id) {
        Ok(sphere.commands())
    } else {
        Err("KnystCommands not found".into())
    }
}

pub fn set_active_sphere(id: SphereId) -> Result<(), Box<dyn std::error::Error>> {
    ACTIVE_KNYST_SPHERE.with(|aks| *aks.borrow_mut() = id);
    ACTIVE_KNYST_SPHERE_COMMANDS.with(|aksc| {
        let kc = get_sphere_commands(id)?;
        let kc = Some(Rc::new(RefCell::new(kc)));
        *aksc.borrow_mut() = kc;
        Ok(())
    })
}

// Return impl KnystCommands to avoid committing to a return type and being able to change the return type through conditional compilation for different platforms
pub fn commands() -> impl KnystCommands {
    if let Some(kc) = ACTIVE_KNYST_SPHERE_COMMANDS.with(|aksc| aksc.borrow().clone()) {
        UnifiedKnystCommands::Real(kc)
    } else {
        // It could be the first time commands is called from this thread in which case we should try to set the default sphere
        let default_sphere = DEFAULT_KNYST_SPHERE.load(std::sync::atomic::Ordering::SeqCst);
        set_active_sphere(SphereId(default_sphere));
        if let Some(kc) = ACTIVE_KNYST_SPHERE_COMMANDS.with(|aksc| aksc.borrow().clone()) {
            UnifiedKnystCommands::Real(kc)
        } else {
            UnifiedKnystCommands::Dummy(DummyKnystCommands)
        }
    }
}

// pub fn test_using() {
//     let sphere = KnystSphere::new("First".to_owned());
//     let _id = register_sphere(sphere).unwrap();
//     // println!("Started sphere {_id:?}");
//     commands().push();
//     commands().push();
//     let sphere = KnystSphere::new("Second".to_owned());
//     let id2 = register_sphere(sphere).unwrap();
//     // println!("Started sphere {id2:?}");
//     commands().push();
//     set_sphere(id2);
//     commands().push();

//     let start = Instant::now();
//     for _ in 0..10000 {
//         commands().push();
//     }
//     let time_taken = start.elapsed();
//     println!("10000 commands took {:?}", time_taken);
// }
