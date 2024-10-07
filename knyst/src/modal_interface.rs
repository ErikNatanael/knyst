//! Knyst is used through a modal interface. A [`KnystSphere`] corresponds to a whole instance of Knyst. Spheres
//! are completely separated from each other. The current sphere and active graph within a sphere is set on a thread by thread basis
//! using thread locals. Most Knyst programs only need one sphere.
//!
//! Interaction with Knyst is done through the [`knyst`](crate) function which will return an object that implements [`KnystCommands`].
//! The implementation depends on the platform.
//!
//! The purpose of this architecture is to allow for a highly ergonomic and concise way of interacting with the graph(s),
//! as well as multiple underlying implementations suitable for different systems. A library targeting Knyst should
//! work the on any platform.
//!
//! Most of the methods of the [`KnystCommands`] trait aren't normally needed by users but, will be used internally by handles.
//! Using [`KnystCommands`] directly instead of using handles is discouraged.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{atomic::AtomicU16, Mutex};

use crate::audio_backend::AudioBackendError;
use crate::resources::{BufferId, WavetableId};

use crate::controller::KnystCommands;
use crate::graph::{GraphSettings, NodeId, Time};
use crate::handles::{GraphHandle, Handle};
use crate::prelude::{CallbackHandle, MultiThreadedKnystCommands};
use crate::sphere::KnystSphere;
use crate::wavetable_aa::Wavetable;

/// A unique id for a KnystSphere.
#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Debug, Hash)]
pub struct SphereId(u16);

/// Implements KnystCommands, but does nothing except reports a warning. This is done in order to
/// enable the infallible thread local state interface.
pub struct DummyKnystCommands;
impl DummyKnystCommands {
    fn report_dummy(&self) {
        eprintln!("No KnystCommands set, command ignored.")
    }
}
/// Represents and possible implementor of [`KnystCommands`]
pub enum UnifiedKnystCommands {
    #[allow(missing_docs)]
    Real(Rc<RefCell<SelectedKnystCommands>>),
    #[allow(missing_docs)]
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
                NodeId::new(u64::MAX)
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
                NodeId::new(u64::MAX)
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
                NodeId::new(u64::MAX)
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
                NodeId::new(u64::MAX)
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
                crate::prelude::Beats,
                &mut MultiThreadedKnystCommands,
            ) -> Option<crate::prelude::Beats>
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

    fn insert_buffer(&mut self, buffer: crate::buffer::Buffer) -> crate::resources::BufferId {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().insert_buffer(buffer),
            UnifiedKnystCommands::Dummy(kc) => {
                kc.report_dummy();
                BufferId::new(&buffer)
            }
        }
    }

    fn remove_buffer(&mut self, buffer_id: crate::resources::BufferId) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().remove_buffer(buffer_id),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn replace_buffer(
        &mut self,
        buffer_id: crate::resources::BufferId,
        buffer: crate::buffer::Buffer,
    ) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().replace_buffer(buffer_id, buffer),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn insert_wavetable(&mut self, wavetable: Wavetable) -> crate::resources::WavetableId {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().insert_wavetable(wavetable),
            UnifiedKnystCommands::Dummy(kc) => {
                kc.report_dummy();
                WavetableId::new()
            }
        }
    }

    fn remove_wavetable(&mut self, wavetable_id: crate::resources::WavetableId) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().remove_wavetable(wavetable_id),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn replace_wavetable(&mut self, id: crate::resources::WavetableId, wavetable: Wavetable) {
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

    fn upload_local_graph(&mut self) -> Option<Handle<GraphHandle>> {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().upload_local_graph(),
            UnifiedKnystCommands::Dummy(kc) => {
                kc.report_dummy();
                None
            }
        }
    }

    fn request_inspection(
        &mut self,
    ) -> std::sync::mpsc::Receiver<crate::inspection::GraphInspection> {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().request_inspection(),
            UnifiedKnystCommands::Dummy(kc) => {
                kc.report_dummy();
                std::sync::mpsc::sync_channel(0).1
            }
        }
    }

    fn to_graph(&mut self, graph_id: crate::graph::GraphId) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().to_graph(graph_id),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn to_top_level_graph(&mut self) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().to_top_level_graph(),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn start_scheduling_bundle(&mut self, time: Time) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().start_scheduling_bundle(time),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn upload_scheduling_bundle(&mut self) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().upload_scheduling_bundle(),
            UnifiedKnystCommands::Dummy(kc) => kc.report_dummy(),
        }
    }

    fn current_graph(&self) -> crate::graph::GraphId {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().current_graph(),
            UnifiedKnystCommands::Dummy(kc) => {
                kc.report_dummy();
                u64::MAX
            }
        }
    }

    fn set_mortality(&mut self, node: NodeId, is_mortal: bool) {
        match self {
            UnifiedKnystCommands::Real(kc) => kc.borrow_mut().set_mortality(node, is_mortal),
            UnifiedKnystCommands::Dummy(kc) => {
                kc.report_dummy();
            }
        }
    }
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

pub(crate) fn register_sphere(sphere: KnystSphere) -> Result<SphereId, SphereError> {
    let mut spheres = match ALL_KNYST_SPHERES.lock() {
        Ok(s) => s,
        Err(poison) => poison.into_inner(),
    };
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
        Err(SphereError::NoMoreSphereIds)
    }
}
pub(crate) fn remove_sphere(sphere_id: SphereId) -> Result<(), SphereError> {
    let mut spheres = match ALL_KNYST_SPHERES.lock() {
        Ok(s) => s,
        Err(poison) => poison.into_inner(),
    };
    if let Some(index) = spheres.iter().position(|(_, id)| *id == sphere_id) {
        let (_old_sphere, _) = spheres.remove(index);
        if ACTIVE_KNYST_SPHERE.with(|aks| *aks.borrow_mut()) == sphere_id {
            if let Some((_, new_active)) = spheres.first() {
                let new_active = *new_active;
                drop(spheres);
                set_active_sphere(new_active)?;
            } else {
                ACTIVE_KNYST_SPHERE_COMMANDS.with(|aksc| {
                    *aksc.borrow_mut() = None;
                });
            }
        }
        Ok(())
    } else {
        Err(SphereError::SphereNotFound)
    }
}

fn get_sphere_commands(sphere_id: SphereId) -> Result<MultiThreadedKnystCommands, SphereError> {
    // If one thread panics while holding a lock to the spheres (which is highly unlikely) it should be fine to just go on accessing the spheres anyway.
    let spheres = match ALL_KNYST_SPHERES.lock() {
        Ok(spheres) => spheres,
        Err(poison_lock) => poison_lock.into_inner(),
    };
    if let Some((sphere, _id)) = spheres.iter().find(|(_s, id)| *id == sphere_id) {
        Ok(sphere.commands())
    } else {
        Err(SphereError::SphereNotFound)
    }
}

/// Set the selected sphere to be active on this thread.
pub fn set_active_sphere(id: SphereId) -> Result<(), SphereError> {
    ACTIVE_KNYST_SPHERE.with(|aks| *aks.borrow_mut() = id);
    ACTIVE_KNYST_SPHERE_COMMANDS.with(|aksc| {
        let kc = get_sphere_commands(id)?;
        let kc = Some(Rc::new(RefCell::new(kc)));
        *aksc.borrow_mut() = kc;
        Ok(())
    })
}

// Return impl KnystCommands to avoid committing to a return type and being able to change the return type through conditional compilation for different platforms
/// Returns an implementor of [`KnystCommands`] which allows interacting with Knyst
pub fn knyst_commands() -> impl KnystCommands {
    if let Some(kc) = ACTIVE_KNYST_SPHERE_COMMANDS.with(|aksc| aksc.borrow().clone()) {
        UnifiedKnystCommands::Real(kc)
    } else {
        // It could be the first time commands is called from this thread in which case we should try to set the default sphere
        let default_sphere = DEFAULT_KNYST_SPHERE.load(std::sync::atomic::Ordering::SeqCst);
        // TODO: report an error if this fails
        set_active_sphere(SphereId(default_sphere)).ok();
        if let Some(kc) = ACTIVE_KNYST_SPHERE_COMMANDS.with(|aksc| aksc.borrow().clone()) {
            UnifiedKnystCommands::Real(kc)
        } else {
            UnifiedKnystCommands::Dummy(DummyKnystCommands)
        }
    }
}

/// Error type for errors having to do with the modal commands system
#[derive(thiserror::Error, Debug)]
pub enum SphereError {
    /// The requested KnystSphere could not be found.
    #[error("The requested KnystSphere could not be found.")]
    SphereNotFound,
    /// You are out of sphere ids so you are unable to register a new sphere. I cannot think of a workload that requires this. If you are sure you need this many spheres, please file a bug report.
    #[error("There are no sphere ids to register a new sphere. If you are sure you need this many spheres, please file a bug report.")]
    NoMoreSphereIds,
    /// There was an error in the audio backend
    #[error("Audio backend error: {0}")]
    AudioBackendError(#[from] AudioBackendError),
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
