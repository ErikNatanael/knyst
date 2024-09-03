//! API for interacting with a running top level [`Graph`] from any number of
//! threads without having to manually keep track of running [`Graph::update`]
//! regularly.
//!
//! [`KnystCommands`] gives you a convenient API for sending messages to the
//! [`Controller`]. The API is similar to calling methods on [`Graph`] directly,
//! but also includes modifying [`Resources`].

#[allow(unused)]
use crate::resources::Resources;
use std::{
    cell::RefCell,
    sync::{atomic::AtomicBool, Arc},
    time::{Duration, Instant},
};

use crate::{
    buffer::Buffer,
    graph::{NodeChanges, ScheduleError, Time},
    inspection::GraphInspection,
    knyst_commands,
    resources::{BufferId, ResourcesCommand, ResourcesResponse, WavetableId},
    wavetable_aa::Wavetable,
};
use crate::{
    graph::{
        connection::{ConnectionBundle, ConnectionError, InputBundle},
        Connection, FreeError, GenOrGraph, GenOrGraphEnum, Graph, GraphId, GraphSettings, NodeId,
        ParameterChange, SimultaneousChanges,
    },
    handles::{GraphHandle, Handle},
    inputs,
    scheduling::MusicalTimeMap,
    time::Beats,
    KnystError,
};
use crossbeam_channel::{unbounded, Receiver, Sender};

/// Encodes commands sent from a [`KnystCommands`]
enum Command {
    Push {
        gen_or_graph: GenOrGraphEnum,
        node_address: NodeId,
        graph_id: GraphId,
        start_time: Time,
    },
    Connect(Connection),
    Disconnect(Connection),
    SetMortality {
        node: NodeId,
        is_mortal: bool,
    },
    FreeNode(NodeId),
    FreeNodeMendConnections(NodeId),
    ScheduleChange(ParameterChange),
    ScheduleChanges(SimultaneousChanges),
    FreeDisconnectedNodes,
    ResourcesCommand(ResourcesCommand),
    ChangeMusicalTimeMap(Box<dyn FnOnce(&mut MusicalTimeMap) + Send>),
    ScheduleBeatCallback(BeatCallback, StartBeat),
    RequestInspection(std::sync::mpsc::SyncSender<GraphInspection>),
}
impl std::fmt::Debug for Command {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Push {
                gen_or_graph,
                node_address,
                graph_id,
                start_time,
            } => f
                .debug_struct("Push")
                .field("gen_or_graph", gen_or_graph)
                .field("node_address", node_address)
                .field("graph_id", graph_id)
                .field("start_time", start_time)
                .finish(),
            Self::Connect(arg0) => f.debug_tuple("Connect").field(arg0).finish(),
            Self::Disconnect(arg0) => f.debug_tuple("Disconnect").field(arg0).finish(),
            Self::FreeNode(arg0) => f.debug_tuple("FreeNode").field(arg0).finish(),
            Self::FreeNodeMendConnections(arg0) => f
                .debug_tuple("FreeNodeMendConnections")
                .field(arg0)
                .finish(),
            Self::ScheduleChange(arg0) => f.debug_tuple("ScheduleChange").field(arg0).finish(),
            Self::ScheduleChanges(arg0) => f.debug_tuple("ScheduleChanges").field(arg0).finish(),
            Self::FreeDisconnectedNodes => write!(f, "FreeDisconnectedNodes"),
            Self::ResourcesCommand(_arg0) => f.debug_tuple("ResourcesCommand").finish(),
            Self::ChangeMusicalTimeMap(_arg0) => f.debug_tuple("ChangeMusicalTimeMap").finish(),
            Self::ScheduleBeatCallback(_arg0, _arg1) => {
                f.debug_tuple("ScheduleBeatCallback").finish()
            }
            Self::RequestInspection(arg0) => {
                f.debug_tuple("RequestInspection").field(arg0).finish()
            }
            Command::SetMortality { node, is_mortal } => f
                .debug_tuple("SetMortality")
                .field(node)
                .field(is_mortal)
                .finish(),
        }
    }
}

/// [`KnystCommands`] sends commands to the [`Controller`] which should hold the
/// top level [`Graph`]. The API is as close as possible to that of an owned
/// [`Graph`].
///
/// This can safely be cloned and sent to a different thread for use.
///
// TODO: What's the best way of referring to a graph? GraphId is unique, but not
// always the handiest. It would be nice to be able to choose to refer to Graphs
// by an identifier e.g. name. In Bevy holding on to GraphIds is easy.
pub trait KnystCommands {
    /// Push a Gen or Graph to the top level Graph without specifying any inputs.
    fn push_without_inputs(&mut self, gen_or_graph: impl GenOrGraph) -> NodeId;
    /// Push a Gen or Graph to the default Graph.
    fn push(&mut self, gen_or_graph: impl GenOrGraph, inputs: impl Into<InputBundle>) -> NodeId;
    /// Push a Gen or Graph to the Graph with the specified id without specifying inputs.
    fn push_to_graph_without_inputs(
        &mut self,
        gen_or_graph: impl GenOrGraph,
        graph_id: GraphId,
    ) -> NodeId;
    /// Push a Gen or Graph to the Graph with the specified id.
    fn push_to_graph(
        &mut self,
        gen_or_graph: impl GenOrGraph,
        graph_id: GraphId,
        inputs: impl Into<InputBundle>,
    ) -> NodeId;
    /// Create a new connections
    fn connect(&mut self, connection: Connection);
    /// Make several connections at once using any of the ConnectionBundle
    /// notations
    fn connect_bundle(&mut self, bundle: impl Into<ConnectionBundle>);
    /// Add a new beat callback. See [`BeatCallback`] for documentation.
    fn schedule_beat_callback(
        &mut self,
        callback: impl FnMut(Beats, &mut MultiThreadedKnystCommands) -> Option<Beats> + Send + 'static,
        start_time: StartBeat,
    ) -> CallbackHandle;
    /// Disconnect (undo) a [`Connection`]
    fn disconnect(&mut self, connection: Connection);
    /// Sets the mortality of a node to mortal (true) or immortal (false). An immortal node cannot be freed.
    fn set_mortality(&mut self, node: NodeId, is_mortal: bool);
    /// Free any nodes that are not currently connected to the graph's outputs
    /// via any chain of connections.
    fn free_disconnected_nodes(&mut self);
    /// Free a node and try to mend connections between the inputs and the
    /// outputs of the node.
    fn free_node_mend_connections(&mut self, node: NodeId);
    /// Free a node.
    fn free_node(&mut self, node: NodeId);
    /// Schedule a change to be made.
    ///
    /// NB: Changes are buffered and the scheduler needs to be regularly updated
    /// for them to be sent to the audio thread. If you are getting your
    /// [`KnystCommands`] through `AudioBackend::start_processing` this is taken
    /// care of automatically.
    fn schedule_change(&mut self, change: ParameterChange);
    /// Schedule multiple changes to be made.
    ///
    /// NB: Changes are buffered and the scheduler needs to be regularly updated
    /// for them to be sent to the audio thread. If you are getting your
    /// [`KnystCommands`] through `AudioBackend::start_processing` this is taken
    /// care of automatically.
    fn schedule_changes(&mut self, changes: SimultaneousChanges);
    /// Inserts a new buffer in the [`Resources`] and returns an id which can be
    /// converted to a key on the audio thread with access to a [`Resources`].
    fn insert_buffer(&mut self, buffer: Buffer) -> BufferId;
    /// Remove a buffer from the [`Resources`]
    fn remove_buffer(&mut self, buffer_id: BufferId);
    /// Replace a buffer in the [`Resources`]
    fn replace_buffer(&mut self, buffer_id: BufferId, buffer: Buffer);
    /// Inserts a new wavetable in the [`Resources`] and returns an id which can be
    /// converted to a key on the audio thread with access to a [`Resources`].
    fn insert_wavetable(&mut self, wavetable: Wavetable) -> WavetableId;
    /// Remove a wavetable from the [`Resources`]
    fn remove_wavetable(&mut self, wavetable_id: WavetableId);
    /// Replace a wavetable in the [`Resources`]
    fn replace_wavetable(&mut self, id: WavetableId, wavetable: Wavetable);
    /// Make a change to the shared [`MusicalTimeMap`]
    fn change_musical_time_map(
        &mut self,
        change_fn: impl FnOnce(&mut MusicalTimeMap) + Send + 'static,
    );
    /// Request a [`GraphInspection`] of the top level graph which will be sent back in the returned channel
    fn request_inspection(&mut self) -> std::sync::mpsc::Receiver<GraphInspection>;

    /// Return the [`GraphSettings`] of the top level graph. This means you
    /// don't have to manually keep track of matching sample rate and block size
    /// for example.
    fn default_graph_settings(&self) -> GraphSettings;
    /// Set knyst commands on the current thread to use the selected GraphId by default
    fn to_graph(&mut self, graph_id: GraphId);
    /// Set knyst commands on the current thread to use the top level GraphId by default
    fn to_top_level_graph(&mut self);
    /// Get the id of the currently active graph
    fn current_graph(&self) -> GraphId;
    /// Creates a new local graph and sets it as the default graph
    fn init_local_graph(&mut self, settings: GraphSettings) -> GraphId;
    /// Upload the local graph to the previously default graph and restore the default graph to that previous default graph.
    fn upload_local_graph(&mut self)
        -> Option<crate::handles::Handle<crate::handles::GraphHandle>>;
    /// Start a scheduling bundle, meaning any change scheduled will not be applied until [`KnystCommands::upload_scheduling_bundle`] is called. Prefer using [`schedule_bundle`] as it is more difficult to misuse.
    fn start_scheduling_bundle(&mut self, time: Time);
    /// Uploads scheduled changes to the graph and schedules them for the time specified in [`KnystCommands::start_scheduling_bundle`]. Prefer [`schedule_bundle`] to help reinforce scoping and potential thread switches.
    fn upload_scheduling_bundle(&mut self);
}

/// Create a new local graph, runs the init function to let you build it, and then uploads it to the active Sphere.
pub fn upload_graph(
    settings: GraphSettings,
    init: impl FnOnce(),
) -> crate::handles::Handle<crate::handles::GraphHandle> {
    knyst_commands().init_local_graph(settings);
    init();
    knyst_commands()
        .upload_local_graph()
        .expect("`upload_graph` initiated a local graph so it should exist")
}

/// Schedules any changes made in the closure at the given time. Currently limited to changes of constant values and spawning new nodes, not new connections.
pub fn schedule_bundle(time: Time, c: impl FnOnce()) {
    knyst_commands().start_scheduling_bundle(time);
    c();
    knyst_commands().upload_scheduling_bundle();
}

#[derive(Clone)]
/// Multi threaded implementation on KnystCommands, default
pub struct MultiThreadedKnystCommands {
    /// Sends Commands to the Controller.
    sender: crossbeam_channel::Sender<Command>,
    /// As pushing to the top level Graph is the default we store the GraphId to that Graph.
    top_level_graph_id: GraphId,
    /// Make the top level graph settings available so that creating a matching sub graph is easy.
    top_level_graph_settings: GraphSettings,
    /// The default graph to push new nodes to
    selected_graph_remote_graph: GraphId,
    /// If changes should be bundled
    bundle_changes: bool,
    /// The vec holding changes to be later scheduled as a bundle
    changes_bundle: Vec<NodeChanges>,
    changes_bundle_time: Time,
}

impl KnystCommands for MultiThreadedKnystCommands {
    /// Push a Gen or Graph to the top level Graph without specifying any inputs.
    fn push_without_inputs(&mut self, gen_or_graph: impl GenOrGraph) -> NodeId {
        self.push(gen_or_graph, inputs![])
    }
    /// Push a Gen or Graph to the default Graph.
    fn push(&mut self, gen_or_graph: impl GenOrGraph, inputs: impl Into<InputBundle>) -> NodeId {
        let node_id = {
            let local_node_id = LOCAL_GRAPH.with_borrow_mut(|g| {
                if let Some(g) = g.last_mut() {
                    let mut node_id = NodeId::new(g.id());
                    g.push_with_existing_address_at_time(
                        gen_or_graph,
                        &mut node_id,
                        self.changes_bundle_time,
                    );
                    Ok(node_id)
                } else {
                    Err(gen_or_graph)
                }
            });
            match local_node_id {
                Ok(node_id) => node_id,
                Err(gen_or_graph) => self
                    .push_to_graph_without_inputs(gen_or_graph, self.selected_graph_remote_graph),
            }
        };
        // Connect any inputs
        let inputs: InputBundle = inputs.into();
        self.connect_bundle(inputs.to(node_id));
        node_id
    }
    /// Push a Gen or Graph to the Graph with the specified id without specifying inputs.
    fn push_to_graph_without_inputs(
        &mut self,
        gen_or_graph: impl GenOrGraph,
        graph_id: GraphId,
    ) -> NodeId {
        let gen_or_graph = gen_or_graph.into_gen_or_graph_enum();
        let found_in_local = LOCAL_GRAPH.with_borrow_mut(|g| {
            if let Some(g) = g.last_mut() {
                if g.id() == graph_id {
                    let mut node_id = NodeId::new(graph_id);
                    if let Err(e) =
                        g.push_with_existing_address_to_graph(gen_or_graph, &mut node_id, g.id())
                    {
                        // TODO: report error
                        // TODO: recover the gen_or_graph from the PushError
                        eprintln!("{e:?}");
                    }
                    Ok(node_id)
                } else {
                    eprintln!("Local graph does not match requested graph");
                    Err(gen_or_graph)
                }
            } else {
                // There is no local graph
                Err(gen_or_graph)
            }
        });
        match found_in_local {
            Ok(node_id) => node_id,
            Err(gen_or_graph) => {
                let new_node_address = NodeId::new(graph_id);
                let command = Command::Push {
                    gen_or_graph,
                    node_address: new_node_address,
                    graph_id,
                    start_time: self.changes_bundle_time,
                };
                self.sender.send(command).unwrap();
                new_node_address
            }
        }
    }
    /// Push a Gen or Graph to the Graph with the specified id.
    fn push_to_graph(
        &mut self,
        gen_or_graph: impl GenOrGraph,
        graph_id: GraphId,
        inputs: impl Into<InputBundle>,
    ) -> NodeId {
        let new_node_address = self.push_to_graph_without_inputs(gen_or_graph, graph_id);
        let inputs: InputBundle = inputs.into();
        self.connect_bundle(inputs.to(new_node_address));
        new_node_address
    }
    /// Create a new connections
    fn connect(&mut self, connection: Connection) {
        // The connection may be in our local graph or remotely. Check local first.
        let found_in_local = LOCAL_GRAPH.with_borrow_mut(|g| {
            if let Some(g) = g.last_mut() {
                match g.connect(connection.clone()) {
                    Ok(()) => true,
                    Err(e) => match e {
                        ConnectionError::GraphNotFound(_) => false,
                        _ => {
                            // TODO: Report this error
                            eprintln!("Error: {e:?}");
                            // We found the correct graph, but there was a different error
                            true
                        }
                    },
                }
            } else {
                false
            }
        });
        if !found_in_local {
            self.sender.send(Command::Connect(connection)).unwrap();
        }
    }
    /// Make several connections at once using any of the ConnectionBundle
    /// notations
    fn connect_bundle(&mut self, bundle: impl Into<ConnectionBundle>) {
        let bundle = bundle.into();
        for c in bundle.as_connections() {
            self.connect(c);
        }
    }
    /// Add a new beat callback. See [`BeatCallback`] for documentation.
    fn schedule_beat_callback(
        &mut self,
        callback: impl FnMut(Beats, &mut MultiThreadedKnystCommands) -> Option<Beats> + Send + 'static,
        start_time: StartBeat,
    ) -> CallbackHandle {
        let c = BeatCallback::new(callback, Beats::ZERO);
        let handle = c.handle();
        let command = Command::ScheduleBeatCallback(c, start_time);
        self.sender.send(command).unwrap();
        handle
    }
    /// Disconnect (undo) a [`Connection`]
    fn disconnect(&mut self, connection: Connection) {
        // The connection may be in our local graph or remotely. Check local first.
        let found_in_local = LOCAL_GRAPH.with_borrow_mut(|g| {
            if let Some(g) = g.last_mut() {
                match g.disconnect(connection.clone()) {
                    Ok(()) => true,
                    Err(e) => match e {
                        ConnectionError::GraphNotFound(_) => false,
                        _ => {
                            // TODO: Report this error
                            eprintln!("Error: {e:?}");
                            // We found the correct graph, but there was a different error
                            true
                        }
                    },
                }
            } else {
                false
            }
        });
        if !found_in_local {
            self.sender.send(Command::Disconnect(connection)).unwrap();
        }
    }
    /// Free any nodes that are not currently connected to the graph's outputs
    /// via any chain of connections.
    fn free_disconnected_nodes(&mut self) {
        self.sender.send(Command::FreeDisconnectedNodes).unwrap();
    }
    /// Free a node and try to mend connections between the inputs and the
    /// outputs of the node.
    fn free_node_mend_connections(&mut self, node: NodeId) {
        self.sender
            .send(Command::FreeNodeMendConnections(node))
            .unwrap();
    }
    /// Free a node.
    fn free_node(&mut self, node: NodeId) {
        self.sender.send(Command::FreeNode(node)).unwrap();
    }
    /// Schedule a change to be made.
    ///
    /// NB: Changes are buffered and the scheduler needs to be regularly updated
    /// for them to be sent to the audio thread. If you are getting your
    /// [`KnystCommands`] through `AudioBackend::start_processing` this is taken
    /// care of automatically.
    fn schedule_change(&mut self, change: ParameterChange) {
        if self.bundle_changes {
            let change = NodeChanges {
                node: change.input.node,
                parameters: vec![(change.input.channel, change.value)],
                offset: None,
            };
            self.changes_bundle.push(change);
        } else {
            LOCAL_GRAPH.with_borrow_mut(|g| {
                if let Some(g) = g.last_mut() {
                    if let Err(e) = g.schedule_change(change) {
                        // TODO: report error
                        // TODO: recover the gen_or_graph from the PushError
                        eprintln!("{e:?}");
                    }
                } else {
                    // There is no local graph
                    self.sender.send(Command::ScheduleChange(change)).unwrap();
                }
            });
        }
    }
    /// Schedule multiple changes to be made.
    ///
    /// NB: Changes are buffered and the scheduler needs to be regularly updated
    /// for them to be sent to the audio thread. If you are getting your
    /// [`KnystCommands`] through `AudioBackend::start_processing` this is taken
    /// care of automatically.
    fn schedule_changes(&mut self, changes: SimultaneousChanges) {
        if self.bundle_changes {
            self.changes_bundle.extend(changes.changes);
        } else {
            let mut all_node_graphs = vec![];
            let time = changes.time;
            for c in &changes.changes {
                if !all_node_graphs.contains(&c.node.graph_id()) {
                    all_node_graphs.push(c.node.graph_id());
                }
            }
            let change_bundles_per_graph = if all_node_graphs.len() < 2 {
                vec![changes.changes]
            } else {
                let mut per_graph = vec![vec![]; all_node_graphs.len()];
                for change in changes.changes {
                    let i = all_node_graphs
                        .iter()
                        .position(|graph| *graph == change.node.graph_id())
                        .unwrap();
                    per_graph[i].push(change);
                }
                per_graph
            };
            for changes in change_bundles_per_graph {
                LOCAL_GRAPH.with_borrow_mut(|g| {
                    if let Some(g) = g.last_mut() {
                        if let Err(e) = g.schedule_changes(changes, time) {
                            // TODO: report error
                            // TODO: recover the gen_or_graph from the PushError
                            eprintln!("Local graph schedule_changes error: {e:?}");
                        }
                    } else {
                        // There is no local graph
                        self.sender
                            .send(Command::ScheduleChanges(SimultaneousChanges {
                                time,
                                changes,
                            }))
                            .unwrap();
                    }
                });
            }
        }
    }
    /// Inserts a new buffer in the [`Resources`] and returns an id which can be
    /// converted to a key on the audio thread with access to a [`Resources`].
    fn insert_buffer(&mut self, buffer: Buffer) -> BufferId {
        let id = BufferId::new(&buffer);
        self.sender
            .send(Command::ResourcesCommand(ResourcesCommand::InsertBuffer {
                id,
                buffer,
            }))
            .unwrap();
        id
    }
    /// Remove a buffer from the [`Resources`]
    fn remove_buffer(&mut self, buffer_id: BufferId) {
        self.sender
            .send(Command::ResourcesCommand(ResourcesCommand::RemoveBuffer {
                id: buffer_id,
            }))
            .unwrap();
    }
    /// Replace a buffer in the [`Resources`]
    fn replace_buffer(&mut self, buffer_id: BufferId, buffer: Buffer) {
        self.sender
            .send(Command::ResourcesCommand(ResourcesCommand::ReplaceBuffer {
                id: buffer_id,
                buffer,
            }))
            .unwrap();
    }
    /// Inserts a new wavetable in the [`Resources`] and returns an id which can be
    /// converted to a key on the audio thread with access to a [`Resources`].
    fn insert_wavetable(&mut self, wavetable: Wavetable) -> WavetableId {
        let id = WavetableId::new();
        self.sender
            .send(Command::ResourcesCommand(
                ResourcesCommand::InsertWavetable { id, wavetable },
            ))
            .unwrap();
        id
    }
    /// Remove a wavetable from the [`Resources`]
    fn remove_wavetable(&mut self, wavetable_id: WavetableId) {
        self.sender
            .send(Command::ResourcesCommand(
                ResourcesCommand::RemoveWavetable { id: wavetable_id },
            ))
            .unwrap();
    }
    /// Replace a wavetable in the [`Resources`]
    fn replace_wavetable(&mut self, id: WavetableId, wavetable: Wavetable) {
        self.sender
            .send(Command::ResourcesCommand(
                ResourcesCommand::ReplaceWavetable { id, wavetable },
            ))
            .unwrap();
    }
    /// Make a change to the shared [`MusicalTimeMap`]
    fn change_musical_time_map(
        &mut self,
        change_fn: impl FnOnce(&mut MusicalTimeMap) + Send + 'static,
    ) {
        self.sender
            .send(Command::ChangeMusicalTimeMap(Box::new(change_fn)))
            .unwrap();
    }
    /// Return the [`GraphSettings`] of the top level graph. This means you
    /// don't have to manually keep track of matching sample rate and block size
    /// for example.
    fn default_graph_settings(&self) -> GraphSettings {
        self.top_level_graph_settings.clone()
    }

    fn init_local_graph(&mut self, settings: GraphSettings) -> GraphId {
        let graph = Graph::new(settings);
        let graph_id = graph.id();
        LOCAL_GRAPH.with_borrow_mut(|g| g.push(graph));
        graph_id
    }

    fn upload_local_graph(&mut self) -> Option<Handle<GraphHandle>> {
        let graph_to_upload = LOCAL_GRAPH.with_borrow_mut(|g| g.pop());
        if let Some(g) = graph_to_upload {
            let num_inputs = g.num_inputs();
            let num_outputs = g.num_outputs();
            let graph_id = g.id();

            let id = self.push_without_inputs(g);
            Some(Handle::new(GraphHandle::new(
                id,
                graph_id,
                num_inputs,
                num_outputs,
            )))
        } else {
            None
        }
    }

    fn request_inspection(&mut self) -> std::sync::mpsc::Receiver<GraphInspection> {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        self.sender
            .send(Command::RequestInspection(sender))
            .unwrap();
        receiver
    }

    fn to_graph(&mut self, graph_id: GraphId) {
        self.selected_graph_remote_graph = graph_id;
    }

    fn to_top_level_graph(&mut self) {
        self.selected_graph_remote_graph = self.top_level_graph_id;
    }

    fn start_scheduling_bundle(&mut self, time: Time) {
        self.bundle_changes = true;
        self.changes_bundle_time = time;
        if !self.changes_bundle.is_empty() {
            eprintln!(
                "Warning: Starting a new scheduling bundle before the previous one was scheduled."
            )
        }
    }

    fn upload_scheduling_bundle(&mut self) {
        self.bundle_changes = false;
        let changes = SimultaneousChanges {
            time: self.changes_bundle_time,
            changes: self.changes_bundle.clone(),
        };
        self.schedule_changes(changes);
        self.changes_bundle.clear();
        self.changes_bundle_time = Time::Immediately;
    }

    fn current_graph(&self) -> GraphId {
        LOCAL_GRAPH.with_borrow_mut(|g| {
            if let Some(g) = g.last_mut() {
                g.id()
            } else {
                self.selected_graph_remote_graph
            }
        })
    }

    fn set_mortality(&mut self, node: NodeId, is_mortal: bool) {
        // The node may be in our local graph or remotely. Check local first.
        let found_in_local = LOCAL_GRAPH.with_borrow_mut(|g| {
            if let Some(g) = g.last_mut() {
                match g.set_node_mortality(node, is_mortal) {
                    Ok(()) => true,
                    Err(e) => match e {
                        ScheduleError::GraphNotFound(_) => false,
                        _ => {
                            // TODO: Report this error
                            eprintln!("Error: {e:?}");
                            // We found the correct graph, but there was a different error
                            true
                        }
                    },
                }
            } else {
                false
            }
        });
        if !found_in_local {
            self.sender
                .send(Command::SetMortality { node, is_mortal })
                .unwrap();
        }
    }
    // /// Create a new Self which pushes to the selected GraphId by default
    // fn to_graph(&self, graph_id: GraphId) -> Self {
    //     let mut k = self.clone();
    //     k.default_graph_id = graph_id;
    //     k
    // }
    // /// Create a new Self which pushes to the top level GraphId by default
    // fn to_top_level_graph(&self) -> Self {
    //     let mut k = self.clone();
    //     k.default_graph_id = self.top_level_graph_id;
    //     k
    // }
}

thread_local! {
    static LOCAL_GRAPH: RefCell<Vec<Graph>> = RefCell::new(Vec::with_capacity(1));
}

/// Handle to modify a running/scheduled callback
pub struct CallbackHandle {
    free_flag: Arc<AtomicBool>,
}

impl CallbackHandle {
    /// Free/delete the callback this handle refers to.
    pub fn free(self) {
        self.free_flag
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }
    pub(crate) fn dummy_new() -> Self {
        Self {
            free_flag: Arc::new(AtomicBool::new(true)),
        }
    }
}

/// The beat on which a callback should start, either an absolute beat value or the next multiple of some number of beats.
pub enum StartBeat {
    /// An absolute time in beat
    Absolute(Beats),
    /// The next multiple of this number of beats
    Multiple(Beats),
}

/// Callback that is scheduled in [`Beats`]. The closure inside the
/// callback should only schedule changes in Beats time guided by the value
/// to start scheduling that is passed to the function.
///
/// The closure takes two parameters: the time to start the next scheduling in
/// Beats time and a `&mut KnystCommands` for scheduling the changes. The
/// timestamp in the first parameter is the start time of the callback plus all
/// the returned beat intervals to wait until the next callback. The callback
/// can return the time to wait until it gets called again or `None` to remove
/// the callback.
pub struct BeatCallback {
    callback: Box<dyn FnMut(Beats, &mut MultiThreadedKnystCommands) -> Option<Beats> + Send>,
    next_timestamp: Beats,
    free_flag: Arc<AtomicBool>,
}
impl BeatCallback {
    /// Create a new [`BeatCallback`] with a given start time
    fn new(
        callback: impl FnMut(Beats, &mut MultiThreadedKnystCommands) -> Option<Beats> + Send + 'static,
        start_time: Beats,
    ) -> Self {
        let free_flag = Arc::new(AtomicBool::new(false));
        Self {
            callback: Box::new(callback),
            next_timestamp: start_time,
            free_flag,
        }
    }
    fn handle(&self) -> CallbackHandle {
        CallbackHandle {
            free_flag: self.free_flag.clone(),
        }
    }
    /// Called by the Controller when it is time to run the callback to schedule
    /// changes in the future.
    fn run_callback(&mut self, k: &mut MultiThreadedKnystCommands) -> CallbackResult {
        if self.free_flag.load(std::sync::atomic::Ordering::SeqCst) {
            CallbackResult::Delete
        } else {
            match (self.callback)(self.next_timestamp, k) {
                Some(time_to_next) => {
                    self.next_timestamp += time_to_next;
                    CallbackResult::Continue
                }
                None => CallbackResult::Delete,
            }
        }
    }
}

enum CallbackResult {
    Continue,
    Delete,
}

/// Receives commands from one or several [`KnystCommands`] that may be on
/// different threads, and applies those to a top level [`Graph`].
pub struct Controller {
    top_level_graph: Graph,
    command_receiver: Receiver<Command>,
    // TODO: Maybe we don't need to store the sender since it can be produced by cloning a ToKnyst
    command_sender: Sender<Command>,
    resources_sender: rtrb::Producer<ResourcesCommand>,
    resources_receiver: rtrb::Consumer<ResourcesResponse>,
    // The queue is for commands that couldn't be applied yet e.g. because a
    // NodeAddress couldn't be resolved because the node had not yet been
    // pushed.
    command_queue: Vec<(Instant, Command)>,
    error_handler: Box<dyn FnMut(KnystError) + Send>,
    beat_callbacks: Vec<BeatCallback>,
}
impl Controller {
    /// Creates a new [`Controller`] taking the top level [`Graph`] to which
    /// commands will be applied and an error handler. You almost never want to
    /// call this in program code; the AudioBackend will create one for you.
    pub fn new(
        top_level_graph: Graph,
        error_handler: impl FnMut(KnystError) + Send + 'static,
        resources_sender: rtrb::Producer<ResourcesCommand>,
        resources_receiver: rtrb::Consumer<ResourcesResponse>,
    ) -> Self {
        let (sender, receiver) = unbounded();
        Self {
            top_level_graph,
            command_receiver: receiver,
            command_sender: sender,
            command_queue: vec![],
            error_handler: Box::new(error_handler),
            resources_receiver,
            resources_sender,
            beat_callbacks: vec![],
        }
    }

    fn apply_command(&mut self, command: Command) {
        let result: Result<(), crate::KnystError> = match command {
            Command::Push {
                gen_or_graph,
                mut node_address,
                graph_id,
                start_time,
            } => {
                if let Err(e) = self
                    .top_level_graph
                    .push_with_existing_address_to_graph_at_time(
                        gen_or_graph,
                        &mut node_address,
                        graph_id,
                        start_time,
                    )
                {
                    Err(From::from(e))
                } else {
                    Ok(())
                }
            }
            Command::Connect(connection) => {
                match self.top_level_graph.connect(connection.clone()) {
                    Ok(_) => Ok(()),
                    Err(e) => match e {
                        ConnectionError::SourceNodeNotPushed
                        | ConnectionError::SinkNodeNotPushed => {
                            self.command_queue
                                .push((Instant::now(), Command::Connect(connection)));
                            Ok(())
                        }
                        _ => Err(From::from(e)),
                    },
                }
            }
            Command::Disconnect(connection) => {
                match self.top_level_graph.disconnect(connection.clone()) {
                    Ok(_) => Ok(()),
                    Err(e) => match e {
                        ConnectionError::SourceNodeNotPushed
                        | ConnectionError::SinkNodeNotPushed => {
                            self.command_queue
                                .push((Instant::now(), Command::Disconnect(connection)));
                            Ok(())
                        }
                        _ => Err(From::from(e)),
                    },
                }
            }
            Command::FreeNode(node) => match self.top_level_graph.free_node(node) {
                Err(e) => {
                    if let FreeError::NodeNotFound = e {
                        self.command_queue
                            .push((Instant::now(), Command::FreeNode(node)));
                        Ok(())
                    } else {
                        Err(KnystError::from(e))
                    }
                }
                _ => Ok(()),
            },
            Command::FreeNodeMendConnections(node) => {
                match self.top_level_graph.free_node_mend_connections(node) {
                    Err(e) => {
                        if let FreeError::NodeNotFound = e {
                            self.command_queue
                                .push((Instant::now(), Command::FreeNodeMendConnections(node)));
                            Ok(())
                        } else {
                            Err(KnystError::from(e))
                        }
                    }
                    _ => Ok(()),
                }
            }
            Command::ScheduleChange(change) => self
                .top_level_graph
                .schedule_change(change)
                .map_err(|e| From::from(e)),
            Command::FreeDisconnectedNodes => self
                .top_level_graph
                .free_disconnected_nodes()
                .map_err(|e| From::from(e)),
            Command::ResourcesCommand(resources_command) => {
                // Try sending it to Resources. If it fails, store it in the queue.
                match self.resources_sender.push(resources_command) {
                    Ok(_) => Ok(()),
                    Err(e) => match e {
                        rtrb::PushError::Full(resources_command) => {
                            self.command_queue.push((
                                Instant::now(),
                                Command::ResourcesCommand(resources_command),
                            ));
                            Ok(())
                        }
                    },
                }
            }
            Command::ChangeMusicalTimeMap(change_fn) => self
                .top_level_graph
                .change_musical_time_map(change_fn)
                .map_err(|e| From::from(e)),
            Command::ScheduleChanges(changes) => {
                let changes_clone = changes.clone();
                match self
                    .top_level_graph
                    .schedule_changes(changes.changes, changes.time)
                {
                    Ok(_) => Ok(()),
                    Err(e) => match e {
                        crate::graph::ScheduleError::GraphNotFound(_node) => {
                            // println!("Didn't find graph for:");
                            println!("Failed to schedule {changes_clone:?}");
                            Err(e.into())
                        }
                        _ => Err(e.into()),
                    },
                }
            }
            Command::ScheduleBeatCallback(mut callback, start_beat) => {
                // Find the start beat
                let current_beats = self.top_level_graph.get_current_time_musical().unwrap();
                let start_timestamp = match start_beat {
                    StartBeat::Absolute(beats) => beats,
                    StartBeat::Multiple(beats) => {
                        let mut i = 1;
                        while beats * Beats::from_beats(i) < current_beats {
                            i += 1;
                        }
                        beats * Beats::from_beats(i)
                    }
                };
                // println!(
                //     "New callback, current beat: {current_beats:?}, start: {start_timestamp:?}"
                // );
                callback.next_timestamp = start_timestamp;
                self.beat_callbacks.push(callback);
                Ok(())
            }
            Command::RequestInspection(sender) => {
                // TODO: Proper error handling
                sender
                    .send(self.top_level_graph.generate_inspection())
                    .unwrap();
                Ok(())
            }
            Command::SetMortality { node, is_mortal } => self
                .top_level_graph
                .set_node_mortality(node, is_mortal)
                .map_err(|e| From::from(e)),
        };

        if let Err(e) = result {
            (*self.error_handler)(e);
        }
    }

    // Receive commands from the queue and apply them to the graph. If
    // `max_commands` commands have been processed, return so that maintenance
    // functions can be run e.g. updating the scheduler.
    //
    // Returns true if all commands in the queue were processed.
    fn receive_and_apply_commands(&mut self, max_commands: usize) -> bool {
        let mut i = 0;
        while let Ok(command) = self.command_receiver.try_recv() {
            // println!("Received command in controller: {:?}", &command);
            self.apply_command(command);
            i += 1;
            if i >= max_commands {
                return false;
            }
        }
        true
    }

    /// Run maintenance tasks: update the graph and run internal maintenance
    fn run_maintenance(&mut self) {
        self.top_level_graph.update();
        while let Ok(response) = self.resources_receiver.pop() {
            match response {
                ResourcesResponse::InsertBuffer(res) => {
                    if let Err(e) = res {
                        (*self.error_handler)(e.into())
                    }
                }
                ResourcesResponse::RemoveBuffer(res) => {
                    if let Err(e) = res {
                        (*self.error_handler)(e.into())
                    }
                }
                ResourcesResponse::ReplaceBuffer(res) => {
                    if let Err(e) = res {
                        (*self.error_handler)(e.into())
                    }
                }
                ResourcesResponse::InsertWavetable(res) => {
                    if let Err(e) = res {
                        (*self.error_handler)(e.into())
                    }
                }
                ResourcesResponse::RemoveWavetable(res) => {
                    if let Err(e) = res {
                        (*self.error_handler)(e.into())
                    }
                }
                ResourcesResponse::ReplaceWavetable(res) => {
                    if let Err(e) = res {
                        (*self.error_handler)(e.into())
                    }
                }
            }
        }
    }

    fn run_callbacks(&mut self) {
        // Get current time in MusicalTime
        let current_time_beats = self.top_level_graph.get_current_time_musical();
        let mut k = self.get_knyst_commands();
        if let Some(current_time_beats) = current_time_beats {
            let mut i = self.beat_callbacks.len();
            while i != 0 {
                let c = &mut self.beat_callbacks[i - 1];
                if c.next_timestamp < current_time_beats
                    || c.next_timestamp.checked_sub(current_time_beats).unwrap()
                        < Beats::from_beats_f32(0.25)
                {
                    if let CallbackResult::Delete = c.run_callback(&mut k) {
                        self.beat_callbacks.remove(i - 1);
                    }
                }
                i -= 1;
            }
        }
    }

    /// Receives messages, applies them and then runs maintenance. Maintenance
    /// includes updating the [`Graph`], sending the changes made to the
    /// audio thread.
    ///
    /// `max_commands_before_update` is the maximum number of commands read from
    /// the queue before forcing maintenance. If you are sending a lot of
    /// commands, fine tuning this can probably reduce latency.
    ///
    /// Returns true if all commands in the queue were processed.
    pub fn run(&mut self, max_commands_before_update: usize) -> bool {
        // Run the callbacks first because they may send commands that would
        // then get picked up and applied just after.
        self.run_callbacks();
        let all_commands_received = self.receive_and_apply_commands(max_commands_before_update);
        self.run_maintenance();
        all_commands_received
    }

    /// Create a [`KnystCommands`] that can communicate with [`Self`]
    pub fn get_knyst_commands(&self) -> MultiThreadedKnystCommands {
        MultiThreadedKnystCommands {
            sender: self.command_sender.clone(),
            top_level_graph_id: self.top_level_graph.id(),
            top_level_graph_settings: self.top_level_graph.graph_settings(),
            selected_graph_remote_graph: self.top_level_graph.id(),
            bundle_changes: false,
            changes_bundle: vec![],
            changes_bundle_time: Time::Immediately,
        }
    }

    /// Consumes the [`Controller`] and moves it to a new thread where it will `run` in a loop.
    pub fn start_on_new_thread(self) -> MultiThreadedKnystCommands {
        let top_level_graph_id = self.top_level_graph.id();
        let top_level_graph_settings = self.top_level_graph.graph_settings();
        let mut controller = self;
        let sender = controller.command_sender.clone();

        std::thread::spawn(move || loop {
            while !controller.run(300) {}
            std::thread::sleep(Duration::from_micros(1));
        });

        MultiThreadedKnystCommands {
            sender,
            top_level_graph_id,
            top_level_graph_settings,
            selected_graph_remote_graph: top_level_graph_id,
            bundle_changes: false,
            changes_bundle: vec![],
            changes_bundle_time: Time::Immediately,
        }
    }
}

/// Simple error handler that just prints the error using `eprintln!`
pub fn print_error_handler(e: KnystError) {
    eprintln!("Error in Controller: {e}");
}

#[cfg(test)]
mod tests {
    use super::schedule_bundle;
    use crate as knyst;
    use crate::{knyst_commands, offline::KnystOffline, prelude::*, trig::once_trig};

    // Outputs its input value + 1
    struct OneGen {}
    #[impl_gen]
    impl OneGen {
        fn new() -> Self {
            Self {}
        }
        #[process]
        fn process(&mut self, passthrough: &[Sample], out: &mut [Sample]) -> GenState {
            for (i, o) in passthrough.iter().zip(out.iter_mut()) {
                *o = *i + 1.0;
            }
            GenState::Continue
        }
    }

    #[test]
    fn schedule_bundle_test() {
        let sr = 44100;
        let mut kt = KnystOffline::new(sr, 64, 0, 1);
        schedule_bundle(crate::graph::Time::Immediately, || {
            graph_output(0, once_trig());
        });
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(5, sr as u64)),
            || {
                graph_output(0, once_trig());
            },
        );
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(10, sr as u64)),
            || {
                graph_output(0, once_trig());
            },
        );
        let mut og = None;
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(16, sr as u64)),
            || {
                og = Some(one_gen());
                graph_output(0, og.unwrap());
            },
        );
        let og = og.unwrap();
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(17, sr as u64)),
            || {
                og.passthrough(2.0);
            },
        );
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(19, sr as u64)),
            || {
                og.passthrough(3.0);
            },
        );
        // Try with the pure KnystCommands methods as well.
        knyst_commands().start_scheduling_bundle(knyst::graph::Time::Seconds(
            Seconds::from_samples(20, sr as u64),
        ));
        og.passthrough(4.0);
        knyst_commands().upload_scheduling_bundle();
        kt.process_block();
        let o = kt.output_channel(0).unwrap();
        dbg!(o);
        assert_eq!(o[0], 1.0);
        assert_eq!(o[1], 0.0);
        assert_eq!(o[4], 0.0);
        assert_eq!(o[5], 1.0);
        assert_eq!(o[6], 0.0);
        assert_eq!(o[10], 1.0);
        assert_eq!(o[11], 0.0);
        assert_eq!(o[16], 1.0);
        assert_eq!(o[17], 3.0);
        assert_eq!(o[19], 4.0);
        assert_eq!(o[20], 5.0);
    }

    #[test]
    fn schedule_bundle_inner_graph_test() {
        let sr = 44100;
        let mut kt = KnystOffline::new(sr, 64, 0, 1);
        // We create a first graph so that the top graph will try to schedule on this one first and fail.
        let mut ignored_graph_node = None;
        let _ignored_graph = upload_graph(knyst_commands().default_graph_settings(), || {
            ignored_graph_node = Some(one_gen());
        });
        let mut inner_graph = None;
        let graph = upload_graph(knyst_commands().default_graph_settings(), || {
            let g = upload_graph(knyst_commands().default_graph_settings(), || ());
            graph_output(0, g);
            inner_graph = Some(g);
        });
        graph_output(0, graph);
        inner_graph.unwrap().activate();
        schedule_bundle(crate::graph::Time::Immediately, || {
            graph_output(0, once_trig());
        });
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(5, sr as u64)),
            || {
                graph_output(0, once_trig());
            },
        );
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(10, sr as u64)),
            || {
                graph_output(0, once_trig());
            },
        );
        let mut og = None;
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(16, sr as u64)),
            || {
                og = Some(one_gen());
                graph_output(0, og.unwrap());
            },
        );
        let og = og.unwrap();
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(17, sr as u64)),
            || {
                og.passthrough(2.0);

                // This will set a value of a node in a different graph, but won't change the output
                ignored_graph_node.unwrap().passthrough(10.);
            },
        );
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(19, sr as u64)),
            || {
                og.passthrough(3.0);
            },
        );
        // Try with the pure KnystCommands methods as well.
        knyst_commands().start_scheduling_bundle(knyst::graph::Time::Seconds(
            Seconds::from_samples(20, sr as u64),
        ));
        og.passthrough(4.0);
        knyst_commands().upload_scheduling_bundle();
        kt.process_block();
        let o = kt.output_channel(0).unwrap();
        dbg!(o);
        assert_eq!(o[0], 1.0);
        assert_eq!(o[1], 0.0);
        assert_eq!(o[4], 0.0);
        assert_eq!(o[5], 1.0);
        assert_eq!(o[6], 0.0);
        assert_eq!(o[10], 1.0);
        assert_eq!(o[11], 0.0);
        assert_eq!(o[16], 1.0);
        assert_eq!(o[17], 3.0);
        assert_eq!(o[19], 4.0);
        assert_eq!(o[20], 5.0);
    }
}
