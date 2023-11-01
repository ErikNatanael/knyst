//! API for interacting with a running top level [`Graph`] from any number of
//! threads without having to manually keep track of running [`Graph::update`]
//! regularly.
//!
//! [`KnystCommands`] gives you a convenient API for sending messages to the
//! [`Controller`]. The API is similar to calling methods on [`Graph`] directly,
//! but also includes modifying [`Resources`].

use std::{
    sync::{atomic::AtomicBool, Arc},
    time::{Duration, Instant},
};

use crate::{
    graph::{
        connection::{ConnectionBundle, ConnectionError, InputBundle},
        Connection, FreeError, GenOrGraph, GenOrGraphEnum, Graph, GraphId, GraphSettings, NodeId,
        ParameterChange, SimultaneousChanges,
    },
    scheduling::MusicalTimeMap,
    time::Superbeats,
    KnystError,
};
use crossbeam_channel::{unbounded, Receiver, Sender};
use knyst_core::{
    buffer::Buffer,
    resources::{BufferId, ResourcesCommand, ResourcesResponse, WavetableId},
    wavetable::Wavetable,
};

/// Encodes commands sent from a [`KnystCommands`]
enum Command {
    Push {
        gen_or_graph: GenOrGraphEnum,
        node_address: NodeId,
        graph_id: GraphId,
    },
    Connect(Connection),
    Disconnect(Connection),
    FreeNode(NodeId),
    FreeNodeMendConnections(NodeId),
    ScheduleChange(ParameterChange),
    ScheduleChanges(SimultaneousChanges),
    FreeDisconnectedNodes,
    ResourcesCommand(ResourcesCommand),
    ChangeMusicalTimeMap(Box<dyn FnOnce(&mut MusicalTimeMap) + Send>),
    ScheduleBeatCallback(BeatCallback, StartBeat),
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
        callback: impl FnMut(Superbeats, &mut MultiThreadedKnystCommands) -> Option<Superbeats>
            + Send
            + 'static,
        start_time: StartBeat,
    ) -> CallbackHandle;
    /// Disconnect (undo) a [`Connection`]
    fn disconnect(&mut self, connection: Connection);
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
    /// Return the [`GraphSettings`] of the top level graph. This means you
    /// don't have to manually keep track of matching sample rate and block size
    /// for example.
    fn default_graph_settings(&self) -> GraphSettings;
    // /// Create a new Self which pushes to the selected GraphId by default
    // /// TODO: modal way to set the current graph instead
    // fn to_graph(&self, graph_id: GraphId) -> Self;
    // /// Create a new Self which pushes to the top level GraphId by default
    // /// TODO: modal way to set the current graph instead
    // fn to_top_level_graph(&self) -> Self;
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
    /// The default graph to push
    default_graph_id: GraphId,
}

impl KnystCommands for MultiThreadedKnystCommands {
    /// Push a Gen or Graph to the top level Graph without specifying any inputs.
    fn push_without_inputs(&mut self, gen_or_graph: impl GenOrGraph) -> NodeId {
        self.push_to_graph_without_inputs(gen_or_graph, self.top_level_graph_id)
    }
    /// Push a Gen or Graph to the default Graph.
    fn push(&mut self, gen_or_graph: impl GenOrGraph, inputs: impl Into<InputBundle>) -> NodeId {
        let addr = self.push_to_graph_without_inputs(gen_or_graph, self.default_graph_id);
        let inputs: InputBundle = inputs.into();
        self.connect_bundle(inputs.to(addr));
        addr
    }
    /// Push a Gen or Graph to the Graph with the specified id without specifying inputs.
    fn push_to_graph_without_inputs(
        &mut self,
        gen_or_graph: impl GenOrGraph,
        graph_id: GraphId,
    ) -> NodeId {
        let new_node_address = NodeId::new();
        let command = Command::Push {
            gen_or_graph: gen_or_graph.into_gen_or_graph_enum(),
            node_address: new_node_address.clone(),
            graph_id,
        };
        self.sender.send(command).unwrap();
        new_node_address
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
        self.sender.send(Command::Connect(connection)).unwrap();
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
        callback: impl FnMut(Superbeats, &mut MultiThreadedKnystCommands) -> Option<Superbeats>
            + Send
            + 'static,
        start_time: StartBeat,
    ) -> CallbackHandle {
        let c = BeatCallback::new(callback, Superbeats::ZERO);
        let handle = c.handle();
        let command = Command::ScheduleBeatCallback(c, start_time);
        self.sender.send(command).unwrap();
        handle
    }
    /// Disconnect (undo) a [`Connection`]
    fn disconnect(&mut self, connection: Connection) {
        self.sender.send(Command::Disconnect(connection)).unwrap();
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
        self.sender.send(Command::ScheduleChange(change)).unwrap();
    }
    /// Schedule multiple changes to be made.
    ///
    /// NB: Changes are buffered and the scheduler needs to be regularly updated
    /// for them to be sent to the audio thread. If you are getting your
    /// [`KnystCommands`] through `AudioBackend::start_processing` this is taken
    /// care of automatically.
    fn schedule_changes(&mut self, changes: SimultaneousChanges) {
        self.sender.send(Command::ScheduleChanges(changes)).unwrap();
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
    Absolute(Superbeats),
    /// The next multiple of this number of beats
    Multiple(Superbeats),
}

/// Callback that is scheduled in [`Superbeats`]. The closure inside the
/// callback should only schedule changes in Superbeats time guided by the value
/// to start scheduling that is passed to the function.
///
/// The closure takes two parameters: the time to start the next scheduling in
/// Superbeats time and a `&mut KnystCommands` for scheduling the changes. The
/// timestamp in the first parameter is the start time of the callback plus all
/// the returned beat intervals to wait until the next callback. The callback
/// can return the time to wait until it gets called again or `None` to remove
/// the callback.
pub struct BeatCallback {
    callback:
        Box<dyn FnMut(Superbeats, &mut MultiThreadedKnystCommands) -> Option<Superbeats> + Send>,
    next_timestamp: Superbeats,
    free_flag: Arc<AtomicBool>,
}
impl BeatCallback {
    /// Create a new [`BeatCallback`] with a given start time
    fn new(
        callback: impl FnMut(Superbeats, &mut MultiThreadedKnystCommands) -> Option<Superbeats>
            + Send
            + 'static,
        start_time: Superbeats,
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
            } => {
                if let Err(e) = self.top_level_graph.push_with_existing_address_to_graph(
                    gen_or_graph,
                    &mut node_address,
                    graph_id,
                ) {
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
                match self.top_level_graph.schedule_changes(changes) {
                    Ok(_) => Ok(()),
                    Err(e) => match e {
                        crate::graph::ScheduleError::GraphNotFound => {
                            // println!("Didn't find graph for:");
                            // println!("{changes_clone:?}");
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
                        while beats * Superbeats::from_beats(i) < current_beats {
                            i += 1;
                        }
                        beats * Superbeats::from_beats(i)
                    }
                };
                println!(
                    "New callback, current beat: {current_beats:?}, start: {start_timestamp:?}"
                );
                callback.next_timestamp = start_timestamp;
                self.beat_callbacks.push(callback);
                Ok(())
            }
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
                        < Superbeats::from_beats_f32(0.25)
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
    /// includes updating the [`Graph`](s), sending the changes made to the
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
            default_graph_id: self.top_level_graph.id(),
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
            default_graph_id: top_level_graph_id,
        }
    }
}

/// Simple error handler that just prints the error using `eprintln!`
pub fn print_error_handler(e: KnystError) {
    eprintln!("Error in Controller: {e}");
}
