use std::time::{Duration, Instant};

use crate::{
    buffer::Buffer,
    graph::{
        Connection, ConnectionError, Gen, GenOrGraph, GenOrGraphEnum, Graph, GraphId, NodeAddress,
        ParameterChange,
    },
    BufferId, KnystError, ResourcesCommand, ResourcesResponse,
};
use crossbeam_channel::{unbounded, Receiver, Sender};

/// Encodes commands sent from a [`KnystCommands`]
enum Command {
    Push {
        gen_or_graph: GenOrGraphEnum,
        node_address: NodeAddress,
        graph_id: GraphId,
    },
    Connect(Connection),
    FreeNode(NodeAddress),
    FreeNodeMendConnections(NodeAddress),
    ScheduleChange(ParameterChange),
    FreeDisconnectedNodes,
    ResourcesCommand(ResourcesCommand),
    // TODO: Commands to change a Resources
}

impl Command {}

/// [`KnystCommands`] sends commands to the [`Controller`] which should hold the
/// top level [`Graph`]. The API is as close as possible to that of an owned
/// [`Graph`].
///
/// This can safely be cloned and sent to a different thread for use.
///
// TODO: Any errors need a mechanism to be sent back, mostly for printing and
// debugging.
//
// TODO: What's the best way of referring to a graph? GraphId is unique, but not
// always the handiest. It would be nice to be able to choose to refer to Graphs
// by an identifier e.g. name. In Bevy holding on to GraphIds is easy.
#[derive(Clone)]
pub struct KnystCommands {
    sender: crossbeam_channel::Sender<Command>,
    top_level_graph_id: GraphId,
}

impl KnystCommands {
    pub fn push(&mut self, gen_or_graph: impl GenOrGraph) -> NodeAddress {
        self.push_to_graph(gen_or_graph, self.top_level_graph_id)
    }
    pub fn push_to_graph(
        &mut self,
        gen_or_graph: impl GenOrGraph,
        graph_id: GraphId,
    ) -> NodeAddress {
        let new_node_address = NodeAddress::new();
        let command = Command::Push {
            gen_or_graph: gen_or_graph.into_gen_or_graph_enum(),
            node_address: new_node_address.clone(),
            graph_id,
        };
        self.sender.send(command).unwrap();
        new_node_address
    }
    pub fn connect(&mut self, connection: Connection) {
        self.sender.send(Command::Connect(connection)).unwrap();
    }
    pub fn free_disconnected_nodes(&mut self) {
        self.sender.send(Command::FreeDisconnectedNodes).unwrap();
    }
    pub fn free_node_mend_connections(&mut self, node: NodeAddress) {
        self.sender
            .send(Command::FreeNodeMendConnections(node))
            .unwrap();
    }
    pub fn free_node(&mut self, node: NodeAddress) {
        self.sender.send(Command::FreeNode(node)).unwrap();
    }
    pub fn schedule_change(&mut self, change: ParameterChange) {
        self.sender.send(Command::ScheduleChange(change)).unwrap();
    }
    pub fn insert_buffer(&mut self, buffer: Buffer) -> BufferId {
        let id = BufferId::new();
        self.sender
            .send(Command::ResourcesCommand(ResourcesCommand::InsertBuffer {
                id,
                buffer,
            }))
            .unwrap();
        id
    }
    pub fn remove_buffer(&mut self, buffer_id: BufferId) {
        self.sender
            .send(Command::ResourcesCommand(ResourcesCommand::RemoveBuffer {
                id: buffer_id,
            }))
            .unwrap();
    }
    pub fn replace_buffer(&mut self, buffer_id: BufferId, buffer: Buffer) {
        self.sender
            .send(Command::ResourcesCommand(ResourcesCommand::ReplaceBuffer {
                id: buffer_id,
                buffer,
            }))
            .unwrap();
    }
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
            Command::FreeNode(node_address) => {
                if let Some(raw_address) = node_address.to_raw() {
                    self.top_level_graph
                        .free_node(raw_address)
                        .map_err(|e| From::from(e))
                } else {
                    self.command_queue
                        .push((Instant::now(), Command::FreeNode(node_address)));
                    Ok(())
                }
            }
            Command::FreeNodeMendConnections(node_address) => {
                if let Some(raw_address) = node_address.to_raw() {
                    self.top_level_graph
                        .free_node_mend_connections(raw_address)
                        .map_err(|e| From::from(e))
                } else {
                    self.command_queue
                        .push((Instant::now(), Command::FreeNode(node_address)));
                    Ok(())
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
        let all_commands_received = self.receive_and_apply_commands(max_commands_before_update);
        self.run_maintenance();
        all_commands_received
    }

    pub fn get_knyst_commands(&self) -> KnystCommands {
        KnystCommands {
            sender: self.command_sender.clone(),
            top_level_graph_id: self.top_level_graph.id(),
        }
    }

    /// Consumes the [`Controller`] and moves it to a new thread where it will `run` in a loop.
    pub fn start_on_new_thread(self) -> KnystCommands {
        let top_level_graph_id = self.top_level_graph.id();
        let mut controller = self;
        let sender = controller.command_sender.clone();

        std::thread::spawn(move || loop {
            while !controller.run(300) {}
            std::thread::sleep(Duration::from_micros(1));
        });

        KnystCommands {
            sender,
            top_level_graph_id,
        }
    }
}

pub fn print_error_handler(e: KnystError) {
    eprintln!("Error in Controller: {e}");
}
