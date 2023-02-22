use std::time::{Duration, Instant};

use crate::graph::{
    Connection, ConnectionError, Gen, GenOrGraph, GenOrGraphEnum, Graph, GraphId, NodeAddress,
    ParameterChange,
};
use crossbeam_channel::{unbounded, Receiver, Sender};

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
    // TODO: Commands to change a Resources
}

impl Command {}

/// [`AsyncKnyst`] communicates asynchronously with the main [`Graph`] on a
/// different thread. The API is as close as possible to that of a owned [`Graph`].
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
pub struct ToKnyst {
    sender: crossbeam_channel::Sender<Command>,
    top_level_graph_id: GraphId,
}

impl ToKnyst {
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
}

pub struct AsyncKnystController {
    top_level_graph: Graph,
    command_receiver: Receiver<Command>,
    // TODO: Maybe we don't need to store the sender since it can be produced by cloning a ToKnyst
    command_sender: Sender<Command>,
    // the queue is for commands that couldn't be applied yet e.g. because a
    // NodeAddress couldn't be resolved because the node had not yet been
    // pushed.
    command_queue: Vec<(Instant, Command)>,
}
impl AsyncKnystController {
    pub fn new(top_level_graph: Graph) -> Self {
        let (sender, receiver) = unbounded();
        Self {
            top_level_graph,
            command_receiver: receiver,
            command_sender: sender,
            command_queue: vec![],
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
        };

        // TODO: Do something better with the result: send it through a channel, store it in a log
        if let Err(e) = result {
            eprintln!("Error in async controller: {e}");
        }
    }

    // Receive commands from the queue and apply them to the graph. If
    // `max_commands` commands have been processed, return so that maintenance
    // functions can be run e.g. updating the scheduler.
    pub fn receive_and_apply_commands(&mut self, max_commands: usize) {
        let mut i = 0;
        while let Ok(command) = self.command_receiver.try_recv() {
            self.apply_command(command);
            i += 1;
            if i >= max_commands {
                break;
            }
        }
    }

    /// Run maintenance tasks: update the graph and run internal maintenance
    pub fn run_maintenance(&mut self) {
        self.top_level_graph.update();
    }
}

pub fn start_async_knyst_thread(top_level_graph: Graph) -> ToKnyst {
    let top_level_graph_id = top_level_graph.id();
    let mut controller = AsyncKnystController::new(top_level_graph);
    let sender = controller.command_sender.clone();

    std::thread::spawn(move || loop {
        controller.receive_and_apply_commands(300);
        controller.run_maintenance();
        std::thread::sleep(Duration::from_micros(1));
    });

    ToKnyst {
        sender,
        top_level_graph_id,
    }
}
