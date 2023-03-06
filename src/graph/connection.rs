use super::{FreeError, NodeAddress, Sample};

/// Connection provides a convenient API for creating connections between nodes in a
/// graph. When used for nodes in a running graph, the shadow engine will translate the
/// Connection to a RtConnection which contains the full Path for finding the correct Graph
/// as fast as possible.
///
/// A node can have any number of connections to/from other nodes or outputs.
/// Multiple constant values will result in the sum of all the constants.
#[derive(Clone)]
pub enum Connection {
    /// node to node
    Node {
        source: NodeAddress,
        from_index: Option<usize>,
        from_label: Option<&'static str>,
        sink: NodeAddress,
        /// if input_index and input_label are both None, the default is index 0
        to_index: Option<usize>,
        to_label: Option<&'static str>,
        /// default: 1
        channels: usize,
        feedback: bool,
    },
    /// constant to node
    Constant {
        value: Sample,
        sink: Option<NodeAddress>,
        to_index: Option<usize>,
        to_label: Option<&'static str>,
    },
    /// node to graph output
    GraphOutput {
        source: NodeAddress,
        from_index: Option<usize>,
        from_label: Option<&'static str>,
        to_index: usize,
        channels: usize,
    },
    /// graph input to node
    GraphInput {
        sink: NodeAddress,
        from_index: usize,
        /// if input_index and input_label are both None, the default is index 0
        to_index: Option<usize>,
        to_label: Option<&'static str>,
        channels: usize,
    },
    Clear {
        node: NodeAddress,
        /// connections to this node
        input_nodes: bool,
        /// constant input values of this node
        input_constants: bool,
        /// connections from this node to other nodes
        output_nodes: bool,
        /// connections from this node to the graph output(s)
        graph_outputs: bool,
        /// connections from the graph inputs to the node
        graph_inputs: bool,
    },
}

pub fn constant(value: Sample) -> Connection {
    Connection::Constant {
        value,
        sink: None,
        to_index: None,
        to_label: None,
    }
}
impl Connection {
    pub fn graph_output(source_node: &NodeAddress) -> Self {
        Self::GraphOutput {
            source: source_node.clone(),
            from_index: Some(0),
            from_label: None,
            to_index: 0,
            channels: 1,
        }
    }
    pub fn graph_input(sink_node: &NodeAddress) -> Self {
        Self::GraphInput {
            sink: sink_node.clone(),
            from_index: 0,
            to_index: None,
            to_label: None,
            channels: 1,
        }
    }
    pub fn clear_constants(node: &NodeAddress) -> Self {
        Self::Clear {
            node: node.clone(),
            input_constants: true,
            input_nodes: false,
            output_nodes: false,
            graph_outputs: false,
            graph_inputs: false,
        }
    }
    pub fn clear_from_nodes(node: &NodeAddress) -> Self {
        Self::Clear {
            node: node.clone(),
            input_constants: false,
            input_nodes: true,
            output_nodes: false,
            graph_outputs: false,
            graph_inputs: false,
        }
    }
    pub fn clear_to_nodes(node: &NodeAddress) -> Self {
        Self::Clear {
            node: node.clone(),
            input_constants: false,
            input_nodes: false,
            output_nodes: true,
            graph_outputs: false,
            graph_inputs: false,
        }
    }
    pub fn clear_to_graph_outputs(node: &NodeAddress) -> Self {
        Self::Clear {
            node: node.clone(),
            input_constants: false,
            input_nodes: false,
            output_nodes: false,
            graph_outputs: true,
            graph_inputs: false,
        }
    }
    pub fn clear_from_graph_inputs(node: &NodeAddress) -> Self {
        Self::Clear {
            node: node.clone(),
            input_constants: false,
            input_nodes: false,
            output_nodes: false,
            graph_outputs: false,
            graph_inputs: true,
        }
    }
    /// Sets the source of a Connection. Only valid for Connection::Node and
    /// Connection::GraphOutput. On other variants it does nothing.
    pub fn from(mut self, source_node: &NodeAddress) -> Self {
        match &mut self {
            Connection::Node { ref mut source, .. } => {
                *source = source_node.clone();
            }
            Connection::GraphInput { .. } => {}
            Connection::Constant { .. } => {}
            Connection::GraphOutput { ref mut source, .. } => {
                *source = source_node.clone();
            }
            Connection::Clear { .. } => {}
        }
        self
    }
    /// Sets the source of a Connection. Does nothing on a
    /// Connection::GraphOutput or Connection::Clear.
    pub fn to(mut self, sink_node: &NodeAddress) -> Self {
        match &mut self {
            Connection::Node { ref mut sink, .. } => {
                *sink = sink_node.clone();
            }
            Connection::GraphInput { ref mut sink, .. } => {
                *sink = sink_node.clone();
            }
            Connection::Constant { ref mut sink, .. } => {
                *sink = Some(sink_node.clone());
            }
            Connection::GraphOutput { .. } => {}
            Connection::Clear { .. } => {}
        }
        self
    }
    pub fn to_label(mut self, label: &'static str) -> Self {
        match &mut self {
            Connection::Node {
                to_label: input_label,
                to_index,
                ..
            } => {
                *input_label = Some(label);
                *to_index = None;
            }
            Connection::Constant {
                to_label: input_label,
                ..
            } => {
                *input_label = Some(label);
            }
            Connection::GraphOutput { .. } => {}
            Connection::GraphInput {
                to_label: input_label,
                to_index,
                ..
            } => {
                *input_label = Some(label);
                *to_index = None;
            }
            Connection::Clear { .. } => {}
        }
        self
    }
    /// Shortcut for `to_label`
    pub fn tl(self, label: &'static str) -> Self {
        self.to_label(label)
    }
    pub fn to_index(mut self, index: usize) -> Self {
        match &mut self {
            Connection::Node {
                to_index: input_index,
                to_label,
                ..
            } => {
                *input_index = Some(index);
                *to_label = None;
            }
            Connection::Constant {
                to_index: input_index,
                ..
            } => {
                *input_index = Some(index);
            }
            Connection::GraphOutput { to_index, .. } => {
                *to_index = index;
            }
            Connection::Clear { .. } => {}
            Connection::GraphInput {
                to_index, to_label, ..
            } => {
                *to_index = Some(index);
                *to_label = None;
            }
        }
        self
    }
    pub fn get_from_index(&self) -> Option<usize> {
        match &self {
            Connection::Node { from_index, .. } => *from_index,
            Connection::GraphOutput { from_index, .. } => *from_index,
            Connection::Clear { .. } => None,
            Connection::Constant { .. } => None,
            Connection::GraphInput { from_index, .. } => Some(*from_index),
        }
    }
    pub fn get_to_index(&self) -> Option<usize> {
        match &self {
            Connection::Node {
                to_index: input_index,
                ..
            } => *input_index,
            Connection::Constant {
                to_index: input_index,
                ..
            } => *input_index,
            Connection::GraphOutput { to_index, .. } => Some(*to_index),
            Connection::Clear { .. } => None,
            Connection::GraphInput { to_index, .. } => *to_index,
        }
    }
    /// Shortcut for `to_index`
    pub fn ti(self, index: usize) -> Self {
        self.to_index(index)
    }
    pub fn from_index(mut self, index: usize) -> Self {
        match &mut self {
            Connection::Node {
                from_index,
                from_label,
                ..
            } => {
                *from_index = Some(index);
                *from_label = None;
            }
            Connection::Constant { .. } => {}
            Connection::GraphOutput {
                from_index,
                from_label,
                ..
            } => {
                *from_index = Some(index);
                *from_label = None;
            }
            Connection::GraphInput { from_index, .. } => {
                *from_index = index;
            }
            Connection::Clear { .. } => {}
        }
        self
    }
    pub fn from_label(mut self, label: &'static str) -> Self {
        match &mut self {
            Connection::Node {
                from_index,
                from_label,
                ..
            } => {
                *from_label = Some(label);
                *from_index = None;
            }
            Connection::Constant { .. } => {}
            Connection::GraphOutput {
                from_index,
                from_label,
                ..
            } => {
                *from_label = Some(label);
                *from_index = None;
            }
            Connection::GraphInput { .. } => {}
            Connection::Clear { .. } => {}
        }
        self
    }
    /// Set how many channels should be connected together. This is most useful
    /// for multi channel connections e.g. stereo -> stereo. Inputs/outputs will
    /// wrap if the `num_channels` value if higher than the number of
    /// inputs/outputs available.
    pub fn channels(mut self, num_channels: usize) -> Self {
        match &mut self {
            Connection::Node { channels, .. } => {
                *channels = num_channels;
            }
            Connection::Constant { .. } => {}
            Connection::GraphOutput { channels, .. } => {
                *channels = num_channels;
            }
            Connection::GraphInput { channels, .. } => {
                *channels = num_channels;
            }
            Connection::Clear { .. } => {}
        }
        self
    }
    pub fn feedback(mut self, activate: bool) -> Self {
        match &mut self {
            Connection::Node { feedback, .. } => {
                *feedback = activate;
            }
            Connection::Constant { .. } => {}
            Connection::GraphOutput { .. } => {}
            Connection::GraphInput { .. } => {}
            Connection::Clear { .. } => {}
        }
        self
    }
    pub fn get_source_node(&self) -> Option<NodeAddress> {
        match self {
            Connection::Node { source, .. } => Some(source.clone()),
            Connection::GraphOutput { source, .. } => Some(source.clone()),
            Connection::GraphInput { .. }
            | Connection::Constant { .. }
            | Connection::Clear { .. } => None,
        }
    }
}

#[derive(thiserror::Error, Debug, PartialEq)]
pub enum ConnectionError {
    #[error("The nodes that you are trying to connect are in different graphs. Nodes can only be connected within a graph.")]
    DifferentGraphs,
    #[error("The graph containing the NodeAdress provided was not found. The node itself may or may not exist.")]
    GraphNotFound,
    #[error("The NodeAddress does not exist. The Node may have been freed already.")]
    NodeNotFound,
    #[error("The given input label (`{0}`) is not available for the given node.")]
    InvalidInputLabel(&'static str),
    #[error("The given output label (`{0}`) is not available for the given node.")]
    InvalidOutputLabel(&'static str),
    #[error("You are trying to connect a node to itself. This can only be done using a feedback connection.")]
    SameNode,
    #[error("The sink node for the connection is not set and is required.")]
    SinkNotSet,
    #[error("You are trying to connect to channels that don't exist, either through direct indexing or a too high `channels` value for the input.")]
    ChannelOutOfBounds,
    #[error("You are referencing a node which has not yet been pushed to a graph and therefore cannot be found.")]
    SourceNodeNotPushed,
    #[error("You are referencing a node which has not yet been pushed to a graph and therefore cannot be found.")]
    SinkNodeNotPushed,
    #[error("The connection change required freeing a node, but the node could not be freed.")]
    NodeFree(#[from] FreeError),
}
