//! Connecting nodes/[`Gen`]s in a Graph is fundamentally done through a
//! [`Connection`]. Additionally, a [`NodeAddress`] has convenience functions to
//! generate [`Connection`]s and the [`ConnectionBundle`] API is often more
//! convenient and ergonomic.
use std::fmt::Display;

use super::{FreeError, GraphId, NodeId, Sample};

/// Connection provides a convenient API for creating connections between nodes in a
/// graph. When used for nodes in a running graph, the shadow engine will translate the
/// Connection to a RtConnection which contains the full Path for finding the correct Graph
/// as fast as possible.
///
/// A node can have any number of connections to/from other nodes or outputs.
/// Multiple constant values will result in the sum of all the constants.
#[derive(Clone, Debug, PartialEq)]
pub enum Connection {
    /// node to node
    Node {
        /// Connect from the output of the source node
        source: NodeId,
        /// What output channel from the source node to connect from.
        ///
        /// Either from_index or from_label can be used, but index takes precedence if both are set.
        from_index: Option<usize>,
        /// What output channel from the source node to connect from.
        ///
        /// Either from_index or from_label can be used, but index takes precedence if both are set.
        from_label: Option<&'static str>,
        /// Connect to the input of the sink node
        sink: NodeId,
        /// What input channel on the sink node to connect to.
        ///
        /// Either to_index or to_label can be used, but index takes precedence if both are set.
        /// if input_index and input_label are both None, the default is index 0
        to_index: Option<usize>,
        /// What input channel on the sink node to connect to.
        ///
        /// Either to_index or to_label can be used, but index takes precedence if both are set.
        to_label: Option<&'static str>,
        /// How many channels to connect. Channels will wrap if this value is
        /// larger than the number of channels on the sink and/or source nodes. Default: 1.
        channels: usize,
        /// Set to true if this connection should be a feedback connection,
        /// meaning gives the values from the previous block, but allows loops
        /// in the [`Graph`].
        feedback: bool,
    },
    /// Set a node input constant. This sets the value immediately if the
    /// [`Graph`] is not running. Otherwise it is equivalent to scheduling a
    /// parameter change as soon as possible.
    Constant {
        /// New constant value
        value: Sample,
        /// The node whose input constant to set
        sink: Option<NodeId>,
        /// What input channel to set the constant value of.
        ///
        /// Either to_index or to_label can be used, but index takes precedence if both are set.
        /// if input_index and input_label are both None, the default is index 0
        to_index: Option<usize>,
        /// What input channel to set the constant value of.
        ///
        /// Either to_index or to_label can be used, but index takes precedence if both are set.
        to_label: Option<&'static str>,
    },
    /// node to graph output
    GraphOutput {
        /// From which node to connect to the graph output.
        source: NodeId,
        /// What output channel from the source node to connect from.
        ///
        /// Either from_index or from_label can be used, but index takes precedence if both are set.
        from_index: Option<usize>,
        /// What output channel from the source node to connect from.
        ///
        /// Either from_index or from_label can be used, but index takes precedence if both are set.
        from_label: Option<&'static str>,
        /// What channel index of the graph outputs to connect to.
        to_index: usize,
        /// How many channels to connect. Channels will wrap if this value is
        /// larger than the number of channels on the sink and/or source nodes. Default: 1.
        channels: usize,
    },
    /// graph input to node
    GraphInput {
        /// To which node a graph input will be connected
        sink: NodeId,
        /// From what index of the [`Graph`] inputs we will connect
        from_index: usize,
        /// What input channel to connect to.
        ///
        /// Either to_index or to_label can be used, but index takes precedence if both are set.
        /// if input_index and input_label are both None, the default is index 0
        to_index: Option<usize>,
        /// What input channel to connect to.
        ///
        /// Either to_index or to_label can be used, but index takes precedence if both are set.
        /// if input_index and input_label are both None, the default is index 0
        to_label: Option<&'static str>,
        /// How many channels to connect. Channels will wrap if this value is
        /// larger than the number of channels on the sink and/or source nodes. Default: 1.
        channels: usize,
    },
    /// Clear connections related to a node
    Clear {
        /// The node whose connections to clear
        node: NodeId,
        /// If true, clear connections to this node
        input_nodes: bool,
        /// If true, clear constant input values of this node
        input_constants: bool,
        /// If true, clear connections from this node to other nodes
        output_nodes: bool,
        /// If true, clear connections from this node to the graph output(s)
        graph_outputs: bool,
        /// If true, clear connections from the graph inputs to the node
        graph_inputs: bool,
        /// Specifies a specific channel to clear. If None (default), clear all.
        channel: Option<NodeChannel>,
    },
    /// Connect between a graph input and its output directly
    GraphInputToOutput {
        /// The index of the first input channel to connect from
        from_input_channel: usize,
        /// The index of the first output channel to connect to
        to_output_channel: usize,
        /// How many channels to connect
        channels: usize,
    },
}
impl Display for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Connection::Node {
                source,
                from_index,
                from_label,
                sink,
                to_index,
                to_label,
                channels,
                feedback,
            } => write!(f, "Connection::Node {:?}:{from_index:?}/{from_label:?} -> {:?}:{to_index:?}/{to_label:?}", source, sink),
            Connection::Constant {
                value,
                sink,
                to_index,
                to_label,
            } => write!(f, "Connection::Constant {} -> {:?}:{to_index:?}/{to_label:?}", value, sink),
            Connection::GraphOutput {
                source,
                from_index,
                from_label,
                to_index,
                channels,
            } => write!(f, "Connection::GraphOutput {:?}:{from_index:?}/{from_label:?} -> GraphOutput", source),
            Connection::GraphInput {
                sink,
                from_index,
                to_index,
                to_label,
                channels,
            } => write!(f, "Connection::GraphInput  {from_index:?} -> {:?}:{to_index:?}/{to_label:?}", sink),
            Connection::Clear {
                node,
                input_nodes,
                input_constants,
                output_nodes,
                graph_outputs,
                graph_inputs,
                channel,
            } => write!(f, "Connection::Clear {:?}", node),
            Connection::GraphInputToOutput {
                from_input_channel,
                to_output_channel,
                channels,
            } => write!(f, "Connection::GraphInputToOutput {from_input_channel} -> {to_output_channel}"),
        }
    }
}

/// Convenience function to create a constant input value change connection
pub fn constant(value: Sample) -> Connection {
    Connection::Constant {
        value,
        sink: None,
        to_index: None,
        to_label: None,
    }
}
impl Connection {
    /// Create a connection from `source_node` to a graph output
    pub fn graph_output(source_node: NodeId) -> Self {
        Self::GraphOutput {
            source: source_node.clone(),
            from_index: Some(0),
            from_label: None,
            to_index: 0,
            channels: 1,
        }
    }
    /// Create a connection from a graph_input to the `sink_node`
    pub fn graph_input(sink_node: NodeId) -> Self {
        Self::GraphInput {
            sink: sink_node.clone(),
            from_index: 0,
            to_index: None,
            to_label: None,
            channels: 1,
        }
    }
    /// Clear input constants of `node`
    pub fn clear_constants(node: NodeId) -> Self {
        Self::Clear {
            node: node.clone(),
            input_constants: true,
            input_nodes: false,
            output_nodes: false,
            graph_outputs: false,
            graph_inputs: false,
            channel: None,
        }
    }
    /// Clear connections from other nodes to this `node`'s input(s)
    pub fn clear_from_nodes(node: NodeId) -> Self {
        Self::Clear {
            node: node.clone(),
            input_constants: false,
            input_nodes: true,
            output_nodes: false,
            graph_outputs: false,
            graph_inputs: false,
            channel: None,
        }
    }
    /// Clear connections from the `node` specified to other nodes
    pub fn clear_to_nodes(node: NodeId) -> Self {
        Self::Clear {
            node: node.clone(),
            input_constants: false,
            input_nodes: false,
            output_nodes: true,
            graph_outputs: false,
            graph_inputs: false,
            channel: None,
        }
    }
    /// Clear connections from the `node` to the graph outputs
    pub fn clear_to_graph_outputs(node: NodeId) -> Self {
        Self::Clear {
            node: node.clone(),
            input_constants: false,
            input_nodes: false,
            output_nodes: false,
            graph_outputs: true,
            graph_inputs: false,
            channel: None,
        }
    }
    /// Clear connections from the graph inputs to the `node`
    pub fn clear_from_graph_inputs(node: NodeId) -> Self {
        Self::Clear {
            node: node.clone(),
            input_constants: false,
            input_nodes: false,
            output_nodes: false,
            graph_outputs: false,
            graph_inputs: true,
            channel: None,
        }
    }
    /// Sets the source of a Connection. Only valid for Connection::Node and
    /// Connection::GraphOutput. On other variants it does nothing.
    pub fn from(mut self, source_node: NodeId) -> Self {
        match &mut self {
            Connection::Node { ref mut source, .. } => {
                *source = source_node;
            }
            Connection::GraphInput { .. } => {}
            Connection::Constant { .. } => {}
            Connection::GraphOutput { ref mut source, .. } => {
                *source = source_node.clone();
            }
            Connection::Clear { .. } => {}
            Connection::GraphInputToOutput { .. } => {}
        }
        self
    }
    /// Sets the source of a Connection. Does nothing on a
    /// Connection::GraphOutput or Connection::Clear.
    pub fn to(mut self, sink_node: NodeId) -> Self {
        match &mut self {
            Connection::Node { ref mut sink, .. } => {
                *sink = sink_node;
            }
            Connection::GraphInput { ref mut sink, .. } => {
                *sink = sink_node;
            }
            Connection::Constant { ref mut sink, .. } => {
                *sink = Some(sink_node);
            }
            Connection::GraphOutput { .. } => {}
            Connection::Clear { .. } => {}
            Connection::GraphInputToOutput { .. } => {}
        }
        self
    }
    /// Set the channel to connect to by label
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
            Connection::GraphInputToOutput { .. } => {}
            Connection::GraphInput {
                to_label: input_label,
                to_index,
                ..
            } => {
                *input_label = Some(label);
                *to_index = None;
            }
            Connection::Clear { channel, .. } => *channel = Some(NodeChannel::Label(label)),
        }
        self
    }
    /// Shortcut for `to_label`
    pub fn tl(self, label: &'static str) -> Self {
        self.to_label(label)
    }
    /// Set the channel to connect to by index
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
            Connection::GraphInputToOutput {
                to_output_channel, ..
            } => *to_output_channel = index,
            Connection::Constant {
                to_index: input_index,
                ..
            } => {
                *input_index = Some(index);
            }
            Connection::GraphOutput { to_index, .. } => {
                *to_index = index;
            }
            Connection::Clear { channel, .. } => *channel = Some(NodeChannel::Index(index)),
            Connection::GraphInput {
                to_index, to_label, ..
            } => {
                *to_index = Some(index);
                *to_label = None;
            }
        }
        self
    }
    /// Get the source channel index of the [`Connection`] if one is set
    pub fn get_from_index(&self) -> Option<usize> {
        match &self {
            Connection::Node { from_index, .. } => *from_index,
            Connection::GraphOutput { from_index, .. } => *from_index,
            Connection::Clear { .. } => None,
            Connection::Constant { .. } => None,
            Connection::GraphInput { from_index, .. } => Some(*from_index),
            Connection::GraphInputToOutput {
                from_input_channel, ..
            } => Some(*from_input_channel),
        }
    }
    /// Get the sink channel index of the [`Connection`] if one is set
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
            Connection::GraphInputToOutput {
                to_output_channel, ..
            } => Some(*to_output_channel),
            Connection::Clear { .. } => None,
            Connection::GraphInput { to_index, .. } => *to_index,
        }
    }
    /// Shortcut for `to_index`
    pub fn ti(self, index: usize) -> Self {
        self.to_index(index)
    }
    /// Set the source channel
    pub fn from_channel(self, channel: impl Into<NodeChannel>) -> Self {
        match channel.into() {
            NodeChannel::Index(index) => self.from_index(index),
            NodeChannel::Label(label) => self.from_label(label),
        }
    }
    /// Set the sink channel
    pub fn to_channel(self, channel: impl Into<NodeChannel>) -> Self {
        match channel.into() {
            NodeChannel::Index(index) => self.to_index(index),
            NodeChannel::Label(label) => self.to_label(label),
        }
    }
    /// Set the source channel by index
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
            Connection::Clear { channel, .. } => *channel = Some(NodeChannel::Index(index)),
            Connection::GraphInputToOutput {
                from_input_channel, ..
            } => *from_input_channel = index,
        }
        self
    }
    /// Set the source channel by label
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
            Connection::GraphInputToOutput { .. } => {}
            Connection::Clear { channel, .. } => *channel = Some(NodeChannel::Label(label)),
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
            Connection::GraphInputToOutput { channels, .. } => {
                *channels = num_channels;
            }
            Connection::Clear { .. } => {}
        }
        self
    }
    /// Set if the connection should be a feedback connection
    pub fn feedback(mut self, activate: bool) -> Self {
        match &mut self {
            Connection::Node { feedback, .. } => {
                *feedback = activate;
            }
            Connection::Constant { .. } => {}
            Connection::GraphOutput { .. } => {}
            Connection::GraphInput { .. } => {}
            Connection::GraphInputToOutput { .. } => {}
            Connection::Clear { .. } => {}
        }
        self
    }
    /// Get the source node address if one is set and the current variant has one.
    pub fn get_source_node(&self) -> Option<NodeId> {
        match self {
            Connection::Node { source, .. } => Some(source.clone()),
            Connection::GraphOutput { source, .. } => Some(source.clone()),
            Connection::GraphInput { .. }
            | Connection::Constant { .. }
            | Connection::Clear { .. }
            | Connection::GraphInputToOutput { .. } => None,
        }
    }
}

#[allow(missing_docs)]
/// Error making a connection inside a [`Graph`]
#[derive(thiserror::Error, Debug, PartialEq)]
pub enum ConnectionError {
    #[error("The nodes that you are trying to connect are in different graphs. Nodes can only be connected within a graph.")]
    DifferentGraphs,
    #[error("The graph containing the NodeAdress provided was not found. The node itself may or may not exist. Connection: {0}")]
    GraphNotFound(Connection),
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

/// Describe a node's input or output channel by index or label
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NodeChannel {
    Label(&'static str),
    Index(usize),
}
impl From<&'static str> for NodeChannel {
    fn from(value: &'static str) -> Self {
        Self::Label(value)
    }
}
impl From<usize> for NodeChannel {
    fn from(value: usize) -> Self {
        Self::Index(value)
    }
}

/// Value for an input constant.
#[derive(Clone, Copy, Debug)]
pub struct Constant(Sample);
impl From<Sample> for Constant {
    fn from(value: Sample) -> Self {
        Self(value)
    }
}

/// A specific output from a specific node.
#[derive(Clone, Debug)]
pub struct NodeInput {
    pub(super) node: NodeId,
    pub(super) channel: NodeChannel,
}
impl From<(NodeId, NodeChannel)> for NodeInput {
    fn from(value: (NodeId, NodeChannel)) -> Self {
        Self {
            node: value.0,
            channel: value.1,
        }
    }
}

/// A specific output from a specific node.
#[derive(Clone, Debug)]
pub struct NodeOutput {
    pub(super) from_node: NodeId,
    pub(super) from_channel: NodeChannel,
}
impl From<(NodeId, NodeChannel)> for NodeOutput {
    fn from(value: (NodeId, NodeChannel)) -> Self {
        Self {
            from_node: value.0,
            from_channel: value.1,
        }
    }
}
impl NodeOutput {
    /// Create a connection from self to the specified node.
    pub fn to_node(&self, node: NodeId) -> Connection {
        let connection = Connection::Node {
            source: self.from_node.clone(),
            from_index: None,
            from_label: None,
            sink: node.clone(),
            to_index: None,
            to_label: None,
            channels: 1,
            feedback: false,
        };
        match self.from_channel {
            NodeChannel::Label(label) => connection.from_label(label),
            NodeChannel::Index(index) => connection.from_index(index),
        }
    }
    /// Create a connection from self to the graph output of the graph the node is in.
    pub fn to_graph_out(&self) -> Connection {
        let connection = Connection::GraphOutput {
            source: self.from_node.clone(),
            from_index: Some(0),
            from_label: None,
            to_index: 0,
            channels: 1,
        };
        match self.from_channel {
            NodeChannel::Label(label) => connection.from_label(label),
            NodeChannel::Index(index) => connection.from_index(index),
        }
    }
}
impl IntoIterator for NodeOutput {
    type Item = NodeOutput;

    type IntoIter = NodeOutputIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        NodeOutputIntoIter { no: Some(self) }
    }
}
/// Iterator for one NodeOutput. Will return one [`NodeOutput`], then `None`
pub struct NodeOutputIntoIter {
    no: Option<NodeOutput>,
}
impl Iterator for NodeOutputIntoIter {
    type Item = NodeOutput;

    fn next(&mut self) -> Option<Self::Item> {
        self.no.take()
    }
}

/// Hold either a constant value or a `NodeOutput`
#[allow(missing_docs)]
#[derive(Clone, Debug)]
pub enum ConstantOrNodeOutput {
    Constant(Constant),
    NodeOutput(NodeOutput),
}
impl From<Constant> for ConstantOrNodeOutput {
    fn from(value: Constant) -> Self {
        Self::Constant(value)
    }
}
impl From<NodeOutput> for ConstantOrNodeOutput {
    fn from(value: NodeOutput) -> Self {
        Self::NodeOutput(value)
    }
}

/// Holds descriptions of several [`Connections`] to the same node. Can be used
/// with the `input!` macro or one of several other syntax variations to connect
/// inputs to a new node.
#[derive(Clone)]
pub struct ConnectionBundle {
    to_node: NodeId,
    inputs: Vec<(NodeChannel, ConstantOrNodeOutput)>,
}

impl ConnectionBundle {
    /// Convert to [`Connection`]s. Returns None if the node whose inputs to set
    /// has not been specified.
    pub fn as_connections(&self) -> Vec<Connection> {
        let mut connections = Vec::with_capacity(self.inputs.len());
        for (channel, input) in &self.inputs {
            connections.push(match input {
                ConstantOrNodeOutput::Constant(con) => {
                    constant(con.0).to(self.to_node).to_channel(*channel)
                }
                ConstantOrNodeOutput::NodeOutput(no) => {
                    no.to_node(self.to_node).to_channel(*channel)
                }
            });
        }
        connections
    }
}

/// `InputBundle` allows you to build connections using a more intuitive less
/// verbose syntax. Can be converted to a [`ConnectionBundle`] by setting the
/// sink node using the [`Self::to`] method.
#[derive(Clone)]
pub struct InputBundle {
    inputs: Vec<(NodeChannel, ConstantOrNodeOutput)>,
}
impl InputBundle {
    /// New empty [`Self`]
    pub fn new() -> Self {
        Self { inputs: Vec::new() }
    }
    /// Set the sink node to generate a [`ConnectionBundle`]
    pub fn to(self, node_address: NodeId) -> ConnectionBundle {
        ConnectionBundle {
            to_node: node_address.clone(),
            inputs: self.inputs,
        }
    }
    /// Add any number of node outputs as inputs to the channel specified.
    pub fn push_node_outputs(
        &mut self,
        channel: NodeChannel,
        node_outputs: impl IntoIterator<Item = NodeOutput>,
    ) {
        for no in node_outputs {
            self.inputs
                .push((channel, ConstantOrNodeOutput::NodeOutput(no)));
        }
    }
    /// Add an input from a source to the `input_channel`
    pub fn push_input(&mut self, input_channel: NodeChannel, source: ConstantOrNodeOutput) {
        self.inputs.push((input_channel, source));
    }
    /// Combine two [`Self`] into one
    pub fn extend(mut self, other: Self) -> Self {
        self.inputs.extend(other.inputs);
        self
    }
}

impl<const N: usize> From<[InputBundle; N]> for InputBundle {
    fn from(array_of_bundles: [InputBundle; N]) -> Self {
        let vec: Vec<InputBundle> = array_of_bundles.into_iter().collect();
        vec.into()
    }
}
impl<const N: usize> From<[(&'static str, Sample); N]> for InputBundle {
    fn from(array_of_bundles: [(&'static str, Sample); N]) -> Self {
        let vec: Vec<InputBundle> = array_of_bundles
            .into_iter()
            .map(|(label, con)| (label, con).into())
            .collect();
        vec.into()
    }
}
impl<const N: usize> From<[(usize, Sample); N]> for InputBundle {
    fn from(array_of_bundles: [(usize, Sample); N]) -> Self {
        let vec: Vec<InputBundle> = array_of_bundles
            .into_iter()
            .map(|(index, con)| (index, con).into())
            .collect();
        vec.into()
    }
}
impl<const N: usize, O: IntoIterator<Item = NodeOutput>> From<[(usize, Sample, O); N]>
    for InputBundle
{
    fn from(array_of_bundles: [(usize, Sample, O); N]) -> Self {
        let vec: Vec<InputBundle> = array_of_bundles
            .into_iter()
            .map(|(index, con, output)| (index, con, output).into())
            .collect();
        vec.into()
    }
}
impl<const N: usize, O: IntoIterator<Item = NodeOutput>> From<[(&'static str, Sample, O); N]>
    for InputBundle
{
    fn from(array_of_bundles: [(&'static str, Sample, O); N]) -> Self {
        let vec: Vec<InputBundle> = array_of_bundles
            .into_iter()
            .map(|(label, con, output)| (label, con, output).into())
            .collect();
        vec.into()
    }
}

impl From<Vec<InputBundle>> for InputBundle {
    fn from(vec_of_bundles: Vec<InputBundle>) -> Self {
        let inputs = vec_of_bundles
            .iter()
            .map(|cb| cb.inputs.clone())
            .flatten()
            .collect();
        Self { inputs }
    }
}
impl From<(&'static str, Sample)> for InputBundle {
    fn from((label, c): (&'static str, Sample)) -> Self {
        Self {
            inputs: vec![(label.into(), Constant(c).into())],
        }
    }
}
impl From<(usize, Sample)> for InputBundle {
    fn from((index, c): (usize, Sample)) -> Self {
        Self {
            inputs: vec![(index.into(), Constant(c).into())],
        }
    }
}
// impl<O: IntoIterator<Item = NodeOutput>> From<(&'static str, O)> for ConnectionBundle {
//     fn from((label, constant, node_outputs): (&'static str, O)) -> Self {
//         let mut inputs = Vec::new();
//         for no in node_outputs {
//             inputs.push((label.into(), no.into()));
//         }
//         Self {
//             to_node: None,
//             inputs,
//         }
//     }
// }
// impl<O: IntoIterator<Item = NodeOutput>> From<(usize, O)> for ConnectionBundle {
//     fn from((index, constant, node_outputs): (usize, O)) -> Self {
//         let mut inputs = Vec::new();
//         for no in node_outputs {
//             inputs.push((index.into(), no.into()));
//         }
//         Self {
//             to_node: None,
//             inputs,
//         }
//     }
// }
impl<O: IntoIterator<Item = NodeOutput>> From<(&'static str, Sample, O)> for InputBundle {
    fn from((label, constant, node_outputs): (&'static str, Sample, O)) -> Self {
        let mut inputs = Vec::new();
        inputs.push((label.into(), Constant(constant).into()));
        for no in node_outputs {
            inputs.push((label.into(), no.into()));
        }
        Self { inputs }
    }
}
impl<O: IntoIterator<Item = NodeOutput>> From<(usize, Sample, O)> for InputBundle {
    fn from((index, constant, node_outputs): (usize, Sample, O)) -> Self {
        let mut inputs = Vec::new();
        inputs.push((index.into(), Constant(constant).into()));
        for no in node_outputs {
            inputs.push((index.into(), no.into()));
        }
        Self { inputs }
    }
}
// impl<
//         L: IntoIterator<Item = &'static str>,
//         S: IntoIterator<Item = Sample>,
//         N: IntoIterator<Item = NodeOutput>,
//     > From<(L, S, N)> for ConnectionBundle
// {
//     fn from((labels, constants, node_outputs): (L, S, N)) -> Self {
//         let mut inputs: Vec<(NodeChannel, ConstantOrNodeOutput)> = Vec::new();
//         let labels = labels.into_iter().collect::<Vec<_>>();
//         let constants = constants.into_iter().collect::<Vec<_>>();
//         let node_outputs = node_outputs.into_iter().collect::<Vec<_>>();
//         let longest = labels.len().max(constants.len()).max(node_outputs.len());
//         let mut labels = labels.into_iter().cycle();
//         let mut constants = constants.into_iter().cycle();
//         let mut node_outputs = node_outputs.into_iter().cycle();
//         for _i in 0..longest {
//             let label = labels.next().unwrap();
//             inputs.push((label.into(), Constant(constants.next().unwrap()).into()));
//             inputs.push((label.into(), node_outputs.next().unwrap().into()));
//         }
//         Self {
//             to_node: None,
//             inputs,
//         }
//     }
// }
// impl<
//         L: IntoIterator<Item = NodeChannel>,
//         S: IntoIterator<Item = Sample>,
//         N: IntoIterator<Item = NodeOutput>,
//     > From<(L, S, N)> for ConnectionBundle
// {
//     fn from((labels, constants, node_outputs): (L, S, N)) -> Self {
//         let mut inputs: Vec<(NodeChannel, ConstantOrNodeOutput)> = Vec::new();
//         let labels = labels.into_iter().collect::<Vec<_>>();
//         let constants = constants.into_iter().collect::<Vec<_>>();
//         let node_outputs = node_outputs.into_iter().collect::<Vec<_>>();
//         let longest = labels.len().max(constants.len()).max(node_outputs.len());
//         let mut labels = labels.into_iter().cycle();
//         let mut constants = constants.into_iter().cycle();
//         let mut node_outputs = node_outputs.into_iter().cycle();
//         for _i in 0..longest {
//             let label = labels.next().unwrap();
//             inputs.push((label.into(), Constant(constants.next().unwrap()).into()));
//             inputs.push((label.into(), node_outputs.next().unwrap().into()));
//         }
//         Self {
//             to_node: None,
//             inputs,
//         }
//     }
// }
// impl<C: IntoIterator<Item = &'static str>, S: IntoIterator<Item = Sample>> From<(C, S)>
//     for ConnectionBundle
// {
//     fn from((labels, constants): (C, S)) -> Self {
//         let mut inputs: Vec<(NodeChannel, ConstantOrNodeOutput)> = Vec::new();
//         let labels = labels.into_iter().collect::<Vec<_>>();
//         let constants = constants.into_iter().collect::<Vec<_>>();
//         let longest = labels.len().max(constants.len());
//         let mut labels = labels.into_iter().cycle();
//         let mut constants = constants.into_iter().cycle();
//         for _i in 0..longest {
//             let label = labels.next().unwrap();
//             inputs.push((label.into(), Constant(constants.next().unwrap()).into()));
//         }
//         Self {
//             to_node: None,
//             inputs,
//         }
//     }
// }
// impl<
//         C: IntoIterator<Item = usize>,
//         S: IntoIterator<Item = Sample>,
//         N: IntoIterator<Item = NodeOutput>,
//     > From<(C, S, N)> for ConnectionBundle
// {
//     fn from((labels, constants, node_outputs): (C, S, N)) -> Self {
//         let mut inputs: Vec<(NodeChannel, ConstantOrNodeOutput)> = Vec::new();
//         let indices = labels.into_iter().collect::<Vec<_>>();
//         let constants = constants.into_iter().collect::<Vec<_>>();
//         let node_outputs = node_outputs.into_iter().collect::<Vec<_>>();
//         let longest = indices.len().max(constants.len()).max(node_outputs.len());
//         let mut indices = indices.into_iter().cycle();
//         let mut constants = constants.into_iter().cycle();
//         let mut node_outputs = node_outputs.into_iter().cycle();
//         for _i in 0..longest {
//             let index = indices.next().unwrap();
//             inputs.push((index.into(), Constant(constants.next().unwrap()).into()));
//             inputs.push((index.into(), node_outputs.next().unwrap().into()));
//         }
//         Self {
//             to_node: None,
//             inputs,
//         }
//     }
// }
//
//

// We cannot allow the pattern (&'static str, IntoIterator<Item = NodeOutput>)
// because "f32 might implement IntoIterator in the future", but maybe we can
// make a macro instead.
/// Create an [`InputBundle`] using the syntax
///
/// ```
/// # use knyst::{inputs, graph::connection::InputBundle};
/// # let from_channel = 0;
/// # let input_constant = 1.0;
/// # let input_node_outputs = vec![];
/// // Connect from channel 0
/// let input_bundle = inputs![(0: input_constant ; input_node_outputs)];
/// ```
///
/// where both `input_constant` and `input_node_outputs` are optional.
/// Multiple tuples can be put in the same `inputs!` invocation.
#[macro_export]
macro_rules! inputs {
    (($channel:literal : $constant:expr)) => {
        InputBundle::from(($channel, $constant))
    };
    (($channel:literal : $constant:expr), $($y:tt), +) => {
        InputBundle::from(($channel, $constant)).extend(inputs!($($y),+))
    };
    (($channel:literal : $constant:expr; $nodes:expr)) => {
        InputBundle::from(($channel, $constant, $nodes))
    };
    (($channel:literal : $constant:expr; $nodes:expr), $($y:tt), +) => {
        InputBundle::from(($channel, $constant, $nodes)).extend(inputs!($($y),+))
    };
    (($channel:literal ; $nodes:expr)) => {
        {
            let mut c = InputBundle::new();
            c.push_node_outputs(($channel).into(), $nodes);
            c
        }
    };
    (($channel:literal ; $nodes:expr), $($y:tt), +) => {
        {
            let mut c = InputBundle::new();
            c.push_node_outputs(($channel).into(), $nodes);
            c.extend(inputs!($($y),+))
        }
    };
    () => {
        InputBundle::new()
    };
}

#[cfg(test)]
mod tests {
    use super::InputBundle;
    use crate::graph::NodeId;

    #[test]
    fn connection_bundles() {
        struct DNode;
        fn push_node(_node: DNode, defaults: impl Into<InputBundle>) -> NodeId {
            let _d = defaults.into();
            NodeId::new()
        }
        fn connect(b: impl Into<InputBundle>) {
            let b = b.into();
            let target_node = NodeId::new();
            b.to(target_node).as_connections();
        }

        let node = NodeId::new();
        let _c: InputBundle = ("freq", 440.0, [node.out(0)]).into();
        let _c: InputBundle = [("freq", 440.0, [node.out(0)])].into();
        connect(vec![
            ("freq", 440.0, [node.out(0), node.out(1)]).into(),
            (1, 0.0).into(),
        ]);
        connect(vec![
            ("freq", 440.0, [node.out(0), node.out(1)]).into(),
            ("phase", 0.0, [node.out(2)]).into(),
            (1, 0.0).into(),
        ]);
        connect(vec![
            ("freq", 440.0, [node.out(0)]).into(),
            ("phase", 0.0).into(),
            ("amp", 0.5).into(),
        ]);
        connect(("freq", 440.0, [node.out(0)]));
        connect((0, 440.0, [node.out(0)]));
        connect((0, 440.0));
        connect(inputs!((0 : 440.0)));
        connect(inputs!(("freq" : 440.; [node.out(0)])));
        connect(inputs!(
            ("freq" : 440. ; [node.out(0)]),
            (1 : 0.0),
            ("amp" ; [node.out(2), node.out(3)])
        ));
        connect(inputs!((0 : 440.0), (1 : 110.)));
        connect(inputs![(0 : 440.0), ("freq" : 440. ; [node.out(0)])]);
        connect([
            ("freq", 440.0),
            ("phase", 0.0),
            ("amp", 0.5),
            ("weirdness", 0.1),
        ]);
        connect([(0, 440.0), (1, 0.0), (2, 0.5), (3, 0.1)]);
        connect([
            ("freq", 440.0, [node.out(0)]),
            ("phase", 0.0, [node.out(1)]),
            ("amp", 0.5, [node.out(2)]),
            ("weirdness", 0.1, [node.out(3)]),
        ]);
        let node = push_node(DNode, [("freq", 440.), ("phase", 0.)]);
        let node = push_node(
            DNode,
            vec![("freq", 0.0, [node.out(0)]).into(), ("phase", 0.0).into()],
        );
        connect([(0, 0.0, [node.out(0)]), (1, 0.0, [node.out(0)])]);
    }
}
