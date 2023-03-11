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

#[derive(Clone, Copy, Debug)]
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

#[derive(Clone, Copy, Debug)]
pub struct Constant(Sample);
impl From<Sample> for Constant {
    fn from(value: Sample) -> Self {
        Self(value)
    }
}

#[derive(Clone, Debug)]
pub struct NodeOutput {
    pub(super) from_node: NodeAddress,
    pub(super) from_channel: NodeChannel,
}

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

/// `ConenctionBundle` allows you to build connections using a more intuitive
/// less verbose syntax.
#[derive(Clone)]
pub struct ConnectionBundle {
    to_node: Option<NodeAddress>,
    inputs: Vec<(NodeChannel, ConstantOrNodeOutput)>,
}
impl ConnectionBundle {
    fn new() -> Self {
        Self {
            to_node: None,
            inputs: Vec::new(),
        }
    }
    pub fn to(&mut self, node_address: &NodeAddress) {
        self.to_node = Some(node_address.clone());
    }
    pub fn push_input(&mut self, input_channel: NodeChannel, source: ConstantOrNodeOutput) {
        self.inputs.push((input_channel, source));
    }
    pub fn as_connections(&self) -> Vec<Connection> {
        todo!()
    }
}

impl<const N: usize> From<[ConnectionBundle; N]> for ConnectionBundle {
    fn from(array_of_bundles: [ConnectionBundle; N]) -> Self {
        let vec: Vec<ConnectionBundle> = array_of_bundles.into_iter().collect();
        vec.into()
    }
}
impl<const N: usize> From<[(&'static str, Sample); N]> for ConnectionBundle {
    fn from(array_of_bundles: [(&'static str, Sample); N]) -> Self {
        let vec: Vec<ConnectionBundle> = array_of_bundles
            .into_iter()
            .map(|(label, con)| (label, con).into())
            .collect();
        vec.into()
    }
}
impl<const N: usize> From<[(usize, Sample); N]> for ConnectionBundle {
    fn from(array_of_bundles: [(usize, Sample); N]) -> Self {
        let vec: Vec<ConnectionBundle> = array_of_bundles
            .into_iter()
            .map(|(index, con)| (index, con).into())
            .collect();
        vec.into()
    }
}
impl<const N: usize> From<[(&'static str, Sample, NodeOutput); N]> for ConnectionBundle {
    fn from(array_of_bundles: [(&'static str, Sample, NodeOutput); N]) -> Self {
        let vec: Vec<ConnectionBundle> = array_of_bundles
            .into_iter()
            .map(|(label, con, output)| (label, con, output).into())
            .collect();
        vec.into()
    }
}

impl From<Vec<ConnectionBundle>> for ConnectionBundle {
    fn from(vec_of_bundles: Vec<ConnectionBundle>) -> Self {
        let inputs = vec_of_bundles
            .iter()
            .map(|cb| cb.inputs.clone())
            .flatten()
            .collect();
        let to_node = vec_of_bundles
            .iter()
            .map(|cn| &cn.to_node)
            .filter(|tn| tn.is_some())
            .last()
            .clone()
            .unwrap_or(&None);
        Self {
            to_node: to_node.clone(),
            inputs,
        }
    }
}
impl From<(&'static str, Sample)> for ConnectionBundle {
    fn from((label, c): (&'static str, Sample)) -> Self {
        Self {
            to_node: None,
            inputs: vec![(label.into(), Constant(c).into())],
        }
    }
}
impl From<(usize, Sample)> for ConnectionBundle {
    fn from((index, c): (usize, Sample)) -> Self {
        Self {
            to_node: None,
            inputs: vec![(index.into(), Constant(c).into())],
        }
    }
}
impl From<(&'static str, Sample, NodeOutput)> for ConnectionBundle {
    fn from((label, constant, node_output): (&'static str, Sample, NodeOutput)) -> Self {
        Self {
            to_node: None,
            inputs: vec![
                (label.into(), Constant(constant).into()),
                (label.into(), node_output.into()),
            ],
        }
    }
}
impl From<(usize, Sample, NodeOutput)> for ConnectionBundle {
    fn from((index, constant, node_output): (usize, Sample, NodeOutput)) -> Self {
        Self {
            to_node: None,
            inputs: vec![
                (index.into(), Constant(constant).into()),
                (index.into(), node_output.into()),
            ],
        }
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
impl<
        L: IntoIterator<Item = NodeChannel>,
        S: IntoIterator<Item = Sample>,
        N: IntoIterator<Item = NodeOutput>,
    > From<(L, S, N)> for ConnectionBundle
{
    fn from((labels, constants, node_outputs): (L, S, N)) -> Self {
        let mut inputs: Vec<(NodeChannel, ConstantOrNodeOutput)> = Vec::new();
        let labels = labels.into_iter().collect::<Vec<_>>();
        let constants = constants.into_iter().collect::<Vec<_>>();
        let node_outputs = node_outputs.into_iter().collect::<Vec<_>>();
        let longest = labels.len().max(constants.len()).max(node_outputs.len());
        let mut labels = labels.into_iter().cycle();
        let mut constants = constants.into_iter().cycle();
        let mut node_outputs = node_outputs.into_iter().cycle();
        for _i in 0..longest {
            let label = labels.next().unwrap();
            inputs.push((label.into(), Constant(constants.next().unwrap()).into()));
            inputs.push((label.into(), node_outputs.next().unwrap().into()));
        }
        Self {
            to_node: None,
            inputs,
        }
    }
}
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
pub struct ConnectionBuilder {
    channel: NodeChannel,
    constant: Option<Sample>,
    node_outputs: Vec<NodeOutput>,
}
impl ConnectionBuilder {
    pub fn new(channel: impl Into<NodeChannel>) -> Self {
        Self {
            channel: channel.into(),
            constant: None,
            node_outputs: vec![],
        }
    }
    pub fn constant(mut self, v: Sample) -> Self {
        self.constant = Some(v);
        self
    }
    pub fn node(mut self, n: NodeOutput) -> Self {
        self.node_outputs.push(n);
        self
    }
}
impl From<ConnectionBuilder> for ConnectionBundle {
    fn from(v: ConnectionBuilder) -> Self {
        let mut cb = ConnectionBundle::new();
        if let Some(c) = v.constant {
            cb.push_input(v.channel, ConstantOrNodeOutput::Constant(c.into()));
        }
        for no in v.node_outputs {
            cb.push_input(v.channel, ConstantOrNodeOutput::NodeOutput(no));
        }
        cb
    }
}
// impl From<Vec<ConnectionBuilder>> for ConnectionBundle {
//     fn from(v: Vec<ConnectionBuilder>) -> Self {
//         todo!()
//     }
// }

mod tests {
    use crate::{
        graph::{
            connection::{ConnectionBuilder, NodeChannel, NodeOutput},
            NodeAddress,
        },
        Sample,
    };

    use super::ConnectionBundle;

    #[test]
    fn connection_bundles() {
        struct DNode;
        fn push_node(_node: DNode, defaults: impl Into<ConnectionBundle>) -> NodeAddress {
            let _d = defaults.into();
            NodeAddress::new()
        }
        fn connect(b: impl Into<ConnectionBundle>) {
            let _b = b.into();
        }

        fn i(chan: impl Into<NodeChannel>) -> ConnectionBuilder {
            ConnectionBuilder::new(chan)
        }
        // let _c: ConnectionBundle = ("freq", 440.0).into();
        let node = NodeAddress::new();
        let _c: ConnectionBundle = ("freq", 440.0, node.out(0)).into();
        let _c: ConnectionBundle = [("freq", 440.0, node.out(0))].into();
        // let _c: ConnectionBundle = [("freq", 440.0, node.out(0)).into()].into();
        //
        // connect(vec![("freq", 440.0, node.out(0)).into(), (1, 0.0).into()]);
        connect(vec![
            ("freq", 440.0, node.out(0)).into(),
            ("phase", 0.0).into(),
            ("amp", 0.5).into(),
        ]);
        connect(("freq", 440.0, node.out(0)));
        connect((0, 440.0, node.out(0)));
        connect((0, 440.0));
        // connect(vec![
        //     i("freq").constant(440.).node(node.out(0)),
        //     i(2).node(node.out(1)),
        //     i("phase").constant(0.0),
        // ]);
        // connect([
        //     ("freq", 440.0, node.out(0)).into(),
        //     ("phase", 0.0).into(),
        //     ("amp", 0.5).into(),
        //     ("weirdness", 0.1).into(),
        // ]);
        // let _c: ConnectionBundle = [
        //     ("freq", 440.0),
        //     ("phase", 0.0),
        //     ("amp", 0.5),
        //     ("weirdness", 0.1),
        // ]
        // .into();
        connect([
            ("freq", 440.0),
            ("phase", 0.0),
            ("amp", 0.5),
            ("weirdness", 0.1),
        ]);
        connect([(0, 440.0), (1, 0.0), (2, 0.5), (3, 0.1)]);
        connect([
            ("freq", 440.0, node.out(0)),
            ("phase", 0.0, node.out(1)),
            ("amp", 0.5, node.out(2)),
            ("weirdness", 0.1, node.out(3)),
        ]);
        let node = push_node(DNode, [("freq", 440.), ("phase", 0.)]);
        let node = push_node(
            DNode,
            vec![("freq", 0.0, node.out(0)).into(), ("phase", 0.0).into()],
        );
        // let node = push_node(
        //     DNode,
        //     vec![
        //         i("freq").constant(0.0).node(node.out(0)),
        //         i("phase").constant(0.5),
        //     ],
        // );
        // let _c: ConnectionBundle = (["freq", "phase"], [440.0, 0.0]).into();
        // (Iter<NodeChannel>, Iter<Sample>, Iter<NodeOutput>)
        let _c: ConnectionBundle = (["freq".into()], [440.0], []).into();
        let _c: ConnectionBundle = ([0.into(), 1.into()], [0.0], [node.out(0)]).into();
        // (Iter<&'static str>, Iter<Sample>, Iter<NodeOutput>)
        // let _c: ConnectionBundle =
        //     (["freq", "phase"], [440.0, 0.0], [node.out(0), node.out(1)]).into();
        // // Send a node.out(0) to both inputs
        // (Iter<usize>, Iter<Sample>, Iter<NodeOutput>)
        // let _c: ConnectionBundle = ([0, 1], [0.0], [node.out(0)]).into();
    }
}
