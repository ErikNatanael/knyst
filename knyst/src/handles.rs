//! Handles are the preferred and most ergonomic way of interacting with Knyst, and may
//! become the only way in the future.
//!
//! ## With your own types
//! For your own [`Gen`]s, the [`impl_gen`] macro will automatically
//! create a handle type for you if you include a `new` function in the impl_gen block. You can also create
//! a handle type yourself by implementing [`HandleData`].
//!
//! If a Gen you want to use does not have a handle type, you can use the [`handle`] function to upload
//! it to the current graph and get a [`GenericHandle`] to it. This handle type can be used in routing, but doesn't
//! have all the type safety features that a custom handle type has, e.g. setting inputs on specifucally named methods.
//!
//!
//! # Example
//! ```
//! use knyst::prelude::*;
//! use knyst::offline::*;
//!
//! let mut kt = KnystOffline::new(128, 64, 0, 1);
//!
//! // Upload a wavetable oscillator to the current graph and set its inputs.
//! let sig = oscillator(WavetableId::cos()).freq(440.);
//! // Create another one
//! let modulator = oscillator(WavetableId::cos()).freq(440.);
//! // Change the input of the first oscillator. You can do basic arithmetic on handles and
//! // the necessary Gens will be added automatically.
//! sig.freq(modulator * 440.);
//! // Lower the amplitude by half. The result of an arithmetic operation is a new
//! // handle representing the outermost operation in the chain, in this case the multiplication.
//! // To still have access to the settings on the inner handles those handles have to be stored.
//! let sig = sig * 0.5;
//! // Output the signal to the graph output. Since we are working on the outermost graph, this is the
//! // output of all of Knyst. The first argument determines what channel index to start outputting to. All
//! // output channels in the `sig` will be output, in this case only 1.
//! graph_output(0, sig);
//!
//! ```

use std::{
    any::Any,
    ops::{Add, Deref, Mul, Sub},
};

use crate::{
    graph::{Change, Connection, GraphId, ParameterChange, SimultaneousChanges},
    prelude::{PowfGen, PowfHandle, SubGen},
    Sample,
};

use crate::{
    graph::{connection::NodeChannel, GenOrGraph, NodeId},
    modal_interface::knyst_commands,
    prelude::{Bus, KnystCommands, MulGen, RangeGen},
};

#[allow(unused)]
use crate::gen::Gen;
#[allow(unused)]
use knyst_macro::impl_gen;

/// Handle
#[derive(Copy, Clone, Debug)]
pub struct Handle<H: Copy> {
    handle: H,
}

impl<H: Copy> Deref for Handle<H> {
    type Target = H;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}
impl Into<GenericHandle> for GraphHandle {
    fn into(self) -> GenericHandle {
        GenericHandle {
            node_id: self.node_id,
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
        }
    }
}
// impl<A: Copy + Into<NodeId>> From<NodeHandle<A>> for NodeId {
//     fn from(value: NodeHandle<A>) -> Self {
//         value.handle.into()
//     }
// }
impl Handle<GenericHandle> {
    /// Create a dummy handle pointing to nothing.
    pub fn void() -> Handle<GenericHandle> {
        Self::new(GenericHandle {
            node_id: NodeId::new(u64::MAX),
            num_inputs: 0,
            num_outputs: 0,
        })
    }
}
impl<A: Copy + HandleData> Handle<A> {
    /// Create a new `Handle` wrapping a specific handle type
    pub fn new(handle: A) -> Self {
        Self { handle }
    }
    /// Repeat all the outputs such that [1, 2, 3] -> [1, 1, 2, 2, 3, 3]
    pub fn repeat_outputs(self, n: usize) -> Handle<RepeatOutputs<A>> {
        Handle::new(RepeatOutputs {
            handle: self.handle,
            repeats: n,
        })
    }
    /// Repeat all the outputs such that [1, 2, 3] -> [1, 1, 2, 2, 3, 3]
    pub fn channels(self, n: usize) -> Handle<Channels<A>> {
        Handle::new(Channels {
            handle: self.handle,
            channels: n,
        })
    }
    /// Take all the output channels to the power of the given exponent
    pub fn powf(self, exponent: impl Into<Input>) -> Handle<PowfHandle> {
        let connecting_channels: Vec<_> = self.out_channels().collect();
        let num_channels = connecting_channels.len();
        let node_id = knyst_commands().push_without_inputs(PowfGen(connecting_channels.len()));
        for (i, (source, chan)) in connecting_channels.into_iter().enumerate() {
            knyst_commands().connect(source.to(node_id).from_channel(chan).to_channel(i + 1));
        }
        Handle::new(PowfHandle {
            node_id,
            num_channels,
        })
        .exponent(exponent)
    }
    /// The non-typed way to set an input channel's value to a constant and/or a handle.
    /// Multiple calls to set the input will add the inputs together, except for constant inputs of which
    /// there can only be one.
    ///
    /// NB: If an index is used for the `channel`, all the channels in `input` will be connected sequentially. If
    /// a label is used, all the channels in the input will be connected to the same labelled channel.
    pub fn set(self, channel: impl Into<NodeChannel>, input: impl Into<Input>) -> Handle<A> {
        let inp = input.into();
        let channel: NodeChannel = channel.into();
        // How to match channel and input_channel iterator?
        match inp {
            Input::Constant(v) => match channel {
                NodeChannel::Label(channel_label) => {
                    if let Some((Sink::Gen(id), _chan)) = self.in_channels().next() {
                        let change =
                            ParameterChange::now(id.input(channel_label), Change::Constant(v));
                        knyst_commands().schedule_change(change);
                    };
                }
                NodeChannel::Index(channel_index) => {
                    if let Some((Sink::Gen(id), chan)) = self.in_channels().nth(channel_index) {
                        let change = ParameterChange::now(id.input(chan), Change::Constant(v));
                        knyst_commands().schedule_change(change);
                    };
                }
            },
            Input::Handle { output_channels } => match channel {
                NodeChannel::Label(channel_label) => {
                    if let Some((in_sink, _)) = self.in_channels().next() {
                        for (out_source, out_chan) in output_channels {
                            knyst_commands().connect(
                                out_source
                                    .to_sink(in_sink)
                                    .from_channel(out_chan)
                                    .to_channel(channel_label),
                            );
                        }
                    }
                }
                NodeChannel::Index(channel_index) => {
                    for ((out_source, out_chan), (in_sink, in_chan)) in
                        output_channels.zip(self.in_channels().skip(channel_index))
                    {
                        knyst_commands().connect(
                            out_source
                                .to_sink(in_sink)
                                .from_channel(out_chan)
                                .to_channel(in_chan),
                        );
                    }
                }
            },
        }
        self
    }
    /// The non-typed way to send a trigger to an input channel
    pub fn trig(self, channel: impl Into<NodeChannel>) -> Handle<A> {
        let channel = channel.into();
        // TODO: better way to send a trigger
        let mut changes = SimultaneousChanges::now();
        for id in self.node_ids() {
            changes.push(id.change().trigger(channel.clone()));
        }
        knyst_commands().schedule_changes(changes);
        self
    }
    /// Sets the mortality for every referenced by the Handle
    pub fn set_mortality(self, is_mortal: bool) -> Handle<A> {
        for id in self.node_ids() {
            knyst_commands().set_mortality(id, is_mortal);
        }
        self
    }
}
impl<H: HandleData + Copy> HandleData for Handle<H> {
    fn out_channels(&self) -> SourceChannelIter {
        self.handle.out_channels()
    }

    fn in_channels(&self) -> SinkChannelIter {
        self.handle.in_channels()
    }

    fn node_ids(&self) -> NodeIdIter {
        self.handle.node_ids()
    }
}

/// Handle to a single output channel from a node.
#[derive(Copy, Clone, Debug)]
pub struct OutputChannelHandle {
    node_id: NodeId,
    channel: NodeChannel,
}
impl HandleData for OutputChannelHandle {
    fn out_channels(&self) -> SourceChannelIter {
        SourceChannelIter::single_channel(self.node_id, self.channel)
    }

    fn in_channels(&self) -> SinkChannelIter {
        SinkChannelIter::None
    }

    fn node_ids(&self) -> NodeIdIter {
        NodeIdIter::Single(self.node_id)
    }
}

/// Handle to a single output channel from a node.
#[derive(Copy, Clone, Debug)]
pub struct InputChannelHandle {
    source: Sink,
    channel: NodeChannel,
}
impl HandleData for InputChannelHandle {
    fn out_channels(&self) -> SourceChannelIter {
        SourceChannelIter::None
    }

    fn in_channels(&self) -> SinkChannelIter {
        match self.source {
            Sink::GraphOutput(graph_id) => {
                let start_index = match self.channel {
                    NodeChannel::Index(index) => index,
                    NodeChannel::Label(_) => unreachable!(),
                };
                SinkChannelIter::GraphOutput {
                    graph_id,
                    start_index,
                    num_channels: 1,
                    current_channel: 0,
                }
            }
            Sink::Gen(node_id) => SinkChannelIter::single_channel(node_id, self.channel),
        }
    }

    fn node_ids(&self) -> NodeIdIter {
        match self.source {
            Sink::GraphOutput(_graph_id) => NodeIdIter::None,
            Sink::Gen(node_id) => NodeIdIter::Single(node_id),
        }
    }
}

/// Handle for a [`Handle::channels`]. Cycles the outputs from the source handle to return a given number of handles.
///
/// # Examples
/// [1, 2].channels(4) -> [1, 2, 1, 2]
///
/// [1, 2, 3, 4, 5, 6].channels(2) -> [1, 2]
#[derive(Copy, Clone, Debug)]
pub struct Channels<H: Copy + HandleData> {
    handle: H,
    channels: usize,
}
impl<H: Copy + HandleData> HandleData for Channels<H> {
    fn out_channels(&self) -> SourceChannelIter {
        let channels = self
            .handle
            .out_channels()
            .cycle()
            .take(self.channels)
            .collect();
        SourceChannelIter::from_vec(channels)
    }

    fn in_channels(&self) -> SinkChannelIter {
        self.handle.in_channels()
    }

    fn node_ids(&self) -> NodeIdIter {
        self.handle.node_ids()
    }
}

/// Handle for a `repeat_outputs`
#[derive(Copy, Clone, Debug)]
pub struct RepeatOutputs<H: Copy + HandleData> {
    handle: H,
    repeats: usize,
}
impl<H: Copy + HandleData> HandleData for RepeatOutputs<H> {
    fn out_channels(&self) -> SourceChannelIter {
        let channels = self
            .handle
            .out_channels()
            .flat_map(|item| std::iter::repeat(item).take(self.repeats + 1))
            .collect();
        SourceChannelIter::from_vec(channels)
    }

    fn in_channels(&self) -> SinkChannelIter {
        self.handle.in_channels()
    }

    fn node_ids(&self) -> NodeIdIter {
        self.handle.node_ids()
    }
}

/// Handle for a - operation
#[derive(Copy, Clone, Debug)]
pub struct SubHandle {
    node_id: NodeId,
    num_out_channels: usize,
}
impl HandleData for SubHandle {
    fn out_channels(&self) -> SourceChannelIter {
        SourceChannelIter::SingleNodeId {
            node_id: self.node_id,
            num_channels: self.num_out_channels,
            current_iter_index: 0,
        }
    }

    fn in_channels(&self) -> SinkChannelIter {
        SinkChannelIter::SingleNodeId {
            node_id: self.node_id,
            num_channels: self.num_out_channels * 2,
            current_iter_index: 0,
        }
    }

    fn node_ids(&self) -> NodeIdIter {
        NodeIdIter::Single(self.node_id)
    }
}

/// Handle for a * operation
#[derive(Copy, Clone, Debug)]
pub struct MulHandle {
    node_id: NodeId,
    num_out_channels: usize,
}
impl HandleData for MulHandle {
    fn out_channels(&self) -> SourceChannelIter {
        SourceChannelIter::SingleNodeId {
            node_id: self.node_id,
            num_channels: self.num_out_channels,
            current_iter_index: 0,
        }
    }

    fn in_channels(&self) -> SinkChannelIter {
        SinkChannelIter::SingleNodeId {
            node_id: self.node_id,
            num_channels: self.num_out_channels * 2,
            current_iter_index: 0,
        }
    }

    fn node_ids(&self) -> NodeIdIter {
        NodeIdIter::Single(self.node_id)
    }
}

/// Handle for a + operation
#[derive(Copy, Clone, Debug)]
pub struct AddHandle {
    node_id: NodeId,
    num_out_channels: usize,
}
impl HandleData for AddHandle {
    fn out_channels(&self) -> SourceChannelIter {
        SourceChannelIter::SingleNodeId {
            node_id: self.node_id,
            num_channels: self.num_out_channels,
            current_iter_index: 0,
        }
    }

    fn in_channels(&self) -> SinkChannelIter {
        SinkChannelIter::SingleNodeId {
            node_id: self.node_id,
            num_channels: self.num_out_channels * 2,
            current_iter_index: 0,
        }
    }

    fn node_ids(&self) -> NodeIdIter {
        NodeIdIter::Single(self.node_id)
    }
}

/// Implemented by all handle types to allow routing between handles.
pub trait HandleData {
    /// All output channels of this `Handle` in order
    fn out_channels(&self) -> SourceChannelIter;
    /// All input channels of this `Handle` in order
    fn in_channels(&self) -> SinkChannelIter;
    /// All `NodeIds` referenced by this `Handle` in any order
    fn node_ids(&self) -> NodeIdIter;

    /// Remove all connections from this handle to any graph output
    fn clear_graph_output_connections(&self) {
        // TODO: Apply only to the selected channels
        for (source, _channel) in self.out_channels() {
            match source {
                Source::GraphInput(graph_id) => {
                    knyst_commands().connect(Connection::ClearGraphInputToOutput {
                        graph_id,
                        from_input_channel: None,
                        to_output_channel: None,
                        channels: None,
                    })
                }
                Source::Gen(id) => knyst_commands().connect(Connection::clear_to_graph_outputs(id)),
            }
        }
    }
    /// Remove all connections from the graph to this handle
    fn clear_graph_input_connections(&self) {
        for id in self.node_ids() {
            knyst_commands().connect(Connection::clear_from_graph_inputs(id));
        }
    }
    /// Remove all connections to inputs to this handle
    fn clear_input_connections(&self) {
        for id in self.node_ids() {
            knyst_commands().connect(Connection::clear_from_nodes(id));
        }
    }
    /// Remove all connections from outputs from this handle
    fn clear_output_connections(&self) {
        for id in self.node_ids() {
            knyst_commands().connect(Connection::clear_to_nodes(id));
        }
    }
    /// Free the node(s) this handle is pointing to
    fn free(&self) {
        for id in self.node_ids() {
            knyst_commands().free_node(id);
        }
    }
    /// Returns a handle to a single channel from this Handle (not type checked)
    fn out(&self, channel: impl Into<NodeChannel>) -> Handle<OutputChannelHandle> {
        let channel = channel.into();
        match channel {
            NodeChannel::Index(i) => {
                let (source, chan) = self.out_channels().nth(i).unwrap();
                match source {
                    Source::GraphInput(_graph_id) => todo!(),
                    Source::Gen(node_id) => Handle::new(OutputChannelHandle {
                        node_id,
                        channel: chan,
                    }),
                }
            }
            NodeChannel::Label(_name) => Handle::new(OutputChannelHandle {
                node_id: self.node_ids().next().unwrap(),
                channel,
            }),
        }
    }
    /// Returns a handle to a single channel from this Handle (not type checked)
    fn input_handle(&self, channel: impl Into<NodeChannel>) -> Handle<InputChannelHandle> {
        let channel = channel.into();
        match channel {
            NodeChannel::Index(i) => {
                let (source, chan) = self.out_channels().nth(i).unwrap();
                match source {
                    Source::GraphInput(_graph_id) => todo!(),
                    Source::Gen(node_id) => Handle::new(InputChannelHandle {
                        source: Sink::Gen(node_id),
                        channel: chan,
                    }),
                }
            }
            NodeChannel::Label(_name) => Handle::new(InputChannelHandle {
                source: Sink::Gen(self.node_ids().next().unwrap()),
                channel,
            }),
        }
    }
}
/// Iterator of `NodeId`s, e.g. for freeing a handle.
#[derive(Clone)]
#[allow(missing_docs)]
pub enum NodeIdIter {
    Single(NodeId),
    Vec(std::vec::IntoIter<NodeId>),
    None,
}
impl Iterator for NodeIdIter {
    type Item = NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            NodeIdIter::Single(node_id) => {
                let id = *node_id;
                *self = NodeIdIter::None;
                Some(id)
            }
            NodeIdIter::Vec(iter) => iter.next(),
            NodeIdIter::None => None,
        }
    }
}

/// An iterator over channels to connect from. Use internally by handles. You won't need to interact with it directly unless you are implementing a handle.
#[derive(Clone, Debug)]
pub enum SourceChannelIter {
    /// All the channels from a single node
    SingleNodeId {
        #[allow(missing_docs)]
        node_id: NodeId,
        #[allow(missing_docs)]
        num_channels: usize,
        #[allow(missing_docs)]
        current_iter_index: usize,
    },
    /// Any number of channels from potentially different source nodes.
    Vec {
        #[allow(missing_docs)]
        channels: Vec<(Source, NodeChannel)>,
        #[allow(missing_docs)]
        current_index: usize,
    },
    /// A single channel
    SingleChannel {
        #[allow(missing_docs)]
        node_id: NodeId,
        #[allow(missing_docs)]
        channel: NodeChannel,
        #[allow(missing_docs)]
        returned: bool,
    },
    /// Channels from a graph input
    GraphInput {
        #[allow(missing_docs)]
        graph_id: GraphId,
        #[allow(missing_docs)]
        start_index: usize,
        #[allow(missing_docs)]
        num_channels: usize,
        #[allow(missing_docs)]
        current_channel: usize,
    },
    /// No channels
    None,
}
impl SourceChannelIter {
    /// Create a `ChannelIter` from a Vec of sources and channels. This is the most flexible option.  
    #[must_use]
    pub fn from_vec(channels: Vec<(Source, NodeChannel)>) -> Self {
        Self::Vec {
            channels,
            current_index: 0,
        }
    }
    /// Create a `ChannelIter` from a single node id and a number of channels and no offset
    #[must_use]
    pub fn single_node_id(node_id: NodeId, num_channels: usize) -> Self {
        Self::SingleNodeId {
            node_id,
            num_channels,
            current_iter_index: 0,
        }
    }
    /// Create a `ChannelIter` with a single channel
    #[must_use]
    pub fn single_channel(node_id: NodeId, channel: NodeChannel) -> Self {
        Self::SingleChannel {
            node_id,
            channel,
            returned: false,
        }
    }
}

/// Represents a source within a Graph, either an input to the graph or a node within the graph.
#[derive(Copy, Clone, Debug)]
pub enum Source {
    #[allow(missing_docs)]
    GraphInput(GraphId),
    #[allow(missing_docs)]
    Gen(NodeId),
}
impl Source {
    /// Create a connection to another node
    #[must_use]
    pub fn to(self, other: NodeId) -> Connection {
        match self {
            Source::GraphInput(_graph_id) => Connection::graph_input(other),
            Source::Gen(node) => node.to(other),
        }
    }
    /// Create a connection to another node
    #[must_use]
    pub fn to_sink(self, other: Sink) -> Connection {
        match self {
            Source::GraphInput(graph_id) => match other {
                Sink::GraphOutput(g_id) => {
                    assert_eq!(graph_id, g_id);
                    Connection::GraphInputToOutput {
                        graph_id,
                        from_input_channel: 0,
                        to_output_channel: 0,
                        channels: 1,
                    }
                }
                Sink::Gen(node_id) => Connection::graph_input(node_id),
            },
            Source::Gen(node) => match other {
                Sink::GraphOutput(_) => node.to_graph_out(),
                Sink::Gen(other_node) => node.to(other_node),
            },
        }
    }
    /// Create a connection to the graph output
    #[must_use]
    pub fn to_graph_out(self) -> Connection {
        match self {
            Source::GraphInput(graph_id) => Connection::GraphInputToOutput {
                graph_id,
                from_input_channel: 0,
                to_output_channel: 0,
                channels: 1,
            },
            Source::Gen(node) => node.to_graph_out(),
        }
    }
}
impl From<NodeId> for Source {
    fn from(value: NodeId) -> Self {
        Self::Gen(value)
    }
}
impl From<&mut NodeId> for Source {
    fn from(value: &mut NodeId) -> Self {
        Self::Gen(*value)
    }
}
impl From<&NodeId> for Source {
    fn from(value: &NodeId) -> Self {
        Self::Gen(*value)
    }
}
impl Iterator for SourceChannelIter {
    type Item = (Source, NodeChannel);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SourceChannelIter::SingleNodeId {
                node_id,
                num_channels,
                current_iter_index,
            } => {
                if *current_iter_index < *num_channels {
                    *current_iter_index += 1;
                    Some((node_id.into(), NodeChannel::Index(*current_iter_index - 1)))
                } else {
                    None
                }
            }
            SourceChannelIter::Vec {
                channels,
                current_index,
            } => {
                let item = channels.get(*current_index).map(|(id, chan)| (*id, *chan));
                *current_index += 1;
                item
            }
            SourceChannelIter::SingleChannel {
                node_id,
                channel,
                returned,
            } => {
                if *returned {
                    None
                } else {
                    *returned = true;
                    Some((node_id.into(), *channel))
                }
            }
            SourceChannelIter::None => None,
            SourceChannelIter::GraphInput {
                graph_id,
                start_index,
                num_channels,
                current_channel,
            } => {
                if current_channel == num_channels {
                    None
                } else {
                    *current_channel += 1;
                    Some((
                        Source::GraphInput(*graph_id),
                        NodeChannel::Index(*start_index + *current_channel - 1),
                    ))
                }
            }
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            SourceChannelIter::SingleNodeId {
                node_id: _,
                num_channels,
                current_iter_index,
            } => {
                let num_left = num_channels - current_iter_index;
                (num_left, Some(num_left))
            }
            SourceChannelIter::Vec {
                channels,
                current_index,
            } => {
                let num_left = channels.len() - current_index;
                (num_left, Some(num_left))
            }
            SourceChannelIter::SingleChannel { returned, .. } => {
                if *returned {
                    (0, Some(0))
                } else {
                    (1, Some(1))
                }
            }
            SourceChannelIter::None => (0, Some(0)),
            SourceChannelIter::GraphInput {
                num_channels,
                current_channel,
                ..
            } => {
                let num_left = num_channels - *current_channel;
                (num_left, Some(num_left))
            }
        }
    }
}

/// An iterator over channels to connect from. Use internally by handles. You won't need to interact with it directly unless you are implementing a handle.
#[derive(Clone, Debug)]
pub enum SinkChannelIter {
    /// All the channels from a single node
    SingleNodeId {
        #[allow(missing_docs)]
        node_id: NodeId,
        #[allow(missing_docs)]
        num_channels: usize,
        #[allow(missing_docs)]
        current_iter_index: usize,
    },
    /// Any number of channels from potentially different source nodes.
    Vec {
        #[allow(missing_docs)]
        channels: Vec<(Sink, NodeChannel)>,
        #[allow(missing_docs)]
        current_index: usize,
    },
    /// A single channel
    SingleChannel {
        #[allow(missing_docs)]
        node_id: NodeId,
        #[allow(missing_docs)]
        channel: NodeChannel,
        #[allow(missing_docs)]
        returned: bool,
    },
    /// Channels to a graph output
    GraphOutput {
        #[allow(missing_docs)]
        graph_id: GraphId,
        #[allow(missing_docs)]
        start_index: usize,
        #[allow(missing_docs)]
        num_channels: usize,
        #[allow(missing_docs)]
        current_channel: usize,
    },
    /// No channels
    None,
}
impl SinkChannelIter {
    /// Create a `ChannelIter` from a Vec of sources and channels. This is the most flexible option.  
    #[must_use]
    pub fn from_vec(channels: Vec<(Sink, NodeChannel)>) -> Self {
        Self::Vec {
            channels,
            current_index: 0,
        }
    }
    /// Create a `ChannelIter` from a single node id and a number of channels and no offset
    #[must_use]
    pub fn single_node_id(node_id: NodeId, num_channels: usize) -> Self {
        Self::SingleNodeId {
            node_id,
            num_channels,
            current_iter_index: 0,
        }
    }
    /// Create a `ChannelIter` with a single channel
    #[must_use]
    pub fn single_channel(node_id: NodeId, channel: NodeChannel) -> Self {
        Self::SingleChannel {
            node_id,
            channel,
            returned: false,
        }
    }
}

/// Represents a source within a Graph, either an input to the graph or a node within the graph.
#[derive(Copy, Clone, Debug)]
pub enum Sink {
    #[allow(missing_docs)]
    GraphOutput(GraphId),
    #[allow(missing_docs)]
    Gen(NodeId),
}
impl From<NodeId> for Sink {
    fn from(value: NodeId) -> Self {
        Self::Gen(value)
    }
}
impl From<&mut NodeId> for Sink {
    fn from(value: &mut NodeId) -> Self {
        Self::Gen(*value)
    }
}
impl From<&NodeId> for Sink {
    fn from(value: &NodeId) -> Self {
        Self::Gen(*value)
    }
}
impl Iterator for SinkChannelIter {
    type Item = (Sink, NodeChannel);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SinkChannelIter::SingleNodeId {
                node_id,
                num_channels,
                current_iter_index,
            } => {
                if *current_iter_index < *num_channels {
                    *current_iter_index += 1;
                    Some((node_id.into(), NodeChannel::Index(*current_iter_index - 1)))
                } else {
                    None
                }
            }
            SinkChannelIter::Vec {
                channels,
                current_index,
            } => {
                let item = channels.get(*current_index).map(|(id, chan)| (*id, *chan));
                *current_index += 1;
                item
            }
            SinkChannelIter::SingleChannel {
                node_id,
                channel,
                returned,
            } => {
                if *returned {
                    None
                } else {
                    *returned = true;
                    Some((node_id.into(), *channel))
                }
            }
            SinkChannelIter::None => None,
            SinkChannelIter::GraphOutput {
                graph_id,
                start_index,
                num_channels,
                current_channel,
            } => {
                if current_channel == num_channels {
                    None
                } else {
                    *current_channel += 1;
                    Some((
                        Sink::GraphOutput(*graph_id),
                        NodeChannel::Index(*start_index + *current_channel - 1),
                    ))
                }
            }
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            SinkChannelIter::SingleNodeId {
                node_id: _,
                num_channels,
                current_iter_index,
            } => {
                let num_left = num_channels - current_iter_index;
                (num_left, Some(num_left))
            }
            SinkChannelIter::Vec {
                channels,
                current_index,
            } => {
                let num_left = channels.len() - current_index;
                (num_left, Some(num_left))
            }
            SinkChannelIter::SingleChannel { returned, .. } => {
                if *returned {
                    (0, Some(0))
                } else {
                    (1, Some(1))
                }
            }
            SinkChannelIter::None => (0, Some(0)),
            SinkChannelIter::GraphOutput {
                num_channels,
                current_channel,
                ..
            } => {
                let num_left = num_channels - *current_channel;
                (num_left, Some(num_left))
            }
        }
    }
}

// Multiplication

impl<A, B> Mul<Handle<A>> for Handle<B>
where
    A: Copy + HandleData,
    B: Copy + HandleData,
{
    type Output = Handle<MulHandle>;

    fn mul(self, rhs: Handle<A>) -> Self::Output {
        let max_channels = self.out_channels().count().max(rhs.out_channels().count());
        let mul_id = knyst_commands().push_without_inputs(MulGen(max_channels));
        for (i, (out0, out1)) in self
            .out_channels()
            .cycle()
            .zip(rhs.out_channels().cycle())
            .take(max_channels)
            .enumerate()
        {
            knyst_commands().connect(out0.0.to(mul_id).from_channel(out0.1).to_channel(i * 2));
            knyst_commands().connect(out1.0.to(mul_id).from_channel(out1.1).to_channel(i * 2 + 1));
        }
        Handle::new(MulHandle {
            node_id: mul_id,
            num_out_channels: max_channels,
        })
    }
}

impl<B: Copy + HandleData> Mul<Handle<B>> for Sample {
    type Output = Handle<MulHandle>;

    fn mul(self, rhs: Handle<B>) -> Self::Output {
        let num_out_channels = rhs.out_channels().collect::<Vec<_>>().len();
        let mul_id = knyst_commands().push_without_inputs(MulGen(num_out_channels));
        for (i, (out0, out1)) in std::iter::repeat(self).zip(rhs.out_channels()).enumerate() {
            knyst_commands().connect(
                crate::graph::connection::constant(out0)
                    .to(mul_id)
                    .to_channel(i * 2),
            );
            knyst_commands().connect(out1.0.to(mul_id).from_channel(out1.1).to_channel(i * 2 + 1));
        }
        Handle::new(MulHandle {
            node_id: mul_id,
            num_out_channels,
        })
    }
}
impl<B: Copy + HandleData> Mul<Sample> for Handle<B> {
    type Output = Handle<MulHandle>;

    fn mul(self, rhs: Sample) -> Self::Output {
        rhs * self
    }
}

// Mul AnyNodeHandle * Handle
impl<B> Mul<AnyNodeHandle> for Handle<B>
where
    B: Copy + HandleData,
{
    type Output = Handle<MulHandle>;

    fn mul(self, rhs: AnyNodeHandle) -> Self::Output {
        let max_channels = self.out_channels().count().max(rhs.out_channels().count());
        let mul_id = knyst_commands().push_without_inputs(MulGen(max_channels));
        for (i, (out0, out1)) in self
            .out_channels()
            .cycle()
            .zip(rhs.out_channels().cycle())
            .take(max_channels)
            .enumerate()
        {
            knyst_commands().connect(out0.0.to(mul_id).from_channel(out0.1).to_channel(i * 2));
            knyst_commands().connect(out1.0.to(mul_id).from_channel(out1.1).to_channel(i * 2 + 1));
        }
        Handle::new(MulHandle {
            node_id: mul_id,
            num_out_channels: max_channels,
        })
    }
}
impl<B> Mul<Handle<B>> for AnyNodeHandle
where
    B: Copy + HandleData,
{
    type Output = Handle<MulHandle>;

    fn mul(self, rhs: Handle<B>) -> Self::Output {
        rhs * self
    }
}

impl Mul<AnyNodeHandle> for Sample {
    type Output = Handle<MulHandle>;

    fn mul(self, rhs: AnyNodeHandle) -> Self::Output {
        let num_out_channels = rhs.out_channels().collect::<Vec<_>>().len();
        let mul_id = knyst_commands().push_without_inputs(MulGen(num_out_channels));
        for (i, (out0, out1)) in std::iter::repeat(self).zip(rhs.out_channels()).enumerate() {
            knyst_commands().connect(
                crate::graph::connection::constant(out0)
                    .to(mul_id)
                    .to_channel(i * 2),
            );
            knyst_commands().connect(out1.0.to(mul_id).from_channel(out1.1).to_channel(i * 2 + 1));
        }
        Handle::new(MulHandle {
            node_id: mul_id,
            num_out_channels,
        })
    }
}
impl Mul<Sample> for AnyNodeHandle {
    type Output = Handle<MulHandle>;

    fn mul(self, rhs: Sample) -> Self::Output {
        rhs * self
    }
}

// Subtraction

impl<A, B> Sub<Handle<A>> for Handle<B>
where
    A: Copy + HandleData,
    B: Copy + HandleData,
{
    type Output = Handle<SubHandle>;

    fn sub(self, rhs: Handle<A>) -> Self::Output {
        let num_out_channels = self.out_channels().collect::<Vec<_>>().len();
        let id = knyst_commands().push_without_inputs(SubGen(num_out_channels));
        for (i, (out0, out1)) in self.out_channels().zip(rhs.out_channels()).enumerate() {
            knyst_commands().connect(out0.0.to(id).from_channel(out0.1).to_channel(i * 2));
            knyst_commands().connect(out1.0.to(id).from_channel(out1.1).to_channel(i * 2 + 1));
        }
        Handle::new(SubHandle {
            node_id: id,
            num_out_channels,
        })
    }
}

impl<B: Copy + HandleData> Sub<Handle<B>> for Sample {
    type Output = Handle<SubHandle>;

    fn sub(self, rhs: Handle<B>) -> Self::Output {
        let num_out_channels = rhs.out_channels().collect::<Vec<_>>().len();
        let id = knyst_commands().push_without_inputs(SubGen(num_out_channels));
        for (i, (out0, out1)) in std::iter::repeat(self).zip(rhs.out_channels()).enumerate() {
            knyst_commands().connect(
                crate::graph::connection::constant(out0)
                    .to(id)
                    .to_channel(i * 2),
            );
            knyst_commands().connect(out1.0.to(id).from_channel(out1.1).to_channel(i * 2 + 1));
        }
        Handle::new(SubHandle {
            node_id: id,
            num_out_channels,
        })
    }
}
impl<B: Copy + HandleData> Sub<Sample> for Handle<B> {
    type Output = Handle<SubHandle>;

    fn sub(self, rhs: Sample) -> Self::Output {
        rhs - self
    }
}

// Sub AnyNodeHandle - Handle
impl<B> Sub<AnyNodeHandle> for Handle<B>
where
    B: Copy + HandleData,
{
    type Output = Handle<SubHandle>;

    fn sub(self, rhs: AnyNodeHandle) -> Self::Output {
        let num_out_channels = self.out_channels().collect::<Vec<_>>().len();
        let id = knyst_commands().push_without_inputs(SubGen(num_out_channels));
        for (i, (out0, out1)) in self.out_channels().zip(rhs.out_channels()).enumerate() {
            knyst_commands().connect(out0.0.to(id).from_channel(out0.1).to_channel(i * 2));
            knyst_commands().connect(out1.0.to(id).from_channel(out1.1).to_channel(i * 2 + 1));
        }
        Handle::new(SubHandle {
            node_id: id,
            num_out_channels,
        })
    }
}
impl<B> Sub<Handle<B>> for AnyNodeHandle
where
    B: Copy + HandleData,
{
    type Output = Handle<SubHandle>;

    fn sub(self, rhs: Handle<B>) -> Self::Output {
        let num_out_channels = self.out_channels().collect::<Vec<_>>().len();
        let id = knyst_commands().push_without_inputs(SubGen(num_out_channels));
        for (i, (out0, out1)) in self.out_channels().zip(rhs.out_channels()).enumerate() {
            knyst_commands().connect(out0.0.to(id).from_channel(out0.1).to_channel(i * 2));
            knyst_commands().connect(out1.0.to(id).from_channel(out1.1).to_channel(i * 2 + 1));
        }
        Handle::new(SubHandle {
            node_id: id,
            num_out_channels,
        })
    }
}

impl Sub<AnyNodeHandle> for Sample {
    type Output = Handle<SubHandle>;

    fn sub(self, rhs: AnyNodeHandle) -> Self::Output {
        rhs - self
    }
}
impl Sub<Sample> for AnyNodeHandle {
    type Output = Handle<SubHandle>;

    fn sub(self, rhs: Sample) -> Self::Output {
        rhs - self
    }
}

// Add
impl<A, B> Add<Handle<A>> for Handle<B>
where
    A: Copy + HandleData,
    B: Copy + HandleData,
{
    type Output = Handle<AddHandle>;

    fn add(self, rhs: Handle<A>) -> Self::Output {
        let num_out_channels = self.out_channels().collect::<Vec<_>>().len();
        let node_id = knyst_commands().push_without_inputs(Bus(num_out_channels));
        for (i, (out0, out1)) in self.out_channels().zip(rhs.out_channels()).enumerate() {
            knyst_commands().connect(out0.0.to(node_id).from_channel(out0.1).to_channel(i));
            knyst_commands().connect(out1.0.to(node_id).from_channel(out1.1).to_channel(i));
        }
        Handle::new(AddHandle {
            node_id,
            num_out_channels,
        })
    }
}

impl<B: Copy + HandleData> Add<Handle<B>> for Sample {
    type Output = Handle<AddHandle>;

    fn add(self, rhs: Handle<B>) -> Self::Output {
        let num_out_channels = rhs.out_channels().collect::<Vec<_>>().len();
        let node_id = knyst_commands().push_without_inputs(Bus(num_out_channels));
        for (i, (out0, out1)) in std::iter::repeat(self).zip(rhs.out_channels()).enumerate() {
            knyst_commands().connect(
                crate::graph::connection::constant(out0)
                    .to(node_id)
                    .to_channel(i),
            );
            knyst_commands().connect(out1.0.to(node_id).from_channel(out1.1).to_channel(i));
        }
        Handle::new(AddHandle {
            node_id,
            num_out_channels,
        })
    }
}
impl<B: Copy + HandleData> Add<Sample> for Handle<B> {
    type Output = Handle<AddHandle>;

    fn add(self, rhs: Sample) -> Self::Output {
        rhs + self
    }
}

// Add Node + AnyNodeHandle
impl<A> Add<Handle<A>> for AnyNodeHandle
where
    A: Copy + HandleData,
{
    type Output = Handle<AddHandle>;

    fn add(self, rhs: Handle<A>) -> Self::Output {
        let num_out_channels = self.out_channels().collect::<Vec<_>>().len();
        let node_id = knyst_commands().push_without_inputs(Bus(num_out_channels));
        for (i, (out0, out1)) in self.out_channels().zip(rhs.out_channels()).enumerate() {
            knyst_commands().connect(out0.0.to(node_id).from_channel(out0.1).to_channel(i));
            knyst_commands().connect(out1.0.to(node_id).from_channel(out1.1).to_channel(i));
        }
        Handle::new(AddHandle {
            node_id,
            num_out_channels,
        })
    }
}
impl<A> Add<AnyNodeHandle> for Handle<A>
where
    A: Copy + HandleData,
{
    type Output = Handle<AddHandle>;

    fn add(self, rhs: AnyNodeHandle) -> Self::Output {
        rhs + self
    }
}

impl Add<AnyNodeHandle> for Sample {
    type Output = Handle<AddHandle>;

    fn add(self, rhs: AnyNodeHandle) -> Self::Output {
        let num_out_channels = rhs.out_channels().collect::<Vec<_>>().len();
        let node_id = knyst_commands().push_without_inputs(Bus(num_out_channels));
        for (i, (out0, out1)) in std::iter::repeat(self).zip(rhs.out_channels()).enumerate() {
            knyst_commands().connect(
                crate::graph::connection::constant(out0)
                    .to(node_id)
                    .to_channel(i),
            );
            knyst_commands().connect(out1.0.to(node_id).from_channel(out1.1).to_channel(i));
        }
        Handle::new(AddHandle {
            node_id,
            num_out_channels,
        })
    }
}
impl Add<Sample> for AnyNodeHandle {
    type Output = Handle<AddHandle>;

    fn add(self, rhs: Sample) -> Self::Output {
        rhs + self
    }
}
/// A safe way to store a Handle without the generic parameter, but keep being able to use it
pub struct AnyNodeHandle {
    _org_handle: Box<dyn Any>,
    in_channel_iter: SinkChannelIter,
    out_channel_iter: SourceChannelIter,
    node_ids: NodeIdIter,
}
impl AnyNodeHandle {
    /// The non-typed way to set an input channel's value
    pub fn set(&self, channel: impl Into<NodeChannel>, input: impl Into<Input>) -> &Self {
        let inp = input.into();
        let channel = channel.into();
        match inp {
            Input::Constant(v) => {
                for id in self.node_ids() {
                    let change = ParameterChange::now(id.input(channel), Change::Constant(v));
                    knyst_commands().schedule_change(change);
                }
            }
            Input::Handle { output_channels } => {
                for (node_id, chan) in output_channels {
                    for id in self.node_ids() {
                        knyst_commands()
                            .connect(node_id.to(id).from_channel(chan).to_channel(channel));
                    }
                }
            }
        }
        self
    }
    /// The non-typed way to send a trigger to an input channel
    pub fn trig(&self, channel: impl Into<NodeChannel>) -> &Self {
        let channel = channel.into();
        // TODO: better way to send a trigger
        let mut changes = SimultaneousChanges::now();
        for id in self.node_ids() {
            changes.push(id.change().trigger(channel.clone()));
        }
        knyst_commands().schedule_changes(changes);
        self
    }
}

impl<H: Copy + HandleData + 'static> From<Handle<H>> for AnyNodeHandle {
    fn from(value: Handle<H>) -> Self {
        AnyNodeHandle {
            _org_handle: Box::new(value),
            in_channel_iter: value.in_channels(),
            out_channel_iter: value.out_channels(),
            node_ids: value.node_ids(),
        }
    }
}
impl HandleData for AnyNodeHandle {
    fn out_channels(&self) -> SourceChannelIter {
        self.out_channel_iter.clone()
    }

    fn in_channels(&self) -> SinkChannelIter {
        self.in_channel_iter.clone()
    }

    fn node_ids(&self) -> NodeIdIter {
        self.node_ids.clone()
    }
}
/// Gathering all kinds of inputs into one
pub enum Input {
    /// Constant value input
    Constant(Sample),
    /// Input from another handle
    #[allow(missing_docs)]
    Handle { output_channels: SourceChannelIter },
}
impl From<Sample> for Input {
    fn from(value: Sample) -> Self {
        Input::Constant(value)
    }
}
impl<H: Copy + HandleData> From<Handle<H>> for Input {
    fn from(value: Handle<H>) -> Self {
        Input::Handle {
            output_channels: value.out_channels(),
        }
    }
}
impl From<&AnyNodeHandle> for Input {
    fn from(value: &AnyNodeHandle) -> Self {
        Input::Handle {
            output_channels: value.out_channel_iter.clone(),
        }
    }
}

/// Marker trait to mark that the output of the Gen will be between -1 and 1 inclusive
pub trait HandleNormalRange {}

/// Handle to a range converter, usually acquired by calling the `.range` method on a supported handle.
#[derive(Clone, Copy, Debug)]
pub struct RangeHandle {
    node_id: NodeId,
    num_out_channels: usize,
}
impl HandleData for RangeHandle {
    fn out_channels(&self) -> SourceChannelIter {
        SourceChannelIter::SingleNodeId {
            node_id: self.node_id,
            num_channels: self.num_out_channels,
            current_iter_index: 0,
        }
    }

    fn in_channels(&self) -> SinkChannelIter {
        SinkChannelIter::SingleNodeId {
            node_id: self.node_id,
            num_channels: self.num_out_channels * 2,
            current_iter_index: 0,
        }
    }

    fn node_ids(&self) -> NodeIdIter {
        NodeIdIter::Single(self.node_id)
    }
}

impl<H: HandleData + Copy + HandleNormalRange> Handle<H> {
    /// Convert the output range from [-1, 1] to [min, max]
    pub fn range(self, min: impl Into<Input>, max: impl Into<Input>) -> Handle<RangeHandle> {
        // Convert to the correct range for the RangeGen
        let input = self * 0.5 + 0.5;
        let num_out_channels = input.out_channels().collect::<Vec<_>>().len();
        let node_id = knyst_commands().push_without_inputs(RangeGen(num_out_channels));
        match min.into() {
            Input::Constant(c) => {
                knyst_commands().connect(
                    crate::graph::connection::constant(c)
                        .to(node_id)
                        .to_channel(0),
                );
            }
            Input::Handle {
                ref mut output_channels,
            } => {
                if let Some((node, chan)) = output_channels.next() {
                    knyst_commands().connect(node.to(node_id).from_channel(chan).to_channel(0));
                } else {
                    // TODO: Error: empty handle as min input to ramp
                }
                if output_channels.next().is_some() {
                    // TODO: Warn: multi channel handle as min input to ramp
                }
            }
        }
        match max.into() {
            Input::Constant(c) => {
                knyst_commands().connect(
                    crate::graph::connection::constant(c)
                        .to(node_id)
                        .to_channel(1),
                );
            }
            Input::Handle {
                ref mut output_channels,
            } => {
                if let Some((node, chan)) = output_channels.next() {
                    knyst_commands().connect(node.to(node_id).from_channel(chan).to_channel(1));
                } else {
                    // TODO: Error: empty handle as max input to ramp
                }
                if output_channels.next().is_some() {
                    // TODO: Warn: multi channel handle as maxinput to ramp
                }
            }
        }
        for (i, out0) in input.out_channels().enumerate() {
            knyst_commands().connect(out0.0.to(node_id).from_channel(out0.1).to_channel(i + 2));
        }
        Handle::new(RangeHandle {
            node_id,
            num_out_channels,
        })
    }
}

/// Handle for a Graph input, usually acquired through the `graph_input` function.
#[derive(Copy, Clone, Debug)]
pub struct GraphInputHandle {
    graph_id: GraphId,
    start_index: usize,
    num_channels: usize,
}

impl HandleData for GraphInputHandle {
    fn out_channels(&self) -> SourceChannelIter {
        SourceChannelIter::GraphInput {
            graph_id: self.graph_id,
            start_index: self.start_index,
            num_channels: self.num_channels,
            current_channel: 0,
        }
    }

    fn in_channels(&self) -> SinkChannelIter {
        SinkChannelIter::None
    }

    fn node_ids(&self) -> NodeIdIter {
        NodeIdIter::None
    }
}

/// Creates a handle to numer of consecutive graph input channels.
#[must_use]
pub fn graph_input(index: usize, num_channels: usize) -> Handle<GraphInputHandle> {
    Handle::new(GraphInputHandle {
        start_index: index,
        num_channels,
        graph_id: knyst_commands().current_graph(),
    })
}

/// Connects all the output channels of input to the graph outputs of its graph, starting at `index`
pub fn graph_output(index: usize, input: impl Into<Input>) {
    let inp = input.into();
    match inp {
        Input::Constant(_v) => {
            // commands().connect(
            //     crate::graph::connection::constant(v)
            //         .to_graph_out()
            //         .to_channel(index),
            // );
            todo!()
        }
        Input::Handle { output_channels } => {
            for (i, (node_id, chan)) in output_channels.enumerate() {
                knyst_commands().connect(
                    node_id
                        .to_graph_out()
                        .from_channel(chan)
                        .to_channel(index + i),
                );
            }
        }
    }
}
/// Get a handle to a single graph output
pub fn graph_output_handle(index: usize) -> Handle<InputChannelHandle> {
    Handle::new(InputChannelHandle {
        source: Sink::GraphOutput(knyst_commands().current_graph()),
        channel: index.into(),
    })
}

/// A handle type that can refer to any Gen and still be Copy. It knows only how many output/input channels the Gen has, not their labels. You can still use labels via the `set` method.
#[derive(Copy, Clone, Debug)]
pub struct GenericHandle {
    node_id: NodeId,
    num_inputs: usize,
    num_outputs: usize,
}
impl GenericHandle {
    /// Create a [`GenericHandle`] for the given node
    pub fn new(node_id: NodeId, num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            node_id,
            num_inputs,
            num_outputs,
        }
    }
    // /// The non-typed way to set an input channel's value
    // pub fn set(
    //     self,
    //     channel: impl Into<NodeChannel>,
    //     input: impl Into<Input>,
    // ) -> Handle<GenericHandle> {
    //     let inp = input.into();
    //     let channel = channel.into();
    //     match inp {
    //         Input::Constant(v) => {
    //             let change = ParameterChange::now(self.node_id.input(channel), Change::Constant(v));
    //             knyst_commands().schedule_change(change);
    //         }
    //         Input::Handle { output_channels } => {
    //             for (i, (node_id, chan)) in output_channels.enumerate() {
    //                 knyst_commands().connect(
    //                     node_id
    //                         .to(self.node_id)
    //                         .from_channel(chan)
    //                         .to_channel(channel)
    //                         .to_channel_offset(i),
    //                 );
    //             }
    //         }
    //     }
    //     Handle::new(self)
    // }
    // /// The non-typed way to send a trigger to an input channel
    // pub fn trig(self, channel: impl Into<NodeChannel>) -> Handle<GenericHandle> {
    //     let _ = channel;
    //     // TODO: better way to send a trigger
    //     let change = ParameterChange::now(self.node_id.input(channel), Change::Trigger);
    //     knyst_commands().schedule_change(change);
    //     todo!(
    //         "Sending trigs this way doesn't work and I don't have time to troubleshoot right now"
    //     );
    //     // Handle::new(self)
    // }
}
impl HandleData for GenericHandle {
    fn out_channels(&self) -> SourceChannelIter {
        SourceChannelIter::SingleNodeId {
            node_id: self.node_id,
            num_channels: self.num_outputs,
            current_iter_index: 0,
        }
    }

    fn in_channels(&self) -> SinkChannelIter {
        SinkChannelIter::single_node_id(self.node_id, self.num_inputs)
    }

    fn node_ids(&self) -> NodeIdIter {
        NodeIdIter::Single(self.node_id)
    }
}
/// Upload the [`Gen`] or graph to the active graph and return a [`GenericHandle`] for routing and setting inputs.
///
/// This is useful when the Gen you want to add doesn't have it's own handle function for some reason. Prefer the
/// type specific handle init function for the type you want.
pub fn handle(gen: impl GenOrGraph) -> Handle<GenericHandle> {
    let num_inputs = gen.num_inputs();
    let num_outputs = gen.num_outputs();
    let node_id = knyst_commands().push_without_inputs(gen);
    Handle::new(GenericHandle {
        node_id,
        num_inputs,
        num_outputs,
    })
}

/// A `Handle` to a `Graph`
#[derive(Copy, Clone, Debug)]
pub struct GraphHandle {
    node_id: NodeId,
    graph_id: GraphId,
    num_inputs: usize,
    num_outputs: usize,
}
impl GraphHandle {
    pub(crate) fn new(
        id: NodeId,
        graph_id: GraphId,
        num_inputs: usize,
        num_outputs: usize,
    ) -> Self {
        Self {
            node_id: id,
            graph_id,
            num_inputs,
            num_outputs,
        }
    }
    /// Return the `GraphId` of the Graph this handle points to.
    #[must_use]
    pub fn graph_id(&self) -> GraphId {
        self.graph_id
    }
    /// Set this graph to the active graph to push new Gens to on the local thread
    pub fn activate(&self) {
        knyst_commands().to_graph(self.graph_id)
    }
    // /// The non-typed way to set an input channel's value
    // pub fn set(
    //     self,
    //     channel: impl Into<NodeChannel>,
    //     input: impl Into<Input>,
    // ) -> Handle<GraphHandle> {
    //     let inp = input.into();
    //     let channel = channel.into();
    //     match inp {
    //         Input::Constant(v) => {
    //             knyst_commands().connect(
    //                 crate::graph::connection::constant(v)
    //                     .to(self.node_id)
    //                     .to_channel(channel),
    //             );
    //         }
    //         Input::Handle { output_channels } => {
    //             for (i, (node_id, chan)) in output_channels.enumerate() {
    //                 knyst_commands().connect(
    //                     node_id
    //                         .to(self.node_id)
    //                         .from_channel(chan)
    //                         .to_channel(channel)
    //                         .to_channel_offset(i),
    //                 );
    //             }
    //         }
    //     }
    //     Handle::new(self)
    // }
}
impl HandleData for GraphHandle {
    fn out_channels(&self) -> SourceChannelIter {
        SourceChannelIter::SingleNodeId {
            node_id: self.node_id,
            num_channels: self.num_outputs,
            current_iter_index: 0,
        }
    }

    fn in_channels(&self) -> SinkChannelIter {
        SinkChannelIter::single_node_id(self.node_id, self.num_inputs)
    }

    fn node_ids(&self) -> NodeIdIter {
        NodeIdIter::Single(self.node_id)
    }
}

#[must_use]
/// Create a Bus Gen with the selected number of channels. One use case is as a single node for setting values later. Another is as a passthrough node for gathering multiple variable sources.
pub fn bus(num_channels: usize) -> Handle<GenericHandle> {
    let node_id = knyst_commands().push_without_inputs(Bus(num_channels));
    Handle::new(GenericHandle {
        node_id,
        num_inputs: num_channels,
        num_outputs: num_channels,
    })
}
