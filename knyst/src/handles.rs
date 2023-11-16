use std::{
    any::Any,
    ops::{Add, Deref, Mul},
};

use crate::{
    controller::CallbackHandle,
    graph::{Change, Connection, GraphId, ParameterChange},
    prelude::PowfGen,
    Sample,
};

use crate::{
    graph::{connection::NodeChannel, GenOrGraph, NodeId},
    modal_interface::commands,
    prelude::{Bus, KnystCommands, MulGen, RampGen},
};

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
// impl<A: Copy + Into<NodeId>> From<NodeHandle<A>> for NodeId {
//     fn from(value: NodeHandle<A>) -> Self {
//         value.handle.into()
//     }
// }
impl Handle<GenericHandle> {
    /// Create a dummy handle pointing to nothing.
    pub fn void() -> Handle<GenericHandle> {
        Self::new(GenericHandle {
            node_id: NodeId::new(),
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
    /// Take all the output channels to the power of the given exponent
    pub fn powf(self, exponent: impl Into<Input>) -> Handle<PowfHandle> {
        let connecting_channels: Vec<_> = self.out_channels().collect();
        let num_channels = connecting_channels.len();
        let node_id = commands().push_without_inputs(PowfGen(connecting_channels.len()));
        for (i, (source, chan)) in connecting_channels.into_iter().enumerate() {
            commands().connect(source.to(node_id).from_channel(chan).to_channel(i + 1));
        }
        Handle::new(PowfHandle {
            node_id,
            num_channels,
        })
        .exponent(exponent)
    }
    /// Free the node(s) this handle is pointing to
    pub fn free(self) {
        for id in self.node_ids() {
            commands().free_node(id);
        }
    }
    /// Returns a handle to a single channel from this Handle (not type checked)
    pub fn out(self, channel: impl Into<NodeChannel>) -> Handle<OutputChannelHandle> {
        let channel = channel.into();
        Handle::new(OutputChannelHandle {
            node_id: self.node_ids().next().unwrap(),
            channel,
        })
    }
}
impl<H: HandleData + Copy> HandleData for Handle<H> {
    fn out_channels(&self) -> ChannelIter {
        self.handle.out_channels()
    }

    fn in_channels(&self) -> ChannelIter {
        self.handle.in_channels()
    }

    fn node_ids(&self) -> NodeIdIter {
        self.handle.node_ids()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PowfHandle {
    node_id: NodeId,
    num_channels: usize,
}
impl PowfHandle {
    /// Set the exponent
    pub fn exponent(self, exponent: impl Into<Input>) -> Handle<Self> {
        let inp = exponent.into();
        match inp {
            Input::Constant(v) => {
                commands().connect(
                    crate::graph::connection::constant(v)
                        .to(self.node_id)
                        .to_channel(0),
                );
            }
            Input::Handle { output_channels } => {
                for (node_id, chan) in output_channels {
                    crate::modal_interface::commands()
                        .connect(node_id.to(self.node_id).from_channel(chan).to_channel(0));
                }
            }
        }
        Handle::new(self)
    }
}
impl HandleData for PowfHandle {
    fn out_channels(&self) -> ChannelIter {
        ChannelIter::single_node_id(self.node_id, self.num_channels)
    }

    fn in_channels(&self) -> ChannelIter {
        ChannelIter::single_node_id(self.node_id, self.num_channels + 1)
    }

    fn node_ids(&self) -> NodeIdIter {
        NodeIdIter::Single(self.node_id)
    }
}

/// Handle to a single output channel from a node.
#[derive(Copy, Clone, Debug)]
pub struct OutputChannelHandle {
    node_id: NodeId,
    channel: NodeChannel,
}
impl HandleData for OutputChannelHandle {
    fn out_channels(&self) -> ChannelIter {
        ChannelIter::single_channel(self.node_id, self.channel)
    }

    fn in_channels(&self) -> ChannelIter {
        ChannelIter::single_node_id(self.node_id, 0)
    }

    fn node_ids(&self) -> NodeIdIter {
        NodeIdIter::Single(self.node_id)
    }
}

/// Handle for a `repeat_outputs`
#[derive(Copy, Clone, Debug)]
pub struct RepeatOutputs<H: Copy + HandleData> {
    handle: H,
    repeats: usize,
}
impl<H: Copy + HandleData> HandleData for RepeatOutputs<H> {
    fn out_channels(&self) -> ChannelIter {
        let channels = self
            .handle
            .out_channels()
            .flat_map(|item| std::iter::repeat(item).take(self.repeats + 1))
            .collect();
        ChannelIter::from_vec(channels)
    }

    fn in_channels(&self) -> ChannelIter {
        self.handle.in_channels()
    }

    fn node_ids(&self) -> NodeIdIter {
        self.handle.node_ids()
    }
}

/// Handle for a * operation
#[derive(Copy, Clone, Debug)]
pub struct MulHandle {
    node_id: NodeId,
    num_out_channels: usize,
}
impl HandleData for MulHandle {
    fn out_channels(&self) -> ChannelIter {
        ChannelIter::SingleNodeId {
            node_id: self.node_id,
            num_channels: self.num_out_channels,
            current_iter_index: 0,
        }
    }

    fn in_channels(&self) -> ChannelIter {
        ChannelIter::SingleNodeId {
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
    fn out_channels(&self) -> ChannelIter {
        ChannelIter::SingleNodeId {
            node_id: self.node_id,
            num_channels: self.num_out_channels,
            current_iter_index: 0,
        }
    }

    fn in_channels(&self) -> ChannelIter {
        ChannelIter::SingleNodeId {
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
    fn out_channels(&self) -> ChannelIter;
    /// All input channels of this `Handle` in order
    fn in_channels(&self) -> ChannelIter;
    /// All `NodeIds` referenced by this `Handle` in any order
    fn node_ids(&self) -> NodeIdIter;
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

/// An iterator over channels to connect to/from. Use internally by handles. You won't need to interact with it directly unless you are implementing a handle.
#[derive(Clone, Debug)]
pub enum ChannelIter {
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
        start_index: usize,
        #[allow(missing_docs)]
        num_channels: usize,
        #[allow(missing_docs)]
        current_channel: usize,
    },
    /// No channels
    None,
}
impl ChannelIter {
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
    GraphInput,
    #[allow(missing_docs)]
    Gen(NodeId),
}
impl Source {
    /// Create a connection to another node
    #[must_use]
    pub fn to(self, other: NodeId) -> Connection {
        match self {
            Source::GraphInput => Connection::graph_input(other),
            Source::Gen(node) => node.to(other),
        }
    }
    /// Create a connection to the graph output
    #[must_use]
    pub fn to_graph_out(self) -> Connection {
        match self {
            Source::GraphInput => Connection::GraphInputToOutput {
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
impl Iterator for ChannelIter {
    type Item = (Source, NodeChannel);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ChannelIter::SingleNodeId {
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
            ChannelIter::Vec {
                channels,
                current_index,
            } => {
                let item = channels.get(*current_index).map(|(id, chan)| (*id, *chan));
                *current_index += 1;
                item
            }
            ChannelIter::SingleChannel {
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
            ChannelIter::None => None,
            ChannelIter::GraphInput {
                start_index,
                num_channels,
                current_channel,
            } => {
                if current_channel == num_channels {
                    None
                } else {
                    *current_channel += 1;
                    Some((
                        Source::GraphInput,
                        NodeChannel::Index(*start_index + *current_channel - 1),
                    ))
                }
            }
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            ChannelIter::SingleNodeId {
                node_id: _,
                num_channels,
                current_iter_index,
            } => {
                let num_left = num_channels - current_iter_index;
                (num_left, Some(num_left))
            }
            ChannelIter::Vec {
                channels,
                current_index,
            } => {
                let num_left = channels.len() - current_index;
                (num_left, Some(num_left))
            }
            ChannelIter::SingleChannel { returned, .. } => {
                if *returned {
                    (0, Some(0))
                } else {
                    (1, Some(1))
                }
            }
            ChannelIter::None => (0, Some(0)),
            ChannelIter::GraphInput {
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

impl<A, B> Mul<Handle<A>> for Handle<B>
where
    A: Copy + HandleData,
    B: Copy + HandleData,
{
    type Output = Handle<MulHandle>;

    fn mul(self, rhs: Handle<A>) -> Self::Output {
        let num_out_channels = self.out_channels().collect::<Vec<_>>().len();
        let mul_id = commands().push_without_inputs(MulGen(num_out_channels));
        for (i, (out0, out1)) in self.out_channels().zip(rhs.out_channels()).enumerate() {
            commands().connect(out0.0.to(mul_id).from_channel(out0.1).to_channel(i * 2));
            commands().connect(out1.0.to(mul_id).from_channel(out1.1).to_channel(i * 2 + 1));
        }
        Handle::new(MulHandle {
            node_id: mul_id,
            num_out_channels,
        })
    }
}

impl<B: Copy + HandleData> Mul<Handle<B>> for Sample {
    type Output = Handle<MulHandle>;

    fn mul(self, rhs: Handle<B>) -> Self::Output {
        let num_out_channels = rhs.out_channels().collect::<Vec<_>>().len();
        let mul_id = commands().push_without_inputs(MulGen(num_out_channels));
        for (i, (out0, out1)) in std::iter::repeat(self).zip(rhs.out_channels()).enumerate() {
            commands().connect(
                crate::graph::connection::constant(out0)
                    .to(mul_id)
                    .to_channel(i * 2),
            );
            commands().connect(out1.0.to(mul_id).from_channel(out1.1).to_channel(i * 2 + 1));
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

// Mull AnyNodeHandle * Handle
impl<B> Mul<AnyNodeHandle> for Handle<B>
where
    B: Copy + HandleData,
{
    type Output = Handle<MulHandle>;

    fn mul(self, rhs: AnyNodeHandle) -> Self::Output {
        let num_out_channels = self.out_channels().collect::<Vec<_>>().len();
        let mul_id = commands().push_without_inputs(MulGen(num_out_channels));
        for (i, (out0, out1)) in self.out_channels().zip(rhs.out_channels()).enumerate() {
            commands().connect(out0.0.to(mul_id).from_channel(out0.1).to_channel(i * 2));
            commands().connect(out1.0.to(mul_id).from_channel(out1.1).to_channel(i * 2 + 1));
        }
        Handle::new(MulHandle {
            node_id: mul_id,
            num_out_channels,
        })
    }
}
impl<B> Mul<Handle<B>> for AnyNodeHandle
where
    B: Copy + HandleData,
{
    type Output = Handle<MulHandle>;

    fn mul(self, rhs: Handle<B>) -> Self::Output {
        let num_out_channels = self.out_channels().collect::<Vec<_>>().len();
        let mul_id = commands().push_without_inputs(MulGen(num_out_channels));
        for (i, (out0, out1)) in self.out_channels().zip(rhs.out_channels()).enumerate() {
            commands().connect(out0.0.to(mul_id).from_channel(out0.1).to_channel(i * 2));
            commands().connect(out1.0.to(mul_id).from_channel(out1.1).to_channel(i * 2 + 1));
        }
        Handle::new(MulHandle {
            node_id: mul_id,
            num_out_channels,
        })
    }
}

impl Mul<AnyNodeHandle> for Sample {
    type Output = Handle<MulHandle>;

    fn mul(self, rhs: AnyNodeHandle) -> Self::Output {
        let num_out_channels = rhs.out_channels().collect::<Vec<_>>().len();
        let mul_id = commands().push_without_inputs(MulGen(num_out_channels));
        for (i, (out0, out1)) in std::iter::repeat(self).zip(rhs.out_channels()).enumerate() {
            commands().connect(
                crate::graph::connection::constant(out0)
                    .to(mul_id)
                    .to_channel(i * 2),
            );
            commands().connect(out1.0.to(mul_id).from_channel(out1.1).to_channel(i * 2 + 1));
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

// Add
impl<A, B> Add<Handle<A>> for Handle<B>
where
    A: Copy + HandleData,
    B: Copy + HandleData,
{
    type Output = Handle<AddHandle>;

    fn add(self, rhs: Handle<A>) -> Self::Output {
        let num_out_channels = self.out_channels().collect::<Vec<_>>().len();
        let node_id = commands().push_without_inputs(Bus(num_out_channels));
        for (i, (out0, out1)) in self.out_channels().zip(rhs.out_channels()).enumerate() {
            commands().connect(out0.0.to(node_id).from_channel(out0.1).to_channel(i));
            commands().connect(out1.0.to(node_id).from_channel(out1.1).to_channel(i));
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
        let node_id = commands().push_without_inputs(Bus(num_out_channels));
        for (i, (out0, out1)) in std::iter::repeat(self).zip(rhs.out_channels()).enumerate() {
            commands().connect(
                crate::graph::connection::constant(out0)
                    .to(node_id)
                    .to_channel(i),
            );
            commands().connect(out1.0.to(node_id).from_channel(out1.1).to_channel(i));
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
        let node_id = commands().push_without_inputs(Bus(num_out_channels));
        for (i, (out0, out1)) in self.out_channels().zip(rhs.out_channels()).enumerate() {
            commands().connect(out0.0.to(node_id).from_channel(out0.1).to_channel(i));
            commands().connect(out1.0.to(node_id).from_channel(out1.1).to_channel(i));
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
        let num_out_channels = self.out_channels().collect::<Vec<_>>().len();
        let node_id = commands().push_without_inputs(Bus(num_out_channels));
        for (i, (out0, out1)) in self.out_channels().zip(rhs.out_channels()).enumerate() {
            commands().connect(out0.0.to(node_id).from_channel(out0.1).to_channel(i));
            commands().connect(out1.0.to(node_id).from_channel(out1.1).to_channel(i));
        }
        Handle::new(AddHandle {
            node_id,
            num_out_channels,
        })
    }
}

impl Add<AnyNodeHandle> for Sample {
    type Output = Handle<AddHandle>;

    fn add(self, rhs: AnyNodeHandle) -> Self::Output {
        let num_out_channels = rhs.out_channels().collect::<Vec<_>>().len();
        let node_id = commands().push_without_inputs(Bus(num_out_channels));
        for (i, (out0, out1)) in std::iter::repeat(self).zip(rhs.out_channels()).enumerate() {
            commands().connect(
                crate::graph::connection::constant(out0)
                    .to(node_id)
                    .to_channel(i),
            );
            commands().connect(out1.0.to(node_id).from_channel(out1.1).to_channel(i));
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
    in_channel_iter: ChannelIter,
    out_channel_iter: ChannelIter,
    node_ids: NodeIdIter,
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
    fn out_channels(&self) -> ChannelIter {
        self.out_channel_iter.clone()
    }

    fn in_channels(&self) -> ChannelIter {
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
    Handle { output_channels: ChannelIter },
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
    fn out_channels(&self) -> ChannelIter {
        ChannelIter::SingleNodeId {
            node_id: self.node_id,
            num_channels: self.num_out_channels,
            current_iter_index: 0,
        }
    }

    fn in_channels(&self) -> ChannelIter {
        ChannelIter::SingleNodeId {
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
        let node_id = commands().push_without_inputs(RampGen(num_out_channels));
        match min.into() {
            Input::Constant(c) => {
                commands().connect(
                    crate::graph::connection::constant(c)
                        .to(node_id)
                        .to_channel(0),
                );
            }
            Input::Handle {
                ref mut output_channels,
            } => {
                if let Some((node, chan)) = output_channels.next() {
                    commands().connect(node.to(node_id).from_channel(chan).to_channel(0));
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
                commands().connect(
                    crate::graph::connection::constant(c)
                        .to(node_id)
                        .to_channel(1),
                );
            }
            Input::Handle {
                ref mut output_channels,
            } => {
                if let Some((node, chan)) = output_channels.next() {
                    commands().connect(node.to(node_id).from_channel(chan).to_channel(1));
                } else {
                    // TODO: Error: empty handle as max input to ramp
                }
                if output_channels.next().is_some() {
                    // TODO: Warn: multi channel handle as maxinput to ramp
                }
            }
        }
        for (i, out0) in input.out_channels().enumerate() {
            commands().connect(out0.0.to(node_id).from_channel(out0.1).to_channel(i + 2));
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
    start_index: usize,
    num_channels: usize,
}

impl HandleData for GraphInputHandle {
    fn out_channels(&self) -> ChannelIter {
        ChannelIter::GraphInput {
            start_index: self.start_index,
            num_channels: self.num_channels,
            current_channel: 0,
        }
    }

    fn in_channels(&self) -> ChannelIter {
        ChannelIter::None
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
                commands().connect(
                    node_id
                        .to_graph_out()
                        .from_channel(chan)
                        .to_channel(index + i),
                );
            }
        }
    }
}

/// A handle type that can refer to any Gen and still be Copy. It knows only how many output/input channels the Gen has, not their labels. You can still use labels via the `set` method.
#[derive(Copy, Clone)]
pub struct GenericHandle {
    node_id: NodeId,
    num_inputs: usize,
    num_outputs: usize,
}
impl GenericHandle {
    pub fn new(node_id: NodeId, num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            node_id,
            num_inputs,
            num_outputs,
        }
    }
    /// The non-typed way to set an input channel's value
    pub fn set(
        self,
        channel: impl Into<NodeChannel>,
        input: impl Into<Input>,
    ) -> Handle<GenericHandle> {
        let inp = input.into();
        let channel = channel.into();
        match inp {
            Input::Constant(v) => {
                commands().connect(
                    crate::graph::connection::constant(v)
                        .to(self.node_id)
                        .to_channel(channel),
                );
            }
            Input::Handle { output_channels } => {
                for (node_id, chan) in output_channels {
                    commands().connect(
                        node_id
                            .to(self.node_id)
                            .from_channel(chan)
                            .to_channel(channel),
                    );
                }
            }
        }
        Handle::new(self)
    }
    /// The non-typed way to send a trigger to an input channel
    pub fn trig(self, channel: impl Into<NodeChannel>) -> Handle<GenericHandle> {
        // TODO: better way to send a trigger
        let change = ParameterChange::now(self.node_id.input(channel), Change::Trigger);
        commands().schedule_change(change);
        Handle::new(self)
    }
}
impl HandleData for GenericHandle {
    fn out_channels(&self) -> ChannelIter {
        ChannelIter::SingleNodeId {
            node_id: self.node_id,
            num_channels: self.num_outputs,
            current_iter_index: 0,
        }
    }

    fn in_channels(&self) -> ChannelIter {
        ChannelIter::single_node_id(self.node_id, self.num_inputs)
    }

    fn node_ids(&self) -> NodeIdIter {
        NodeIdIter::Single(self.node_id)
    }
}
/// Upload the Gen to the Graph and return a `GenericHandle` for routing and setting inputs.
///
/// This is useful when the Gen you want to add doesn't have it's own handle function for some reason.
pub fn handle(gen: impl GenOrGraph) -> Handle<GenericHandle> {
    let num_inputs = gen.num_inputs();
    let num_outputs = gen.num_outputs();
    let node_id = commands().push_without_inputs(gen);
    Handle::new(GenericHandle {
        node_id,
        num_inputs,
        num_outputs,
    })
}

/// A `Handle` to a `Graph`
#[derive(Copy, Clone)]
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
}
impl HandleData for GraphHandle {
    fn out_channels(&self) -> ChannelIter {
        ChannelIter::SingleNodeId {
            node_id: self.node_id,
            num_channels: self.num_outputs,
            current_iter_index: 0,
        }
    }

    fn in_channels(&self) -> ChannelIter {
        ChannelIter::single_node_id(self.node_id, self.num_inputs)
    }

    fn node_ids(&self) -> NodeIdIter {
        NodeIdIter::Single(self.node_id)
    }
}

#[must_use]
/// Create a Bus Gen with the selected number of channels. One use case is as a single node for setting values later. Another is as a passthrough node for gathering multiple variable sources.
pub fn bus(num_channels: usize) -> Handle<GenericHandle> {
    let node_id = commands().push_without_inputs(Bus(num_channels));
    Handle::new(GenericHandle {
        node_id,
        num_inputs: num_channels,
        num_outputs: num_channels,
    })
}
