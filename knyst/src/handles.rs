use std::{
    any::Any,
    ops::{Add, Deref, Mul},
};

use crate::{
    graph::{Change, Connection, ParameterChange, SimultaneousChanges},
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
    pub fn void() -> Handle<GenericHandle> {
        Self::new(GenericHandle {
            node_id: NodeId::new(),
            num_inputs: 0,
            num_outputs: 0,
        })
    }
}
impl<A: Copy + HandleData> Handle<A> {
    pub fn new(handle: A) -> Self {
        Self { handle }
    }
    pub fn repeat_outputs(self, n: usize) -> Handle<RepeatOutputs<A>> {
        Handle::new(RepeatOutputs {
            handle: self.handle,
            repeats: n,
        })
    }
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

pub trait HandleData {
    /// All output channels of this `Handle` in order
    fn out_channels(&self) -> ChannelIter;
    /// All input channels of this `Handle` in order
    fn in_channels(&self) -> ChannelIter;
    /// All `NodeIds` referenced by this `Handle` in any order
    fn node_ids(&self) -> NodeIdIter;
}
#[derive(Clone)]
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
#[derive(Clone, Debug)]
pub enum ChannelIter {
    SingleNodeId {
        node_id: NodeId,
        num_channels: usize,
        current_iter_index: usize,
    },
    Vec {
        channels: Vec<(Node, NodeChannel)>,
        current_index: usize,
    },
    SingleChannel {
        node_id: NodeId,
        channel: NodeChannel,
        returned: bool,
    },
    GraphInput {
        start_index: usize,
        num_channels: usize,
        current_channel: usize,
    },
    None,
}
impl ChannelIter {
    pub fn from_vec(channels: Vec<(Node, NodeChannel)>) -> Self {
        Self::Vec {
            channels,
            current_index: 0,
        }
    }
    pub fn single_node_id(node_id: NodeId, num_channels: usize) -> Self {
        Self::SingleNodeId {
            node_id,
            num_channels,
            current_iter_index: 0,
        }
    }
    pub fn single_channel(node_id: NodeId, channel: NodeChannel) -> Self {
        Self::SingleChannel {
            node_id,
            channel,
            returned: false,
        }
    }
}
#[derive(Copy, Clone, Debug)]
pub enum Node {
    GraphInput,
    Gen(NodeId),
}
impl Node {
    pub fn to(self, other: NodeId) -> Connection {
        match self {
            Node::GraphInput => Connection::graph_input(other),
            Node::Gen(node) => node.to(other),
        }
    }
    pub fn to_graph_out(self) -> Connection {
        match self {
            Node::GraphInput => Connection::GraphInputToOutput {
                from_input_channel: 0,
                to_output_channel: 0,
                channels: 1,
            },
            Node::Gen(node) => node.to_graph_out(),
        }
    }
}
impl From<NodeId> for Node {
    fn from(value: NodeId) -> Self {
        Self::Gen(value)
    }
}
impl From<&mut NodeId> for Node {
    fn from(value: &mut NodeId) -> Self {
        Self::Gen(*value)
    }
}
impl From<&NodeId> for Node {
    fn from(value: &NodeId) -> Self {
        Self::Gen(*value)
    }
}
impl Iterator for ChannelIter {
    type Item = (Node, NodeChannel);

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
                        Node::GraphInput,
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
                start_index,
                num_channels,
                current_channel,
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
/// A safe way to store a Handle without the generic parameter, but keep being able to use it
pub struct AnyNodeHandle {
    org_handle: Box<dyn Any>,
    in_channel_iter: ChannelIter,
    out_channel_iter: ChannelIter,
    node_ids: NodeIdIter,
}

impl<H: Copy + HandleData + 'static> From<Handle<H>> for AnyNodeHandle {
    fn from(value: Handle<H>) -> Self {
        AnyNodeHandle {
            org_handle: Box::new(value),
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
    Constant(Sample),
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
    pub fn range(mut self, min: impl Into<Input>, max: impl Into<Input>) -> Handle<RangeHandle> {
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

/// Creates a handle to numer of consecutive graph_input channels.
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
        Input::Constant(v) => {
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

#[derive(Copy, Clone)]
pub struct GenericHandle {
    node_id: NodeId,
    num_inputs: usize,
    num_outputs: usize,
}
impl GenericHandle {
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
                for (i, (node_id, chan)) in output_channels.enumerate() {
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

#[derive(Copy, Clone)]
pub struct GraphHandle {
    node_id: NodeId,
    num_inputs: usize,
    num_outputs: usize,
}
impl GraphHandle {
    pub(crate) fn new(id: NodeId, num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            node_id: id,
            num_inputs,
            num_outputs,
        }
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
