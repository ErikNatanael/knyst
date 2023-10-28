use std::{
    any::Any,
    ops::{Add, Deref, Mul},
};

use knyst_core::Sample;

use crate::{
    graph::{connection::NodeChannel, Bus, MulGen, NodeId},
    modal_interface::commands,
    prelude::KnystCommands,
};

#[derive(Copy, Clone, Debug)]
pub struct NodeHandle<H: Copy> {
    handle: H,
}

impl<H: Copy> Deref for NodeHandle<H> {
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
impl<A: Copy + NodeHandleData> NodeHandle<A> {
    pub fn new(handle: A) -> Self {
        Self { handle }
    }
    pub fn repeat_outputs(self, n: usize) -> NodeHandle<RepeatOutputs<A>> {
        NodeHandle::new(RepeatOutputs {
            handle: self.handle,
            repeats: n,
        })
    }
}
impl<H: NodeHandleData + Copy> NodeHandleData for NodeHandle<H> {
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
pub struct RepeatOutputs<H: Copy + NodeHandleData> {
    handle: H,
    repeats: usize,
}
impl<H: Copy + NodeHandleData> NodeHandleData for RepeatOutputs<H> {
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
impl NodeHandleData for MulHandle {
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
impl NodeHandleData for AddHandle {
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

pub trait NodeHandleData {
    /// All output channels of this `NodeHandle` in order
    fn out_channels(&self) -> ChannelIter;
    /// All input channels of this `NodeHandle` in order
    fn in_channels(&self) -> ChannelIter;
    /// All `NodeIds` referenced by thi`NodeHandle` in any order
    fn node_ids(&self) -> NodeIdIter;
}
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
#[derive(Debug)]
pub enum ChannelIter {
    SingleNodeId {
        node_id: NodeId,
        num_channels: usize,
        current_iter_index: usize,
    },
    Vec {
        channels: Vec<(NodeId, NodeChannel)>,
        current_index: usize,
    },
}
impl ChannelIter {
    pub fn from_vec(channels: Vec<(NodeId, NodeChannel)>) -> Self {
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
}
impl Iterator for ChannelIter {
    type Item = (NodeId, NodeChannel);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ChannelIter::SingleNodeId {
                node_id,
                num_channels,
                current_iter_index,
            } => {
                if *current_iter_index < *num_channels {
                    *current_iter_index += 1;
                    Some((*node_id, NodeChannel::Index(*current_iter_index - 1)))
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
        }
    }
}

impl<A, B> Mul<NodeHandle<A>> for NodeHandle<B>
where
    A: Copy + NodeHandleData,
    B: Copy + NodeHandleData,
{
    type Output = NodeHandle<MulHandle>;

    fn mul(self, rhs: NodeHandle<A>) -> Self::Output {
        let num_out_channels = self.out_channels().collect::<Vec<_>>().len();
        let mul_id = commands().push_without_inputs(MulGen(num_out_channels));
        for (i, (out0, out1)) in self.out_channels().zip(rhs.out_channels()).enumerate() {
            commands().connect(out0.0.to(mul_id).from_channel(out0.1).to_channel(i * 2));
            commands().connect(out1.0.to(mul_id).from_channel(out1.1).to_channel(i * 2 + 1));
        }
        NodeHandle::new(MulHandle {
            node_id: mul_id,
            num_out_channels,
        })
    }
}

impl<B: Copy + NodeHandleData> Mul<NodeHandle<B>> for f32 {
    type Output = NodeHandle<MulHandle>;

    fn mul(self, rhs: NodeHandle<B>) -> Self::Output {
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
        NodeHandle::new(MulHandle {
            node_id: mul_id,
            num_out_channels,
        })
    }
}
impl<B: Copy + NodeHandleData> Mul<f32> for NodeHandle<B> {
    type Output = NodeHandle<MulHandle>;

    fn mul(self, rhs: f32) -> Self::Output {
        rhs * self
    }
}
// Add
impl<A, B> Add<NodeHandle<A>> for NodeHandle<B>
where
    A: Copy + NodeHandleData,
    B: Copy + NodeHandleData,
{
    type Output = NodeHandle<AddHandle>;

    fn add(self, rhs: NodeHandle<A>) -> Self::Output {
        let num_out_channels = self.out_channels().collect::<Vec<_>>().len();
        let node_id = commands().push_without_inputs(Bus(num_out_channels));
        for (i, (out0, out1)) in self.out_channels().zip(rhs.out_channels()).enumerate() {
            commands().connect(out0.0.to(node_id).from_channel(out0.1).to_channel(i));
            commands().connect(out1.0.to(node_id).from_channel(out1.1).to_channel(i));
        }
        NodeHandle::new(AddHandle {
            node_id,
            num_out_channels,
        })
    }
}

impl<B: Copy + NodeHandleData> Add<NodeHandle<B>> for f32 {
    type Output = NodeHandle<AddHandle>;

    fn add(self, rhs: NodeHandle<B>) -> Self::Output {
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
        NodeHandle::new(AddHandle {
            node_id,
            num_out_channels,
        })
    }
}
impl<B: Copy + NodeHandleData> Add<f32> for NodeHandle<B> {
    type Output = NodeHandle<AddHandle>;

    fn add(self, rhs: f32) -> Self::Output {
        rhs + self
    }
}
/// A safe way to store a NodeHandle without the generic parameter, but keep being able to use it
pub struct AnyNodeHandle {
    org_handle: Box<dyn Any>,
    in_channel_iter: ChannelIter,
    out_channel_iter: ChannelIter,
    node_ids: NodeIdIter,
}

impl<H: Copy + NodeHandleData + 'static> From<NodeHandle<H>> for AnyNodeHandle {
    fn from(value: NodeHandle<H>) -> Self {
        AnyNodeHandle {
            org_handle: Box::new(value),
            in_channel_iter: value.in_channels(),
            out_channel_iter: value.out_channels(),
            node_ids: value.node_ids(),
        }
    }
}
/// Gathering all kinds of inputs into one
pub enum Input {
    Constant(Sample),
    Handle { output_channels: ChannelIter },
}
impl From<f32> for Input {
    fn from(value: f32) -> Self {
        Input::Constant(value)
    }
}
impl<H: Copy + NodeHandleData> From<NodeHandle<H>> for Input {
    fn from(value: NodeHandle<H>) -> Self {
        Input::Handle {
            output_channels: value.out_channels(),
        }
    }
}
impl From<&AnyNodeHandle> for Input {
    fn from(value: &AnyNodeHandle) -> Self {
        todo!()
    }
}

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
