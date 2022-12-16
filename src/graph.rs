//! [`Graph`] is the audio graph, the core of Knyst. Implement [`Gen`] and you can be a `Node`.
//!
//! To build an audio graph, add generators (anything implementing the [`Gen`]
//! trait) to a graph and add connections between them.
//!
//! ```
//! # use knyst::prelude::*;
//! # use knyst::wavetable::*;
//! let graph_settings = GraphSettings {
//!     block_size: 64,
//!     sample_rate: 44100.,
//!     num_outputs: 2,
//!     ..Default::default()
//! };
//! let mut graph = Graph::new(graph_settings);
//! // Adding a node gives you an address to that node
//! let sine_node_address = graph.push_gen(WavetableOscillatorOwned::new(Wavetable::sine()));
//! // Connecting the node to the graph output
//! graph.connect(sine_node_address.to_graph_out())?;
//! // You need to commit changes if the graph is running.
//! graph.commit_changes();
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//! To produce an output from the [`Graph`] we need to turn it into a node. If
//! you want to listen to the graph output the easiest way is to use and audio
//! backend. Have a look at [`RunGraph`] if you want to do non-realtime
//! synthesis or implement your own backend.

// #[cfg(loom)]
// use loom::cell::UnsafeCell;
#[cfg(loom)]
use loom::sync::atomic::Ordering;

// #[cfg(not(loom))]
use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet};
use std::mem;
#[cfg(not(loom))]
use std::sync::atomic::Ordering;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::Resources;
/// The graph consists of (simplified)
/// 1. a list of nodes
/// 2. lists of edges that are inputs per node, outputs of the graph and inputs from the graph input to a node
/// 3. lists of feedback edges that take the values of a node last iteration
///
/// The order that the nodes are processed is calculated based on the edges/connections between nodes using
/// depth-first search from the graph output. Nodes not connected to the graph output will still be run
/// because this is the most intuitive behaviour.
///
/// Each node contains a trait object on the heap with the sound generating object, a Box<dyn Gen> Each
/// edge/connection specifies between which output/input of the nodes data is mapped.

pub type Sample = f32;
pub type GraphId = u64;

/// Get a unique id for a Graph from this by using `fetch_add`
static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(0);

/// An address to a specific Node. The graph_id is constant indepentently of where the graph is (inside some
/// other graph), so it always points to a specific Node in a specific Graph.
#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq)]
pub struct NodeAddress {
    graph_id: GraphId,
    key: NodeKey,
}

impl NodeAddress {
    pub fn to(&self, sink_node: NodeAddress) -> Connection {
        Connection::Node {
            source: *self,
            from_index: Some(0),
            from_label: None,
            sink: sink_node,
            to_index: None,
            to_label: None,
            channels: 1,
            feedback: false,
        }
    }
    pub fn to_graph_out(&self) -> Connection {
        Connection::GraphOutput {
            source: *self,
            from_index: Some(0),
            from_label: None,
            to_index: 0,
            channels: 1,
        }
    }
    pub fn feedback_to(&self, sink_node: NodeAddress) -> Connection {
        Connection::Node {
            source: *self,
            from_index: Some(0),
            from_label: None,
            sink: sink_node,
            to_index: None,
            to_label: None,
            channels: 1,
            feedback: true,
        }
    }
}

pub struct GraphInput;
impl GraphInput {
    pub fn to(sink_node: NodeAddress) -> Connection {
        Connection::GraphInput {
            from_index: 0,
            sink: sink_node,
            to_index: None,
            to_label: None,
            channels: 1,
        }
    }
}

// graph.connect(sine.to(GraphOut))
// graph.connect(sine.to(GRAPH_OUT))
// graph.connect(sine.to_graph_out())
// graph.connect(GraphIn::to(sine).from_index(1).to_label("freq"))
// graph.connect(GRAPH_IN.to(sine).from_index(1).to_label("freq"))

#[derive(Clone, Copy)]
pub struct ParameterChange {
    pub time: TimeKind,
    pub node: NodeAddress,
    pub input_index: Option<usize>,
    pub input_label: Option<&'static str>,
    pub value: Sample,
}

impl ParameterChange {
    pub fn absolute_samples(node: NodeAddress, value: Sample, absolute_timestamp: u64) -> Self {
        Self {
            node,
            value,
            time: TimeKind::AbsoluteSample(absolute_timestamp),
            input_index: None,
            input_label: None,
        }
    }
    pub fn relative_duration(node: NodeAddress, value: Sample, from_now: Duration) -> Self {
        Self {
            node,
            value,
            time: TimeKind::DurationFromNow(from_now),
            input_index: None,
            input_label: None,
        }
    }
    pub fn now(node: NodeAddress, value: Sample) -> Self {
        Self {
            node,
            value,
            time: TimeKind::DurationFromNow(Duration::from_millis(0)),
            input_index: None,
            input_label: None,
        }
    }
    pub fn index(self, index: usize) -> Self {
        self.i(index)
    }
    pub fn i(mut self, index: usize) -> Self {
        self.input_index = Some(index);
        self.input_label = None;
        self
    }
    pub fn label(self, label: &'static str) -> Self {
        self.l(label)
    }
    pub fn l(mut self, label: &'static str) -> Self {
        self.input_label = Some(label);
        self.input_index = None;
        self
    }
}

#[derive(Clone, Copy)]
pub enum TimeKind {
    DurationFromNow(Duration),
    AbsoluteSample(u64),
}

/// Connection provides a convenient API for creating connections between nodes in a
/// graph. When used for nodes in a running graph, the shadow engine will translate the
/// Connection to a RtConnection which contains the full Path for finding the correct Graph
/// as fast as possible.
///
/// A node can have any number of connections to/from other nodes or outputs.
/// Multiple constant values will result in the sum of all the constants.
#[derive(Clone, Copy)]
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
    pub fn graph_output(source_node: NodeAddress) -> Self {
        Self::GraphOutput {
            source: source_node,
            from_index: Some(0),
            from_label: None,
            to_index: 0,
            channels: 1,
        }
    }
    pub fn graph_input(sink_node: NodeAddress) -> Self {
        Self::GraphInput {
            sink: sink_node,
            from_index: 0,
            to_index: None,
            to_label: None,
            channels: 1,
        }
    }
    pub fn clear_constants(node: NodeAddress) -> Self {
        Self::Clear {
            node,
            input_constants: true,
            input_nodes: false,
            output_nodes: false,
            graph_outputs: false,
            graph_inputs: false,
        }
    }
    pub fn clear_inputs(node: NodeAddress) -> Self {
        Self::Clear {
            node,
            input_constants: false,
            input_nodes: true,
            output_nodes: false,
            graph_outputs: false,
            graph_inputs: false,
        }
    }
    pub fn clear_node_outputs(node: NodeAddress) -> Self {
        Self::Clear {
            node,
            input_constants: false,
            input_nodes: false,
            output_nodes: true,
            graph_outputs: false,
            graph_inputs: false,
        }
    }
    pub fn clear_graph_outputs(node: NodeAddress) -> Self {
        Self::Clear {
            node,
            input_constants: false,
            input_nodes: false,
            output_nodes: false,
            graph_outputs: true,
            graph_inputs: false,
        }
    }
    pub fn clear_graph_inputs(node: NodeAddress) -> Self {
        Self::Clear {
            node,
            input_constants: false,
            input_nodes: false,
            output_nodes: false,
            graph_outputs: false,
            graph_inputs: true,
        }
    }
    /// Sets the source of a Connection. Only valid for Connection::Node and
    /// Connection::GraphOutput. On other variants it does nothing.
    pub fn from(mut self, source_node: NodeAddress) -> Self {
        match &mut self {
            Connection::Node { source, .. } => {
                *source = source_node;
            }
            Connection::GraphInput { .. } => {}
            Connection::Constant { .. } => {}
            Connection::GraphOutput { source, .. } => {
                *source = source_node;
            }
            Connection::Clear { .. } => {}
        }
        self
    }
    /// Sets the source of a Connection. Does nothing on a
    /// Connection::GraphOutput or Connection::Clear.
    pub fn to(mut self, sink_node: NodeAddress) -> Self {
        match &mut self {
            Connection::Node { sink, .. } => {
                *sink = sink_node;
            }
            Connection::GraphInput { sink, .. } => {
                *sink = sink_node;
            }
            Connection::Constant { sink, .. } => {
                *sink = Some(sink_node);
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
            Connection::Node { source, .. } => Some(*source),
            Connection::GraphOutput { source, .. } => Some(*source),
            Connection::GraphInput { .. }
            | Connection::Constant { .. }
            | Connection::Clear { .. } => None,
        }
    }
}

/// One task to complete, for the node graph Safety: Uses raw pointers to nodes
/// and buffers. A node and its buffers may not be touched from the Graph while
/// a Task containing pointers to it is running. This is guaranteed by an atomic
/// generation counter in GraphGen/GraphGenCommunicator to allow the Graph to
/// free nodes once they are no longer used, and by the Arc pointer to the nodes
/// owned by both Graph and GraphGen so that if Graph is dropped, the pointers
/// are still valid.
struct Task {
    /// The node key may be used to send a message to the Graph to free the node in this Task
    node_key: NodeKey,
    input_constants: *mut [Sample],
    /// inputs to copy from the graph inputs (whole buffers) in the form `(from_graph_input_index, to_node_input_index)`
    graph_inputs_to_copy: Vec<(usize, usize)>,
    /// list of tuples of single floats in the form `(from, to)` where the `from` points to an output of a different node and the `to` points to the input buffer.
    inputs_to_copy: Vec<(*mut Sample, *mut Sample)>,
    input_buffers: NodeBufferRef,
    gen: *mut dyn Gen,
    output_buffers_first_ptr: *mut Sample,
    block_size: usize,
    num_outputs: usize,
}
impl Task {
    fn init_constants(&mut self) {
        // Copy all constants
        let node_constants = unsafe { &*self.input_constants };
        for (channel, &constant) in node_constants.iter().enumerate() {
            self.input_buffers.fill_channel(constant, channel);
        }
    }
    fn apply_constant_change(&mut self, change: &ScheduledChange, start_sample_in_block: usize) {
        match change.kind {
            ScheduledChangeKind::Constant { index, value } => {
                let node_constants = unsafe { &mut *self.input_constants };
                node_constants[index] = value;
                for i in start_sample_in_block..self.input_buffers.block_size() {
                    self.input_buffers.write(value, index, i);
                }
            }
        }
    }
    fn run(&mut self, graph_inputs: &NodeBufferRef, resources: &mut Resources) -> GenState {
        // let node = unsafe { &mut *self.node_ptr };
        // Copy all inputs
        for (from, to) in &self.inputs_to_copy {
            unsafe {
                **to += **from;
            }
        }
        // Copy all graph inputs
        for (graph_input_index, node_input_index) in &self.graph_inputs_to_copy {
            for i in 0..self.input_buffers.block_size {
                self.input_buffers.write(
                    graph_inputs.read(*graph_input_index, i),
                    *node_input_index,
                    i,
                );
            }
        }
        // Process node
        // node.process(&self.input_buffers, resources)
        let mut outputs = NodeBufferRef {
            buf: self.output_buffers_first_ptr,
            num_channels: self.num_outputs,
            block_size: self.block_size,
        };
        let ctx = GenContext {
            inputs: &self.input_buffers,
            outputs: &mut outputs,
        };
        assert!(!self.gen.is_null());
        unsafe { (*self.gen).process(ctx, resources) }
    }
}

unsafe impl Send for Task {}

/// Copy the entire channel from `input_index` from `input_buffers` to the
/// `graph_output_index` channel of the outputs buffer. Used to copy from a node
/// to the output of a Graph.
struct OutputTask {
    input_buffers: NodeBufferRef,
    input_index: usize,
    graph_output_index: usize,
}
unsafe impl Send for OutputTask {}

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
    #[error("The connection change required freeing a node, but the node could not be freed.")]
    NodeFree(#[from] FreeError),
}

#[derive(thiserror::Error, Debug, PartialEq)]
pub enum FreeError {
    #[error("The graph containing the NodeAdress provided was not found. The node itself may or may not exist.")]
    GraphNotFound,
    #[error("The NodeAddress does not exist. The Node may have been freed already.")]
    NodeNotFound,
    #[error("The free action required making a new connection, but the connection failed.")]
    ConnectionError(#[from] Box<ConnectionError>),
}
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    #[error("The graph containing the NodeAdress provided was not found. The node itself may or may not exist.")]
    GraphNotFound,
    #[error("The NodeAddress does not exist. The Node may have been freed already.")]
    NodeNotFound,
    #[error("The input label specified was not registered for the node: `{0}`")]
    InputLabelNotFound(&'static str),
    #[error("No scheduler was created for the Graph so the change cannot be scheduled. This is likely because this Graph was not yet added to another Graph or split into a Node.")]
    SchedulerNotCreated,
}

/// This trait is not meant to be implemented by users. In almost all situations
/// you instead want to implement the [`Gen`] trait.
///
/// ToNode allows us to generically push either something that implements Gen or
/// a Graph using the same API.
pub trait GenOrGraph {
    fn components(self) -> (Option<Graph>, Box<dyn Gen + Send>);
}

impl<T: Gen + Send + 'static> GenOrGraph for T {
    fn components(self) -> (Option<Graph>, Box<dyn Gen + Send>) {
        (None, Box::new(self))
    }
}
impl GenOrGraph for Graph {
    fn components(mut self) -> (Option<Graph>, Box<dyn Gen + Send>) {
        if self.block_size() != self.block_size() {
            panic!("Warning: You are pushing a graph with a different block size. The library is not currently equipped to handle this. In a future version this will work seamlesly.")
        }
        if self.sample_rate != self.sample_rate {
            eprintln!("Warning: You are pushing a graph with a different sample rate. This is currently allowed, but expect bugs unless you deal with resampling manually.")
        }
        // Create the GraphGen from the new Graph
        let gen = self.create_graph_gen().unwrap();
        (Some(self), Box::new(gen))
    }
}

/// If it implements Gen, it can be a `Node` in a [`Graph`].
pub trait Gen {
    /// The input and output buffers are both indexed using \[in/out_index\]\[sample_index\].
    ///
    /// - *inputs*: The inputs to the Gen filled with the relevant values. May be any size the same or larger
    /// than the number of inputs to this particular Gen.
    ///
    /// - *outputs*: The buffer to place the result of the Gen inside. This buffer may contain any data and
    /// will not be zeroed. If the output should be zero, the Gen needs to write zeroes into the output
    /// buffer. This buffer will be correctly sized to hold the number of outputs that the Gen requires.
    fn process(&mut self, ctx: GenContext, resources: &mut Resources) -> GenState;
    /// The number of inputs this `Gen` takes. Determines how big the input buffer is.
    fn num_inputs(&self) -> usize;
    /// The number of outputs this `Gen` produces. Determines how big the output buffer is.
    fn num_outputs(&self) -> usize;
    /// Initialize buffers etc.
    /// Default: noop
    fn init(&mut self, _sample_rate: Sample) {}
    /// Return a label for a given input channel index. This sets the label in the [`Connection`] API.
    fn input_desc(&self, _input: usize) -> &'static str {
        ""
    }
    /// Return a label for a given output channel index. This sets the label in the [`Connection`] API.
    fn output_desc(&self, _output: usize) -> &'static str {
        ""
    }
    /// A name identifying this `Gen`.
    fn name(&self) -> &'static str {
        "no_name"
    }
}

/// Wrapper around a buffer of output samples. Panics at improper usage.
///
/// The number of channels corresponds to the number of inputs/outputs specified
/// for the [`Gen`] this is given to. Trying to access channels outside of what
/// has been declared will result in a panic at runtime.
///
/// The NodeBuffer does not own the data it points to, it is just a wrapper
/// around a pointer to the buffer to make accessing it more convenient.
///
/// The samples are stored in a `*mut [Sample]` layout out one channel after the
/// other as if it was a `*mut[*mut [Sample]]`. In benchmarks this was as fast as
/// iterating over multiple `&mut[Sample]`
///
pub struct NodeBufferRef {
    buf: *mut Sample,
    num_channels: usize,
    block_size: usize,
}
impl NodeBufferRef {
    /// Produces a buffer with 0 channels that you cannot read from or write to, but it may be useful as a first input to a node that has no inputs.
    pub fn null_buffer() -> Self {
        Self {
            buf: std::ptr::null_mut(),
            num_channels: 0,
            block_size: 0,
        }
    }
    #[inline]
    pub fn write(&mut self, value: f32, channel: usize, sample_index: usize) {
        assert!(channel < self.num_channels);
        assert!(sample_index < self.block_size);
        assert!(!self.buf.is_null());
        let buffer_slice = unsafe {
            std::slice::from_raw_parts_mut(self.buf, self.num_channels * self.block_size)
        };
        unsafe {
            *buffer_slice.get_unchecked_mut(channel * self.block_size + sample_index) = value
        };
    }
    /// Decouples the lifetime of the returned slice from self so that slices to different channels can be returned.
    #[inline]
    pub fn get_channel_mut<'a, 'b>(&'a mut self, channel: usize) -> &'b mut [f32] {
        assert!(channel < self.num_channels);
        assert!(!self.buf.is_null());
        let channel_offset = channel * self.block_size;
        let buffer_channel_slice = unsafe {
            let ptr_at_channel = self.buf.add(channel_offset);
            std::slice::from_raw_parts_mut(ptr_at_channel, self.block_size)
        };
        buffer_channel_slice
    }
    #[inline]
    pub fn get_channel(&self, channel: usize) -> &[f32] {
        assert!(channel < self.num_channels);
        assert!(!self.buf.is_null());
        let channel_offset = channel * self.block_size;
        let buffer_channel_slice = unsafe {
            let ptr_at_channel = self.buf.add(channel_offset);
            std::slice::from_raw_parts(ptr_at_channel, self.block_size)
        };
        buffer_channel_slice
    }
    #[inline]
    pub fn add(&mut self, value: f32, channel: usize, sample_index: usize) {
        assert!(channel < self.num_channels);
        assert!(sample_index < self.block_size);
        assert!(!self.buf.is_null());
        unsafe {
            let buffer_slice =
                std::slice::from_raw_parts_mut(self.buf, self.num_channels * self.block_size);
            let sample =
                (*buffer_slice).get_unchecked_mut(channel * self.block_size + sample_index);
            *sample += value
        };
    }
    #[inline]
    pub fn read(&self, channel: usize, sample_index: usize) -> Sample {
        assert!(channel < self.num_channels);
        assert!(sample_index < self.block_size);
        assert!(!self.buf.is_null());
        unsafe {
            let buffer_slice =
                std::slice::from_raw_parts_mut(self.buf, self.num_channels * self.block_size);
            *(*buffer_slice).get_unchecked(channel * self.block_size + sample_index)
        }
    }
    #[inline]
    fn fill_channel(&mut self, value: f32, channel: usize) {
        assert!(channel < self.num_channels);
        assert!(!self.buf.is_null());
        let channel_offset = channel * self.block_size;
        let buffer_channel_slice = unsafe {
            let ptr_at_channel = self.buf.add(channel_offset);
            std::slice::from_raw_parts_mut(ptr_at_channel, self.block_size)
        };
        for sample in buffer_channel_slice {
            *sample = value;
        }
    }
    #[inline]
    fn fill(&mut self, value: f32) {
        assert!(!self.buf.is_null());
        let buffer_slice = unsafe {
            std::slice::from_raw_parts_mut(self.buf, self.num_channels * self.block_size)
        };
        for sample in buffer_slice {
            *sample = value;
        }
    }
    fn ptr_to_sample(&mut self, channel: usize, sample_index: usize) -> *mut Sample {
        assert!(channel < self.num_channels);
        assert!(sample_index < self.block_size);
        unsafe { self.buf.add(channel * self.block_size + sample_index) }
    }
    pub fn channels(&self) -> usize {
        self.num_channels
    }
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

/// Gives access to the inputs and outputs buffers of a node for processing.
pub struct GenContext<'a, 'b> {
    pub inputs: &'a NodeBufferRef,
    pub outputs: &'b mut NodeBufferRef,
}
impl<'a, 'b> GenContext<'a, 'b> {
    /// Returns the current block size
    pub fn block_size(&self) -> usize {
        self.outputs.block_size
    }
}

type ProcessFn = Box<dyn FnMut(GenContext, &mut Resources) -> GenState + Send>;

/// Convenience struct to create a [`Gen`] from a closure.
///
/// Inputs and outputs are declared by chaining calls to the
/// [`ClosureGen::input`] and [`ClosureGen::output`] methods in the order the
/// inputs should be.
///
/// # Example
/// ```
/// use knyst::prelude::*;
/// use fastapprox::fast::tanh;
/// let closure_gen = gen(move |ctx, _resources| {
///     let out0 = ctx.outputs.get_channel_mut(0);
///     let out1 = ctx.outputs.get_channel_mut(1);
///     for ((((o0, o1), i0), i1), dist) in out0
///         .iter_mut()
///         .zip(out1.iter_mut())
///         .zip(ctx.inputs.get_channel(0).iter())
///         .zip(ctx.inputs.get_channel(1).iter())
///         .zip(ctx.inputs.get_channel(2).iter())
///     {
///         *o0 = tanh(*i0 * dist.max(1.0)) * 0.5;
///         *o1 = tanh(*i1 * dist.max(1.0)) * 0.5;
///     }
///     GenState::Continue
/// })
/// .output("out0")
/// .output("out1")
/// .input("in0")
/// .input("in1")
/// .input("distortion");
/// ```
pub struct ClosureGen {
    process_fn: ProcessFn,
    outputs: Vec<&'static str>,
    inputs: Vec<&'static str>,
    name: &'static str,
}
/// Alias for [`ClosureGen::new`]. See [`ClosureGen`] for more information.
pub fn gen(
    process: impl FnMut(GenContext, &mut Resources) -> GenState + 'static + Send,
) -> ClosureGen {
    ClosureGen {
        process_fn: Box::new(process),
        ..Default::default()
    }
}
impl ClosureGen {
    /// Create a [`ClosureGen`] with the given closure, 0 outputs and 0 inputs.
    /// Add inputs/outputs with the respective functions.
    pub fn new(
        process: impl FnMut(GenContext, &mut Resources) -> GenState + 'static + Send,
    ) -> Self {
        gen(process)
    }
    /// Adds an output. The order of outputs depends on the order they are added.
    pub fn output(mut self, output_name: &'static str) -> Self {
        self.outputs.push(output_name);
        self
    }
    /// Adds an input. The order of inputs depends on the order they are added.
    pub fn input(mut self, input_name: &'static str) -> Self {
        self.inputs.push(input_name);
        self
    }
    /// Set the name of the ClosureGen.
    pub fn name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }
}
impl Default for ClosureGen {
    fn default() -> Self {
        Self {
            process_fn: Box::new(|_ctx, _resources| GenState::Continue),
            outputs: Default::default(),
            inputs: Default::default(),
            name: "ClosureGen",
        }
    }
}

impl Gen for ClosureGen {
    fn process(&mut self, ctx: GenContext, resources: &mut Resources) -> GenState {
        (self.process_fn)(ctx, resources)
    }

    fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    fn init(&mut self, _sample_rate: Sample) {}

    fn input_desc(&self, input: usize) -> &'static str {
        self.inputs.get(input).unwrap_or(&"")
    }

    fn output_desc(&self, output: usize) -> &'static str {
        self.outputs.get(output).unwrap_or(&"")
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

/// The Node should return Continue unless it needs to free itself or the Graph it is in.
///
/// No promise is made as to when the node or the Graph will be freed so the Node needs to do the right thing
/// if run again the next block. E.g. a node returning `FreeSelfMendConnections` is expected to act as a
/// connection bridge from its non constant inputs to its outputs as if it weren't there. Only inputs with a
/// corresponding output should be passed through, e.g. in\[0\] -> out\[0\], in\[1\] -> out\[1\], in\[2..5\] go nowhere.
///
/// The FreeGraph and FreeGraphMendConnections values also return the relative
/// sample in the current block after which the graph should return 0 or connect
/// its non constant inputs to its outputs.
#[derive(Debug, Clone, Copy)]
pub enum GenState {
    /// Continue running
    Continue,
    /// Free the node containing the Gen
    FreeSelf,
    /// Free the node containing the Gen, bridging its input node(s) to its output node(s).
    FreeSelfMendConnections,
    /// Free the graph containing the node containing the Gen.
    FreeGraph(usize),
    /// Free the graph containing the node containing the Gen, bridging its input node(s) to its output node(s).
    FreeGraphMendConnections(usize),
}

new_key_type! {
    /// Node identifier in a specific Graph. For referring to a Node outside of the context of a Graph, use NodeAddress instead.
    struct NodeKey;
}

/// Pass to Graph::new to set the options the Graph is created with in an ergonomic and clear way.
#[derive(Clone, Copy, Debug)]
pub struct GraphSettings {
    /// The number of inputs to the Graph
    pub num_inputs: usize,
    /// The maximum number of inputs to a Node contained in the Graph
    pub max_node_inputs: usize,
    /// The number of outputs from the Graph
    pub num_outputs: usize,
    /// The block size this Graph uses for processing.
    pub block_size: usize,
    /// The maximum number of nodes that can be added to the graph.
    pub num_nodes: usize,
    /// The sample rate this Graph uses for processing.
    pub sample_rate: Sample,
    /// The number of messages that can be sent through any of the ring buffers.
    /// Ring buffers are used pass information back and forth between the audio
    /// thread (GraphGen) and the Graph.
    pub ring_buffer_size: usize,
    /// How much time is added to every *relative* scheduling event to ensure the Change has time to travel to the GraphGen.
    pub latency: Duration,
}

impl Default for GraphSettings {
    fn default() -> Self {
        GraphSettings {
            num_inputs: 0,
            num_outputs: 2,
            max_node_inputs: 8,
            block_size: 64,
            num_nodes: 1024,
            sample_rate: 48000.,
            ring_buffer_size: 100,
            latency: Duration::from_millis(4),
        }
    }
}

// Hold on to an allocation and drop it when we're done. Can be easily wrapped in an Arc
struct OwnedRawBuffer {
    ptr: *mut [f32],
}
impl Drop for OwnedRawBuffer {
    fn drop(&mut self) {
        unsafe { drop(Box::from_raw(self.ptr)) }
    }
}

/// A [`Graph`] contains nodes, which are wrappers around a dyn [`Gen`], and
/// connections between those nodes. Connections can eiterh be normal/forward
/// connections or feedback connections. Graphs can themselves be used as
/// [`Gen`]s in a different [`Graph`].
///
/// To run a [`Graph`] it has to be split so that parts of it are mirrored in a
/// `GraphGen` (private). This is done internally when calling [`Graph::push_graph`].
/// You can also do it yourself using a [`RunGraph`]. The [`Graph`] behaves
/// slightly differently when split:
///
/// - changes to constants are always scheduled to be performed by the
/// `GraphGen` instead of applied directly
/// - adding/freeing nodes/connections are scheduled to be done as soon as possible when it is safe to do so and [`Graph::commit_changes`] is called
///
/// # Manipulating the [`Graph`]
/// - [`Graph::push`] creates a node from a [`Gen`] or a [`Graph`], returning a [`NodeAddress`] which is a handle to that node.
/// - [`Graph::connect`] uses [`Connection`] to add or clear connections between nodes and the [`Graph`] they are in. You can also connect a constant value to a node input.
/// - [`Graph::commit_changes`] recalculates node order and applies changes to connections and nodes to the running `GraphGen` from the next time it is called. It also tries to free any resources it can that have been previously removed.
/// - [`Graph::schedule_change`] adds a parameter/input constant change to the scheduler.
/// - [`Graph::update`] updates the internal scheduler so that it sends scheduled updates that are soon due on to the `GraphGen`. This has to be called regularly if you are scheduling changes. Changes are only sent on to the GraphGen when they are soon due for performance reasons.
///
/// When a node has been added, it cannot be retreived from the [`Graph`]. You can, however, send any data you want out from the [`Gen::process`] trait method using your own synchronisation method e.g. a channel.
///
/// # Example
/// ```
/// use knyst::prelude::*;
/// use knyst::wavetable::*;
/// use knyst::graph::RunGraph;
/// let graph_settings = GraphSettings {
///     block_size: 64,
///     sample_rate: 44100.,
///     num_outputs: 2,
///     ..Default::default()
/// };
/// let mut graph = Graph::new(graph_settings);
/// let resources = Resources::new(ResourcesSettings {
///     sample_rate: 44100.,
///     ..Default::default()
/// });
/// let mut run_graph = RunGraph::new(&mut graph, resources)?;
/// // Adding a node gives you an address to that node
/// let sine_node_address = graph.push_gen(WavetableOscillatorOwned::new(Wavetable::sine()));
/// // Connecting the node to the graph output
/// graph.connect(sine_node_address.to_graph_out())?;
/// // Set the frequency of the oscillator to 220 Hz. This will
/// // be converted to a scheduled change because the graph is running.
/// graph.connect(constant(220.0).to(sine_node_address).to_label("freq"))?;
/// // You need to commit changes if the graph is running.
/// graph.commit_changes();
/// // You also need to update the scheduler to send messages to the audio thread.
/// graph.update();
/// // Process one block of audio data. If you are using an
/// // [`crate::AudioBackend`] you don't need to worry about this step.
/// run_graph.process_block();
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Graph {
    id: GraphId,
    nodes: Arc<UnsafeCell<SlotMap<NodeKey, Node>>>,
    node_keys_to_free_when_safe: Vec<(NodeKey, Arc<AtomicBool>)>,
    node_keys_pending_removal: HashSet<NodeKey>,
    /// A list of input edges for every node, sharing the same index as the node
    node_input_edges: SecondaryMap<NodeKey, Vec<Edge>>,
    node_input_index_to_name: SecondaryMap<NodeKey, Vec<&'static str>>,
    node_input_name_to_index: SecondaryMap<NodeKey, HashMap<&'static str, usize>>,
    node_output_index_to_name: SecondaryMap<NodeKey, Vec<&'static str>>,
    node_output_name_to_index: SecondaryMap<NodeKey, HashMap<&'static str, usize>>,
    /// List of feedback input edges for every node. The NodeKey in the tuple is the index of the FeedbackNode doing the buffering
    node_feedback_edges: SecondaryMap<NodeKey, Vec<FeedbackEdge>>,
    node_feedback_node_key: SecondaryMap<NodeKey, NodeKey>,
    node_order: Vec<NodeKey>,
    disconnected_nodes: Vec<NodeKey>,
    feedback_node_indices: Vec<NodeKey>,
    /// If a node is a graph, that graph will be added with the same key here.
    graphs_per_node: SecondaryMap<NodeKey, Graph>,
    /// The outputs of the Graph
    output_edges: Vec<Edge>,
    /// The edges from the graph inputs to nodes, one Vec per node. `source` in the edge is really the sink here.
    graph_input_edges: SecondaryMap<NodeKey, Vec<Edge>>,
    num_inputs: usize,
    num_outputs: usize,
    block_size: usize,
    sample_rate: Sample,
    ring_buffer_size: usize,
    initiated: bool,
    /// Used for processing every node, index using \[input_num\]\[sample_in_block\]
    // inputs_buffers: Vec<Box<[Sample]>>,
    /// A pointer to an allocation that is being used for the inputs to nodes, and aliased in the inputs_buffers
    inputs_buffers_ptr: Arc<OwnedRawBuffer>,
    max_node_inputs: usize,
    graph_gen_communicator: Option<GraphGenCommunicator>,
    /// The duration added to all changes scheduled to a relative time so that they have time to travel to the GraphGen.
    latency: Duration,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new(GraphSettings::default())
    }
}

impl Graph {
    pub fn new(options: GraphSettings) -> Self {
        let GraphSettings {
            num_inputs,
            num_outputs,
            max_node_inputs,
            block_size,
            num_nodes,
            sample_rate,
            ring_buffer_size,
            latency,
        } = options;
        let inputs_buffers_ptr = Box::<[Sample]>::into_raw(
            vec![0.0 as Sample; block_size * max_node_inputs].into_boxed_slice(),
        );
        let inputs_buffers_ptr = Arc::new(OwnedRawBuffer {
            ptr: inputs_buffers_ptr,
        });
        let id = NEXT_GRAPH_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let nodes = Arc::new(UnsafeCell::new(SlotMap::with_capacity_and_key(num_nodes)));
        let node_input_edges = SecondaryMap::with_capacity(num_nodes);
        let node_feedback_edges = SecondaryMap::with_capacity(num_nodes);
        let graph_input_edges = SecondaryMap::with_capacity(num_nodes);
        Self {
            id,
            nodes,
            node_input_edges,
            node_input_index_to_name: SecondaryMap::with_capacity(num_nodes),
            node_input_name_to_index: SecondaryMap::with_capacity(num_nodes),
            node_output_index_to_name: SecondaryMap::with_capacity(num_nodes),
            node_output_name_to_index: SecondaryMap::with_capacity(num_nodes),
            node_feedback_node_key: SecondaryMap::with_capacity(num_nodes),
            node_feedback_edges,
            node_order: Vec::with_capacity(num_nodes),
            disconnected_nodes: vec![],
            node_keys_to_free_when_safe: vec![],
            node_keys_pending_removal: HashSet::new(),
            feedback_node_indices: vec![],
            graphs_per_node: SecondaryMap::with_capacity(num_nodes),
            output_edges: vec![],
            graph_input_edges,
            num_inputs,
            num_outputs,
            block_size,
            sample_rate,
            latency,
            initiated: false,
            inputs_buffers_ptr,
            max_node_inputs,
            ring_buffer_size,
            graph_gen_communicator: None,
        }
    }
    /// Create a node that will run this graph. This will fail if a Node or Gen has already been created from the Graph since only one Gen is allowed to exist per Graph.
    ///
    /// Only use this for manually running the main Graph (the Graph containing all other Graphs). For adding a Graph to another Graph, use the push_graph() method.
    fn split_and_create_node(&mut self) -> Result<Node, String> {
        let block_size = self.block_size();
        let graph_gen = self.create_graph_gen()?;
        let mut node = Node::new("graph", Box::new(graph_gen));
        node.init(block_size, self.sample_rate);
        Ok(node)
    }

    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }
    /// Returns a number including both active nodes and nodes waiting to be safely freed
    pub fn num_stored_nodes(&self) -> usize {
        self.get_nodes().len()
    }
    pub fn push(&mut self, to_node: impl GenOrGraph) -> NodeAddress {
        let (graph, gen) = to_node.components();
        let address = self.push_node(Node::new(gen.name(), gen));
        if let Some(graph) = graph {
            self.graphs_per_node.insert(address.key, graph);
        }
        address
    }
    /// Add a graph as a node in this graph. This will allow you to change the Graph you added later on as needed.
    #[deprecated(since = "0.3.1", note = "please use `push` instead")]
    pub fn push_graph(&mut self, mut graph: Graph) -> NodeAddress {
        if graph.block_size() != self.block_size() {
            panic!("Warning: You are pushing a graph with a different block size. The library is not currently equipped to handle this. In a future version this will work seamlesly.")
        }
        if graph.sample_rate != self.sample_rate {
            eprintln!("Warning: You are pushing a graph with a different sample rate. This is currently allowed, but expect bugs unless you deal with resampling manually.")
        }
        // Create the GraphGen from the new Graph
        let gen = graph.create_graph_gen().unwrap();
        // Add the GraphGen to this Graph as a Node
        let address = self.push_gen_private(gen);
        // Add the Graph to this Graph's graph list
        self.graphs_per_node.insert(address.key, graph);
        address
    }
    /// Add anything that implements Gen to this Graph as a node.
    #[deprecated(since = "0.3.1", note = "please use `push` instead")]
    pub fn push_gen<G: Gen + Send + 'static>(&mut self, gen: G) -> NodeAddress {
        self.push_node(Node::new(gen.name(), Box::new(gen)))
    }
    /// Convert a Gen to a Node and add it to the Graph. Used to be public, but now it is private in favour of `Graph::push()`
    fn push_gen_private<G: Gen + Send + 'static>(&mut self, gen: G) -> NodeAddress {
        self.push_node(Node::new(gen.name(), Box::new(gen)))
    }
    /// Add a node to this Graph. The Node will be (re)initialised with the correct block size for this Graph.
    ///
    /// Making it not public means Graphs cannot be accidentally added, but a Node<Graph> can still be created for the top level one if preferred.
    fn push_node(&mut self, mut node: Node) -> NodeAddress {
        if node.num_inputs() > self.max_node_inputs {
            eprintln!("Warning: You are trying to add a node with more inputs than the maximum for this Graph. Try increasing the maximum number of node inputs in the GraphSettings.");
        }
        let nodes = self.get_nodes();
        if nodes.capacity() == nodes.len() {
            eprintln!("Error: Trying to push a node into a Graph that is at capacity. Try increasing the number of node slots and make sure you free the nodes you don't need.");
        }
        let input_index_to_name = node.input_indices_to_names();
        let input_name_to_index = input_index_to_name
            .iter()
            .enumerate()
            .map(|(i, &name)| (name, i))
            .collect();
        let output_index_to_name = node.output_indices_to_names();
        let output_name_to_index = output_index_to_name
            .iter()
            .enumerate()
            .map(|(i, &name)| (name, i))
            .collect();
        node.init(self.block_size, self.sample_rate);
        let key = self.get_nodes_mut().insert(node);
        self.node_input_edges.insert(key, vec![]);
        self.node_feedback_edges.insert(key, vec![]);
        self.graph_input_edges.insert(key, vec![]);
        self.node_input_index_to_name
            .insert(key, input_index_to_name);
        self.node_input_name_to_index
            .insert(key, input_name_to_index);
        self.node_output_index_to_name
            .insert(key, output_index_to_name);
        self.node_output_name_to_index
            .insert(key, output_name_to_index);
        NodeAddress {
            graph_id: self.id,
            key,
        }
    }
    /// Remove all nodes in this graph and all its subgraphs that are not connected to anything.
    pub fn free_disconnected_nodes(&mut self) -> Result<(), FreeError> {
        // The easiest way to do it would be to store disconnected nodes after
        // calculating the node order of the graph. (i.e. all the nodes that
        // weren't visited)

        let disconnected_nodes = std::mem::take(&mut self.disconnected_nodes);
        for node in disconnected_nodes {
            match self.free_node(NodeAddress {
                key: node,
                graph_id: self.id,
            }) {
                Ok(_) => (),
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
    /// Remove the node and connect its input edges to the sinks of its output edges
    pub fn free_node_mend_connections(
        &mut self,
        node: impl Into<NodeAddress>,
    ) -> Result<(), FreeError> {
        // For every input of the node, connect the nodes connected
        // to it to all the nodes taking input from the corresponding output.
        //
        // E.g. node1output1 -> input1, node2output2 -> input1, output1 ->
        // node3input2, output2 -> node4input3. Connect node1output1 and
        // node2output2 to node3input2. Since input2 has no connections nothing
        // is conencted to node4input3.
        //
        let node = node.into();
        if node.graph_id == self.id {
            // Does the Node exist?
            if !self.get_nodes_mut().contains_key(node.key) {
                return Err(FreeError::NodeNotFound);
            }
            let num_inputs = self.node_input_index_to_name.get(node.key).expect("Since the key exists in the Graph it should have a corresponding node_input_index_to_name Vec").len();
            let num_outputs= self.node_output_index_to_name.get(node.key).expect("Since the key exists in the Graph it should have a corresponding node_output_index_to_name Vec").len();
            let inputs_to_bridge = num_inputs.min(num_outputs);
            let mut outputs = vec![vec![]; inputs_to_bridge];
            for (destination_node_key, edge_vec) in &self.node_input_edges {
                for edge in edge_vec {
                    if edge.source == node.key && edge.from_output_index < inputs_to_bridge {
                        outputs[edge.from_output_index].push(Connection::Node {
                            source: NodeAddress {
                                graph_id: self.id,
                                key: node.key,
                            },
                            from_index: Some(edge.from_output_index),
                            from_label: None,
                            sink: NodeAddress {
                                graph_id: self.id,
                                key: destination_node_key,
                            },
                            to_index: Some(edge.to_input_index),
                            to_label: None,
                            channels: 1,
                            feedback: false,
                        });
                    }
                }
            }
            for graph_output in &self.output_edges {
                if graph_output.source == node.key
                    && graph_output.from_output_index < inputs_to_bridge
                {
                    outputs[graph_output.from_output_index].push(Connection::GraphOutput {
                        source: NodeAddress {
                            graph_id: self.id,
                            key: node.key,
                        },
                        from_index: Some(graph_output.from_output_index),
                        from_label: None,
                        to_index: graph_output.to_input_index,
                        channels: 1,
                    });
                }
            }
            let mut inputs = vec![vec![]; inputs_to_bridge];
            for (inout_index, bridge_input) in inputs.iter_mut().enumerate().take(inputs_to_bridge)
            {
                for input in self
                    .node_input_edges
                    .get(node.key)
                    .expect("Since the key exists in the Graph its edge Vec should also exist")
                {
                    if input.to_input_index == inout_index {
                        bridge_input.push(*input);
                    }
                }
            }
            let mut graph_inputs = vec![vec![]; inputs_to_bridge];
            for graph_input in self
                .graph_input_edges
                .get(node.key)
                .expect("Since the key exists in the graph its graph input Vec should also exist")
            {
                if graph_input.to_input_index < inputs_to_bridge {
                    graph_inputs[graph_input.to_input_index].push(Connection::GraphInput {
                        sink: NodeAddress {
                            graph_id: self.id,
                            key: node.key,
                        },
                        from_index: graph_input.from_output_index,
                        to_index: Some(graph_input.to_input_index),
                        to_label: None,
                        channels: 1,
                    });
                }
            }
            // We have all the edges, now connect them
            for (inputs, outputs) in inputs.into_iter().zip(outputs.iter()) {
                for input in inputs {
                    for output in outputs {
                        let connection = output;
                        self.connect(connection.from(NodeAddress {
                            key: input.source,
                            graph_id: self.id,
                        }))
                        .expect("Mended connections should be guaranteed to succeed");
                    }
                }
            }
            for (graph_inputs, outputs) in graph_inputs.into_iter().zip(outputs.iter()) {
                for input in graph_inputs {
                    for output in outputs {
                        let connection = input;
                        match self.connect(
                            connection
                                .to(output.get_source_node().unwrap())
                                .to_index(output.get_to_index().unwrap()),
                        ) {
                            Ok(_) => (),
                            Err(e) => return Err(FreeError::ConnectionError(Box::new(e))),
                        }
                    }
                }
            }
            // All connections have been mended/bridged, now free the node
            self.free_node(node)
        } else {
            // Try to find the graph containing the node by asking all the graphs in this graph to free the node
            for (_key, graph) in &mut self.graphs_per_node {
                match graph.free_node_mend_connections(node) {
                    Ok(_) => return Ok(()),
                    Err(e) => match e {
                        FreeError::GraphNotFound => (),
                        _ => return Err(e),
                    },
                }
            }
            Err(FreeError::GraphNotFound)
        }
    }
    /// Remove the node and any edges to/from the node. This may lead to other nodes being disconnected from the output and therefore not be run, but they will not be freed.
    pub fn free_node(&mut self, node: impl Into<NodeAddress>) -> Result<(), FreeError> {
        let node = node.into();
        if node.graph_id == self.id {
            // Does the Node exist?
            if !self.get_nodes_mut().contains_key(node.key) {
                return Err(FreeError::NodeNotFound);
            }

            // Remove all edges leading to the node
            self.node_input_edges.remove(node.key);
            self.graph_input_edges.remove(node.key);
            // feedback from the freed node requires removing the feedback node and all edges from the feedback node
            self.node_feedback_edges.remove(node.key);
            // Remove all edges leading from the node to other nodes
            for (_k, input_edges) in &mut self.node_input_edges {
                let mut i = 0;
                while i < input_edges.len() {
                    if input_edges[i].source == node.key {
                        input_edges.remove(i);
                    } else {
                        i += 1;
                    }
                }
            }
            // Remove all edges leading from the node to the Graph output
            {
                let mut i = 0;
                while i < self.output_edges.len() {
                    if self.output_edges[i].source == node.key {
                        self.output_edges.remove(i);
                    } else {
                        i += 1;
                    }
                }
            }
            // Remove all feedback edges leading from or to the node
            let mut nodes_to_free = HashSet::new();
            if let Some(&feedback_node) = self.node_feedback_node_key.get(node.key) {
                // The node that is being freed has a feedback node attached to it. Free that as well.
                let graph_id = self.id;
                nodes_to_free.insert(NodeAddress {
                    graph_id,
                    key: feedback_node,
                });
                self.node_feedback_node_key.remove(node.key);
            }
            for (feedback_key, feedback_edges) in &mut self.node_feedback_edges {
                if !feedback_edges.is_empty() {
                    let mut i = 0;
                    while i < feedback_edges.len() {
                        if feedback_edges[i].source == node.key
                            || feedback_edges[i].feedback_destination == node.key
                        {
                            feedback_edges.remove(i);
                        } else {
                            i += 1;
                        }
                    }
                    if feedback_edges.is_empty() {
                        // The feedback node has no more edges to it: free it
                        let graph_id = self.id;
                        nodes_to_free.insert(NodeAddress {
                            graph_id,
                            key: feedback_key,
                        });
                        // TODO: Will this definitely remove all feedback node
                        // key references? Can a feedback node be manually freed
                        // in a different way?
                        let mut node_feedback_node_belongs_to = None;
                        for (source_node, &feedback_node) in &self.node_feedback_node_key {
                            if feedback_node == feedback_key {
                                node_feedback_node_belongs_to = Some(source_node);
                            }
                        }
                        if let Some(key) = node_feedback_node_belongs_to {
                            self.node_feedback_node_key.remove(key);
                        }
                    }
                }
            }
            for na in nodes_to_free {
                self.free_node(na)?;
            }
            if let Some(ggc) = &mut self.graph_gen_communicator {
                // The GraphGen has been created so we have to be more careful
                self.node_keys_to_free_when_safe
                    .push((node.key, ggc.next_change_flag.clone()));
                self.node_keys_pending_removal.insert(node.key);
            } else {
                // The GraphGen has not been created so we can do things the easy way
                self.graphs_per_node.remove(node.key);
                self.get_nodes_mut().remove(node.key);
            }
        } else {
            // Try to find the graph containing the node by asking all the graphs in this graph to free the node
            for (_key, graph) in &mut self.graphs_per_node {
                match graph.free_node(node) {
                    Ok(_) => return Ok(()),
                    Err(e) => match e {
                        FreeError::GraphNotFound => (),
                        _ => {
                            return Err(e);
                        }
                    },
                }
            }
            return Err(FreeError::GraphNotFound);
        }
        Ok(())
    }

    pub fn schedule_change(&mut self, change: ParameterChange) -> Result<(), ScheduleError> {
        if change.node.graph_id == self.id {
            // Does the Node exist?
            if !self.get_nodes_mut().contains_key(change.node.key) {
                return Err(ScheduleError::NodeNotFound);
            }
            let index = if let Some(label) = change.input_label {
                if let Some(label_index) = self.node_input_name_to_index[change.node.key].get(label)
                {
                    *label_index
                } else {
                    return Err(ScheduleError::InputLabelNotFound(label));
                }
            } else if let Some(index) = change.input_index {
                index
            } else {
                0
            };
            if let Some(ggc) = &mut self.graph_gen_communicator {
                // The GraphGen has been created so we have to be more careful
                let change_kind = ScheduledChangeKind::Constant {
                    index,
                    value: change.value,
                };
                match change.time {
                    TimeKind::DurationFromNow(d) => {
                        ggc.scheduler
                            .schedule_local_time(change.node.key, change_kind, d)
                    }
                    TimeKind::AbsoluteSample(absolute_timestamp) => ggc
                        .scheduler
                        .schedule_absolute_sample(change.node.key, change_kind, absolute_timestamp),
                }
            } else {
                return Err(ScheduleError::SchedulerNotCreated);
            }
        } else {
            // Try to find the graph containing the node by asking all the graphs in this graph to free the node
            for (_key, graph) in &mut self.graphs_per_node {
                match graph.schedule_change(change) {
                    Ok(_) => return Ok(()),
                    Err(e) => match e {
                        ScheduleError::GraphNotFound => (),
                        _ => return Err(e),
                    },
                }
            }
            return Err(ScheduleError::GraphNotFound);
        }
        Ok(())
    }
    /// Disconnect the given connection if it exists. Will return Ok if the Connection doesn't exist, but the data inside it is correct and the graph could be found.
    ///
    /// Disconnecting a constant means setting that constant input to 0. Disconnecting a feedback edge will remove the feedback node under the hood if there are no remaining edges to it. Disconnecting a Connection::Clear will do the same thing as "connecting" it: clear edges according to its parameters.
    pub fn disconnect(&mut self, connection: Connection) -> Result<(), ConnectionError> {
        let mut try_disconnect_in_child_graphs = |connection| {
            for (_key, graph) in &mut self.graphs_per_node {
                match graph.disconnect(connection) {
                    Ok(_) => return Ok(()),
                    Err(e) => match &e {
                        ConnectionError::NodeNotFound => {
                            // The correct graph was found, but the node wasn't in it.
                            return Err(ConnectionError::NodeNotFound);
                        }
                        // We continue trying other graphs
                        ConnectionError::GraphNotFound => (),
                        _ => return Err(e),
                    },
                }
            }
            Err(ConnectionError::GraphNotFound)
        };

        match connection {
            Connection::Node {
                source,
                from_index,
                from_label,
                sink,
                to_index: input_index,
                to_label: input_label,
                channels,
                feedback,
            } => {
                if source.graph_id != sink.graph_id {
                    return Err(ConnectionError::DifferentGraphs);
                }
                if source.graph_id != self.id {
                    return try_disconnect_in_child_graphs(connection);
                }
                if source.key == sink.key {
                    return Err(ConnectionError::SameNode);
                }
                let to_index = if input_index.is_some() {
                    if let Some(i) = input_index {
                        i
                    } else {
                        0
                    }
                } else if input_label.is_some() {
                    if let Some(label) = input_label {
                        // unwrap() is okay because the hashmap is generated for every node when it is inserted
                        if let Some(index) = self.input_index_from_label(sink.key, label) {
                            index
                        } else {
                            return Err(ConnectionError::InvalidInputLabel(label));
                        }
                    } else {
                        0
                    }
                } else {
                    0
                };
                let from_index = if from_index.is_some() {
                    if let Some(i) = from_index {
                        i
                    } else {
                        0
                    }
                } else if from_label.is_some() {
                    if let Some(label) = from_label {
                        // unwrap() is okay because the hashmap is generated for every node when it is inserted
                        if let Some(index) = self.output_index_from_label(sink.key, label) {
                            index
                        } else {
                            return Err(ConnectionError::InvalidOutputLabel(label));
                        }
                    } else {
                        0
                    }
                } else {
                    0
                };
                // Alternative way to get the num_inputs without accessing the node
                if channels + to_index > self.node_input_index_to_name.get(sink.key).unwrap().len()
                {
                    return Err(ConnectionError::ChannelOutOfBounds);
                }
                if !feedback {
                    let edge_list = &mut self.node_input_edges[sink.key];
                    let mut i = 0;
                    while i < edge_list.len() {
                        if edge_list[i].source == source.key
                            && edge_list[i].from_output_index >= from_index
                            && edge_list[i].from_output_index < from_index + channels
                            && edge_list[i].to_input_index >= to_index
                            && edge_list[i].to_input_index < to_index + channels
                        {
                            edge_list.remove(i);
                        } else {
                            i += 1;
                        }
                    }
                } else if let Some(&feedback_node) = self.node_feedback_node_key.get(source.key) {
                    // Remove feedback edges
                    let feedback_edge_list = &mut self.node_feedback_edges[feedback_node];
                    let mut i = 0;
                    while i < feedback_edge_list.len() {
                        if feedback_edge_list[i].source == source.key
                            && feedback_edge_list[i].feedback_destination == sink.key
                            && feedback_edge_list[i].from_output_index >= from_index
                            && feedback_edge_list[i].from_output_index < from_index + channels
                            && feedback_edge_list[i].to_input_index >= from_index
                            && feedback_edge_list[i].to_input_index < from_index + channels
                        {
                            feedback_edge_list.remove(i);
                        } else {
                            i += 1;
                        }
                    }
                    // Remove inputs to the sink

                    let edge_list = &mut self.node_input_edges[sink.key];

                    let mut i = 0;
                    while i < edge_list.len() {
                        if edge_list[i].source == feedback_node
                            && edge_list[i].from_output_index >= from_index
                            && edge_list[i].from_output_index < from_index + channels
                            && edge_list[i].to_input_index >= to_index
                            && edge_list[i].to_input_index < to_index + channels
                        {
                            edge_list.remove(i);
                        } else {
                            i += 1;
                        }
                    }

                    if feedback_edge_list.is_empty() {
                        self.free_node(NodeAddress {
                            key: feedback_node,
                            graph_id: self.id,
                        })?;
                    }
                }
            }
            Connection::Constant {
                value: _,
                sink,
                to_index: input_index,
                to_label: input_label,
            } => {
                if let Some(sink) = sink {
                    if sink.graph_id != self.id {
                        return try_disconnect_in_child_graphs(connection);
                    }

                    let input = if input_index.is_some() {
                        if let Some(i) = input_index {
                            i
                        } else {
                            0
                        }
                    } else if input_label.is_some() {
                        if let Some(label) = input_label {
                            if let Some(index) = self.input_index_from_label(sink.key, label) {
                                index
                            } else {
                                return Err(ConnectionError::InvalidInputLabel(label));
                            }
                        } else {
                            0
                        }
                    } else {
                        0
                    };
                    if let Some(ggc) = &mut self.graph_gen_communicator {
                        ggc.scheduler.schedule_asap(
                            sink.key,
                            ScheduledChangeKind::Constant {
                                index: input,
                                value: 0.0,
                            },
                        );
                    } else {
                        // No GraphGen exists so we can set the constant directly.
                        self.get_nodes_mut()[sink.key].set_constant(0.0, input);
                    }
                } else {
                    return Err(ConnectionError::SinkNotSet);
                }
            }
            Connection::GraphOutput {
                source,
                from_index,
                from_label,
                to_index,
                channels,
            } => {
                if source.graph_id != self.id {
                    return try_disconnect_in_child_graphs(connection);
                }
                if channels + to_index > self.num_outputs {
                    return Err(ConnectionError::ChannelOutOfBounds);
                }

                let from_index = if from_index.is_some() {
                    if let Some(i) = from_index {
                        i
                    } else {
                        0
                    }
                } else if from_label.is_some() {
                    if let Some(label) = from_label {
                        // unwrap() is okay because the hashmap is generated for every node when it is inserted
                        if let Some(index) = self.output_index_from_label(source.key, label) {
                            index
                        } else {
                            return Err(ConnectionError::InvalidOutputLabel(label));
                        }
                    } else {
                        0
                    }
                } else {
                    0
                };
                let edge_list = &mut self.output_edges;
                let mut i = 0;
                while i < edge_list.len() {
                    if edge_list[i].source == source.key
                        && edge_list[i].from_output_index >= from_index
                        && edge_list[i].from_output_index < from_index + channels
                        && edge_list[i].to_input_index >= to_index
                        && edge_list[i].to_input_index < to_index + channels
                    {
                        edge_list.remove(i);
                    } else {
                        i += 1;
                    }
                }
            }
            Connection::GraphInput {
                sink,
                from_index,
                to_index,
                to_label,
                channels,
            } => {
                if sink.graph_id != self.id {
                    return try_disconnect_in_child_graphs(connection);
                }
                let to_index = if to_index.is_some() {
                    if let Some(i) = to_index {
                        i
                    } else {
                        0
                    }
                } else if to_label.is_some() {
                    if let Some(label) = to_label {
                        if let Some(index) = self.input_index_from_label(sink.key, label) {
                            index
                        } else {
                            return Err(ConnectionError::InvalidInputLabel(label));
                        }
                    } else {
                        0
                    }
                } else {
                    0
                };
                if channels + to_index > self.num_outputs {
                    return Err(ConnectionError::ChannelOutOfBounds);
                }
                let edge_list = &mut self.graph_input_edges[sink.key];
                let mut i = 0;
                while i < edge_list.len() {
                    if edge_list[i].source == sink.key
                        && edge_list[i].from_output_index >= from_index
                        && edge_list[i].from_output_index < from_index + channels
                        && edge_list[i].to_input_index >= to_index
                        && edge_list[i].to_input_index < to_index + channels
                    {
                        edge_list.remove(i);
                    } else {
                        i += 1;
                    }
                }
            }
            Connection::Clear { .. } => {
                return self.connect(connection);
            }
        }
        Ok(())
    }
    /// Create or clear a connection in the Graph. Will call child Graphs until
    /// the graph containing the nodes is found or return an error if the right
    /// Graph or Node cannot be found.
    pub fn connect(&mut self, connection: Connection) -> Result<(), ConnectionError> {
        let mut try_connect_to_graphs = |connection| {
            for (_key, graph) in &mut self.graphs_per_node {
                match graph.connect(connection) {
                    Ok(_) => return Ok(()),
                    Err(e) => match &e {
                        ConnectionError::NodeNotFound => {
                            // The correct graph was found, but the node wasn't in it.
                            return Err(ConnectionError::NodeNotFound);
                        }
                        // We continue trying other graphs
                        ConnectionError::GraphNotFound => (),
                        _ => return Err(e),
                    },
                }
            }
            Err(ConnectionError::GraphNotFound)
        };
        match connection {
            Connection::Node {
                source,
                from_index,
                from_label,
                sink,
                to_index: input_index,
                to_label: input_label,
                channels,
                feedback,
            } => {
                if source.graph_id != sink.graph_id {
                    return Err(ConnectionError::DifferentGraphs);
                }
                if source.graph_id != self.id {
                    return try_connect_to_graphs(connection);
                }
                if source.key == sink.key {
                    return Err(ConnectionError::SameNode);
                }
                let to_index = if input_index.is_some() {
                    if let Some(i) = input_index {
                        i
                    } else {
                        0
                    }
                } else if input_label.is_some() {
                    if let Some(label) = input_label {
                        // unwrap() is okay because the hashmap is generated for every node when it is inserted
                        if let Some(index) = self.input_index_from_label(sink.key, label) {
                            index
                        } else {
                            return Err(ConnectionError::InvalidInputLabel(label));
                        }
                    } else {
                        0
                    }
                } else {
                    0
                };
                let from_index = if from_index.is_some() {
                    if let Some(i) = from_index {
                        i
                    } else {
                        0
                    }
                } else if from_label.is_some() {
                    if let Some(label) = from_label {
                        // unwrap() is okay because the hashmap is generated for every node when it is inserted
                        if let Some(index) = self.output_index_from_label(sink.key, label) {
                            index
                        } else {
                            return Err(ConnectionError::InvalidOutputLabel(label));
                        }
                    } else {
                        0
                    }
                } else {
                    0
                };
                // Alternative way to get the num_inputs without accessing the node
                if channels + to_index > self.node_input_index_to_name.get(sink.key).unwrap().len()
                {
                    return Err(ConnectionError::ChannelOutOfBounds);
                }
                if !feedback {
                    let edge_list = &mut self.node_input_edges[sink.key];
                    for i in 0..channels {
                        edge_list.push(Edge {
                            from_output_index: from_index + i,
                            source: source.key,
                            to_input_index: to_index + i,
                        });
                    }
                } else {
                    // Create a feedback node if there isn't one.
                    let feedback_node_index =
                        if let Some(&index) = self.node_feedback_node_key.get(source.key) {
                            index
                        } else {
                            let num_outputs = self.get_nodes_mut()[source.key].num_outputs();
                            let feedback_node = FeedbackGen::node(num_outputs);
                            let adress = self.push_node(feedback_node);
                            self.feedback_node_indices.push(adress.key);
                            self.node_feedback_node_key.insert(source.key, adress.key);
                            adress.key
                        };
                    // Create feedback edges leading to the FeedbackNode from
                    // the source and normal edges leading from the FeedbackNode
                    // to the sink.
                    let edge_list = &mut self.node_input_edges[sink.key];
                    for i in 0..channels {
                        edge_list.push(Edge {
                            from_output_index: from_index + i,
                            source: feedback_node_index,
                            to_input_index: to_index + i,
                        });
                    }
                    let edge_list = &mut self.node_feedback_edges[feedback_node_index];
                    for i in 0..channels {
                        edge_list.push(FeedbackEdge {
                            from_output_index: from_index + i,
                            source: source.key,
                            to_input_index: from_index + i,
                            feedback_destination: sink.key,
                        });
                    }
                }
            }
            Connection::Constant {
                value,
                sink,
                to_index: input_index,
                to_label: input_label,
            } => {
                if let Some(sink) = sink {
                    if sink.graph_id != self.id {
                        return try_connect_to_graphs(connection);
                    }

                    let input = if input_index.is_some() {
                        if let Some(i) = input_index {
                            i
                        } else {
                            0
                        }
                    } else if input_label.is_some() {
                        if let Some(label) = input_label {
                            if let Some(index) = self.input_index_from_label(sink.key, label) {
                                index
                            } else {
                                return Err(ConnectionError::InvalidInputLabel(label));
                            }
                        } else {
                            0
                        }
                    } else {
                        0
                    };
                    if let Some(ggc) = &mut self.graph_gen_communicator {
                        ggc.scheduler.schedule_asap(
                            sink.key,
                            ScheduledChangeKind::Constant {
                                index: input,
                                value,
                            },
                        );
                    } else {
                        // No GraphGen exists so we can set the constant directly.
                        self.get_nodes_mut()[sink.key].set_constant(value, input);
                    }
                } else {
                    return Err(ConnectionError::SinkNotSet);
                }
            }
            Connection::GraphOutput {
                source,
                from_index,
                from_label,
                to_index,
                channels,
            } => {
                if source.graph_id != self.id {
                    return try_connect_to_graphs(connection);
                }
                if channels + to_index > self.num_outputs {
                    return Err(ConnectionError::ChannelOutOfBounds);
                }
                let from_index = if from_index.is_some() {
                    if let Some(i) = from_index {
                        i
                    } else {
                        0
                    }
                } else if from_label.is_some() {
                    if let Some(label) = from_label {
                        if let Some(index) = self.output_index_from_label(source.key, label) {
                            index
                        } else {
                            return Err(ConnectionError::InvalidOutputLabel(label));
                        }
                    } else {
                        0
                    }
                } else {
                    0
                };
                for i in 0..channels {
                    self.output_edges.push(Edge {
                        source: source.key,
                        from_output_index: from_index + i,
                        to_input_index: to_index + i,
                    })
                }
            }
            Connection::GraphInput {
                sink,
                from_index,
                to_index,
                to_label,
                channels,
            } => {
                if sink.graph_id != self.id {
                    return try_connect_to_graphs(connection);
                }
                let to_index = if to_index.is_some() {
                    if let Some(i) = to_index {
                        i
                    } else {
                        0
                    }
                } else if to_label.is_some() {
                    if let Some(label) = to_label {
                        if let Some(index) = self.input_index_from_label(sink.key, label) {
                            index
                        } else {
                            return Err(ConnectionError::InvalidInputLabel(label));
                        }
                    } else {
                        0
                    }
                } else {
                    0
                };
                if channels + to_index > self.num_outputs {
                    return Err(ConnectionError::ChannelOutOfBounds);
                }
                for i in 0..channels {
                    self.graph_input_edges[sink.key].push(Edge {
                        source: sink.key,
                        from_output_index: from_index + i,
                        to_input_index: to_index + i,
                    })
                }
            }
            Connection::Clear {
                node,
                input_nodes,
                input_constants,
                output_nodes,
                graph_outputs,
                graph_inputs,
            } => {
                if node.graph_id != self.id {
                    return try_connect_to_graphs(connection);
                }
                if input_nodes {
                    let mut nodes_to_free = HashSet::new();
                    for input_edge in &self.node_input_edges[node.key] {
                        if self.feedback_node_indices.contains(&input_edge.source) {
                            // The edge is from a feedback node. Remove the corresponding feedback edge.
                            let feedback_edges = &mut self.node_feedback_edges[input_edge.source];
                            let mut i = 0;
                            while i < feedback_edges.len() {
                                if feedback_edges[i].feedback_destination == node.key
                                    && feedback_edges[i].from_output_index
                                        == input_edge.from_output_index
                                {
                                    feedback_edges.remove(i);
                                } else {
                                    i += 1;
                                }
                            }
                            if feedback_edges.is_empty() {
                                let graph_id = self.id;
                                nodes_to_free.insert(NodeAddress {
                                    graph_id,
                                    key: input_edge.source,
                                });
                            }
                        }
                    }
                    self.node_input_edges[node.key].clear();
                    for na in nodes_to_free {
                        self.free_node(na)?;
                    }
                }
                if graph_inputs {
                    self.graph_input_edges[node.key].clear();
                }
                if input_constants {
                    // Clear input constants by scheduling them all to be set to 0 now
                    let num_node_inputs = self.get_nodes_mut()[node.key].num_inputs();
                    if let Some(ggc) = &mut self.graph_gen_communicator {
                        // The GraphGen has been created so we have to be more careful
                        for index in 0..num_node_inputs {
                            let change_kind = ScheduledChangeKind::Constant { index, value: 0.0 };
                            ggc.scheduler.schedule_asap(node.key, change_kind);
                        }
                    } else {
                        // We are fine to set the constants on the node
                        // directly. In fact we have to because the Scheduler
                        // doesn't exist.
                        for index in 0..num_node_inputs {
                            self.get_nodes_mut()[node.key].set_constant(0.0, index);
                        }
                    }
                }
                if output_nodes {
                    for (_key, edges) in &mut self.node_input_edges {
                        let mut i = 0;
                        while i < edges.len() {
                            if edges[i].source == node.key {
                                edges.remove(i);
                            } else {
                                i += 1;
                            }
                        }
                    }
                }
                if graph_outputs {
                    let mut i = 0;
                    while i < self.output_edges.len() {
                        if self.output_edges[i].source == node.key {
                            self.output_edges.remove(i);
                        } else {
                            i += 1;
                        }
                    }
                }
            }
        }
        Ok(())
    }
    fn input_index_from_label(&self, node: NodeKey, label: &'static str) -> Option<usize> {
        if let Some(&index) = self
            .node_input_name_to_index
            .get(node)
            .unwrap() // Always exists. If it doesn't it's a major bug.
            .get(label)
        {
            Some(index)
        } else {
            None
        }
    }
    fn output_index_from_label(&self, node: NodeKey, label: &'static str) -> Option<usize> {
        if let Some(&index) = self
            .node_output_name_to_index
            .get(node)
            .unwrap() // Always exists. If it doesn't it's a major bug.
            .get(label)
        {
            Some(index)
        } else {
            None
        }
    }
    fn depth_first_search(
        &self,
        visited: &mut HashSet<NodeKey>,
        nodes_to_process: &mut Vec<NodeKey>,
    ) -> Vec<NodeKey> {
        let mut stack = Vec::with_capacity(self.get_nodes().capacity());
        while !nodes_to_process.is_empty() {
            let node_index = nodes_to_process.pop().unwrap();
            stack.push(node_index);
            // cloning the input edges here to avoid unsafe
            let input_edges = &self.node_input_edges[node_index];
            for edge in input_edges {
                if !visited.contains(&edge.source) {
                    nodes_to_process.push(edge.source);
                    visited.insert(edge.source);
                }
            }
        }
        stack
    }
    fn get_deepest_output_node(&self, start_node: NodeKey, visited: &HashSet<NodeKey>) -> NodeKey {
        let mut last_connected_node_index = start_node;
        let mut last_connected_output_node_index = start_node;
        loop {
            let mut found_later_node = false;
            for (key, input_edges) in self.node_input_edges.iter() {
                for input_edge in input_edges {
                    if input_edge.source == last_connected_node_index
                        && !visited.contains(&input_edge.source)
                    {
                        last_connected_node_index = key;
                        found_later_node = true;

                        // check if it's an output node
                        for edge in &self.output_edges {
                            if last_connected_node_index == edge.source {
                                last_connected_output_node_index = last_connected_node_index;
                            }
                        }
                        break;
                    }
                }
                if found_later_node {
                    break;
                }
            }
            if !found_later_node {
                break;
            }
        }
        last_connected_output_node_index
    }
    /// Calculate the node order of the graph based on the outputs
    /// Post-ordered depth first search
    /// NB: Not real-time safe
    pub fn calculate_node_order(&mut self) {
        self.node_order.clear();
        // Add feedback nodes first, their order doesn't matter
        self.node_order.extend(self.feedback_node_indices.iter());
        // Set the visited status for all nodes to false
        let mut visited = HashSet::new();
        // add the feedback node indices as visited
        for &feedback_node_index in &self.feedback_node_indices {
            visited.insert(feedback_node_index);
        }
        let mut nodes_to_process = Vec::with_capacity(self.get_nodes_mut().capacity());
        for edge in &self.output_edges {
            // The same source node may be present in multiple output edges e.g.
            // for stereo so we need to check if visited. One output may also
            // depend on another. Therefore we need to make sure to start with
            // the deepest output nodes only.
            let deepest_node = self.get_deepest_output_node(edge.source, &visited);
            if !visited.contains(&deepest_node) {
                nodes_to_process.push(deepest_node);
                visited.insert(deepest_node);
            }
        }

        let stack = self.depth_first_search(&mut visited, &mut nodes_to_process);
        self.node_order.extend(stack.into_iter().rev());

        // Check if feedback nodes need to be added to the node order
        let mut feedback_node_order_addition = vec![];
        for (_key, feedback_edges) in self.node_feedback_edges.iter() {
            for feedback_edge in feedback_edges {
                if !visited.contains(&feedback_edge.source) {
                    // The source of this feedback_edge needs to be added to the
                    // node order at the end. Check if it's the input to any
                    // other node and start a depth first search from the last
                    // node.
                    let mut last_connected_node_index = feedback_edge.source;
                    let mut last_connected_not_visited_ni = feedback_edge.source;
                    loop {
                        let mut found_later_node = false;
                        for (key, input_edges) in self.node_input_edges.iter() {
                            for input_edge in input_edges {
                                if input_edge.source == last_connected_node_index {
                                    last_connected_node_index = key;

                                    if !visited.contains(&key) {
                                        last_connected_not_visited_ni = key;
                                    }
                                    found_later_node = true;
                                    break;
                                }
                            }
                            if found_later_node {
                                break;
                            }
                        }
                        if !found_later_node {
                            break;
                        }
                    }
                    // Do a depth first search from `last_connected_node_index`
                    nodes_to_process.clear();
                    visited.insert(last_connected_not_visited_ni);
                    nodes_to_process.push(last_connected_not_visited_ni);
                    let stack = self.depth_first_search(&mut visited, &mut nodes_to_process);
                    feedback_node_order_addition.extend(stack);
                }
            }
        }
        self.node_order
            .extend(feedback_node_order_addition.into_iter().rev());

        // Add all remaining nodes. These are not currently connected to anything.
        let mut remaining_nodes = vec![];
        for (node_key, _node) in self.get_nodes() {
            if !visited.contains(&node_key) && !self.node_keys_pending_removal.contains(&node_key) {
                remaining_nodes.push(node_key);
            }
        }
        self.node_order.extend(remaining_nodes.iter());
        self.disconnected_nodes = remaining_nodes;
    }
    pub fn block_size(&self) -> usize {
        self.block_size
    }
    pub fn num_nodes(&self) -> usize {
        self.get_nodes().len()
    }

    /// NB: Not real time safe
    fn generate_tasks(&mut self) -> Vec<Task> {
        let mut tasks = vec![];
        // Safety: No other thread will access the SlotMap. All we're doing with the buffers is taking pointers; there's no manipulation.
        let nodes = unsafe { &mut *self.nodes.get() };
        let first_sample = self.inputs_buffers_ptr.ptr.cast::<f32>();
        for &node_key in &self.node_order {
            let num_inputs = nodes[node_key].num_inputs();
            let mut input_buffers = NodeBufferRef {
                buf: first_sample,
                num_channels: num_inputs,
                block_size: self.block_size,
            };
            // Collect inputs into the node's input buffer
            let input_edges = &self.node_input_edges[node_key];
            let graph_input_edges = &self.graph_input_edges[node_key];
            let feedback_input_edges = &self.node_feedback_edges[node_key];

            let mut inputs_to_copy = vec![];
            let mut graph_inputs_to_copy = vec![];

            for input_edge in input_edges {
                let source = &nodes[input_edge.source];
                let mut output_values = source.output_buffers();
                for sample_index in 0..self.block_size {
                    let from_channel = input_edge.from_output_index;
                    let to_channel = input_edge.to_input_index;
                    inputs_to_copy.push((
                        output_values.ptr_to_sample(from_channel, sample_index),
                        input_buffers.ptr_to_sample(to_channel, sample_index),
                    ));
                }
            }
            for input_edge in graph_input_edges {
                graph_inputs_to_copy
                    .push((input_edge.from_output_index, input_edge.to_input_index));
            }
            for feedback_edge in feedback_input_edges {
                let source = &nodes[feedback_edge.source];
                let mut output_values = source.output_buffers();
                for i in 0..self.block_size {
                    inputs_to_copy.push((
                        output_values.ptr_to_sample(feedback_edge.from_output_index, i),
                        input_buffers.ptr_to_sample(feedback_edge.from_output_index, i),
                    ));
                }
            }
            let node = &nodes[node_key];
            tasks.push(Task {
                node_key,
                inputs_to_copy,
                graph_inputs_to_copy,
                input_buffers,
                input_constants: node.input_constants,
                gen: node.gen,
                output_buffers_first_ptr: node.output_buffers_first_ptr,
                block_size: node.block_size,
                num_outputs: node.num_outputs,
            });
        }
        tasks
    }
    fn generate_output_tasks(&mut self) -> Vec<OutputTask> {
        let mut output_tasks = vec![];
        for output_edge in &self.output_edges {
            let source = &self.get_nodes()[output_edge.source];
            let graph_output_index = output_edge.to_input_index;
            output_tasks.push(OutputTask {
                input_buffers: source.output_buffers(),
                input_index: output_edge.from_output_index,
                graph_output_index,
            });
        }
        output_tasks
    }
    /// Only one GraphGen can be created from a Graph, since otherwise nodes in
    /// the graph could be run multiple times.
    fn create_graph_gen(&mut self) -> Result<GraphGen, String> {
        if self.graph_gen_communicator.is_some() {
            return Err(
                "create_graph_gen: GraphGenCommunicator already existed for this graph".to_owned(),
            );
        }
        self.init();
        let tasks = self.generate_tasks().into_boxed_slice();
        let output_tasks = self.generate_output_tasks().into_boxed_slice();
        let task_data = TaskData {
            applied: Arc::new(AtomicBool::new(false)),
            tasks,
            output_tasks,
        };
        // let task_data = Box::into_raw(Box::new(task_data));
        // let task_data_ptr = Arc::new(AtomicPtr::new(task_data));
        let (free_node_queue_producer, free_node_queue_consumer) =
            RingBuffer::<(NodeKey, GenState)>::new(self.ring_buffer_size);
        let (new_task_data_producer, new_task_data_consumer) =
            RingBuffer::<TaskData>::new(self.ring_buffer_size);
        let (task_data_to_be_dropped_producer, task_data_to_be_dropped_consumer) =
            RingBuffer::<TaskData>::new(self.ring_buffer_size);
        let (scheduler, schedule_receiver) = Scheduler::new(self.sample_rate, 300, self.latency);

        let graph_gen_communicator = GraphGenCommunicator {
            free_node_queue_consumer,
            scheduler,
            task_data_to_be_dropped_consumer,
            new_task_data_producer,
            next_change_flag: task_data.applied.clone(),
            timestamp: Arc::new(AtomicU64::new(0)),
        };

        let graph_gen = GraphGen {
            current_task_data: task_data,
            block_size: self.block_size,
            num_outputs: self.num_outputs,
            num_inputs: self.num_inputs,
            graph_state: GenState::Continue,
            sample_counter: 0,
            timestamp: graph_gen_communicator.timestamp.clone(),
            free_node_queue_producer,
            schedule_receiver,
            _arc_nodes: self.nodes.clone(),
            task_data_to_be_dropped_producer,
            new_task_data_consumer,
            _arc_inputs_buffers_ptr: self.inputs_buffers_ptr.clone(),
        };
        self.graph_gen_communicator = Some(graph_gen_communicator);
        Ok(graph_gen)
    }

    /// Initialise buffers, return parameters to zero etc.
    fn init(&mut self) {
        self.calculate_node_order();
        let block_size = self.block_size;
        let sample_rate = self.sample_rate;
        for (_key, n) in self.get_nodes_mut() {
            n.init(block_size, sample_rate);
        }
        // self.tasks = self.generate_tasks();
        // self.output_tasks = self.generate_output_tasks();
        self.initiated = true;
    }

    /// Applies the latest changes to connections and added nodes in the graph on the audio thread and updates the scheduler.
    pub fn commit_changes(&mut self) {
        if self.graph_gen_communicator.is_some() {
            self.free_old();
            self.calculate_node_order();
            let output_tasks = self.generate_output_tasks().into_boxed_slice();
            let tasks = self.generate_tasks().into_boxed_slice();
            if let Some(ggc) = &mut self.graph_gen_communicator {
                ggc.send_updated_tasks(tasks, output_tasks);
            }
        }
        for (_key, graph) in &mut self.graphs_per_node {
            graph.commit_changes();
        }
    }

    /// This function needs to be run regularly to be sure that scheduled changes are carried out.
    pub fn update(&mut self) {
        if let Some(ggc) = &mut self.graph_gen_communicator {
            ggc.update();
        }
        for (_key, graph) in &mut self.graphs_per_node {
            graph.update();
        }
    }

    /// Check if there are any old nodes or other resources that have been
    /// removed from the graph and can now be freed since they are no longer
    /// used on the audio thread.
    fn free_old(&mut self) {
        // See if the GraphGen has reported any nodes that should be freed
        let free_queue = if let Some(ggc) = &mut self.graph_gen_communicator {
            ggc.get_nodes_to_free()
        } else {
            vec![]
        };
        for (key, state) in free_queue {
            let address = NodeAddress {
                graph_id: self.id,
                key,
            };
            match state {
                GenState::FreeSelf => {
                    // If the node key cannot be found it was probably freed
                    // already and added multiple times to the queue. We can
                    // just ignore it.
                    self.free_node(address).ok();
                }
                GenState::FreeSelfMendConnections => {
                    self.free_node_mend_connections(address).ok();
                }
                GenState::FreeGraph(_) => unreachable!(),
                GenState::FreeGraphMendConnections(_) => unreachable!(),
                GenState::Continue => unreachable!(),
            };
        }
        // Remove old nodes
        let nodes = unsafe { &mut *self.nodes.get() };
        let mut i = 0;
        while i < self.node_keys_to_free_when_safe.len() {
            let (key, flag) = &self.node_keys_to_free_when_safe[i];
            if flag.load(Ordering::SeqCst) {
                nodes.remove(*key);
                // If the node was a graph, free the graph as well (it will be returned and  dropped here)
                // The Graph should be dropped after the GraphGen Node.
                self.graphs_per_node.remove(*key);
                self.node_keys_pending_removal.remove(key);
                self.node_keys_to_free_when_safe.remove(i);
            } else {
                i += 1;
            }
        }
    }
    fn get_nodes_mut(&mut self) -> &mut SlotMap<NodeKey, Node> {
        unsafe { &mut *self.nodes.get() }
    }
    fn get_nodes(&self) -> &SlotMap<NodeKey, Node> {
        unsafe { &*self.nodes.get() }
    }
}

/// Safety: The GraphGen is given access to an Arc<UnsafeCell<SlotMap<NodeKey,
/// Node>>, but won't use it unless the Graph is dropped and it needs to keep
/// the SlotMap alive, and the drop it when the GraphGen is dropped.
unsafe impl Send for Graph {}

impl Gen for GraphGen {
    fn name(&self) -> &'static str {
        "GraphGen"
    }
    fn process(&mut self, ctx: GenContext, resources: &mut Resources) -> GenState {
        match self.graph_state {
            GenState::Continue => {
                // TODO: Support output with a different block size, i.e. local buffering and running this graph more or less often than the parent graph
                //
                //
                let mut do_empty_buffer = None;
                let mut do_mend_connections = None;
                let num_new_task_data = self.new_task_data_consumer.slots();
                if num_new_task_data > 0 {
                    if let Ok(td_chunk) = self.new_task_data_consumer.read_chunk(num_new_task_data)
                    {
                        for td in td_chunk {
                            // Setting `applied` to true signals that the new TaskData have been received and old data can be dropped
                            td.applied.store(true, Ordering::SeqCst);
                            let old_td = std::mem::replace(&mut self.current_task_data, td);
                            match self.task_data_to_be_dropped_producer.push(old_td) {
                            Ok(_) => (),
                            Err(e) => eprintln!("RingBuffer for TaskData to be dropped was full. Please increase the size of the RingBuffer. The GraphGen will drop the TaskData here instead. e: {e}"),
                        }
                        }
                    }
                }

                let task_data = &mut self.current_task_data;
                let TaskData {
                    applied: _,
                    tasks,
                    output_tasks,
                } = task_data;

                let changes = self.schedule_receiver.changes();

                // Run the tasks
                for task in tasks.iter_mut() {
                    task.init_constants();
                    // If there are any changes to the constants of the node, apply them here
                    let mut i = 0;
                    while i < changes.len() {
                        let change = &changes[i];
                        if change.key == task.node_key {
                            let sample_to_apply = if change.timestamp < self.sample_counter {
                                if change.timestamp != 0 {
                                    // timestamps of 0 simply means as fast as possible. It is not an error or issue.
                                    eprintln!("Warning: Scheduled change was applied late. Consider increasing latency.");
                                }
                                0
                            } else {
                                change.timestamp - self.sample_counter
                            } as usize;
                            if sample_to_apply < self.block_size {
                                task.apply_constant_change(change, sample_to_apply);
                                // TODO: This is inefficient since the the first
                                // changes are the most likely to be removed,
                                // and are the most expensive to remove. Either
                                // the changes can be in reverse order, but then
                                // pushing into the list always puts new changes
                                // in the wrong place, or many changes can be
                                // removed all at once after applying them.
                                changes.remove(i);
                            } else {
                                i += 1;
                            }
                        } else {
                            i += 1;
                        }
                    }
                    match task.run(ctx.inputs, resources) {
                        GenState::Continue => (),
                        GenState::FreeSelf => {
                            // We don't care if it fails since if it does the
                            // node will still exist, return FreeSelf and get
                            // added to the queue next block.
                            self.free_node_queue_producer
                                .push((task.node_key, GenState::FreeSelf))
                                .ok();
                        }
                        GenState::FreeSelfMendConnections => {
                            self.free_node_queue_producer
                                .push((task.node_key, GenState::FreeSelfMendConnections))
                                .ok();
                        }
                        GenState::FreeGraph(from_sample_nr) => {
                            self.graph_state = GenState::FreeSelf;
                            do_empty_buffer = Some(from_sample_nr);
                        }
                        GenState::FreeGraphMendConnections(from_sample_nr) => {
                            self.graph_state = GenState::FreeSelfMendConnections;
                            do_mend_connections = Some(from_sample_nr);
                        }
                    }
                }

                // Set the output of the graph
                // Zero the output buffer.
                ctx.outputs.fill(0.0);
                for output_task in output_tasks.iter() {
                    let input_values = output_task
                        .input_buffers
                        .get_channel(output_task.input_index);
                    let output = ctx.outputs.get_channel_mut(output_task.graph_output_index);
                    for i in 0..self.block_size {
                        let value = input_values[i];
                        // Since many nodes may write to the same output we
                        // need to add the outputs together. The Node makes no promise as to the content of
                        // the output buffer provided.
                        output[i] += value;
                    }
                }
                if let Some(from_relative_sample_nr) = do_empty_buffer {
                    for channel in 0..ctx.outputs.channels() {
                        let output = ctx.outputs.get_channel_mut(channel);
                        for sample in &mut output[from_relative_sample_nr..] {
                            *sample = 0.0;
                        }
                    }
                }
                if let Some(from_relative_sample_nr) = do_mend_connections {
                    let num_channels = ctx.inputs.channels().min(ctx.outputs.channels());
                    for channel in 0..num_channels {
                        let input = ctx.inputs.get_channel(channel);
                        let output = ctx.outputs.get_channel_mut(channel);
                        // TODO: Check if the input is constant or not first. Only non-constant inputs should be passed through (because it's "mending" the connection)
                        for (i, o) in input[from_relative_sample_nr..]
                            .iter()
                            .zip(output[from_relative_sample_nr..].iter_mut())
                        {
                            *o = *i;
                        }
                    }
                }
                self.sample_counter += self.block_size as u64;
                self.timestamp.store(self.sample_counter, Ordering::SeqCst);
            }
            GenState::FreeSelf => {
                ctx.outputs.fill(0.0);
            }
            GenState::FreeSelfMendConnections => {
                let num_channels = ctx.inputs.channels().min(ctx.outputs.channels());
                for channel in 0..num_channels {
                    let input = ctx.inputs.get_channel(channel);
                    let output = ctx.outputs.get_channel_mut(channel);
                    // TODO: Check if the input is constant or not first. Only non-constant inputs should be passed through (because it's "mending" the connection)
                    for (i, o) in input.iter().zip(output.iter_mut()) {
                        *o = *i;
                    }
                }
            }
            // These are unreachable because they are converted to the FreeSelf
            // variants at the Graph level which is this level.
            GenState::FreeGraph(_) => unreachable!(),
            GenState::FreeGraphMendConnections(_) => unreachable!(),
        }
        self.graph_state
    }
    fn num_inputs(&self) -> usize {
        self.num_inputs
    }
    fn num_outputs(&self) -> usize {
        self.num_outputs
    }
}

/// This gets placed as a dyn Gen in a Node in a Graph. It's how the Graph gets
/// run. The Graph communicates with the GraphGen in a thread safe way.
///
/// # Safety
/// Using this struct is safe only if used in conjunction with the
/// Graph. The Graph owns nodes and gives its corresponding GraphGen raw
/// pointers to them through Tasks, but it never accesses or deallocates a node
/// while it can be accessed by the [`GraphGen`] through a Task. The [`GraphGen`]
/// mustn't use the _arc_nodes field; it is only there to make sure the nodes
/// don't get dropped.
struct GraphGen {
    block_size: usize,
    num_outputs: usize,
    num_inputs: usize,
    current_task_data: TaskData,
    // This Arc is cloned from the Graph and exists so that if the Graph gets dropped, the GraphGen can continue on without segfaulting.
    _arc_nodes: Arc<UnsafeCell<SlotMap<NodeKey, Node>>>,
    // This Arc makes sure the input buffers allocation is valid for as long as it needs to be
    _arc_inputs_buffers_ptr: Arc<OwnedRawBuffer>,
    graph_state: GenState,
    /// Stores the number of completed samples, updated at the end of a block
    sample_counter: u64,
    timestamp: Arc<AtomicU64>,
    schedule_receiver: ScheduleReceiver,
    free_node_queue_producer: rtrb::Producer<(NodeKey, GenState)>,
    task_data_to_be_dropped_producer: rtrb::Producer<TaskData>,
    new_task_data_consumer: rtrb::Consumer<TaskData>,
}

/// Safety: This impl of Send is required because of the Arc<UnsafeCell<...>> in
/// GraphGen. The _arc_nodes field of GraphGen exists only so that the nodes
/// won't get dropped if the Graph is dropped. The UnsafeCell will never be used
/// to access the data from within GraphGen.
unsafe impl Send for GraphGen {}

struct ScheduledChange {
    timestamp: u64,
    key: NodeKey,
    kind: ScheduledChangeKind,
}
enum ScheduledChangeKind {
    Constant { index: usize, value: Sample },
}

struct Scheduler {
    start_ts: Instant,
    sample_rate: u64,
    /// if the ts of the change is less than this number of samples in the future, send it to the GraphGen
    max_duration_to_send: u64,
    rb_producer: rtrb::Producer<ScheduledChange>,
    /// Changes waiting to be sent to the GraphGen because they are too far into the future
    scheduling_queue: Vec<ScheduledChange>,
    latency: u64,
}
impl Scheduler {
    fn new(sample_rate: Sample, capacity: usize, latency: Duration) -> (Self, ScheduleReceiver) {
        let (rb_producer, rb_consumer) = RingBuffer::new(capacity);
        (
            Scheduler {
                start_ts: Instant::now(),
                sample_rate: sample_rate as u64,
                max_duration_to_send: (sample_rate * 0.5) as u64,
                scheduling_queue: vec![],
                rb_producer,
                latency: (latency.as_secs_f32() * sample_rate) as u64,
            },
            ScheduleReceiver::new(rb_consumer, capacity),
        )
    }
    fn schedule_absolute_sample(
        &mut self,
        key: NodeKey,
        change: ScheduledChangeKind,
        absolute_timestamp: u64,
    ) {
        self.scheduling_queue.push(ScheduledChange {
            timestamp: absolute_timestamp,
            key,
            kind: change,
        });
    }
    fn schedule_asap(&mut self, key: NodeKey, change: ScheduledChangeKind) {
        self.scheduling_queue.push(ScheduledChange {
            timestamp: 0,
            key,
            kind: change,
        });
    }
    fn schedule_local_time(
        &mut self,
        key: NodeKey,
        change: ScheduledChangeKind,
        duration_from_now: Duration,
    ) {
        let timestamp = ((self.start_ts.elapsed() + duration_from_now).as_secs_f64()
            * self.sample_rate as f64) as u64
            + self.latency;
        self.scheduling_queue.push(ScheduledChange {
            timestamp,
            key,
            kind: change,
        });
    }
    fn update(&mut self, timestamp: u64) {
        // scheduled updates should always be sorted before they are sent, in case there are several changes to the same thing
        self.scheduling_queue.sort_unstable_by_key(|s| s.timestamp);

        let mut i = 0;
        while i < self.scheduling_queue.len() {
            if timestamp > self.scheduling_queue[i].timestamp
                || self.scheduling_queue[i].timestamp - timestamp < self.max_duration_to_send
            {
                let change = self.scheduling_queue.remove(i);
                if let Err(e) = self.rb_producer.push(change) {
                    eprintln!("Unable to push scheduled change into RingBuffer: {e}")
                }
            } else {
                i += 1;
            }
        }
    }
}

struct ScheduleReceiver {
    rb_consumer: rtrb::Consumer<ScheduledChange>,
    schedule_queue: Vec<ScheduledChange>,
}
impl ScheduleReceiver {
    fn new(rb_consumer: rtrb::Consumer<ScheduledChange>, capacity: usize) -> Self {
        Self {
            rb_consumer,
            schedule_queue: Vec::with_capacity(capacity),
        }
    }
    /// TODO: Return only a slice of changes that should be applied this block and then remove them all at once.
    fn changes(&mut self) -> &mut Vec<ScheduledChange> {
        let num_new_changes = self.rb_consumer.slots();
        if num_new_changes > 0 {
            // Only try to read so many changes there is room for in the queue
            let changes_to_read =
                num_new_changes.min(self.schedule_queue.capacity() - self.schedule_queue.len());
            match self.rb_consumer.read_chunk(changes_to_read) {
                Ok(chunk) => {
                    for change in chunk {
                        self.schedule_queue.push(change);
                    }

                    self.schedule_queue.sort_unstable_by_key(|s| s.timestamp);
                }
                Err(e) => {
                    eprintln!("Failed to receive changes in ScheduleReceiver: {e}");
                }
            }
        }
        &mut self.schedule_queue
    }
}

/// This data is sent via a boxed TaskData converted to a raw pointer.
///
/// Safety: The tasks or output_tasks may not be moved while there is a raw
/// pointer to the TaskData. If there is a problem, the Boxes in TaskData may
/// need to be raw pointers.
struct TaskData {
    // `applied` must be set to true when the running GraphGen receives it. This
    // signals that the changes in this TaskData have been applied and certain
    // Nodes may be dropped.
    applied: Arc<AtomicBool>,
    tasks: Box<[Task]>,
    output_tasks: Box<[OutputTask]>,
}

struct GraphGenCommunicator {
    // The number of updates applied to this GraphGen. Add by
    // `updates_available` every time it finishes a block. It is a u16 so that
    // when it overflows generation - some_generation_number fits in an i32.
    scheduler: Scheduler,
    timestamp: Arc<AtomicU64>,
    next_change_flag: Arc<AtomicBool>,
    free_node_queue_consumer: rtrb::Consumer<(NodeKey, GenState)>,
    task_data_to_be_dropped_consumer: rtrb::Consumer<TaskData>,
    new_task_data_producer: rtrb::Producer<TaskData>,
}

unsafe impl Send for GraphGenCommunicator {}

impl GraphGenCommunicator {
    fn free_old(&mut self) {
        // If there are discarded tasks, check if they can be removed
        //
        let num_to_remove = self.task_data_to_be_dropped_consumer.slots();
        let chunk = self
            .task_data_to_be_dropped_consumer
            .read_chunk(num_to_remove);
        if let Ok(chunk) = chunk {
            for td in chunk {
                drop(td);
            }
        }
    }

    /// Sends the updated tasks to the GraphGen. NB: Always check if any
    /// resoruces in the Graph can be freed before running this.
    /// GraphGenCommunicator will free its own resources.
    fn send_updated_tasks(&mut self, tasks: Box<[Task]>, output_tasks: Box<[OutputTask]>) {
        self.free_old();

        let current_change_flag =
            mem::replace(&mut self.next_change_flag, Arc::new(AtomicBool::new(false)));

        let td = TaskData {
            applied: current_change_flag,
            tasks,
            output_tasks,
        };
        if let Err(e) = self.new_task_data_producer.push(td) {
            eprintln!(
                "Unable to push new TaskData to the GraphGen. Please increase RingBuffer size. {e}"
            )
        }
    }

    /// Run periodically to make sure the scheduler passes messages on to the GraphGen
    fn update(&mut self) {
        let timestamp = self.timestamp.load(Ordering::SeqCst);
        self.scheduler.update(timestamp);
    }
    fn get_nodes_to_free(&mut self) -> Vec<(NodeKey, GenState)> {
        let num_items = self.free_node_queue_consumer.slots();
        let chunk = self.free_node_queue_consumer.read_chunk(num_items);
        if let Ok(chunk) = chunk {
            chunk.into_iter().collect()
        } else {
            vec![]
        }
    }
}

/// Node is a very unsafe struct. Be very careful when changing it.
///
/// - The Gen is not allowed to be replaced.
/// - The input_constants buffer mustn't be deallocated until the Node is dropped.
/// - init must not be called after the Node has started being used on the audio thread.
/// - the number of inputs/outputs of the Gen must not change.
///
/// It also must not be dropped while a Task exists with pointers to the buffers
/// of the Node. Graph has a mechanism to ensure this.
struct Node {
    // name: &'static str,
    /// index by input_channel
    input_constants: *mut [Sample],
    /// index by `output_channel * block_size + sample_index`
    output_buffers: *mut [Sample],
    output_buffers_first_ptr: *mut Sample,
    block_size: usize,
    num_outputs: usize,
    gen: *mut (dyn Gen + Send),
}

unsafe impl Send for Node {}

impl Node {
    pub fn new(_name: &'static str, gen: Box<dyn Gen + Send>) -> Self {
        let num_outputs = gen.num_outputs();
        let block_size = 0;
        let output_buffers =
            Box::<[Sample]>::into_raw(vec![0.0; num_outputs * block_size].into_boxed_slice());

        let output_buffers_first_ptr = std::ptr::null_mut();

        Node {
            // name,
            input_constants: Box::into_raw(
                vec![0.0 as Sample; gen.num_inputs()].into_boxed_slice(),
            ),
            gen: Box::into_raw(gen),
            output_buffers,
            output_buffers_first_ptr,
            num_outputs,
            block_size,
        }
    }
    // pub fn name(&self) -> &'static str {
    //     self.name
    // }
    /// *Allocates memory*
    /// Allocates enough memory for the given block size
    pub fn init(&mut self, block_size: usize, sample_rate: Sample) {
        // Free the previous buffer
        unsafe {
            drop(Box::from_raw(self.output_buffers));
        }
        self.output_buffers =
            Box::<[Sample]>::into_raw(vec![0.0; self.num_outputs * block_size].into_boxed_slice());
        self.output_buffers_first_ptr = if block_size * self.num_outputs > 0 {
            // Get the pointer to the first f32 in the block without limiting its scope or going through a reference
            self.output_buffers.cast::<f32>()
        } else {
            std::ptr::null_mut()
        };
        self.block_size = block_size;
        unsafe {
            (*self.gen).init(sample_rate);
        }
    }
    /// Use the embedded Gen to generate values that are placed in the
    /// output_buffer. The Graph will have already filled the input buffer with
    /// the correct values.
    #[inline]
    pub fn process(
        &mut self,
        input_buffers: &NodeBufferRef,
        resources: &mut Resources,
    ) -> GenState {
        let mut outputs = NodeBufferRef {
            buf: self.output_buffers_first_ptr,
            num_channels: self.num_outputs,
            block_size: self.block_size,
        };
        let ctx = GenContext {
            inputs: input_buffers,
            outputs: &mut outputs,
        };
        unsafe { (*self.gen).process(ctx, resources) }
    }
    pub fn set_constant(&mut self, value: Sample, input_index: usize) {
        unsafe {
            (*self.input_constants)[input_index] = value;
        }
    }
    pub fn output_buffers(&self) -> NodeBufferRef {
        NodeBufferRef {
            buf: self.output_buffers_first_ptr,
            num_channels: self.num_outputs,
            block_size: self.block_size,
        }
    }
    pub fn num_inputs(&self) -> usize {
        // self.gen.num_inputs()
        // Not dynamic dispatch, may be faster
        unsafe { &*self.input_constants }.len()
    }
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }
    pub fn input_indices_to_names(&self) -> Vec<&'static str> {
        let mut list = vec![];
        for i in 0..self.num_inputs() {
            list.push(unsafe { (*self.gen).input_desc(i) });
        }
        list
    }
    pub fn output_indices_to_names(&self) -> Vec<&'static str> {
        let mut list = vec![];
        for i in 0..self.num_outputs() {
            list.push(unsafe { (*self.gen).output_desc(i) });
        }
        list
    }
}
impl Drop for Node {
    fn drop(&mut self) {
        drop(unsafe { Box::from_raw(self.gen) });
        drop(unsafe { Box::from_raw(self.input_constants) });
        drop(unsafe { Box::from_raw(self.output_buffers) });
    }
}

#[derive(thiserror::Error, Debug)]
pub enum RunGraphError {
    #[error("Unable to create a node from the Graph: {0}")]
    CouldNodeCreateNode(String),
}

/// Wrapper around a [`Graph`] `Node` with convenience methods to run the
/// Graph, either from an audio thread or for non-real time purposes.
pub struct RunGraph {
    graph_node: Node,
    resources: Resources,
    input_buffer_ptr: *mut f32,
    input_buffer_length: usize,
    input_node_buffer_ref: NodeBufferRef,
    output_node_buffer_ref: NodeBufferRef,
}

impl RunGraph {
    /// Prepare the necessary resources for running the graph. This will fail if
    /// the Graph passed in has already been turned into a Node somewhere else,
    /// for example if it has been pushed to a different Graph.
    pub fn new(graph: &mut Graph, resources: Resources) -> Result<Self, RunGraphError> {
        match graph.split_and_create_node() {
            Ok(graph_node) => {
                let input_buffer_length = graph_node.num_inputs() * graph_node.block_size;
                let input_buffer_ptr = if input_buffer_length != 0 {
                    let input_buffer = vec![0.0 as Sample; input_buffer_length].into_boxed_slice();
                    let input_buffer = Box::into_raw(input_buffer);
                    // Safety: we just created the slice of non-zero length
                    input_buffer.cast::<f32>()
                } else {
                    std::ptr::null_mut()
                };
                let input_node_buffer_ref = NodeBufferRef {
                    buf: input_buffer_ptr,
                    num_channels: graph_node.num_inputs(),
                    block_size: graph_node.block_size,
                };
                // Safety: The Nodes get initiated and their buffers allocated
                // when the Graph is split and the Node is created from it.
                // Therefore, we can safely store a reference to a buffer in
                // that Node here. We want to store it to be able to return a
                // reference to it instead of an owned value which includes a
                // raw pointer.
                let output_node_buffer_ref = graph_node.output_buffers();
                Ok(Self {
                    graph_node,
                    resources,
                    input_buffer_length,
                    input_buffer_ptr,
                    input_node_buffer_ref,
                    output_node_buffer_ref,
                })
            }
            Err(e) => Err(RunGraphError::CouldNodeCreateNode(e)),
        }
    }
    /// Returns the input buffer that will be read as the input of the main
    /// graph. For example, you may want to fill this with the sound inputs of
    /// the sound card. The buffer does not get emptied automatically. If you
    /// don't change it between calls to [`RunGraph::process_block`], its content will be static.
    #[inline]
    pub fn graph_input_buffers(&mut self) -> &mut NodeBufferRef {
        &mut self.input_node_buffer_ref
    }
    /// Run the Graph for one block using the inputs currently stored in the
    /// input buffer. The results can be accessed through the output buffer
    /// through [`RunGraph::graph_output_buffers`].
    pub fn process_block(&mut self) {
        self.graph_node
            .process(&self.input_node_buffer_ref, &mut self.resources);
    }
    /// Return a reference to the buffer holding the output of the [`Graph`].
    /// Channels which have no [`Connection`]/graph edge to them will be 0.0.
    pub fn graph_output_buffers(&self) -> &NodeBufferRef {
        &self.output_node_buffer_ref
    }
    pub fn block_size(&self) -> usize {
        self.output_node_buffer_ref.block_size()
    }
}

impl Drop for RunGraph {
    fn drop(&mut self) {
        // Safety: The slice is allocated when the RunGraph is created with the
        // given size, unless that size is 0 in which case no allocation is
        // made.
        if !self.input_buffer_ptr.is_null() {
            unsafe {
                drop(Box::from_raw(std::slice::from_raw_parts_mut(
                    self.input_buffer_ptr,
                    self.input_buffer_length,
                )));
            }
        }
    }
}
unsafe impl Send for RunGraph {}

/// Buffers the output of a node from last block to simplify feedback nodes and
/// make sure they work in all possible graphs.
///
/// Without buffering, the ordering of nodes becomes very complicated for
/// certain graphs and impossible for others while keeping the feedback
/// consistent to 1 block as far as I can tell.
struct FeedbackGen {
    num_channels: usize,
}

impl FeedbackGen {
    pub fn node(num_channels: usize) -> Node {
        Node::new("feedback_node", Box::new(Self { num_channels }))
    }
}

impl Gen for FeedbackGen {
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        for channel in 0..self.num_channels {
            for sample_index in 0..ctx.block_size() {
                ctx.outputs.write(
                    ctx.inputs.read(channel, sample_index),
                    channel,
                    sample_index,
                );
            }
        }
        GenState::Continue
    }
    fn num_outputs(&self) -> usize {
        self.num_channels
    }
    fn num_inputs(&self) -> usize {
        self.num_channels
    }
}
#[derive(Clone, Debug, Copy)]
struct Edge {
    source: NodeKey,
    /// the output index on the destination node
    from_output_index: usize,
    /// the input index on the origin node where the input from the node is placed
    to_input_index: usize,
}
impl Edge {}

/// Edge containing all metadata for a feedback connection since a feedback
/// connection includes several things that may need to be freed together:
/// - a node
/// - a feedback edge
/// - a normal edge
#[derive(Clone, Debug, Copy)]
struct FeedbackEdge {
    source: NodeKey,
    /// the output index on the destination node
    from_output_index: usize,
    /// the input index on the origin node where the input from the node is placed
    to_input_index: usize,
    /// If the source node is freed we want to remove the normal edge to the destination node.
    feedback_destination: NodeKey,
}

/// When input 0 changes, move smoothly to the new value over the time in seconds given by input 1.
#[derive(Default)]
pub struct Ramp {
    // Compare with the current value. If there is change, recalculate the step.
    last_value: Sample,
    // Compare with the current value. If there is change, recalculate the step.
    last_time: Sample,
    current_value: Sample,
    step: Sample,
    sample_rate: Sample,
}

impl Ramp {
    pub fn new() -> Self {
        Self::default()
    }
}
impl Gen for Ramp {
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        for i in 0..ctx.block_size() {
            let value = ctx.inputs.read(0, i);
            let time = ctx.inputs.read(1, i);
            let mut recalculate = false;
            if value != self.last_value {
                self.last_value = value;
                recalculate = true;
            }
            if time != self.last_time {
                self.last_time = time;
                recalculate = true;
            }
            if recalculate {
                let num_samples = (time * self.sample_rate).floor();
                self.step = (value - self.current_value) / num_samples;
            }
            if (self.current_value - value).abs() < 0.0001 {
                self.current_value = value;
                self.step = 0.;
            }
            self.current_value += self.step;
            ctx.outputs.write(self.current_value, 0, i);
        }
        GenState::Continue
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn init(&mut self, sample_rate: Sample) {
        self.sample_rate = sample_rate;
    }

    fn input_desc(&self, input: usize) -> &'static str {
        match input {
            0 => "value",
            1 => "time",
            _ => "",
        }
    }

    fn output_desc(&self, output: usize) -> &'static str {
        match output {
            0 => "ramped_value",
            _ => "",
        }
    }

    fn name(&self) -> &'static str {
        "Ramp"
    }
}
/// Multiply two inputs together and produce one output.
///
/// # Example
/// ```
/// use knyst::prelude::*;
/// use knyst::wavetable::*;
/// use knyst::graph::RunGraph;
/// let graph_settings = GraphSettings {
///     block_size: 64,
///     sample_rate: 44100.,
///     num_outputs: 2,
///     ..Default::default()
/// };
/// let mut graph = Graph::new(graph_settings);
/// let resources = Resources::new(ResourcesSettings {
///     sample_rate: 44100.,
///     ..Default::default()
/// });
/// let mut run_graph = RunGraph::new(&mut graph, resources)?;
/// let mult = graph.push_gen(Mult);
/// // Connecting the node to the graph output
/// graph.connect(mult.to_graph_out())?;
/// // Multiply 5 by 9
/// graph.connect(constant(5.).to(mult).to_index(0))?;
/// graph.connect(constant(9.).to(mult).to_index(1))?;
/// // You need to commit changes and update if the graph is running.
/// graph.commit_changes();
/// graph.update();
/// run_graph.process_block();
/// assert_eq!(run_graph.graph_output_buffers().read(0, 0), 45.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Mult;
impl Gen for Mult {
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        for i in 0..ctx.block_size() {
            let value = ctx.inputs.read(0, i) * ctx.inputs.read(1, i);
            ctx.outputs.write(value, 0, i);
        }
        GenState::Continue
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn input_desc(&self, input: usize) -> &'static str {
        match input {
            0 => "value0",
            1 => "value1",
            _ => "",
        }
    }

    fn output_desc(&self, output: usize) -> &'static str {
        match output {
            0 => "product",
            _ => "",
        }
    }

    fn name(&self) -> &'static str {
        "Mult"
    }
}
/// Pan a mono signal to stereo using the cos/sine pan law. Pan value should be
/// between -1 and 1, 0 being in the center.
///
/// ```rust
/// use knyst::prelude::*;
/// use knyst::graph::RunGraph;
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let sample_rate = 44100.;
///     let block_size = 8;
///     let resources = Resources::new(ResourcesSettings {
///         sample_rate,
///         ..Default::default()
///     });
///     let graph_settings = GraphSettings {
///         block_size,
///         sample_rate,
///         num_outputs: 2,
///         ..Default::default()
///     };
///     let mut graph: Graph = Graph::new(graph_settings);
///     let pan = graph.push_gen(PanMonoToStereo);
///     // The signal is a constant 1.0
///     graph.connect(constant(1.).to(pan).to_label("signal"))?;
///     // Pan to the left
///     graph.connect(constant(-1.).to(pan).to_label("pan"))?;
///     graph.connect(pan.to_graph_out().channels(2))?;
///     graph.commit_changes();
///     graph.update();
///     let mut run_graph = RunGraph::new(&mut graph, resources)?;
///     run_graph.process_block();
///     assert!(run_graph.graph_output_buffers().read(0, 0) > 0.9999);
///     assert!(run_graph.graph_output_buffers().read(1, 0) < 0.0001);
///     // Pan to the right
///     graph.connect(constant(1.).to(pan).to_label("pan"))?;
///     graph.commit_changes();
///     graph.update();
///     run_graph.process_block();
///     assert!(run_graph.graph_output_buffers().read(0, 0) < 0.0001);
///     assert!(run_graph.graph_output_buffers().read(1, 0) > 0.9999);
///     // Pan to center
///     graph.connect(constant(0.).to(pan).to_label("pan"))?;
///     graph.commit_changes();
///     graph.update();
///     run_graph.process_block();
///     assert_eq!(run_graph.graph_output_buffers().read(0, 0), 0.7070929);
///     assert_eq!(run_graph.graph_output_buffers().read(1, 0), 0.7070929);
///     assert_eq!(
///         run_graph.graph_output_buffers().read(0, 0),
///         run_graph.graph_output_buffers().read(1, 0)
///     );
///     Ok(())
/// }
/// ```
// TODO: Implement multiple different pan laws, maybe as a generic.
pub struct PanMonoToStereo;
impl Gen for PanMonoToStereo {
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        for i in 0..ctx.block_size() {
            let signal = ctx.inputs.read(0, i);
            // The equation needs pan to be in the range [0, 1]
            let pan = ctx.inputs.read(1, i) * 0.5 + 0.5;
            let pan_pos_radians = pan * std::f32::consts::FRAC_PI_2;
            let left_gain = fastapprox::fast::cos(pan_pos_radians);
            let right_gain = fastapprox::fast::sin(pan_pos_radians);
            ctx.outputs.write(signal * left_gain, 0, i);
            ctx.outputs.write(signal * right_gain, 1, i);
        }
        GenState::Continue
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn num_outputs(&self) -> usize {
        2
    }

    fn input_desc(&self, input: usize) -> &'static str {
        match input {
            0 => "signal",
            1 => "pan",
            _ => "",
        }
    }

    fn output_desc(&self, output: usize) -> &'static str {
        match output {
            0 => "left",
            1 => "right",
            _ => "",
        }
    }

    fn name(&self) -> &'static str {
        "PanMonoToStereo"
    }
}

struct NaiveSine {
    phase: Sample,
    phase_step: Sample,
    amp: Sample,
}
impl NaiveSine {
    pub fn update_freq(&mut self, freq: Sample, sample_rate: Sample) {
        self.phase_step = freq / sample_rate;
    }
}

impl Gen for NaiveSine {
    fn process(&mut self, ctx: GenContext, resources: &mut Resources) -> GenState {
        for i in 0..ctx.block_size() {
            let freq = ctx.inputs.read(0, i);
            let amp = ctx.inputs.read(1, i);
            self.update_freq(freq, resources.sample_rate);
            self.amp = amp;
            ctx.outputs.write(self.phase.cos() * self.amp, 0, i);
            self.phase += self.phase_step;
            if self.phase > 1.0 {
                self.phase -= 1.0;
            }
        }
        GenState::Continue
    }
    fn input_desc(&self, input: usize) -> &'static str {
        match input {
            0 => "freq",
            1 => "amp",
            _ => "",
        }
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        1
    }
}

use rtrb::RingBuffer;
use slotmap::{new_key_type, SecondaryMap, SlotMap};
#[cfg(test)]
mod tests {

    use std::sync::atomic::AtomicBool;

    use crate::ResourcesSettings;

    use super::*;
    fn null_input() -> NodeBufferRef {
        NodeBufferRef::null_buffer()
    }
    // Outputs its input value + 1
    struct OneGen {}
    impl Gen for OneGen {
        fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
            for i in 0..ctx.block_size() {
                ctx.outputs.write(ctx.inputs.read(0, i) + 1.0, 0, i);
            }
            GenState::Continue
        }
        fn input_desc(&self, input: usize) -> &'static str {
            match input {
                0 => "passthrough",
                _ => "",
            }
        }

        fn num_inputs(&self) -> usize {
            1
        }

        fn num_outputs(&self) -> usize {
            1
        }
    }
    struct DummyGen {
        counter: f32,
    }
    impl Gen for DummyGen {
        fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
            for i in 0..ctx.block_size() {
                self.counter += 1.;
                ctx.outputs
                    .write(ctx.inputs.read(0, i) + self.counter, 0, i);
            }
            GenState::Continue
        }

        fn input_desc(&self, input: usize) -> &'static str {
            match input {
                0 => "counter",
                _ => "",
            }
        }

        fn num_inputs(&self) -> usize {
            1
        }

        fn num_outputs(&self) -> usize {
            1
        }
    }
    fn graph_node(graph: &mut Graph) -> Node {
        graph.split_and_create_node().unwrap()
    }
    fn test_resources_settings() -> ResourcesSettings {
        ResourcesSettings::default()
    }

    #[test]
    fn create_graph() {
        let mut graph: Graph = Graph::new(GraphSettings::default());
        let node = Node::new("Dummy", Box::new(DummyGen { counter: 0.0 }));
        let node_id = graph.push_node(node);
        graph.connect(Connection::graph_output(node_id)).unwrap();
        graph.init();
        let mut resources = Resources::new(test_resources_settings());
        let mut graph_node = graph_node(&mut graph);
        graph_node.process(&null_input(), &mut resources);
        assert_eq!(graph_node.output_buffers().read(0, 31), 32.0);
    }
    #[test]
    fn multiple_nodes() {
        let mut graph: Graph = Graph::new(GraphSettings {
            block_size: 4,
            ..Default::default()
        });
        let node1 = Node::new("Dummy", Box::new(DummyGen { counter: 0.0 }));
        let node2 = Node::new("Dummy", Box::new(DummyGen { counter: 0.0 }));
        let node3 = Node::new("Dummy", Box::new(DummyGen { counter: 0.0 }));
        let node_id1 = graph.push_node(node1);
        let node_id2 = graph.push_node(node2);
        let node_id3 = graph.push_node(node3);
        graph.connect(node_id1.to(node_id3)).unwrap();
        graph.connect(node_id2.to(node_id3)).unwrap();
        graph.connect(Connection::graph_output(node_id3)).unwrap();
        let mut graph_node = graph_node(&mut graph);
        let mut resources = Resources::new(test_resources_settings());
        graph_node.process(&null_input(), &mut resources);
        assert_eq!(graph_node.output_buffers().read(0, 2), 9.0);
    }
    #[test]
    fn constant_inputs() {
        const BLOCK: usize = 16;
        let mut graph: Graph = Graph::new(GraphSettings {
            block_size: BLOCK,
            ..Default::default()
        });
        let node = Node::new("Dummy", Box::new(DummyGen { counter: 0.0 }));
        let node_id = graph.push_node(node);
        graph.connect(constant(0.5).to(node_id)).unwrap();
        graph.connect(Connection::graph_output(node_id)).unwrap();
        let resources = Resources::new(test_resources_settings());
        let mut run_graph = RunGraph::new(&mut graph, resources).unwrap();
        run_graph.process_block();
        assert_eq!(run_graph.graph_output_buffers().read(0, BLOCK - 1), 16.5);
    }
    #[test]
    fn graph_in_a_graph() {
        const BLOCK: usize = 16;
        const CONSTANT_INPUT_TO_NODE: Sample = 0.25;
        let mut inner_graph: Graph = Graph::new(GraphSettings {
            block_size: BLOCK,
            ..Default::default()
        });
        let node = Node::new("Dummy", Box::new(DummyGen { counter: 0.0 }));
        let node_id = inner_graph.push_node(node);
        inner_graph
            .connect(Connection::graph_output(node_id))
            .unwrap();
        let mut graph: Graph = Graph::new(GraphSettings {
            block_size: BLOCK,
            ..Default::default()
        });
        let inner_graph_node_id = graph.push(inner_graph);
        graph
            .connect(Connection::graph_output(inner_graph_node_id).channels(2))
            .unwrap();
        // Parent graphs should propagate a connection to its child graphs without issue
        graph
            .connect(constant(CONSTANT_INPUT_TO_NODE).to(node_id))
            .unwrap();
        let resources = Resources::new(test_resources_settings());
        let mut run_graph = RunGraph::new(&mut graph, resources).unwrap();
        graph.commit_changes();
        graph.update();
        run_graph.process_block();
        assert_eq!(
            run_graph.graph_output_buffers().read(0, BLOCK - 2),
            (BLOCK - 1) as Sample + CONSTANT_INPUT_TO_NODE
        );
    }
    #[test]
    fn feedback_in_graph() {
        // v-------------<
        // |   2 -> 3 -> 4
        // |   |    |
        // fbv fbv fbv
        //   0     -> 1 -> out
        const BLOCK: usize = 4;
        let mut graph: Graph = Graph::new(GraphSettings {
            block_size: BLOCK,
            ..Default::default()
        });
        let n0 = graph.push_node(Node::new("Dummy", Box::new(DummyGen { counter: 0.0 })));
        let n1 = graph.push_node(Node::new("Dummy", Box::new(DummyGen { counter: 0.0 })));
        let n2 = graph.push_node(Node::new("Dummy", Box::new(DummyGen { counter: 4.0 })));
        let n3 = graph.push_node(Node::new("Dummy", Box::new(DummyGen { counter: 5.0 })));
        let n4 = graph.push_node(Node::new("Dummy", Box::new(DummyGen { counter: 1.0 })));
        graph.connect(Connection::graph_output(n1)).unwrap();
        graph
            .connect(Connection::graph_output(n0).to_index(1))
            .unwrap();
        graph.connect(n0.to(n1)).unwrap();
        graph.connect(n2.to(n3)).unwrap();
        graph.connect(n3.to(n4)).unwrap();
        graph.connect(n2.to(n0).feedback(true)).unwrap();
        graph.connect(n3.to(n1).feedback(true)).unwrap();
        graph.connect(n4.to(n0).feedback(true)).unwrap();
        let mut graph_node = graph_node(&mut graph);
        let mut resources = Resources::new(test_resources_settings());
        graph_node.process(&null_input(), &mut resources);
        let block1 = vec![2.0f32, 4., 6., 8.];
        let block2 = vec![39f32, 47., 55., 63.];
        for (&output, expected) in graph_node
            .output_buffers()
            .get_channel(0)
            .iter()
            .zip(block1)
        {
            assert_eq!(
                output,
                expected,
                "block1 failed with o: {}, expected: {}, n1: {:#?}, n0: {:#?}",
                output,
                expected,
                graph_node.output_buffers().get_channel(0),
                graph_node.output_buffers().get_channel(1)
            );
        }
        graph_node.process(&null_input(), &mut resources);
        for (&output, expected) in graph_node
            .output_buffers()
            .get_channel(0)
            .iter()
            .zip(block2)
        {
            assert_eq!(
                output,
                expected,
                "block2 failed with o: {}, expected: {}, n1: {:#?}, n0: {:#?}",
                output,
                expected,
                graph_node.output_buffers().get_channel(0),
                graph_node.output_buffers().get_channel(1)
            );
        }
    }

    #[test]
    fn changing_the_graph() {
        const BLOCK: usize = 1;
        let mut graph: Graph = Graph::new(GraphSettings {
            block_size: BLOCK,
            ..Default::default()
        });
        let resources = Resources::new(test_resources_settings());
        let mut run_graph = RunGraph::new(&mut graph, resources).unwrap();
        let triangular_sequence = |n| (n * (n + 1)) / 2;
        for i in 0..10 {
            let node = Node::new("Dummy", Box::new(DummyGen { counter: 0.0 }));
            let node_id = graph.push_node(node);
            graph.connect(constant(0.5).to(node_id)).unwrap();
            graph.connect(Connection::graph_output(node_id)).unwrap();
            graph.commit_changes();
            graph.update();
            run_graph.process_block();
            assert_eq!(
                run_graph.graph_output_buffers().read(0, 0),
                (i + 1) as f32 * 0.5 + triangular_sequence(i + 1) as f32,
                "i: {}, output: {}, expected: {}",
                i,
                run_graph.graph_output_buffers().read(0, 0),
                (i + 1) as f32 * 0.5 + triangular_sequence(i + 1) as f32,
            );
        }
    }

    #[test]
    fn graph_inputs() {
        const BLOCK: usize = 1;
        let mut graph: Graph = Graph::new(GraphSettings {
            block_size: BLOCK,
            num_inputs: 3,
            ..Default::default()
        });
        let resources = Resources::new(test_resources_settings());
        let mut run_graph = RunGraph::new(&mut graph, resources).unwrap();
        let node = Node::new("Dummy", Box::new(DummyGen { counter: 0.0 }));
        let node_id = graph.push_node(node);
        graph
            .connect(
                Connection::graph_input(node_id)
                    .from_index(2)
                    .to_label("counter"),
            )
            .unwrap();
        graph.connect(Connection::graph_output(node_id)).unwrap();
        graph.commit_changes();
        run_graph.graph_input_buffers().fill_channel(10.0, 2);
        run_graph.process_block();
        assert_eq!(run_graph.graph_output_buffers().read(0, 0), 11.0);
    }

    #[test]
    fn remove_nodes() {
        const BLOCK: usize = 1;
        let mut graph: Graph = Graph::new(GraphSettings {
            block_size: BLOCK,
            ..Default::default()
        });
        let mut graph_node = graph_node(&mut graph);
        let mut resources = Resources::new(test_resources_settings());
        let mut nodes = vec![];
        let mut last_node = None;
        for _ in 0..10 {
            let node = graph.push(OneGen {});
            if let Some(last) = last_node.take() {
                graph.connect(node.to(last)).unwrap();
            } else {
                graph.connect(Connection::graph_output(node)).unwrap();
            }
            last_node = Some(node);
            nodes.push(node);
        }
        graph.commit_changes();
        graph_node.process(&null_input(), &mut resources);
        assert_eq!(graph_node.output_buffers().read(0, 0), 10.0);
        graph.free_node(nodes[9]).unwrap();
        graph.commit_changes();
        graph_node.process(&null_input(), &mut resources);
        assert_eq!(graph_node.output_buffers().read(0, 0), 9.0);
        graph.free_node(nodes[4]).unwrap();
        graph.commit_changes();
        graph_node.process(&null_input(), &mut resources);
        assert_eq!(graph_node.output_buffers().read(0, 0), 4.0);
        graph.connect(nodes[5].to(nodes[3])).unwrap();
        graph.commit_changes();
        graph_node.process(&null_input(), &mut resources);
        assert_eq!(graph_node.output_buffers().read(0, 0), 8.0);
        for node in &nodes[1..4] {
            graph.free_node(*node).unwrap();
        }
        graph.commit_changes();
        graph_node.process(&null_input(), &mut resources);
        assert_eq!(graph_node.output_buffers().read(0, 0), 1.0);
        assert_eq!(graph.free_node(nodes[4]), Err(FreeError::NodeNotFound));
    }

    #[test]
    fn parallel_mutation() {
        // Here we just want to check that the program doesn't crash/segfault when changing the graph while running the GraphGen.
        const BLOCK: usize = 1;
        let mut graph: Graph = Graph::new(GraphSettings {
            block_size: BLOCK,
            ..Default::default()
        });
        let resources = Resources::new(test_resources_settings());
        let mut run_graph = RunGraph::new(&mut graph, resources).unwrap();
        let mut nodes = vec![];
        let mut last_node = None;
        let done_flag = Arc::new(AtomicBool::new(false));
        let audio_done_flag = done_flag.clone();
        let audio_thread = std::thread::spawn(move || {
            while !audio_done_flag.load(Ordering::SeqCst) {
                run_graph.process_block();
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        });
        for _ in 0..10 {
            let node = graph.push(OneGen {});
            if let Some(last) = last_node.take() {
                graph.connect(node.to(last)).unwrap();
            } else {
                graph.connect(Connection::graph_output(node)).unwrap();
            }
            last_node = Some(node);
            nodes.push(node);
            graph.commit_changes();
            std::thread::sleep(std::time::Duration::from_millis(3));
        }
        for node in nodes.into_iter().rev() {
            graph.free_node(node).unwrap();
            graph.commit_changes();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let mut nodes = vec![];
        last_node = None;
        for _ in 0..10 {
            let node = graph.push(DummyGen { counter: 0. });
            if let Some(last) = last_node.take() {
                graph.connect(node.to(last)).unwrap();
            } else {
                graph.connect(Connection::graph_output(node)).unwrap();
            }
            last_node = Some(node);
            nodes.push(node);
            graph.commit_changes();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        nodes.shuffle(&mut rng);
        for node in nodes.into_iter() {
            graph.free_node(node).unwrap();
            graph.commit_changes();
            graph.update();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        drop(graph);
        done_flag.store(true, Ordering::SeqCst);
        audio_thread.join().unwrap();
    }
    struct SelfFreeing {
        samples_countdown: usize,
        value: Sample,
        mend: bool,
    }
    impl Gen for SelfFreeing {
        fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
            for i in 0..ctx.block_size() {
                if self.samples_countdown == 0 {
                    if self.mend {
                        ctx.outputs.write(ctx.inputs.read(0, i), 0, i);
                    } else {
                        ctx.outputs.write(0., 0, i);
                    }
                } else {
                    ctx.outputs.write(ctx.inputs.read(0, i) + self.value, 0, i);
                    self.samples_countdown -= 1;
                }
            }
            if self.samples_countdown == 0 {
                if self.mend {
                    GenState::FreeSelfMendConnections
                } else {
                    GenState::FreeSelf
                }
            } else {
                GenState::Continue
            }
        }

        fn num_inputs(&self) -> usize {
            1
        }

        fn num_outputs(&self) -> usize {
            1
        }
    }

    #[test]
    fn self_freeing_nodes() {
        const BLOCK: usize = 1;
        let mut graph: Graph = Graph::new(GraphSettings {
            block_size: BLOCK,
            ..Default::default()
        });
        let mut graph_node = graph_node(&mut graph);
        let mut resources = Resources::new(test_resources_settings());
        let n0 = graph.push(SelfFreeing {
            samples_countdown: 2,
            value: 1.,
            mend: true,
        });
        let n1 = graph.push(SelfFreeing {
            samples_countdown: 1,
            value: 1.,
            mend: true,
        });
        let n2 = graph.push(SelfFreeing {
            samples_countdown: 4,
            value: 2.,
            mend: false,
        });
        let n3 = graph.push(SelfFreeing {
            samples_countdown: 5,
            value: 3.,
            mend: false,
        });
        graph.connect(Connection::graph_output(n0)).unwrap();
        graph.connect(n1.to(n0)).unwrap();
        graph.connect(n2.to(n1)).unwrap();
        graph.connect(n3.to(n2)).unwrap();

        graph.commit_changes();
        assert_eq!(graph.num_nodes(), 4);
        graph_node.process(&null_input(), &mut resources);
        assert_eq!(graph_node.output_buffers().read(0, 0), 7.0);
        graph.commit_changes();
        // Still 4 since the node has been added to the free node queue, but there
        // hasn't been a new generation in the GraphGen yet.
        assert_eq!(graph.num_nodes(), 4);
        graph_node.process(&null_input(), &mut resources);
        assert_eq!(graph_node.output_buffers().read(0, 0), 6.0);
        graph.commit_changes();
        assert_eq!(graph.num_nodes(), 3);
        graph_node.process(&null_input(), &mut resources);
        assert_eq!(graph_node.output_buffers().read(0, 0), 5.0);
        graph.commit_changes();
        assert_eq!(graph.num_nodes(), 2);
        graph_node.process(&null_input(), &mut resources);
        assert_eq!(graph_node.output_buffers().read(0, 0), 5.0);
        graph.commit_changes();
        assert_eq!(graph.num_nodes(), 2);
        graph_node.process(&null_input(), &mut resources);
        assert_eq!(graph_node.output_buffers().read(0, 0), 0.0);
        graph.commit_changes();
        assert_eq!(graph.num_nodes(), 1);
        graph_node.process(&null_input(), &mut resources);
        graph.commit_changes();
        assert_eq!(graph.num_nodes(), 0);
    }
    #[test]
    fn scheduling() {
        const BLOCK: usize = 4;
        let mut graph: Graph = Graph::new(GraphSettings {
            block_size: BLOCK,
            latency: Duration::from_millis(0),
            ..Default::default()
        });
        let mut graph_node = graph_node(&mut graph);
        let mut resources = Resources::new(test_resources_settings());
        let node0 = graph.push(OneGen {});
        let node1 = graph.push(OneGen {});
        graph.connect(Connection::graph_output(node0)).unwrap();
        graph.connect(node1.to(node0)).unwrap();
        graph.connect(constant(2.).to(node1)).unwrap();
        graph.commit_changes();
        graph
            .schedule_change(ParameterChange::absolute_samples(node0, 1.0, 3).i(0))
            .unwrap();
        graph
            .schedule_change(ParameterChange::absolute_samples(node0, 2.0, 2).l("passthrough"))
            .unwrap();
        graph
            .schedule_change(ParameterChange::absolute_samples(node1, 10.0, 7).i(0))
            .unwrap();
        // Schedule far into the future
        graph
            .schedule_change(
                ParameterChange::relative_duration(node1, 3000.0, Duration::from_secs(100)).i(0),
            )
            .unwrap();
        graph.update();
        graph_node.process(&null_input(), &mut resources);
        assert_eq!(graph_node.output_buffers().read(0, 0), 4.0);
        assert_eq!(graph_node.output_buffers().read(0, 1), 4.0);
        assert_eq!(graph_node.output_buffers().read(0, 2), 6.0);
        assert_eq!(graph_node.output_buffers().read(0, 3), 5.0);
        graph
            .schedule_change(ParameterChange::absolute_samples(node0, 0.0, 5).i(0))
            .unwrap();
        graph
            .schedule_change(ParameterChange::absolute_samples(node1, 0.0, 6).l("passthrough"))
            .unwrap();
        assert_eq!(
            graph.schedule_change(ParameterChange::absolute_samples(node1, 100.0, 6).l("pasta")),
            Err(ScheduleError::InputLabelNotFound("pasta"))
        );
        graph.update();
        graph_node.process(&null_input(), &mut resources);
        assert_eq!(graph_node.output_buffers().read(0, 0), 5.0);
        assert_eq!(graph_node.output_buffers().read(0, 1), 4.0);
        assert_eq!(graph_node.output_buffers().read(0, 2), 2.0);
        assert_eq!(graph_node.output_buffers().read(0, 3), 12.0);
        // Schedule "now", but this will be a few hundred samples into the
        // future depending on the time it takes to run this test.
        graph
            .schedule_change(ParameterChange::now(node1, 1000.0).i(0))
            .unwrap();
        graph.update();
        // Move a few hundred samples into the future
        for _ in 0..600 {
            graph_node.process(&null_input(), &mut resources);
            graph.update();
        }
        // If this fails, it may be because the machine is too slow. Try
        // increasing the number of iterations above.
        assert_eq!(graph_node.output_buffers().read(0, 0), 1002.0);
    }
    #[test]
    fn index_routing() {
        let graph_settings = GraphSettings {
            block_size: 4,
            sample_rate: 44100.,
            num_outputs: 2,
            ..Default::default()
        };
        let mut graph = Graph::new(graph_settings);
        let resources = Resources::new(ResourcesSettings {
            sample_rate: 44100.,
            ..Default::default()
        });
        let mut run_graph = RunGraph::new(&mut graph, resources).unwrap();
        let mult = graph.push(Mult);
        let five = graph.push(
            gen(move |ctx, _resources| {
                for o0 in ctx.outputs.get_channel_mut(0) {
                    *o0 = 5.;
                }
                GenState::Continue
            })
            .output("o"),
        );
        let nine = graph.push(
            gen(move |ctx, _resources| {
                for o0 in ctx.outputs.get_channel_mut(0) {
                    *o0 = 9.;
                }
                GenState::Continue
            })
            .output("o"),
        );
        // Connecting the node to the graph output
        graph.connect(mult.to_graph_out()).unwrap();
        // Multiply 5 by 9
        graph.connect(five.to(mult).to_index(0)).unwrap();
        graph.connect(nine.to(mult).to_index(1)).unwrap();
        // You need to commit changes and update if the graph is running.
        graph.commit_changes();
        graph.update();
        run_graph.process_block();
        assert_eq!(run_graph.graph_output_buffers().read(0, 0), 45.0);
    }
    #[test]
    fn index_routing_advanced() {
        let graph_settings = GraphSettings {
            block_size: 4,
            sample_rate: 44100.,
            num_outputs: 2,
            ..Default::default()
        };
        let mut graph = Graph::new(graph_settings);
        let resources = Resources::new(ResourcesSettings {
            sample_rate: 44100.,
            ..Default::default()
        });
        let mut run_graph = RunGraph::new(&mut graph, resources).unwrap();
        let multiplier = graph.push(
            gen(move |ctx, _resources| {
                for i in 0..ctx.block_size() {
                    let i0 = ctx.inputs.read(0, i) * 1.;
                    let i1 = ctx.inputs.read(1, i) * 2.;
                    let i2 = ctx.inputs.read(2, i) * 3.;
                    let i3 = ctx.inputs.read(3, i) * 4.;
                    let i4 = ctx.inputs.read(4, i) * 5.;
                    let i5 = ctx.inputs.read(5, i) * 6.;
                    let mut value = i0;
                    for i in [i1, i2, i3, i4, i5] {
                        if i != 0.0 {
                            value *= i;
                        }
                    }
                    ctx.outputs.write(value, 0, i);
                }
                GenState::Continue
            })
            .output("o")
            .input("i0")
            .input("i1")
            .input("i2")
            .input("i3")
            .input("i4")
            .input("i5"),
        );
        let five = graph.push(
            gen(move |ctx, _resources| {
                for o0 in ctx.outputs.get_channel_mut(0) {
                    *o0 = 5.;
                }
                GenState::Continue
            })
            .output("o"),
        );
        let nine = graph.push(
            gen(move |ctx, _resources| {
                for o0 in ctx.outputs.get_channel_mut(0) {
                    *o0 = 9.;
                }
                GenState::Continue
            })
            .output("o"),
        );
        let numberer = graph.push(
            gen(move |ctx, _resources| {
                for i in 0..ctx.block_size() {
                    ctx.outputs.write(0.0, 0, i);
                    ctx.outputs.write(1.0, 1, i);
                    ctx.outputs.write(2.0, 2, i);
                    ctx.outputs.write(3.0, 3, i);
                    ctx.outputs.write(4.0, 4, i);
                    ctx.outputs.write(5.0, 5, i);
                    ctx.outputs.write(6.0, 6, i);
                    ctx.outputs.write(7.0, 7, i);
                    ctx.outputs.write(8.0, 8, i);
                    ctx.outputs.write(9.0, 9, i);
                }
                GenState::Continue
            })
            .output("o0")
            .output("o1")
            .output("o2")
            .output("o3")
            .output("o4")
            .output("o5")
            .output("o6")
            .output("o7")
            .output("o8")
            .output("o9"),
        );
        // Connecting the node to the graph output
        graph.connect(multiplier.to_graph_out()).unwrap();
        // Multiply 5 by 9
        graph.connect(nine.to(multiplier).to_index(0)).unwrap();
        graph.connect(five.to(multiplier).to_index(1)).unwrap();
        // You need to commit changes and update if the graph is running.
        graph.commit_changes();
        graph.update();
        run_graph.process_block();
        assert_eq!(run_graph.graph_output_buffers().read(0, 0), 90.0);

        graph.disconnect(five.to(multiplier).to_index(1)).unwrap();
        graph.connect(five.to(multiplier).to_index(3)).unwrap();
        graph.commit_changes();
        graph.update();
        run_graph.process_block();
        assert_eq!(run_graph.graph_output_buffers().read(0, 0), 180.0);

        graph.connect(five.to(multiplier).to_index(4)).unwrap();
        graph.connect(five.to(multiplier).to_label("i5")).unwrap();
        graph.commit_changes();
        graph.update();
        run_graph.process_block();
        assert_eq!(run_graph.graph_output_buffers().read(0, 0), 135000.0);

        graph.connect(Connection::clear_node_outputs(five)).unwrap();
        graph.commit_changes();
        graph.update();
        run_graph.process_block();
        assert_eq!(run_graph.graph_output_buffers().read(0, 0), 9.0);

        graph
            .connect(numberer.to(multiplier).from_index(2).to_index(4))
            .unwrap();
        graph
            .connect(numberer.to(multiplier).from_index(7).to_index(2))
            .unwrap();
        graph.commit_changes();
        graph.update();
        run_graph.process_block();
        assert_eq!(run_graph.graph_output_buffers().read(0, 0), 1890.0);
    }
}
