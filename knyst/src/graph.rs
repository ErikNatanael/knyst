//! [`Graph`] is the audio graph, the core of Knyst. Implement [`Gen`] and you can be a `Node`.
//!
//! To build an audio graph, add generators (anything implementing the [`Gen`]
//! trait) to a graph and add connections between them.
//!
//! ```
//! # use knyst::prelude::*;
//! # use knyst::osc::*;
//! # use knyst_core::wavetable::*;
//! let graph_settings = GraphSettings {
//!     block_size: 64,
//!     sample_rate: 44100.,
//!     num_outputs: 2,
//!     ..Default::default()
//! };
//! let mut graph = Graph::new(graph_settings);
//! // Adding a node gives you an address to that node
//! let sine_node_address = graph.push(WavetableOscillatorOwned::new(Wavetable::sine()));
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

use knyst_core::gen::{Gen, GenContext, GenState};
use knyst_core::Sample;
// #[cfg(loom)]
// use loom::cell::UnsafeCell;
#[cfg(loom)]
use loom::sync::atomic::Ordering;

#[macro_use]
pub mod connection;
mod graph_gen;
mod node;
pub mod run_graph;
pub use connection::Connection;
use connection::ConnectionError;
pub use knyst_core::node_buffer::NodeBufferRef;
use node::Node;
pub use run_graph::{RunGraph, RunGraphSettings};

use crate::inspection::{EdgeInspection, EdgeSource, GraphInspection, NodeInspection};
use crate::scheduling::MusicalTimeMap;
use crate::time::{Superbeats, Superseconds};
use rtrb::RingBuffer;
use slotmap::{new_key_type, SecondaryMap, SlotMap};

use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet};
use std::mem;
#[cfg(not(loom))]
use std::sync::atomic::Ordering;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use self::connection::{ConnectionBundle, NodeChannel, NodeOutput};

use knyst_core::resources::Resources;
/// The graph consists of (simplified)
/// 1. a list of nodes
/// 2. lists of edges that are inputs per node, outputs of the graph and inputs from the graph input to a node
/// 3. lists of feedback edges that take the values of a node last iteration
///
/// The order that the nodes are processed is calculated based on the edges/connections between nodes using
/// depth-first search from the graph output. Nodes not connected to the graph output will still be run
/// because this is the most intuitive behaviour.
///
/// Each node contains a trait object on the heap with the sound generating object, a `Box<dyn Gen>` Each
/// edge/connection specifies between which output/input of the nodes data is mapped.

/// Unique id identifying a [`Graph`]. Is set from an atomic any time a [`Graph`] is created.
pub type GraphId = u64;

/// Get a unique id for a Graph from this by using `fetch_add`
static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(0);
/// NodeAddresses need to be unique and hashable and this unique number makes it so.
static NEXT_ADDRESS_ID: AtomicU64 = AtomicU64::new(0);

/// An address to a specific Node. The graph_id is constant indepentently of where the graph is (inside some
/// other graph), so it always points to a specific Node in a specific Graph.
#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq)]
pub struct RawNodeAddress {
    key: NodeKey,
    graph_id: GraphId,
}

/// A handle to an address to a specific node. This handle will be updated with
/// the node key and the graph id once those are available. This means that a
/// handle can be created before the node is inserted into a [`Graph`]
#[derive(Clone, Debug)]
pub struct NodeAddress {
    unique_id: u64,
    graph_id: Arc<RwLock<Option<GraphId>>>,
    node_key: Arc<RwLock<Option<NodeKey>>>,
}

impl PartialEq for NodeAddress {
    fn eq(&self, other: &Self) -> bool {
        self.unique_id == other.unique_id
    }
}
impl Eq for NodeAddress {}

impl std::hash::Hash for NodeAddress {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.unique_id.hash(state);
    }
}

impl From<RawNodeAddress> for NodeAddress {
    fn from(raw_address: RawNodeAddress) -> Self {
        let mut a = NodeAddress::new();
        a.set_graph_id(raw_address.graph_id);
        a.set_node_key(raw_address.key);
        a
    }
}

impl NodeAddress {
    /// Create a new [`Self`] not yet connected to a specific node, but with a unique id.
    pub fn new() -> Self {
        Self {
            unique_id: NEXT_ADDRESS_ID.fetch_add(1, Ordering::Relaxed),
            graph_id: Arc::new(RwLock::new(None)),
            node_key: Arc::new(RwLock::new(None)),
        }
    }

    fn set_node_key(&mut self, node_key: NodeKey) {
        match self.node_key.write() {
            Ok(mut lock) => {
                *lock = Some(node_key);
            }
            Err(e) => {
                panic!(
                    "NodeAddress internal node_key RwLock was poisoned. This is unacceptable. {e}"
                );
            }
        }
    }
    fn set_graph_id(&mut self, graph_id: GraphId) {
        match self.graph_id.write() {
            Ok(mut lock) => {
                *lock = Some(graph_id);
            }
            Err(e) => {
                panic!(
                    "NodeAddress internal graph_id RwLock was poisoned. This is unacceptable. {e}"
                );
            }
        }
    }
    fn node_key(&self) -> Option<NodeKey> {
        match self.node_key.read() {
            Ok(data) => *data,
            Err(e) => {
                panic!(
                    "NodeAddress internal node_key RwLock was poisoned. This is unacceptable. {e}"
                );
            }
        }
    }
    /// Retreive [`GraphId`] if one is set.
    pub fn graph_id(&self) -> Option<GraphId> {
        match self.graph_id.read() {
            Ok(data) => *data,
            Err(e) => {
                panic!(
                    "NodeAddress internal graph_id RwLock was poisoned. This is unacceptable. {e}"
                );
            }
        }
    }
    /// Convert to a [`RawNodeAddress`] if possible.
    pub fn to_raw(&self) -> Option<RawNodeAddress> {
        match (self.node_key(), self.graph_id()) {
            (Some(node_key), Some(graph_id)) => Some(RawNodeAddress {
                key: node_key,
                graph_id,
            }),
            _ => None,
        }
    }
    /// Create a [`NodeOutput`] based on `self` and a specific channel.
    pub fn out(&self, channel: impl Into<connection::NodeChannel>) -> NodeOutput {
        NodeOutput {
            from_node: self.clone(),
            from_channel: channel.into(),
        }
    }
}

impl NodeAddress {
    /// Create a [`Connection`] from `self` to `sink_node`
    pub fn to(&self, sink_node: &NodeAddress) -> Connection {
        Connection::Node {
            source: self.clone(),
            from_index: Some(0),
            from_label: None,
            sink: sink_node.clone(),
            to_index: None,
            to_label: None,
            channels: 1,
            feedback: false,
        }
    }
    /// Create a [`Connection`] from `self` to a graph output.
    pub fn to_graph_out(&self) -> Connection {
        Connection::GraphOutput {
            source: self.clone(),
            from_index: Some(0),
            from_label: None,
            to_index: 0,
            channels: 1,
        }
    }
    /// Create a feedback [`Connection`] from `self` to `sink_node`.
    pub fn feedback_to(&self, sink_node: &NodeAddress) -> Connection {
        Connection::Node {
            source: self.clone(),
            from_index: Some(0),
            from_label: None,
            sink: sink_node.clone(),
            to_index: None,
            to_label: None,
            channels: 1,
            feedback: true,
        }
    }
    /// Create a NodeChanges for setting parameters or triggers. Use in
    /// conjunction with [`SimultaneousChanges`] and `schedule_changes`.
    pub fn change(&self) -> NodeChanges {
        NodeChanges::new(self.clone())
    }
}

/// Convenience struct for the notation `GraphInput::to(node)`
pub struct GraphInput;
impl GraphInput {
    /// Create a [`Connection`] from a graph input to `sink_node`
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

#[derive(Clone, Debug)]
/// Bundle multiple changes to multiple nodes. This is usually required to be
/// certain that they get applied in the same frame, except if using absolute
/// time.
///
/// For now, every Node has to be located in the same graph.
pub struct SimultaneousChanges {
    /// When to apply the parameter change(s)
    pub time: Time,
    /// What changes to apply at the same time
    pub changes: Vec<NodeChanges>,
}
impl SimultaneousChanges {
    /// Empty `Self` set to be scheduled as soon as possible
    pub fn now() -> Self {
        Self {
            time: Time::Immediately,
            changes: vec![],
        }
    }
    /// Empty `Self` set to be scheduled a specified wall clock duration from now + latency.
    pub fn duration_from_now(duration: Duration) -> Self {
        Self {
            time: Time::DurationFromNow(duration),
            changes: vec![],
        }
    }
    /// Empty `Self` set to be scheduled a specified beat time.
    pub fn beats(beats: Superbeats) -> Self {
        Self {
            time: Time::Beats(beats),
            changes: vec![],
        }
    }
    /// Push a new [`NodeChanges`] into the list of changes that will be scheduled.
    pub fn push(&mut self, node_changes: NodeChanges) -> &mut Self {
        self.changes.push(node_changes);
        self
    }
}
#[derive(Clone, Debug)]
/// Multiple changes to the input values of a certain node. See [`Change`] for
/// what kinds of changes can be scheduled. All methods return Self so it can be chained:
///
/// # Example
/// ```rust
/// use knyst::graph::{NodeChanges, NodeAddress, SimultaneousChanges};
/// // In reality, use a NodeAddress that points to a node.
/// let my_node_address = NodeAddress::new();
/// let node_changes = NodeChanges::new(my_node_address).set("freq", 442.0).set("amp", 0.5).trigger("reset");
/// // equivalent to:
/// // let node_changes = my_node_address.change().set("freq", 442.0).set("amp", 0.5).trigger("reset");
/// let mut simultaneous_changes = SimultaneousChanges::now();
/// simultaneous_changes.push(node_changes);
/// ```
pub struct NodeChanges {
    node: NodeAddress,
    parameters: Vec<(NodeChannel, Change)>,
    offset: Option<TimeOffset>,
}
impl NodeChanges {
    /// New set of changes for a certain node
    pub fn new(node: NodeAddress) -> Self {
        Self {
            node,
            parameters: vec![],
            offset: None,
        }
    }
    /// Adds a new value to set for a specific channel. Can be chained.
    pub fn set(mut self, channel: impl Into<NodeChannel>, value: Sample) -> Self {
        self.parameters
            .push((channel.into(), Change::Constant(value)));
        self
    }
    /// Adds a trigger for a specific channel. Can be chained.
    pub fn trigger(mut self, channel: impl Into<NodeChannel>) -> Self {
        self.parameters.push((channel.into(), Change::Trigger));
        self
    }
    /// Add this time offset to all the [`Change`]s in the [`NodeChanges`]. If
    /// you want different offsets for different settings to the same node you
    /// can create multiple [`NodeChanges`].
    ///
    /// With the `Time::Immediately` timing variant for the
    /// [`SimultaneousChanges`] only positive offsets are allowed.
    pub fn time_offset(mut self, offset: TimeOffset) -> Self {
        self.offset = Some(offset);
        self
    }
}
#[derive(Clone, Copy, Debug)]
/// Types of changes that can be scheduled for a node in a Graph.
pub enum Change {
    /// Change an input constant to a new value. This value will be added to any
    /// values from node inputs, or to 0 if there are no inputs from other
    /// nodes.
    Constant(Sample),
    /// Schedule a trigger for one sample. A trigger is defined by the
    /// [`trig::is_trigger`] function.
    Trigger,
}

/// A parameter (input constant) change to be scheduled on a [`Graph`].
#[derive(Clone)]
pub struct ParameterChange {
    /// When to apply the parameter change
    pub time: Time,
    /// What node to change
    pub node: NodeAddress,
    /// Channel index
    pub input_index: Option<usize>,
    /// Channel label, index takes precedence if both are set
    pub input_label: Option<&'static str>,
    /// New value of the input constant
    pub value: Sample,
}

impl ParameterChange {
    /// Schedule a change at a specific time in [`Superbeats`]
    pub fn beats(node: NodeAddress, value: Sample, beats: Superbeats) -> Self {
        Self {
            node,
            value,
            time: Time::Beats(beats),
            input_index: None,
            input_label: None,
        }
    }
    /// Schedule a change at a specific time in [`Superseconds`]
    pub fn superseconds(node: NodeAddress, value: Sample, superseconds: Superseconds) -> Self {
        Self {
            node,
            value,
            time: Time::Superseconds(superseconds),
            input_index: None,
            input_label: None,
        }
    }
    /// Schedule a change at a duration from right now.
    pub fn duration_from_now(node: NodeAddress, value: Sample, from_now: Duration) -> Self {
        Self {
            node,
            value,
            time: Time::DurationFromNow(from_now),
            input_index: None,
            input_label: None,
        }
    }
    /// Schedule a change as soon as possible.
    pub fn now(node: NodeAddress, value: Sample) -> Self {
        Self {
            node,
            value,
            time: Time::Immediately,
            input_index: None,
            input_label: None,
        }
    }
    /// Set the index of the input channel to change
    pub fn index(self, index: usize) -> Self {
        self.i(index)
    }
    /// Set the index of the input channel to change
    pub fn i(mut self, index: usize) -> Self {
        self.input_index = Some(index);
        self.input_label = None;
        self
    }
    /// Set the label of the input channel to change
    pub fn label(self, label: &'static str) -> Self {
        self.l(label)
    }
    /// Set the label of the input channel to change
    pub fn l(mut self, label: &'static str) -> Self {
        self.input_label = Some(label);
        self.input_index = None;
        self
    }
}

/// Used to specify a time when a parameter change should be applied.
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug)]
pub enum Time {
    Beats(Superbeats),
    DurationFromNow(Duration),
    Superseconds(Superseconds),
    Immediately,
}

/// Newtype for using a [`Time`] as an offset.
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug)]
pub enum TimeOffset {
    Frames(i64),
    Superseconds(Relative<Superseconds>),
}

impl TimeOffset {
    /// Convert self to a time offset in frames/samples
    pub fn to_frames(&self, sample_rate: u64) -> i64 {
        match self {
            TimeOffset::Frames(frame_offset) => *frame_offset,
            TimeOffset::Superseconds(relative_supserseconds) => match relative_supserseconds {
                Relative::Before(s) => (s.to_samples(sample_rate) as i64) * -1,
                Relative::After(s) => s.to_samples(sample_rate) as i64,
            },
        }
    }
}

/// Used to denote a relative value of an unsigned type, essentially adding a
/// sign.
#[derive(Clone, Copy, Debug)]
pub enum Relative<T> {
    /// A value before the value it is modifying i.e. a negative offset.
    Before(T),
    /// A value after the value it is modifying i.e. a positive offset.
    After(T),
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
    /// When a node is scheduled to start a certain sample this will hold that
    /// time in samples at the local Graph sample rate. Otherwise 0.
    start_node_at_sample: u64,
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
            ScheduledChangeKind::Trigger { index } => {
                self.input_buffers.write(1.0, index, start_sample_in_block);
            }
        }
    }
    fn run(
        &mut self,
        graph_inputs: &NodeBufferRef,
        resources: &mut Resources,
        sample_rate: Sample,
        sample_time_at_block_start: u64,
    ) -> GenState {
        // Copy all inputs
        for (from, to) in &self.inputs_to_copy {
            unsafe {
                **to += **from;
            }
        }
        // Copy all graph inputs
        for (graph_input_index, node_input_index) in &self.graph_inputs_to_copy {
            for i in 0..self.input_buffers.block_size() {
                self.input_buffers.write(
                    graph_inputs.read(*graph_input_index, i),
                    *node_input_index,
                    i,
                );
            }
        }
        if self.start_node_at_sample <= sample_time_at_block_start {
            // Process node
            let mut outputs = NodeBufferRef::new(
                self.output_buffers_first_ptr,
                self.num_outputs,
                self.block_size,
            );
            let ctx = GenContext {
                inputs: &self.input_buffers,
                outputs: &mut outputs,
                sample_rate,
            };
            assert!(!self.gen.is_null());
            unsafe { (*self.gen).process(ctx, resources) }
        } else if ((self.start_node_at_sample - sample_time_at_block_start) as usize)
            < self.block_size
        {
            // The node should start running this block, but only part of the block
            let new_block_size = self.block_size
                - ((self.start_node_at_sample - sample_time_at_block_start) as usize);

            // Process node
            let mut outputs = NodeBufferRef::new(
                self.output_buffers_first_ptr,
                self.num_outputs,
                self.block_size,
            );
            let partial_inputs =
                unsafe { self.input_buffers.to_partial_block_size(new_block_size) };
            let mut partial_outputs = unsafe { outputs.to_partial_block_size(new_block_size) };
            let ctx = GenContext {
                inputs: &partial_inputs,
                outputs: &mut partial_outputs,
                sample_rate,
            };
            assert!(!self.gen.is_null());
            unsafe { (*self.gen).process(ctx, resources) }
        } else {
            // It's not time to run the node yet, just continue
            GenState::Continue
        }
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

/// Error pushing a new node (Gen or Graph) to a Graph
#[allow(missing_docs)]
#[derive(thiserror::Error, Debug)]
pub enum PushError {
    #[error("The target graph was not found. The GenOrGraph that was pushed is returned.")]
    GraphNotFound(GenOrGraphEnum),
}

/// Error freeing a node in a Graph
#[allow(missing_docs)]
#[derive(thiserror::Error, Debug, PartialEq)]
pub enum FreeError {
    #[error(
        "The graph containing the NodeAdress provided was not found. The node itself may or may not exist."
    )]
    GraphNotFound,
    #[error("The NodeAddress does not exist. The Node may have been freed already.")]
    NodeNotFound,
    #[error("The free action required making a new connection, but the connection failed.")]
    ConnectionError(#[from] Box<connection::ConnectionError>),
}
#[allow(missing_docs)]
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    #[error(
        "The graph containing the NodeAdress provided was not found. The node itself may or may not exist."
    )]
    GraphNotFound,
    #[error("The NodeAddress does not exist. The Node may have been freed already.")]
    NodeNotFound,
    #[error("The input label specified was not registered for the node: `{0}`")]
    InputLabelNotFound(&'static str),
    #[error(
        "No scheduler was created for the Graph so the change cannot be scheduled. This is likely because this Graph was not yet added to another Graph or split into a Node."
    )]
    SchedulerNotCreated,
    #[error("A lock for writing to the MusicalTimeMap cannot be acquired.")]
    MusicalTimeMapCannotBeWrittenTo,
}

/// Holds either a boxed [`Gen`] or a [`Graph`]
#[allow(missing_docs)]
pub enum GenOrGraphEnum {
    Gen(Box<dyn Gen + Send>),
    Graph(Graph),
}

impl std::fmt::Debug for GenOrGraphEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            GenOrGraphEnum::Gen(gen) => write!(f, "Gen: {}", gen.name()),
            GenOrGraphEnum::Graph(graph) => write!(f, "Graph: {}, {}", graph.id, graph.name),
        }
    }
}

impl GenOrGraphEnum {
    fn components(
        self,
        parent_graph_block_size: usize,
        parent_graph_sample_rate: Sample,
        parent_graph_oversampling: Oversampling,
    ) -> (Option<Graph>, Box<dyn Gen + Send>) {
        match self {
            GenOrGraphEnum::Gen(boxed_gen) => (None, boxed_gen),
            GenOrGraphEnum::Graph(graph) => graph.components(
                parent_graph_block_size,
                parent_graph_sample_rate,
                parent_graph_oversampling,
            ),
        }
    }
}

impl<T: GenOrGraph> From<T> for GenOrGraphEnum {
    fn from(value: T) -> Self {
        value.into_gen_or_graph_enum()
    }
}
// impl GenOrGraph for GenOrGraphEnum {
//     fn components(self) -> (Option<Graph>, Box<dyn Gen + Send>) {
//         match self {
//             GenOrGraphEnum::Gen(boxed_gen) => (None, boxed_gen),
//             GenOrGraphEnum::Graph(graph) => graph.components(),
//         }
//     }
//     fn into_gen_or_graph_enum(self) -> GenOrGraphEnum {
//         self
//     }
// }

/// This trait is not meant to be implemented by users. In almost all situations
/// you instead want to implement the [`Gen`] trait.
///
/// ToNode allows us to generically push either something that implements Gen or
/// a Graph using the same API.
#[allow(missing_docs)]
pub trait GenOrGraph {
    fn components(
        self,
        parent_graph_block_size: usize,
        parent_graph_sample_rate: Sample,
        parent_graph_oversampling: Oversampling,
    ) -> (Option<Graph>, Box<dyn Gen + Send>);
    fn into_gen_or_graph_enum(self) -> GenOrGraphEnum;
}

impl<T: Gen + Send + 'static> GenOrGraph for T {
    fn components(
        self,
        _parent_graph_block_size: usize,
        _parent_graph_sample_rate: Sample,
        _parent_graph_oversampling: Oversampling,
    ) -> (Option<Graph>, Box<dyn Gen + Send>) {
        (None, Box::new(self))
    }
    fn into_gen_or_graph_enum(self) -> GenOrGraphEnum {
        GenOrGraphEnum::Gen(Box::new(self))
    }
}
// impl<T: Gen + Send> GenOrGraph for Box<T> {
//     fn components(self) -> (Option<Graph>, Box<dyn Gen + Send>) {
//         (None, self)
//     }
//     fn into_gen_or_graph_enum(self) -> GenOrGraphEnum {
//         GenOrGraphEnum::Gen(self)
//     }
// }
impl GenOrGraph for Graph {
    fn components(
        mut self,
        parent_graph_block_size: usize,
        parent_graph_sample_rate: Sample,
        parent_graph_oversampling: Oversampling,
    ) -> (Option<Graph>, Box<dyn Gen + Send>) {
        if self.block_size() > parent_graph_block_size && self.num_inputs() > 0 {
            panic!(
                "Warning: You are pushing a graph with a larger block size and with Graph inputs. An inner Graph with a larger block size cannot have inputs since the inputs for the entire inner block would not have been calculated yet."
            );
        }
        if self.sample_rate != parent_graph_sample_rate {
            eprintln!(
                "Warning: You are pushing a graph with a different sample rate. This is currently allowed, but no automatic resampling will be allowed."
            );
        }
        if self.oversampling.as_usize() < parent_graph_oversampling.as_usize() {
            panic!(
                "You tried to push an inner graph with lower oversampling than its parent. This is not currently allowed."
            );
        }
        // Create the GraphGen from the new Graph
        let gen = self
            .create_graph_gen(
                parent_graph_block_size,
                parent_graph_sample_rate,
                parent_graph_oversampling,
            )
            .unwrap();
        (Some(self), gen)
    }
    fn into_gen_or_graph_enum(self) -> GenOrGraphEnum {
        GenOrGraphEnum::Graph(self)
    }
}

type ProcessFn = Box<dyn (FnMut(GenContext, &mut Resources) -> GenState) + Send>;

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
///     let mut outputs = ctx.outputs.iter_mut();
///     let out0 = outputs.next().unwrap();
///     let out1 = outputs.next().unwrap();
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
    process: impl (FnMut(GenContext, &mut Resources) -> GenState) + 'static + Send,
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
        process: impl (FnMut(GenContext, &mut Resources) -> GenState) + 'static + Send,
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

new_key_type! {
    /// Node identifier in a specific Graph. For referring to a Node outside of the context of a Graph, use NodeAddress instead.
    struct NodeKey;
}

/// Describes the oversampling applied to a graph
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Oversampling {
    X1,
    X2,
    // X4,
    // X8,
    // X16,
    // X32,
}
impl Oversampling {
    /// Convert the oversampling ratio to a usize
    pub fn as_usize(&self) -> usize {
        match self {
            Oversampling::X1 => 1,
            Oversampling::X2 => 2,
            // Oversampling::X4 => 4,
            // Oversampling::X8 => 8,
            // Oversampling::X16 => 16,
            // Oversampling::X32 => 32,
        }
    }
    /// Create Self from a ratio if that ratio is supported
    pub fn from_usize(x: usize) -> Option<Self> {
        match x {
            1 => Some(Oversampling::X1),
            2 => Some(Oversampling::X2),
            // 4 => Some(Oversampling::X4),
            // 8 => Some(Oversampling::X8),
            // 16 => Some(Oversampling::X16),
            // 32 => Some(Oversampling::X32),
            _ => None,
        }
    }
}

/// Pass to `Graph::new` to set the options the Graph is created with in an ergonomic and clear way.
#[derive(Clone, Debug)]
pub struct GraphSettings {
    /// The name of the Graph
    pub name: String,
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
    /// The oversampling factor
    pub oversampling: Oversampling,
    /// The number of messages that can be sent through any of the ring buffers.
    /// Ring buffers are used pass information back and forth between the audio
    /// thread (GraphGen) and the Graph.
    pub ring_buffer_size: usize,
}

impl Default for GraphSettings {
    fn default() -> Self {
        GraphSettings {
            name: String::new(),
            num_inputs: 0,
            num_outputs: 2,
            max_node_inputs: 8,
            block_size: 64,
            num_nodes: 1024,
            sample_rate: 48000.0,
            oversampling: Oversampling::X1,
            ring_buffer_size: 1000,
        }
    }
}

/// Hold on to an allocation and drop it when we're done. Can be easily wrapped
/// in an Arc. This ensures we free the memory.
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
/// `GraphGen` (private). This is done internally when calling [`Graph::push`]ing a [`Graph`].
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
/// use knyst_core::wavetable::*;
/// use knyst::osc::*;
/// use knyst::graph::RunGraph;
/// let graph_settings = GraphSettings {
///     block_size: 64,
///     sample_rate: 44100.,
///     num_outputs: 2,
///     ..Default::default()
/// };
/// let mut graph = Graph::new(graph_settings);
/// let resources = Resources::new(ResourcesSettings::default());
/// let (mut run_graph, _, _) = RunGraph::new(&mut graph, resources, RunGraphSettings::default())?;
/// // Adding a node gives you an address to that node
/// let sine_node_address = graph.push(WavetableOscillatorOwned::new(Wavetable::sine()));
/// // Connecting the node to the graph output
/// graph.connect(sine_node_address.to_graph_out())?;
/// // Set the frequency of the oscillator to 220 Hz. This will
/// // be converted to a scheduled change because the graph is running.
/// graph.connect(constant(220.0).to(&sine_node_address).to_label("freq"))?;
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
    name: String,
    nodes: Arc<UnsafeCell<SlotMap<NodeKey, Node>>>,
    node_keys_to_free_when_safe: Vec<(NodeKey, Arc<AtomicBool>)>,
    /// Set of keys pending removal to easily check if a node is pending
    /// removal. TODO: Maybe it's actually faster and easier to just look
    /// through node_keys_to_free_when_safe than to bother with a HashSet since
    /// this list will almost always be tiny.
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
    /// Every NodeAddress is generated with a unique ID in order to have a
    /// stable hash/eq. Therefore, it needs to be stored here for retreival for
    /// inspection etc.
    node_addresses: SecondaryMap<NodeKey, NodeAddress>,
    node_order: Vec<NodeKey>,
    disconnected_nodes: Vec<NodeKey>,
    feedback_node_indices: Vec<NodeKey>,
    /// If a node is a graph, that graph will be added with the same key here.
    graphs_per_node: SecondaryMap<NodeKey, Graph>,
    /// The outputs of the Graph
    output_edges: Vec<Edge>,
    /// The edges from the graph inputs to nodes, one Vec per node. `source` in the edge is really the sink here.
    graph_input_edges: SecondaryMap<NodeKey, Vec<Edge>>,
    /// If changes have been made that require recalculating the graph this will be set to true.
    recalculation_required: bool,
    num_inputs: usize,
    num_outputs: usize,
    block_size: usize,
    sample_rate: Sample,
    oversampling: Oversampling,
    ring_buffer_size: usize,
    initiated: bool,
    /// Used for processing every node, index using \[input_num\]\[sample_in_block\]
    // inputs_buffers: Vec<Box<[Sample]>>,
    /// A pointer to an allocation that is being used for the inputs to nodes, and aliased in the inputs_buffers
    inputs_buffers_ptr: Arc<OwnedRawBuffer>,
    max_node_inputs: usize,
    graph_gen_communicator: Option<GraphGenCommunicator>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new(GraphSettings::default())
    }
}

impl Graph {
    /// Create a new empty [`Graph`] with a unique atomically generated [`GraphId`]
    pub fn new(options: GraphSettings) -> Self {
        let GraphSettings {
            name,
            num_inputs,
            num_outputs,
            max_node_inputs,
            block_size,
            num_nodes,
            sample_rate,
            oversampling,
            ring_buffer_size,
        } = options;
        let inputs_buffers_ptr = Box::<[Sample]>::into_raw(
            vec![0.0 as Sample; block_size * oversampling.as_usize() * max_node_inputs]
                .into_boxed_slice(),
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
            name,
            nodes,
            node_input_edges,
            node_input_index_to_name: SecondaryMap::with_capacity(num_nodes),
            node_input_name_to_index: SecondaryMap::with_capacity(num_nodes),
            node_output_index_to_name: SecondaryMap::with_capacity(num_nodes),
            node_output_name_to_index: SecondaryMap::with_capacity(num_nodes),
            node_feedback_node_key: SecondaryMap::with_capacity(num_nodes),
            node_feedback_edges,
            node_addresses: SecondaryMap::with_capacity(num_nodes),
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
            oversampling,
            initiated: false,
            inputs_buffers_ptr,
            max_node_inputs,
            ring_buffer_size,
            graph_gen_communicator: None,
            recalculation_required: false,
        }
    }
    /// Create a node that will run this graph. This will fail if a Node or Gen has already been created from the Graph since only one Gen is allowed to exist per Graph.
    ///
    /// Only use this for manually running the main Graph (the Graph containing all other Graphs). For adding a Graph to another Graph, use the push_graph() method.
    fn split_and_create_top_level_node(&mut self) -> Result<Node, String> {
        let block_size = self.block_size();
        // For the top level node, we will set the parent to its own settings, but without oversampling.
        let graph_gen =
            self.create_graph_gen(self.block_size, self.sample_rate, Oversampling::X1)?;
        let mut node = Node::new("graph", graph_gen);
        node.init(block_size, self.sample_rate);
        self.recalculation_required = true;
        Ok(node)
    }

    /// Return the number of input channels to the Graph
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }
    /// Return the number of output channels from the Graph
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }
    /// Return a [`GraphSettings`] matching this [`Graph`]
    pub fn graph_settings(&self) -> GraphSettings {
        GraphSettings {
            name: self.name.clone(),
            num_inputs: self.num_inputs,
            max_node_inputs: self.max_node_inputs,
            num_outputs: self.num_outputs,
            block_size: self.block_size,
            num_nodes: self.get_nodes().capacity(),
            sample_rate: self.sample_rate,
            oversampling: self.oversampling,
            ring_buffer_size: self.ring_buffer_size,
        }
    }
    /// Returns a number including both active nodes and nodes waiting to be safely freed
    pub fn num_stored_nodes(&self) -> usize {
        self.get_nodes().len()
    }
    #[allow(missing_docs)]
    pub fn id(&self) -> GraphId {
        self.id
    }
    /// Push something implementing [`Gen`] or a [`Graph`] to self creating a
    /// new node whose address is returned.
    pub fn push(&mut self, to_node: impl Into<GenOrGraphEnum>) -> NodeAddress {
        self.push_to_graph(to_node, self.id).unwrap()
    }
    /// Push something implementing [`Gen`] or a [`Graph`] to self creating a
    /// new node whose address is returned. The node will start at `start_time`.
    pub fn push_at_time(
        &mut self,
        to_node: impl Into<GenOrGraphEnum>,
        start_time: Superseconds,
    ) -> NodeAddress {
        self.push_to_graph_at_time(to_node, self.id, start_time)
            .unwrap()
    }
    /// Push something implementing [`Gen`] or a [`Graph`] to the graph with the
    /// id provided, creating a new node whose address is returned.
    pub fn push_to_graph(
        &mut self,
        to_node: impl Into<GenOrGraphEnum>,
        graph_id: GraphId,
    ) -> Result<NodeAddress, PushError> {
        let mut new_node_address = NodeAddress::new();
        self.push_with_existing_address_to_graph(to_node, &mut new_node_address, graph_id)?;
        Ok(new_node_address)
    }
    /// Push something implementing [`Gen`] or a [`Graph`] to the graph with the
    /// id provided, creating a new node whose address is returned.
    pub fn push_to_graph_at_time(
        &mut self,
        to_node: impl Into<GenOrGraphEnum>,
        graph_id: GraphId,
        start_time: Superseconds,
    ) -> Result<NodeAddress, PushError> {
        let mut new_node_address = NodeAddress::new();
        self.push_with_existing_address_to_graph_at_time(
            to_node,
            &mut new_node_address,
            graph_id,
            start_time,
        )?;
        Ok(new_node_address)
    }
    /// Push something implementing [`Gen`] or a [`Graph`] to self, storing its
    /// address in the NodeAddress provided.
    pub fn push_with_existing_address(
        &mut self,
        to_node: impl Into<GenOrGraphEnum>,
        node_address: &mut NodeAddress,
    ) {
        self.push_with_existing_address_to_graph(to_node, node_address, self.id)
            .unwrap()
    }
    /// Push something implementing [`Gen`] or a [`Graph`] to self, storing its
    /// address in the NodeAddress provided. The node will start processing at
    /// the `start_time`.
    pub fn push_with_existing_address_at_time(
        &mut self,
        to_node: impl Into<GenOrGraphEnum>,
        node_address: &mut NodeAddress,
        start_time: Superseconds,
    ) {
        self.push_with_existing_address_to_graph_at_time(to_node, node_address, self.id, start_time)
            .unwrap()
    }
    /// Push something implementing [`Gen`] or a [`Graph`] to the graph with the
    /// id provided, storing its address in the NodeAddress provided.
    pub fn push_with_existing_address_to_graph(
        &mut self,
        to_node: impl Into<GenOrGraphEnum>,
        node_address: &mut NodeAddress,
        graph_id: GraphId,
    ) -> Result<(), PushError> {
        self.push_with_existing_address_to_graph_at_time(
            to_node,
            node_address,
            graph_id,
            Superseconds::ZERO,
        )
    }
    /// Push something implementing [`Gen`] or a [`Graph`] to the graph with the
    /// id provided, storing its address in the NodeAddress provided. The node
    /// will start processing at the `start_time`.
    pub fn push_with_existing_address_to_graph_at_time(
        &mut self,
        to_node: impl Into<GenOrGraphEnum>,
        node_address: &mut NodeAddress,
        graph_id: GraphId,
        start_time: Superseconds,
    ) -> Result<(), PushError> {
        if graph_id == self.id {
            let (graph, gen) =
                to_node
                    .into()
                    .components(self.block_size, self.sample_rate, self.oversampling);
            let mut node = Node::new(gen.name(), gen);
            let start_sample_time = start_time.to_samples(self.sample_rate as u64);
            node.start_at_sample(start_sample_time);
            self.push_node(node, node_address);
            if let Some(mut graph) = graph {
                // Important: we must start the scheduler here if the current
                // graph is started, otherwise it will never start.
                if let Some(ggc) = &mut self.graph_gen_communicator {
                    if let Scheduler::Running {
                        start_ts,
                        latency_in_samples,
                        musical_time_map,
                        ..
                    } = &mut ggc.scheduler
                    {
                        let clock_update = ClockUpdate {
                            timestamp: ggc.timestamp.clone(),
                            clock_sample_rate: self.sample_rate,
                        };
                        let latency = Duration::from_secs_f64(
                            *latency_in_samples / (self.sample_rate as f64),
                        );
                        graph.start_scheduler(
                            latency,
                            *start_ts,
                            &Some(clock_update),
                            musical_time_map,
                        );
                    }
                }
                self.graphs_per_node
                    .insert(node_address.node_key().unwrap(), graph);
            }
            Ok(())
        } else {
            // Try to find the graph containing the node by asking all the graphs in this graph to free the node
            let mut to_node = to_node.into();
            for (_key, graph) in &mut self.graphs_per_node {
                match graph.push_with_existing_address_to_graph(to_node, node_address, graph_id) {
                    Ok(_) => {
                        return Ok(());
                    }
                    // Return the error unless it's a GraphNotFound in which case we continue trying
                    Err(e) => match e {
                        PushError::GraphNotFound(returned_to_node) => {
                            to_node = returned_to_node;
                        }
                    },
                }
            }
            Err(PushError::GraphNotFound(to_node))
        }
    }
    /// Add a node to this Graph. The Node will be (re)initialised with the
    /// correct block size for this Graph.
    ///
    /// Provide a [`NodeAddress`] so that a pre-created async NodeAddress can be
    /// connected to this node. A new NodeAddress can also be passed in. Either
    /// way, it will be connected to this node.
    ///
    /// Making it not public means Graphs cannot be accidentally added, but a
    /// Node<Graph> can still be created for the top level one if preferred.
    fn push_node(&mut self, mut node: Node, node_address: &mut NodeAddress) {
        if node.num_inputs() > self.max_node_inputs {
            eprintln!(
                "Warning: You are trying to add a node with more inputs than the maximum for this Graph. Try increasing the maximum number of node inputs in the GraphSettings."
            );
        }
        let nodes = self.get_nodes();
        if nodes.capacity() == nodes.len() {
            eprintln!(
                "Error: Trying to push a node into a Graph that is at capacity. Try increasing the number of node slots and make sure you free the nodes you don't need."
            );
        }
        self.recalculation_required = true;
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
        node.init(
            self.block_size * self.oversampling.as_usize(),
            self.sample_rate * (self.oversampling.as_usize() as f32),
        );
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

        node_address.set_graph_id(self.id);
        node_address.set_node_key(key);
        self.node_addresses.insert(key, node_address.clone());
    }
    /// Remove all nodes in this graph and all its subgraphs that are not connected to anything.
    pub fn free_disconnected_nodes(&mut self) -> Result<(), FreeError> {
        // The easiest way to do it would be to store disconnected nodes after
        // calculating the node order of the graph. (i.e. all the nodes that
        // weren't visited)
        // TODO: This method should be infallible because any error is an internal bug.

        let disconnected_nodes = std::mem::take(&mut self.disconnected_nodes);
        if disconnected_nodes.len() > 0 {
            self.recalculation_required = true;
        }
        for node in disconnected_nodes {
            match self.free_node(RawNodeAddress {
                key: node,
                graph_id: self.id,
            }) {
                Ok(_) => (),
                Err(e) => {
                    return Err(e);
                }
            }
        }
        Ok(())
    }
    /// Remove the node and connect its input edges to the sinks of its output edges
    pub fn free_node_mend_connections(
        &mut self,
        node: impl Into<RawNodeAddress>,
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
            self.recalculation_required = true;

            let num_inputs = self.node_input_index_to_name
                .get(node.key)
                .expect(
                    "Since the key exists in the Graph it should have a corresponding node_input_index_to_name Vec"
                )
                .len();
            let num_outputs = self.node_output_index_to_name
                .get(node.key)
                .expect(
                    "Since the key exists in the Graph it should have a corresponding node_output_index_to_name Vec"
                )
                .len();
            let inputs_to_bridge = num_inputs.min(num_outputs);
            // First collect all the connections that should be bridged so that they are in one place
            let mut outputs = vec![vec![]; inputs_to_bridge];
            for (destination_node_key, edge_vec) in &self.node_input_edges {
                for edge in edge_vec {
                    if edge.source == node.key && edge.from_output_index < inputs_to_bridge {
                        outputs[edge.from_output_index].push(Connection::Node {
                            source: (RawNodeAddress {
                                graph_id: self.id,
                                key: node.key,
                            })
                            .into(),
                            from_index: Some(edge.from_output_index),
                            from_label: None,
                            sink: (RawNodeAddress {
                                graph_id: self.id,
                                key: destination_node_key,
                            })
                            .into(),
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
                        source: (RawNodeAddress {
                            graph_id: self.id,
                            key: node.key,
                        })
                        .into(),
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
                        sink: (RawNodeAddress {
                            graph_id: self.id,
                            key: node.key,
                        })
                        .into(),
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
                        // We are not certain that the input node has as many
                        // outputs as the node being freed.
                        let num_node_outputs =
                            self.get_nodes().get(input.source).unwrap().num_outputs();
                        let mut connection = output.clone();
                        if let Some(connection_from_index) = connection.get_from_index() {
                            connection =
                                connection.from_index(connection_from_index % num_node_outputs);
                        }
                        self.connect(
                            connection.from(
                                &(RawNodeAddress {
                                    key: input.source,
                                    graph_id: self.id,
                                })
                                .into(),
                            ),
                        )
                        .expect("Mended connections should be guaranteed to succeed");
                    }
                }
            }
            for (graph_inputs, outputs) in graph_inputs.into_iter().zip(outputs.iter()) {
                for input in graph_inputs {
                    for output in outputs {
                        let connection = input.clone();
                        match self.connect(
                            connection
                                .to(&output.get_source_node().unwrap())
                                .to_index(output.get_to_index().unwrap()),
                        ) {
                            Ok(_) => (),
                            Err(e) => {
                                return Err(FreeError::ConnectionError(Box::new(e)));
                            }
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
                    Ok(_) => {
                        return Ok(());
                    }
                    Err(e) => match e {
                        FreeError::GraphNotFound => (),
                        _ => {
                            return Err(e);
                        }
                    },
                }
            }
            Err(FreeError::GraphNotFound)
        }
    }
    /// Remove the node and any edges to/from the node. This may lead to other nodes being disconnected from the output and therefore not be run, but they will not be freed.
    pub fn free_node(&mut self, node: impl Into<RawNodeAddress>) -> Result<(), FreeError> {
        let node = node.into();
        if node.graph_id == self.id {
            // Does the Node exist?
            if !self.get_nodes_mut().contains_key(node.key) {
                return Err(FreeError::NodeNotFound);
            }

            self.recalculation_required = true;

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
                nodes_to_free.insert(RawNodeAddress {
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
                        nodes_to_free.insert(RawNodeAddress {
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
                    Ok(_) => {
                        return Ok(());
                    }
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

    fn start_scheduler(
        &mut self,
        latency: Duration,
        start_ts: Instant,
        clock_update: &Option<ClockUpdate>,
        musical_time_map: &Arc<RwLock<MusicalTimeMap>>,
    ) {
        if let Some(ggc) = &mut self.graph_gen_communicator {
            if let Some(clock_update) = &clock_update {
                ggc.send_clock_update(clock_update.clone()); // Make sure all the clocks in the GraphGens are in sync.
            }
            ggc.scheduler.start(
                self.sample_rate * (self.oversampling.as_usize() as f32),
                self.block_size * self.oversampling.as_usize(),
                latency,
                start_ts,
                musical_time_map.clone(),
            );
        }
        for (_key, graph) in &mut self.graphs_per_node {
            graph.start_scheduler(latency, start_ts, clock_update, musical_time_map);
        }
    }
    /// Returns the current audio thread time in Superbeats based on the
    /// MusicalTimeMap, or None if it is not available (e.g. if the Graph has
    /// not been started yet).
    pub fn get_current_time_musical(&self) -> Option<Superbeats> {
        if let Some(ggc) = &self.graph_gen_communicator {
            let ts_samples = ggc.timestamp.load(Ordering::Relaxed);
            let seconds = (ts_samples as f64) / (self.sample_rate as f64);
            ggc.scheduler
                .seconds_to_musical_time_superbeats(Superseconds::from_seconds_f64(seconds))
        } else {
            None
        }
    }
    /// Generate inspection metadata for this graph and all sub graphs. Can be
    /// used to generate static or dynamic inspection and manipulation tools.
    pub fn generate_inspection(&self) -> GraphInspection {
        let real_nodes = self.get_nodes();
        // Maps a node key to the index in the Vec
        let mut node_key_processed = Vec::with_capacity(real_nodes.len());
        let mut nodes = Vec::with_capacity(real_nodes.len());
        for (node_key, node) in real_nodes {
            let graph_inspection = self
                .graphs_per_node
                .get(node_key)
                .map(|graph| graph.generate_inspection());

            nodes.push(NodeInspection {
                name: node.name.to_string(),
                address: self.node_addresses
                    .get(node_key)
                    .expect("All nodes should have their addresses stored.")
                    .clone(),
                input_channels: self.node_input_index_to_name
                    .get(node_key)
                    .expect(
                        "All nodes should have a list of input channel names made when pushed to the graph."
                    )
                    .iter()
                    .map(|&s| s.to_string())
                    .collect(),
                output_channels: self.node_output_index_to_name
                    .get(node_key)
                    .expect(
                        "All nodes should have a list of output channel names made when pushed to the graph."
                    )
                    .iter()
                    .map(|&s| s.to_string())
                    .collect(),
                // Leave empty for now, fill later
                input_edges: vec![],
                graph_inspection,
            });
            node_key_processed.push(node_key);
        }
        // Convert edges to the inspection native format of indices to the list of nodes
        for (node_key, _node) in real_nodes {
            let mut input_edges = Vec::new();
            if let Some(edges) = self.node_input_edges.get(node_key) {
                for edge in edges {
                    let index = node_key_processed
                        .iter()
                        .position(|&k| k == edge.source)
                        .unwrap();
                    input_edges.push(EdgeInspection {
                        source: EdgeSource::Node(index),
                        from_index: edge.from_output_index,
                        to_index: edge.to_input_index,
                    });
                }
            }
            if let Some(edges) = self.graph_input_edges.get(node_key) {
                for edge in edges {
                    input_edges.push(EdgeInspection {
                        source: EdgeSource::Graph,
                        from_index: edge.from_output_index,
                        to_index: edge.to_input_index,
                    });
                }
            }
            let index = node_key_processed
                .iter()
                .position(|&k| k == node_key)
                .unwrap();
            nodes[index].input_edges = input_edges;
        }
        let mut graph_output_input_edges = Vec::new();
        for edge in &self.output_edges {
            let index = node_key_processed
                .iter()
                .position(|&k| k == edge.source)
                .unwrap();
            graph_output_input_edges.push(EdgeInspection {
                source: EdgeSource::Node(index),
                from_index: edge.from_output_index,
                to_index: edge.to_input_index,
            });
        }
        let unconnected_nodes = self
            .disconnected_nodes
            .iter()
            .map(|&disconnected_key| {
                node_key_processed
                    .iter()
                    .position(|&key| key == disconnected_key)
                    .unwrap()
            })
            .collect();
        let nodes_pending_removal = self
            .node_keys_to_free_when_safe
            .iter()
            .map(|&(freed_key, _)| {
                node_key_processed
                    .iter()
                    .position(|&key| key == freed_key)
                    .unwrap()
            })
            .collect();

        GraphInspection {
            nodes,
            unconnected_nodes,
            nodes_pending_removal,
            graph_output_input_edges,
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
            graph_id: self.id,
        }
    }

    /// Schedule changes to input channel constants. The changes will only be
    /// applied if the [`Graph`] is running and its scheduler is regularly
    /// updated.
    pub fn schedule_changes(&mut self, changes: SimultaneousChanges) -> Result<(), ScheduleError> {
        let mut scheduler_changes = vec![];
        for node_changes in &changes.changes {
            let node = &node_changes.node;
            let change_pairs = &node_changes.parameters;
            let time_offset = node_changes.offset;
            if let Some(raw_node_address) = node.to_raw() {
                if raw_node_address.graph_id == self.id {
                    // Does the Node exist?
                    if !self.get_nodes_mut().contains_key(raw_node_address.key) {
                        return Err(ScheduleError::NodeNotFound);
                    }
                    for (channel, change) in change_pairs {
                        let index = match channel {
                            NodeChannel::Label(label) => {
                                if let Some(label_index) =
                                    self.node_input_name_to_index[raw_node_address.key].get(label)
                                {
                                    *label_index
                                } else {
                                    return Err(ScheduleError::InputLabelNotFound(label));
                                }
                            }
                            NodeChannel::Index(index) => *index,
                        };

                        let change_kind = match change {
                            Change::Constant(value) => ScheduledChangeKind::Constant {
                                index,
                                value: *value,
                            },
                            Change::Trigger => ScheduledChangeKind::Trigger { index },
                        };
                        scheduler_changes.push((raw_node_address.key, change_kind, time_offset));
                    }
                } else {
                    // Try to find the graph containing the node by asking all the graphs in this graph to free the node
                    for (_key, graph) in &mut self.graphs_per_node {
                        match graph.schedule_changes(changes.clone()) {
                            Ok(_) => {
                                return Ok(());
                            }
                            Err(e) => match e {
                                ScheduleError::GraphNotFound => (),
                                _ => {
                                    return Err(e);
                                }
                            },
                        }
                    }
                    return Err(ScheduleError::GraphNotFound);
                }
            } else {
                // TODO: Add the change to a queue?
                todo!();
            }
        }
        if let Some(ggc) = &mut self.graph_gen_communicator {
            ggc.scheduler.schedule(scheduler_changes, changes.time);
        } else {
            return Err(ScheduleError::SchedulerNotCreated);
        }
        Ok(())
    }
    /// Schedule a change to an input channel constant. The change will only be
    /// applied if the [`Graph`] is running and its scheduler is regularly
    /// updated.
    pub fn schedule_change(&mut self, change: ParameterChange) -> Result<(), ScheduleError> {
        if let Some(raw_node_address) = change.node.to_raw() {
            if raw_node_address.graph_id == self.id {
                // Does the Node exist?
                if !self.get_nodes_mut().contains_key(raw_node_address.key) {
                    return Err(ScheduleError::NodeNotFound);
                }
                let index = if let Some(label) = change.input_label {
                    if let Some(label_index) =
                        self.node_input_name_to_index[raw_node_address.key].get(label)
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
                    ggc.scheduler
                        .schedule(vec![(raw_node_address.key, change_kind, None)], change.time);
                } else {
                    return Err(ScheduleError::SchedulerNotCreated);
                }
            } else {
                // Try to find the graph containing the node by asking all the graphs in this graph to free the node
                for (_key, graph) in &mut self.graphs_per_node {
                    match graph.schedule_change(change.clone()) {
                        Ok(_) => {
                            return Ok(());
                        }
                        Err(e) => match e {
                            ScheduleError::GraphNotFound => (),
                            _ => {
                                return Err(e);
                            }
                        },
                    }
                }
                return Err(ScheduleError::GraphNotFound);
            }
        } else {
            // TODO: Add the change to a queue?
            todo!();
        }
        Ok(())
    }
    /// Disconnect the given connection if it exists. Will return Ok if the Connection doesn't exist, but the data inside it is correct and the graph could be found.
    ///
    /// Disconnecting a constant means setting that constant input to 0. Disconnecting a feedback edge will remove the feedback node under the hood if there are no remaining edges to it. Disconnecting a Connection::Clear will do the same thing as "connecting" it: clear edges according to its parameters.
    pub fn disconnect(&mut self, connection: Connection) -> Result<(), ConnectionError> {
        let mut try_disconnect_in_child_graphs = |connection: Connection| {
            for (_key, graph) in &mut self.graphs_per_node {
                match graph.disconnect(connection.clone()) {
                    Ok(_) => {
                        return Ok(());
                    }
                    Err(e) => match &e {
                        ConnectionError::NodeNotFound => {
                            // The correct graph was found, but the node wasn't in it.
                            return Err(ConnectionError::NodeNotFound);
                        }
                        // We continue trying other graphs
                        ConnectionError::GraphNotFound => (),
                        _ => {
                            return Err(e);
                        }
                    },
                }
            }
            Err(ConnectionError::GraphNotFound)
        };

        match connection {
            Connection::Node {
                ref source,
                from_index,
                from_label,
                ref sink,
                to_index: input_index,
                to_label: input_label,
                channels,
                feedback,
            } => {
                let source = if let Some(raw_source) = source.to_raw() {
                    raw_source
                } else {
                    return Err(ConnectionError::SourceNodeNotPushed);
                };
                let sink = if let Some(raw_sink) = sink.to_raw() {
                    raw_sink
                } else {
                    return Err(ConnectionError::SinkNodeNotPushed);
                };
                if source.graph_id != sink.graph_id {
                    return Err(ConnectionError::DifferentGraphs);
                }
                if source.graph_id != self.id {
                    return try_disconnect_in_child_graphs(connection.clone());
                }
                if source.key == sink.key {
                    return Err(ConnectionError::SameNode);
                }
                if !self.get_nodes().contains_key(source.key) {
                    return Err(ConnectionError::NodeNotFound);
                }
                if !self.get_nodes().contains_key(sink.key) {
                    return Err(ConnectionError::NodeNotFound);
                }
                if self.node_keys_pending_removal.contains(&source.key) {
                    return Err(ConnectionError::NodeNotFound);
                }
                if self.node_keys_pending_removal.contains(&sink.key) {
                    return Err(ConnectionError::NodeNotFound);
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
                        self.free_node(RawNodeAddress {
                            key: feedback_node,
                            graph_id: self.id,
                        })?;
                    }
                }
            }
            Connection::Constant {
                value: _,
                ref sink,
                to_index: input_index,
                to_label: input_label,
            } => {
                if let Some(sink) = sink {
                    let sink = if let Some(raw_sink) = sink.to_raw() {
                        raw_sink
                    } else {
                        return Err(ConnectionError::SinkNodeNotPushed);
                    };
                    if sink.graph_id != self.id {
                        return try_disconnect_in_child_graphs(connection.clone());
                    }
                    if !self.get_nodes().contains_key(sink.key) {
                        return Err(ConnectionError::NodeNotFound);
                    }
                    if self.node_keys_pending_removal.contains(&sink.key) {
                        return Err(ConnectionError::NodeNotFound);
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
                        ggc.scheduler.schedule(
                            vec![(
                                sink.key,
                                ScheduledChangeKind::Constant {
                                    index: input,
                                    value: 0.0,
                                },
                                None,
                            )],
                            Time::Immediately,
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
                ref source,
                from_index,
                from_label,
                to_index,
                channels,
            } => {
                let source = if let Some(raw_source) = source.to_raw() {
                    raw_source
                } else {
                    return Err(ConnectionError::SinkNodeNotPushed);
                };
                if source.graph_id != self.id {
                    return try_disconnect_in_child_graphs(connection.clone());
                }
                if !self.get_nodes().contains_key(source.key) {
                    return Err(ConnectionError::NodeNotFound);
                }
                if self.node_keys_pending_removal.contains(&source.key) {
                    return Err(ConnectionError::NodeNotFound);
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
                ref sink,
                from_index,
                to_index,
                to_label,
                channels,
            } => {
                let sink = if let Some(raw_sink) = sink.to_raw() {
                    raw_sink
                } else {
                    return Err(ConnectionError::SinkNodeNotPushed);
                };
                if sink.graph_id != self.id {
                    return try_disconnect_in_child_graphs(connection.clone());
                }
                if !self.get_nodes().contains_key(sink.key) {
                    return Err(ConnectionError::NodeNotFound);
                }
                if self.node_keys_pending_removal.contains(&sink.key) {
                    return Err(ConnectionError::NodeNotFound);
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
        // If no error was encountered we end up here and a recalculation is required.
        self.recalculation_required = true;
        Ok(())
    }
    /// Make several connections at once. If one connection fails the function
    /// will return and any remaining connections will be lost.
    pub fn connect_bundle(
        &mut self,
        bundle: impl Into<ConnectionBundle>,
    ) -> Result<(), ConnectionError> {
        let bundle = bundle.into();
        for con in bundle.as_connections() {
            match self.connect(con) {
                Ok(_) => (),
                Err(e) => {
                    return Err(e);
                }
            }
        }
        Ok(())
    }
    /// Create or clear a connection in the Graph. Will call child Graphs until
    /// the graph containing the nodes is found or return an error if the right
    /// Graph or Node cannot be found.
    pub fn connect(&mut self, connection: Connection) -> Result<(), ConnectionError> {
        let mut try_connect_to_graphs = |connection: Connection| {
            for (_key, graph) in &mut self.graphs_per_node {
                match graph.connect(connection.clone()) {
                    Ok(_) => {
                        return Ok(());
                    }
                    Err(e) => match &e {
                        ConnectionError::NodeNotFound => {
                            // The correct graph was found, but the node wasn't in it.
                            return Err(ConnectionError::NodeNotFound);
                        }
                        // We continue trying other graphs
                        ConnectionError::GraphNotFound => (),
                        _ => {
                            return Err(e);
                        }
                    },
                }
            }
            Err(ConnectionError::GraphNotFound)
        };
        match connection {
            Connection::Node {
                ref source,
                from_index,
                from_label,
                ref sink,
                to_index: input_index,
                to_label: input_label,
                channels,
                feedback,
            } => {
                // Convert from NodeAddress to RawNodeAddress, returning an error if it isn't possible
                let source = if let Some(raw_source) = source.to_raw() {
                    raw_source
                } else {
                    return Err(ConnectionError::SourceNodeNotPushed);
                };
                let sink = if let Some(raw_sink) = sink.to_raw() {
                    raw_sink
                } else {
                    return Err(ConnectionError::SinkNodeNotPushed);
                };
                // Check that the nodes are valid
                if source.graph_id != sink.graph_id {
                    return Err(ConnectionError::DifferentGraphs);
                }
                if source.graph_id != self.id {
                    return try_connect_to_graphs(connection);
                }
                if source.key == sink.key {
                    return Err(ConnectionError::SameNode);
                }
                if !self.get_nodes().contains_key(source.key) {
                    return Err(ConnectionError::NodeNotFound);
                }
                if !self.get_nodes().contains_key(sink.key) {
                    return Err(ConnectionError::NodeNotFound);
                }
                if self.node_keys_pending_removal.contains(&source.key) {
                    return Err(ConnectionError::NodeNotFound);
                }
                if self.node_keys_pending_removal.contains(&sink.key) {
                    return Err(ConnectionError::NodeNotFound);
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
                // Alternative way to get the num_inputs and outputs without accessing the node
                let num_source_outputs = self
                    .node_output_index_to_name
                    .get(source.key)
                    .unwrap()
                    .len();
                let num_sink_inputs = self.node_input_index_to_name.get(sink.key).unwrap().len();
                if !feedback {
                    let edge_list = &mut self.node_input_edges[sink.key];
                    for i in 0..channels {
                        edge_list.push(Edge {
                            // wrap channels if there are too many
                            from_output_index: (from_index + i) % num_source_outputs,
                            source: source.key,
                            // wrap channels if there are too many
                            to_input_index: (to_index + i) % num_sink_inputs,
                        });
                    }
                } else {
                    // Create a feedback node if there isn't one.
                    let feedback_node_key =
                        if let Some(&index) = self.node_feedback_node_key.get(source.key) {
                            index
                        } else {
                            let feedback_node = FeedbackGen::node(num_source_outputs);
                            let mut feedback_node_address = NodeAddress::new();
                            self.push_node(feedback_node, &mut feedback_node_address);
                            let address = feedback_node_address.to_raw().unwrap();
                            self.feedback_node_indices.push(address.key);
                            self.node_feedback_node_key.insert(source.key, address.key);
                            address.key
                        };
                    // Create feedback edges leading to the FeedbackNode from
                    // the source and normal edges leading from the FeedbackNode
                    // to the sink.
                    let edge_list = &mut self.node_input_edges[sink.key];
                    for i in 0..channels {
                        edge_list.push(Edge {
                            from_output_index: (from_index + i) % num_source_outputs,
                            source: feedback_node_key,
                            to_input_index: (to_index + i) % num_sink_inputs,
                        });
                    }
                    let edge_list = &mut self.node_feedback_edges[feedback_node_key];
                    for i in 0..channels {
                        edge_list.push(FeedbackEdge {
                            from_output_index: (from_index + i) % num_source_outputs,
                            source: source.key,
                            to_input_index: (from_index + i) % num_source_outputs,
                            feedback_destination: sink.key,
                        });
                    }
                }
            }
            Connection::Constant {
                value,
                ref sink,
                to_index: input_index,
                to_label: input_label,
            } => {
                if let Some(sink) = sink {
                    // Convert from NodeAddress to RawNodeAddress, returning an error if it isn't possible
                    let Some(sink) = sink.to_raw() else {
                        return Err(ConnectionError::SinkNodeNotPushed);
                    };
                    // If the sink node does not belong in this graph, try a sub graph of this graph
                    if sink.graph_id != self.id {
                        return try_connect_to_graphs(connection);
                    }
                    if !self.get_nodes().contains_key(sink.key) {
                        return Err(ConnectionError::NodeNotFound);
                    }
                    if self.node_keys_pending_removal.contains(&sink.key) {
                        return Err(ConnectionError::NodeNotFound);
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
                        ggc.scheduler.schedule(
                            vec![(
                                sink.key,
                                ScheduledChangeKind::Constant {
                                    index: input,
                                    value,
                                },
                                None,
                            )],
                            Time::Immediately,
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
                ref source,
                from_index,
                from_label,
                to_index,
                channels,
            } => {
                // Convert from NodeAddress to RawNodeAddress, returning an error if it isn't possible
                let source = if let Some(raw_source) = source.to_raw() {
                    raw_source
                } else {
                    return Err(ConnectionError::SourceNodeNotPushed);
                };

                // If the source belongs in a different graph, search for the right graph among the sub graphs
                if source.graph_id != self.id {
                    return try_connect_to_graphs(connection);
                }
                if !self.get_nodes().contains_key(source.key) {
                    return Err(ConnectionError::NodeNotFound);
                }
                if self.node_keys_pending_removal.contains(&source.key) {
                    return Err(ConnectionError::NodeNotFound);
                }

                let num_source_outputs = self
                    .node_output_index_to_name
                    .get(source.key)
                    .unwrap()
                    .len();
                // if channels + to_index > self.num_outputs {
                //     return Err(ConnectionError::ChannelOutOfBounds);
                // }
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
                        from_output_index: (from_index + i) % num_source_outputs,
                        to_input_index: (to_index + i) % self.num_outputs,
                    });
                }
            }
            Connection::GraphInput {
                ref sink,
                from_index,
                to_index,
                to_label,
                channels,
            } => {
                // Convert from NodeAddress to RawNodeAddress, returning an error if it isn't possible
                let sink = if let Some(raw_sink) = sink.to_raw() {
                    raw_sink
                } else {
                    return Err(ConnectionError::SinkNodeNotPushed);
                };

                if sink.graph_id != self.id {
                    return try_connect_to_graphs(connection);
                }
                if !self.get_nodes().contains_key(sink.key) {
                    return Err(ConnectionError::NodeNotFound);
                }
                if self.node_keys_pending_removal.contains(&sink.key) {
                    return Err(ConnectionError::NodeNotFound);
                }
                // Find the index number, potentially from the label
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
                    });
                }
            }
            Connection::Clear {
                ref node,
                input_nodes,
                input_constants,
                output_nodes,
                graph_outputs,
                graph_inputs,
                channel,
            } => {
                // Convert from NodeAddress to RawNodeAddress, returning an error if it isn't possible
                let node = if let Some(raw_sink) = node.to_raw() {
                    raw_sink
                } else {
                    return Err(ConnectionError::SinkNodeNotPushed);
                };

                if node.graph_id != self.id {
                    return try_connect_to_graphs(connection);
                }
                if !self.get_nodes().contains_key(node.key) {
                    return Err(ConnectionError::NodeNotFound);
                }
                if self.node_keys_pending_removal.contains(&node.key) {
                    return Err(ConnectionError::NodeNotFound);
                }
                let channel_index = match channel {
                    Some(c) => {
                        let index = match c {
                            connection::NodeChannel::Label(label) => {
                                if let Some(index) = self.input_index_from_label(node.key, label) {
                                    index
                                } else {
                                    return Err(ConnectionError::InvalidInputLabel(label));
                                }
                            }
                            connection::NodeChannel::Index(i) => i,
                        };
                        Some(index)
                    }
                    None => None,
                };
                if input_nodes {
                    let mut nodes_to_free = HashSet::new();

                    if let Some(index) = channel_index {
                        for input_edge in &self.node_input_edges[node.key] {
                            // Check that it is an input edge to the selected input index
                            if input_edge.to_input_index == index {
                                if self.feedback_node_indices.contains(&input_edge.source) {
                                    // The edge is from a feedback node. Remove the corresponding feedback edge.
                                    let feedback_edges =
                                        &mut self.node_feedback_edges[input_edge.source];
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
                                        nodes_to_free.insert(RawNodeAddress {
                                            graph_id,
                                            key: input_edge.source,
                                        });
                                    }
                                }
                            }
                        }
                        let edges = &mut self.node_input_edges[node.key];
                        let mut i = 0;
                        while i > edges.len() {
                            if edges[i].to_input_index == index {
                                edges.remove(i);
                            } else {
                                i += 1;
                            }
                        }
                    } else {
                        for input_edge in &self.node_input_edges[node.key] {
                            if self.feedback_node_indices.contains(&input_edge.source) {
                                // The edge is from a feedback node. Remove the corresponding feedback edge.
                                let feedback_edges =
                                    &mut self.node_feedback_edges[input_edge.source];
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
                                    nodes_to_free.insert(RawNodeAddress {
                                        graph_id,
                                        key: input_edge.source,
                                    });
                                }
                            }
                        }
                        self.node_input_edges[node.key].clear();
                    }
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
                        if let Some(index) = channel_index {
                            let change_kind = ScheduledChangeKind::Constant { index, value: 0.0 };
                            ggc.scheduler.schedule_now(node.key, change_kind);
                        } else {
                            for index in 0..num_node_inputs {
                                let change_kind =
                                    ScheduledChangeKind::Constant { index, value: 0.0 };
                                ggc.scheduler.schedule_now(node.key, change_kind);
                            }
                        }
                    } else {
                        // We are fine to set the constants on the node
                        // directly. In fact we have to because the Scheduler
                        // doesn't exist.
                        if let Some(index) = channel_index {
                            self.get_nodes_mut()[node.key].set_constant(0.0, index);
                        } else {
                            for index in 0..num_node_inputs {
                                self.get_nodes_mut()[node.key].set_constant(0.0, index);
                            }
                        }
                    }
                }
                if output_nodes {
                    for (_key, edges) in &mut self.node_input_edges {
                        let mut i = 0;
                        while i < edges.len() {
                            if edges[i].source == node.key {
                                if let Some(index) = channel_index {
                                    if edges[i].from_output_index == index {
                                        edges.remove(i);
                                    } else {
                                        i += 1;
                                    }
                                } else {
                                    edges.remove(i);
                                }
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
                            if let Some(index) = channel_index {
                                if self.output_edges[i].from_output_index == index {
                                    self.output_edges.remove(i);
                                } else {
                                    i += 1;
                                }
                            } else {
                                self.output_edges.remove(i);
                            }
                        } else {
                            i += 1;
                        }
                    }
                }
            }
        }
        self.recalculation_required = true;
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
    /// Apply a change to the internal [`MusicalTimeMap`] shared between all
    /// [`Graphs`] in a tree. Call this only on the top level Graph.
    ///
    /// # Errors
    /// If the scheduler was not yet created, meaning the Graph is not running.
    pub fn change_musical_time_map(
        &mut self,
        change_fn: impl FnOnce(&mut MusicalTimeMap),
    ) -> Result<(), ScheduleError> {
        if let Some(ggc) = &mut self.graph_gen_communicator {
            ggc.scheduler.change_musical_time_map(change_fn)
        } else {
            Err(ScheduleError::SchedulerNotCreated)
        }
    }
    fn depth_first_search(
        &self,
        visited: &mut HashSet<NodeKey>,
        nodes_to_process: &mut Vec<NodeKey>,
    ) -> Vec<NodeKey> {
        let mut stack = Vec::with_capacity(self.get_nodes().capacity());
        while let Some(node_index) = nodes_to_process.pop() {
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
            for (key, input_edges) in &self.node_input_edges {
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
    /// Returns the block size of the [`Graph`], not corrected for oversampling.
    pub fn block_size(&self) -> usize {
        self.block_size
    }
    /// Return the number of nodes currently held by the Graph, including nodes
    /// that are queued to be freed, but have not yet been freed.
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
            let mut input_buffers = NodeBufferRef::new(
                first_sample,
                num_inputs,
                self.block_size * self.oversampling.as_usize(),
            );
            // Collect inputs into the node's input buffer
            let input_edges = &self.node_input_edges[node_key];
            let graph_input_edges = &self.graph_input_edges[node_key];
            let feedback_input_edges = &self.node_feedback_edges[node_key];

            let mut inputs_to_copy = vec![];
            let mut graph_inputs_to_copy = vec![];

            for input_edge in input_edges {
                let source = &nodes[input_edge.source];
                let mut output_values = source.output_buffers();
                for sample_index in 0..self.block_size * self.oversampling.as_usize() {
                    let from_channel = input_edge.from_output_index;
                    let to_channel = input_edge.to_input_index;
                    inputs_to_copy.push((
                        unsafe { output_values.ptr_to_sample(from_channel, sample_index) },
                        unsafe { input_buffers.ptr_to_sample(to_channel, sample_index) },
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
                for i in 0..self.block_size * self.oversampling.as_usize() {
                    inputs_to_copy.push((
                        unsafe { output_values.ptr_to_sample(feedback_edge.from_output_index, i) },
                        unsafe { input_buffers.ptr_to_sample(feedback_edge.from_output_index, i) },
                    ));
                }
            }
            let node = &nodes[node_key];
            tasks.push(node.to_task(
                node_key,
                inputs_to_copy,
                graph_inputs_to_copy,
                input_buffers,
            ));
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
    /// Only one `GraphGen` can be created from a Graph, since otherwise nodes in
    /// the graph could be run multiple times.
    fn create_graph_gen(
        &mut self,
        parent_graph_block_size: usize,
        parent_graph_sample_rate: Sample,
        parent_graph_oversampling: Oversampling,
    ) -> Result<Box<dyn Gen + Send>, String> {
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
        let scheduler = Scheduler::new();

        let scheduler_buffer_size = self.ring_buffer_size;
        let (scheduled_change_producer, rb_consumer) = RingBuffer::new(scheduler_buffer_size);
        let (clock_update_producer, clock_update_consumer) = RingBuffer::new(10);
        let schedule_receiver =
            ScheduleReceiver::new(rb_consumer, clock_update_consumer, scheduler_buffer_size);

        let graph_gen_communicator = GraphGenCommunicator {
            free_node_queue_consumer,
            scheduler,
            scheduled_change_producer,
            clock_update_producer,
            task_data_to_be_dropped_consumer,
            new_task_data_producer,
            next_change_flag: task_data.applied.clone(),
            timestamp: Arc::new(AtomicU64::new(0)),
        };

        let graph_gen = graph_gen::make_graph_gen(
            self.sample_rate,
            parent_graph_sample_rate,
            task_data,
            self.block_size,
            parent_graph_block_size,
            self.oversampling,
            parent_graph_oversampling,
            self.num_outputs,
            self.num_inputs,
            graph_gen_communicator.timestamp.clone(),
            free_node_queue_producer,
            schedule_receiver,
            self.nodes.clone(),
            task_data_to_be_dropped_producer,
            new_task_data_consumer,
            self.inputs_buffers_ptr.clone(),
        );
        self.graph_gen_communicator = Some(graph_gen_communicator);
        Ok(graph_gen)
    }

    /// Initialise buffers, return parameters to zero etc.
    fn init(&mut self) {
        self.calculate_node_order();
        let block_size = self.block_size;
        let sample_rate = self.sample_rate;
        let oversampling = self.oversampling;
        for (_key, n) in self.get_nodes_mut() {
            n.init(
                block_size * oversampling.as_usize(),
                sample_rate * (oversampling.as_usize() as f32),
            );
        }
        // self.tasks = self.generate_tasks();
        // self.output_tasks = self.generate_output_tasks();
        self.initiated = true;
    }

    /// Applies the latest changes to connections and added nodes in the graph on the audio thread and updates the scheduler.
    pub fn commit_changes(&mut self) {
        if self.graph_gen_communicator.is_some() {
            // We need to run free_old to know if there are nodes to free and hence a recalculation required.
            self.free_old();
            if self.recalculation_required {
                self.calculate_node_order();
                let output_tasks = self.generate_output_tasks().into_boxed_slice();
                let tasks = self.generate_tasks().into_boxed_slice();
                if let Some(ggc) = &mut self.graph_gen_communicator {
                    ggc.send_updated_tasks(tasks, output_tasks);
                }
                self.recalculation_required = false;
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
        self.commit_changes();
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
            let address = RawNodeAddress {
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
                GenState::FreeGraph(_) | GenState::FreeGraphMendConnections(_) => unreachable!(),
                GenState::Continue => unreachable!(),
            }
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

    /// Create a dump of all nodes in the Graph. Currently only useful for
    /// debugging.
    pub fn dump_nodes(&self) -> Vec<NodeDump> {
        let mut dump = Vec::new();
        let nodes = self.get_nodes();
        for key in nodes.keys() {
            if let Some(graph) = self.graphs_per_node.get(key) {
                dump.push(NodeDump::Graph(graph.dump_nodes()));
            } else {
                // unwrap
                dump.push(NodeDump::Node(
                    nodes
                        .get(key)
                        .expect("key from `nodes` should still be valid, but isn't")
                        .name
                        .to_string(),
                ));
            }
        }
        dump
    }
}

#[allow(missing_docs)]
#[derive(Clone, Debug)]
pub enum NodeDump {
    Node(String),
    Graph(Vec<NodeDump>),
}

/// Safety: The `GraphGen` is given access to an Arc<UnsafeCell<SlotMap<NodeKey,
/// Node>>, but won't use it unless the Graph is dropped and it needs to keep
/// the `SlotMap` alive, and the drop it when the `GraphGen` is dropped.
unsafe impl Send for Graph {}

/// The internal representation of a scheduled change to a running graph. This
/// is what gets sent to the `GraphGen`.
#[derive(Clone, Copy, Debug)]
struct ScheduledChange {
    /// timestamp in samples in the current Graph's sample rate
    timestamp: u64,
    key: NodeKey,
    kind: ScheduledChangeKind,
}
#[derive(Clone, Copy, Debug)]
enum ScheduledChangeKind {
    Constant { index: usize, value: Sample },
    Trigger { index: usize },
}
// #[derive(Eq, Copy, Clone)]
// enum AudioThreadTimestamp {
//     Samples(u64),
//     ASAP,
// }
// impl AudioThreadTimestamp {
//     fn to_samples_from_now(&self, sample_counter: u64) -> u64 {
//         match self {
//             AudioThreadTimestamp::Samples(ts) => ts - sample_counter,
//             AudioThreadTimestamp::ASAP => 0,
//         }
//     }
// }
// impl PartialEq for AudioThreadTimestamp {
//     fn eq(&self, other: &Self) -> bool {
//         match (self, other) {
//             (Self::Samples(l0), Self::Samples(r0)) => l0 == r0,
//             _ => core::mem::discriminant(self) == core::mem::discriminant(other),
//         }
//     }
// }
// impl PartialOrd for AudioThreadTimestamp {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         Some(match (self, other) {
//             (Self::ASAP, Self::ASAP) => std::cmp::Ordering::Equal,
//             (Self::ASAP, Self::Samples(_)) => std::cmp::Ordering::Less,
//             (Self::Samples(_), Self::ASAP) => std::cmp::Ordering::More,
//             (Self::Samples(s0), Self::Samples(s1)) => s0.partial_cmp(s1).unwrap(),
//         })
//     }
// }
// impl Ord for AudioThreadTimestamp {
//     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//         self.partial_cmp(other).unwrap()
//     }
// }

type SchedulingQueueItem = (
    Vec<(NodeKey, ScheduledChangeKind, Option<TimeOffset>)>,
    Time,
);

/// The Scheduler handles scheduled changes and communicates parameter changes
/// directly to the audio thread through a ring buffer.
///
/// Schedulers are synced so that all sub Graphs from a parent Graph have the
/// same starting time stamp.
///
/// Before a Graph is running and changes scheduled will be stored in a queue.
enum Scheduler {
    Stopped {
        scheduling_queue: Vec<SchedulingQueueItem>,
    },
    Running {
        /// The starting time of the audio thread graph, relative to which time also
        /// passes for the audio thread. This is the timestamp that is used to
        /// convert wall clock time to number of samples since the audio thread
        /// started.
        start_ts: Instant,
        /// Sample rate including oversampling
        sample_rate: u64,
        /// if the ts of the change is less than this number of samples in the future, send it to the GraphGen
        max_duration_to_send: u64,
        /// Changes waiting to be sent to the GraphGen because they are too far into the future
        scheduling_queue: Vec<ScheduledChange>,
        latency_in_samples: f64,
        musical_time_map: Arc<RwLock<MusicalTimeMap>>,
    },
}

impl Scheduler {
    fn new() -> Self {
        Scheduler::Stopped {
            scheduling_queue: vec![],
        }
    }
    fn start(
        &mut self,
        sample_rate: Sample,
        block_size: usize,
        latency: Duration,
        audio_thread_start_ts: Instant,
        musical_time_map: Arc<RwLock<MusicalTimeMap>>,
    ) {
        match self {
            Scheduler::Stopped {
                ref mut scheduling_queue,
            } => {
                // "Take" the scheduling queue out, replacing it with an empty vec which should be cheap
                let scheduling_queue = mem::take(scheduling_queue);
                // How far into the future messages are sent to the GraphGen.
                // This needs to be at least 2 * block_size since the timestamp
                // this is compared to is loaded atomically from the GraphGen
                // and there might be a race condition if less than 2 blocks of
                // events are sent.
                let max_duration_to_send =
                    ((sample_rate * 0.5) as u64).max((block_size as u64) * 2);
                let mut new_scheduler = Scheduler::Running {
                    start_ts: audio_thread_start_ts,
                    #[allow(clippy::cast_possible_truncation)]
                    sample_rate: sample_rate as u64,
                    max_duration_to_send,
                    scheduling_queue: vec![],
                    latency_in_samples: latency.as_secs_f64() * (sample_rate as f64),
                    musical_time_map,
                };
                for (changes, time) in scheduling_queue {
                    new_scheduler.schedule(changes, time);
                }
                *self = new_scheduler;
            }
            Scheduler::Running { .. } => (),
        }
    }
    fn schedule(
        &mut self,
        changes: Vec<(NodeKey, ScheduledChangeKind, Option<TimeOffset>)>,
        time: Time,
    ) {
        match self {
            Scheduler::Stopped { scheduling_queue } => scheduling_queue.push((changes, time)),
            Scheduler::Running {
                start_ts,
                sample_rate,
                max_duration_to_send: _,
                scheduling_queue,
                latency_in_samples: latency,
                musical_time_map,
            } => {
                let offset_to_frames = |time_offset: Option<TimeOffset>| {
                    if let Some(to) = time_offset {
                        to.to_frames(*sample_rate)
                    } else {
                        0
                    }
                };
                match time {
                    Time::DurationFromNow(duration_from_now) => {
                        let mut timestamp = ((start_ts.elapsed() + duration_from_now).as_secs_f64()
                            * (*sample_rate as f64)
                            + *latency) as u64;
                        for (key, change_kind, time_offset) in changes {
                            let frame_offset = offset_to_frames(time_offset);
                            if frame_offset >= 0 {
                                timestamp = timestamp
                                    .checked_add(frame_offset as u64)
                                    .unwrap_or_else(|| {
                                        eprintln!(
                                            "Used a time offset that made the timestamp overflow: {frame_offset:?}"
                                        );
                                        timestamp
                                    });
                            } else {
                                timestamp = timestamp
                                    .checked_sub((frame_offset * -1) as u64)
                                    .unwrap_or_else(|| {
                                        eprintln!(
                                            "Used a time offset that made the timestamp overflow: {frame_offset:?}"
                                        );
                                        timestamp
                                    });
                            }
                            scheduling_queue.push(ScheduledChange {
                                timestamp,
                                key,
                                kind: change_kind,
                            });
                        }
                    }
                    Time::Superseconds(superseconds) => {
                        let mut timestamp = superseconds.to_samples(*sample_rate);
                        for (key, change_kind, time_offset) in changes {
                            let frame_offset = offset_to_frames(time_offset);
                            if frame_offset >= 0 {
                                timestamp = timestamp
                                    .checked_add(frame_offset as u64)
                                    .unwrap_or_else(|| {
                                        eprintln!(
                                            "Used a time offset that made the timestamp overflow: {frame_offset:?}"
                                        );
                                        timestamp
                                    });
                            } else {
                                timestamp = timestamp
                                    .checked_sub((frame_offset * -1) as u64)
                                    .unwrap_or_else(|| {
                                        eprintln!(
                                            "Used a time offset that made the timestamp overflow: {frame_offset:?}"
                                        );
                                        timestamp
                                    });
                            }
                            scheduling_queue.push(ScheduledChange {
                                timestamp,
                                key,
                                kind: change_kind,
                            });
                        }
                    }
                    Time::Beats(mt) => {
                        // TODO: Remove unwrap, return a Result
                        let mtm = musical_time_map.read().unwrap();
                        let duration_from_start =
                            Duration::from_secs_f64(mtm.musical_time_to_secs_f64(mt));
                        let mut timestamp = (duration_from_start.as_secs_f64()
                            * (*sample_rate as f64)
                            + *latency) as u64;
                        for (key, change_kind, time_offset) in changes {
                            let frame_offset = offset_to_frames(time_offset);
                            if frame_offset >= 0 {
                                timestamp = timestamp
                                    .checked_add(frame_offset as u64)
                                    .unwrap_or_else(|| {
                                        eprintln!(
                                            "Used a time offset that made the timestamp overflow: {frame_offset:?}"
                                        );
                                        timestamp
                                    });
                            } else {
                                timestamp = timestamp
                                    .checked_sub((frame_offset * -1) as u64)
                                    .unwrap_or_else(|| {
                                        eprintln!(
                                            "Used a time offset that made the timestamp overflow: {frame_offset:?}"
                                        );
                                        timestamp
                                    });
                            }
                            scheduling_queue.push(ScheduledChange {
                                timestamp,
                                key,
                                kind: change_kind,
                            });
                        }
                    }
                    Time::Immediately => {
                        for (key, change_kind, time_offset) in changes {
                            let frame_offset = offset_to_frames(time_offset);
                            let mut timestamp: u64 = 0;
                            if frame_offset >= 0 {
                                timestamp = timestamp
                                    .checked_add(frame_offset as u64)
                                    .unwrap_or_else(|| {
                                        eprintln!(
                                            "Used a time offset that made the timestamp overflow: {frame_offset:?}"
                                        );
                                        timestamp
                                    });
                            } else {
                                timestamp = timestamp
                                    .checked_sub((frame_offset * -1) as u64)
                                    .unwrap_or_else(|| {
                                        eprintln!(
                                            "Used a time offset that made the timestamp overflow: {frame_offset:?}"
                                        );
                                        timestamp
                                    });
                            }
                            scheduling_queue.push(ScheduledChange {
                                timestamp,
                                key,
                                kind: change_kind,
                            });
                        }
                    }
                }
            }
        }
    }
    pub fn change_musical_time_map(
        &mut self,
        change_fn: impl FnOnce(&mut MusicalTimeMap),
    ) -> Result<(), ScheduleError> {
        match self {
            Scheduler::Stopped { .. } => Err(ScheduleError::SchedulerNotCreated),
            Scheduler::Running {
                musical_time_map, ..
            } => match musical_time_map.write() {
                Ok(mut mtm) => {
                    change_fn(&mut *mtm);
                    Ok(())
                }
                Err(_) => Err(ScheduleError::MusicalTimeMapCannotBeWrittenTo),
            },
        }
    }
    /// Schedules a change to be applied at the time of calling the function + the latency setting.
    fn schedule_now(&mut self, key: NodeKey, change: ScheduledChangeKind) {
        self.schedule(
            vec![(key, change, None)],
            Time::DurationFromNow(Duration::new(0, 0)),
        )
    }
    fn update(&mut self, timestamp: u64, rb_producer: &mut rtrb::Producer<ScheduledChange>) {
        match self {
            Scheduler::Stopped { .. } => (),
            Scheduler::Running {
                max_duration_to_send,
                scheduling_queue,
                ..
            } => {
                // scheduled updates should always be sorted before they are sent, in case there are several changes to the same thing
                scheduling_queue.sort_unstable_by_key(|s| s.timestamp);

                let mut i = 0;
                while i < scheduling_queue.len() {
                    if timestamp > scheduling_queue[i].timestamp
                        || scheduling_queue[i].timestamp - timestamp < *max_duration_to_send
                    {
                        let change = scheduling_queue.remove(i);
                        if let Err(e) = rb_producer.push(change) {
                            eprintln!("Unable to push scheduled change into RingBuffer: {e}");
                        }
                    } else {
                        i += 1;
                    }
                }
            }
        }
    }
    fn seconds_to_musical_time_superbeats(&self, ts: Superseconds) -> Option<Superbeats> {
        match self {
            Scheduler::Stopped { .. } => None,
            Scheduler::Running {
                musical_time_map, ..
            } => {
                let mtm = musical_time_map.read().unwrap();
                Some(mtm.superseconds_to_superbeats(ts))
            }
        }
    }
}
#[derive(Clone, Debug)]
struct ClockUpdate {
    /// A sample counter in a different graph. This Arc must never deallocate when dropped.
    timestamp: Arc<AtomicU64>,
    /// The sample rate of the graph that the timestamp above comes from. Used
    /// to convert between timestamps in different sample rates.
    clock_sample_rate: f32,
}

struct ScheduleReceiver {
    rb_consumer: rtrb::Consumer<ScheduledChange>,
    schedule_queue: Vec<ScheduledChange>,
    clock_update_consumer: rtrb::Consumer<ClockUpdate>,
}
impl ScheduleReceiver {
    fn new(
        rb_consumer: rtrb::Consumer<ScheduledChange>,
        clock_update_consumer: rtrb::Consumer<ClockUpdate>,
        capacity: usize,
    ) -> Self {
        Self {
            rb_consumer,
            schedule_queue: Vec::with_capacity(capacity),
            clock_update_consumer,
        }
    }
    fn clock_update(&mut self, sample_rate: f32) -> Option<u64> {
        let mut new_timestamp = None;
        while let Ok(clock) = self.clock_update_consumer.pop() {
            let samples = clock.timestamp.load(Ordering::SeqCst);
            if sample_rate == clock.clock_sample_rate {
                new_timestamp = Some(samples);
            } else {
                new_timestamp = Some(
                    (((samples as f64) / (clock.clock_sample_rate as f64)) * (sample_rate as f64))
                        as u64,
                );
            }
        }
        new_timestamp
    }
    /// TODO: Return only a slice of changes that should be applied this block and then remove them all at once.
    fn changes(&mut self) -> &mut Vec<ScheduledChange> {
        let num_new_changes = self.rb_consumer.slots();
        if num_new_changes > 0 {
            // Only try to read so many changes there is room for in the queue
            let changes_to_read =
                num_new_changes.min(self.schedule_queue.capacity() - self.schedule_queue.len());
            if changes_to_read == 0 {
                eprintln!("ScheduleReceiver: Unable to read any changes, queue is full.");
            }
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
    /// For sending clock updates to the audio thread
    clock_update_producer: rtrb::Producer<ClockUpdate>,
    /// The ring buffer for sending scheduled changes to the audio thread
    scheduled_change_producer: rtrb::Producer<ScheduledChange>,
    timestamp: Arc<AtomicU64>,
    /// The next change flag to be attached to a task update. When the changes
    /// in the update have been applied on the audio thread, this flag till be
    /// set to true. Its purpose is to make sure nodes can be safely dropped
    /// because they are guaranteed not to be accessed on the audio thread. This
    /// is done by each node to be deleted having a clone of this flag which
    /// corresponds to the update when that node was removed from the Tasks
    /// list.
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

    fn send_clock_update(&mut self, clock_update: ClockUpdate) {
        self.clock_update_producer.push(clock_update).unwrap();
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
        self.scheduler
            .update(timestamp, &mut self.scheduled_change_producer);
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

/// Move to the target value exponentially by -60db attenuation over the specified time.
/// *Inputs*
/// 0. "value"
/// 1. "time" in seconds
/// *Outputs*
/// 0. "smoothed_value"
#[derive(Default)]
pub struct Lag {
    // Compare with the current value. If there is change, recalculate the mix.
    last_time: Sample,
    current_value: Sample,
    mix: Sample,
    sample_rate: Sample,
}

impl Lag {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self::default()
    }
}
impl Gen for Lag {
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        for i in 0..ctx.block_size() {
            let value = ctx.inputs.read(0, i);
            let time = ctx.inputs.read(1, i);
            if time != self.last_time {
                self.last_time = time;
                let num_samples = (time * self.sample_rate).floor();
                self.mix = 1.0 - (0.001_f32).powf(1.0 / num_samples);
            }
            self.current_value += (value - self.current_value) * self.mix;
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

    fn init(&mut self, _block_size: usize, sample_rate: Sample) {
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
            0 => "smoothed_value",
            _ => "",
        }
    }

    fn name(&self) -> &'static str {
        "Lag"
    }
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
    #[allow(missing_docs)]
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
                self.step = 0.0;
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

    fn init(&mut self, _block_size: usize, sample_rate: Sample) {
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
/// use knyst_core::wavetable::*;
/// use knyst::graph::RunGraph;
/// let graph_settings = GraphSettings {
///     block_size: 64,
///     sample_rate: 44100.,
///     num_outputs: 2,
///     ..Default::default()
/// };
/// let mut graph = Graph::new(graph_settings);
/// let resources = Resources::new(ResourcesSettings::default());
/// let (mut run_graph, _, _) = RunGraph::new(&mut graph, resources, RunGraphSettings::default())?;
/// let mult = graph.push(Mult);
/// // Connecting the node to the graph output
/// graph.connect(mult.to_graph_out())?;
/// // Multiply 5 by 9
/// graph.connect(constant(5.).to(&mult).to_index(0))?;
/// graph.connect(constant(9.).to(&mult).to_index(1))?;
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
        let block_size = ctx.block_size();
        #[allow(unused_mut)]
        let mut in0 = &ctx.inputs.get_channel(0)[..block_size];
        #[allow(unused_mut)]
        let mut in1 = &ctx.inputs.get_channel(1)[..block_size];
        #[allow(unused_mut)]
        let mut out_buf = &mut ctx.outputs.iter_mut().next().unwrap()[..block_size];
        // fallback
        #[cfg(not(feature = "unstable"))]
        {
            for i in 0..block_size {
                out_buf[i] = in0[i] * in1[i];
            }
        }
        #[cfg(feature = "unstable")]
        {
            use std::simd::f32x2;
            let simd_width = 2;
            for _ in 0..block_size / simd_width {
                let s_in0 = f32x2::from_slice(&in0[..simd_width]);
                let s_in1 = f32x2::from_slice(&in1[..simd_width]);
                let product = s_in0 * s_in1;
                product.copy_to_slice(out_buf);
                in0 = &in0[simd_width..];
                in1 = &in1[simd_width..];
                out_buf = &mut out_buf[simd_width..];
            }
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
/// Bus(channels)
///
/// Convenience Gen for collecting many signals to one node address. Inputs will
/// be copied to the corresponding outputs.
pub struct Bus(pub usize);
impl Gen for Bus {
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        let mut out_bufs = ctx.outputs.iter_mut();
        for channel in 0..self.0 {
            let in_buf = ctx.inputs.get_channel(channel);
            let out_buf = out_bufs.next().unwrap();
            out_buf.copy_from_slice(in_buf);
        }
        GenState::Continue
    }

    fn num_inputs(&self) -> usize {
        self.0
    }

    fn num_outputs(&self) -> usize {
        self.0
    }

    fn name(&self) -> &'static str {
        "Bus"
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
///     let resources = Resources::new(ResourcesSettings::default());
///     let graph_settings = GraphSettings {
///         block_size,
///         sample_rate,
///         num_outputs: 2,
///         ..Default::default()
///     };
///     let mut graph: Graph = Graph::new(graph_settings);
///     let pan = graph.push(PanMonoToStereo);
///     // The signal is a constant 1.0
///     graph.connect(constant(1.).to(&pan).to_label("signal"))?;
///     // Pan to the left
///     graph.connect(constant(-1.).to(&pan).to_label("pan"))?;
///     graph.connect(pan.to_graph_out().channels(2))?;
///     graph.commit_changes();
///     graph.update();
///     let (mut run_graph, _, _) = RunGraph::new(&mut graph, resources, RunGraphSettings::default())?;
///     run_graph.process_block();
///     assert!(run_graph.graph_output_buffers().read(0, 0) > 0.9999);
///     assert!(run_graph.graph_output_buffers().read(1, 0) < 0.0001);
///     // Pan to the right
///     graph.connect(constant(1.).to(&pan).to_label("pan"))?;
///     graph.commit_changes();
///     graph.update();
///     run_graph.process_block();
///     assert!(run_graph.graph_output_buffers().read(0, 0) < 0.0001);
///     assert!(run_graph.graph_output_buffers().read(1, 0) > 0.9999);
///     // Pan to center
///     graph.connect(constant(0.).to(&pan).to_label("pan"))?;
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
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        for i in 0..ctx.block_size() {
            let freq = ctx.inputs.read(0, i);
            let amp = ctx.inputs.read(1, i);
            self.update_freq(freq, ctx.sample_rate);
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

#[cfg(test)]
mod tests;