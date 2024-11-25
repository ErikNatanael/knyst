//! [`Graph`] is the audio graph, the core of Knyst. Implement [`Gen`] and you can be a `Node`.
//!
//! To build an audio graph, add generators (anything implementing the [`Gen`]
//! trait) to a graph and add connections between them.
//!
//! ```
//! # use knyst::prelude::*;
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

use crate::gen::{Gen, GenContext, GenState};
#[allow(unused)]
use crate::trig;
use crate::{BlockSize, Sample};
use knyst_macro::impl_gen;
// For using impl_gen inside the knyst crate
use crate as knyst;

#[macro_use]
pub mod connection;
mod graph_gen;
mod node;
pub mod run_graph;
pub use crate::node_buffer::NodeBufferRef;
pub use connection::Connection;
use connection::ConnectionError;
use node::Node;
pub use run_graph::{RunGraph, RunGraphSettings};

use crate::inspection::{EdgeInspection, EdgeSource, GraphInspection, NodeInspection};
use crate::scheduling::MusicalTimeMap;
use crate::time::{Beats, Seconds};
use rtrb::RingBuffer;
use slotmap::{new_key_type, SecondaryMap, SlotMap};

use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet};
use std::mem;
use std::sync::atomic::Ordering;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use self::connection::{ConnectionBundle, NodeChannel, NodeInput, NodeOutput};

use crate::resources::Resources;
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
/// NodeIds need to be unique and hashable and this unique number makes it so.
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
#[derive(Clone, Copy, Debug)]
pub struct NodeId {
    unique_id: u64,
    /// If the graph_id is not set, every graph needs to search for the node id.
    graph_id: GraphId,
    // graph_id: Arc<RwLock<Option<GraphId>>>,
    // node_key: Arc<RwLock<Option<NodeKey>>>,
}

impl PartialEq for NodeId {
    fn eq(&self, other: &Self) -> bool {
        self.unique_id == other.unique_id
    }
}
impl Eq for NodeId {}

impl std::hash::Hash for NodeId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.unique_id.hash(state);
    }
}

impl NodeId {
    /// Create a new [`Self`] not yet connected to a specific node, but with a unique id.
    pub fn new(graph_id: GraphId) -> Self {
        Self {
            unique_id: NEXT_ADDRESS_ID.fetch_add(1, Ordering::Relaxed),
            graph_id,
        }
    }
    // /// Set the graph id of this NodeId. If you don't know the GraphId yet you can leave it unset. Setting it may speed up edits to the node routing, especially if the node is in a deep graph.
    // pub fn set_graph_id(&mut self, id: GraphId) {
    //     self.graph_id = Some(id);
    // }
    /// Retreive [`GraphId`] if one is set.
    pub fn graph_id(&self) -> GraphId {
        self.graph_id
    }
    /// Create a [`NodeOutput`] based on `self` and a specific channel.
    pub fn out(&self, channel: impl Into<connection::NodeChannel>) -> NodeOutput {
        NodeOutput {
            from_node: self.clone(),
            from_channel: channel.into(),
        }
    }
    /// Create a [`NodeOutput`] based on `self` and a specific channel.
    pub fn input(&self, channel: impl Into<connection::NodeChannel>) -> NodeInput {
        NodeInput {
            node: self.clone(),
            channel: channel.into(),
        }
    }
}

impl NodeId {
    /// Create a [`Connection`] from `self` to `sink_node`
    pub fn to(&self, sink_node: NodeId) -> Connection {
        Connection::Node {
            source: self.clone(),
            from_index: Some(0),
            from_label: None,
            sink: sink_node.clone(),
            to_index: None,
            to_label: None,
            channels: 1,
            feedback: false,
            to_index_offset: 0,
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
    pub fn feedback_to(&self, sink_node: NodeId) -> Connection {
        Connection::Node {
            source: self.clone(),
            from_index: Some(0),
            from_label: None,
            sink: sink_node.clone(),
            to_index: None,
            to_label: None,
            channels: 1,
            feedback: true,
            to_index_offset: 0,
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
    pub fn to(sink_node: NodeId) -> Connection {
        Connection::GraphInput {
            from_index: 0,
            sink: sink_node,
            to_index: None,
            to_label: None,
            channels: 1,
            to_index_offset: 0,
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
    pub fn beats(beats: Beats) -> Self {
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
/// use knyst::graph::{NodeChanges, NodeId, SimultaneousChanges};
/// // In reality, use a real NodeAddress that points to a node.
/// let my_node = NodeId::new(0);
/// let node_changes = NodeChanges::new(my_node).set("freq", 442.0).set("amp", 0.5).trigger("reset");
/// // equivalent to:
/// // let node_changes = my_node.change().set("freq", 442.0).set("amp", 0.5).trigger("reset");
/// let mut simultaneous_changes = SimultaneousChanges::now();
/// simultaneous_changes.push(node_changes);
/// ```
pub struct NodeChanges {
    pub(crate) node: NodeId,
    pub(crate) parameters: Vec<(NodeChannel, Change)>,
    pub(crate) offset: Option<TimeOffset>,
}
impl NodeChanges {
    /// New set of changes for a certain node
    pub fn new(node: NodeId) -> Self {
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
#[derive(Clone, Copy, Debug, PartialEq)]
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

impl From<f32> for Change {
    fn from(value: f32) -> Self {
        Self::Constant(value as Sample)
    }
}
impl From<f64> for Change {
    fn from(value: f64) -> Self {
        Self::Constant(value as Sample)
    }
}

/// A parameter (input constant) change to be scheduled on a [`Graph`].
#[derive(Clone, Debug)]
pub struct ParameterChange {
    /// When to apply the parameter change
    pub time: Time,
    /// What node to change
    pub input: NodeInput,
    /// New value of the input constant
    pub value: Change,
}

impl ParameterChange {
    /// Schedule a change at a specific time in [`Beats`]
    pub fn beats(channel: NodeInput, value: impl Into<Change>, beats: Beats) -> Self {
        Self {
            input: channel,
            value: value.into(),
            time: Time::Beats(beats),
        }
    }
    /// Schedule a change at a specific time in [`Seconds`]
    pub fn seconds(channel: NodeInput, value: impl Into<Change>, seconds: Seconds) -> Self {
        Self {
            input: channel,
            value: value.into(),
            time: Time::Seconds(seconds),
        }
    }
    /// Schedule a change at a duration from right now.
    pub fn duration_from_now(
        channel: NodeInput,
        value: impl Into<Change>,
        from_now: Duration,
    ) -> Self {
        Self {
            input: channel,
            value: value.into(),
            time: Time::DurationFromNow(from_now),
        }
    }
    /// Schedule a change as soon as possible.
    pub fn now(channel: NodeInput, value: impl Into<Change>) -> Self {
        Self {
            input: channel,
            value: value.into(),
            time: Time::Immediately,
        }
    }
}

/// Used to specify a time when a parameter change should be applied.
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug)]
pub enum Time {
    Beats(Beats),
    DurationFromNow(Duration),
    Seconds(Seconds),
    Immediately,
}

/// Newtype for using a [`Time`] as an offset.
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug)]
pub enum TimeOffset {
    Frames(i64),
    Seconds(Relative<Seconds>),
}

impl TimeOffset {
    /// Convert self to a time offset in frames/samples
    pub fn to_frames(&self, sample_rate: u64) -> i64 {
        match self {
            TimeOffset::Frames(frame_offset) => *frame_offset,
            TimeOffset::Seconds(relative_supserseconds) => match relative_supserseconds {
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

/// Stores whether an input can be copied or needs to be added. If there's only a single input to a channel that channel can be copied.
#[derive(Clone, Copy, Debug)]
enum CopyOrAdd {
    Copy,
    Add,
}

/// One task to complete, for the node graph
///
/// Safety: Uses raw pointers to nodes and buffers. A node and its buffers may
/// not be touched from the Graph while a Task containing pointers to it is
/// running. This is guaranteed by an atomic generation counter in
/// GraphGen/GraphGenCommunicator to allow the Graph to free nodes once they are
/// no longer used, and by the Arc pointer to the nodes owned by both Graph and
/// GraphGen so that if Graph is dropped, the pointers are still valid.
struct Task {
    /// The node key may be used to send a message to the Graph to free the node in this Task
    node_key: NodeKey,
    input_constants: *mut [Sample],
    /// inputs to copy from the graph inputs (whole buffers) in the form `(from_graph_input_index, to_node_input_index)`
    graph_inputs_to_copy: Vec<(usize, usize)>,
    /// list of tuples of single floats in the form `(from, to)` where the `from` points to an output of a different node and the `to` points to the input buffer.
    // inputs_to_copy: Vec<(*mut Sample, *mut Sample)>,
    /// list of tuples of single floats in the form `(from, to, block_size)` where the `from` points to an output of a different node and the `to` points to the input buffer.
    inputs_to_copy: Vec<(*mut Sample, *mut Sample, usize, CopyOrAdd)>,
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
    #[inline]
    fn init_constants(&mut self) {
        // Copy all constants
        let node_constants = unsafe { &*self.input_constants };
        for (channel, &constant) in node_constants.iter().enumerate() {
            self.input_buffers.fill_channel(constant, channel);
        }
    }
    #[inline]
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
    #[inline]
    fn run(
        &mut self,
        graph_inputs: &NodeBufferRef,
        resources: &mut Resources,
        sample_rate: Sample,
        sample_time_at_block_start: u64,
    ) -> GenState {
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
        // Copy all inputs
        for (from, to, block_size, _copy_or_add) in &self.inputs_to_copy {
            let from_slice = unsafe { std::slice::from_raw_parts(*from, *block_size) };
            let to_slice = unsafe { std::slice::from_raw_parts_mut(*to, *block_size) };

            for (from, to) in from_slice.iter().zip(to_slice.iter_mut()) {
                *to += *from;
            }

            // TODO: This fails when there is one node input to a node and a constant input as
            // well. We need a change to allow only one input per node for this optimisation to work properly.
            // match copy_or_add {
            //     CopyOrAdd::Copy => {
            //         // This is way faster, but requires a single input edge per channel
            //         unsafe { to.copy_from_nonoverlapping(*from, *block_size) };
            //     }
            //     CopyOrAdd::Add => {
            //         let from_slice = unsafe { std::slice::from_raw_parts(*from, *block_size) };
            //         let to_slice = unsafe { std::slice::from_raw_parts_mut(*to, *block_size) };

            //         for (from, to) in from_slice.iter().zip(to_slice.iter_mut()) {
            //             *to += *from;
            //         }
            //     }
            // }
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

struct InputToOutputTask {
    graph_input_index: usize,
    graph_output_index: usize,
}

/// Copy the entire channel from `input_index` from `input_buffers` to the
/// `graph_output_index` channel of the outputs buffer. Used to copy from a node
/// to the output of a Graph.
struct OutputTask {
    input_buffers: NodeBufferRef,
    input_index: usize,
    graph_output_index: usize,
}
impl std::fmt::Debug for OutputTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputTask")
            .field("input_index", &self.input_index)
            .field("graph_output_index", &self.graph_output_index)
            .finish()
    }
}
unsafe impl Send for OutputTask {}

/// Error pushing a new node (Gen or Graph) to a Graph
#[allow(missing_docs)]
#[derive(thiserror::Error, Debug)]
pub enum PushError {
    #[error("The graph was not started and the given start time of a Gen could therefore not be calculated: `{0:?}`.")]
    InvalidStartTimeOnUnstartedGraph(Time),
    #[error("The target graph (`{target_graph}`) was not found. The GenOrGraph that was pushed is returned.")]
    GraphNotFound {
        g: GenOrGraphEnum,
        target_graph: GraphId,
    },
}

/// Error freeing a node in a Graph
#[allow(missing_docs)]
#[derive(thiserror::Error, Debug, PartialEq)]
pub enum FreeError {
    #[error(
        "The graph containing the NodeAdress provided was not found. The node itself may or may not exist."
    )]
    GraphNotFound,
    #[error("The NodeId does not exist. The Node may have been freed already.")]
    NodeNotFound,
    #[error(
        "The node you tried to free has been marked as immortal. Make it mortal before freeing."
    )]
    ImmortalNode,
    #[error("The free action required making a new connection, but the connection failed.")]
    ConnectionError(#[from] Box<connection::ConnectionError>),
}
#[allow(missing_docs)]
#[derive(thiserror::Error, Debug, PartialEq)]
pub enum ScheduleError {
    #[error("Changes for nodes in different graphs were attempted to be scheduled together.`")]
    DifferentGraphs,
    #[error("The graph containing the NodeId provided was not found: `{0:?}`")]
    GraphNotFound(NodeId),
    #[error("The NodeId does not exist. The Node may have been freed already.")]
    NodeNotFound,
    #[error("The input label specified was not registered for the node: `{0}`")]
    InputLabelNotFound(&'static str),
    #[error(
        "No scheduler was created for the Graph so the change cannot be scheduled. This is likely because this Graph was not yet added to another Graph or split into a Node."
    )]
    SchedulerNotCreated,
    #[error("A lock for writing to the MusicalTimeMap cannot be acquired.")]
    MusicalTimeMapCannotBeWrittenTo,
    #[error("Tried to schedule change `{change:?}` to non existing input `{channel:?}` for node `{node_name}`")]
    InputOutOfRange {
        node_name: String,
        channel: NodeChannel,
        change: Change,
    },
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
    fn num_outputs(&self) -> usize;
    fn num_inputs(&self) -> usize;
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

    fn num_outputs(&self) -> usize {
        self.num_outputs()
    }

    fn num_inputs(&self) -> usize {
        self.num_inputs()
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

    fn num_outputs(&self) -> usize {
        self.num_outputs()
    }

    fn num_inputs(&self) -> usize {
        self.num_inputs()
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
    /// Node identifier in a specific Graph. For referring to a Node outside of the context of a Graph, use NodeId instead.
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

impl GraphSettings {
    /// Set the num_inputs to a new value
    pub fn num_inputs(mut self, num_inputs: usize) -> Self {
        self.num_inputs = num_inputs;
        self
    }
    /// Set the num_outputs to a new value
    pub fn num_outputs(mut self, num_outputs: usize) -> Self {
        self.num_outputs = num_outputs;
        self
    }
    /// Set the oversampling to a new value
    pub fn oversampling(mut self, oversampling: Oversampling) -> Self {
        self.oversampling = oversampling;
        self
    }
    /// Set the oversampling to a new value
    pub fn block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }
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
    ptr: *mut [Sample],
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
/// - [`Graph::push`] creates a node from a [`Gen`] or a [`Graph`], returning a [`NodeId`] which is a handle to that node.
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
    name: String,
    /// The nodes are stored here on the heap, functionally Pinned until dropped.
    /// The `Arc` guarantees that the `GraphGen` will keep the allocations alive even if the `Graph` is dropped.
    nodes: Arc<UnsafeCell<SlotMap<NodeKey, Node>>>,
    node_keys_to_free_when_safe: Vec<(NodeKey, Arc<AtomicBool>)>,
    buffers_to_free_when_safe: Vec<Arc<OwnedRawBuffer>>,
    new_inputs_buffers_ptr: bool,
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
    /// Every NodeAddress is generated with a unique ID. This map can convert between NodeKeys and NodeIds.
    node_ids: SecondaryMap<NodeKey, NodeId>,
    /// If a node can be freed or not. A node can be made immortal to avoid accidentally removing it.
    node_mortality: SecondaryMap<NodeKey, bool>,
    node_order: Vec<NodeKey>,
    disconnected_nodes: Vec<NodeKey>,
    feedback_node_indices: Vec<NodeKey>,
    /// If a node is a graph, that graph will be added with the same key here.
    graphs_per_node: SecondaryMap<NodeKey, Graph>,
    /// The outputs of the Graph
    output_edges: Vec<Edge>,
    /// The edges from the graph inputs to nodes, one Vec per node. `source` in the edge is really the sink here.
    graph_input_edges: SecondaryMap<NodeKey, Vec<Edge>>,
    /// Edges going straight from a graph input to a graph output
    graph_input_to_output_edges: Vec<InterGraphEdge>,
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
    /// For storing changes made before the graph is started. When the GraphGen is created, the changes will be scheduled on the scheduler.
    scheduled_changes_queue: Vec<(
        Vec<(NodeKey, ScheduledChangeKind, Option<TimeOffset>)>,
        Time,
    )>,
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
        let graph_input_to_output_edges = Vec::new();
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
            node_ids: SecondaryMap::with_capacity(num_nodes),
            node_mortality: SecondaryMap::with_capacity(num_nodes),
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
            buffers_to_free_when_safe: vec![],
            new_inputs_buffers_ptr: false,
            graph_input_to_output_edges,
            scheduled_changes_queue: vec![],
        }
    }
    /// Create a node that will run this graph. This will fail if a Node or Gen has already been created from the Graph since only one Gen is allowed to exist per Graph.
    ///
    /// Only use this for manually running the main Graph (the Graph containing all other Graphs). For adding a Graph to another Graph, use the push_graph() method.
    fn split_and_create_top_level_node(&mut self, node_id: NodeId) -> Result<Node, String> {
        let block_size = self.block_size();
        // For the top level node, we will set the parent to its own settings, but without oversampling.
        let graph_gen =
            self.create_graph_gen(self.block_size, self.sample_rate, Oversampling::X1)?;
        let mut node = Node::new("graph", graph_gen);
        node.init(block_size, self.sample_rate, node_id);
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
    /// Set the mortality of a node. An immortal node (false) cannot be manually freed and will not be accidentally freed e.g. through [`Graph::free_disconnected_nodes`].
    pub fn set_node_mortality(
        &mut self,
        node_id: NodeId,
        is_mortal: bool,
    ) -> Result<(), ScheduleError> {
        let mut node_might_be_in_this_graph = true;
        if node_id.graph_id() != self.id {
            node_might_be_in_this_graph = false;
        }
        if node_might_be_in_this_graph {
            if let Some(key) = Self::key_from_id(&self.node_ids, node_id) {
                // Does the Node exist?
                if !self.get_nodes_mut().contains_key(key) {
                    return Err(ScheduleError::NodeNotFound);
                }
                self.node_mortality[key] = is_mortal;
            }
        }
        if !node_might_be_in_this_graph {
            // Try to find the graph containing the node by asking all the graphs in this graph to free the node
            for (_key, graph) in &mut self.graphs_per_node {
                match graph.set_node_mortality(node_id, is_mortal) {
                    Ok(_) => {
                        return Ok(());
                    }
                    Err(e) => match e {
                        ScheduleError::GraphNotFound(_) => (),
                        _ => {
                            return Err(e);
                        }
                    },
                }
            }
            return Err(ScheduleError::GraphNotFound(node_id));
        }
        Ok(())
    }
    /// Push something implementing [`Gen`] or a [`Graph`] to self creating a
    /// new node whose address is returned.
    pub fn push(&mut self, to_node: impl Into<GenOrGraphEnum>) -> NodeId {
        self.push_to_graph(to_node, self.id).unwrap()
    }
    /// Push something implementing [`Gen`] or a [`Graph`] to self creating a
    /// new node whose address is returned. The node will start at `start_time`.
    pub fn push_at_time(&mut self, to_node: impl Into<GenOrGraphEnum>, start_time: Time) -> NodeId {
        self.push_to_graph_at_time(to_node, self.id, start_time)
            .unwrap()
    }
    /// Push something implementing [`Gen`] or a [`Graph`] to the graph with the
    /// id provided, creating a new node whose address is returned.
    pub fn push_to_graph(
        &mut self,
        to_node: impl Into<GenOrGraphEnum>,
        graph_id: GraphId,
    ) -> Result<NodeId, PushError> {
        let mut new_node_address = NodeId::new(graph_id);
        self.push_with_existing_address_to_graph(to_node, &mut new_node_address, graph_id)?;
        Ok(new_node_address)
    }
    /// Push something implementing [`Gen`] or a [`Graph`] to the graph with the
    /// id provided, creating a new node whose address is returned.
    pub fn push_to_graph_at_time(
        &mut self,
        to_node: impl Into<GenOrGraphEnum>,
        graph_id: GraphId,
        start_time: Time,
    ) -> Result<NodeId, PushError> {
        let mut new_node_address = NodeId::new(graph_id);
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
        node_address: &mut NodeId,
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
        node_address: &mut NodeId,
        start_time: Time,
    ) {
        self.push_with_existing_address_to_graph_at_time(to_node, node_address, self.id, start_time)
            .unwrap()
    }
    /// Push something implementing [`Gen`] or a [`Graph`] to the graph with the
    /// id provided, storing its address in the NodeAddress provided.
    pub fn push_with_existing_address_to_graph(
        &mut self,
        to_node: impl Into<GenOrGraphEnum>,
        node_address: &mut NodeId,
        graph_id: GraphId,
    ) -> Result<(), PushError> {
        self.push_with_existing_address_to_graph_at_time(
            to_node,
            node_address,
            graph_id,
            Time::Immediately,
        )
    }
    /// Push something implementing [`Gen`] or a [`Graph`] to the graph with the
    /// id provided, storing its address in the NodeAddress provided. The node
    /// will start processing at the `start_time`.
    pub fn push_with_existing_address_to_graph_at_time(
        &mut self,
        to_node: impl Into<GenOrGraphEnum>,
        node_address: &mut NodeId,
        graph_id: GraphId,
        start_time: Time,
    ) -> Result<(), PushError> {
        if graph_id == self.id {
            let (graph, gen) =
                to_node
                    .into()
                    .components(self.block_size, self.sample_rate, self.oversampling);
            let mut start_timestamp = 0;

            let mut scheduler_ts = false;
            if let Some(ggc) = &mut self.graph_gen_communicator {
                if let Some(ts) = ggc.scheduler.time_to_frames_timestamp(start_time) {
                    start_timestamp = ts;
                    scheduler_ts = true;
                }
            }
            if !scheduler_ts {
                match start_time {
                    Time::Beats(_) => {
                        return Err(PushError::InvalidStartTimeOnUnstartedGraph(start_time))
                    }
                    Time::DurationFromNow(_) => {
                        return Err(PushError::InvalidStartTimeOnUnstartedGraph(start_time))
                    }
                    Time::Seconds(s) => {
                        start_timestamp = s.to_samples(
                            self.sample_rate as u64 * self.oversampling.as_usize() as u64,
                        )
                    }
                    Time::Immediately => start_timestamp = 0,
                }
            }
            let mut node = Node::new(gen.name(), gen);
            node.start_at_sample(start_timestamp);
            let node_key = self.push_node(node, node_address);
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
                self.graphs_per_node.insert(node_key, graph);
            }
            node_address.graph_id = self.id;
            Ok(())
        } else {
            // Try to find the graph containing the node by asking all the graphs in this graph to free the node
            let mut to_node = to_node.into();
            for (_key, graph) in &mut self.graphs_per_node {
                match graph.push_with_existing_address_to_graph_at_time(
                    to_node,
                    node_address,
                    graph_id,
                    start_time,
                ) {
                    Ok(_) => {
                        return Ok(());
                    }
                    // Return the error unless it's a GraphNotFound in which case we continue trying
                    Err(e) => match e {
                        PushError::GraphNotFound {
                            g: returned_to_node,
                            target_graph: _graph_id,
                        } => {
                            to_node = returned_to_node;
                        }
                        _ => return Err(e),
                    },
                }
            }
            Err(PushError::GraphNotFound {
                g: to_node,
                target_graph: graph_id,
            })
        }
    }
    /// Increase the maximum number of inputs a node can have. This means reallocating the inputs buffer and marking the old for deletion.
    fn increase_max_node_inputs(&mut self, new_max_node_inputs: usize) {
        let inputs_buffers_ptr = Box::<[Sample]>::into_raw(
            vec![
                0.0 as Sample;
                self.block_size * self.oversampling.as_usize() * new_max_node_inputs
            ]
            .into_boxed_slice(),
        );
        let inputs_buffers_ptr = Arc::new(OwnedRawBuffer {
            ptr: inputs_buffers_ptr,
        });
        let old_input_buffers_ptr = mem::replace(&mut self.inputs_buffers_ptr, inputs_buffers_ptr);
        if self.graph_gen_communicator.is_some() {
            // The GraphGen has been created so we have to be more careful
            self.buffers_to_free_when_safe.push(old_input_buffers_ptr);
            // Send the new inputs buffers to the GraphGen together with the next TaskData
            self.new_inputs_buffers_ptr = true;
        } else {
            // The GraphGen has not been created so there is nothing running on the audio thread we can do things the easy way
            assert_eq!(
                std::sync::Arc::<OwnedRawBuffer>::strong_count(&old_input_buffers_ptr),
                1
            );
            drop(old_input_buffers_ptr);
        }
        self.max_node_inputs = new_max_node_inputs;
    }
    /// Add a node to this Graph. The Node will be (re)initialised with the
    /// correct block size for this Graph.
    ///
    /// Provide a [`NodeId`] so that a pre-created NodeId can be
    /// connected to this node. A new NodeId can also be passed in. Either
    /// way, it will be connected to this node.
    ///
    /// Making it not public means Graphs cannot be accidentally added, but a
    /// Node<Graph> can still be created for the top level one if preferred.
    fn push_node(&mut self, mut node: Node, node_id: &mut NodeId) -> NodeKey {
        if node_id.graph_id() != self.id {
            eprintln!("Warning: Pushing node to NodeId with a GraphId matching a different Graph than the current Graph.")
        }
        if node.num_inputs() > self.max_node_inputs {
            self.increase_max_node_inputs(node.num_inputs());
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
            self.sample_rate * (self.oversampling.as_usize() as Sample),
            *node_id,
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
        self.node_mortality.insert(key, true);

        self.node_ids.insert(key, node_id.clone());
        key
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
            match self.free_node_from_key(node) {
                Ok(_) => (),
                Err(e) => match e {
                    // If a node is immortal we ignore it
                    FreeError::ImmortalNode => (),
                    // TODO: Report error instead and continue freeing the rest of the disconnected nodes
                    _ => return Err(e),
                },
            }
        }
        Ok(())
    }
    /// Get the NodeKey for a NodeId if it exists in this graph.
    fn key_from_id(node_ids: &SecondaryMap<NodeKey, NodeId>, node_id: NodeId) -> Option<NodeKey> {
        node_ids
            .iter()
            .find(|(_key, &id)| id == node_id)
            .map(|(key, _id)| key)
    }
    fn id_from_key(&self, node_key: NodeKey) -> Option<NodeId> {
        self.node_ids.get(node_key).map(|id| *id)
    }
    fn free_node_mend_connections_from_key(&mut self, node_key: NodeKey) -> Result<(), FreeError> {
        // Does the Node exist?
        if !self.get_nodes_mut().contains_key(node_key) {
            return Err(FreeError::NodeNotFound);
        }
        self.recalculation_required = true;

        let num_inputs = self.node_input_index_to_name
                .get(node_key)
                .expect(
                    "Since the key exists in the Graph it should have a corresponding node_input_index_to_name Vec"
                )
                .len();
        let num_outputs = self.node_output_index_to_name
                .get(node_key)
                .expect(
                    "Since the key exists in the Graph it should have a corresponding node_output_index_to_name Vec"
                )
                .len();
        let inputs_to_bridge = num_inputs.min(num_outputs);
        // First collect all the connections that should be bridged so that they are in one place
        let mut outputs = vec![vec![]; inputs_to_bridge];
        for (destination_node_key, edge_vec) in &self.node_input_edges {
            for edge in edge_vec {
                if edge.source == node_key && edge.from_output_index < inputs_to_bridge {
                    outputs[edge.from_output_index].push(Connection::Node {
                        source: self.id_from_key(node_key).unwrap(),
                        from_index: Some(edge.from_output_index),
                        from_label: None,
                        sink: self.id_from_key(destination_node_key).unwrap(),
                        to_index: Some(edge.to_input_index),
                        to_label: None,
                        channels: 1,
                        feedback: false,
                        to_index_offset: 0,
                    });
                }
            }
        }
        for graph_output in &self.output_edges {
            if graph_output.source == node_key && graph_output.from_output_index < inputs_to_bridge
            {
                outputs[graph_output.from_output_index].push(Connection::GraphOutput {
                    source: self.id_from_key(node_key).unwrap(),
                    from_index: Some(graph_output.from_output_index),
                    from_label: None,
                    to_index: graph_output.to_input_index,
                    channels: 1,
                });
            }
        }
        let mut inputs = vec![vec![]; inputs_to_bridge];
        for (inout_index, bridge_input) in inputs.iter_mut().enumerate().take(inputs_to_bridge) {
            for input in self
                .node_input_edges
                .get(node_key)
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
            .get(node_key)
            .expect("Since the key exists in the graph its graph input Vec should also exist")
        {
            if graph_input.to_input_index < inputs_to_bridge {
                graph_inputs[graph_input.to_input_index].push(Connection::GraphInput {
                    sink: self.id_from_key(node_key).unwrap(),
                    from_index: graph_input.from_output_index,
                    to_index: Some(graph_input.to_input_index),
                    to_label: None,
                    channels: 1,
                    to_index_offset: 0,
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
                    self.connect(connection.from(self.id_from_key(input.source).unwrap()))
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
                            .to(output.get_source_node().unwrap())
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
        self.free_node_from_key(node_key)
    }
    /// Remove the node and connect its input edges to the sinks of its output edges
    pub fn free_node_mend_connections(&mut self, node: NodeId) -> Result<(), FreeError> {
        // For every input of the node, connect the nodes connected
        // to it to all the nodes taking input from the corresponding output.
        //
        // E.g. node1output1 -> input1, node2output2 -> input1, output1 ->
        // node3input2, output2 -> node4input3. Connect node1output1 and
        // node2output2 to node3input2. Since input2 has no connections nothing
        // is conencted to node4input3.
        //
        let mut node_might_be_in_graph = true;
        if node.graph_id != self.id {
            node_might_be_in_graph = false;
        }
        if node_might_be_in_graph {
            if let Some((node_key, _id)) = self.node_ids.iter().find(|(_key, &id)| node == id) {
                return self.free_node_mend_connections_from_key(node_key);
            }
        }
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
    fn clear_feedback_for_node(&mut self, node_key: NodeKey) -> Result<(), FreeError> {
        // Remove all feedback edges leading from or to the node
        let mut nodes_to_free = HashSet::new();
        if let Some(&feedback_node) = self.node_feedback_node_key.get(node_key) {
            // The node that is being freed has a feedback node attached to it. Free that as well.
            nodes_to_free.insert(feedback_node);
            self.node_feedback_node_key.remove(node_key);
        }
        for (feedback_key, feedback_edges) in &mut self.node_feedback_edges {
            if !feedback_edges.is_empty() {
                let mut i = 0;
                while i < feedback_edges.len() {
                    if feedback_edges[i].source == node_key
                        || feedback_edges[i].feedback_destination == node_key
                    {
                        feedback_edges.remove(i);
                    } else {
                        i += 1;
                    }
                }
                if feedback_edges.is_empty() {
                    // The feedback node has no more edges to it: free it
                    nodes_to_free.insert(feedback_key);
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
            self.free_node_from_key(na)?;
        }
        Ok(())
    }
    fn free_node_from_key(&mut self, node_key: NodeKey) -> Result<(), FreeError> {
        // Does the Node exist?
        if !self.get_nodes_mut().contains_key(node_key) {
            return Err(FreeError::NodeNotFound);
        }
        if !self.node_mortality[node_key] {
            return Err(FreeError::ImmortalNode);
        }

        self.recalculation_required = true;

        // Remove all edges leading to the node
        self.node_input_edges.remove(node_key);
        self.graph_input_edges.remove(node_key);
        // feedback from the freed node requires removing the feedback node and all edges from the feedback node
        self.node_feedback_edges.remove(node_key);
        // Remove all edges leading from the node to other nodes
        for (_k, input_edges) in &mut self.node_input_edges {
            let mut i = 0;
            while i < input_edges.len() {
                if input_edges[i].source == node_key {
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
                if self.output_edges[i].source == node_key {
                    self.output_edges.remove(i);
                } else {
                    i += 1;
                }
            }
        }
        self.clear_feedback_for_node(node_key)?;
        if let Some(ggc) = &mut self.graph_gen_communicator {
            // The GraphGen has been created so we have to be more careful
            self.node_keys_to_free_when_safe
                .push((node_key, ggc.next_change_flag.clone()));
            self.node_keys_pending_removal.insert(node_key);
        } else {
            // The GraphGen has not been created so we can do things the easy way
            self.graphs_per_node.remove(node_key);
            self.get_nodes_mut().remove(node_key);
        }
        Ok(())
    }
    /// Remove the node and any edges to/from the node. This may lead to other nodes being disconnected from the output and therefore not be run, but they will not be freed.
    pub fn free_node(&mut self, node: NodeId) -> Result<(), FreeError> {
        let mut node_might_be_in_graph = true;
        if node.graph_id() != self.id {
            node_might_be_in_graph = false;
        }
        if node_might_be_in_graph {
            if let Some((node_key, _id)) = self.node_ids.iter().find(|(_key, &id)| node == id) {
                self.free_node_from_key(node_key)?;
            } else {
                node_might_be_in_graph = false;
            }
        }
        if !node_might_be_in_graph {
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
                self.sample_rate * (self.oversampling.as_usize() as Sample),
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
    /// Returns the current audio thread time in Beats based on the
    /// MusicalTimeMap, or None if it is not available (e.g. if the Graph has
    /// not been started yet).
    pub fn get_current_time_musical(&self) -> Option<Beats> {
        if let Some(ggc) = &self.graph_gen_communicator {
            let ts_samples = ggc.timestamp.load(Ordering::Relaxed);
            let seconds = (ts_samples as f64) / (self.sample_rate as f64);
            ggc.scheduler
                .seconds_to_musical_time_beats(Seconds::from_seconds_f64(seconds))
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
                address: self.node_ids
                    .get(node_key)
                    .expect("All nodes should have their ids stored.")
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
            if let Some(index) = node_key_processed.iter().position(|&k| k == node_key) {
                nodes[index].input_edges = input_edges;
            }
        }
        let mut graph_output_input_edges = Vec::new();
        for edge in &self.output_edges {
            if let Some(index) = node_key_processed.iter().position(|&k| k == edge.source) {
                graph_output_input_edges.push(EdgeInspection {
                    source: EdgeSource::Node(index),
                    from_index: edge.from_output_index,
                    to_index: edge.to_input_index,
                });
            }
        }
        let unconnected_nodes = self
            .disconnected_nodes
            .iter()
            .filter_map(|&disconnected_key| {
                node_key_processed
                    .iter()
                    .position(|&key| key == disconnected_key)
            })
            .collect();
        // Note: keys from `node_keys_to_free_when_safe` very rarely cannot be found in the `node_key_processed`. Why?
        let nodes_pending_removal = self
            .node_keys_to_free_when_safe
            .iter()
            .filter_map(|&(freed_key, _)| {
                node_key_processed.iter().position(|&key| key == freed_key)
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
    /// updated. [`NodeChanges`] must all be in the same Graph. If you want to
    /// schedule chagnes to nodes in multiple graphs, separate them and call
    /// [`Graph::schedule_changes`] multiple times.
    pub fn schedule_changes(
        &mut self,
        node_changes: Vec<NodeChanges>,
        time: Time,
    ) -> Result<(), ScheduleError> {
        // assert all changes are for the same graph
        if node_changes.is_empty() {
            return Ok(());
        }
        let first_graph = node_changes[0].node.graph_id();
        for nc in &node_changes {
            if nc.node.graph_id() != first_graph {
                return Err(ScheduleError::DifferentGraphs);
            }
        }
        let mut scheduler_changes = vec![];
        for node_changes in &node_changes {
            let node = node_changes.node;
            let change_pairs = &node_changes.parameters;
            let time_offset = node_changes.offset;
            let mut node_might_be_in_this_graph = true;
            if node.graph_id() != self.id {
                node_might_be_in_this_graph = false;
            }
            if node_might_be_in_this_graph {
                if let Some((key, _id)) = self.node_ids.iter().find(|(_key, &id)| id == node) {
                    // Does the Node exist?
                    if !self.get_nodes_mut().contains_key(key) {
                        // TODO: Report error
                        eprintln!("Scheduled change for a node that no longer exists");
                    }
                    for (channel, change) in change_pairs {
                        let index = match channel {
                            NodeChannel::Label(label) => {
                                if let Some(label_index) =
                                    self.node_input_name_to_index[key].get(label)
                                {
                                    *label_index
                                } else {
                                    return Err(ScheduleError::InputLabelNotFound(label));
                                }
                            }
                            NodeChannel::Index(index) => *index,
                        };
                        if index >= self.node_input_index_to_name[key].len() {
                            return Err(ScheduleError::InputOutOfRange {
                                node_name: self.get_nodes()[key].name.to_string(),
                                channel: channel.clone(),
                                change: change.clone(),
                            });
                        }

                        let change_kind = match change {
                            Change::Constant(value) => ScheduledChangeKind::Constant {
                                index,
                                value: *value,
                            },
                            Change::Trigger => ScheduledChangeKind::Trigger { index },
                        };
                        scheduler_changes.push((key, change_kind, time_offset));
                    }
                } else {
                    node_might_be_in_this_graph = false;
                }
            }
            if !node_might_be_in_this_graph {
                // Try to find the graph containing the node by asking all the graphs in this graph to free the node
                let mut found_graph = false;
                for (_key, graph) in &mut self.graphs_per_node {
                    match graph.schedule_changes(vec![node_changes.clone()], time) {
                        Ok(_) => {
                            found_graph = true;
                            break;
                        }
                        Err(e) => match e {
                            ScheduleError::GraphNotFound { .. } => (),
                            _ => {
                                return Err(e);
                            }
                        },
                    }
                }
                if !found_graph {
                    return Err(ScheduleError::GraphNotFound(node_changes.node));
                }
            }
        }
        if let Some(ggc) = &mut self.graph_gen_communicator {
            ggc.scheduler.schedule(scheduler_changes, time);
        } else {
            self.scheduled_changes_queue.push((scheduler_changes, time));
        }
        Ok(())
    }
    /// Schedule a change to an input channel constant. The change will only be
    /// applied if the [`Graph`] is running and its scheduler is regularly
    /// updated.
    pub fn schedule_change(&mut self, change: ParameterChange) -> Result<(), ScheduleError> {
        let mut node_might_be_in_this_graph = true;
        if change.input.node.graph_id() != self.id {
            node_might_be_in_this_graph = false;
        }
        if node_might_be_in_this_graph {
            if let Some(key) = Self::key_from_id(&self.node_ids, change.input.node) {
                // Does the Node exist?
                if !self.get_nodes_mut().contains_key(key) {
                    return Err(ScheduleError::NodeNotFound);
                }
                let index = match change.input.channel {
                    NodeChannel::Label(label) => {
                        if let Some(label_index) = self.node_input_name_to_index[key].get(label) {
                            *label_index
                        } else {
                            return Err(ScheduleError::InputLabelNotFound(label));
                        }
                    }
                    NodeChannel::Index(i) => i,
                };
                if index >= self.node_input_index_to_name[key].len() {
                    return Err(ScheduleError::InputOutOfRange {
                        node_name: self.get_nodes()[key].name.to_string(),
                        channel: change.input.channel,
                        change: change.value,
                    });
                }
                let change_kind = match change.value {
                    Change::Constant(c) => ScheduledChangeKind::Constant { index, value: c },
                    Change::Trigger => ScheduledChangeKind::Trigger { index },
                };
                if let Some(ggc) = &mut self.graph_gen_communicator {
                    ggc.scheduler
                        .schedule(vec![(key, change_kind, None)], change.time);
                } else {
                    self.scheduled_changes_queue
                        .push((vec![(key, change_kind, None)], change.time));
                }
            }
        }
        if !node_might_be_in_this_graph {
            // Try to find the graph containing the node by asking all the graphs in this graph to free the node
            for (_key, graph) in &mut self.graphs_per_node {
                match graph.schedule_change(change.clone()) {
                    Ok(_) => {
                        return Ok(());
                    }
                    Err(e) => match e {
                        ScheduleError::GraphNotFound(_) => (),
                        _ => {
                            return Err(e);
                        }
                    },
                }
            }
            return Err(ScheduleError::GraphNotFound(change.input.node));
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
                    Err(e) => match e {
                        ConnectionError::NodeNotFound(c) => {
                            // The correct graph was found, but the node wasn't in it.
                            return Err(ConnectionError::NodeNotFound(c));
                        }
                        // We continue trying other graphs
                        ConnectionError::GraphNotFound(_connection) => (),
                        _ => {
                            return Err(e);
                        }
                    },
                }
            }
            Err(ConnectionError::GraphNotFound(connection))
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
                to_index_offset,
            } => {
                if source.graph_id() != sink.graph_id() {
                    return Err(ConnectionError::DifferentGraphs(connection.clone()));
                }
                if source.graph_id() != self.id {
                    return try_disconnect_in_child_graphs(connection.clone());
                }
                let Some(source_key) = Self::key_from_id(&self.node_ids, *source) else {
                    return try_disconnect_in_child_graphs(connection.clone());
                };
                let Some(sink_key) = Self::key_from_id(&self.node_ids, *sink) else {
                    return try_disconnect_in_child_graphs(connection.clone());
                };
                if source_key == sink_key {
                    return Err(ConnectionError::SameNode);
                }
                if !self.get_nodes().contains_key(source_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
                }
                if !self.get_nodes().contains_key(sink_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
                }
                if self.node_keys_pending_removal.contains(&source_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
                }
                if self.node_keys_pending_removal.contains(&sink_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
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
                        if let Some(index) = self.input_index_from_label(sink_key, label) {
                            index
                        } else {
                            return Err(ConnectionError::InvalidInputLabel(label));
                        }
                    } else {
                        0
                    }
                } else {
                    0
                } + to_index_offset;
                let from_index = if from_index.is_some() {
                    if let Some(i) = from_index {
                        i
                    } else {
                        0
                    }
                } else if from_label.is_some() {
                    if let Some(label) = from_label {
                        // unwrap() is okay because the hashmap is generated for every node when it is inserted
                        if let Some(index) = self.output_index_from_label(sink_key, label) {
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
                if channels + from_index
                    > self
                        .node_output_index_to_name
                        .get(source_key)
                        .unwrap()
                        .len()
                {
                    return Err(ConnectionError::SourceChannelOutOfBounds);
                }
                // Alternative way to get the num_inputs without accessing the node
                if channels + to_index > self.node_input_index_to_name.get(sink_key).unwrap().len()
                {
                    return Err(ConnectionError::DestinationChannelOutOfBounds);
                }
                if !feedback {
                    let edge_list = &mut self.node_input_edges[sink_key];
                    let mut i = 0;
                    while i < edge_list.len() {
                        if edge_list[i].source == source_key
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
                } else if let Some(&feedback_node) = self.node_feedback_node_key.get(source_key) {
                    // Remove feedback edges
                    let feedback_edge_list = &mut self.node_feedback_edges[feedback_node];
                    let mut i = 0;
                    while i < feedback_edge_list.len() {
                        if feedback_edge_list[i].source == source_key
                            && feedback_edge_list[i].feedback_destination == sink_key
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

                    let edge_list = &mut self.node_input_edges[sink_key];

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
                        self.free_node_from_key(feedback_node)?;
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
                    if sink.graph_id() != self.id {
                        return try_disconnect_in_child_graphs(connection.clone());
                    }
                    let Some(sink_key) = Self::key_from_id(&self.node_ids, *sink) else {
                        return try_disconnect_in_child_graphs(connection.clone());
                    };
                    if !self.get_nodes().contains_key(sink_key) {
                        return Err(ConnectionError::NodeNotFound(connection.clone()));
                    }
                    if self.node_keys_pending_removal.contains(&sink_key) {
                        return Err(ConnectionError::NodeNotFound(connection.clone()));
                    }

                    let input = if input_index.is_some() {
                        if let Some(i) = input_index {
                            i
                        } else {
                            0
                        }
                    } else if input_label.is_some() {
                        if let Some(label) = input_label {
                            if let Some(index) = self.input_index_from_label(sink_key, label) {
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
                                sink_key,
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
                        self.get_nodes_mut()[sink_key].set_constant(0.0, input);
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
                if source.graph_id != self.id {
                    return try_disconnect_in_child_graphs(connection.clone());
                }
                let Some((source_key, _)) = self.node_ids.iter().find(|(_key, &id)| id == *source)
                else {
                    return try_disconnect_in_child_graphs(connection.clone());
                };
                if !self.get_nodes().contains_key(source_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
                }
                if self.node_keys_pending_removal.contains(&source_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
                }
                if channels + to_index > self.num_outputs {
                    return Err(ConnectionError::DestinationChannelOutOfBounds);
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
                        if let Some(index) = self.output_index_from_label(source_key, label) {
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
                    if edge_list[i].source == source_key
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
                to_index_offset,
            } => {
                if sink.graph_id != self.id {
                    return try_disconnect_in_child_graphs(connection.clone());
                }
                let Some((sink_key, _)) = self.node_ids.iter().find(|(_key, &id)| id == *sink)
                else {
                    return try_disconnect_in_child_graphs(connection.clone());
                };
                if !self.get_nodes().contains_key(sink_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
                }
                if self.node_keys_pending_removal.contains(&sink_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
                }
                let to_index = if to_index.is_some() {
                    if let Some(i) = to_index {
                        i
                    } else {
                        0
                    }
                } else if to_label.is_some() {
                    if let Some(label) = to_label {
                        if let Some(index) = self.input_index_from_label(sink_key, label) {
                            index
                        } else {
                            return Err(ConnectionError::InvalidInputLabel(label));
                        }
                    } else {
                        0
                    }
                } else {
                    0
                } + to_index_offset;
                if channels + to_index > self.num_outputs {
                    return Err(ConnectionError::DestinationChannelOutOfBounds);
                }
                let edge_list = &mut self.graph_input_edges[sink_key];
                let mut i = 0;
                while i < edge_list.len() {
                    if edge_list[i].source == sink_key
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
            Connection::GraphInputToOutput {
                from_input_channel,
                to_output_channel,
                channels,
                graph_id,
            } => {
                if graph_id != self.id {
                    return try_disconnect_in_child_graphs(connection.clone());
                }
                // input/output are confusing here, use from and to for guidance
                for i in (0..self.graph_input_to_output_edges.len()).rev() {
                    let edge = self.graph_input_to_output_edges[i];
                    if edge.from_output_index >= from_input_channel
                        && edge.from_output_index < (from_input_channel + channels)
                        && edge.to_input_index >= to_output_channel
                        && edge.to_input_index < (to_output_channel + channels)
                    {
                        self.graph_input_to_output_edges.remove(i);
                        self.recalculation_required = true;
                    }
                }
            }
            Connection::ClearGraphInputToOutput {
                graph_id,
                from_input_channel,
                to_output_channel,
                channels,
            } => {
                if graph_id != self.id {
                    return try_disconnect_in_child_graphs(connection.clone());
                }
                let from_input_channel = from_input_channel.unwrap_or(0);
                let to_output_channel = to_output_channel.unwrap_or(0);
                let channels = channels.unwrap_or(self.num_inputs.max(self.num_outputs));
                // input/output are confusing here, use from and to for guidance
                for i in (0..self.graph_input_to_output_edges.len()).rev() {
                    let edge = self.graph_input_to_output_edges[i];
                    if edge.from_output_index >= from_input_channel
                        && edge.from_output_index < (from_input_channel + channels)
                        && edge.to_input_index >= to_output_channel
                        && edge.to_input_index < (to_output_channel + channels)
                    {
                        self.graph_input_to_output_edges.remove(i);
                        self.recalculation_required = true;
                    }
                }
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
                    Err(e) => match e {
                        ConnectionError::NodeNotFound(c) => {
                            // The correct graph was found, but the node wasn't in it.
                            return Err(ConnectionError::NodeNotFound(c));
                        }
                        // We continue trying other graphs
                        ConnectionError::GraphNotFound(_connection) => (),
                        _ => {
                            return Err(e);
                        }
                    },
                }
            }
            Err(ConnectionError::GraphNotFound(connection))
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
                to_index_offset,
            } => {
                if source.graph_id() != sink.graph_id() {
                    return Err(ConnectionError::DifferentGraphs(connection.clone()));
                }
                if source.graph_id != self.id {
                    return try_connect_to_graphs(connection.clone());
                }
                let Some((source_key, _)) = self.node_ids.iter().find(|(_key, &id)| id == *source)
                else {
                    return try_connect_to_graphs(connection.clone());
                };
                let Some((sink_key, _)) = self.node_ids.iter().find(|(_key, &id)| id == *sink)
                else {
                    return try_connect_to_graphs(connection.clone());
                };
                if source_key == sink_key {
                    return Err(ConnectionError::SameNode);
                }
                if !self.get_nodes().contains_key(source_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
                }
                if !self.get_nodes().contains_key(sink_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
                }
                if self.node_keys_pending_removal.contains(&source_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
                }
                if self.node_keys_pending_removal.contains(&sink_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
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
                        if let Some(index) = self.input_index_from_label(sink_key, label) {
                            index
                        } else {
                            return Err(ConnectionError::InvalidInputLabel(label));
                        }
                    } else {
                        0
                    }
                } else {
                    0
                } + to_index_offset;
                let from_index = if from_index.is_some() {
                    if let Some(i) = from_index {
                        i
                    } else {
                        0
                    }
                } else if from_label.is_some() {
                    if let Some(label) = from_label {
                        if let Some(index) = self.output_index_from_label(source_key, label) {
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
                    .get(source_key)
                    .unwrap()
                    .len();
                let num_sink_inputs = self.node_input_index_to_name.get(sink_key).unwrap().len();
                if !feedback {
                    let edge_list = &mut self.node_input_edges[sink_key];
                    for i in 0..channels {
                        edge_list.push(Edge {
                            // wrap channels if there are too many
                            from_output_index: (from_index + i) % num_source_outputs,
                            source: source_key,
                            // wrap channels if there are too many
                            to_input_index: (to_index + i) % num_sink_inputs,
                        });
                    }
                } else {
                    // Create a feedback node if there isn't one.
                    let feedback_node_key =
                        if let Some(&index) = self.node_feedback_node_key.get(source_key) {
                            index
                        } else {
                            let feedback_node = FeedbackGen::node(num_source_outputs);
                            let mut feedback_node_address = NodeId::new(self.id);
                            let key = self.push_node(feedback_node, &mut feedback_node_address);
                            self.feedback_node_indices.push(key);
                            self.node_feedback_node_key.insert(source_key, key);
                            key
                        };
                    // Create feedback edges leading to the FeedbackNode from
                    // the source and normal edges leading from the FeedbackNode
                    // to the sink.
                    let edge_list = &mut self.node_input_edges[sink_key];
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
                            source: source_key,
                            to_input_index: (from_index + i) % num_source_outputs,
                            feedback_destination: sink_key,
                        });
                    }
                }

                self.recalculation_required = true;
            }
            Connection::Constant {
                value,
                ref sink,
                to_index: input_index,
                to_label: input_label,
            } => {
                if let Some(sink) = sink {
                    if sink.graph_id != self.id {
                        return try_connect_to_graphs(connection.clone());
                    }
                    let Some((sink_key, _)) = self.node_ids.iter().find(|(_key, &id)| id == *sink)
                    else {
                        return try_connect_to_graphs(connection.clone());
                    };
                    if !self.get_nodes().contains_key(sink_key) {
                        return Err(ConnectionError::NodeNotFound(connection.clone()));
                    }
                    if self.node_keys_pending_removal.contains(&sink_key) {
                        return Err(ConnectionError::NodeNotFound(connection.clone()));
                    }

                    let input = if input_index.is_some() {
                        if let Some(i) = input_index {
                            i
                        } else {
                            0
                        }
                    } else if input_label.is_some() {
                        if let Some(label) = input_label {
                            if let Some(index) = self.input_index_from_label(sink_key, label) {
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
                                sink_key,
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
                        self.get_nodes_mut()[sink_key].set_constant(value, input);
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
                if source.graph_id != self.id {
                    return try_connect_to_graphs(connection.clone());
                }
                let Some((source_key, _)) = self.node_ids.iter().find(|(_key, &id)| id == *source)
                else {
                    return try_connect_to_graphs(connection.clone());
                };
                if !self.get_nodes().contains_key(source_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
                }
                if self.node_keys_pending_removal.contains(&source_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
                }

                let num_source_outputs = self
                    .node_output_index_to_name
                    .get(source_key)
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
                        if let Some(index) = self.output_index_from_label(source_key, label) {
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
                        source: source_key,
                        from_output_index: (from_index + i) % num_source_outputs,
                        to_input_index: (to_index + i) % self.num_outputs,
                    });
                }

                self.recalculation_required = true;
            }
            Connection::GraphInput {
                ref sink,
                from_index,
                to_index,
                to_label,
                channels,
                to_index_offset,
            } => {
                if sink.graph_id != self.id {
                    return try_connect_to_graphs(connection.clone());
                }
                let Some((sink_key, _)) = self.node_ids.iter().find(|(_key, &id)| id == *sink)
                else {
                    return try_connect_to_graphs(connection.clone());
                };
                if !self.get_nodes().contains_key(sink_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
                }
                if self.node_keys_pending_removal.contains(&sink_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
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
                        if let Some(index) = self.input_index_from_label(sink_key, label) {
                            index
                        } else {
                            return Err(ConnectionError::InvalidInputLabel(label));
                        }
                    } else {
                        0
                    }
                } else {
                    0
                } + to_index_offset;
                if channels + from_index > self.num_inputs {
                    return Err(ConnectionError::SourceChannelOutOfBounds);
                }
                if channels + to_index > self.node_input_index_to_name[sink_key].len() {
                    return Err(ConnectionError::DestinationChannelOutOfBounds);
                }
                for i in 0..channels {
                    self.graph_input_edges[sink_key].push(Edge {
                        source: sink_key,
                        from_output_index: from_index + i,
                        to_input_index: to_index + i,
                    });
                }

                self.recalculation_required = true;
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
                if node.graph_id != self.id {
                    return try_connect_to_graphs(connection.clone());
                }
                let Some((node_key, _)) = self.node_ids.iter().find(|(_key, &id)| id == *node)
                else {
                    return try_connect_to_graphs(connection.clone());
                };
                if !self.get_nodes().contains_key(node_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
                }
                if self.node_keys_pending_removal.contains(&node_key) {
                    return Err(ConnectionError::NodeNotFound(connection.clone()));
                }

                self.recalculation_required = true;
                let channel_index = match channel {
                    Some(c) => {
                        let index = match c {
                            connection::NodeChannel::Label(label) => {
                                if let Some(index) = self.input_index_from_label(node_key, label) {
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
                        for input_edge in &self.node_input_edges[node_key] {
                            // Check that it is an input edge to the selected input index
                            if input_edge.to_input_index == index {
                                if self.feedback_node_indices.contains(&input_edge.source) {
                                    // The edge is from a feedback node. Remove the corresponding feedback edge.
                                    let feedback_edges =
                                        &mut self.node_feedback_edges[input_edge.source];
                                    let mut i = 0;
                                    while i < feedback_edges.len() {
                                        if feedback_edges[i].feedback_destination == node_key
                                            && feedback_edges[i].from_output_index
                                                == input_edge.from_output_index
                                        {
                                            feedback_edges.remove(i);
                                        } else {
                                            i += 1;
                                        }
                                    }
                                    if feedback_edges.is_empty() {
                                        nodes_to_free.insert(input_edge.source);
                                    }
                                }
                            }
                        }
                        let edges = &mut self.node_input_edges[node_key];
                        let mut i = 0;
                        while i > edges.len() {
                            if edges[i].to_input_index == index {
                                edges.remove(i);
                            } else {
                                i += 1;
                            }
                        }
                    } else {
                        for input_edge in &self.node_input_edges[node_key] {
                            if self.feedback_node_indices.contains(&input_edge.source) {
                                // The edge is from a feedback node. Remove the corresponding feedback edge.
                                let feedback_edges =
                                    &mut self.node_feedback_edges[input_edge.source];
                                let mut i = 0;
                                while i < feedback_edges.len() {
                                    if feedback_edges[i].feedback_destination == node_key
                                        && feedback_edges[i].from_output_index
                                            == input_edge.from_output_index
                                    {
                                        feedback_edges.remove(i);
                                    } else {
                                        i += 1;
                                    }
                                }
                                if feedback_edges.is_empty() {
                                    nodes_to_free.insert(input_edge.source);
                                }
                            }
                        }
                        self.node_input_edges[node_key].clear();
                    }
                    for na in nodes_to_free {
                        self.free_node_from_key(na)?;
                    }
                }
                if graph_inputs {
                    self.graph_input_edges[node_key].clear();
                }
                if input_constants {
                    // Clear input constants by scheduling them all to be set to 0 now
                    let num_node_inputs = self.get_nodes_mut()[node_key].num_inputs();
                    if let Some(ggc) = &mut self.graph_gen_communicator {
                        // The GraphGen has been created so we have to be more careful
                        if let Some(index) = channel_index {
                            let change_kind = ScheduledChangeKind::Constant { index, value: 0.0 };
                            ggc.scheduler.schedule_now(node_key, change_kind);
                        } else {
                            for index in 0..num_node_inputs {
                                let change_kind =
                                    ScheduledChangeKind::Constant { index, value: 0.0 };
                                ggc.scheduler.schedule_now(node_key, change_kind);
                            }
                        }
                    } else {
                        // We are fine to set the constants on the node
                        // directly. In fact we have to because the Scheduler
                        // doesn't exist.
                        if let Some(index) = channel_index {
                            self.get_nodes_mut()[node_key].set_constant(0.0, index);
                        } else {
                            for index in 0..num_node_inputs {
                                self.get_nodes_mut()[node_key].set_constant(0.0, index);
                            }
                        }
                    }
                }
                if output_nodes {
                    for (_key, edges) in &mut self.node_input_edges {
                        let mut i = 0;
                        while i < edges.len() {
                            if edges[i].source == node_key {
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
                        if self.output_edges[i].source == node_key {
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
            Connection::GraphInputToOutput {
                graph_id,
                from_input_channel,
                to_output_channel,
                channels,
            } => {
                if graph_id != self.id {
                    return try_connect_to_graphs(connection.clone());
                }
                // TODO Check for duplicates
                for i in 0..channels {
                    self.graph_input_to_output_edges.push(InterGraphEdge {
                        from_output_index: from_input_channel + i,
                        to_input_index: to_output_channel + i,
                    })
                }

                self.recalculation_required = true;
            }
            Connection::ClearGraphInputToOutput { .. } => self.disconnect(connection.clone())?,
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
    /// Apply a change to the internal [`MusicalTimeMap`] shared between all
    /// [`Graph`]s in a tree. Call this only on the top level Graph.
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
    /// Goes through all of the nodes that are connected to nodes in `nodes_to_process` and adds them to the list in
    /// reverse depth first order.
    ///
    fn depth_first_search(
        &self,
        visited: &mut HashSet<NodeKey>,
        nodes_to_process: &mut Vec<NodeKey>,
    ) -> Vec<NodeKey> {
        let mut node_order = Vec::with_capacity(self.get_nodes().capacity());
        while !nodes_to_process.is_empty() {
            let node_index = *nodes_to_process.last().unwrap();

            let input_edges = &self.node_input_edges[node_index];
            let mut found_unvisited = false;
            // There is probably room for optimisation here by managing to
            // not iterate the edges multiple times.
            for edge in input_edges {
                if !visited.contains(&edge.source) {
                    nodes_to_process.push(edge.source);
                    visited.insert(edge.source);
                    found_unvisited = true;
                    break;
                }
            }
            if !found_unvisited {
                node_order.push(nodes_to_process.pop().unwrap());
            }
        }
        node_order
    }
    /// Looks for the deepest (furthest away from the graph output) node that is also an output node, i.e.
    /// a node that is both an output node and an input to another node which is eventually connected to
    /// an output is deeper than a node which is only connected to an output.
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
        self.node_order.extend(stack.into_iter());

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
            .extend(feedback_node_order_addition.into_iter());

        // Add all remaining nodes. These are not currently connected to anything.
        let mut remaining_nodes = vec![];
        for (node_key, _node) in self.get_nodes() {
            if !visited.contains(&node_key) && !self.node_keys_pending_removal.contains(&node_key) {
                remaining_nodes.push(node_key);
            }
        }
        self.node_order.extend(remaining_nodes.iter());
        self.disconnected_nodes = remaining_nodes;
        // debug
        // let nodes = self.get_nodes();
        // for (i, n) in self.node_order.iter().enumerate() {
        //     let name = nodes.get(*n).unwrap().name;
        //     println!("{i}: {name}, {n:?}");
        // }
        // dbg!(&self.node_order);
        // dbg!(&self.disconnected_nodes);
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
        let first_sample = self.inputs_buffers_ptr.ptr.cast::<Sample>();
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
            let mut inputs_per_channel = vec![0; num_inputs];
            for input_edge in input_edges {
                inputs_per_channel[input_edge.to_input_index] += 1;
            }
            for input_edge in feedback_input_edges {
                inputs_per_channel[input_edge.to_input_index] += 1;
            }

            let copy_or_add: Vec<_> = inputs_per_channel
                .into_iter()
                .map(|num| {
                    if num > 1 {
                        CopyOrAdd::Add
                    } else {
                        CopyOrAdd::Copy
                    }
                })
                .collect();

            for input_edge in input_edges {
                let source = &nodes[input_edge.source];
                let mut output_values = source.output_buffers();
                let from_channel = input_edge.from_output_index;
                let to_channel = input_edge.to_input_index;
                inputs_to_copy.push((
                    unsafe { output_values.ptr_to_sample(from_channel, 0) },
                    unsafe { input_buffers.ptr_to_sample(to_channel, 0) },
                    self.block_size * self.oversampling.as_usize(),
                    copy_or_add[to_channel],
                ));
            }
            for input_edge in graph_input_edges {
                graph_inputs_to_copy
                    .push((input_edge.from_output_index, input_edge.to_input_index));
            }
            for feedback_edge in feedback_input_edges {
                let source = &nodes[feedback_edge.source];
                let mut output_values = source.output_buffers();
                inputs_to_copy.push((
                    unsafe { output_values.ptr_to_sample(feedback_edge.from_output_index, 0) },
                    unsafe { input_buffers.ptr_to_sample(feedback_edge.from_output_index, 0) },
                    self.block_size * self.oversampling.as_usize(),
                    copy_or_add[feedback_edge.from_output_index],
                ));
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
    fn generate_input_to_output_tasks(&mut self) -> Vec<InputToOutputTask> {
        let mut output_tasks = vec![];
        for output_edge in &self.graph_input_to_output_edges {
            output_tasks.push(InputToOutputTask {
                graph_input_index: output_edge.from_output_index,
                graph_output_index: output_edge.to_input_index,
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
            new_inputs_buffers_ptr: Some(self.inputs_buffers_ptr.clone()),
            input_to_output_tasks: self.generate_input_to_output_tasks().into_boxed_slice(),
        };
        // let task_data = Box::into_raw(Box::new(task_data));
        // let task_data_ptr = Arc::new(AtomicPtr::new(task_data));
        let (free_node_queue_producer, free_node_queue_consumer) =
            RingBuffer::<(NodeKey, GenState)>::new(self.ring_buffer_size);
        let (new_task_data_producer, new_task_data_consumer) =
            RingBuffer::<TaskData>::new(self.ring_buffer_size);
        let (task_data_to_be_dropped_producer, task_data_to_be_dropped_consumer) =
            RingBuffer::<TaskData>::new(self.ring_buffer_size);
        let mut scheduler = Scheduler::new();
        for (schedule_changes, time) in self.scheduled_changes_queue.drain(..) {
            scheduler.schedule(schedule_changes, time);
        }

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
        for (key, n) in unsafe { &mut *self.nodes.get() } {
            let id = self.node_ids[key];
            n.init(
                block_size * oversampling.as_usize(),
                sample_rate * (oversampling.as_usize() as Sample),
                id,
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
                let input_to_output_tasks =
                    self.generate_input_to_output_tasks().into_boxed_slice();
                let tasks = self.generate_tasks().into_boxed_slice();
                if let Some(ggc) = &mut self.graph_gen_communicator {
                    let new_inputs_buffers_ptr = if self.new_inputs_buffers_ptr {
                        Some(self.inputs_buffers_ptr.clone())
                    } else {
                        None
                    };
                    ggc.send_updated_tasks(
                        tasks,
                        output_tasks,
                        input_to_output_tasks,
                        new_inputs_buffers_ptr,
                    );
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
        self.commit_changes();
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
            match state {
                GenState::FreeSelf => {
                    // If the node key cannot be found it was probably freed
                    // already and added multiple times to the queue. We can
                    // just ignore it.
                    self.free_node_from_key(key).ok();
                }
                GenState::FreeSelfMendConnections => {
                    self.free_node_mend_connections_from_key(key).ok();
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
        // Remove old buffers
        if !self.buffers_to_free_when_safe.is_empty() {
            let mut i = self.buffers_to_free_when_safe.len() - 1;
            loop {
                if Arc::<OwnedRawBuffer>::strong_count(&self.buffers_to_free_when_safe[i]) == 1 {
                    self.buffers_to_free_when_safe.remove(i);
                }
                if i == 0 {
                    break;
                }
                i -= 1;
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
    /// If a change is unable to be applied on the audio thread for many
    /// blocks it has to be removed to make space. This counter counts the blocks
    /// since the change first expired.
    removal_countdown: u8,
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
    /// Converts a [`Time`] to a number of frames from the start time of the graph
    fn time_to_frames_timestamp(&mut self, time: Time) -> Option<u64> {
        match self {
            Scheduler::Stopped { .. } => None,
            Scheduler::Running {
                start_ts,
                sample_rate,
                latency_in_samples: latency,
                musical_time_map,
                ..
            } => {
                Some(match time {
                    Time::DurationFromNow(duration_from_now) => {
                        ((start_ts.elapsed() + duration_from_now).as_secs_f64()
                            * (*sample_rate as f64)
                            + *latency) as u64
                    }
                    Time::Seconds(seconds) => seconds.to_samples(*sample_rate),
                    Time::Beats(mt) => {
                        // TODO: Remove unwrap, return a Result
                        let mtm = musical_time_map.read().unwrap();
                        let duration_from_start =
                            Duration::from_secs_f64(mtm.musical_time_to_secs_f64(mt));
                        let timestamp = (duration_from_start.as_secs_f64() * (*sample_rate as f64)
                            + *latency) as u64;
                        timestamp
                    }
                    Time::Immediately => 0,
                })
            }
        }
    }
    fn schedule(
        &mut self,
        changes: Vec<(NodeKey, ScheduledChangeKind, Option<TimeOffset>)>,
        time: Time,
    ) {
        let timestamp = self.time_to_frames_timestamp(time);
        match self {
            Scheduler::Stopped { scheduling_queue } => scheduling_queue.push((changes, time)),
            Scheduler::Running {
                sample_rate,
                max_duration_to_send: _,
                scheduling_queue,
                ..
            } => {
                // timestamp will be Some if the Scheduler is running
                let timestamp = timestamp.unwrap();
                let offset_to_frames = |time_offset: Option<TimeOffset>| {
                    if let Some(to) = time_offset {
                        to.to_frames(*sample_rate)
                    } else {
                        0
                    }
                };
                for (key, change_kind, time_offset) in changes {
                    let frame_offset = offset_to_frames(time_offset);
                    let mut ts = timestamp;
                    if frame_offset >= 0 {
                        ts = ts
                                    .checked_add(frame_offset as u64)
                                    .unwrap_or_else(|| {
                                        eprintln!(
                                            "Used a time offset that made the timestamp overflow: {frame_offset:?}"
                                        );
                                        timestamp
                                    });
                    } else {
                        ts = ts
                                    .checked_sub((frame_offset * -1) as u64)
                                    .unwrap_or_else(|| {
                                        eprintln!(
                                            "Used a time offset that made the timestamp overflow: {frame_offset:?}"
                                        );
                                        timestamp
                                    });
                    }
                    scheduling_queue.push(ScheduledChange {
                        timestamp: ts,
                        key,
                        kind: change_kind,
                        removal_countdown: 0,
                    });
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
    fn seconds_to_musical_time_beats(&self, ts: Seconds) -> Option<Beats> {
        match self {
            Scheduler::Stopped { .. } => None,
            Scheduler::Running {
                musical_time_map, ..
            } => {
                let mtm = musical_time_map.read().unwrap();
                Some(mtm.seconds_to_beats(ts))
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
    clock_sample_rate: Sample,
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
    fn clock_update(&mut self, sample_rate: Sample) -> Option<u64> {
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
    input_to_output_tasks: Box<[InputToOutputTask]>,
    // if the inputs buffers have been replaced, replace the Arc to them in the GraphGen as well. This avoids the scenario of the buffers being dropped if the Graph is dropped, but the GraphGen is still running.
    new_inputs_buffers_ptr: Option<Arc<OwnedRawBuffer>>,
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
    fn send_updated_tasks(
        &mut self,
        tasks: Box<[Task]>,
        output_tasks: Box<[OutputTask]>,
        input_to_output_tasks: Box<[InputToOutputTask]>,
        new_inputs_buffers_ptr: Option<Arc<OwnedRawBuffer>>,
    ) {
        self.free_old();

        let current_change_flag =
            mem::replace(&mut self.next_change_flag, Arc::new(AtomicBool::new(false)));

        let td = TaskData {
            applied: current_change_flag,
            tasks,
            output_tasks,
            new_inputs_buffers_ptr,
            input_to_output_tasks,
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

#[derive(Clone, Debug, Copy)]
struct InterGraphEdge {
    /// the output index on the destination node
    from_output_index: usize,
    /// the input index on the origin node where the input from the node is placed
    to_input_index: usize,
}

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
/// let resources = Resources::new(ResourcesSettings::default());
/// let (mut run_graph, _, _) = RunGraph::new(&mut graph, resources, RunGraphSettings::default())?;
/// let mult = graph.push(Mult);
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
#[impl_gen]
impl Mult {
    #[process]
    fn process(
        #[allow(unused)] &mut self,
        block_size: BlockSize,
        #[allow(unused)] mut value0: &[Sample],
        #[allow(unused)] mut value1: &[Sample],
        #[allow(unused)] mut product: &mut [Sample],
    ) -> GenState {
        // fallback
        #[cfg(not(feature = "unstable"))]
        {
            for i in 0..*block_size {
                product[i] = value0[i] * value1[i];
            }
        }
        #[cfg(feature = "unstable")]
        {
            use std::simd::f32x2;
            let simd_width = 2;
            for _ in 0..(*block_size / simd_width) {
                let s_in0 = f32x2::from_slice(&value0[..simd_width]);
                let s_in1 = f32x2::from_slice(&value1[..simd_width]);
                let prod = s_in0 * s_in1;
                prod.copy_to_slice(product);
                value0 = &value0[simd_width..];
                value1 = &value1[simd_width..];
                product = &mut product[simd_width..];
            }
        }
        GenState::Continue
    }
}

#[cfg(test)]
mod tests;
