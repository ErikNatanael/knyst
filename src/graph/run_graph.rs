//! Provides the [`RunGraph`] struct which safely wraps a running [`Graph`] and
//! provides access to its outputs. Used internally by implementations of
//! `AudioBackend`, but it can also be used directly for custom environments or
//! for offline processing.

use std::{
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

use rtrb::RingBuffer;

use crate::{scheduling::MusicalTimeMap, Resources};

use super::{node::Node, Graph, NodeBufferRef, Sample};

/// Wrapper around a [`Graph`] `Node` with convenience methods to run the
/// Graph, either from an audio thread or for non-real time purposes.
pub struct RunGraph {
    graph_node: Node,
    graph_sample_rate: Sample,
    resources: Resources,
    input_buffer_ptr: *mut f32,
    input_buffer_length: usize,
    input_node_buffer_ref: NodeBufferRef,
    output_node_buffer_ref: NodeBufferRef,
    resources_command_receiver: rtrb::Consumer<crate::ResourcesCommand>,
    resources_response_sender: rtrb::Producer<crate::ResourcesResponse>,
}

impl RunGraph {
    /// Prepare the necessary resources for running the graph. This will fail if
    /// the Graph passed in has already been turned into a Node somewhere else,
    /// for example if it has been pushed to a different Graph.
    ///
    /// Returns Self as well as ring buffer channels to apply changes to
    /// Resources.
    pub fn new(
        graph: &mut Graph,
        resources: Resources,
        settings: RunGraphSettings,
    ) -> Result<
        (
            Self,
            rtrb::Producer<crate::ResourcesCommand>,
            rtrb::Consumer<crate::ResourcesResponse>,
        ),
        RunGraphError,
    > {
        match graph.split_and_create_top_level_node() {
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
                let input_node_buffer_ref = NodeBufferRef::new(
                    input_buffer_ptr,
                    graph_node.num_inputs(),
                    graph_node.block_size,
                );
                let graph_sample_rate = graph.sample_rate;
                let musical_time_map = Arc::new(RwLock::new(MusicalTimeMap::new()));
                //                TODO: Start the scheduler of the Graph
                // Safety: The Nodes get initiated and their buffers allocated
                // when the Graph is split and the Node is created from it.
                // Therefore, we can safely store a reference to a buffer in
                // that Node here. We want to store it to be able to return a
                // reference to it instead of an owned value which includes a
                // raw pointer.
                let output_node_buffer_ref = graph_node.output_buffers();

                let scheduler_start_time_stamp = Instant::now();
                graph.start_scheduler(
                    settings.scheduling_latency,
                    scheduler_start_time_stamp,
                    &None,
                    &musical_time_map,
                );
                // Run a first update to make sure any queued changes get sent to the GraphGen
                graph.update();
                // Create ring buffer channels for communicating with Resources
                let (resources_command_sender, resources_command_receiver) = RingBuffer::new(50);
                let (resources_response_sender, resources_response_receiver) = RingBuffer::new(50);
                Ok((
                    Self {
                        graph_node,
                        graph_sample_rate,
                        resources,
                        input_buffer_length,
                        input_buffer_ptr,
                        input_node_buffer_ref,
                        output_node_buffer_ref,
                        resources_command_receiver,
                        resources_response_sender,
                    },
                    resources_command_sender,
                    resources_response_receiver,
                ))
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
    /// Receive and apply any commands to modify the [`Resources`]. Commands are
    /// normally sent by a [`KnystCommands`] via a [`Controller`].
    pub fn run_resources_communication(&mut self, max_commands_to_process: usize) {
        let mut i = 0;
        while let Ok(command) = self.resources_command_receiver.pop() {
            if let Err(e) = self
                .resources_response_sender
                .push(self.resources.apply_command(command))
            {
                eprintln!(
                    "Warning: A ResourcesResponse could not be sent back from the RunGraph. This may lead to dropping on the audio thread. {e}"
                );
            }
            i += 1;
            if i >= max_commands_to_process {
                break;
            }
        }
    }
    /// Run the Graph for one block using the inputs currently stored in the
    /// input buffer. The results can be accessed through the output buffer
    /// through [`RunGraph::graph_output_buffers`].
    pub fn process_block(&mut self) {
        self.graph_node.process(
            &self.input_node_buffer_ref,
            self.graph_sample_rate,
            &mut self.resources,
        );
    }
    /// Return a reference to the buffer holding the output of the [`Graph`].
    /// Channels which have no [`Connection`]/graph edge to them will be 0.0.
    pub fn graph_output_buffers(&self) -> &NodeBufferRef {
        &self.output_node_buffer_ref
    }
    /// Returns the block size of the resulting output and input buffer(s) of
    /// the top level [`Graph`].
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

#[allow(missing_docs)]
#[derive(thiserror::Error, Debug, PartialEq)]
pub enum RunGraphError {
    #[error("Unable to create a node from the Graph: {0}")]
    CouldNodeCreateNode(String),
}

/// Settings for a [`RunGraph`]
pub struct RunGraphSettings {
    /// How much time is added to every *relative* scheduling event to ensure the Change has time to travel to the GraphGen.
    pub scheduling_latency: Duration,
}
impl Default for RunGraphSettings {
    fn default() -> Self {
        Self {
            scheduling_latency: Duration::from_millis(50),
        }
    }
}
