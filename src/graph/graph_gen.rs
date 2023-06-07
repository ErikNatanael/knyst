use std::{
    cell::UnsafeCell,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use rtrb::Producer;
use slotmap::SlotMap;

use crate::{filter::hiir::StandardDownsampler2X, Resources};

use super::{
    node::Node, Gen, GenContext, GenState, NodeBufferRef, NodeKey, Oversampling, OwnedRawBuffer,
    Sample, ScheduleReceiver, TaskData,
};

pub(super) fn make_graph_gen(
    sample_rate: f32,
    parent_sample_rate: f32,
    current_task_data: TaskData,
    block_size: usize,
    parent_block_size: usize,
    oversampling: Oversampling,
    parent_oversampling: Oversampling,
    num_outputs: usize,
    num_inputs: usize,
    timestamp: Arc<AtomicU64>,
    free_node_queue_producer: Producer<(NodeKey, GenState)>,
    schedule_receiver: ScheduleReceiver,
    arc_nodes: Arc<UnsafeCell<SlotMap<NodeKey, Node>>>,
    task_data_to_be_dropped_producer: rtrb::Producer<TaskData>,
    new_task_data_consumer: rtrb::Consumer<TaskData>,
    arc_inputs_buffers_ptr: Arc<OwnedRawBuffer>,
) -> Box<dyn Gen + Send> {
    let graph_gen = Box::new(GraphGen {
        sample_rate: sample_rate * oversampling.as_usize() as f32,
        current_task_data,
        block_size: block_size * oversampling.as_usize(),
        num_outputs,
        num_inputs,
        graph_state: GenState::Continue,
        sample_counter: 0,
        timestamp,
        free_node_queue_producer,
        schedule_receiver,
        _arc_nodes: arc_nodes,
        task_data_to_be_dropped_producer,
        new_task_data_consumer,
        _arc_inputs_buffers_ptr: arc_inputs_buffers_ptr,
    });
    // TODO:
    // If the graph is the same as the parent Graph, do no conversion.
    // Otherwise, first convert oversampling to that of the parent if necessary.
    // Then convert sample rate further if necessary.
    // Then convert block size if necessary.
    let graph_gen: Box<dyn Gen + Send + 'static> = if oversampling != parent_oversampling {
        if num_inputs > 0 {
            panic!("Oversampling is currently not implemented for Graphs with inputs. This will be supported in the future.");
        }
        let graph_oversampling_converter = GraphConvertOversampling2XGen::new(
            Node::new("GraphConvertOversamplingGen", graph_gen),
            parent_sample_rate as usize * parent_oversampling.as_usize(),
            sample_rate as usize * oversampling.as_usize(),
            // Use the inner block size here because that conversion is done afterwards if necessary
            block_size * parent_oversampling.as_usize(),
            block_size * oversampling.as_usize(),
        );
        Box::new(graph_oversampling_converter)
    } else {
        graph_gen
    };
    if parent_block_size != block_size {
        if parent_block_size < block_size && num_inputs > 0 {
            panic!("An inner Graph cannot have inputs if it has a larger block size since the inputs will not be sufficiently filled.");
        }
        let graph_block_converter_gen = GraphBlockConverterGen::new(
            Node::new("GraphBlockConverterGen", graph_gen),
            parent_block_size,
            block_size,
        );
        Box::new(graph_block_converter_gen)
    } else {
        graph_gen
    }
}

/// Contains a GraphGen which it will run inside itself, converting block size
/// and sample rate to that of the parent Graph of this Graph.
///
/// Note that if the inner Graph has a larger block size it cannot have any
/// inputs since these inputs would only be half filled.
pub(super) struct GraphBlockConverterGen {
    graph_gen_node: Node,
    // The block size of the
    inner_block_size: usize,
    parent_block_size: usize,
    species: BlockConverterSpecies,
}

enum BlockConverterSpecies {
    InnerSmallOuterLarge {
        /// How many times the inner block size goes into the outer block size
        num_batches: usize,
        /// This buffer is used to hold the inputs to the inner graph so that they
        /// can be passed to the node. It only needs to have as many channels as the
        /// inner node has inputs. It could have been a Vec if not for the interface
        /// for calling Node::process which needs to work with raw pointers in every
        /// other situation.
        inputs_buffers_ptr: Arc<OwnedRawBuffer>,
    },
    InnerLargeOuterSmall {
        /// How many times the outer block size goes into the inner block size
        num_batches: usize,
        /// Keeping track of how many batches we have already taken from the
        /// inner graph and if we need to process another inner block
        batch_counter: usize,
        gen_state: GenState,
    },
}

impl GraphBlockConverterGen {
    pub fn new(graph_gen_node: Node, parent_block_size: usize, inner_block_size: usize) -> Self {
        let species = if parent_block_size > inner_block_size {
            let inputs_buffers_ptr = Box::<[Sample]>::into_raw(
                vec![0.0 as Sample; inner_block_size * graph_gen_node.num_inputs()]
                    .into_boxed_slice(),
            );
            let inputs_buffers_ptr = Arc::new(OwnedRawBuffer {
                ptr: inputs_buffers_ptr,
            });
            BlockConverterSpecies::InnerSmallOuterLarge {
                num_batches: parent_block_size / inner_block_size,
                inputs_buffers_ptr,
            }
        } else {
            BlockConverterSpecies::InnerLargeOuterSmall {
                num_batches: inner_block_size / parent_block_size,
                // Set to max from the start so that one block is immediately processed
                batch_counter: inner_block_size / parent_block_size,
                gen_state: GenState::Continue,
            }
        };
        Self {
            graph_gen_node,
            inner_block_size,
            parent_block_size,
            species,
        }
    }
    pub fn convert_inner_large_outer_small(
        graph_gen_node: &mut Node,
        ctx: GenContext,
        resources: &mut Resources,
        parent_block_size: usize,
        num_batches: usize,
        batch_counter: &mut usize,
        gen_state: &mut GenState,
    ) {
        if *batch_counter == num_batches {
            *gen_state =
                graph_gen_node.process(&NodeBufferRef::null_buffer(), ctx.sample_rate, resources);
            *batch_counter = 0;
        }

        let offset = *batch_counter * parent_block_size;
        let mut node_outputs = graph_gen_node.output_buffers();
        for (output_channel, node_output) in ctx.outputs.iter_mut().zip(node_outputs.iter_mut()) {
            for sample in 0..parent_block_size {
                output_channel[sample] = node_output[sample + offset];
            }
        }
        *batch_counter += 1;
    }
    pub fn convert_inner_small_outer_large(
        &mut self,
        ctx: GenContext,
        resources: &mut Resources,
        num_batches: usize,
        input_buffers: &mut NodeBufferRef,
    ) -> GenState {
        // We need to run our inner graph_gen many times to fill the outer block.

        for batch in 0..num_batches {
            let batch_sample_offset = batch * self.inner_block_size;
            // Copy inputs from input to this graph
            for input_num in 0..input_buffers.channels() {
                for sample in 0..self.inner_block_size {
                    input_buffers.write(
                        ctx.inputs.read(input_num, sample + batch_sample_offset),
                        input_num,
                        sample,
                    );
                }
            }
            // GraphBlockConverterGen does not convert sample rate so just
            // use the sample rate from upstream
            let returned_gen_state =
                self.graph_gen_node
                    .process(&input_buffers, ctx.sample_rate, resources);
            // Copy the outputs of the node to the outputs of this Gen
            for output_num in 0..ctx.outputs.channels() {
                for sample in 0..self.inner_block_size {
                    ctx.outputs.write(
                        self.graph_gen_node
                            .output_buffers()
                            .read(output_num, sample),
                        output_num,
                        sample + batch_sample_offset,
                    );
                }
            }
            match returned_gen_state {
                GenState::Continue => (),
                GenState::FreeSelf
                | GenState::FreeSelfMendConnections
                | GenState::FreeGraph(_)
                | GenState::FreeGraphMendConnections(_) => {
                    // zero the following batches and then break
                    for batch_left in (batch + 1)..num_batches {
                        let batch_sample_offset = batch_left * self.inner_block_size;
                        for output_num in 0..ctx.outputs.channels() {
                            for sample in 0..self.inner_block_size {
                                ctx.outputs
                                    .write(0.0, output_num, sample + batch_sample_offset);
                            }
                        }
                    }
                    return returned_gen_state;
                }
            }
        }
        GenState::Continue
    }
}

impl Gen for GraphBlockConverterGen {
    fn process(&mut self, ctx: GenContext, resources: &mut Resources) -> GenState {
        match &mut self.species {
            BlockConverterSpecies::InnerSmallOuterLarge {
                num_batches,
                inputs_buffers_ptr,
            } => {
                let input_buffers_first_sample = inputs_buffers_ptr.ptr.cast::<f32>();
                let mut input_buffers = NodeBufferRef::new(
                    input_buffers_first_sample,
                    self.graph_gen_node.num_inputs(),
                    self.inner_block_size,
                );
                let num_batches = *num_batches;
                self.convert_inner_small_outer_large(
                    ctx,
                    resources,
                    num_batches,
                    &mut input_buffers,
                )
            }
            BlockConverterSpecies::InnerLargeOuterSmall {
                num_batches,
                batch_counter,
                gen_state,
            } => {
                GraphBlockConverterGen::convert_inner_large_outer_small(
                    &mut self.graph_gen_node,
                    ctx,
                    resources,
                    self.parent_block_size,
                    *num_batches,
                    batch_counter,
                    gen_state,
                );
                *gen_state
            }
        }
    }

    fn num_inputs(&self) -> usize {
        self.graph_gen_node.num_inputs()
    }

    fn num_outputs(&self) -> usize {
        self.graph_gen_node.num_outputs()
    }

    fn init(&mut self, _block_size: usize, sample_rate: Sample) {
        self.graph_gen_node.init(self.inner_block_size, sample_rate);
    }

    fn input_desc(&self, input: usize) -> &'static str {
        self.graph_gen_node.input_desc(input)
    }

    fn output_desc(&self, output: usize) -> &'static str {
        self.graph_gen_node.output_desc(output)
    }

    fn name(&self) -> &'static str {
        self.graph_gen_node.name
    }
}

unsafe impl Send for GraphBlockConverterGen {}

pub(super) struct GraphConvertOversampling2XGen {
    graph_gen_node: Node,
    /// Inner GraphGen sample rate with oversampling applied
    inner_sample_rate: Sample,
    /// Inner block size with oversampling applied
    inner_block_size: usize,
    downsamplers: Vec<StandardDownsampler2X>,
}

impl GraphConvertOversampling2XGen {
    pub(super) fn new(
        graph_gen_node: Node,
        parent_sample_rate: usize,
        inner_sample_rate: usize,
        parent_block_size: usize,
        inner_block_size: usize,
    ) -> Self {
        // let num_input_channels = graph_gen_node.num_inputs();
        let num_output_channels = graph_gen_node.num_outputs();
        let oversampling_rate = if inner_sample_rate > parent_sample_rate {
            assert_eq!(inner_sample_rate % parent_sample_rate, 0);
            assert_eq!(
                parent_block_size * inner_sample_rate / parent_sample_rate,
                inner_block_size
            );
            inner_sample_rate / parent_sample_rate
        } else {
            assert_eq!(parent_sample_rate % inner_sample_rate, 0);
            assert_eq!(
                inner_block_size * parent_sample_rate / inner_sample_rate,
                parent_block_size
            );
            parent_sample_rate / inner_sample_rate
        };
        assert!(oversampling_rate == 2);
        // let inputs_oversampled_buffers = vec![vec![0.0; inner_block_size]; num_input_channels];
        // // let inputs_buffer =
        // //     vec![
        // //         vec![0.0; parent_block_size * inner_sample_rate / parent_sample_rate];
        // //         graph_gen_node.num_inputs()
        // //     ];
        // // let outputs_buffer =
        // //     vec![vec![0.0 as Sample; parent_block_size]; graph_gen_node.num_outputs()];
        // let inner_outputs_conversion =
        //     vec![vec![0.0 as Sample; inner_block_size]; graph_gen_node.num_outputs()];
        // // let inner_outputs_conversion = output_resampler.input_buffer_allocate();
        // let parent_inputs_conversion =
        //     vec![vec![0.0 as Sample; parent_block_size]; graph_gen_node.num_inputs()];
        // // let parent_inputs_conversion = input_resampler.input_buffer_allocate();

        // let parent_inputs_oversampled_buffers_ptr = Box::<[Sample]>::into_raw(
        //     vec![0.0 as Sample; inner_block_size * graph_gen_node.num_inputs()].into_boxed_slice(),
        // );
        // let parent_inputs_oversampled_buffers_ptr = Arc::new(OwnedRawBuffer {
        //     ptr: parent_inputs_oversampled_buffers_ptr,
        // });

        let downsamplers = vec![StandardDownsampler2X::new(); num_output_channels];
        Self {
            graph_gen_node,
            inner_sample_rate: inner_sample_rate as Sample,
            inner_block_size,
            downsamplers,
        }
    }
}
impl Gen for GraphConvertOversampling2XGen {
    fn process(&mut self, ctx: GenContext, resources: &mut Resources) -> GenState {
        // let input_buffers = if self.graph_gen_node.num_inputs() > 0 {
        //     // Convert inputs to the inner sample rate
        //     for i in 0..ctx.inputs.channels() {
        //         let inp = ctx.inputs.get_channel(i);
        //         let conv = &mut self.parent_inputs_conversion[i];
        //         for (&inp, conv) in inp.iter().zip(conv.iter_mut()) {
        //             *conv = inp;
        //         }
        //     }
        //     // TODO: Upsample here without rubato
        //     self.input_resampler.process_into_buffer(
        //         self.parent_inputs_conversion.as_slice(),
        //         &mut self.inputs_oversampled_buffers,
        //         None,
        //     );
        //     // TODO: Can a Vec based output be converted to a NodeBufferRef for free?
        //     // Copy into out NodeBufferRef structure
        //     let input_buffers_first = self.inputs_oversampled_buffers_ptr.ptr.cast::<f32>();
        //     let mut converted_inputs = NodeBufferRef::new(
        //         input_buffers_first,
        //         self.graph_gen_node.num_inputs(),
        //         self.inner_block_size,
        //     );
        //     assert_eq!(ctx.inputs.channels(), self.graph_gen_node.num_inputs());
        //     for (vec_channel, noderef_channel) in self
        //         .inputs_oversampled_buffers
        //         .iter()
        //         .zip(converted_inputs.iter_mut())
        //     {
        //         for (&vec_val, noderef_val) in vec_channel.iter().zip(noderef_channel.iter_mut()) {
        //             *noderef_val = vec_val;
        //         }
        //     }

        //     converted_inputs
        // } else {
        //     NodeBufferRef::null_buffer()
        // };
        let input_buffers = NodeBufferRef::null_buffer();

        // The GraphGen does nothing with the sample rate parameter sent here,
        // replacing it by its own sample rate.
        let gen_state = self.graph_gen_node.process(&input_buffers, 0., resources);
        // dbg!(self.graph_gen_node.output_buffers().get_channel(0));

        // TODO: Remove this step if we can get NodeBufferRef compatible with &[AsRef<[f32]>]
        /*
                for (vec_channel, noderef_channel) in self
                    .inner_outputs_conversion
                    .iter_mut()
                    .zip(self.graph_gen_node.output_buffers().iter())
                {
                    for (vec_val, &noderef_val) in vec_channel.iter_mut().zip(noderef_channel.iter()) {
                        *vec_val = noderef_val;
                    }
                }

                // dbg!(self.graph_gen_node.output_buffers().get_channel(0));
                // dbg!(&self.inner_outputs_conversion);
                // Convert outputs to the parent sample rate
                let _res = self.output_resampler.process_into_buffer(
                    self.inner_outputs_conversion.as_slice(),
                    &mut self.outputs_buffer,
                    None,
                );

                for (vec_channel, noderef_channel) in self.outputs_buffer.iter().zip(ctx.outputs.iter_mut())
                {
                    for (&vec_val, noderef_val) in vec_channel.iter().zip(noderef_channel.iter_mut()) {
                        *noderef_val = vec_val;
                    }
                }
        */
        // TODO: Use an antialiasing filter on the output here before copying
        // for ((from_graph, to_out), filter) in self
        //     .graph_gen_node
        //     .output_buffers()
        //     .iter()
        //     .zip(ctx.outputs.iter_mut())
        //     .zip(self.downsampling_filter.iter_mut())
        // {
        //     assert!(from_graph.len() / self.oversampling_rate == to_out.len());
        //     for i in 0..(self.inner_block_size / self.oversampling_rate) {
        //         to_out[i] = from_graph[i * self.oversampling_rate];
        //     }
        // }
        for ((from_graph, to_out), downsampler) in self
            .graph_gen_node
            .output_buffers()
            .iter()
            .zip(ctx.outputs.iter_mut())
            .zip(self.downsamplers.iter_mut())
        {
            // If we want more than 2x we need to stack downsamplers
            assert!(from_graph.len() / 2 == to_out.len());
            downsampler.process_block(from_graph, to_out);
        }
        // dbg!(ctx.outputs.get_channel(0));
        gen_state
    }

    fn num_inputs(&self) -> usize {
        self.graph_gen_node.num_inputs()
    }

    fn num_outputs(&self) -> usize {
        self.graph_gen_node.num_outputs()
    }

    fn init(&mut self, _block_size: usize, _sample_rate: Sample) {
        self.graph_gen_node
            .init(self.inner_block_size, self.inner_sample_rate);
    }

    fn input_desc(&self, input: usize) -> &'static str {
        self.graph_gen_node.input_desc(input)
    }

    fn output_desc(&self, output: usize) -> &'static str {
        self.graph_gen_node.output_desc(output)
    }

    fn name(&self) -> &'static str {
        self.graph_gen_node.name
    }
}

unsafe impl Send for GraphConvertOversampling2XGen {}

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
pub(super) struct GraphGen {
    // block_size with oversampling applied
    block_size: usize,
    // sample_rate with oversampling applied
    sample_rate: Sample,
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
                // Check if there is a new clock to update to
                if let Some(new_sample_counter) =
                    self.schedule_receiver.clock_update(self.sample_rate)
                {
                    self.sample_counter = new_sample_counter;
                }
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
                                    // TODO: Send this off of the audio thread
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
                    match task.run(ctx.inputs, resources, self.sample_rate, self.sample_counter) {
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

                // Remove changes that have expired for tasks that don't exist
                // (anymore, removed nodes). Otherwise they will accumulate
                // until there is a crash.
                let mut i = 0;
                while i < changes.len() {
                    let change = &changes[i];
                    if change.timestamp < self.sample_counter {
                        changes.remove(i);
                    } else {
                        i += 1;
                    }
                }

                // Set the output of the graph
                // Zero the output buffer.
                ctx.outputs.fill(0.0);
                for output_task in output_tasks.iter() {
                    let input_values = output_task
                        .input_buffers
                        .get_channel(output_task.input_index);
                    // Safety: We always drop the&mut refernce before requesting
                    // another one so we cannot hold mutliple references to the
                    // same channnel.
                    let output =
                        unsafe { ctx.outputs.get_channel_mut(output_task.graph_output_index) };
                    for i in 0..self.block_size {
                        let value = input_values[i];
                        // Since many nodes may write to the same output we
                        // need to add the outputs together. The Node makes no promise as to the content of
                        // the output buffer provided.
                        output[i] += value;
                    }
                }
                if let Some(from_relative_sample_nr) = do_empty_buffer {
                    for channel in ctx.outputs.iter_mut() {
                        for sample in &mut channel[from_relative_sample_nr..] {
                            *sample = 0.0;
                        }
                    }
                }
                if let Some(from_relative_sample_nr) = do_mend_connections {
                    let channels = ctx.inputs.channels().min(ctx.outputs.channels());
                    for channel in 0..channels {
                        let input = ctx.inputs.get_channel(channel);
                        // Safety: We are only holding one &mut to a channel at a time.
                        let output = unsafe { ctx.outputs.get_channel_mut(channel) };
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
                let channels = ctx.inputs.channels().min(ctx.outputs.channels());
                for channel in 0..channels {
                    let input = ctx.inputs.get_channel(channel);
                    // Safety: We are only holding a &mut to one channel at a time.
                    let output = unsafe { ctx.outputs.get_channel_mut(channel) };
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

/// Safety: This impl of Send is required because of the Arc<UnsafeCell<...>> in
/// GraphGen. The _arc_nodes field of GraphGen exists only so that the nodes
/// won't get dropped if the Graph is dropped. The UnsafeCell will never be used
/// to access the data from within GraphGen.
unsafe impl Send for GraphGen {}
