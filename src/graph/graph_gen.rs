use std::{
    cell::UnsafeCell,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use rtrb::Producer;
use slotmap::SlotMap;

use crate::Resources;

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
    let graph_gen = GraphGen {
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
    };
    // TODO:
    // If the graph is the same as the parent Graph, do no conversion.
    // Otherwise, first convert oversampling to that of the parent if necessary.
    // Then convert sample rate further if necessary.
    // Then convert block size if necessary.
    if parent_block_size != block_size {
        if parent_block_size < block_size && num_inputs > 0 {
            panic!("An inner Graph cannot have inputs if it has a larger block size since the inputs will not be sufficiently filled.");
        }
        let graph_block_converter_gen = GraphBlockConverterGen::new(
            Node::new("GraphBlockConverterGen", Box::new(graph_gen)),
            parent_block_size,
            block_size,
        );
        Box::new(graph_block_converter_gen)
    } else {
        Box::new(graph_gen)
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
    // TODO: Input buffer needs buffering because it gets passed to the Node as an NodeBufferRef
    /// This buffer is used to hold the inputs to the inner graph so that they
    /// can be passed to the node. It only needs to have as many channels as the
    /// inner node has inputs. It could have been a Vec if not for the interface
    /// for calling Node::process which needs to work with raw pointers in every
    /// other situation.
    inputs_buffers_ptr: Arc<OwnedRawBuffer>,
}

impl GraphBlockConverterGen {
    pub fn new(graph_gen_node: Node, parent_block_size: usize, inner_block_size: usize) -> Self {
        let inputs_buffers_ptr = Box::<[Sample]>::into_raw(
            vec![0.0 as Sample; inner_block_size * graph_gen_node.num_inputs()].into_boxed_slice(),
        );
        let inputs_buffers_ptr = Arc::new(OwnedRawBuffer {
            ptr: inputs_buffers_ptr,
        });
        Self {
            graph_gen_node,
            inner_block_size,
            parent_block_size,
            inputs_buffers_ptr,
        }
    }
}

impl Gen for GraphBlockConverterGen {
    fn process(&mut self, ctx: GenContext, resources: &mut Resources) -> GenState {
        let mut gen_state = GenState::Continue;
        if self.inner_block_size < self.parent_block_size {
            // We need to run our inner graph_gen many times to fill the outer block.
            let num_batches = self.parent_block_size / self.inner_block_size;
            let input_buffers_first_sample = self.inputs_buffers_ptr.ptr.cast::<f32>();
            let mut input_buffers = NodeBufferRef::new(
                input_buffers_first_sample,
                self.graph_gen_node.num_inputs(),
                self.inner_block_size,
            );

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
                gen_state = returned_gen_state;
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
                                    ctx.outputs.write(
                                        0.0,
                                        output_num,
                                        sample + batch_sample_offset,
                                    );
                                }
                            }
                        }
                        break;
                    }
                }
            }
        } else {
            // Inner graph has a larger block size than outer
            todo!()
        }
        gen_state
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
                    for channel in ctx.outputs.split_mut() {
                        for sample in &mut channel[from_relative_sample_nr..] {
                            *sample = 0.0;
                        }
                    }
                }
                if let Some(from_relative_sample_nr) = do_mend_connections {
                    let num_channels = ctx.inputs.channels().min(ctx.outputs.channels());
                    for channel in 0..num_channels {
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
                let num_channels = ctx.inputs.channels().min(ctx.outputs.channels());
                for channel in 0..num_channels {
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
