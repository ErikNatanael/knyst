use crate::{node_buffer::NodeBufferRef, Resources, Sample};

use super::{CopyOrAdd, Gen, GenContext, GenState, NodeId, NodeKey, Task};

/// Node is a very unsafe struct. Be very careful when changing it.
///
/// - The Gen is not allowed to be replaced. It is functionally Pinned until dropped.
///   It must be kept as a raw pointer to the allocation not to break rust aliasing
///   rules. `UnsafeCell` is not enough since an owned UnsafeCell implies unique access
///   and can give a &mut reference in safe code.
/// - The input_constants buffer mustn't be deallocated until the Node is dropped.
/// - init must not be called after the Node has started being used on the audio thread.
/// - the number of inputs/outputs of the Gen must not change.
///
/// It also must not be dropped while a Task exists with pointers to the buffers
/// of the Node. Graph has a mechanism to ensure this using an atomic generation counter.
pub(super) struct Node {
    pub(super) name: &'static str,
    /// index by input_channel
    input_constants: *mut [Sample],
    /// index by `output_channel * block_size + sample_index`
    output_buffers: *mut [Sample],
    output_buffers_first_ptr: *mut Sample,
    pub block_size: usize,
    num_outputs: usize,
    gen: *mut (dyn Gen + Send),
    start_node_at_sample: u64,
}

unsafe impl Send for Node {}

impl Node {
    pub fn new(name: &'static str, gen: Box<dyn Gen + Send>) -> Self {
        let num_outputs = gen.num_outputs();
        let block_size = 0;
        let output_buffers =
            Box::<[Sample]>::into_raw(vec![0.0; num_outputs * block_size].into_boxed_slice());

        let output_buffers_first_ptr = std::ptr::null_mut();

        Node {
            name,
            input_constants: Box::into_raw(
                vec![0.0 as Sample; gen.num_inputs()].into_boxed_slice(),
            ),
            gen: Box::into_raw(gen),
            output_buffers,
            output_buffers_first_ptr,
            num_outputs,
            block_size,
            start_node_at_sample: 0,
        }
    }
    pub(super) fn start_at_sample(&mut self, sample_time: u64) {
        self.start_node_at_sample = sample_time;
    }
    pub(super) fn to_task(
        &self,
        node_key: NodeKey,
        inputs_to_copy: Vec<(*mut Sample, *mut Sample, usize, CopyOrAdd)>,
        graph_inputs_to_copy: Vec<(usize, usize)>,
        input_buffers: NodeBufferRef,
    ) -> Task {
        Task {
            node_key,
            inputs_to_copy,
            graph_inputs_to_copy,
            input_buffers,
            input_constants: self.input_constants,
            gen: self.gen,
            output_buffers_first_ptr: self.output_buffers_first_ptr,
            block_size: self.block_size,
            num_outputs: self.num_outputs,
            start_node_at_sample: self.start_node_at_sample,
        }
    }
    // pub fn name(&self) -> &'static str {
    //     self.name
    // }
    /// *Allocates memory*
    /// Allocates enough memory for the given block size
    pub fn init(&mut self, block_size: usize, sample_rate: Sample, node_id: NodeId) {
        // Free the previous buffer
        unsafe {
            drop(Box::from_raw(self.output_buffers));
        }
        self.output_buffers =
            Box::<[Sample]>::into_raw(vec![0.0; self.num_outputs * block_size].into_boxed_slice());
        self.output_buffers_first_ptr = if block_size * self.num_outputs > 0 {
            // Get the pointer to the first Sample in the block without limiting its scope or going through a reference
            self.output_buffers.cast::<Sample>()
        } else {
            std::ptr::null_mut()
        };
        self.block_size = block_size;
        unsafe {
            (*self.gen).init(block_size, sample_rate, node_id);
        }
    }
    /// Use the embedded Gen to generate values that are placed in the
    /// output_buffer. The Graph will have already filled the input buffer with
    /// the correct values.
    #[inline]
    pub(super) fn process(
        &mut self,
        input_buffers: &NodeBufferRef,
        sample_rate: Sample,
        resources: &mut Resources,
    ) -> GenState {
        let mut outputs = NodeBufferRef::new(
            self.output_buffers_first_ptr,
            self.num_outputs,
            self.block_size,
        );
        let ctx = GenContext {
            inputs: input_buffers,
            outputs: &mut outputs,
            sample_rate,
        };
        unsafe { (*self.gen).process(ctx, resources) }
    }
    pub fn set_constant(&mut self, value: Sample, input_index: usize) {
        unsafe {
            (*self.input_constants)[input_index] = value;
        }
    }
    pub fn output_buffers(&self) -> NodeBufferRef {
        NodeBufferRef::new(
            self.output_buffers_first_ptr,
            self.num_outputs,
            self.block_size,
        )
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
    pub(super) fn input_desc(&self, input: usize) -> &'static str {
        unsafe { (*self.gen).input_desc(input) }
    }

    pub(super) fn output_desc(&self, output: usize) -> &'static str {
        unsafe { (*self.gen).output_desc(output) }
    }
}
impl Drop for Node {
    fn drop(&mut self) {
        drop(unsafe { Box::from_raw(self.gen) });
        drop(unsafe { Box::from_raw(self.input_constants) });
        drop(unsafe { Box::from_raw(self.output_buffers) });
    }
}
