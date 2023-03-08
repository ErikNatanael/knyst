use crate::{Resources, Sample};

use super::{Gen, GenContext, GenState, NodeKey, Task};

/// Node is a very unsafe struct. Be very careful when changing it.
///
/// - The Gen is not allowed to be replaced.
/// - The input_constants buffer mustn't be deallocated until the Node is dropped.
/// - init must not be called after the Node has started being used on the audio thread.
/// - the number of inputs/outputs of the Gen must not change.
///
/// It also must not be dropped while a Task exists with pointers to the buffers
/// of the Node. Graph has a mechanism to ensure this.
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
        inputs_to_copy: Vec<(*mut Sample, *mut Sample)>,
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
            (*self.gen).init(block_size, sample_rate);
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
    block_start_offset: usize,
}
impl NodeBufferRef {
    /// Produces a buffer with 0 channels that you cannot read from or write to,
    /// but it may be useful as a first input to a node that has no inputs.
    pub fn null_buffer() -> Self {
        Self {
            buf: std::ptr::null_mut(),
            num_channels: 0,
            block_size: 0,
            block_start_offset: 0,
        }
    }

    pub(crate) fn new(buf: *mut Sample, num_channels: usize, block_size: usize) -> Self {
        Self {
            buf,
            num_channels,
            block_size,
            block_start_offset: 0,
        }
    }
    /// Write a value to a single sample
    #[inline]
    pub fn write(&mut self, value: f32, channel: usize, sample_index: usize) {
        assert!(channel < self.num_channels);
        assert!(sample_index < self.block_size - self.block_start_offset);
        assert!(!self.buf.is_null());
        let buffer_slice = unsafe {
            std::slice::from_raw_parts_mut(self.buf, self.num_channels * self.block_size)
        };
        unsafe {
            *buffer_slice.get_unchecked_mut(
                channel * self.block_size + sample_index + self.block_start_offset,
            ) = value
        };
    }
    /// Decouples the lifetime of the returned slice from self so that slices to
    /// different channels can be returned.
    ///
    /// Safety: We guarantee that the channels don't overlap with other channels. Getting multiple
    /// mutable references to the same channel this way is, however, UB. Use
    /// `split_mut` for an iterator to all channels instead.
    #[inline]
    pub(crate) unsafe fn get_channel_mut<'a, 'b>(&'a mut self, channel: usize) -> &'b mut [f32] {
        assert!(channel < self.num_channels);
        assert!(!self.buf.is_null());
        let channel_offset = channel * self.block_size;
        let buffer_channel_slice = unsafe {
            let ptr_at_channel = self.buf.add(channel_offset + self.block_start_offset);
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
            let ptr_at_channel = self.buf.add(channel_offset + self.block_start_offset);
            std::slice::from_raw_parts(ptr_at_channel, self.block_size - self.block_start_offset)
        };
        buffer_channel_slice
    }
    /// Adds the given value to the selected sample
    #[inline]
    pub fn add(&mut self, value: f32, channel: usize, sample_index: usize) {
        assert!(channel < self.num_channels);
        assert!(sample_index < self.block_size - self.block_start_offset);
        assert!(!self.buf.is_null());
        unsafe {
            let buffer_slice =
                std::slice::from_raw_parts_mut(self.buf, self.num_channels * self.block_size);
            let sample = (*buffer_slice).get_unchecked_mut(
                channel * self.block_size + sample_index + self.block_start_offset,
            );
            *sample += value
        };
    }
    /// Read one sample from a channel
    #[inline]
    pub fn read(&self, channel: usize, sample_index: usize) -> Sample {
        assert!(channel < self.num_channels);
        assert!(sample_index < self.block_size - self.block_start_offset);
        assert!(!self.buf.is_null());
        unsafe {
            let buffer_slice =
                std::slice::from_raw_parts_mut(self.buf, self.num_channels * self.block_size);
            *(*buffer_slice)
                .get_unchecked(channel * self.block_size + sample_index + self.block_start_offset)
        }
    }
    /// Fill a channel with one and the same value
    #[inline]
    pub fn fill_channel(&mut self, value: f32, channel: usize) {
        assert!(channel < self.num_channels);
        assert!(!self.buf.is_null());
        let channel_offset = channel * self.block_size;
        let buffer_channel_slice = unsafe {
            let ptr_at_channel = self.buf.add(channel_offset + self.block_start_offset);
            std::slice::from_raw_parts_mut(
                ptr_at_channel,
                self.block_size - self.block_start_offset,
            )
        };
        for sample in buffer_channel_slice {
            *sample = value;
        }
    }
    /// Fill all channels in the buffer with one and the same value
    #[inline]
    pub fn fill(&mut self, value: f32) {
        assert!(!self.buf.is_null());
        let buffer_slice = unsafe {
            std::slice::from_raw_parts_mut(self.buf, self.num_channels * self.block_size)
        };
        for sample in buffer_slice {
            *sample = value;
        }
    }
    pub(crate) unsafe fn ptr_to_sample(
        &mut self,
        channel: usize,
        sample_index: usize,
    ) -> *mut Sample {
        assert!(channel < self.num_channels);
        assert!(sample_index < self.block_size);
        unsafe {
            self.buf
                .add(channel * self.block_size + sample_index + self.block_start_offset)
        }
    }
    /// returns the number of channels
    pub fn channels(&self) -> usize {
        self.num_channels
    }
    /// returns the block size for the buffer
    pub fn block_size(&self) -> usize {
        self.block_size - self.block_start_offset
    }
    /// Returns an iterator to the channels in this NodeBufferRef
    pub fn split_mut(&mut self) -> NodeBufferRefSplitMut {
        NodeBufferRefSplitMut {
            buf: self.buf,
            num_channels: self.num_channels,
            block_size: self.block_size,
            block_start_offset: self.block_start_offset,
            _phantom_data: std::marker::PhantomData,
        }
    }
    /// Create a version of self where the block is offset from the start. This
    /// is useful for starting a node some way into a block.
    ///
    /// Safety: You mustn't use the original NodeBufferRef before this one is dropped.
    pub(crate) unsafe fn to_partial_block_size(&mut self, new_block_size: usize) -> Self {
        Self {
            buf: self.buf,
            num_channels: self.num_channels,
            block_size: self.block_size,
            block_start_offset: self.block_size - new_block_size,
        }
    }
}
/// A variant of a NodeBufferRef that can yield &mut[Sample] to all the channels.
pub struct NodeBufferRefSplitMut<'a> {
    buf: *mut Sample,
    num_channels: usize,
    block_size: usize,
    block_start_offset: usize,
    _phantom_data: std::marker::PhantomData<&'a mut ()>,
}
impl<'a> Iterator for NodeBufferRefSplitMut<'a> {
    type Item = &'a mut [Sample];

    fn next(&mut self) -> Option<Self::Item> {
        if self.num_channels == 0 {
            return None;
        }
        assert!(!self.buf.is_null());
        let buffer_channel_slice = unsafe {
            let channel_start_ptr = self.buf.add(self.block_start_offset);
            std::slice::from_raw_parts_mut(
                channel_start_ptr,
                self.block_size - self.block_start_offset,
            )
        };
        unsafe {
            self.buf = self.buf.add(self.block_size);
        }
        self.num_channels -= 1;
        Some(buffer_channel_slice)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.num_channels, Some(self.num_channels))
    }
}
