//! Knyst uses its own wrapper around a buffer allocation: [`NodeBufferRef`]. Buffers are only 
//! dropped from a non-audio thread once no pointers to the allocation persist using an
//! application specific atomic flag mechanism.
//! 
//! You should avoid manually interacting with the NodeBufferRef if you can, e.g. by
//! using the [`impl_gen`] macro instead of a manual [`Gen`] implementation.
//! 
#[allow(unused)]
use crate::gen::Gen;
#[allow(unused)]
use knyst_macro::impl_gen;
use crate::Sample;

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
    pub fn write(&mut self, value: Sample, channel: usize, sample_index: usize) {
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
    /// # Safety
    /// We guarantee that the channels don't overlap with other channels. Getting multiple
    /// mutable references to the same channel this way is, however, UB. Use
    /// `split_mut` for an iterator to all channels instead.
    #[inline]
    pub unsafe fn get_channel_mut<'a, 'b>(&'a mut self, channel: usize) -> &'b mut [Sample] {
        assert!(channel < self.num_channels);
        assert!(!self.buf.is_null());
        let channel_offset = channel * self.block_size;
        let buffer_channel_slice = unsafe {
            let ptr_at_channel = self.buf.add(channel_offset + self.block_start_offset);
            std::slice::from_raw_parts_mut(ptr_at_channel, self.block_size)
        };
        buffer_channel_slice
    }
    /// Returns a single channel of data.
    ///
    /// # Safety
    /// This is safe because the slices are guaranteed not to overlap
    /// and multiple immutable references are allowed.
    #[inline]
    pub fn get_channel(&self, channel: usize) -> &[Sample] {
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
    pub fn add(&mut self, value: Sample, channel: usize, sample_index: usize) {
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
    pub fn fill_channel(&mut self, value: Sample, channel: usize) {
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
    pub fn fill(&mut self, value: Sample) {
        assert!(!self.buf.is_null());
        let buffer_slice = unsafe {
            std::slice::from_raw_parts_mut(self.buf, self.num_channels * self.block_size)
        };
        for sample in buffer_slice {
            *sample = value;
        }
    }
    /// Create a mutable pointer to a specific sample in the buffer.
    ///
    /// # Safety
    /// - You mustn't access, modify or drop `self` until the last use of any pointers created using this function.
    pub unsafe fn ptr_to_sample(&mut self, channel: usize, sample_index: usize) -> *mut Sample {
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
    pub fn iter(&mut self) -> NodeBufferRefIter {
        NodeBufferRefIter {
            buf: self.buf,
            num_channels: self.num_channels,
            block_size: self.block_size,
            block_start_offset: self.block_start_offset,
            _phantom_data: std::marker::PhantomData,
        }
    }
    /// Returns an iterator to the channels in this NodeBufferRef
    pub fn iter_mut(&mut self) -> NodeBufferRefIterMut {
        NodeBufferRefIterMut {
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
    /// # Safety
    /// You mustn't use the original NodeBufferRef before dropping the one returned from this method.
    pub unsafe fn to_partial_block_size(&mut self, new_block_size: usize) -> Self {
        Self {
            buf: self.buf,
            num_channels: self.num_channels,
            block_size: self.block_size,
            block_start_offset: self.block_size - new_block_size,
        }
    }
}
/// A variant of a NodeBufferRef that can yield &mut[Sample] to all the channels.
pub struct NodeBufferRefIterMut<'a> {
    buf: *mut Sample,
    num_channels: usize,
    block_size: usize,
    block_start_offset: usize,
    _phantom_data: std::marker::PhantomData<&'a mut ()>,
}
impl<'a> Iterator for NodeBufferRefIterMut<'a> {
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

/// A variant of a NodeBufferRef that can yield &[Sample] per channel.
pub struct NodeBufferRefIter<'a> {
    buf: *mut Sample,
    num_channels: usize,
    block_size: usize,
    block_start_offset: usize,
    _phantom_data: std::marker::PhantomData<&'a mut ()>,
}
impl<'a> Iterator for NodeBufferRefIter<'a> {
    type Item = &'a [Sample];

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
