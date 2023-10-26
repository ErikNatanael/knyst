pub mod buffer;
pub mod gen;
pub mod node_buffer;
pub mod resources;
pub mod wavetable;
pub mod xorrng;

use std::ops::{Deref, DerefMut};

pub use resources::Resources;

/// The current sample type used throughout Knyst
pub type Sample = f32;

/// Newtype for a sample rate to identify it in function signatures. Derefs to a `Sample` for easy use on the audio thread.
#[derive(Copy, Clone, Debug)]
pub struct SampleRate(Sample);

impl SampleRate {
    pub fn to_f64(self) -> f64 {
        self.0 as f64
    }
}

impl Deref for SampleRate {
    type Target = Sample;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SampleRate {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<SampleRate> for f64 {
    fn from(value: SampleRate) -> Self {
        value.0 as f64
    }
}

impl From<f32> for SampleRate {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

#[derive(Copy, Clone, Debug)]
/// BlockSize.
///
/// Can be an unorthodox block size value in the event of a partial block at the beginning of a node's existence in the graph.
pub struct BlockSize(usize);

impl From<BlockSize> for usize {
    fn from(value: BlockSize) -> Self {
        value.0
    }
}
impl From<usize> for BlockSize {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl Deref for BlockSize {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BlockSize {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
