//! # Delay
//! This module contains some basic delay Gens

use crate as knyst;
use crate::gen::GenState;
use crate::BlockSize;
use crate::SampleRate;
use knyst_macro::impl_gen;

use crate::time::Seconds;
use crate::Sample;

/// Delay by an integer number of samples, no interpolation. This is good for e.g. triggers.
///
/// *inputs*
/// 0. "signal": input signal, the signal to be delayed
/// 1. "delay_time": the delay time in seconds (will be truncated to the nearest sample)
/// *outputs*
/// 0. "signal": the delayed signal
pub struct SampleDelay {
    buffer: Vec<Sample>,
    write_position: usize,
    max_delay_length: Seconds,
}
impl SampleDelay {}

#[impl_gen]
impl SampleDelay {
    #[new]
    /// Create a new SampleDelay with a maximum delay time.
    pub fn new(max_delay_length: Seconds) -> Self {
        Self {
            buffer: vec![0.0; 0],
            max_delay_length,
            write_position: 0,
        }
    }
    #[process]
    fn process(
        &mut self,
        signal: &[Sample],
        delay_time: &[Sample],
        output: &mut [Sample],
        sample_rate: SampleRate,
    ) -> GenState {
        let sig_buf = signal;
        let time_buf = delay_time;
        let out_buf = output;
        for ((&input, &time), o) in sig_buf.iter().zip(time_buf).zip(out_buf.iter_mut()) {
            self.buffer[self.write_position] = input;
            let delay_samples = (time * *sample_rate) as usize;
            *o = self.buffer
                [(self.write_position + self.buffer.len() - delay_samples) % self.buffer.len()];
            self.write_position = (self.write_position + 1) % self.buffer.len();
        }
        GenState::Continue
    }

    #[init]
    fn init(&mut self, sample_rate: SampleRate) {
        self.buffer =
            vec![0.0; (self.max_delay_length.to_seconds_f64() * sample_rate.to_f64()) as usize];
        self.write_position = 0;
    }
}

/// Schroeder (?) allpass interpolation
#[derive(Clone, Copy, Debug)]
pub struct AllpassInterpolator {
    coeff: f64,
    prev_input: f64,
    prev_output: f64,
}

impl AllpassInterpolator {
    /// Create a new and reset AllpassInterpolator
    pub fn new() -> Self {
        Self {
            coeff: 0.0,
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }
    /// Reset any state to 0
    pub fn clear(&mut self) {
        self.prev_input = 0.0;
        self.prev_output = 0.0;
    }
    /// Set the fractional number of frames in the delay time that we want to interpolate over
    pub fn set_delta(&mut self, delta: f64) {
        self.coeff = (1.0 - delta) / (1.0 + delta);
    }
    /// Interpolate between the input sample and the previous value using the last set delta.
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let output = self.coeff * (input - self.prev_output) + self.prev_input;
        self.prev_output = output;
        self.prev_input = input;
        output
    }
}

/// Delay with allpass interpolation between adjacent samples, no feedback.
#[derive(Clone, Debug)]
pub struct AllpassDelay {
    buffer: Vec<f64>,
    buffer_size: usize,
    frame: usize,
    num_frames: usize,
    allpass: AllpassInterpolator,
}

impl AllpassDelay {
    #[allow(missing_docs)]
    #[must_use]
    pub fn new(buffer_size: usize) -> Self {
        let buffer = vec![0.0; buffer_size];
        Self {
            buffer,
            buffer_size,
            frame: 0,
            num_frames: 1,
            allpass: AllpassInterpolator::new(),
        }
    }
    /// Read the current frame from the delay and allpass interpolate. Read before `write_and_advance` for the correct sample.
    pub fn read(&mut self) -> f64 {
        let index = self.frame % self.buffer.len();
        self.allpass.process_sample(self.buffer[index])
    }
    /// Set the delay time in number of frames/samples
    pub fn set_delay_in_frames(&mut self, num_frames: f64) {
        self.num_frames = num_frames.floor() as usize;
        self.allpass.set_delta(num_frames - self.num_frames as f64);
    }
    /// Clear the internal sample buffer
    pub fn clear(&mut self) {
        for sample in &mut self.buffer {
            *sample = 0.0;
        }
        self.allpass.clear();
    }
    /// Reset the delay with a new length in frames
    pub fn set_delay_in_frames_and_clear(&mut self, num_frames: f64) {
        for sample in &mut self.buffer {
            *sample = 0.0;
        }
        self.set_delay_in_frames(num_frames);
        // println!(
        //     "num_frames: {}, delta: {}",
        //     self.num_frames,
        //     (num_frames - self.num_frames as f64)
        // );
    }
    /// Write a new value into the delay after incrementing the sample pointer.
    pub fn write_and_advance(&mut self, input: f64) {
        self.frame += 1;
        let index = (self.frame + self.num_frames) % self.buffer_size;
        self.buffer[index] = input;
    }
}

/// A Schroeder allpass delay with feedback. Wraps the `AllpassDelay`
#[derive(Clone, Debug)]
pub struct AllpassFeedbackDelay {
    /// The feedback value of the delay. Can be set directly.
    pub feedback: f64,
    previous_delay_time: Sample,
    allpass_delay: AllpassDelay,
}
#[impl_gen]
impl AllpassFeedbackDelay {
    #[allow(missing_docs)]
    #[must_use]
    pub fn new(max_delay_samples: usize) -> Self {
        let allpass_delay = AllpassDelay::new(max_delay_samples);
        let s = Self {
            feedback: 0.,
            allpass_delay,
            previous_delay_time: max_delay_samples as Sample,
        };
        s
    }
    /// Set the delay length counted in frames/samples
    pub fn set_delay_in_frames(&mut self, delay_length: f64) {
        self.allpass_delay.set_delay_in_frames(delay_length);
    }
    /// Clear any values in the delay
    pub fn clear(&mut self) {
        self.allpass_delay.clear();
    }
    // fn calculate_values(&mut self) {
    //     self.feedback = (0.001 as Sample).powf(self.delay_time / self.decay_time.abs())
    //         * self.decay_time.signum();
    //     let delay_samples = self.delay_time * self.sample_rate;
    //     self.allpass_delay.set_num_frames(delay_samples as f64);
    // }
    /// Process on sample
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let delayed_sig = self.allpass_delay.read();
        let delay_write = delayed_sig * self.feedback + input;
        self.allpass_delay.write_and_advance(delay_write);

        delayed_sig - self.feedback * delay_write
    }
    /// Process a block. This will still run sample by sample inside.
    pub fn process(
        &mut self,
        input: &[Sample],
        feedback: &[Sample],
        delay_time: &[Sample],
        output: &mut [Sample],
        sample_rate: SampleRate,
    ) -> GenState {
        for (((&input, &feedback), &delay_time), output) in input
            .iter()
            .zip(feedback)
            .zip(delay_time)
            .zip(output.iter_mut())
        {
            if delay_time != self.previous_delay_time {
                self.set_delay_in_frames(delay_time as f64 * sample_rate.to_f64());
                self.previous_delay_time = delay_time;
            }
            self.feedback = f64::from(feedback);
            *output = self.process_sample(f64::from(input)) as Sample;
        }
        GenState::Continue
    }
}

/// A sample delay with a static number of samples of delay
///
/// # Examples
/// ```
/// # use knyst::prelude::*;
/// use knyst::gen::delay::StaticSampleDelay;
/// let mut delay = StaticSampleDelay::new(4);
/// assert_eq!(delay.read(), 0.0);
/// delay.write_and_advance(1.0);
/// assert_eq!(delay.read(), 0.0);
/// delay.write_and_advance(2.0);
/// assert_eq!(delay.read(), 0.0);
/// delay.write_and_advance(3.0);
/// assert_eq!(delay.read(), 0.0);
/// delay.write_and_advance(4.0);
/// assert_eq!(delay.read(), 1.0);
/// delay.write_and_advance(0.0);
/// assert_eq!(delay.read(), 2.0);
/// delay.write_and_advance(0.0);
/// assert_eq!(delay.read(), 3.0);
/// delay.write_and_advance(0.0);
/// assert_eq!(delay.read(), 4.0);
/// delay.write_and_advance(0.0);
/// assert_eq!(delay.read(), 0.0);
/// delay.write_and_advance(0.0);
/// delay.write_block_and_advance(&[1.0, 2.0]);
/// let mut block = [0.0; 2];
/// delay.read_block(&mut block);
/// delay.write_block_and_advance(&[3.0, 4.0]);
/// delay.read_block(&mut block);
/// assert_eq!(block, [1.0, 2.0]);
/// delay.write_block_and_advance(&[5.0, 6.0]);
/// delay.read_block(&mut block);
/// assert_eq!(block, [3.0, 4.0]);
/// delay.write_block_and_advance(&[0.0, 0.0]);
/// delay.read_block(&mut block);
/// assert_eq!(block, [5.0, 6.0]);
/// ```
pub struct StaticSampleDelay {
    buffer: Vec<Sample>,
    /// The current read/write position in the buffer. Public because in some situations it is more efficient to have access to it directly.
    pub position: usize,
    delay_length: usize,
}
#[impl_gen]
impl StaticSampleDelay {
    #[must_use]
    /// Create a new Self. delay_length_in_samples must be more than 0
    ///
    /// # Panics
    /// If delay_length_in_samples is 0
    pub fn new(delay_length_in_samples: usize) -> Self {
        assert!(delay_length_in_samples != 0);
        Self {
            buffer: vec![0.0; delay_length_in_samples],
            position: 0,
            delay_length: delay_length_in_samples,
        }
    }

    /// Set a new delay length for the delay. Real time safe. If the given length is longer than the max delay length it will be set to the max delay length.
    #[inline]
    pub fn set_delay_length(&mut self, delay_length_in_samples: usize) {
        self.delay_length = delay_length_in_samples.min(self.buffer.len());
    }
    /// Set a new delay length for the delay as a fraction of the entire delay length. Real time safe.
    pub fn set_delay_length_fraction(&mut self, fraction: Sample) {
        self.delay_length = (self.buffer.len() as Sample * fraction) as usize;
    }

    /// Read a whole block at a time. Only use this if the delay time is longer than 1 block.
    #[inline]
    pub fn read_block(&mut self, output: &mut [Sample]) {
        let block_size = output.len();
        assert!(self.buffer.len() >= block_size);
        let read_end = self.position + block_size;
        if read_end <= self.buffer.len() {
            output.copy_from_slice(&self.buffer[self.position..read_end]);
        } else {
            // block wraps around
            let read_end = read_end % self.delay_length;
            output[0..(block_size - read_end)].copy_from_slice(&self.buffer[self.position..]);
            output[(block_size - read_end)..].copy_from_slice(&self.buffer[0..read_end]);
        }
    }
    /// Write a whole block at a time. Only use this if the delay time is longer than 1 block. Advances the frame pointer.
    #[inline]
    pub fn write_block_and_advance(&mut self, input: &[Sample]) {
        let block_size = input.len();
        assert!(self.buffer.len() >= block_size);
        let write_end = self.position + block_size;
        if write_end <= self.buffer.len() {
            self.buffer[self.position..write_end].copy_from_slice(&input);
        } else {
            // block wraps around
            let write_end = write_end % self.delay_length;
            self.buffer[self.position..].copy_from_slice(&input[0..block_size - write_end]);
            self.buffer[0..write_end].copy_from_slice(&input[block_size - write_end..]);
        }
        self.position = (self.position + block_size) % self.buffer.len();
    }
    /// Read a sample from the buffer. Does not advance the frame pointer. Read first, then write.
    #[inline]
    pub fn read(&mut self) -> Sample {
        self.buffer[self.position]
    }
    /// Read a sample from the buffer at a given position. Does not advance the frame pointer. Read first, then write.
    pub fn read_at(&mut self, index: usize) -> Sample {
        self.buffer[index]
    }
    /// Read a sample from the buffer at a given position with linear interpolation. Does not advance the frame pointer. Read first, then write.
    pub fn read_at_lin(&mut self, index: Sample) -> Sample {
        let mut low = index.floor() as usize;
        let mut high = index.ceil() as usize;
        while low >= self.buffer.len() {
            low -= self.buffer.len();
            high -= self.buffer.len();
        }
        if high >= self.buffer.len() {
            high -= self.buffer.len();
        }
        let low_sample = self.buffer[low];
        let high_sample = self.buffer[high];
        low_sample + (high_sample - low_sample) * index.fract()
    }
    /// Write a sample to the buffer. Advances the frame pointer.
    #[inline]
    pub fn write_and_advance(&mut self, input: Sample) {
        self.buffer[self.position] = input;
        self.position = (self.position + 1) % self.delay_length;
    }
    /// Process one block of the delay. Will choose block based processing at runtime if the delay time is larger than the block size.
    #[inline]
    pub fn process(
        &mut self,
        input: &[Sample],
        output: &mut [Sample],
        block_size: BlockSize,
    ) -> GenState {
        if self.buffer.len() > *block_size {
            self.read_block(output);
            self.write_block_and_advance(input);
        } else {
            for (i, o) in input.iter().zip(output.iter_mut()) {
                *o = self.read();
                self.write_and_advance(*i);
            }
        }
        GenState::Continue
    }
}
