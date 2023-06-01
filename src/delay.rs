//! # Delay
//! This module contains some basic delay Gens

use crate::graph::Gen;
use crate::time::Superseconds;
use crate::Resources;
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
    max_delay_length: Superseconds,
}
impl SampleDelay {
    /// Create a new SampleDelay with a maximum delay time.
    pub fn new(max_delay_length: Superseconds) -> Self {
        Self {
            buffer: vec![0.0; 0],
            max_delay_length,
            write_position: 0,
        }
    }
}

impl Gen for SampleDelay {
    fn process(
        &mut self,
        ctx: crate::graph::GenContext,
        _resources: &mut Resources,
    ) -> crate::graph::GenState {
        let sig_buf = ctx.inputs.get_channel(0);
        let time_buf = ctx.inputs.get_channel(1);
        let out_buf = ctx.outputs.iter_mut().next().unwrap();
        for ((&input, &time), o) in sig_buf.iter().zip(time_buf).zip(out_buf.iter_mut()) {
            self.buffer[self.write_position] = input;
            let delay_samples = (time * ctx.sample_rate) as usize;
            *o = self.buffer
                [(self.write_position + self.buffer.len() - delay_samples) % self.buffer.len()];
            self.write_position = (self.write_position + 1) % self.buffer.len();
        }
        crate::graph::GenState::Continue
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn init(&mut self, _block_size: usize, sample_rate: crate::graph::Sample) {
        self.buffer =
            vec![0.0; (self.max_delay_length.to_seconds_f64() * sample_rate as f64) as usize];
        self.write_position = 0;
    }

    fn input_desc(&self, input: usize) -> &'static str {
        match input {
            0 => "signal",
            1 => "delay_time",
            _ => "",
        }
    }

    fn output_desc(&self, _output: usize) -> &'static str {
        "signal"
    }

    fn name(&self) -> &'static str {
        "SampleDelay"
    }
}

/*
#[derive(Clone, Debug)]
struct Grain {
    /// phase and read_ptr are two representations of the same thing, but phase is needed to index into the window buffer
    phase: Sample,
    phase_step: Sample,
    read_ptr: usize,
    active: bool,
}

impl Grain {
    fn inactive() -> Self {
        Self {
            phase: 0.0,
            phase_step: 0.0,
            read_ptr: 0,
            active: false,
        }
    }
    fn start(&mut self, start_pos: usize, phase_step: Sample) {
        self.active = true;
        self.phase = 0.0;
        self.phase_step = phase_step;
        self.read_ptr = start_pos;
    }
    fn update(&mut self) {
        self.phase += self.phase_step;
        self.read_ptr += 1;
        if self.phase >= 1.0 {
            self.active = false;
        }
    }
}
/// A ring buffer recording for a certain length of time and playing random snippets from the buffer
#[derive(Clone, Debug)]
pub struct FragmentedDelay {
    buffer: Vec<Sample>,
    write_ptr: usize,
    /// Grain window
    window: Wavetable,
    grains: Vec<Grain>,
    grain_index: usize,
    // In samples
    jump_distance_max: usize,
    // In samples
    jump_distance_min: usize,
    /// jump_distance_max - jump_distance_min
    jump_distance_width: usize,
    // In samples
    duration_max: usize,
    duration_min: usize,
    duration_width: usize, // optimisation to avoid duration_max-duration_min every sample
    sample_rate: Sample,
}
impl FragmentedDelay {
    pub fn new(
        recording_time: Sample,
        delay_duration_min: Sample,
        delay_duration_max: Sample,
        jump_distance_min: Sample,
        jump_distance_max: Sample,
        sample_rate: Sample,
    ) -> Self {
        let jump_distance_max = (jump_distance_max * sample_rate) as usize;
        let jump_distance_min = (jump_distance_min * sample_rate) as usize;
        let duration_max = (delay_duration_max * sample_rate) as usize;
        let duration_min = (delay_duration_min * sample_rate) as usize;
        let window = Wavetable::hann_window(4096);
        Self {
            buffer: vec![0.0; (recording_time * sample_rate).ceil() as usize],
            write_ptr: 0,
            window,
            grains: vec![Grain::inactive(); 20],
            grain_index: 0,
            jump_distance_max,
            jump_distance_min,
            jump_distance_width: jump_distance_max - jump_distance_min,
            duration_max,
            duration_min,
            duration_width: duration_max - duration_min,
            sample_rate,
        }
    }
    pub fn set_delay_duration_interval(&mut self, min: Sample, max: Sample) {
        self.duration_min = (min * self.sample_rate) as usize;
        self.duration_max = (max * self.sample_rate) as usize;
        self.duration_width = self.duration_max - self.duration_min;
    }
    /// Set the minimum and maximum time in seconds that the delay playback will jump back in time
    pub fn set_jump_interval(&mut self, min: Sample, max: Sample) {
        if max < min {
            panic!("set_jump_interval max is smaller than min");
        }
        self.jump_distance_min = (min * self.sample_rate) as usize;
        self.jump_distance_max = ((max * self.sample_rate) as usize).max(1);
        self.jump_distance_width = self.jump_distance_max - self.jump_distance_min;
    }
    #[inline(always)]
    pub fn process(&mut self, resources: &mut Resources, input: Sample) -> Sample {
        // First write and then read, since if the delay is 0 we should read the
        // current input and not one buffer size away
        //
        // Write the input into the buffer
        self.buffer[self.write_ptr] = input;
        // Advance write ptr
        self.write_ptr += 1;
        if self.write_ptr >= self.buffer.len() {
            self.write_ptr = 0;
        }
        // Read output from buffer
        let mut out = 0.0;
        for g in &mut self.grains {
            if g.active {
                out += self.buffer[g.read_ptr % self.buffer.len()] * self.window.get(g.phase);
                g.update();
            }
        }

        // Check if we need to add a new grain
        if resources.rng.f32() > 0.99966 {
            let duration = self.duration_min + resources.rng.usize(0..self.duration_width);
            let jump_distance =
                self.jump_distance_min + resources.rng.usize(0..self.jump_distance_width);
            // Add the buffer size to avoid usize underrun
            let mut start_pos = self.buffer.len() + self.write_ptr - jump_distance;
            while start_pos > self.buffer.len() {
                start_pos -= self.buffer.len();
            }
            let phase_step = duration as Sample / self.sample_rate.powi(2) as Sample;
            self.grains[self.grain_index].start(start_pos, phase_step);
            self.grain_index = (self.grain_index + 1) % self.grains.len();
        }

        // self.read_ptr += 1;
        // if self.read_ptr >= self.buffer.len() {
        //     self.read_ptr = 0;
        // }
        // // Check if we need to set a new read ptr position
        // if self.read_ptr == self.read_ptr_end {
        //     let duration = self.duration_min + resources.rng.usize(0..self.duration_width);
        //     let jump_distance =
        //         self.jump_distance_min + resources.rng.usize(0..self.jump_distance_width);
        //     // Add the buffer size to avoid usize underrun
        //     self.read_ptr = self.buffer.len() + self.write_ptr - jump_distance;
        //     while self.read_ptr > self.buffer.len() {
        //         self.read_ptr -= self.buffer.len();
        //     }
        //     self.read_ptr_end = self.read_ptr + duration;
        //     while self.read_ptr_end > self.buffer.len() {
        //         self.read_ptr_end -= self.buffer.len();
        //     }
        // }

        out
    }
}
*/
