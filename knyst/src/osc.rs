//! Contains some commonly used oscillators

use knyst_core::{
    buffer::BufferKey,
    gen::{Gen, GenContext, GenState, StopAction},
    resources::{BufferId, IdOrKey, WavetableId, WavetableKey},
    wavetable::{Wavetable, WavetablePhase, FRACTIONAL_PART, TABLE_SIZE},
    Resources, Sample,
};

/// Oscillator using a shared [`Wavetable`] stored in a [`Resources`]
/// *inputs*
/// 0. "freq": Frequency of oscillation
/// *outputs*
/// 0. "sig": Output signal
#[derive(Debug, Clone)]
pub struct Oscillator {
    step: u32,
    phase: WavetablePhase,
    wavetable: IdOrKey<WavetableId, WavetableKey>,
    amp: Sample,
    freq_to_phase_inc: f64,
}

#[allow(missing_docs)]
impl Oscillator {
    #[must_use]
    pub fn new(wavetable: IdOrKey<WavetableId, WavetableKey>) -> Self {
        Oscillator {
            step: 0,
            phase: WavetablePhase(0),
            wavetable,
            amp: 1.0,
            freq_to_phase_inc: 0., // set to a real value in init
        }
    }
    #[inline]
    pub fn set_freq(&mut self, freq: Sample) {
        self.step = (freq as f64 * self.freq_to_phase_inc) as u32;
    }
    #[inline]
    pub fn set_amp(&mut self, amp: Sample) {
        self.amp = amp;
    }
    #[inline]
    pub fn reset_phase(&mut self) {
        self.phase.0 = 0;
    }
}
impl Gen for Oscillator {
    fn process(&mut self, ctx: GenContext, resources: &mut Resources) -> GenState {
        let output = ctx.outputs.iter_mut().next().unwrap();
        let freq_buf = ctx.inputs.get_channel(0);
        let wt_key = match self.wavetable {
            IdOrKey::Id(id) => {
                if let Some(key) = resources.wavetable_key_from_id(id) {
                    self.wavetable = IdOrKey::Key(key);
                    key
                } else {
                    output.fill(0.0);
                    return GenState::Continue;
                }
            }
            IdOrKey::Key(key) => key,
        };
        if let Some(wt) = resources.wavetable(wt_key) {
            for (&freq, o) in freq_buf.iter().zip(output.iter_mut()) {
                self.set_freq(freq);
                self.phase.increase(self.step);
                *o = wt.get(self.phase) * self.amp;
            }
        } else {
            output.fill(0.0);
        }
        GenState::Continue
    }
    fn input_desc(&self, input: usize) -> &'static str {
        match input {
            0 => "freq",
            _ => "",
        }
    }
    fn output_desc(&self, output: usize) -> &'static str {
        match output {
            0 => "sig",
            _ => "",
        }
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn init(&mut self, _block_size: usize, sample_rate: Sample) {
        self.freq_to_phase_inc =
            TABLE_SIZE as f64 * f64::from(FRACTIONAL_PART) * (1.0 / f64::from(sample_rate));
    }
    fn name(&self) -> &'static str {
        "Oscillator"
    }
}

/// Reads a sample from a buffer and outputs it. In a multi channel [`Buffer`] only the first channel will be read.
/// TODO: Support rate through an argument with a default constant of 1
#[derive(Clone, Debug)]
pub struct BufferReader {
    buffer_key: IdOrKey<BufferId, BufferKey>,
    /// read pointer in samples
    read_pointer: f64,
    rate: f64,
    base_rate: f64, // The basic rate for playing the buffer at normal speed
    /// true if Self has finished reading the buffer
    finished: bool,
    /// true if the [`BufferReader`] should loop the buffer
    pub looping: bool,
    stop_action: StopAction,
}

impl BufferReader {
    #[allow(missing_docs)]
    pub fn new(
        buffer_key: IdOrKey<BufferId, BufferKey>,
        rate: f64,
        stop_action: StopAction,
    ) -> Self {
        BufferReader {
            buffer_key,
            read_pointer: 0.0,
            base_rate: 0.0, // initialise to the correct value the first time next() is called
            rate,
            finished: false,
            looping: false,
            stop_action,
        }
    }
    /// Jump back to the start of the buffer
    pub fn reset(&mut self) {
        self.jump_to(0.0);
    }
    /// Jump to a specific point in the buffer in samples
    pub fn jump_to(&mut self, new_pointer_pos: f64) {
        self.read_pointer = new_pointer_pos;
        self.finished = false;
    }
}

impl Gen for BufferReader {
    fn process(&mut self, ctx: GenContext, resources: &mut Resources) -> GenState {
        let mut stop_sample = None;

        let mut outputs = ctx.outputs.iter_mut();
        let output0 = outputs.next().unwrap();
        if !self.finished {
            if let IdOrKey::Id(id) = self.buffer_key {
                match resources.buffer_key_from_id(id) {
                    Some(key) => self.buffer_key = IdOrKey::Key(key),
                    None => (),
                }
            }
            if let IdOrKey::Key(buffer_key) = self.buffer_key {
                if let Some(buffer) = &mut resources.buffer(buffer_key) {
                    // Initialise the base rate if it hasn't been set
                    if self.base_rate == 0.0 {
                        self.base_rate = buffer.buf_rate_scale(ctx.sample_rate);
                    }

                    for (i, out) in output0.iter_mut().enumerate() {
                        let samples = buffer.get_interleaved((self.read_pointer) as usize);
                        *out = samples[0];
                        // println!("out: {}", sample);
                        self.read_pointer += self.base_rate * self.rate;
                        if self.read_pointer >= buffer.num_frames() {
                            self.finished = true;
                            if self.looping {
                                self.reset();
                            }
                        }
                        if self.finished {
                            stop_sample = Some(i + 1);
                            break;
                        }
                    }
                } else {
                    // Output zeroes if the buffer doesn't exist.
                    // TODO: Send error back to the user that the buffer doesn't exist without interrupting the audio thread.
                    // eprintln!("Error: BufferReader: buffer doesn't exist in Resources");
                    stop_sample = Some(0);
                }
            }
        } else {
            stop_sample = Some(0);
        }
        if let Some(stop_sample) = stop_sample {
            for out in output0[stop_sample..].iter_mut() {
                *out = 0.;
            }
            self.stop_action.to_gen_state(stop_sample)
        } else {
            GenState::Continue
        }
    }

    fn num_inputs(&self) -> usize {
        0
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn input_desc(&self, _input: usize) -> &'static str {
        ""
    }

    fn output_desc(&self, _output: usize) -> &'static str {
        "out"
    }

    fn name(&self) -> &'static str {
        "BufferReader"
    }
}

/// Play back a buffer with multiple channels. You cannot change the number of
/// channels after pushing this to a graph. If the buffer has fewer channels
/// than `num_channels`, the remaining outputs will be left at their current
/// value, not zeroed.
#[derive(Clone, Debug)]
pub struct BufferReaderMulti {
    buffer_key: BufferKey,
    read_pointer: f64,
    rate: f64,
    num_channels: usize,
    base_rate: f64, // The basic rate for playing the buffer at normal speed
    finished: bool,
    /// true if the BufferReaderMulti should loop the buffer
    pub looping: bool,
    stop_action: StopAction,
}

impl BufferReaderMulti {
    #[allow(missing_docs)]
    pub fn new(buffer_key: BufferKey, rate: f64, stop_action: StopAction) -> Self {
        Self {
            buffer_key,
            read_pointer: 0.0,
            base_rate: 0.0, // initialise to the correct value the first time next() is called
            rate,
            num_channels: 1,
            finished: false,
            looping: false,
            stop_action,
        }
    }
    /// Set looping
    pub fn looping(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }
    /// Set the number of channels to read and play
    pub fn channels(mut self, num_channels: usize) -> Self {
        self.num_channels = num_channels;
        self
    }
    /// Jump back to the start of the buffer
    pub fn reset(&mut self) {
        self.jump_to(0.0);
    }
    /// Jump to a specific point in time in samples
    pub fn jump_to(&mut self, new_pointer_pos: f64) {
        self.read_pointer = new_pointer_pos;
        self.finished = false;
    }
}

impl Gen for BufferReaderMulti {
    fn process(&mut self, ctx: GenContext, resources: &mut crate::Resources) -> GenState {
        let mut stop_sample = None;
        if !self.finished {
            if let Some(buffer) = &mut resources.buffer(self.buffer_key) {
                // Initialise the base rate if it hasn't been set
                if self.base_rate == 0.0 {
                    self.base_rate = buffer.buf_rate_scale(ctx.sample_rate);
                }
                for i in 0..ctx.block_size() {
                    let samples = buffer.get_interleaved((self.read_pointer) as usize);
                    for (out_num, sample) in samples.iter().take(self.num_channels).enumerate() {
                        ctx.outputs.write(*sample, out_num, i);
                    }
                    self.read_pointer += self.base_rate * self.rate;
                    if self.read_pointer >= buffer.num_frames() {
                        self.finished = true;
                        if self.looping {
                            self.reset();
                        }
                    }
                    if self.finished {
                        stop_sample = Some(i + 1);
                        break;
                    }
                }
            } else {
                // Output zeroes if the buffer doesn't exist.
                // TODO: Send error back to the user that the buffer doesn't exist without interrupting the audio thread.
                // eprintln!("Error: BufferReader: buffer doesn't exist in Resources");
                stop_sample = Some(0);
            }
        } else {
            stop_sample = Some(0);
        }
        if let Some(stop_sample) = stop_sample {
            let mut outputs = ctx.outputs.iter_mut();
            let output = outputs.next().unwrap();
            for out in output[stop_sample..].iter_mut() {
                *out = 0.;
            }
            self.stop_action.to_gen_state(stop_sample)
        } else {
            GenState::Continue
        }
    }

    fn num_inputs(&self) -> usize {
        0
    }

    fn num_outputs(&self) -> usize {
        self.num_channels
    }

    fn output_desc(&self, output: usize) -> &'static str {
        if output < self.num_channels {
            output_str(output)
        } else {
            ""
        }
    }

    fn name(&self) -> &'static str {
        "BufferReader"
    }
}

fn output_str(num: usize) -> &'static str {
    match num {
        0 => "output0",
        1 => "output1",
        2 => "output2",
        3 => "output3",
        4 => "output4",
        5 => "output5",
        6 => "output6",
        7 => "output7",
        8 => "output8",
        9 => "output9",
        10 => "output10",
        _ => "",
    }
}

/// Osciallator with an owned Wavetable
/// *inputs*
/// 0. "freq": The frequency of oscillation
/// *outputs*
/// 0. "sig": The signal
#[derive(Debug, Clone)]
pub struct WavetableOscillatorOwned {
    step: u32,
    phase: WavetablePhase,
    wavetable: Wavetable,
    amp: Sample,
    freq_to_phase_inc: f64,
}

impl WavetableOscillatorOwned {
    #[allow(missing_docs)]
    #[must_use]
    pub fn new(wavetable: Wavetable) -> Self {
        WavetableOscillatorOwned {
            step: 0,
            phase: WavetablePhase(0),
            wavetable,
            amp: 1.0,
            freq_to_phase_inc: 0.0, // set to a real value in init
        }
    }
    /// Set the frequency of the oscillation. This will be overwritten by the
    /// input frequency if used as a Gen.
    pub fn set_freq(&mut self, freq: Sample) {
        self.step = (freq as f64 * self.freq_to_phase_inc) as u32;
    }
    /// Set the amplitude of the signal.
    pub fn set_amp(&mut self, amp: Sample) {
        self.amp = amp;
    }
    /// Reset the phase of the oscillator.
    pub fn reset_phase(&mut self) {
        self.phase.0 = 0;
    }

    /// Generate the next sample given the current settings.
    #[inline(always)]
    #[must_use]
    pub fn next_sample(&mut self) -> Sample {
        // Use the phase to index into the wavetable
        // self.wavetable.get_linear_interp(temp_phase) * self.amp
        let sample = self.wavetable.get(self.phase) * self.amp;
        self.phase.increase(self.step);
        sample
    }
}

impl Gen for WavetableOscillatorOwned {
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        let output = ctx.outputs.iter_mut().next().unwrap();
        let freq_buf = ctx.inputs.get_channel(0);
        assert!(freq_buf.len() == output.len());
        for (&freq, o) in freq_buf.iter().zip(output.iter_mut()) {
            self.set_freq(freq);
            *o = self.next_sample();
        }
        GenState::Continue
    }
    fn input_desc(&self, input: usize) -> &'static str {
        match input {
            0 => "freq",
            _ => "",
        }
    }
    fn output_desc(&self, output: usize) -> &'static str {
        match output {
            0 => "sig",
            _ => "",
        }
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn init(&mut self, _block_size: usize, sample_rate: Sample) {
        self.reset_phase();
        self.freq_to_phase_inc =
            TABLE_SIZE as f64 * FRACTIONAL_PART as f64 * (1.0 / sample_rate as f64);
    }
    fn name(&self) -> &'static str {
        "WavetableOscillatorOwned"
    }
}
