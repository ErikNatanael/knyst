//! Contains some commonly used oscillators

#[allow(unused)]
use crate::buffer::Buffer;
use crate::{
    buffer::BufferKey,
    gen::{Gen, GenContext, GenState, StopAction},
    prelude::Seconds,
    resources::{BufferId, IdOrKey, WavetableId, WavetableKey},
    wavetable::{WavetablePhase, FRACTIONAL_PART, TABLE_SIZE},
    wavetable_aa::Wavetable,
    Resources, Sample, SampleRate,
};
use knyst_macro::impl_gen;

// Necessary to use impl_gen from inside the knyst crate
use crate::{self as knyst, handles::HandleData, modal_interface::knyst_commands};

/// Oscillator using a shared [`Wavetable`] stored in a [`Resources`]. Assumes the wavetable has normal range for the `range` method on the Handle.
/// *inputs*
/// 0. "freq": Frequency of oscillation
/// *outputs*
/// 0. "sig": Output signal
#[derive(Debug, Clone)]
pub struct Oscillator {
    step: u32,
    phase: WavetablePhase,
    wavetable: IdOrKey<WavetableId, WavetableKey>,
    freq_to_phase_inc: f64,
}

#[allow(missing_docs)]
#[impl_gen(range = normal)]
impl Oscillator {
    #[new]
    #[must_use]
    pub fn new(wavetable: impl Into<IdOrKey<WavetableId, WavetableKey>>) -> Self {
        Oscillator {
            step: 0,
            phase: WavetablePhase(0),
            wavetable: wavetable.into(),
            freq_to_phase_inc: 0., // set to a real value in init
        }
    }
    #[inline]
    pub fn set_freq(&mut self, freq: Sample) {
        self.step = (freq as f64 * self.freq_to_phase_inc) as u32;
    }
    #[inline]
    pub fn reset_phase(&mut self) {
        self.phase.0 = 0;
    }
    #[process]
    pub fn process(
        &mut self,
        freq: &[Sample],
        sig: &mut [Sample],
        resources: &mut Resources,
    ) -> GenState {
        let wt_key = match self.wavetable {
            IdOrKey::Id(id) => {
                if let Some(key) = resources.wavetable_key_from_id(id) {
                    self.wavetable = IdOrKey::Key(key);
                    key
                } else {
                    sig.fill(0.0);
                    return GenState::Continue;
                }
            }
            IdOrKey::Key(key) => key,
        };
        if let Some(wt) = resources.wavetable(wt_key) {
            for (&f, o) in freq.iter().zip(sig.iter_mut()) {
                *o = wt.get_linear_interp(self.phase, f);
                self.set_freq(f);
                self.phase.increase(self.step);
                // TODO: Set a buffer of phase values and request them all from the wavetable all at the same time. Should enable SIMD in the wavetable lookup.
            }
        } else {
            sig.fill(0.0);
        }
        GenState::Continue
    }
    #[init]
    pub fn init(&mut self, sample_rate: SampleRate) {
        self.freq_to_phase_inc =
            TABLE_SIZE as f64 * f64::from(FRACTIONAL_PART) * (1.0 / f64::from(sample_rate));
    }
}
// impl Gen for Oscillator {
//     fn process(&mut self, ctx: GenContext, resources: &mut Resources) -> GenState {
//         let output = ctx.outputs.iter_mut().next().unwrap();
//         let freq_buf = ctx.inputs.get_channel(0);
//     }
//     fn input_desc(&self, input: usize) -> &'static str {
//         match input {
//             0 => "freq",
//             _ => "",
//         }
//     }
//     fn output_desc(&self, output: usize) -> &'static str {
//         match output {
//             0 => "sig",
//             _ => "",
//         }
//     }
//     fn num_outputs(&self) -> usize {
//         1
//     }
//     fn num_inputs(&self) -> usize {
//         1
//     }
//     fn init(&mut self, _block_size: usize, sample_rate: Sample) {
//         self.freq_to_phase_inc =
//             TABLE_SIZE as f64 * f64::from(FRACTIONAL_PART) * (1.0 / f64::from(sample_rate));
//     }
//     fn name(&self) -> &'static str {
//         "Oscillator"
//     }
// }

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
    start_time: Seconds,
}

#[impl_gen]
impl BufferReader {
    #[allow(missing_docs)]
    #[must_use]
    pub fn new(
        buffer: impl Into<IdOrKey<BufferId, BufferKey>>,
        rate: f64,
        looping: bool,
        stop_action: StopAction,
    ) -> Self {
        BufferReader {
            buffer_key: buffer.into(),
            read_pointer: 0.0,
            base_rate: 0.0,
            rate,
            finished: false,
            looping,
            stop_action,
            start_time: Seconds::ZERO,
        }
    }
    /// Jump back to the start of the buffer
    fn reset(&mut self) {
        self.jump_to(0.0);
    }
    /// Jump to a specific point in the buffer in samples
    fn jump_to(&mut self, new_pointer_pos: f64) {
        self.read_pointer = new_pointer_pos;
        self.finished = false;
    }
    /// Jump to a specific point in the buffer in samples. Has to be called before processing starts.
    pub fn start_at(mut self, start_time: Seconds) -> Self {
        self.start_time = start_time;
        self
    }
    /// Process block
    pub fn process(
        &mut self,
        out: &mut [Sample],
        resources: &mut Resources,
        sample_rate: SampleRate,
    ) -> GenState {
        let mut stop_sample = None;
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
                    // TODO: Move this to init? Would require the buffer to be inserted though, i.e. no inserting the buffer late.
                    if self.base_rate == 0.0 {
                        self.base_rate = buffer.buf_rate_scale(sample_rate);
                        // Also init start time since this would be the first block
                        let start_frame = self.start_time.to_samples(buffer.sample_rate() as u64);
                        self.jump_to(start_frame as f64);
                    }

                    for (i, o) in out.iter_mut().enumerate() {
                        if self.read_pointer >= buffer.num_frames() {
                            self.finished = true;
                            if self.looping {
                                self.reset();
                            }
                        }
                        if self.finished {
                            stop_sample = Some(i);
                            break;
                        }
                        let samples = buffer.get_interleaved((self.read_pointer) as usize);
                        *o = samples[0];
                        // println!("out: {}", sample);
                        self.read_pointer += self.base_rate * self.rate;
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
            for o in out[stop_sample..].iter_mut() {
                *o = 0.;
            }
            self.stop_action.to_gen_state(stop_sample)
        } else {
            GenState::Continue
        }
    }
}

/// Play back a buffer with multiple channels. You cannot change the number of
/// channels after pushing this to a graph. If the buffer has fewer channels
/// than `num_channels`, the remaining outputs will be left at their current
/// value, not zeroed.
#[derive(Clone, Debug)]
pub struct BufferReaderMulti {
    buffer_key: IdOrKey<BufferId, BufferKey>,
    read_pointer: f64,
    rate: f64,
    num_channels: usize,
    base_rate: f64, // The basic rate for playing the buffer at normal speed
    finished: bool,
    /// true if the BufferReaderMulti should loop the buffer
    pub looping: bool,
    stop_action: StopAction,
}

// TODO: Make this generic over the number of inputs? How would that interact with the impl_gen macro?
impl BufferReaderMulti {
    #[allow(missing_docs)]
    pub fn new(
        buffer: impl Into<IdOrKey<BufferId, BufferKey>>,
        rate: f64,
        stop_action: StopAction,
    ) -> Self {
        Self {
            buffer_key: buffer.into(),
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
    /// Upload to the current graph, returning a handle to the new node
    pub fn upload(self) -> knyst::handles::Handle<BufferReaderMultiHandle> {
        let num_channels = self.num_channels;
        let id = knyst::prelude::KnystCommands::push_without_inputs(&mut knyst_commands(), self);
        knyst::handles::Handle::new(BufferReaderMultiHandle {
            node_id: id,
            num_channels,
        })
    }
}

impl Gen for BufferReaderMulti {
    fn process(&mut self, ctx: GenContext, resources: &mut crate::Resources) -> GenState {
        let mut stop_sample = None;
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
                        self.base_rate = buffer.buf_rate_scale(ctx.sample_rate.into());
                    }
                    for i in 0..ctx.block_size() {
                        let samples = buffer.get_interleaved((self.read_pointer) as usize);
                        for (out_num, sample) in samples.iter().take(self.num_channels).enumerate()
                        {
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
        "BufferReaderMulti"
    }
}

/// Upload a [`BufferReaderMulti`] to the current graph and return a handle to it.
pub fn buffer_reader_multi(
    buffer: BufferId,
    rate: f64,
    looping: bool,
    stop_action: StopAction,
) -> knyst::handles::Handle<BufferReaderMultiHandle> {
    let gen = BufferReaderMulti::new(buffer, rate, stop_action)
        .looping(looping)
        .channels(buffer.num_channels());
    let num_channels = buffer.num_channels();
    let id = knyst::prelude::KnystCommands::push_without_inputs(&mut knyst_commands(), gen);
    knyst::handles::Handle::new(BufferReaderMultiHandle {
        node_id: id,
        num_channels,
    })
}
/// Handle to a [`BufferReaderMulti`]
#[derive(Clone, Copy, Debug)]
pub struct BufferReaderMultiHandle {
    node_id: knyst::graph::NodeId,
    num_channels: usize,
}
impl HandleData for BufferReaderMultiHandle {
    fn out_channels(&self) -> knyst::handles::SourceChannelIter {
        knyst::handles::SourceChannelIter::single_node_id(self.node_id, self.num_channels)
    }

    fn in_channels(&self) -> knyst::handles::SinkChannelIter {
        knyst::handles::SinkChannelIter::single_node_id(self.node_id, 0)
    }

    fn node_ids(&self) -> knyst::handles::NodeIdIter {
        knyst::handles::NodeIdIter::Single(self.node_id)
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
    freq: Sample,
}

impl WavetableOscillatorOwned {
    /// Set the frequency of the oscillation. This will be overwritten by the
    /// input frequency if used as a Gen.
    pub fn set_freq(&mut self, freq: Sample) {
        self.freq = freq;
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
        let sample = self.wavetable.get(self.phase, self.freq) * self.amp;
        self.phase.increase(self.step);
        sample
    }
}

#[impl_gen(range=normal)]
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
            freq: 0.,
        }
    }
    fn process(&mut self, freq: &[Sample], sig: &mut [Sample]) -> GenState {
        assert!(freq.len() == sig.len());
        for (&freq, o) in freq.iter().zip(sig.iter_mut()) {
            self.set_freq(freq);
            *o = self.next_sample();
        }
        GenState::Continue
    }
    fn init(&mut self, sample_rate: SampleRate) {
        self.reset_phase();
        self.freq_to_phase_inc =
            TABLE_SIZE as f64 * FRACTIONAL_PART as f64 * (1.0 / sample_rate.to_f64());
    }
}

/// Linear ramp from 0 to 1 at a given frequency. Will alias at higher frequencies.
struct Phasor {
    phase: f64,
    freq_to_phase_step_mult: f64,
}

#[impl_gen]
impl Phasor {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {
            phase: 0.0,
            freq_to_phase_step_mult: 0.0,
        }
    }
    #[allow(missing_docs)]
    pub fn init(&mut self, sample_rate: SampleRate) {
        self.freq_to_phase_step_mult = 1.0_f64 / (sample_rate.to_f64() - 0.0);
    }
    #[allow(missing_docs)]
    pub fn process(&mut self, freq: &[Sample], output: &mut [Sample]) -> GenState {
        for (freq, out) in freq.iter().zip(output.iter_mut()) {
            *out = self.phase as Sample;
            let step = *freq as f64 * self.freq_to_phase_step_mult;
            self.phase += step;
            while self.phase >= 1.0 {
                self.phase -= 1.0;
            }
        }
        GenState::Continue
    }
}
/// Sawtooth wave starting at 0. Linear ramp from 0 to 1, then -1 to 0, at a given frequency. Will alias at higher frequencies.
struct AliasingSaw {
    phase: f64,
    freq_to_phase_step_mult: f64,
}

#[impl_gen]
impl AliasingSaw {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {
            phase: 0.0,
            freq_to_phase_step_mult: 0.0,
        }
    }
    #[allow(missing_docs)]
    pub fn init(&mut self, sample_rate: SampleRate) {
        self.freq_to_phase_step_mult = 2.0_f64 / (sample_rate.to_f64());
    }
    #[allow(missing_docs)]
    pub fn process(&mut self, freq: &[Sample], output: &mut [Sample]) -> GenState {
        for (freq, out) in freq.iter().zip(output.iter_mut()) {
            *out = self.phase as Sample;
            let step = *freq as f64 * self.freq_to_phase_step_mult;
            self.phase += step;
            while self.phase >= 1.0 {
                self.phase -= 2.0;
            }
        }
        GenState::Continue
    }
}

#[cfg(test)]
mod tests {
    use crate::offline::KnystOffline;

    use super::*;
    use knyst::prelude::*;
    #[test]
    fn phasor_test() {
        let mut kt = KnystOffline::new(128, 64, 0, 1);
        let p = phasor().freq(2.0);
        graph_output(0, p);
        kt.process_block();
        let output = kt.output_channel(0).unwrap();
        dbg!(1.0 / 128.0);
        assert_eq!(output[0], 0.0);
        assert!((output[63] - 1.0).abs() < 0.1, "{}", output[63]);
        // Test that it keeps in phase over time
        kt.process_block();
        kt.process_block();
        kt.process_block();
        let output = kt.output_channel(0).unwrap();
        assert!((output[0]).abs() < 0.1, "{}", output[0]);
        assert!((output[63] - 1.0).abs() < 0.1, "{}", output[63]);
    }
    #[test]
    fn saw_test() {
        let mut kt = KnystOffline::new(64, 64, 0, 1);
        let p = aliasing_saw().freq(2.0);
        graph_output(0, p);
        kt.process_block();
        let output = kt.output_channel(0).unwrap();
        dbg!(output);
        assert_eq!(output[0], 0.0);
        assert!((output[7] - 0.5).abs() < 0.1, "{}", output[7]);
        assert!((output[15] - 1.0).abs() < 0.1, "{}", output[15]);
        assert!((output[16] + 1.0).abs() < 0.1, "{}", output[16]);
        assert!(output[31].abs() < 0.1, "{}", output[31]);
        kt.process_block();
        let output = kt.output_channel(0).unwrap();
        assert!((output[0]).abs() < 0.1);
        assert!((output[7] - 0.5).abs() < 0.1, "{}", output[7]);
        assert!((output[15] - 1.0).abs() < 0.1, "{}", output[15]);
        assert!((output[16] + 1.0).abs() < 0.1, "{}", output[16]);
        assert!(output[31].abs() < 0.1, "{}", output[31]);
    }
}
