use crate::{Resources, Sample};

use crate::graph::{Gen, GenState};
// use std::f64::consts::PI;
use crate::xorrng::XOrShift32Rng;
use std::f32::consts::PI;

pub const SINE_WAVETABLE: WavetableIndex = 0;

// We could later turn WavetableIndex into a generational index if we'd want
pub type WavetableIndex = usize;

#[derive(Debug, Clone)]
pub struct Wavetable {
    buffer: Vec<Sample>, // Box<[Sample; 131072]>,
    // Store the size as an f64 to find fractional indexes without typecasting
    size: Sample,
}

impl Wavetable {
    pub fn new(wavetable_size: usize) -> Self {
        let w_size = if !is_power_of_2(wavetable_size) {
            // Make a power of two by taking the log2 and discarding the fractional part of the answer and then squaring again
            ((wavetable_size as f64).log2() as usize).pow(2)
        } else {
            wavetable_size
        };
        let buffer = vec![0.0; w_size];
        Wavetable {
            buffer,
            size: w_size as Sample,
        }
    }
    pub fn from_buffer(buffer: Vec<Sample>) -> Self {
        let size = buffer.len() as Sample;
        Self { buffer, size }
    }
    pub fn sine(wavetable_size: usize) -> Self {
        let mut wt = Wavetable::new(wavetable_size);
        // Fill buffer with a sine
        for i in 0..wavetable_size {
            wt.buffer[i] = ((i as Sample / wt.size) * PI * 2.0).sin();
        }
        wt
    }
    pub fn multi_sine(wavetable_size: usize, num_harmonics: usize) -> Self {
        let mut wt = Wavetable::new(wavetable_size);
        wt.fill_sine(num_harmonics, 1.0);
        wt.add_noise(0.95, (num_harmonics as f64 * 3723.83626).floor() as u32);
        wt.normalize();
        wt
    }
    pub fn crazy(wavetable_size: usize, seed: u32) -> Self {
        let mut wt = Wavetable::new(wavetable_size);
        let mut xorrng = XOrShift32Rng::new(seed);
        wt.fill_sine(16, 1.0);
        for _ in 0..(xorrng.gen_u32() % 3 + 1) {
            wt.fill_sine(16, (xorrng.gen_f32() * 32.0).floor());
        }
        wt.add_noise(1.0 - xorrng.gen_f64() * 0.05, seed + wavetable_size as u32);
        wt.normalize();
        wt
    }
    /// Produces a Hann window
    pub fn hann_window(wavetable_size: usize) -> Self {
        let mut wt = Wavetable::new(wavetable_size);
        // This approach was heavily influenced by the SuperCollider Signal implementation
        wt.fill(0.5);
        wt.add_sine(1.0, 0.5, -0.5 * PI);
        wt
    }
    /// Produces a Hamming window
    pub fn hamming_window(wavetable_size: usize) -> Self {
        let mut wt = Wavetable::new(wavetable_size);
        // This approach was heavily influenced by the SuperCollider Signal implementation
        wt.fill(0.53836);
        wt.add_sine(1.0, 0.46164, -0.5 * PI);
        wt
    }
    /// Produces a Sine window
    pub fn sine_window(wavetable_size: usize) -> Self {
        let mut wt = Wavetable::new(wavetable_size);
        // This approach was heavily influenced by the SuperCollider Signal implementation
        wt.add_sine(0.5, 1.0, 0.0);
        wt
    }
    pub fn fill(&mut self, value: Sample) {
        for sample in &mut self.buffer {
            *sample = value;
        }
    }
    pub fn add_sine(&mut self, freq: Sample, amplitude: Sample, phase: Sample) {
        let step = (freq * PI * 2.0) / self.size;
        let mut phase = phase;
        for sample in &mut self.buffer {
            *sample += phase.sin() * amplitude;
            phase += step;
        }
    }
    pub fn fill_sine(&mut self, num_harmonics: usize, freq: Sample) {
        for n in 0..num_harmonics {
            let start_phase = 0.0;
            let harmonic_amp = match n {
                0 => 1.0,
                _ => ((num_harmonics - n) as Sample / (num_harmonics) as Sample) * 0.5,
            };
            for i in 0..self.size as usize {
                self.buffer[i] += ((i as Sample / self.size) * PI * 2.0 * freq * (n + 1) as Sample
                    + start_phase)
                    .sin()
                    * harmonic_amp;
            }
        }
    }
    pub fn add_saw(&mut self, num_harmonics: usize, amp: Sample) {
        for i in 0..num_harmonics {
            let start_phase = 0.0;
            let harmonic_amp = 1.0 / ((i + 1) as Sample * PI);
            for k in 0..self.buffer.len() {
                self.buffer[k] +=
                    ((k as Sample / self.buffer.len() as Sample * PI * 2.0 * (i + 1) as Sample
                        + start_phase)
                        .sin()
                        * harmonic_amp)
                        * amp;
            }
        }
    }
    pub fn add_odd_harmonics(&mut self, num_harmonics: usize, amp_falloff: Sample) {
        for i in 0..num_harmonics {
            let start_phase = match i {
                0 => 0.0,
                _ => (-1.0 as Sample).powi(i as i32 + 2),
            };
            // an amp_falloff of 2.0 gives triangle wave approximation
            let harmonic_amp = 1.0 / ((i * 2 + 1) as Sample).powf(amp_falloff);
            // Add this odd harmonic to the buffer
            for k in 0..self.buffer.len() {
                self.buffer[k] += (k as Sample / self.buffer.len() as Sample
                    * PI
                    * 2.0
                    * ((i * 2) as Sample + 1.0)
                    + start_phase)
                    .sin()
                    * harmonic_amp;
            }
        }
    }
    pub fn add_noise(&mut self, probability: f64, seed: u32) {
        let mut xorrng = XOrShift32Rng::new(seed);
        for sample in &mut self.buffer {
            if xorrng.gen_f64() > probability {
                *sample += xorrng.gen_f32() - 0.5;
                if *sample > 1.0 {
                    *sample -= 1.0;
                }
                if *sample < -1.0 {
                    *sample += 1.0;
                }
            }
        }
    }
    pub fn multiply(&mut self, mult: Sample) {
        for sample in &mut self.buffer {
            *sample *= mult;
        }
    }
    pub fn normalize(&mut self) {
        // Find highest absolute value
        let mut loudest_sample = 0.0;
        for sample in &self.buffer {
            if sample.abs() > loudest_sample {
                loudest_sample = sample.abs();
            }
        }
        // Scale buffer
        let scaler = 1.0 / loudest_sample;
        for sample in &mut self.buffer {
            *sample *= scaler;
        }
    }

    /// Linearly interpolate between the value in between which the phase points.
    /// The phase is assumed to be 0 <= phase < 1
    #[inline]
    pub fn get_linear_interp(&self, phase: Sample) -> Sample {
        let index = self.size * phase;
        let mix = index.fract();
        let v0 = self.buffer[index.floor() as usize];
        let v1 = self.buffer[index.ceil() as usize % self.buffer.len()];
        // let (v0, v1) = unsafe {
        //     (
        //         self.buffer.get_unchecked(index.floor() as usize),
        //         self.buffer
        //             .get_unchecked(index.ceil() as usize % self.buffer.len()),
        //     )
        // };
        v0 + (v1 - v0) * mix
    }

    /// Get the closest sample with no interpolation
    #[inline]
    pub fn get(&self, phase: Sample) -> Sample {
        let index = (self.size * phase) as usize;
        // self.buffer[index]
        unsafe { *self.buffer.get_unchecked(index) }
    }
}

fn is_power_of_2(num: usize) -> bool {
    return num > 0 && num & (num - 1) == 0;
}

pub struct WavetableArena {
    wavetables: Vec<Option<Wavetable>>,
    next_free_index: WavetableIndex,
    _freed_indexes: Vec<WavetableIndex>,
}

impl WavetableArena {
    pub fn new() -> Self {
        let mut wavetables = Vec::with_capacity(20);
        for _ in 0..20 {
            wavetables.push(None);
        }
        WavetableArena {
            wavetables,
            next_free_index: 0,
            _freed_indexes: vec![],
        }
    }
    pub fn get(&self, index: WavetableIndex) -> &Option<Wavetable> {
        &self.wavetables[index]
    }
    pub fn add(&mut self, wavetable: Wavetable) -> WavetableIndex {
        // TODO: In order to do this safely in an audio thread we should pass the old value on to a helper thread for deallocation
        // since dropping it here would probably deallocate it.
        let _old_wavetable = self.wavetables[self.next_free_index].replace(wavetable);
        let index = self.next_free_index;
        self.next_free_index += 1;
        // TODO: Check that the next free index is within the bounds of the wavetables Vec or else use the indexes that have been freed
        index
    }
}

/// Osciallator with an owned Wavetable
#[derive(Debug, Clone)]
pub struct WavetableOscillatorOwned {
    step: Sample,
    phase: Sample,
    wavetable: Wavetable,
    amp: Sample,
    sample_rate: Sample,
}

impl WavetableOscillatorOwned {
    pub fn new(wavetable: Wavetable, sample_rate: Sample) -> Self {
        WavetableOscillatorOwned {
            step: 0.0,
            phase: 0.0,
            wavetable,
            amp: 1.0,
            sample_rate,
        }
    }
    pub fn from_freq(wavetable: Wavetable, sample_rate: Sample, freq: Sample, amp: Sample) -> Self {
        let mut osc = Self::new(wavetable, sample_rate);
        osc.amp = amp;
        osc.set_freq(freq);
        osc
    }
    pub fn set_freq(&mut self, freq: Sample) {
        self.step = freq / self.sample_rate;
    }
    pub fn set_amp(&mut self, amp: Sample) {
        self.amp = amp;
    }
    pub fn reset_phase(&mut self) {
        self.phase = 0.0;
    }

    #[inline(always)]
    pub fn next(&mut self) -> Sample {
        let temp_phase = self.phase;
        self.phase += self.step;
        self.phase -= self.phase.floor();
        // Use the phase to index into the wavetable
        // self.wavetable.get_linear_interp(temp_phase) * self.amp
        self.wavetable.get(temp_phase) * self.amp
    }
}

#[derive(Clone)]
pub struct Oscillator {
    step: Sample,
    phase: Sample,
    wavetable: WavetableIndex,
    amp: Sample,
    sample_rate: Sample,
}

impl Oscillator {
    pub fn new(wavetable: WavetableIndex, sample_rate: Sample) -> Self {
        Oscillator {
            step: 0.0,
            phase: 0.0,
            wavetable,
            amp: 1.0,
            sample_rate,
        }
    }
    pub fn from_freq(
        wavetable: WavetableIndex,
        sample_rate: Sample,
        freq: Sample,
        amp: Sample,
    ) -> Self {
        let mut osc = Oscillator::new(wavetable, sample_rate);
        osc.amp = amp;
        osc.set_freq(freq);
        osc
    }
    pub fn set_freq(&mut self, freq: Sample) {
        self.step = freq / self.sample_rate;
    }
    pub fn set_amp(&mut self, amp: Sample) {
        self.amp = amp;
    }
    pub fn reset_phase(&mut self) {
        self.phase = 0.0;
    }
    #[inline]
    fn next(&mut self, resources: &mut Resources) -> Sample {
        let temp_phase = self.phase;
        self.phase += self.step;
        while self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        // Use the phase to index into the wavetable
        match resources.wavetable_arena.get(self.wavetable) {
            Some(wt) => wt.get(temp_phase) * self.amp,
            None => {
                eprintln!("Wavetable doesn't exist: {}", self.wavetable);
                0.0
            }
        }
    }
}
impl Gen for Oscillator {
    fn process(
        &mut self,
        inputs: &[Box<[Sample]>],
        outputs: &mut [Box<[Sample]>],
        resources: &mut Resources,
    ) -> GenState {
        let output = &mut outputs[0];
        let freq_buf = &inputs[0];
        for (&freq, o) in freq_buf.iter().zip(output.iter_mut()) {
            self.set_freq(freq);
            *o = self.next(resources);
        }
        GenState::Continue
    }
    fn input_desc(&self, input: usize) -> &'static str {
        match input {
            0 => "freq",
            _ => "",
        }
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn num_inputs(&self) -> usize {
        1
    }
}
