//! Wavetable synthesis

use slotmap::new_key_type;

use crate::{Resources, Sample};

use crate::graph::{Gen, GenContext, GenState};
// use std::f64::consts::PI;
use crate::xorrng::XOrShift32Rng;
use std::f32::consts::PI;

/// Decides the number of samples per [`Wavetable`] buffer, and therefore also
/// the number of high bits used for the phase indexing into the wavetable. With
/// the current u32 phase, this can be maximum 16.
pub const TABLE_POWER: u32 = 14;
/// TABLE_SIZE is 2^TABLE_POWER
pub const TABLE_SIZE: usize = 2_usize.pow(TABLE_POWER);
/// The high mask is used to 0 everything above the table size so that adding
/// further would have the same effect as wrapping.
pub const TABLE_HIGH_MASK: u32 = TABLE_SIZE as u32 - 1;
/// Max number of the fractional part of a integer phase. Currently, 16 bits are used for the fractional part.
pub const FRACTIONAL_PART: u32 = 65536;

// We could later turn WavetableIndex into a generational index if we'd want
pub type WavetableIndex = usize;

new_key_type! {
    /// Key for selecting a wavetable that has been added to Resources.
    pub struct WavetableKey;
}

/// Wavetable is a standardised wavetable with a buffer of samples, as well as a
/// separate buffer with the difference between the current sample and the next.
/// The wavetable is of size [`TABLE_SIZE`] and can be indexed using a [`Phase`].
///
/// It is not safe to modify the wavetable while it is being used on the audio
/// thread, even if no Node is currently reading from it, because most modifying
/// operations may allocate.
#[derive(Debug, Clone)]
pub struct Wavetable {
    buffer: Vec<Sample>,      // Box<[Sample; 131072]>,
    diff_buffer: Vec<Sample>, // Box<[Sample; 131072]>,
}

impl Default for Wavetable {
    fn default() -> Self {
        let buffer = vec![0.0; TABLE_SIZE];
        let diff_buffer = vec![0.0; TABLE_SIZE];
        Wavetable {
            buffer,
            diff_buffer,
        }
    }
}

impl Wavetable {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Recalculate the difference between samples in the buffer.
    ///
    /// The [`Wavetable`] contains a buffer with the difference between each
    /// sample of the buffer for efficiency reason.
    pub fn update_diff_buffer(&mut self) {
        let diff_buffer: Vec<f32> = self
            .buffer
            .iter()
            .zip(self.buffer.iter().skip(1).cycle())
            .map(|(&a, &b)| b - a)
            .collect();
        self.diff_buffer = diff_buffer;
    }
    /// Create a [`Wavetable`] from an existing buffer.
    ///
    /// # Errors
    /// The buffer has to be of [`TABLE_SIZE`] length, otherwise an error will be returned.
    pub fn from_buffer(buffer: Vec<Sample>) -> Result<Self, String> {
        if buffer.len() != TABLE_SIZE {
            return Err(format!(
                "Invalid size buffer for a wavetable: {}. Wavetables must be of size {}",
                buffer.len(),
                TABLE_SIZE,
            ));
        }
        let mut w = Self {
            buffer,
            ..Default::default()
        };
        w.update_diff_buffer();
        Ok(w)
    }
    /// Create a new [`Wavetable`] and populate it using the closure/function provided.
    #[must_use]
    pub fn from_closure<F>(f: F) -> Self
    where
        F: FnOnce(&mut [Sample]),
    {
        let mut w = Self::default();
        f(&mut w.buffer);
        w.update_diff_buffer();
        w
    }
    #[must_use]
    pub fn sine() -> Self {
        let wavetable_size = TABLE_SIZE;
        let mut wt = Wavetable::new();
        // Fill buffer with a sine
        for i in 0..wavetable_size {
            wt.buffer[i] = ((i as Sample / TABLE_SIZE as f32) * PI * 2.0).sin();
        }
        wt.update_diff_buffer();
        wt
    }
    #[must_use]
    pub fn multi_sine(num_harmonics: usize) -> Self {
        let mut wt = Wavetable::new();
        wt.fill_sine(num_harmonics, 1.0);
        wt.add_noise(0.95, (num_harmonics as f64 * 3723.83626).floor() as u32);
        wt.normalize();
        wt.update_diff_buffer();
        wt
    }
    #[must_use]
    pub fn crazy(seed: u32) -> Self {
        let wavetable_size = TABLE_SIZE;
        let mut wt = Wavetable::new();
        let mut xorrng = XOrShift32Rng::new(seed);
        wt.fill_sine(16, 1.0);
        for _ in 0..(xorrng.gen_u32() % 3 + 1) {
            wt.fill_sine(16, (xorrng.gen_f32() * 32.0).floor());
        }
        wt.add_noise(1.0 - xorrng.gen_f64() * 0.05, seed + wavetable_size as u32);
        wt.normalize();
        wt.update_diff_buffer();
        wt
    }
    #[must_use]
    /// Produces a Hann window
    pub fn hann_window() -> Self {
        let mut wt = Wavetable::new();
        // This approach was heavily influenced by the SuperCollider Signal implementation
        wt.fill(0.5);
        wt.add_sine(1.0, 0.5, -0.5 * PI);
        wt.update_diff_buffer();
        wt
    }
    /// Produces a Hamming window
    #[must_use]
    pub fn hamming_window() -> Self {
        let mut wt = Wavetable::new();
        // This approach was heavily influenced by the SuperCollider Signal implementation
        wt.fill(0.53836);
        wt.add_sine(1.0, 0.46164, -0.5 * PI);
        wt.update_diff_buffer();
        wt
    }
    /// Produces a Sine window
    #[must_use]
    pub fn sine_window() -> Self {
        let mut wt = Wavetable::new();
        // This approach was heavily influenced by the SuperCollider Signal implementation
        wt.add_sine(0.5, 1.0, 0.0);
        wt.update_diff_buffer();
        wt
    }
    /// Fill the wavetable buffer with some value
    pub fn fill(&mut self, value: Sample) {
        for sample in &mut self.buffer {
            *sample = value;
        }
        self.update_diff_buffer();
    }
    /// Add a sine wave with the given parameters to the wavetable. Note that
    /// the frequency is relative to the wavetable. If adding a sine wave of
    /// frequency 2.0 Hz and then playing the wavetable at frequency 200 Hz that
    /// sine wave will sound at 400 Hz.
    pub fn add_sine(&mut self, freq: Sample, amplitude: Sample, phase: Sample) {
        let step = (freq * PI * 2.0) / TABLE_SIZE as f32;
        let mut phase = phase;
        for sample in &mut self.buffer {
            *sample += phase.sin() * amplitude;
            phase += step;
        }
        self.update_diff_buffer();
    }
    /// Add a number of harmonics to the wavetable, starting at frequency `freq`.
    pub fn fill_sine(&mut self, num_harmonics: usize, freq: Sample) {
        for n in 0..num_harmonics {
            let start_phase = 0.0;
            let harmonic_amp = match n {
                0 => 1.0,
                _ => ((num_harmonics - n) as Sample / (num_harmonics) as Sample) * 0.5,
            };
            for i in 0..TABLE_SIZE {
                self.buffer[i] +=
                    ((i as Sample / TABLE_SIZE as f32) * PI * 2.0 * freq * (n + 1) as Sample
                        + start_phase)
                        .sin()
                        * harmonic_amp;
            }
        }
        self.update_diff_buffer();
    }
    /// Add a naive sawtooth wave to the wavetable.
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
        self.update_diff_buffer();
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
        self.update_diff_buffer();
    }
    /// Add noise to the wavetable using [`XOrShift32Rng`], keeping the wavetable within +/- 1.0
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
        self.update_diff_buffer();
    }
    /// Multiply all values of the wavetable by a given amount.
    pub fn multiply(&mut self, mult: Sample) {
        for sample in &mut self.buffer {
            *sample *= mult;
        }
        self.update_diff_buffer();
    }
    /// Normalize the amplitude of the wavetable to 1.0.
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
        self.update_diff_buffer();
    }

    /// Linearly interpolate between the value in between which the phase points.
    /// The phase is assumed to be 0 <= phase < 1
    #[inline]
    #[must_use]
    pub fn get_linear_interp(&self, phase: Phase) -> Sample {
        let index = phase.integer_component();
        let mix = phase.fractional_component_f32();
        self.buffer[index] + self.diff_buffer[index] * mix
    }

    /// Get the closest sample with no interpolation
    #[inline]
    #[must_use]
    pub fn get(&self, phase: Phase) -> Sample {
        unsafe { *self.buffer.get_unchecked(phase.integer_component()) }
    }
}

/// Osciallator with an owned Wavetable
#[derive(Debug, Clone)]
pub struct WavetableOscillatorOwned {
    step: u32,
    phase: Phase,
    wavetable: Wavetable,
    amp: Sample,
}

impl WavetableOscillatorOwned {
    #[must_use]
    pub fn new(wavetable: Wavetable) -> Self {
        WavetableOscillatorOwned {
            step: 0,
            phase: Phase(0),
            wavetable,
            amp: 1.0,
        }
    }
    #[must_use]
    pub fn from_freq(wavetable: Wavetable, sample_rate: Sample, freq: Sample, amp: Sample) -> Self {
        let mut osc = Self::new(wavetable);
        osc.amp = amp;
        osc.step = ((freq / sample_rate) * TABLE_SIZE as f32) as u32;
        osc
    }
    pub fn set_freq(&mut self, freq: Sample, resources: &mut Resources) {
        self.step = (freq as f64 * resources.freq_to_phase_inc) as u32;
    }
    pub fn set_amp(&mut self, amp: Sample) {
        self.amp = amp;
    }
    pub fn reset_phase(&mut self) {
        self.phase.0 = 0;
    }

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
    fn process(&mut self, ctx: GenContext, resources: &mut Resources) -> GenState {
        let output = ctx.outputs.get_channel_mut(0);
        let freq_buf = ctx.inputs.get_channel(0);
        for (&freq, o) in freq_buf.iter().zip(output.iter_mut()) {
            self.set_freq(freq, resources);
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
    fn num_outputs(&self) -> usize {
        1
    }
    fn num_inputs(&self) -> usize {
        1
    }
}

/// Fixed point phase, making use of the TABLE_* constants; compatible with Wavetable
#[derive(Debug, Clone, Copy)]
pub struct Phase(pub u32);

impl Phase {
    #[must_use]
    #[inline]
    pub fn integer_component(&self) -> usize {
        // This will fill with zeroes unless going above 31 bits of shift, in
        // which case it will overflow. The mask will remove anything above the
        // bits we use for our table size, so we don't need 2^16 size tables.
        ((self.0 >> 16) & TABLE_HIGH_MASK) as usize
    }
    /// Returns the fractional component, but as the lower bits of a u32.
    #[must_use]
    #[inline]
    pub fn fractional_component(&self) -> u32 {
        const FRACTIONAL_MASK: u32 = u16::MAX as u32;
        self.0 & FRACTIONAL_MASK
    }
    /// Returns the fractional component of the phase.
    #[must_use]
    #[inline]
    pub fn fractional_component_f32(&self) -> f32 {
        const FRACTIONAL_MASK: u32 = u16::MAX as u32;
        (self.0 & FRACTIONAL_MASK) as f32 / FRACTIONAL_MASK as f32
    }
    /// Increase the phase by the given step. The step should be the
    /// frequency/sample_rate * [`TABLE_SIZE`] * [`FRACTIONAL_PART`]. [`Resources`]
    /// holds a value `freq_to_phase_inc` which, multiplied by the desired
    /// frequency, gives this step value.
    #[inline]
    pub fn increase(&mut self, add: u32) {
        self.0 = self.0.wrapping_add(add);
    }
}

/// Don't use! This is the same as Phase, but stored in an f32. Last time I
/// benchmarked it was significantly slower. I'm leaving it here so that I, next
/// time I wonder if the floating point phase is indeed slower, can run the
/// benchmark and find out.
pub struct PhaseF32(pub f32);

impl PhaseF32 {
    #[inline]
    pub fn index_mix(&self) -> (usize, f32) {
        const TABLE_SIZE_F32: f32 = TABLE_SIZE as f32;
        let value = self.0 * TABLE_SIZE_F32;
        (value as usize, value.fract())
    }
    #[inline]
    pub fn increase(&mut self, add: f32) {
        // This is the fastest version I have found, faster than % 1.0 and .fract()
        self.0 += add;
        while self.0 >= 1.0 {
            self.0 -= 1.0;
        }
    }
}

#[derive(Debug, Clone)]
pub struct Oscillator {
    step: u32,
    phase: Phase,
    wavetable: WavetableKey,
    amp: Sample,
}

impl Oscillator {
    #[must_use]
    pub fn new(wavetable: WavetableKey) -> Self {
        Oscillator {
            step: 0,
            phase: Phase(0),
            wavetable,
            amp: 1.0,
        }
    }
    #[must_use]
    pub fn from_freq(
        wavetable: WavetableKey,
        sample_rate: Sample,
        freq: Sample,
        amp: Sample,
    ) -> Self {
        let mut osc = Oscillator::new(wavetable);
        osc.amp = amp;
        osc.step = ((freq / sample_rate) * TABLE_SIZE as f32) as u32;
        osc
    }
    #[inline]
    pub fn set_freq(&mut self, freq: Sample, resources: &mut Resources) {
        self.step = (freq as f64 * resources.freq_to_phase_inc) as u32;
    }
    #[inline]
    pub fn set_amp(&mut self, amp: Sample) {
        self.amp = amp;
    }
    #[inline]
    pub fn reset_phase(&mut self) {
        self.phase.0 = 0;
    }
    #[inline]
    #[must_use]
    fn next(&mut self, resources: &mut Resources) -> Sample {
        // Use the phase to index into the wavetable
        let sample = if let Some(wt) = resources.wavetables.get(self.wavetable) {
            wt.get(self.phase) * self.amp
        } else {
            eprintln!("Wavetable doesn't exist: {:?}", self.wavetable);
            0.0
        };
        self.phase.increase(self.step);
        sample
    }
}
impl Gen for Oscillator {
    fn process(&mut self, ctx: GenContext, resources: &mut Resources) -> GenState {
        let output = ctx.outputs.get_channel_mut(0);
        let freq_buf = ctx.inputs.get_channel(0);
        for (&freq, o) in freq_buf.iter().zip(output.iter_mut()) {
            self.set_freq(freq, resources);
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
