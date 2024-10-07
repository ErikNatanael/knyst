//! Wavetable synthesis

use crate::wavetable::{WavetablePhase, TABLE_SIZE};
use crate::Sample;

// use std::f64::consts::PI;
use crate::xorrng::XOrShift32Rng;
use std::f64::consts::PI;

/// Holds a table with some partial range of harmonics of the full waveform
#[derive(Debug, Clone)]
struct PartialTable {
    buffer: Vec<Sample>,      // Box<[Sample; 131072]>,
    diff_buffer: Vec<Sample>, // Box<[Sample; 131072]>,
}

impl Default for PartialTable {
    fn default() -> Self {
        let buffer = vec![0.0; TABLE_SIZE];
        let diff_buffer = vec![0.0; TABLE_SIZE];
        Self {
            buffer,
            diff_buffer,
        }
    }
}
impl PartialTable {
    pub fn new() -> Self {
        Self::default()
    }
    /// Recalculate the difference between samples in the buffer.
    ///
    /// The [`PartialTable`] contains a buffer with the difference between each
    /// sample of the buffer for efficiency reason.
    pub fn update_diff_buffer(&mut self) {
        // let diff_buffer: Vec<Sample> = self
        //     .buffer
        //     .iter()
        //     .zip(self.buffer.iter().skip(1).cycle())
        //     .map(|(&a, &b)| a - b)
        //     .collect();
        let mut diff_buffer = vec![0.0; self.buffer.len()];
        for (i, diff) in diff_buffer.iter_mut().enumerate() {
            *diff = self.buffer[(i + 1) % self.buffer.len()] - self.buffer[i];
        }
        assert_eq!(self.buffer[1] - self.buffer[0], diff_buffer[0]);
        assert_eq!(
            self.buffer[0] - self.buffer.iter().last().unwrap(),
            *diff_buffer.iter().last().unwrap()
        );
        self.diff_buffer = diff_buffer;
    }
    /// Create a [`PartialTable`] from an existing buffer.
    ///
    /// # Errors
    /// The buffer has to be of [`TABLE_SIZE`] length, otherwise an error will be returned.
    pub fn set_from_buffer(&mut self, buffer: Vec<Sample>) {
        // TODO: FFT of the buffer for anti-aliasing
        self.buffer = buffer;
        self.update_diff_buffer();
    }
    /// Create a new wavetable containing a sine wave. For audio, you often want a cosine instead since it starts at 0 to avoid discontinuities.
    #[must_use]
    pub fn sine() -> Self {
        let wavetable_size = TABLE_SIZE;
        let mut wt = Self::new();
        // Fill buffer with a sine
        for i in 0..wavetable_size {
            wt.buffer[i] = ((i as f64 / TABLE_SIZE as f64) * PI * 2.0).sin() as Sample;
        }
        wt.update_diff_buffer();
        wt
    }
    /// Create a new wavetable containing a cosine wave.
    #[must_use]
    pub fn cosine() -> Self {
        let wavetable_size = TABLE_SIZE;
        let mut wt = Self::new();
        // Fill buffer with a sine
        for i in 0..wavetable_size {
            wt.buffer[i] = ((i as f64 / TABLE_SIZE as f64) * PI * 2.0).cos() as Sample;
        }
        wt.update_diff_buffer();
        wt
    }
    /// Create a new wavetable containing an aliasing sawtooth wave
    #[must_use]
    pub fn aliasing_saw() -> Self {
        let wavetable_size = TABLE_SIZE;
        let mut wt = Self::new();
        // Fill buffer with a sine
        let per_sample = 2.0 / wavetable_size as Sample;
        for i in 0..wavetable_size {
            wt.buffer[i] = -1. + per_sample * i as Sample;
        }
        wt.update_diff_buffer();
        wt
    }
    // #[must_use]
    // pub fn crazy(seed: u32) -> Self {
    //     let wavetable_size = TABLE_SIZE;
    //     let mut wt = Wavetable::new();
    //     let mut xorrng = XOrShift32Rng::new(seed);
    //     wt.fill_sine(16, 1.0);
    //     for _ in 0..(xorrng.gen_u32() % 3 + 1) {
    //         wt.fill_sine(16, (xorrng.gen_f32() * 32.0).floor());
    //     }
    //     wt.add_noise(1.0 - xorrng.gen_f64() * 0.05, seed + wavetable_size as u32);
    //     wt.normalize();
    //     wt.update_diff_buffer();
    //     wt
    // }
    #[must_use]
    /// Produces a Hann window
    pub fn hann_window() -> Self {
        let mut wt = Self::new();
        // This approach was heavily influenced by the SuperCollider Signal implementation
        wt.fill(0.5);
        wt.add_sine(1.0, 0.5, -0.5 * PI as Sample);
        wt.update_diff_buffer();
        wt
    }
    /// Produces a Hamming window
    #[must_use]
    pub fn hamming_window() -> Self {
        let mut wt = Self::new();
        // This approach was heavily influenced by the SuperCollider Signal implementation
        wt.fill(0.53836);
        wt.add_sine(1.0, 0.46164, -0.5 * PI as Sample);
        wt.update_diff_buffer();
        wt
    }
    /// Produces a Sine window
    #[must_use]
    pub fn sine_window() -> Self {
        let mut wt = Self::new();
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
        let step = (freq * PI as Sample * 2.0) / TABLE_SIZE as Sample;
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
            let harmonic_freq = freq * (n + 1) as Sample;
            for i in 0..TABLE_SIZE {
                self.buffer[i] +=
                    ((i as Sample / TABLE_SIZE as Sample) * PI as Sample * 2.0 * harmonic_freq
                        + start_phase)
                        .sin()
                        * harmonic_amp;
            }
        }
        self.update_diff_buffer();
    }
    /// Add a naive sawtooth wave to the wavetable.
    pub fn add_saw(&mut self, start_harmonic: usize, end_harmonic: usize, amp: Sample) {
        for i in start_harmonic..=end_harmonic {
            let start_phase = 0.0;
            let harmonic_amp = 1.0 / ((i + 1) as Sample * PI as Sample);
            for k in 0..self.buffer.len() {
                self.buffer[k] += ((k as Sample / self.buffer.len() as Sample
                    * PI as Sample
                    * 2.0
                    * (i + 1) as Sample
                    + start_phase)
                    .sin()
                    * harmonic_amp)
                    * amp;
            }
        }
        self.update_diff_buffer();
    }
    /// Add a number of odd harmonics to the wavetable. `amp_falloff` is the
    /// exponential falloff as we go to higher harmonics, a value of 0.0 is no
    /// falloff.
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
                    * PI as Sample
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
    /// TODO: anti-aliasing
    pub fn add_noise(&mut self, probability: f64, seed: u32) {
        let mut xorrng = XOrShift32Rng::new(seed);
        for sample in &mut self.buffer {
            if xorrng.gen_f64() > probability {
                *sample += xorrng.gen_f32() as Sample - 0.5;
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

    /// Linearly interpolate between the value in between which the phase points.
    /// The phase is assumed to be 0 <= phase < 1
    #[inline]
    #[must_use]
    pub fn get_linear_interp(&self, phase: WavetablePhase) -> Sample {
        let index = phase.integer_component();
        let mix = phase.fractional_component_f32() as Sample;
        self.buffer[index] + self.diff_buffer[index] * mix
    }

    /// Get the closest sample with no interpolation
    #[inline]
    #[must_use]
    pub fn get(&self, phase: WavetablePhase) -> Sample {
        unsafe { *self.buffer.get_unchecked(phase.integer_component()) }
    }
}

const TABLE_AA_SPACING: Sample = 1.5;
/// Converts a certain frequency to the corresponding wavetable
fn freq_to_table_index(freq: f32) -> usize {
    // let mut index = 0;
    // let mut freq = freq;
    // loop {
    //     if freq < 32. {
    //         return index;
    //     }
    //     freq /= TABLE_AA_SPACING;
    //     index += 1;
    // }

    // For TABLE_AA_SPACING == 1.5
    let f = freq;
    if f <= 32.0 {
        0
    } else if f <= 48.0 {
        1
    } else if f <= 72.0 {
        2
    } else if f <= 108.0 {
        3
    } else if f <= 162.0 {
        4
    } else if f <= 243.0 {
        5
    } else if f <= 364.5 {
        6
    } else if f <= 546.75 {
        7
    } else if f <= 820.125 {
        8
    } else if f <= 1230.1875 {
        9
    } else if f <= 1845.2813 {
        10
    } else if f <= 2767.9219 {
        11
    } else if f <= 4151.883 {
        12
    } else if f <= 6227.824 {
        13
    } else if f <= 9341.736 {
        14
    } else if f <= 14012.6045 {
        15
    } else {
        16
    }
}
fn table_index_to_max_freq_produced(index: usize) -> Sample {
    32. * TABLE_AA_SPACING.powi(index as i32)
}
fn table_index_to_max_harmonic(index: usize) -> usize {
    // The higher this freq, the lower the number of harmonics
    let max_freq_produced = table_index_to_max_freq_produced(index);
    let max_harmonic_freq = 20000.0;
    (max_harmonic_freq / max_freq_produced) as usize
}

/// Wavetable is a standardised wavetable with a buffer of samples, as well as a
/// separate buffer with the difference between the current sample and the next.
/// The wavetable is of size [`TABLE_SIZE`] and can be indexed using a [`WavetablePhase`].
///
/// It is not safe to modify the wavetable while it is being used on the audio
/// thread, even if no Node is currently reading from it, because most modifying
/// operations may allocate.
#[derive(Debug, Clone)]
pub struct Wavetable {
    partial_tables: Vec<PartialTable>,
}

impl Default for Wavetable {
    fn default() -> Self {
        let num_tables = freq_to_table_index(20000.0) + 1;
        Wavetable {
            partial_tables: vec![PartialTable::default(); num_tables],
        }
    }
}

impl Wavetable {
    /// Create an empyu wavetable
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Recalculate the difference between samples in the buffer.
    ///
    /// The [`Wavetable`] contains a buffer with the difference between each
    /// sample of the buffer for efficiency reason.
    pub fn update_diff_buffer(&mut self) {
        for table in &mut self.partial_tables {
            table.update_diff_buffer()
        }
    }
    /// Create a [`Wavetable`] from an existing buffer. TODO: anti-aliasing
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
        // TODO: FFT of the buffer for anti-aliasing
        let mut s = Self::default();
        for table in &mut s.partial_tables {
            table.set_from_buffer(buffer.clone());
        }
        Ok(s)
    }
    /// Create a new [`Wavetable`] and populate it using the closure/function provided. TODO: anti-aliasing
    #[must_use]
    pub fn from_closure<F>(f: F) -> Self
    where
        F: FnOnce(&mut [Sample]),
    {
        let mut w = Self::default();
        let mut buffer = vec![0.0; TABLE_SIZE];
        f(&mut buffer);
        // TODO: FFT of buffer for anti-aliasing
        for table in &mut w.partial_tables {
            table.set_from_buffer(buffer.clone());
        }
        w
    }
    /// Create a new wavetable containing a sine wave. For audio, you often want a cosine instead since it starts at 0 to avoid discontinuities.
    #[must_use]
    pub fn sine() -> Self {
        let mut wt = Wavetable::new();
        for table in &mut wt.partial_tables {
            *table = PartialTable::sine();
        }
        wt
    }
    /// Create a new wavetable containing a cosine wave.
    #[must_use]
    pub fn cosine() -> Self {
        let mut wt = Wavetable::new();
        for table in &mut wt.partial_tables {
            *table = PartialTable::cosine();
        }
        wt
    }
    /// Create a new wavetable containing an aliasing sawtooth wave
    #[must_use]
    pub fn aliasing_saw() -> Self {
        let mut wt = Wavetable::new();
        for table in &mut wt.partial_tables {
            *table = PartialTable::aliasing_saw();
        }
        wt
    }
    // #[must_use]
    // pub fn crazy(seed: u32) -> Self {
    //     let wavetable_size = TABLE_SIZE;
    //     let mut wt = Wavetable::new();
    //     let mut xorrng = XOrShift32Rng::new(seed);
    //     wt.fill_sine(16, 1.0);
    //     for _ in 0..(xorrng.gen_u32() % 3 + 1) {
    //         wt.fill_sine(16, (xorrng.gen_f32() * 32.0).floor());
    //     }
    //     wt.add_noise(1.0 - xorrng.gen_f64() * 0.05, seed + wavetable_size as u32);
    //     wt.normalize();
    //     wt.update_diff_buffer();
    //     wt
    // }
    #[must_use]
    /// Produces a Hann window
    pub fn hann_window() -> Self {
        let mut wt = Wavetable::new();
        for table in &mut wt.partial_tables {
            *table = PartialTable::hann_window();
        }
        wt
    }
    /// Produces a Hamming window
    #[must_use]
    pub fn hamming_window() -> Self {
        let mut wt = Wavetable::new();
        for table in &mut wt.partial_tables {
            *table = PartialTable::hamming_window();
        }
        wt
    }
    /// Produces a Sine window
    #[must_use]
    pub fn sine_window() -> Self {
        let mut wt = Wavetable::new();
        for table in &mut wt.partial_tables {
            *table = PartialTable::sine_window();
        }
        wt
    }
    /// Fill the wavetable buffer with some value
    pub fn fill(&mut self, value: Sample) {
        for table in &mut self.partial_tables {
            table.fill(value);
        }
    }
    /// Add a sine wave with the given parameters to the wavetable. Note that
    /// the frequency is relative to the wavetable. If adding a sine wave of
    /// frequency 2.0 Hz and then playing the wavetable at frequency 200 Hz that
    /// sine wave will sound at 400 Hz.
    pub fn add_sine(&mut self, freq: Sample, amplitude: Sample, phase: Sample) {
        for (i, table) in self.partial_tables.iter_mut().enumerate() {
            if freq.ceil() as usize <= table_index_to_max_harmonic(i) {
                table.add_sine(freq, amplitude, phase);
            }
        }
    }
    /// Add a number of harmonics to the wavetable, starting at frequency `freq`.
    pub fn fill_sine(&mut self, num_harmonics: usize, freq: Sample) {
        for (i, table) in self.partial_tables.iter_mut().enumerate() {
            table.fill_sine(
                num_harmonics.min((table_index_to_max_harmonic(i) as Sample * freq) as usize),
                freq,
            );
        }
    }
    /// Add a naive sawtooth wave to the wavetable.
    pub fn add_aliasing_saw(&mut self, num_harmonics: usize, amp: Sample) {
        for (i, table) in self.partial_tables.iter_mut().enumerate() {
            table.add_saw(0, num_harmonics.min(table_index_to_max_harmonic(i)), amp);
        }
    }
    /// Add a sawtooth wave starting from a specific harmonic
    pub fn add_saw(&mut self, start_harmonic: usize, end_harmonic: usize, amp: Sample) {
        for (i, table) in self.partial_tables.iter_mut().enumerate() {
            let end_harmonic = end_harmonic.min(table_index_to_max_harmonic(i));
            if end_harmonic > start_harmonic {
                table.add_saw(start_harmonic, end_harmonic, amp);
            }
        }
    }
    /// Add a number of odd harmonics to the wavetable. `amp_falloff` is the
    /// exponential falloff as we go to higher harmonics, a value of 0.0 is no
    /// falloff.
    pub fn add_odd_harmonics(&mut self, num_harmonics: usize, amp_falloff: Sample) {
        for (i, table) in self.partial_tables.iter_mut().enumerate() {
            table.add_odd_harmonics(
                num_harmonics.min(table_index_to_max_harmonic(i)),
                amp_falloff,
            );
        }
    }
    /// Add noise to the wavetable using [`XOrShift32Rng`], keeping the wavetable within +/- 1.0
    pub fn add_noise(&mut self, probability: f64, seed: u32) {
        for table in self.partial_tables.iter_mut() {
            table.add_noise(probability, seed);
        }
    }
    /// Multiply all values of the wavetable by a given amount.
    pub fn multiply(&mut self, mult: Sample) {
        for table in self.partial_tables.iter_mut() {
            table.multiply(mult);
        }
    }
    /// Normalize the amplitude of the wavetable to 1.0 based on the wavetable most rich in harmonics. Interference from high partials out of phase could normalize high pitch tables to more or less than 1.0.
    pub fn normalize(&mut self) {
        // Find highest absolute value
        let mut loudest_sample = 0.0;
        for sample in &self.partial_tables[0].buffer {
            if sample.abs() > loudest_sample {
                loudest_sample = sample.abs();
            }
        }
        // Scale all tables by the same amount
        let scaler = 1.0 / loudest_sample;
        for table in self.partial_tables.iter_mut() {
            table.multiply(scaler);
        }
    }

    /// Linearly interpolate between the value in between which the phase points.
    /// The phase is assumed to be 0 <= phase < 1
    #[inline]
    #[must_use]
    pub fn get_linear_interp(&self, phase: WavetablePhase, freq: Sample) -> Sample {
        let table_index = freq_to_table_index(freq);
        self.partial_tables[table_index.min(self.partial_tables.len())].get_linear_interp(phase)
    }

    /// Get the closest sample with no interpolation
    #[inline]
    #[must_use]
    pub fn get(&self, phase: WavetablePhase, freq: Sample) -> Sample {
        let table_index = freq_to_table_index(freq);
        self.partial_tables[table_index.min(self.partial_tables.len())].get(phase)
    }
}

#[cfg(test)]
mod tests {
    use crate::wavetable_aa::{table_index_to_max_freq_produced, table_index_to_max_harmonic};

    use super::freq_to_table_index;

    #[test]
    fn table_nr_from_freq() {
        freq_to_table_index(0.0);
        freq_to_table_index(20.0);
        freq_to_table_index(20000.0);
        dbg!(freq_to_table_index(0.0));
        dbg!(freq_to_table_index(20.0));
        dbg!(freq_to_table_index(20000.0));
        let max_index = freq_to_table_index(20050.) + 1;
        println!("Max freq produced:");
        for i in 0..max_index {
            dbg!(table_index_to_max_freq_produced(i));
        }
        println!("Num harmonics for table:");
        for i in 0..max_index {
            println!(
                "{i}: max_harmonics: {} max_freq: {}",
                table_index_to_max_harmonic(i),
                table_index_to_max_freq_produced(i)
            );
        }
        assert!(table_index_to_max_freq_produced(freq_to_table_index(20000.)) >= 20000.);
        assert!(table_index_to_max_freq_produced(freq_to_table_index(20.)) >= 20.);
        assert!(table_index_to_max_freq_produced(freq_to_table_index(200.)) >= 200.);
    }
}
