//! Randja compressor
//! ported from <https://www.musicdsp.org/en/latest/Synthesis/272-randja-compressor.html?highlight=compressor>
use crate::{self as knyst, prelude::GenState};
use knyst_macro::impl_gen;

use crate::Sample;

/// Randja compressor
/// ported from <https://www.musicdsp.org/en/latest/Synthesis/272-randja-compressor.html?highlight=compressor>
pub struct RandjaCompressor {
    threshold: Sample,
    attack: Sample,
    release: Sample,
    envelope_decay: Sample,
    transfer_a: Sample,
    transfer_b: Sample,
    env: Sample,
    gain: Sample,
    output: Sample,
}
#[impl_gen]
impl RandjaCompressor {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {
            threshold: 1.,
            attack: 0.,
            release: 0.,
            envelope_decay: 0.,
            output: 1.,
            transfer_a: 0.,
            transfer_b: 1.,
            env: 0.,
            gain: 1.,
        }
    }
    /// Set the gain threshold in amplitude (0.0, 1.0]
    pub fn set_threshold(&mut self, threshold: Sample) -> &mut Self {
        self.threshold = threshold;
        self.transfer_b = self.output * self.threshold.powf(-self.transfer_a);
        self
    }
    /// Sets the ratio; 'value' must be in range [0.0; 1.0] with 0.0
    /// representing a oo:1 ratio, 0.5 a 2:1 ratio; 1.0 a 1:1 ratio and so on.
    pub fn set_ratio(&mut self, ratio: f32) {
        self.transfer_a = ratio - 1.;
        self.transfer_b = self.output * self.threshold.powf(-self.transfer_a);
    }

    /// Sets the attack time; 'value' gives the attack time in _samples_
    pub fn set_attack(&mut self, attack: f32) {
        self.attack = (-1. / attack).exp();
    }
    /// Sets the release time; 'value' gives the release time in _samples_
    pub fn set_release(&mut self, release: Sample) {
        self.release = (-1. / release).exp();
        self.envelope_decay = (-4. / release).exp();
    }
    /// Sets the output gain; 'value' represents the gain, where 0dBFS is 1.0
    /// (see set_threshold())
    pub fn set_output(&mut self, output: Sample) {
        self.output = output;
        self.transfer_b = self.output * self.threshold.powf(-self.transfer_a);
    }
    /// Resets the internal state of the compressor (gain reduction)
    pub fn reset(&mut self) {
        self.env = 0.;
        self.gain = 1.;
    }
    #[allow(missing_docs)]
    pub fn process(
        &mut self,
        input_left: &[Sample],
        input_right: &[Sample],
        output_left: &mut [Sample],
        output_right: &mut [Sample],
    ) -> GenState {
        // for (il, ir, ol, or) in izip!(input_left, input_right, output_left, output_right) {
        for (((il, ir), ol), or) in input_left
            .iter()
            .zip(input_right)
            .zip(output_left.iter_mut())
            .zip(output_right.iter_mut())
        {
            let det = il.abs().max(ir.abs()) + 10e-30;
            self.env = if det >= self.env {
                det
            } else {
                det + self.envelope_decay * (self.env - det)
            };
            let transfer_gain = if self.env > self.threshold {
                self.env.powf(self.transfer_a) * self.transfer_b
            } else {
                self.output
            };
            self.gain = if transfer_gain < self.gain {
                transfer_gain + self.attack * (self.gain - transfer_gain)
            } else {
                transfer_gain + self.release * (self.gain - transfer_gain)
            };
            *ol = il * self.gain;
            *or = ir * self.gain;
        }
        GenState::Continue
    }
}
