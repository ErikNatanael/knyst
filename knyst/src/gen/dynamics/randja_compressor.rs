//! Randja compressor
//! ported from <https://www.musicdsp.org/en/latest/Synthesis/272-randja-compressor.html?highlight=compressor>
use crate::{self as knyst, prelude::GenState};
use itertools::izip;
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
    pub fn set_threshold(&mut self, threshold: Sample) {
        self.threshold = threshold;
        self.transfer_b = self.output * self.threshold.powf(-self.transfer_a);
    }
    pub fn set_ratio(&mut self, ratio: f32) {
        self.transfer_a = ratio - 1.;
        self.transfer_b = self.output * self.threshold.powf(-self.transfer_a);
    }

    pub fn set_attack(&mut self, attack: f32) {
        self.attack = (1. / attack).exp();
    }
    pub fn set_release(&mut self, release: Sample) {
        self.release = (-1. / release).exp();
        self.envelope_decay = (-4. / release).exp();
    }
    pub fn set_output(&mut self, output: Sample) {
        self.output = output;
        self.transfer_b = self.output * self.threshold.powf(-self.transfer_a);
    }
    pub fn reset(&mut self) {
        self.env = 0.;
        self.gain = 1.;
    }
    pub fn process(
        &mut self,
        input_left: &[Sample],
        input_right: &[Sample],
        output_left: &mut [Sample],
        output_right: &mut [Sample],
    ) -> GenState {
        for (il, ir, ol, or) in izip!(input_left, input_right, output_left, output_right) {
            let det = il.abs().max(ir.abs()) + 10e-30;
            self.env = if det >= self.env {
                det
            } else {
                det + self.envelope_decay * (self.env - det)
            };
            let mut transfer_gain = if self.env > self.threshold {
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
/*
#include <cmath>
#define max(a,b) (a>b?a:b)

class compressor
{

    private:
            float   threshold;
            float   attack, release, envelope_decay;
            float   output;
            float   transfer_A, transfer_B;
            float   env, gain;

    public:
    compressor()
    {
            threshold = 1.f;
            attack = release = envelope_decay = 0.f;
            output = 1.f;

            transfer_A = 0.f;
            transfer_B = 1.f;

            env = 0.f;
            gain = 1.f;
    }

    void set_threshold(float value)
    {
            threshold = value;
            transfer_B = output * pow(threshold,-transfer_A);
    }


    void set_ratio(float value)
    {
            transfer_A = value-1.f;
            transfer_B = output * pow(threshold,-transfer_A);
    }


    void set_attack(float value)
    {
            attack = exp(-1.f/value);
    }


    void et_release(float value)
    {
            release = exp(-1.f/value);
            envelope_decay = exp(-4.f/value); /* = exp(-1/(0.25*value)) */
    }


    void set_output(float value)
    {
            output = value;
            transfer_B = output * pow(threshold,-transfer_A);
    }


    void reset()
    {
            env = 0.f; gain = 1.f;
    }


    __forceinline void process(float *input_left, float *input_right,float *output_left, float *output_right,       int frames)
    {
            float det, transfer_gain;
            for(int i=0; i<frames; i++)
            {
                    det = max(fabs(input_left[i]),fabs(input_right[i]));
                    det += 10e-30f; /* add tiny DC offset (-600dB) to prevent denormals */

                    env = det >= env ? det : det+envelope_decay*(env-det);

                    transfer_gain = env > threshold ? pow(env,transfer_A)*transfer_B:output;

                    gain = transfer_gain < gain ?
                                                    transfer_gain+attack *(gain-transfer_gain):
                                                    transfer_gain+release*(gain-transfer_gain);

                    output_left[i] = input_left[i] * gain;
                    output_right[i] = input_right[i] * gain;
            }
    }


    __forceinline void process(double *input_left, double *input_right,     double *output_left, double *output_right,int frames)
    {
            double det, transfer_gain;
            for(int i=0; i<frames; i++)
            {
                    det = max(fabs(input_left[i]),fabs(input_right[i]));
                    det += 10e-30f; /* add tiny DC offset (-600dB) to prevent denormals */

                    env = det >= env ? det : det+envelope_decay*(env-det);

                    transfer_gain = env > threshold ? pow(env,transfer_A)*transfer_B:output;

                    gain = transfer_gain < gain ?
                                                    transfer_gain+attack *(gain-transfer_gain):
                                                    transfer_gain+release*(gain-transfer_gain);

                    output_left[i] = input_left[i] * gain;
                    output_right[i] = input_right[i] * gain;
            }
    }

};
*/
