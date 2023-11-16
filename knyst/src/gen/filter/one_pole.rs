//! One pole filters make good and cheap lowpass 6dB/octave rolloff filters.
//! It is also good for removing zipping from parameter changes.

use crate as knyst;
use knyst_macro::impl_gen;

use crate::{prelude::GenState, Sample, SampleRate};
// To use it as a DC blocker:
//
// `OnePole *dcBlockerLp = new OnePole(10.0 / sampleRate);`
// for each sample:
// `sample -= dcBlockerLp->process(sample);`
/// Internal functionality for the `OnePoleLpf` and `OnePoleHpf` Gens.
#[derive(Debug, Clone, Copy)]
pub struct OnePole<T> {
    last_output: T,
    a0: T,
    b1: T,
    hp: bool,
}

impl<
        T: num_traits::Float
            + num_traits::FromPrimitive
            + num_traits::FloatConst
            + num_traits::One
            + num_traits::Zero
            + std::fmt::Display,
    > OnePole<T>
{
    pub fn new() -> Self {
        Self {
            last_output: T::from_f32(0.0).unwrap(),
            a0: T::from_f32(1.0).unwrap(),
            b1: T::from_f32(0.0).unwrap(),
            hp: false,
        }
    }
    pub fn reset(&mut self) {
        self.last_output = T::zero();
    }
    pub fn set_freq_lowpass(&mut self, freq: T, sample_rate: T) {
        // let freq: T = freq
        //     .max(T::zero())
        //     .min(sample_rate * T::from_f32(0.5).unwrap());
        // if freq > sample_rate * T::from_f32(0.5).unwrap() {
        //     println!("OnePole freq out of bounds: {freq}");
        // }
        let f: T = freq / sample_rate;
        let b_tmp: T = (T::from_f64(-2.0_f64).unwrap() * num_traits::FloatConst::PI() * f).exp();
        self.b1 = b_tmp;
        self.a0 = T::from_f64(1.0_f64).unwrap() - self.b1;
        self.hp = false;
    }
    // TODO: Not verified to set the frequency correctly. In fact, I suspect it doesn't
    pub fn set_freq_highpass(&mut self, freq: T, sample_rate: T) {
        // let x = T::from_f32(2.).unwrap() * FloatConst::PI() * (freq / sample_rate);
        // let p = (T::from_f32(2.).unwrap() + x.cos())
        //     - ((T::from_f32(2.0).unwrap() + x.cos()).powi(2) - T::one()).sqrt();
        // self.b1 = p * T::from_f32(-1.0).unwrap();
        // self.a0 = p - T::one();
        // self.set_freq_lowpass(freq, sample_rate);
        // self.a0 = self.b1 - T::one();
        // self.b1 = self.b1 * T::from_f32(-1.0).unwrap();
        self.set_freq_lowpass(freq, sample_rate);
        self.hp = true;
    }
    pub fn process(&mut self, input: T) -> T {
        self.last_output = input * self.a0 + self.last_output * self.b1;
        if self.hp {
            input - self.last_output
        } else {
            self.last_output
        }
    }
    pub fn process_lp(&mut self, input: T) -> T {
        self.last_output = input * self.a0 + self.last_output * self.b1;
        self.last_output
    }
    pub fn process_hp(&mut self, input: T) -> T {
        self.last_output = input * self.a0 + self.last_output * self.b1;
        input - self.last_output
    }
    pub fn cheap_tuning_compensation_lpf(&self) -> T {
        T::from_f32(-2.).unwrap() * (T::one() - self.b1).ln()
    }
    pub fn phase_delay(fstringhz: T, fcutoffhz: T) -> T {
        fstringhz.atan2(fcutoffhz) * T::from_f32(-1.).unwrap()
    }
}
pub struct OnePoleLpf {
    pub op: OnePole<f64>,
    last_freq: f32,
}
#[impl_gen]
impl OnePoleLpf {
    pub fn new() -> Self {
        let mut op = OnePole::new();
        op.set_freq_lowpass(19000., 44100.);
        Self {
            op,
            last_freq: 19000.,
        }
    }
    pub fn process(
        &mut self,
        sample_rate: SampleRate,
        sig: &[Sample],
        cutoff_freq: &[Sample],
        output: &mut [Sample],
    ) -> GenState {
        for ((&i, &cutoff), o) in sig.iter().zip(cutoff_freq.iter()).zip(output.iter_mut()) {
            if cutoff != self.last_freq {
                self.op.set_freq_lowpass(cutoff as f64, *sample_rate as f64);
                self.last_freq = cutoff;
            }
            *o = self.op.process_lp(i as f64) as f32;
        }
        GenState::Continue
    }
}
pub struct OnePoleHpf {
    pub op: OnePole<f64>,
    last_freq: f32,
}
#[impl_gen]
impl OnePoleHpf {
    pub fn new() -> Self {
        let mut op = OnePole::new();
        op.set_freq_highpass(19000., 44100.);
        Self {
            op,
            last_freq: 19000.,
        }
    }
    fn process(
        &mut self,
        sample_rate: SampleRate,
        sig: &[Sample],
        cutoff_freq: &[Sample],
        output: &mut [Sample],
    ) -> GenState {
        for ((&i, &cutoff), o) in sig.iter().zip(cutoff_freq.iter()).zip(output.iter_mut()) {
            if cutoff != self.last_freq {
                self.op
                    .set_freq_highpass(cutoff as f64, *sample_rate as f64);
                self.last_freq = cutoff;
            }
            *o = self.op.process_hp(i as f64) as f32;
        }
        GenState::Continue
    }
}
