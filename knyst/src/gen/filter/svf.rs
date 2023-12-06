use knyst_macro::impl_gen;

use crate::prelude::GenState;
use crate::{Sample, SampleRate};
use crate as knyst;

#[derive(Clone, Copy, Debug)]
pub enum SvfFilterType {
    Low,
    High,
    Band,
    Notch,
    Peak,
    All,
    Bell,
    LowShelf,
    HighShelf,
}
pub struct GeneralSvfFilter {
    ty: SvfFilterType,
    cutoff_freq: Sample,
    q: Sample,
    gain_db: Sample,
    // state
    ic1eq: Sample,
    ic2eq: Sample,
    // coefficients
    a1: Sample,
    a2: Sample,
    a3: Sample,
    m0: Sample,
    m1: Sample,
    m2: Sample,
}

#[impl_gen]
impl GeneralSvfFilter {
    pub fn new(ty: SvfFilterType, cutoff_freq: Sample, q: Sample, gain_db: Sample) -> Self {
        Self {
            ic1eq: 0.,
            ic2eq: 0.,
            a1: 0.,
            a2: 0.,
            a3: 0.,
            m0: 0.,
            m1: 0.,
            m2: 0.,
            ty,
            cutoff_freq,
            q,
            gain_db,
        }
    }
    pub fn set_coeffs(&mut self, cutoff: Sample, q: Sample, gain_db: Sample, sample_rate: Sample) {
        match self.ty {
            SvfFilterType::Low => {
                let g = ((std::f64::consts::PI as Sample * cutoff) / sample_rate).tan();
                let k = 1.0 / q;
                self.a1 = 1.0 / (1.0 + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = 0.;
                self.m1 = 0.;
                self.m2 = 1.;
            }
            SvfFilterType::Band => {
                let g = ((std::f64::consts::PI as Sample * cutoff) / sample_rate).tan();
                let k = 1.0 / q;
                self.a1 = 1.0 / (1.0 + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = 0.;
                self.m1 = 1.;
                self.m2 = 0.;
            }
            SvfFilterType::High => {
                let g = ((std::f64::consts::PI as Sample * cutoff) / sample_rate).tan();
                let k = 1.0 / q;
                self.a1 = 1.0 / (1.0 + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = 1.;
                self.m1 = -k;
                self.m2 = -1.;
            }
            SvfFilterType::Notch => {
                let g = ((std::f64::consts::PI as Sample * cutoff) / sample_rate).tan();
                let k = 1.0 / q;
                self.a1 = 1.0 / (1.0 + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = 1.;
                self.m1 = -k;
                self.m2 = 0.;
            }
            SvfFilterType::Peak => {
                let g = ((std::f64::consts::PI as Sample * cutoff) / sample_rate).tan();
                let k = 1.0 / q;
                self.a1 = 1.0 / (1.0 + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = 1.;
                self.m1 = -k;
                self.m2 = -2.;
            }
            SvfFilterType::All => {
                let g = ((std::f64::consts::PI as Sample * cutoff) / sample_rate).tan();
                let k = 1.0 / q;
                self.a1 = 1.0 / (1.0 + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = 1.;
                self.m1 = -2. * k;
                self.m2 = 0.;
            }
            SvfFilterType::Bell => {
                let amp = (10.0 as Sample).powf(gain_db / 40.);
                let g =
                    ((std::f64::consts::PI as Sample * cutoff) / sample_rate).tan() / amp.sqrt();
                let k = 1.0 / (q * amp);
                self.a1 = 1.0 / (1.0 + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = 1.;
                self.m1 = k * (amp * amp - 1.);
                self.m2 = 0.;
            }
            SvfFilterType::LowShelf => {
                let amp = (10.0 as Sample).powf(gain_db / 40.);
                let g =
                    ((std::f64::consts::PI as Sample * cutoff) / sample_rate).tan() / amp.sqrt();
                let k = 1.0 / q;
                self.a1 = 1.0 / (1.0 + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = 1.;
                self.m1 = k * (amp - 1.);
                self.m2 = amp * amp - 1.;
            }
            SvfFilterType::HighShelf => {
                let amp = (10.0 as Sample).powf(gain_db / 40.);
                let g =
                    ((std::f64::consts::PI as Sample * cutoff) / sample_rate).tan() * amp.sqrt();
                let k = 1.0 / q;
                self.a1 = 1.0 / (1.0 + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = amp * amp;
                self.m1 = k * (1. - amp) * amp;
                self.m2 = 1.0 - amp * amp;
            }
        }
    }
    pub fn init(&mut self, sample_rate: SampleRate) {
      self.set_coeffs(self.cutoff_freq, self.q, self.gain_db, *sample_rate);
    }
    // TODO: This is vectorisable such that multiple filters can be run at once, e.g. multiple channels with the same coefficients
    pub fn process_sample(&mut self, v0: Sample) -> Sample {
        let GeneralSvfFilter {
            ic1eq,
            ic2eq,
            a1,
            a2,
            a3,
            m0,
            m1,
            m2,
            ..
        } = self;
        let v3 = v0 - *ic2eq;
        let v1 = *a1 * *ic1eq + *a2 * v3;
        let v2 = *ic2eq + *a2 * *ic1eq + *a3 * v3;
        *ic1eq = 2. * v1 - *ic1eq;
        *ic2eq = 2. * v2 - *ic2eq;

        *m0 * v0 + *m1 * v1 + *m2 * v2
    }
    pub fn process(&mut self, input: &[Sample], output: &mut [Sample]) -> GenState {
      for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = self.process_sample(*inp);
      }
      GenState::Continue
    }
}
