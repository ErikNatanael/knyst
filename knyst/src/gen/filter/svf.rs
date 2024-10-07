//! SVF filter for all your EQ needs
//!
//! Implemented based on [a technical paper by Andrew Simper, Cytomic, 2013](https://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf) also available at <https://cytomic.com/technical-papers/>
//!
use knyst_macro::impl_gen;

use crate as knyst;
use crate::prelude::GenState;
use crate::{Sample, SampleRate};

/// Different supported filter types
#[derive(Clone, Copy, Debug)]
pub enum SvfFilterType {
    #[allow(missing_docs)]
    Low,
    #[allow(missing_docs)]
    High,
    #[allow(missing_docs)]
    Band,
    #[allow(missing_docs)]
    Notch,
    #[allow(missing_docs)]
    Peak,
    #[allow(missing_docs)]
    All,
    #[allow(missing_docs)]
    Bell,
    #[allow(missing_docs)]
    LowShelf,
    #[allow(missing_docs)]
    HighShelf,
}
/// A versatile EQ filter implementation
///
/// Implemented based on [a technical paper by Andrew Simper, Cytomic, 2013](https://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf) also available at <https://cytomic.com/technical-papers/>
#[derive(Clone, Debug)]
pub struct SvfFilter {
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
impl SvfFilter {
    #[allow(missing_docs)]
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
    /// Set the coefficients for the currently set filter type. `gain_db` is only used for Bell, HighShelf and LowShelf.
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
    #[allow(missing_docs)]
    pub fn init(&mut self, sample_rate: SampleRate) {
        self.set_coeffs(self.cutoff_freq, self.q, self.gain_db, *sample_rate);
    }
    // TODO: This is vectorisable such that multiple filters can be run at once, e.g. multiple channels with the same coefficients
    #[allow(missing_docs)]
    pub fn process_sample(&mut self, v0: Sample) -> Sample {
        let SvfFilter {
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
    #[allow(missing_docs)]
    pub fn process(&mut self, input: &[Sample], output: &mut [Sample]) -> GenState {
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = self.process_sample(*inp);
        }
        GenState::Continue
    }
}

/// Svf high shelf filter with controllable settings.
///
/// Uses the [`SvfFilter`] under the hood, but allows for setting the filter parameters at block rate
pub struct SvfHighShelf {
    svf: SvfFilter,
    last_cutoff: Sample,
    last_gain: Sample,
    last_q: Sample,
}
#[impl_gen]
impl SvfHighShelf {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {
            svf: SvfFilter::new(SvfFilterType::HighShelf, 20000., 1.0, 0.0),
            last_cutoff: 20000.,
            last_gain: 0.0,
            last_q: 1.0,
        }
    }
    #[allow(missing_docs)]
    pub fn process(
        &mut self,
        input: &[Sample],
        cutoff_freq: &[Sample],
        gain: &[Sample],
        q: &[Sample],
        output: &mut [Sample],
        sample_rate: SampleRate,
    ) -> GenState {
        if cutoff_freq[0] != self.last_cutoff || gain[0] != self.last_gain || q[0] != self.last_q {
            self.svf
                .set_coeffs(cutoff_freq[0], q[0], gain[0], *sample_rate);
        }
        self.svf.process(input, output);
        GenState::Continue
    }
}

/// Svf filter with controllable settings.
///
/// Uses the [`SvfFilter`] under the hood, but allows for setting the filter parameters at block rate
pub struct SvfDynamic {
    svf: SvfFilter,
    last_cutoff: Sample,
    last_gain: Sample,
    last_q: Sample,
}
#[impl_gen]
impl SvfDynamic {
    #[allow(missing_docs)]
    pub fn new(ty: SvfFilterType) -> Self {
        Self {
            svf: SvfFilter::new(ty, 2000., 1.0, 0.0),
            last_cutoff: 20000.,
            last_gain: 0.0,
            last_q: 1.0,
        }
    }
    #[allow(missing_docs)]
    pub fn process(
        &mut self,
        input: &[Sample],
        cutoff_freq: &[Sample],
        gain: &[Sample],
        q: &[Sample],
        output: &mut [Sample],
        sample_rate: SampleRate,
    ) -> GenState {
        if cutoff_freq[0] != self.last_cutoff || gain[0] != self.last_gain || q[0] != self.last_q {
            // A q value of 0.0 will result in NaN
            let mut safe_q = q[0];
            if safe_q <= 0.0 {
                safe_q = f32::MIN;
            }
            self.svf
                .set_coeffs(cutoff_freq[0], safe_q, gain[0], *sample_rate);
            self.last_cutoff = cutoff_freq[0];
            self.last_gain = gain[0];
            self.last_q = q[0];
        }
        self.svf.process(input, output);
        GenState::Continue
    }
}
