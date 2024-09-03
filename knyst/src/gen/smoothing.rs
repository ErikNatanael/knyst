use crate as knyst;
use crate::{gen::GenState, BlockSize, Sample, SampleRate};
use knyst_macro::impl_gen;

/// Move to the target value exponentially by -60db attenuation over the specified time.
/// *Inputs*
/// 0. "value"
/// 1. "time" in seconds
///
/// *Outputs*
/// 0. "smoothed_value"
#[derive(Default)]
pub struct Lag {
    // Compare with the current value. If there is change, recalculate the mix.
    last_time: Sample,
    current_value: Sample,
    mix: Sample,
    sample_rate: Sample,
}

#[impl_gen]
impl Lag {
    #[new]
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self::default()
    }
    #[process]
    #[allow(missing_docs)]
    pub fn process(
        &mut self,
        value: &[Sample],
        time: &[Sample],
        smoothed_value: &mut [Sample],
        block_size: BlockSize,
    ) -> GenState {
        for i in 0..*block_size {
            let value = value[i];
            let time = time[i];
            if time != self.last_time {
                self.last_time = time;
                let num_samples = (time * self.sample_rate).floor();
                self.mix = 1.0 - (0.001 as Sample).powf(1.0 / num_samples);
            }
            self.current_value += (value - self.current_value) * self.mix;
            smoothed_value[i] = self.current_value;
        }
        GenState::Continue
    }
    #[init]
    #[allow(missing_docs)]
    pub fn init(&mut self, sample_rate: SampleRate) {
        self.sample_rate = *sample_rate;
    }
}
/// When input 0 changes, move smoothly to the new value over the time in seconds given by input 1.
#[derive(Default)]
pub struct Ramp {
    // Compare with the current value. If there is change, recalculate the step.
    last_value: Sample,
    // Compare with the current value. If there is change, recalculate the step.
    last_time: Sample,
    current_value: Sample,
    step: Sample,
    num_steps_to_go: usize,
}

impl Ramp {}
#[impl_gen]
impl Ramp {
    #[new]
    #[allow(missing_docs)]
    #[must_use]
    pub fn new(start_value: Sample) -> Self {
        Self {
            last_value: start_value,
            last_time: 1.0,
            current_value: start_value,
            step: 0.,
            num_steps_to_go: 0,
        }
    }
    #[process]
    #[allow(missing_docs)]
    pub fn process(
        &mut self,
        value: &[Sample],
        time: &[Sample],
        ramped_value: &mut [Sample],
        block_size: BlockSize,
        sample_rate: SampleRate,
    ) -> GenState {
        for i in 0..*block_size {
            let mut recalculate = false;
            let v = value[i];
            let t = time[i];
            if v != self.last_value {
                self.last_value = v;
                recalculate = true;
            }
            if t != self.last_time {
                self.last_time = t;
                recalculate = true;
            }
            if recalculate {
                let num_samples = (t * *sample_rate).floor();
                self.step = (v - self.current_value) / num_samples;
                self.num_steps_to_go = num_samples as usize;
            }
            if self.num_steps_to_go == 0 {
                self.current_value = v;
                self.step = 0.0;
            } else {
                self.current_value += self.step;
                self.num_steps_to_go -= 1;
            }
            ramped_value[i] = self.current_value;
        }
        GenState::Continue
    }

    #[init]
    #[allow(missing_docs)]
    pub fn init(&mut self) {
        self.num_steps_to_go = 0;
    }
}
