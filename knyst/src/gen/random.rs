use std::sync::atomic::AtomicU64;

use knyst_macro::impl_gen;

use crate::wavetable::{FRACTIONAL_PART, TABLE_SIZE};
use crate::{self as knyst, SampleRate};
use crate::{wavetable::WavetablePhase, Sample};

use super::GenState;

/// Random numbers with linear interpolation with new values at some frequency. Freq is sampled at control rate only.
pub struct RandomLin {
    rng: fastrand::Rng,
    current_value: Sample,
    current_change_width: Sample,
    // when phase reaches 1 we choose a new value
    phase: Sample,
    freq_to_phase_inc: Sample,
}

/// Used to seed random number generating Gens to create a deterministic result as long as all Gens are created in the same order from start.
pub static NEXT_SEED: AtomicU64 = AtomicU64::new(0);

#[impl_gen]
impl RandomLin {
    pub fn new() -> Self {
        let mut rng = fastrand::Rng::with_seed(
            NEXT_SEED.fetch_add(1, std::sync::atomic::Ordering::SeqCst) * 94 + 53,
        );
        Self {
            current_value: rng.f32() as Sample,
            phase: 0.0,
            rng,
            freq_to_phase_inc: 0.0,
            current_change_width: 0.0,
        }
    }

    pub fn init(&mut self, sample_rate: SampleRate) {
        self.freq_to_phase_inc = 1.0 / *sample_rate;
        self.new_value();
    }

    #[inline]
    fn new_value(&mut self) {
        let old_target = self.current_value + self.current_change_width;
        let new = self.rng.f32();
        self.current_value = old_target;
        self.current_change_width = new - old_target;
        self.phase = 0.0;
        // dbg!(old_target, self.current_value, self.current_change_width);
    }

    pub fn process(&mut self, freq: &[Sample], output: &mut [Sample]) -> GenState {
        let phase_step = freq[0] * self.freq_to_phase_inc;

        for out in output {
            *out = self.current_value + self.phase * self.current_change_width;
            self.phase += phase_step;

            if self.phase >= 1.0 {
                self.new_value();
            }
        }

        GenState::Continue
    }
}