// Generate code that compiles, the Gen will not be used in any way
#![allow(dead_code)]
#![allow(unused)]
use knyst_macro::impl_gen;

use knyst::prelude::{GenState, Sample, SampleRate};

struct Sine {
    phase: f32,
}
#[impl_gen]
impl Sine {
    #[process]
    fn process(
        &mut self,
        freq: &[Sample],
        phase: &[Sample],
        out0: &mut [Sample],
        sample_rate: SampleRate,
    ) -> GenState {
        for ((freq, phase), out) in freq.iter().zip(phase).zip(out0) {
            *out = (self.phase + phase).sin();
            self.phase += freq / *sample_rate;
        }
        GenState::Continue
    }

    #[init]
    pub fn init(&mut self, sample_rate: SampleRate) {
        let _ = sample_rate;
    }
}

fn main() {}
