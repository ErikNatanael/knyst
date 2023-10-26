// Generate code that compiles, the Gen will not be used in any way
#![allow(dead_code)]
#![allow(unused)]
use knyst_macro::gen;

use knyst_core::{gen::GenState, Sample};

struct Sine {
    phase: f32,
}
#[gen]
impl Sine {
    #[process]
    fn process(&mut self, freq: &[Sample], phase: &[Sample], out0: &mut [Sample]) -> GenState {
        todo!()
    }
}

fn main() {}
