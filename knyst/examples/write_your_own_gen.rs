use std::f32::consts::TAU;

use anyhow::Result;
use knyst::audio_backend::{CpalBackend, CpalBackendOptions};
use knyst::controller::print_error_handler;
use knyst::prelude::*;
#[warn(missing_docs)]
struct NaiveSine {
    phase: Sample,
    phase_step: Sample,
    amp: Sample,
}
impl NaiveSine {
    pub fn update_freq(&mut self, freq: Sample, sample_rate: Sample) {
        self.phase_step = (freq * TAU) / sample_rate;
    }
}

#[impl_gen]
impl NaiveSine {
    pub fn new() -> Self {
        Self {
            phase: 0.0,
            phase_step: 0.0,
            amp: 1.0,
        }
    }
    #[process]
    pub fn process(
        &mut self,
        freq: &[Sample],
        amp: &[Sample],
        sig: &mut [Sample],
        block_size: BlockSize,
        sample_rate: SampleRate,
    ) -> GenState {
        for i in 0..*block_size {
            let freq = freq[i];
            self.amp = amp[i];
            self.update_freq(freq, *sample_rate);
            sig[i] = self.phase.cos() * self.amp;
            self.phase += self.phase_step;
            if self.phase > TAU {
                self.phase -= TAU;
            }
        }
        GenState::Continue
    }
}

fn main() -> Result<()> {
    // Start knyst
    let mut backend = CpalBackend::new(CpalBackendOptions::default())?;
    // let mut backend = JackBackend::new("Knyst<3JACK")?;
    let _sphere = KnystSphere::start(
        &mut backend,
        SphereSettings {
            ..Default::default()
        },
        print_error_handler,
    );

    // std::thread::sleep(std::time::Duration::from_millis(1000));
    // Play our NaiveSine
    graph_output(0, naive_sine().freq(300.).amp(0.3).channels(2));
    println!("Playing a sine wave at 300 Hz and an amplitude of 0.3");
    println!("Press [ENTER] to exit");
    // Wait for new line
    let mut s = String::new();
    let _ = std::io::stdin().read_line(&mut s);
    Ok(())
}
