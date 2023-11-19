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
        self.phase_step = freq / sample_rate;
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
            let amp = amp[i];
            self.update_freq(freq, *sample_rate);
            self.amp = amp;
            sig[i] = self.phase.cos() * self.amp;
            self.phase += self.phase_step;
            if self.phase > 1.0 {
                self.phase -= 1.0;
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
    // Play our NaiveSine
    graph_output(0, naive_sine() * 0.5);
    Ok(())
}
