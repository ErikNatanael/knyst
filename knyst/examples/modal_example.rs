use anyhow::Result;
use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions},
    handles::{graph_output, NodeHandle},
    osc::{wavetable_oscillator_owned, WavetableOscillatorOwnedHandle},
    prelude::*,
    sphere::{KnystSphere, SphereSettings},
};
fn main() -> Result<()> {
    let mut backend = CpalBackend::new(CpalBackendOptions::default())?;
    let _sphere = KnystSphere::start(&mut backend, SphereSettings::default());

    // let root_freq = var(440.); // todo: add nodes which are a constant value, could be a Bus(1) or a special Gen // todo: add nodes which are a constant value, could be a Bus(1) or a special Gen
    let freq =
        (sine().freq(sine().freq(sine().freq(0.01) * 0.05 + 0.07) * 200.0 + 200.0) * 100.0) + 440.;
    let node0 = sine();
    node0.freq(freq);
    let modulator = sine();
    modulator.freq(sine().freq(0.09) * -5.0 + 6.0);
    graph_output(0, (node0 * modulator * 0.25).repeat_outputs(1));

    let mut input = String::new();
    loop {
        match std::io::stdin().read_line(&mut input) {
            Ok(n) => {
                println!("{} bytes read", n);
                println!("{}", input.trim());
                let input = input.trim();
                if let Ok(freq) = input.parse::<usize>() {
                    node0.freq(freq as f32);
                } else if input == "q" {
                    break;
                }
            }
            Err(error) => println!("error: {}", error),
        }
        input.clear();
    }
    Ok(())
}

fn sine() -> NodeHandle<WavetableOscillatorOwnedHandle> {
    wavetable_oscillator_owned(Wavetable::sine())
}
