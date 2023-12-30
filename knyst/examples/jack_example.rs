use anyhow::Result;
use knyst::{audio_backend::JackBackend, controller::print_error_handler, prelude::*};

fn main() -> Result<()> {
    let mut backend = JackBackend::new("Knyst<3JACK")?;

    let sample_rate = backend.sample_rate() as Sample;
    let block_size = backend.block_size().unwrap_or(64);
    println!("sr: {sample_rate}, block: {block_size}");

    let _sphere = KnystSphere::start(
        &mut backend,
        SphereSettings {
            num_inputs: 1,
            num_outputs: 2,
            ..Default::default()
        },
        print_error_handler,
    );
    // Owned wavetable oscillator at 440 Hz
    let node0 = wavetable_oscillator_owned(Wavetable::sine()).freq(440.);
    // We can also use a shared wavetable oscillator which reads its wavetable from `Resources`. A cosine wavetable
    // is created by default.
    let modulator = oscillator(WavetableId::cos()).freq(5.);

    graph_output(0, node0 * (modulator * 0.25 + 0.5));
    let node1 = wavetable_oscillator_owned(Wavetable::sine()).freq(220.);
    let modulator = oscillator(WavetableId::cos()).freq(3.);
    graph_output(1, node1 * (modulator * 0.25 + 0.5));
    let mut input = String::new();
    loop {
        match std::io::stdin().read_line(&mut input) {
            Ok(_n) => {
                let input = input.trim();
                if let Ok(freq) = input.parse::<usize>() {
                    node0.freq(freq as f32);
                    println!("Setting freq to {freq}");
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
