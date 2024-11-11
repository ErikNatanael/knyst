use anyhow::Result;
#[allow(unused)]
use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions, JackBackend},
    controller::print_error_handler,
    prelude::*,
};

/// Sets up both an owned and shared wavetable oscillator, and connects them to audio output channels.
///
/// Listens for user input to dynamically adjust the frequency of the first oscillator.
///
fn main() -> Result<()> {
    let _backend = setup();

    // Owned wavetable oscillator at 440 Hz
    let node0 = wavetable_oscillator_owned(Wavetable::sine()).freq(440.);
    // We can also use a shared wavetable oscillator which reads it's wavetable from `Resources`. A cosine wavetable
    // is created by default.
    let modulator = oscillator(WavetableId::cos()).freq(5.);
    // Output to the zeroth (left) channel
    graph_output(0, node0 * (modulator * 0.25 + 0.5));

    let node1 = wavetable_oscillator_owned(Wavetable::sine()).freq(220.);
    let modulator = oscillator(WavetableId::cos()).freq(3.);
    // Output a different sound to the first (right) channel
    graph_output(1, node1 * (modulator * 0.25 + 0.5));

    // Respond to user input. This kind of interaction can be put in a different thread and/or in an async runtime.
    println!("Playing a sine wave with 440 Hz and 220 Hz");
    println!("Enter a new frequency for the left channel followed by [ENTER]");
    println!("Press [q] to quit");
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

/// Initializes the audio backend and starts a `KnystSphere` for audio processing.
/// Start with an automatic helper thread for scheduling changes and managing resources.
/// If you want to manage the `Controller` yourself, use `start_return_controller`.
///
/// The backend is returned here because it would otherwise be dropped at the end of setup()
fn setup() -> impl AudioBackend {
    let mut backend =
        CpalBackend::new(CpalBackendOptions::default()).expect("Unable to connect to CPAL backend");
    // Uncomment the line below and comment the line above to use the JACK backend instead
    // let mut backend = JackBackend::new("Knyst<3JACK").expect("Unable to start JACK backend");

    let _sphere_id = KnystSphere::start(
        &mut backend,
        SphereSettings {
            ..Default::default()
        },
        print_error_handler,
    )
    .ok();
    backend
}
