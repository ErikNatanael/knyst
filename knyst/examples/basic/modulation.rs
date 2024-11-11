use anyhow::Result;
#[allow(unused)]
use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions, JackBackend},
    controller::print_error_handler,
    prelude::*,
};

/// Configures a wavetable oscillator to output modulated audio.
///
/// The function performs the following steps:
/// 1. Creates an owned wavetable oscillator set at 110 Hz.
/// 2. Outputs the oscillator to the left and right channels at 30% volume each, modulating it with a frequency of 2 Hz.
/// 3. Waits for user to press ENTER.
///
/// The modulation signal is a sine which has a range of values between -1 and 1. To prevent
/// zero-crossings, the signal is shifted up by dividing by 2 (so the range becomes -0.5 to 0.5) and
/// then adding 0.5 (so the range becomes 0 to 1).
///
fn main() -> Result<()> {
    let _backend = setup();

    let node0 = wavetable_oscillator_owned(Wavetable::sine()).freq(110.);

    let modulator = oscillator(WavetableId::cos()).freq(2.);

    graph_output(0, node0 * 0.3 * (modulator * 0.5 + 0.5));
    graph_output(1, node0 * 0.3 * (modulator * 0.5 + 0.5));

    println!("Playing a modulated sine wave at 110 Hz at an amplitude of 0.3");
    println!("Press [ENTER] to exit");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
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
