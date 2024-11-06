use anyhow::Result;
use knyst::envelope::envelope_gen;
#[allow(unused)]
use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions, JackBackend},
    controller::print_error_handler,
    prelude::*,
};

/// Initializes the audio backend, sets up the `KnystSphere`, and configures a wavetable oscillator to output audio.
///
/// The function performs the following steps:
/// 1. Initializes the audio backend and starts a `KnystSphere`.
/// 2. Sets the active sphere for audio processing.
/// 3. Creates an owned wavetable oscillator set at 110 Hz.
/// 4. Outputs the oscillator to the left and right channels at 30% volume each.
/// 5. Frequency of the oscillator is changed by a frequency envelope.
/// 6. Waits for user to press ENTER.
///
/// The envelope has a starting value of 0.0. The envelope consists of points that are in the
/// format (level, duration). The duration is given in seconds, and indicates the: time to reach the
/// point from the start value or from the preceding point.
///
fn main() -> Result<()> {
    let _backend = setup();

    let frequency_envelope = envelope_gen(
        110.0,
        vec![
            (220.0, 2.0),
            (440.5, 2.),
            (220.25, 4.0),
            (77.5, 2.0),
            (220.1, 2.0),
            (330.0, 6.0),
        ],
        knyst::envelope::SustainMode::NoSustain,
        StopAction::Continue,
    );

    let node0 = wavetable_oscillator_owned(Wavetable::sine()).freq(frequency_envelope);

    graph_output(0, node0 * 0.3);
    graph_output(1, node0 * 0.3);

    println!("Playing a sine wave with an envelope at 110 Hz at an amplitude of 0.3");
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
///
/// # Returns
///
/// A tuple containing:
/// - `impl AudioBackend`: The initialized audio backend, either CPAL or JACK.
/// - `Option<SphereId>`: The optional unique identifier for the created `KnystSphere`.
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
