use dialog::DialogBox;
use std::time::Duration;

use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions},
    controller::print_error_handler,
    knyst_commands,
    prelude::*,
};

/// Plays back 10 seconds of an audio file chosen by the user.
fn main() -> anyhow::Result<()> {
    let _backend = setup();

    // Get file path to a sound file from the user
    let file_path = dialog::FileSelection::new("Please select an audio file")
        .title("Open audio file")
        .show()
        .expect("Could not display dialog box")
        .unwrap();

    // Insert the buffer before sending Resources to the audio thread
    let sound_buffer = Buffer::from_sound_file(file_path)?;
    let channels = sound_buffer.num_channels();
    let buffer = knyst_commands().insert_buffer(sound_buffer);

    // Create a node which reads the buffer we inserted earlier and plays it back
    // `channels(2)` means we are expecting a stereo buffer
    // `looping(false)` means the buffer will not be looped
    let buf_playback_node = BufferReaderMulti::new(buffer, 1.0, StopAction::FreeSelf)
        .channels(channels)
        .looping(true)
        .upload();

    // Connect the buffer to the outputs, connecting 2 channels (a mono buffer will be played in both the left and right channels)
    graph_output(0, buf_playback_node.channels(2));

    println!("Playing back sound for 10 seconds");
    std::thread::sleep(Duration::from_millis(10000));
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
