use dialog::DialogBox;
use std::time::Duration;

use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions},
    controller::print_error_handler,
    knyst_commands,
    prelude::*,
};

fn main() -> anyhow::Result<()> {
    let mut backend = CpalBackend::new(CpalBackendOptions::default())?;
    // let mut backend = JackBackend::new("Knyst<3JACK")?;
    let _sphere = KnystSphere::start(
        &mut backend,
        SphereSettings {
            num_inputs: 0,
            num_outputs: 2,
            ..Default::default()
        },
        print_error_handler,
    );

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
