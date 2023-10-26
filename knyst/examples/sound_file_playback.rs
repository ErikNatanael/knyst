use dialog::DialogBox;
use std::time::Duration;

use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions},
    controller,
    osc::BufferReaderMulti,
    prelude::*,
};

fn main() -> anyhow::Result<()> {
    // Create the backend to get the backend settings needed to create a Graph with the correct block size and sample rate etc.
    let mut backend = CpalBackend::new(CpalBackendOptions::default())?;

    let sample_rate = backend.sample_rate() as f32;
    let block_size = backend.block_size().unwrap_or(64);
    let mut resources = Resources::new(ResourcesSettings {
        ..Default::default()
    });

    // Get file path to a sound file from the user
    let file_path = dialog::FileSelection::new("Please select an audio file")
        .title("Open audio file")
        .show()
        .expect("Could not display dialog box")
        .unwrap();
    // Insert the buffer before sending Resources to the audio thread
    let buffer = resources.insert_buffer(Buffer::from_sound_file(file_path)?)?;

    let graph_settings = GraphSettings {
        block_size,
        sample_rate,
        num_outputs: backend.num_outputs(),
        ..Default::default()
    };
    println!("{graph_settings:#?}");
    // Create the Graph with the settings above
    let g: Graph = Graph::new(graph_settings);
    // The backend will split the Graph into two
    let mut k = backend.start_processing(
        g,
        resources,
        RunGraphSettings::default(),
        controller::print_error_handler,
    )?;

    // Create a node which reads the buffer we inserted earlier and plays it back
    // `channels(2)` means we are expecting a stereo buffer
    // `looping(false)` means the buffer will not be looped
    let buf_playback_node = k.push(
        BufferReaderMulti::new(buffer, 1.0, StopAction::FreeSelf)
            .channels(2)
            .looping(true),
        inputs!(),
    );
    // Connect the buffer to the outputs.
    k.connect(buf_playback_node.to_graph_out().channels(2));

    println!("Playing back sound for 10 seconds");
    std::thread::sleep(Duration::from_millis(10000));
    Ok(())
}
