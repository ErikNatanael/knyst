use dialog::DialogBox;
use std::time::Duration;

use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions},
    buffer::BufferReaderMulti,
    prelude::*,
};

fn main() -> anyhow::Result<()> {
    // Create the backend to get the backend settings needed to create a Graph with the correct block size and sample rate etc.
    let mut backend = CpalBackend::new(CpalBackendOptions::default())?;

    let sample_rate = backend.sample_rate() as f32;
    let block_size = backend.block_size().unwrap_or(64);
    let mut resources = Resources::new(ResourcesSettings {
        sample_rate,
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
        latency: Duration::from_millis(100),
        num_outputs: backend.num_outputs(),
        ..Default::default()
    };
    println!("{graph_settings:#?}");
    // Create the Graph with the settings above
    let mut g: Graph = Graph::new(graph_settings);
    // The backend will split the Graph into two
    backend.start_processing(&mut g, resources)?;

    let buf_playback_node =
        g.push(BufferReaderMulti::new(buffer, 1.0, StopAction::FreeSelf).channels(2));
    g.connect(buf_playback_node.to_graph_out().channels(2))?;

    g.commit_changes();
    g.update();
    println!("Playing back sound for 10 seconds");
    std::thread::sleep(Duration::from_millis(10000));
    Ok(())
}
