use anyhow::Result;
use knyst::{
    async_api,
    audio_backend::{CpalBackend, CpalBackendOptions},
    graph::Mult,
    prelude::*,
    wavetable::{Wavetable, WavetableOscillatorOwned},
};
use std::io::Write;
use std::time::Duration;

use termion;
use termion::input::TermRead;
use termion::raw::IntoRawMode;

fn main() -> Result<()> {
    let mut backend = CpalBackend::new(CpalBackendOptions::default())?;

    let sample_rate = backend.sample_rate() as f32;
    let block_size = backend.block_size().unwrap_or(64);
    let resources = Resources::new(ResourcesSettings {
        sample_rate,
        ..Default::default()
    });
    let mut top_level_graph = Graph::new(GraphSettings {
        block_size,
        sample_rate,
        num_outputs: backend.num_outputs(),
        ..Default::default()
    });
    backend.start_processing(
        &mut top_level_graph,
        resources,
        RunGraphSettings {
            scheduling_latency: Duration::from_millis(100),
        },
    )?;
    // We start the asynchronous api on another thread and receive an object
    // with which we can communicate with the graph.
    //
    // TODO: Combine with start_processing or something to ensure we have access
    // to modifying resources as well.
    let mut tk = async_api::start_async_knyst_thread(top_level_graph);
    // let sub_graph = Graph::new(GraphSettings {
    //     block_size,
    //     sample_rate,
    //     num_outputs: backend.num_outputs(),
    //     ..Default::default()
    // });
    // let sub_graph_id = sub_graph.id();

    // Nodes are pushed to the top level graph if no graph id is specified
    let node0 = tk.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    tk.connect(constant(440.).to(&node0).to_label("freq"));
    let modulator = tk.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    tk.connect(constant(5.).to(&modulator).to_label("freq"));
    let mod_amp = tk.push(Mult);
    tk.connect(modulator.to(&mod_amp));
    tk.connect(constant(0.25).to(&mod_amp).to_index(1));
    let amp = tk.push(Mult);
    tk.connect(node0.to(&amp));
    tk.connect(constant(0.5).to(&amp).to_index(1));
    tk.connect(mod_amp.to(&node0).to_label("freq"));
    tk.connect(amp.to_graph_out());
    tk.connect(amp.to_graph_out().to_index(1));

    // Set terminal to raw mode to allow reading stdin one key at a time
    let mut stdout = std::io::stdout().into_raw_mode().unwrap();

    // Use asynchronous stdin
    let mut stdin = termion::async_stdin().keys();
    loop {
        // Read input (if any)
        let input = stdin.next();

        // If a key was pressed
        if let Some(Ok(key)) = input {
            match key {
                // Exit if 'q' is pressed
                termion::event::Key::Char('q') => break,
                // Else print the pressed key
                termion::event::Key::Char(c) => {
                    write!(
                        stdout,
                        "{}{}Play your keyboard, q: quit, key pressed: {:?}",
                        termion::clear::All,
                        termion::cursor::Goto(1, 1),
                        key
                    )
                    .unwrap();

                    // Change the frequency of the nodes based on what key was pressed
                    let new_freq = character_to_hz(c);
                    tk.schedule_change(ParameterChange::now(node0.clone(), new_freq).l("freq"));
                    tk.schedule_change(
                        ParameterChange::now(modulator.clone(), new_freq * 5.).l("freq"),
                    );
                    tk.schedule_change(ParameterChange::now(mod_amp.clone(), new_freq * 2.0).i(1));

                    stdout.lock().flush().unwrap();
                }
                _ => {
                    write!(
                        stdout,
                        "{}{}Play your keyboard, q: quit, key pressed: {:?}",
                        termion::clear::All,
                        termion::cursor::Goto(1, 1),
                        key
                    )
                    .unwrap();
                }
            }
        }
        // Short sleep time to minimise latency
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    Ok(())
}

fn degree_53_to_hz(degree: f32, root: f32) -> f32 {
    root * 2.0_f32.powf(degree / 53.)
}
fn character_to_hz(c: char) -> f32 {
    degree_53_to_hz(
        match c {
            'a' => 0,
            'z' => 8,
            's' => 9,
            'w' => 5,
            'e' => 14,
            'd' => 17,
            'r' => 22,
            'f' => 26,
            't' => 31,
            'g' => 36,
            'y' => 39,
            'b' => 43,
            'h' => 45,
            'u' => 48,
            'j' => 53,
            'k' => 62,
            _ => 0,
        } as f32,
        200.,
    )
}
