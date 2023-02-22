use anyhow::Result;
use knyst::{
    async_api,
    audio_backend::{CpalBackend, CpalBackendOptions},
    graph::{Mult, NodeAddress},
    prelude::*,
    wavetable::{Wavetable, WavetableOscillatorOwned},
};
use rand::{seq::SliceRandom, thread_rng, Rng};
use std::io::Write;
use std::time::Duration;

use termion;
use termion::input::TermRead;
use termion::raw::IntoRawMode;

static ROOT_FREQ: f32 = 200.;

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
    let num_outputs = backend.num_outputs();
    // We start the asynchronous api on another thread and receive an object
    // with which we can communicate with the graph.
    //
    // TODO: Combine with start_processing or something to ensure we have access
    // to modifying resources as well.
    let mut tk = async_api::start_async_knyst_thread(top_level_graph);

    // Nodes are pushed to the top level graph if no graph id is specified
    let node0 = tk.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    tk.connect(constant(ROOT_FREQ).to(&node0).to_label("freq"));
    let modulator = tk.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    tk.connect(constant(5.).to(&modulator).to_label("freq"));
    let mod_amp = tk.push(Mult);
    tk.connect(modulator.to(&mod_amp));
    tk.connect(constant(0.25).to(&mod_amp).to_index(1));
    let amp = tk.push(Mult);
    tk.connect(node0.to(&amp));
    tk.connect(constant(0.25).to(&amp).to_index(1));
    tk.connect(mod_amp.to(&node0).to_label("freq"));
    tk.connect(amp.to_graph_out());
    tk.connect(amp.to_graph_out().to_index(1));

    // Have a background thread play some harmony
    {
        let chords = vec![
            vec![0, 17, 31],
            vec![53, 17, 31],
            vec![0, 22, 36],
            vec![0, 22, 39],
            vec![-5, 9, 31],
        ];
        let mut k = tk.clone();
        std::thread::spawn(move || {
            let mut rng = thread_rng();
            let mut sub_graph = Graph::new(GraphSettings {
                block_size,
                sample_rate,
                num_outputs,
                name: String::from("subgraph"),
                ..Default::default()
            });

            let sub_graph_id = sub_graph.id();
            let sub_graph = k.push(sub_graph);
            k.connect(sub_graph.to_graph_out().channels(2));
            let mut harmony_wavetable = Wavetable::sine();
            harmony_wavetable.add_odd_harmonics(8, 1.3);
            harmony_wavetable.normalize();
            let harmony_nodes: Vec<NodeAddress> = (0..chords[0].len())
                .map(|_| {
                    let node = k.push_to_graph(
                        WavetableOscillatorOwned::new(harmony_wavetable.clone()),
                        sub_graph_id,
                    );
                    let amp = k.push_to_graph(Mult, sub_graph_id);

                    k.connect(constant(400.).to(&node).to_label("freq"));
                    k.connect(node.to(&amp));
                    k.connect(constant(0.05).to(&amp).to_index(1));
                    k.connect(amp.to_graph_out());
                    k.connect(amp.to_graph_out().to_index(1));
                    node
                })
                .collect();
            loop {
                let new_chord = chords.choose(&mut rng).unwrap();
                for (i, node) in harmony_nodes.iter().enumerate() {
                    k.schedule_change(
                        ParameterChange::now(
                            node.clone(),
                            degree_53_to_hz(new_chord[i] as f32, ROOT_FREQ * 2.0),
                        )
                        .label("freq"),
                    );
                }

                std::thread::sleep(Duration::from_millis(rng.gen_range(500..2500)));
            }
        });
    }
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
                    tk.schedule_change(ParameterChange::now(mod_amp.clone(), new_freq * 0.1).i(1));

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
            'i' => 5 + 53,
            'k' => 62,
            'o' => 14 + 53,
            'l' => 17 + 53,
            'p' => 22 + 53,
            'ö' => 26 + 53,
            'å' => 31 + 53,
            'ä' => 36 + 53,
            _ => 0,
        } as f32,
        ROOT_FREQ * 4.,
    )
}
