use anyhow::Result;
use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions},
    controller::{self, KnystCommands},
    graph::{ClosureGen, Mult, NodeAddress},
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

struct State {
    potential_reverb_inputs: Vec<NodeAddress>,
    reverb_node: Option<NodeAddress>,
    block_size: usize,
    sample_rate: f32,
}

fn main() -> Result<()> {
    let mut backend = CpalBackend::new(CpalBackendOptions::default())?;

    let sample_rate = backend.sample_rate() as f32;
    let block_size = backend.block_size().unwrap_or(64);
    let resources = Resources::new(ResourcesSettings {
        sample_rate,
        ..Default::default()
    });
    let top_level_graph = Graph::new(GraphSettings {
        block_size,
        sample_rate,
        num_outputs: backend.num_outputs(),
        ..Default::default()
    });
    let mut k = backend.start_processing(
        top_level_graph,
        resources,
        RunGraphSettings {
            scheduling_latency: Duration::from_millis(100),
        },
        controller::print_error_handler,
    )?;
    let num_outputs = backend.num_outputs();

    // Nodes are pushed to the top level graph if no graph id is specified
    let node0 = k.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    k.connect(constant(ROOT_FREQ).to(&node0).to_label("freq"));
    let modulator = k.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    k.connect(constant(5.).to(&modulator).to_label("freq"));
    let mod_amp = k.push(Mult);
    k.connect(modulator.to(&mod_amp));
    k.connect(constant(0.25).to(&mod_amp).to_index(1));
    let amp = k.push(Mult);
    k.connect(node0.to(&amp));
    k.connect(constant(0.25).to(&amp).to_index(1));
    k.connect(mod_amp.to(&node0).to_label("freq"));
    k.connect(amp.to_graph_out());
    k.connect(amp.to_graph_out().to_index(1));

    let sub_graph = Graph::new(GraphSettings {
        block_size,
        sample_rate,
        num_outputs,
        name: String::from("subgraph"),
        ..Default::default()
    });

    let sub_graph_id = sub_graph.id();
    let sub_graph = k.push(sub_graph);
    k.connect(sub_graph.to_graph_out().channels(2));

    // Store the nodes that would be connected to the reverb if it's toggled on.
    let potential_reverb_inputs = vec![sub_graph, amp];
    let mut state = State {
        potential_reverb_inputs,
        reverb_node: None,
        block_size,
        sample_rate,
    };

    // Have a background thread play some harmony
    {
        let chords = vec![
            vec![0, 17, 31],
            vec![53, 17, 31],
            vec![0, 22, 36],
            vec![0, 22, 39],
            vec![-5, 9, 31],
        ];
        let mut k = k.clone();
        std::thread::spawn(move || {
            let mut rng = thread_rng();
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

                std::thread::sleep(Duration::from_millis(1000));
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

                    if !handle_special_keys(c, k.clone(), &mut state) {
                        // Change the frequency of the nodes based on what key was pressed
                        let new_freq = character_to_hz(c);
                        k.schedule_change(ParameterChange::now(node0.clone(), new_freq).l("freq"));
                        k.schedule_change(
                            ParameterChange::now(modulator.clone(), new_freq * 5.).l("freq"),
                        );
                        k.schedule_change(
                            ParameterChange::now(mod_amp.clone(), new_freq * 0.1).i(1),
                        );
                    }

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

fn handle_special_keys(c: char, mut k: KnystCommands, state: &mut State) -> bool {
    match c {
        'm' => {
            // Load a buffer and play it back
            let files = rfd::FileDialog::new()
                .add_filter("wav", &["wav"])
                .pick_file();
            if let Some(file_path) = files {
                match Buffer::from_sound_file(file_path) {
                    Ok(buffer) => {
                        let id = k.insert_buffer(buffer);
                        let reader =
                            BufferReader::new(knyst::IdOrKey::Id(id), 1.0, StopAction::FreeSelf);
                        let reader = k.push(reader);
                        k.connect(reader.to_graph_out());
                        k.connect(reader.to_graph_out().to_index(1));
                    }
                    Err(e) => eprintln!("Error opening sound buffer: {e}"),
                }
            }
            true
        }
        'n' => {
            match state.reverb_node.take() {
                Some(reverb_node) => k.free_node_mend_connections(reverb_node),
                None => {
                    let reverb_node = insert_reverb(
                        k,
                        &state.potential_reverb_inputs,
                        state.sample_rate as f64,
                        state.block_size,
                    );
                    state.reverb_node = Some(reverb_node);
                }
            }
            true
        }
        _ => false,
    }
}

fn insert_reverb(
    mut k: KnystCommands,
    inputs: &[NodeAddress],
    sample_rate: f64,
    _block_size: usize,
) -> NodeAddress {
    let mix = 0.5;
    let reverb = k.push(fundsp_reverb_gen(sample_rate, mix));
    for input in inputs {
        // k.disconnect(input.clone().to_graph_out());
        // k.disconnect(input.clone().to_graph_out().to_index(1));
        k.disconnect(Connection::clear_to_graph_outputs(&input));
        k.connect(input.clone().to(&reverb));
        k.connect(input.clone().to(&reverb).to_index(1));
    }
    k.connect(reverb.to_graph_out().channels(1));
    k.connect(reverb.to_graph_out().to_index(1));
    reverb
}

fn fundsp_reverb_gen(sample_rate: f64, mix: f32) -> ClosureGen {
    use fundsp::audiounit::AudioUnit32;
    let mut fundsp_graph = {
        use fundsp::hacker32::*;
        //let mut c = c * 0.1;
        let mut c = multipass() & mix * reverb_stereo(10.0, 5.0);
        c.reset(Some(sample_rate));
        c
    };

    gen(move |ctx, _resources| {
        let in0 = ctx.inputs.get_channel(0);
        let in1 = ctx.inputs.get_channel(1);
        let out0 = ctx.outputs.get_channel_mut(0);
        let out1 = ctx.outputs.get_channel_mut(1);
        let mut output = [out0, out1];
        let input = [in0, in1];
        fundsp_graph.process(ctx.block_size(), input.as_slice(), output.as_mut_slice());
        GenState::Continue
    })
    .output("out0")
    .output("out1")
    .input("in0")
    .input("in1")
}
