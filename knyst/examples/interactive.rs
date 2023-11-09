//! This example is supposed to exemplify the most important features of Knyst:
//! - starting an audio backend
//! - pushing nodes
//! - making connections
//! - inner graphs
//! - async and multi threaded usage of KnystCommands
//! - scheduling changes
//! - interactivity
//! - wrapping other dsp libraries (fundsp in this case)
//! - writing a custom error handler

use anyhow::Result;
use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions},
    controller::MultiThreadedKnystCommands,
    envelope::{Curve, Envelope},
    graph::{ClosureGen, Mult, NodeId},
    inputs,
    prelude::*,
    trig::OnceTrig,
};
use rand::{seq::SliceRandom, thread_rng, Rng};
use std::{
    io::Write,
    sync::{atomic::AtomicBool, mpsc::Receiver},
};
use std::{sync::Arc, time::Duration};

use termion::input::TermRead;
use termion::raw::IntoRawMode;

static ROOT_FREQ: Sample = 200.;

struct State {
    potential_reverb_inputs: Vec<NodeId>,
    reverb_node: Option<NodeId>,
    harmony_wavetable_id: WavetableId,
    block_size: usize,
    tokio_trigger: Arc<AtomicBool>,
    lead_env_address: NodeId,
    error_strings: Vec<String>,
    error_receiver: Receiver<String>,
    // store a NodeAddress pointing to nothing to simulate getting an error
    invalid_node: NodeId,
}

// TODO: Use the musical time map for scheduling and sometimes change the tempo
fn main() -> Result<()> {
    let mut backend = CpalBackend::new(CpalBackendOptions::default())?;

    let sample_rate = backend.sample_rate() as Sample;
    let block_size = backend.block_size().unwrap_or(64);
    dbg!(block_size);
    let resources = Resources::new(ResourcesSettings::default());
    let top_level_graph = Graph::new(GraphSettings {
        block_size,
        sample_rate,
        oversampling: knyst::graph::Oversampling::X1,
        num_outputs: backend.num_outputs(),
        ..Default::default()
    });
    let (error_sender, error_receiver) = std::sync::mpsc::channel();
    let mut k = backend.start_processing(
        top_level_graph,
        resources,
        RunGraphSettings {
            scheduling_latency: Duration::from_millis(100),
        },
        Box::new(move |error| {
            error_sender.send(format!("{error}")).unwrap();
        }),
    )?;
    let num_outputs = backend.num_outputs();

    // This NodeAddress will very soon point to nothing because it will run and
    // then immediately free itself
    let invalid_node = k.push(OnceTrig::new(), inputs![]);

    // Nodes are pushed to the top level graph if no graph id is specified
    let modulator = k.push(
        WavetableOscillatorOwned::new(Wavetable::sine()),
        // The `inputs` macro accepts multiple tuples in the format
        // `(channel : constant ; node_outputs)` where the constant and the node_outputs are
        // optional
        inputs!(("freq" : 5.)),
    );
    let mod_amp = k.push(Mult, inputs!((0 ; modulator.out(0)), (1 : 0.25)));
    let node0 = k.push(
        WavetableOscillatorOwned::new(Wavetable::sine()),
        inputs!(("freq": ROOT_FREQ ; mod_amp.out(0))),
    );
    let env = Envelope {
        // Points are given in the format (value, time_to_reach_value)
        points: vec![(0.25, 0.02), (0.125, 0.1), (0.0, 0.5)],
        curves: vec![Curve::Linear, Curve::Linear, Curve::Exponential(2.0)],
        ..Default::default()
    };
    let env = k.push(env.to_gen(), inputs!());
    let amp = k.push(Mult, inputs!((0 ; node0.out(0)), (1 ; env.out(0))));
    k.connect(amp.to_graph_out().channels(2));

    let sub_graph = Graph::new(GraphSettings {
        // block_size: block_size / 2,
        block_size,
        sample_rate,
        num_outputs,
        oversampling: knyst::graph::Oversampling::X2,
        name: String::from("subgraph"),
        num_inputs: 0,
        ..Default::default()
    });

    let sub_graph_id = sub_graph.id();
    let sub_graph = k.push(sub_graph, inputs!());
    k.connect(sub_graph.to_graph_out().channels(2));

    // Store the nodes that would be connected to the reverb if it's toggled on.
    let potential_reverb_inputs = vec![sub_graph, amp];

    let mut harmony_wavetable = Wavetable::sine();
    harmony_wavetable.add_odd_harmonics(8, 1.3);
    harmony_wavetable.normalize();
    let harmony_wavetable_id = k.insert_wavetable(harmony_wavetable);

    // Have a background thread play some harmony
    {
        let chords = vec![
            vec![0, 17, 31],
            vec![53, 17, 31],
            vec![0, 22, 36],
            vec![9, 22, 45],
            vec![9, 17, 45],
            vec![0, 9, 22],
        ];
        let mut k = k.clone();
        std::thread::spawn(move || {
            let mut rng = thread_rng();
            let harmony_nodes: Vec<NodeId> = (0..chords[0].len())
                .map(|i| {
                    let node = k.push_to_graph(
                        Oscillator::new(harmony_wavetable_id),
                        sub_graph_id,
                        ("freq", 400.),
                    );
                    let sine_amp = k.push_to_graph(
                        WavetableOscillatorOwned::new(Wavetable::sine()),
                        sub_graph_id,
                        inputs!(("freq" : rng.gen_range(0.1..0.4))),
                    );
                    let sine_amp_mul = k.push_to_graph(
                        Mult,
                        sub_graph_id,
                        inputs!((0 ; sine_amp.out(0)), (1 : 0.4 * 0.03)),
                    );
                    let amp = k.push_to_graph(
                        Mult,
                        sub_graph_id,
                        inputs!((0 ; node.out(0)), (1 : 0.03 ; sine_amp_mul.out(0))),
                    );
                    // Pan at positions -0.5, 0.0, 0.5
                    let pan = k.push_to_graph(
                        PanMonoToStereo,
                        sub_graph_id,
                        inputs!(("signal" ; amp.out(0)), ("pan" : (i as f32 - 1.0) * 0.5)),
                    );
                    k.connect(pan.to_graph_out().channels(2));
                    node
                })
                .collect();
            // Change to a new chord
            loop {
                let new_chord = chords.choose(&mut rng).unwrap();
                for (i, node) in harmony_nodes.iter().enumerate() {
                    k.schedule_change(
                        ParameterChange::now(
                            node.clone(),
                            degree_53_to_hz(new_chord[i] as Sample, ROOT_FREQ * 2.0),
                        )
                        .label("freq"),
                    );
                    std::thread::sleep(Duration::from_millis(rng.gen::<u64>() % 1500 + 500));
                }

                std::thread::sleep(Duration::from_millis(rng.gen::<u64>() % 1000 + 1000));
            }
        });
    }

    // Start the tokio subsystem
    let tokio_trigger = Arc::new(AtomicBool::new(false));

    {
        let k = k.clone();
        let trigger = tokio_trigger.clone();
        std::thread::spawn(move || tokio_knyst(k, trigger));
    }
    let mut state = State {
        potential_reverb_inputs,
        reverb_node: None,
        block_size,
        harmony_wavetable_id,
        tokio_trigger,
        lead_env_address: env.clone(),
        error_strings: vec![],
        error_receiver,
        invalid_node,
    };

    // Set terminal to raw mode to allow reading stdin one key at a time
    let mut stdout = std::io::stdout().into_raw_mode().unwrap();

    // Use asynchronous stdin
    let mut stdin = termion::async_stdin().keys();
    let lines = [
        "Play your keyboard",
        "a-ä and w-å: trigger a new note on the monophonic synth",
        "q: quit",
        "m: load and play sound file",
        "n: toggle reverb",
        "b: replace wavetable for harmony notes",
        "v: trigger a little melody using async",
        "c: tries to allocate, in debug mode this will panic (will mess with the terminal)",
        "x: make an invalid connection which will create an error",
    ];
    write!(stdout, "{}", termion::clear::All,).unwrap();
    for (y, line) in lines.into_iter().enumerate() {
        write!(stdout, "{}{line}", termion::cursor::Goto(1, y as u16 + 1)).unwrap();
    }
    stdout.lock().flush().unwrap();
    loop {
        // Check if we have received any new errors and print all of them
        while let Ok(error) = state.error_receiver.try_recv() {
            state.error_strings.push(error);
        }
        write!(
            stdout,
            "{}Errors received ({}):",
            termion::cursor::Goto(1, lines.len() as u16 + 4),
            state.error_strings.len(),
        )
        .unwrap();
        for (y, e) in state.error_strings.iter().enumerate() {
            write!(
                stdout,
                "{}{e}",
                termion::cursor::Goto(3, lines.len() as u16 + y as u16 + 5)
            )
            .unwrap();
        }
        // Read input (if any)
        let input = stdin.next();

        // If a key was pressed
        if let Some(Ok(key)) = input {
            match key {
                // Exit if 'q' is pressed
                termion::event::Key::Char('q') => break,
                // Else print the pressed key
                termion::event::Key::Char(c) => {
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
                        // Trigger the envelope to restart
                        let trig = k.push(OnceTrig::new(), inputs!());
                        k.connect(trig.to(state.lead_env_address).to_label("restart_trig"));
                        write!(
                            stdout,
                            "{}Triggered note with frequency {new_freq}",
                            termion::cursor::Goto(1, lines.len() as u16 + 2)
                        )
                        .unwrap();
                        stdout.lock().flush().unwrap();
                    }
                }
                _ => (),
            }
        }
        // Put the cursor in the top left
        write!(stdout, "{}", termion::cursor::Goto(1, 1)).unwrap();
        // Short sleep time to minimise latency
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    Ok(())
}

fn degree_53_to_hz(degree: Sample, root: Sample) -> Sample {
    root * (2.0 as Sample).powf(degree / 53.)
}
fn character_to_hz(c: char) -> Sample {
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
        } as Sample,
        ROOT_FREQ * 4.,
    )
}

/// Perform different actions depending on the key
fn handle_special_keys(c: char, mut k: MultiThreadedKnystCommands, state: &mut State) -> bool {
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
                            BufferReader::new(IdOrKey::Id(id), 1.0, false, StopAction::FreeSelf);
                        let reader = k.push(reader, inputs!());
                        k.connect(reader.to_graph_out());
                        k.connect(reader.to_graph_out().to_index(1));
                    }
                    Err(e) => eprintln!("Error opening sound buffer: {e}"),
                }
            }
            true
        }
        'n' => {
            // Toggle the fundsp reverb on or off
            match state.reverb_node.take() {
                Some(reverb_node) => k.free_node_mend_connections(reverb_node),
                None => {
                    let reverb_node =
                        insert_reverb(k, &state.potential_reverb_inputs, state.block_size);
                    state.reverb_node = Some(reverb_node);
                }
            }
            true
        }
        'b' => {
            // Replace the wavetable used for the harmony nodes scheduled on a different thread
            let mut new_harmony_wavetable = Wavetable::sine();
            new_harmony_wavetable.add_odd_harmonics(
                rand::random::<usize>() % 16 + 1,
                rand::random::<Sample>() + 1.0,
            );
            new_harmony_wavetable.fill_sine(rand::random::<usize>() % 16 + 4, 1.0);
            new_harmony_wavetable.normalize();
            k.replace_wavetable(state.harmony_wavetable_id, new_harmony_wavetable);
            true
        }
        'v' => {
            // Send a trigger to the async tokio routine
            state
                .tokio_trigger
                .store(true, std::sync::atomic::Ordering::SeqCst);
            true
        }
        'c' => {
            // Here's an example of what you mustn't do. If you run this program
            // in debug mode it should panic because of assert_no_alloc.
            k.push(
                gen(|ctx, _resources| {
                    let out = ctx.outputs.iter_mut().next().unwrap();
                    let new_allocation: Vec<Sample> = ctx
                        .inputs
                        .get_channel(0)
                        .iter()
                        .map(|v| v.powf(2.5))
                        .collect();
                    for (o, &new_alloc) in out.iter_mut().zip(new_allocation.iter()) {
                        *o = new_alloc;
                    }
                    GenState::Continue
                })
                .output("out")
                .input("in"),
                inputs!(),
            );
            true
        }
        'x' => {
            // Produce an error by making an invalid connection
            k.connect(state.invalid_node.to_graph_out());
            true
        }
        _ => false,
    }
}

fn insert_reverb(
    mut k: MultiThreadedKnystCommands,
    inputs: &[NodeId],
    _block_size: usize,
) -> NodeId {
    let mix = 0.5;
    let reverb = k.push(fundsp_reverb_gen(mix), inputs!());
    for &input in inputs {
        // Clear all connections from this node to the outputs of the graph it is in.
        k.disconnect(Connection::clear_to_graph_outputs(input));
        // Connect the node to the newly created reverb instead
        k.connect(input.clone().to(reverb));
        k.connect(input.clone().to(reverb).to_index(1));
    }
    k.connect(reverb.to_graph_out().channels(2));
    reverb
}

/// Create a Gen containing a fundsp graph
///
/// Note: This currently allocates on the audio thread, the creator of fundsp is
/// working on a solution
fn fundsp_reverb_gen(mix: f32) -> ClosureGen {
    let mut fundsp_graph = {
        use fundsp::hacker32::*;
        let mut c = multipass() & mix * reverb_stereo(10.0, 5.0);
        c.allocate(); // Allocate any buffers needed internally ahead of time.
        c.reset();
        c
    };

    gen(move |ctx, _resources| {
        let in0 = ctx.inputs.get_channel(0);
        let in1 = ctx.inputs.get_channel(1);
        let block_size = ctx.block_size();
        let mut outputs = ctx.outputs.iter_mut();
        let out0 = outputs.next().unwrap();
        let out1 = outputs.next().unwrap();
        // With an f32 fundsp AudioUnit we can pass input/output buffers
        // straight to the fundsp process method to avoid copying.
        let mut output = [out0, out1];
        let input = [in0, in1];
        fundsp_graph.process(block_size, input.as_slice(), output.as_mut_slice());
        GenState::Continue
    })
    .output("out0")
    .output("out1")
    .input("in0")
    .input("in1")
}

// Start a tokio async runtime to demonstrate that it works. This is an
// alternative to using standard threads.
#[tokio::main]
async fn tokio_knyst(k: MultiThreadedKnystCommands, mut trigger: Arc<AtomicBool>) {
    let mut rng = thread_rng();
    loop {
        receive_trigger(&mut trigger).await;
        let k = k.clone();
        let speed = rng.gen_range(0.1..0.6);
        tokio::spawn(async move {
            play_a_little_tune(k, speed).await;
        });
    }
}

async fn receive_trigger(trigger: &mut Arc<AtomicBool>) {
    while !trigger.load(std::sync::atomic::Ordering::SeqCst) {
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
    }
    trigger.store(false, std::sync::atomic::Ordering::SeqCst);
}

async fn play_a_little_tune(mut k: MultiThreadedKnystCommands, speed: Sample) {
    let melody = vec![
        (17, 1.),
        (22, 1.),
        (31, 1.),
        (36, 1.),
        (45, 0.25),
        (53, 0.25),
        (58, 0.25),
        (62, 0.25),
        (53 + 17, 6.),
    ];
    for (degree_53, beats) in melody {
        let freq = degree_53_to_hz(degree_53 as Sample + 53., ROOT_FREQ);
        spawn_note(&mut k, freq, beats * speed).await;
        tokio::time::sleep(tokio::time::Duration::from_secs_f32((beats * speed) as f32)).await;
    }
}

async fn spawn_note(k: &mut MultiThreadedKnystCommands, freq: Sample, length_seconds: Sample) {
    let mut settings = k.default_graph_settings();
    settings.num_outputs = 1;
    settings.num_inputs = 0;
    let mut note_graph = Graph::new(settings);
    let mut wavetable = Wavetable::new();
    let num_harmonics = (15000. / freq) as usize;
    wavetable.add_aliasing_saw(num_harmonics, 1.0);
    let sig = note_graph.push(WavetableOscillatorOwned::new(wavetable.clone()));
    note_graph
        .connect(constant(freq).to(sig).to_label("freq"))
        .unwrap();
    let env = Envelope {
        points: vec![(0.15, 0.05), (0.07, 0.1), (0.0, length_seconds)],
        stop_action: StopAction::FreeGraph,
        ..Default::default()
    };
    let env = note_graph.push(env.to_gen());
    let amp = note_graph.push(Mult);
    note_graph.connect(sig.to(amp)).unwrap();
    note_graph.connect(env.to(amp).to_index(1)).unwrap();
    note_graph.connect(amp.to_graph_out()).unwrap();
    let note_graph = k.push(note_graph, inputs!());
    k.connect(note_graph.to_graph_out().channels(2));
}
