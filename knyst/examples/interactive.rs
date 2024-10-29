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
    envelope::{envelope_gen, Envelope, EnvelopeGenHandle},
    gen::delay::{allpass_feedback_delay, static_sample_delay},
    graph::{Mult, NodeId},
    handles::{AnyNodeHandle, HandleData},
    inputs, knyst_commands,
    prelude::*,
    trig::once_trig,
};
use rand::{seq::SliceRandom, thread_rng, Rng};
use std::{
    io::Write,
    sync::{atomic::AtomicBool, mpsc::Receiver},
};
use std::{sync::Arc, time::Duration};

use termion::input::TermRead;
use termion::raw::IntoRawMode;

const ROOT_FREQ: Sample = 200.;

struct State {
    potential_delay_inputs: Vec<AnyNodeHandle>,
    delay_node: Option<AnyNodeHandle>,
    harmony_wavetable_id: WavetableId,
    tokio_trigger: Arc<AtomicBool>,
    lead_env: Handle<EnvelopeGenHandle>,
    error_strings: Vec<String>,
    error_receiver: Receiver<String>,
    // store a NodeAddress pointing to nothing to simulate getting an error
    invalid_node: AnyNodeHandle,
}

// TODO: Use the musical time map for scheduling and sometimes change the tempo
fn main() -> Result<()> {
    let (error_sender, error_receiver) = std::sync::mpsc::channel();

    let mut backend = CpalBackend::new(CpalBackendOptions::default())?;
    // let mut backend = JackBackend::new("Knyst<3JACK")?;
    let _sphere = KnystSphere::start(
        &mut backend,
        SphereSettings {
            num_inputs: 0,
            num_outputs: 2,
            ..Default::default()
        },
        Box::new(move |error| {
            error_sender.send(format!("{error}")).unwrap();
        }),
    );

    // This Handle will very soon point to nothing because it will run and
    // then immediately free itself
    let invalid_node = once_trig();

    let mod_amp = bus(1).set(0, 0.25);
    let freq_bus = bus(1).set(0, ROOT_FREQ);
    let modulator = wavetable_oscillator_owned(Wavetable::sine()).freq(freq_bus * 5.);
    let node0 = oscillator(WavetableId::cos()).freq(freq_bus + (modulator * mod_amp));

    let env = envelope_gen(
        0.0,
        vec![(0.25, 0.02), (0.125, 0.1), (0.0, 2.5)],
        knyst::envelope::SustainMode::NoSustain,
        StopAction::Continue,
    );
    let node = node0 * env;
    graph_output(0, node.repeat_outputs(1));

    let sub_graph = upload_graph(
        knyst_commands()
            .default_graph_settings()
            .oversampling(knyst::graph::Oversampling::X2),
        || {},
    );

    let sub_graph_id = sub_graph.graph_id();
    graph_output(0, sub_graph);

    // Store the nodes that would be connected to the delay if it's toggled on.
    let potential_delay_inputs: Vec<AnyNodeHandle> = vec![sub_graph.into(), node.into()];

    let mut k = knyst_commands();
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
        std::thread::spawn(move || {
            let mut k = knyst_commands();
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
                    k.schedule_change(ParameterChange::now(
                        node.input("freq"),
                        degree_53_to_hz(new_chord[i] as Sample, ROOT_FREQ * 2.0),
                    ));
                    std::thread::sleep(Duration::from_millis(rng.gen::<u64>() % 1500 + 500));
                }

                std::thread::sleep(Duration::from_millis(rng.gen::<u64>() % 1000 + 1000));
            }
        });
    }

    // Start the tokio subsystem
    let tokio_trigger = Arc::new(AtomicBool::new(false));

    {
        let trigger = tokio_trigger.clone();
        std::thread::spawn(move || tokio_knyst(trigger));
    }
    let mut state = State {
        potential_delay_inputs,
        delay_node: None,
        harmony_wavetable_id,
        tokio_trigger,
        lead_env: env.into(),
        error_strings: vec![],
        error_receiver,
        invalid_node: invalid_node.into(),
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
        "n: toggle delay",
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
                    if !handle_special_keys(c, &mut state) {
                        // Change the frequency of the nodes based on what key was pressed
                        let new_freq = character_to_hz(c);
                        freq_bus.set(0, new_freq);
                        mod_amp.set(0, new_freq * 0.2);
                        // Trigger the envelope to restart
                        state.lead_env.restart_trig();
                        write!(
                            stdout,
                            "{}Triggered note with frequency {new_freq:.2}                  ",
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
fn handle_special_keys(c: char, state: &mut State) -> bool {
    let mut k = knyst_commands();
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
            match state.delay_node.take() {
                Some(reverb_node) => {
                    k.free_node_mend_connections(reverb_node.node_ids().next().unwrap())
                }
                None => {
                    let delay_node = insert_delay(&state.potential_delay_inputs);
                    state.delay_node = Some(delay_node);
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
            graph_output(0, &state.invalid_node);
            true
        }
        _ => false,
    }
}

fn insert_delay(inputs: &[AnyNodeHandle]) -> AnyNodeHandle {
    let delay = allpass_feedback_delay(48000).feedback(0.5).delay_time(0.3);
    graph_output(0, delay);
    graph_output(1, static_sample_delay(62).input(delay));
    for input in inputs {
        // Clear all connections from this node to the outputs of the graph it is in.
        knyst_commands().disconnect(Connection::clear_to_graph_outputs(
            input.node_ids().next().unwrap(),
        ));
        // Connect the node to the newly created reverb instead
        delay.input(input);
    }
    delay.into()
}

// Start a tokio async runtime to demonstrate that it works. This is an
// alternative to using standard threads.
//
// When using a multi threaded async runtime, it's important to keep in mind to activate
// the correct graph you want to push to after every `await` since, if it is scheduled
// on a new thread, that thread will have thread locals with different graph settings.
// `await`ing inside of building a local graph or scheduling a bundle is also very likely to break.
#[tokio::main]
async fn tokio_knyst(mut trigger: Arc<AtomicBool>) {
    let mut rng = thread_rng();
    loop {
        receive_trigger(&mut trigger).await;
        let speed = rng.gen_range(0.1..0.6);
        tokio::spawn(async move {
            play_a_little_tune(speed).await;
        });
    }
}

async fn receive_trigger(trigger: &mut Arc<AtomicBool>) {
    while !trigger.load(std::sync::atomic::Ordering::SeqCst) {
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
    }
    trigger.store(false, std::sync::atomic::Ordering::SeqCst);
}

async fn play_a_little_tune(speed: Sample) {
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
        spawn_note(freq, beats * speed).await;
        tokio::time::sleep(tokio::time::Duration::from_secs_f32((beats * speed) as f32)).await;
    }
}

async fn spawn_note(freq: Sample, length_seconds: Sample) {
    let mut k = knyst_commands();
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
