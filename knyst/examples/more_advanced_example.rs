use anyhow::Result;
use knyst::gen::delay::allpass_feedback_delay;
#[allow(unused)]
use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions, JackBackend},
    controller::print_error_handler,
    envelope::Envelope,
    handles::{graph_output, handle, Handle},
    modal_interface::knyst_commands,
    prelude::{delay::static_sample_delay, *},
    sphere::{KnystSphere, SphereSettings},
};
use rand::{thread_rng, Rng};
fn main() -> Result<()> {
    let mut backend =
        CpalBackend::new(CpalBackendOptions::default()).expect("Unable to connect to CPAL backend");
    // Uncomment the line below and comment the line above to use the JACK backend instead
    // let mut backend = JackBackend::new("Knyst<3JACK").expect("Unable to start JACK backend");
    let _sphere = KnystSphere::start(
        &mut backend,
        SphereSettings {
            num_inputs: 0,
            num_outputs: 2,
            ..Default::default()
        },
        print_error_handler,
    );

    let delay = allpass_feedback_delay(48000).feedback(0.9).delay_time(0.23);
    // Output delay with a small additional delay to create a rudimentary stereo illusion
    graph_output(0, (delay + static_sample_delay(87).input(delay)) * 0.5);
    graph_output(1, (delay + static_sample_delay(62).input(delay)) * 0.5);
    let outer_graph = upload_graph(knyst_commands().default_graph_settings(), || {});
    // Nothing is passed into the delay until now. Pass any output of the "outer_graph" into the delay.
    delay.input(outer_graph);
    // Make the "outer_graph" the active graph, meaning any new nodes are pushed to it.
    outer_graph.activate();

    let mut rng = thread_rng();
    // Create a node for the frequency value so that this value can be changed once and
    // propagated to many other nodes.
    let freq_var = bus(1).set(0, 440.);
    for _ in 0..10 {
        let freq = (sine().freq(
            sine()
                .freq(
                    sine()
                        .freq(0.01)
                        .range(0.02, rng.gen_range(0.05..0.3 as Sample)),
                )
                .range(0.0, 400.),
        ) * 100.0)
            + freq_var;
        let node0 = sine();
        node0.freq(freq);
        let modulator = sine();
        modulator.freq(sine().freq(0.09) * -5.0 + 6.0);
        graph_output(0, (node0 * modulator * 0.025).repeat_outputs(1));
    }

    std::thread::spawn(move || {
        // We're on a new thread so we have to activate the graph we are targeting again.
        outer_graph.activate();
        for &ratio in [1.0, 1.5, 5. / 4.].iter().cycle() {
            // new graph
            let graph = upload_graph(
                knyst_commands().default_graph_settings().num_inputs(1),
                || {
                    // Since freq_var is in a different graph we can pipe it in via a graph input
                    let freq_var = graph_input(0, 1);
                    let sig = sine().freq(freq_var * ratio).out("sig") * 0.25;
                    let env = Envelope {
                        points: vec![(1.0, 0.005), (0.0, 0.5)],
                        stop_action: StopAction::FreeGraph,
                        ..Default::default()
                    };
                    let sig = sig * handle(env.to_gen());
                    // let sig = sig * handle(env.to_gen());
                    graph_output(0, sig.repeat_outputs(1));
                },
            );
            // Make sure we also pass the freq_var signal in
            graph.set(0, freq_var);
            outer_graph.activate();
            // Add the direct signal of the graph together with a delay
            let sig = graph + static_sample_delay(48 * 500).input(graph.out(0));
            // Output to the outer graph
            graph_output(0, sig.repeat_outputs(1));
            std::thread::sleep(std::time::Duration::from_millis(2500));
        }
    });

    // Create a repeating arpeggio on a different thread
    std::thread::spawn(move || {
        // We're on a new thread so we have to activate the graph we are targeting again.
        outer_graph.activate();
        loop {
            for &ratio in [1.0, 5. / 4., 1.5, 7. / 4., 2., 17. / 8.].iter() {
                // new graph
                let graph = upload_graph(
                    knyst_commands().default_graph_settings().num_inputs(1),
                    || {
                        // Since freq_var is in a different graph we can pipe it in via a graph input
                        let freq_var = graph_input(0, 1);
                        let sig = sine().freq(freq_var * ratio).out("sig") * 0.25;
                        let env = Envelope {
                            points: vec![(1.0, 0.005), (0.0, 0.5)],
                            stop_action: StopAction::FreeGraph,
                            ..Default::default()
                        };
                        let sig = sig * handle(env.to_gen());
                        // let sig = sig * handle(env.to_gen());
                        graph_output(0, sig.repeat_outputs(1));
                    },
                );
                // Make sure we also pass the freq_var signal in
                graph.set(0, freq_var);
                outer_graph.activate();
                // Output to the outer graph
                graph_output(0, graph.repeat_outputs(1) * 0.3);
                std::thread::sleep(std::time::Duration::from_millis(250));
            }
            std::thread::sleep(std::time::Duration::from_millis(3500));
        }
    });

    let mut input = String::new();
    loop {
        println!("Input a frequency for the root note, or 'q' to quit: ");
        match std::io::stdin().read_line(&mut input) {
            Ok(_n) => {
                let input = input.trim();
                if let Ok(freq) = input.parse::<f32>() {
                    println!("New freq: {}", input.trim());
                    freq_var.set(0, freq as Sample);
                } else if input == "q" {
                    break;
                }
            }
            Err(error) => println!("error: {}", error),
        }
        input.clear();
    }
    Ok(())
}

fn sine() -> Handle<OscillatorHandle> {
    oscillator(WavetableId::cos())
}
