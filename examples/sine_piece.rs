use std::time::Duration;

use fastapprox::fast::tanh;
use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions},
    envelope::{Curve, Envelope},
    graph::{GenState, Mult, NodeAddress, Ramp},
    prelude::*,
    wavetable::{Wavetable, WavetableOscillatorOwned},
};
use rand::{prelude::SliceRandom, thread_rng, Rng};

fn main() -> anyhow::Result<()> {
    // Create the backend to get the backend settings needed to create a Graph with the correct block size and sample rate etc.
    let mut backend = CpalBackend::new(CpalBackendOptions::default())?;

    let sample_rate = backend.sample_rate() as f32;
    let block_size = backend.block_size().unwrap_or(64);
    let resources = Resources::new(ResourcesSettings {
        sample_rate,
        ..Default::default()
    });
    let graph_settings = GraphSettings {
        block_size,
        sample_rate,
        latency: Duration::from_millis(100),
        num_outputs: backend.num_outputs(),
        ..Default::default()
    };
    println!("{graph_settings:#?}");
    // Create the Graph with the settings above
    let mut graph: Graph = Graph::new(graph_settings);
    // The backend will split the Graph into two
    backend.start_processing(&mut graph, resources)?;

    // Create a sine Gen to modulate the distortion parameter of the output_node below.
    let dist_sine = graph.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    // Connect the constant 0.03 to the input names "freq" on the node "dist_sine"
    graph.connect(constant(0.03).to(dist_sine).to_label("freq"))?;
    let dist_sine_mul = graph.push(Mult);
    // Multiply the dist_sine by 5.0, giving it the range of +- 5.0 at 0.3 Hz
    graph.connect(dist_sine.to(dist_sine_mul))?;
    graph.connect(constant(1.5).to(dist_sine_mul).to_index(1))?;

    // Make a custom Gen that adds some distortion to the output with stereo
    // inputs and outputs. You could also implement the Gen trait for your own
    // struct.
    let output_node = graph.push(
        gen(move |ctx, _resources| {
            let out0 = ctx.outputs.get_channel_mut(0);
            let out1 = ctx.outputs.get_channel_mut(1);
            for ((((o0, o1), i0), i1), dist) in out0
                .iter_mut()
                .zip(out1.iter_mut())
                .zip(ctx.inputs.get_channel(0).iter())
                .zip(ctx.inputs.get_channel(1).iter())
                .zip(ctx.inputs.get_channel(2).iter())
            {
                *o0 = tanh(*i0 * dist.max(1.0)) * 0.5;
                *o1 = tanh(*i1 * dist.max(1.0)) * 0.5;
            }
            GenState::Continue
        })
        .output("out0")
        .output("out1")
        .input("in0")
        .input("in1")
        .input("distortion"),
    );
    // Create a Ramp for smooth transitions between distortion values.
    let dist_ramp = graph.push(Ramp::new());
    graph.connect(dist_ramp.to(output_node).to_label("distortion"))?;
    graph.connect(dist_sine_mul.to(dist_ramp).to_label("value"))?;
    graph.connect(constant(1.5).to(dist_ramp).to_label("value"))?;
    graph.connect(constant(0.5).to(dist_ramp).to_label("time"))?;
    graph.connect(output_node.to_graph_out().channels(2))?;

    let mut rng = thread_rng();
    let chord = vec![1.0, 5. / 4., 3. / 2., 7. / 4., 2.0, 17. / 8.];
    let mut fundamental = 440.0;
    loop {
        // Change the distortion value offset to -1.5. Note that we're setting
        // the input value of the Ramp which is connected to the distortion
        // value.
        graph.schedule_change(ParameterChange::now(dist_ramp, -1.5).label("value"))?;
        // After scheduling a change, we need to update the graph scheduler for
        // it to pass changes on to the audio thread.
        graph.update();
        for _ in 0..32 {
            let attack = 0.5;
            let freq = chord.choose(&mut rng).unwrap() * fundamental;
            add_sine(freq, attack, 1.0, graph_settings, output_node, &mut graph)?;
            add_sine(
                freq * 0.999,
                attack,
                0.3,
                graph_settings,
                output_node,
                &mut graph,
            )?;
            add_sine(
                freq * 1.001,
                attack,
                0.5,
                graph_settings,
                output_node,
                &mut graph,
            )?;
            std::thread::sleep(Duration::from_secs_f32(0.15));
        }
        graph.connect(
            constant(rng.gen::<f32>() * 1.5)
                .to(dist_ramp)
                .to_label("value"),
        )?;
        for _ in 0..1 {
            let attack = 0.02;
            let freq = chord.choose(&mut rng).unwrap() * fundamental;
            add_sine(freq, attack, 2.0, graph_settings, output_node, &mut graph)?;
            let freq = chord.choose(&mut rng).unwrap() * fundamental * 0.125;
            add_sine(freq, 5.0, 10.0, graph_settings, output_node, &mut graph)?;
            std::thread::sleep(Duration::from_secs_f32(0.5));
            for _ in 0..15 {
                let attack = rng.gen::<f32>().powi(2) * 0.25 + 0.001;
                let freq = chord.choose(&mut rng).unwrap() * fundamental;
                add_sine(freq, attack, 1.0, graph_settings, output_node, &mut graph)?;
                add_sine(
                    freq * 0.999,
                    attack,
                    0.3,
                    graph_settings,
                    output_node,
                    &mut graph,
                )?;
                add_sine(
                    freq * 1.001,
                    attack,
                    0.5,
                    graph_settings,
                    output_node,
                    &mut graph,
                )?;
                std::thread::sleep(Duration::from_secs_f32(rng.gen::<f32>() * 0.35 + 0.15));
            }
        }
        fundamental *= chord.choose(&mut rng).unwrap();
        while fundamental > 880. {
            fundamental *= 0.5;
        }
    }
}

/// Returns a Graph containing a sine oscillator multiplied by an envelope that frees the Graph when it reaches the end.
fn sine_tone_graph(
    freq: f32,
    attack: f32,
    amp: f32,
    duration_secs: f32,
    graph_settings: GraphSettings,
) -> anyhow::Result<Graph> {
    let mut g = Graph::new(graph_settings);
    let sin = g.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    g.connect(constant(freq).to(sin).to_label("freq"))?;
    let env = Envelope {
        points: vec![(amp, attack), (0.0, duration_secs)],
        curves: vec![Curve::Linear, Curve::Exponential(2.0)],
        stop_action: StopAction::FreeGraph,
        ..Default::default()
    };
    let env = env.to_gen();
    let env = g.push(env);
    let mult = g.push(Mult);
    g.connect(sin.to(mult))?;
    g.connect(env.to(mult).to_index(1))?;
    g.connect(Connection::graph_output(mult))?;
    g.connect(Connection::graph_output(mult).to_index(1))?;
    g.commit_changes();
    Ok(g)
}

/// Add sines with the settings in the parameters and some stereo enhancing effects to the main graph.
fn add_sine(
    freq: f32,
    attack: f32,
    duration_secs: f32,
    graph_settings: GraphSettings,
    output_node: NodeAddress,
    main_graph: &mut Graph,
) -> anyhow::Result<()> {
    let node = main_graph.push(sine_tone_graph(
        freq,
        attack,
        0.01,
        duration_secs,
        graph_settings,
    )?);
    main_graph.connect(node.to(output_node))?;
    // Make the right side sine a different pitch to enhance the stereo effect.
    let node = main_graph.push(sine_tone_graph(
        freq * 1.002,
        attack * 1.12,
        0.01,
        duration_secs,
        graph_settings,
    )?);
    main_graph.connect(node.to(output_node).to_index(1))?;
    main_graph.commit_changes();
    main_graph.update();
    Ok(())
}
