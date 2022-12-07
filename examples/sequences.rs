use std::time::Duration;

use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions},
    envelope::{Curve, Envelope},
    graph::{GenState, Mult, NodeAddress},
    prelude::*,
    wavetable::{Wavetable, WavetableOscillatorOwned},
    ResourcesSettings,
};
use rand::{prelude::SliceRandom, thread_rng, Rng};
fn main() -> anyhow::Result<()> {
    // Write some pitch sequences. The pitches are in 53edo degrees, but you can just as well use 12tet.
    #[rustfmt::skip]
    let seq1_0 = vec![
        0, -22, 9, -22, 14, -22, 0, -22,
        9, -22, 14, -22, 22, -22, 9, -22,
        14, -22, 22, -22, 31, -22, 14, -22,
        9, -22, 14, -22, 22, -22, 9, -22,
    ];
    #[rustfmt::skip]
    let seq2_0 = vec![
        14, 9, 0, 9,
        9, 0, 48 - 53, 9,
        9, 0, 43 - 53, 39 - 53,
        31 - 53, 9, 36, 22,
    ];
    let seq4_0 = vec![31, 14, 22, 9, 14, 0, 48 - 53, 9];

    #[rustfmt::skip]
    let seq1_1 = vec![
        22, 0, 31, 0, 39, 0, 22, 0,
        31, 0, 39, 0, 45, 0, 31, 0,
        39, 0, 45, 0, 53, 0, 39, 0,
        31, 0, 39, 0, 44, 0, 31, 0,
    ];
    #[rustfmt::skip]
    let seq2_1 = vec![
        39, 31, 22, 31,
        31, 22, 17, 31,
        22, 17, 8, 22,
        0, 22, 17, 53,
    ];
    let seq4_1 = vec![53, 39, 45, 31, 39, 22, 17, 31];

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
    let mut graph: Graph = Graph::new(graph_settings);
    backend.start_processing(&mut graph, resources)?;

    // Create a custom output node. Right now, it just acts like a brickwall limiter.
    let output_node = graph.push(
        gen(move |ctx, _resources| {
            let out0 = ctx.outputs.get_channel_mut(0);
            let out1 = ctx.outputs.get_channel_mut(1);
            for (((o0, o1), i0), i1) in out0
                .iter_mut()
                .zip(out1.iter_mut())
                .zip(ctx.inputs.get_channel(0).iter())
                .zip(ctx.inputs.get_channel(1).iter())
            {
                *o0 = i0.clamp(-0.9, 0.9);
                *o1 = i1.clamp(-0.9, 0.9);
            }
            GenState::Continue
        })
        .output("out0")
        .output("out1")
        .input("in0")
        .input("in1"),
    );
    // Connect the output_node to the output of the main graph with two channels, which are mapped 0 -> 0, 1 -> 1
    graph.connect(output_node.to_graph_out().channels(2))?;

    // This local struct helps us keep track of the form of the piece based on the loop count.
    struct Section {
        loop_count: usize,
    }
    impl Section {
        fn fast_part(&self) -> bool {
            self.loop_count % 4 == 2 || self.loop_count % 4 == 3
        }
        fn ostinato(&self) -> bool {
            self.loop_count % 2 == 1
        }
        fn high_octave(&self) -> f32 {
            if self.fast_part() || self.ostinato() {
                2.0
            } else {
                1.0
            }
        }
    }

    let mut section = Section { loop_count: 0 };

    let mut rng = thread_rng();
    let mut fundamental = 440.0;
    loop {
        // First play the minor sequences, the ones names *_0
        for beat_counter in 0..64 {
            if section.fast_part() && beat_counter % 1 == 0 {
                let freq = degree_to_freq(31, fundamental);
                let amp = rng.gen::<f32>() * 0.5 + 0.25;
                add_sine(
                    freq * vec![0.125, 0.25].choose(&mut rng).unwrap(),
                    amp,
                    0.02,
                    0.05,
                    graph_settings,
                    output_node,
                    &mut graph,
                )?;
            }
            if section.ostinato() && beat_counter % 2 == 0 {
                let i = beat_counter / 2;
                let amp = [0.4, 0.2, 0.6, 0.2, 1.0, 0.2, 0.4, 0.2][i % 8];
                let freq = degree_to_freq(seq1_0[(beat_counter / 2) % seq1_0.len()], fundamental);
                add_sine(
                    freq * 0.5,
                    amp,
                    0.001,
                    0.25,
                    graph_settings,
                    output_node,
                    &mut graph,
                )?;
            }
            if beat_counter % 4 == 0 {
                let i = beat_counter / 4;
                let amp = [0.8, 0.6, 0.4, 0.7][i % 4];
                let freq = degree_to_freq(seq2_0[(beat_counter / 4) % seq2_0.len()], fundamental);
                add_sine(freq, amp, 0.1, 0.5, graph_settings, output_node, &mut graph)?;
            }
            if beat_counter % 8 == 0 {
                let i = beat_counter / 8;
                let amp = [1.0, 0.5, 0.8, 0.5][i % 4];
                let freq = degree_to_freq(seq4_0[(beat_counter / 8) % seq4_0.len()], fundamental);
                add_sine(freq, amp, 0.1, 1.2, graph_settings, output_node, &mut graph)?;
            }

            std::thread::sleep(Duration::from_millis(150));
        }
        // Then play the major sequences, the ones names *_1
        let f_degrees = vec![0, 22, 31, 39, 45, 53, 17 + 53, 22 + 53, 17 + 53, 8 + 53, 45];
        for beat_counter in 0..64 {
            if section.fast_part() && beat_counter % 1 == 0 {
                let amp = rng.gen::<f32>() * 0.5 + 0.25;
                let degree = f_degrees[beat_counter % f_degrees.len()];
                let freq = degree_to_freq(degree, fundamental);
                add_sine(
                    freq * 0.125 * 2.0_f32.powi((beat_counter / f_degrees.len()) as i32),
                    amp,
                    rng.gen::<f32>().powi(3) * 0.2,
                    0.20,
                    graph_settings,
                    output_node,
                    &mut graph,
                )?;
            }
            if section.ostinato() && beat_counter % 2 == 0 {
                let i = beat_counter / 2;
                let amp = [0.4, 0.2, 0.6, 0.2, 1.0, 0.2, 0.4, 0.2][i % 8];
                let freq = degree_to_freq(seq1_1[(beat_counter / 2) % seq1_1.len()], fundamental);
                add_sine(
                    freq * 0.5,
                    amp,
                    0.001,
                    0.25,
                    graph_settings,
                    output_node,
                    &mut graph,
                )?;
            }
            if beat_counter % 4 == 0 {
                let i = beat_counter / 4;
                let amp = [1.0, 0.6, 0.3, 0.5][i % 4];
                let freq = degree_to_freq(seq2_1[(beat_counter / 4) % seq2_1.len()], fundamental);
                add_sine(
                    freq * section.high_octave(),
                    amp,
                    0.1,
                    0.9,
                    graph_settings,
                    output_node,
                    &mut graph,
                )?;
            }
            if beat_counter % 8 == 0 {
                let i = beat_counter / 8;
                let amp = [1.0, 0.6, 0.8, 0.5][i % 4];
                let freq = degree_to_freq(seq4_1[(beat_counter / 8) % seq4_1.len()], fundamental);
                add_sine(
                    freq * section.high_octave(),
                    amp,
                    0.2,
                    1.2,
                    graph_settings,
                    output_node,
                    &mut graph,
                )?;
            }

            if section.loop_count % 4 == 3 {
                fundamental = rng.gen::<f32>() * 200. + 300.;
            }
            std::thread::sleep(Duration::from_millis(150));
        }
        section.loop_count += 1;
        if section.loop_count % 4 == 0 {
            fundamental = rng.gen::<f32>() * 200. + 300.;
        }
    }
}

fn degree_to_freq(degree: i32, root: f32) -> f32 {
    root * 2.0_f32.powf(degree as f32 / 53.0)
}

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
    g.connect(mult.to_graph_out())?;
    g.connect(mult.to_graph_out().to_index(1))?;
    g.commit_changes();
    Ok(g)
}

fn add_sine(
    freq: f32,
    amp: f32,
    attack: f32,
    duration_secs: f32,
    graph_settings: GraphSettings,
    output_node: NodeAddress,
    main_graph: &mut Graph,
) -> anyhow::Result<()> {
    let node = main_graph.push(sine_tone_graph(
        freq,
        attack,
        0.05 * amp,
        duration_secs,
        graph_settings,
    )?);
    main_graph.connect(node.to(output_node))?;
    main_graph.connect(node.to(output_node).to_index(1))?;
    main_graph.commit_changes();
    main_graph.update();
    Ok(())
}
