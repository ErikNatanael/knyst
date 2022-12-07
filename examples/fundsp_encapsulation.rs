//! This example demonstrates putting a static fundsp graph into a Knyst Graph
//! and listening to the output. It encapsulates a subset of the fundsp "beep"
//! example code, running it through a tanh function with a gain controlled by a
//! sine wave with an offset.
//!
//! It is helpful to keep the fundsp code encapsulated because its imports may
//! interfere with other function names.

use std::time::Duration;

use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions},
    graph::{Mult, Ramp},
    prelude::*,
    wavetable::{Wavetable, WavetableOscillatorOwned},
};
use rand::{thread_rng, Rng};

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
        latency: Duration::from_millis(200),
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

    let mut fundsp_graph = {
        use fundsp::hacker::*;
        let sample_rate = sample_rate as f64;

        // The following code is an excerpt from the beep.rs example for fundsp:
        // https://github.com/SamiPerttu/fundsp/blob/master/examples/beep.rs

        // FM synthesis.
        let f = 110.0;
        let m = 5.0;
        let c = oversample(sine_hz(f) * f * m + f >> sine());

        // Apply Moog filter.
        let c = (c | lfo(|t| (xerp11(110.0, 11000.0, sin_hz(0.15, t)), 0.6))) >> moog();

        let c = c >> split::<U2>();

        //let c = fundsp::sound::risset_glissando(false);

        // Add chorus.
        let c = c >> (chorus(0, 0.0, 0.01, 0.5) | chorus(1, 0.0, 0.01, 0.5));

        // Add flanger.
        let c = c
            >> (flanger(0.6, 0.005, 0.01, |t| lerp11(0.005, 0.01, sin_hz(0.1, t)))
                | flanger(0.6, 0.005, 0.01, |t| lerp11(0.005, 0.01, cos_hz(0.1, t))));

        let mut c = c
        >> (declick() | declick())
        >> (dcblock() | dcblock())
        //>> (multipass() & 0.2 * reverb_stereo(10.0, 3.0))
        >> limiter_stereo((1.0, 5.0));
        //let mut c = c * 0.1;
        c.reset(Some(sample_rate));
        c
    };

    // Make a custom Gen that adds some distortion to the output with stereo
    // inputs and outputs. You could also implement the Gen trait for your own
    // struct.
    let output_node = graph.push(
        gen(move |ctx, _resources| {
            let out0 = ctx.outputs.get_channel_mut(0);
            let out1 = ctx.outputs.get_channel_mut(1);
            for ((o0, o1), dist) in out0
                .iter_mut()
                .zip(out1.iter_mut())
                .zip(ctx.inputs.get_channel(0).iter())
            {
                use fundsp::hacker::*;
                let frame = fundsp_graph.get_stereo();
                // Apply some tanh distortion, getting the distortion amount
                *o0 = tanh(frame.0 as f32 * dist.max(1.0)) * 0.5;
                *o1 = tanh(frame.1 as f32 * dist.max(1.0)) * 0.5;
            }
            GenState::Continue
        })
        .output("out0")
        .output("out1")
        .input("distortion"),
    );
    // Create a Ramp for smooth transitions between distortion values.
    let dist_ramp = graph.push(Ramp::new());
    graph.connect(dist_ramp.to(output_node).to_label("distortion"))?;
    graph.connect(dist_sine_mul.to(dist_ramp).to_label("value"))?;
    graph.connect(constant(1.5).to(dist_ramp).to_label("value"))?;
    graph.connect(constant(0.05).to(dist_ramp).to_label("time"))?;
    graph.connect(output_node.to_graph_out().channels(2))?;
    graph.commit_changes();
    graph.update();

    let mut rng = thread_rng();
    loop {
        // Change the distortion value offset to a random value. Note that we're setting
        // the input value of the Ramp which is connected to the distortion
        // value.
        graph.schedule_change(
            ParameterChange::now(dist_ramp, rng.gen::<f32>() * 5.0 - 2.5).label("value"),
        )?;
        // After scheduling a change, we need to update the graph scheduler for
        // it to pass changes on to the audio thread.
        graph.update();
        std::thread::sleep(Duration::from_secs_f32(0.1));
    }
}
