use anyhow::Result;
use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions},
    graph::Mult,
    prelude::*,
    wavetable::{Wavetable, WavetableOscillatorOwned},
};
use std::time::Duration;

fn main() -> Result<()> {
    let mut backend = CpalBackend::new(CpalBackendOptions::default())?;

    let sample_rate = backend.sample_rate() as f32;
    let block_size = backend.block_size().unwrap_or(64);
    let resources = Resources::new(ResourcesSettings {
        sample_rate,
        ..Default::default()
    });
    let mut graph: Graph = Graph::new(GraphSettings {
        block_size,
        sample_rate,
        latency: Duration::from_millis(100),
        num_outputs: backend.num_outputs(),
        ..Default::default()
    });
    backend.start_processing(&mut graph, resources)?;
    let node0 = graph.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    graph.connect(constant(440.).to(node0).to_label("freq"))?;
    let modulator = graph.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    graph.connect(constant(5.).to(modulator).to_label("freq"))?;
    let mod_amp = graph.push(Mult);
    graph.connect(modulator.to(mod_amp))?;
    graph.connect(constant(0.25).to(mod_amp).to_index(1))?;
    let amp = graph.push(Mult);
    graph.connect(node0.to(amp))?;
    graph.connect(constant(0.5).to(amp).to_index(1))?;
    graph.connect(mod_amp.to(amp).to_index(1))?;
    graph.connect(amp.to_graph_out())?;
    graph.connect(amp.to_graph_out().to_index(1))?;
    graph.commit_changes();
    graph.update(); // Required because constant connections get converted to
                    // scheduled changes when the graph is running.
    let mut input = String::new();
    loop {
        match std::io::stdin().read_line(&mut input) {
            Ok(n) => {
                println!("{} bytes read", n);
                println!("{}", input.trim());
                let input = input.trim();
                if let Ok(freq) = input.parse::<usize>() {
                    graph.schedule_change(ParameterChange::now(node0, freq as f32).l("freq"))?;
                    graph.update();
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
