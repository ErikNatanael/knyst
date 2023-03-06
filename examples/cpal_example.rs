use anyhow::Result;
use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions},
    controller::print_error_handler,
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
        ..Default::default()
    });
    let graph: Graph = Graph::new(GraphSettings {
        block_size,
        sample_rate,
        num_outputs: backend.num_outputs(),
        ..Default::default()
    });
    let mut k = backend
        .start_processing(
            graph,
            resources,
            RunGraphSettings {
                scheduling_latency: Duration::from_millis(100),
            },
            print_error_handler,
        )
        .unwrap();
    let node0 = k.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    k.connect(constant(440.).to(&node0).to_label("freq"));
    let modulator = k.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    k.connect(constant(5.).to(&modulator).to_label("freq"));
    let mod_amp = k.push(Mult);
    k.connect(modulator.to(&mod_amp));
    k.connect(constant(0.25).to(&mod_amp).to_index(1));
    let amp = k.push(Mult);
    k.connect(node0.to(&amp));
    k.connect(constant(0.5).to(&amp).to_index(1));
    k.connect(mod_amp.to(&amp).to_index(1));
    k.connect(amp.to_graph_out());
    k.connect(amp.to_graph_out().to_index(1));

    let mut input = String::new();
    loop {
        match std::io::stdin().read_line(&mut input) {
            Ok(n) => {
                println!("{} bytes read", n);
                println!("{}", input.trim());
                let input = input.trim();
                if let Ok(freq) = input.parse::<usize>() {
                    k.schedule_change(ParameterChange::now(node0.clone(), freq as f32).l("freq"));
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
