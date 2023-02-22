use anyhow::Result;
use knyst::{
    async_api,
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
    let mut graph = Graph::new(GraphSettings {
        block_size,
        sample_rate,
        num_outputs: backend.num_outputs(),
        ..Default::default()
    });
    backend.start_processing(
        &mut graph,
        resources,
        RunGraphSettings {
            scheduling_latency: Duration::from_millis(100),
        },
    )?;
    let mut tk = async_api::start_async_knyst_thread(graph);
    let sub_graph = Graph::new(GraphSettings {
        block_size,
        sample_rate,
        num_outputs: backend.num_outputs(),
        ..Default::default()
    });
    let sub_graph_id = sub_graph.id();
    let node0 = tk.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    tk.connect(constant(440.).to(&node0).to_label("freq"));
    let modulator = tk.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    tk.connect(constant(5.).to(&modulator).to_label("freq"));
    let mod_amp = tk.push(Mult);
    tk.connect(modulator.to(&mod_amp));
    tk.connect(constant(0.25).to(&mod_amp).to_index(1));
    let amp = tk.push(Mult);
    tk.connect(node0.to(&amp));
    tk.connect(constant(0.5).to(&amp).to_index(1));
    tk.connect(mod_amp.to(&amp).to_index(1));
    tk.connect(amp.to_graph_out());
    tk.connect(amp.to_graph_out().to_index(1));
    let mut input = String::new();
    loop {
        match std::io::stdin().read_line(&mut input) {
            Ok(n) => {
                println!("{} bytes read", n);
                println!("{}", input.trim());
                let input = input.trim();
                if let Ok(freq) = input.parse::<usize>() {
                    tk.schedule_change(ParameterChange::now(node0.clone(), freq as f32).l("freq"));
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
