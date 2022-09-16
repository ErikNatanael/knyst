use std::time::Duration;

use knyst::{
    audio_backend::JackBackend,
    graph::Mult,
    prelude::*,
    wavetable::{Wavetable, WavetableOscillatorOwned},
};
fn main() {
    let mut backend = JackBackend::new("knyst").unwrap();

    let sample_rate = backend.sample_rate() as f32;
    let block_size = backend.block_size().unwrap_or(64);
    println!("sr: {sample_rate}, block: {block_size}");
    let resources = Resources::new(sample_rate);
    let mut graph: Graph = Graph::new(GraphSettings {
        block_size,
        sample_rate,
        latency: Duration::from_millis(0),
        ..Default::default()
    });
    backend.start_processing(&mut graph, resources).unwrap();
    let node0 = graph.push_gen(WavetableOscillatorOwned::new(Wavetable::sine()));
    graph
        .connect(constant(440.).to(node0).to_label("freq"))
        .unwrap();
    let modulator = graph.push_gen(WavetableOscillatorOwned::new(Wavetable::sine()));
    graph
        .connect(constant(5.).to(modulator).to_label("freq"))
        .unwrap();
    let mod_amp = graph.push_gen(Mult);
    graph.connect(modulator.to(mod_amp)).unwrap();
    graph
        .connect(constant(0.25).to(mod_amp).to_index(1))
        .unwrap();
    let amp = graph.push_gen(Mult);
    graph.connect(node0.to(amp)).unwrap();
    graph.connect(constant(0.5).to(amp).to_index(1)).unwrap();
    graph.connect(mod_amp.to(amp).to_index(1)).unwrap();
    graph.connect(amp.to_graph_out()).unwrap();
    graph.connect(amp.to_graph_out().to_index(1)).unwrap();
    graph.commit_changes();
    graph.update(); // Required because constant connections get converted to
                    // scheduled changes when the graph is running.
    let mut input = String::new();
    loop {
        match std::io::stdin().read_line(&mut input) {
            Ok(_n) => {
                let input = input.trim();
                if let Ok(freq) = input.parse::<usize>() {
                    graph
                        .schedule_change(ParameterChange::now(node0, freq as f32).l("freq"))
                        .unwrap();
                    graph.update();
                    println!("Setting freq to {freq}");
                } else if input == "q" {
                    break;
                }
            }
            Err(error) => println!("error: {}", error),
        }
        input.clear();
    }
}
