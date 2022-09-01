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
    let block_size = backend.block_size();
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
    graph.connect(constl(440., "freq").to_node(node0)).unwrap();
    let modulator = graph.push_gen(WavetableOscillatorOwned::new(Wavetable::sine()));
    graph
        .connect(constl(5., "freq").to_node(modulator))
        .unwrap();
    let mod_amp = graph.push_gen(Mult);
    graph.connect(modulator.to(mod_amp)).unwrap();
    graph.connect(consti(0.25, 1).to_node(mod_amp)).unwrap();
    let amp = graph.push_gen(Mult);
    graph.connect(node0.to(amp)).unwrap();
    graph.connect(consti(0.5, 1).to_node(amp)).unwrap();
    graph.connect(mod_amp.to(amp).to_index(1)).unwrap();
    graph.connect(Connection::out(amp)).unwrap();
    graph.connect(Connection::out(amp).to_index(1)).unwrap();
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
