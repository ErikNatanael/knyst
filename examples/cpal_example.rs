use std::time::Duration;

use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions},
    prelude::*,
    wavetable::{Wavetable, WavetableOscillatorOwned},
};
fn main() {
    let mut backend = CpalBackend::new(CpalBackendOptions::default()).unwrap();

    let sample_rate = backend.sample_rate() as f32;
    let block_size = backend.block_size();
    let resources = Resources::new(sample_rate);
    let mut graph: Graph = Graph::new(GraphSettings {
        block_size,
        sample_rate,
        latency: Duration::from_millis(0),
        num_outputs: backend.num_outputs(),
        ..Default::default()
    });
    backend.start_processing(&mut graph, resources).unwrap();
    let node0 = graph.push_gen(WavetableOscillatorOwned::new(Wavetable::sine()));
    graph.connect(Connection::out(node0)).unwrap();
    graph.connect(Connection::out(node0).to_index(1)).unwrap();
    graph.connect(constl(440., "freq").to_node(node0)).unwrap();
    graph.commit_changes();
    let mut input = String::new();
    loop {
        match std::io::stdin().read_line(&mut input) {
            Ok(n) => {
                println!("{} bytes read", n);
                println!("{}", input.trim());
                let input = input.trim();
                if let Ok(freq) = input.parse::<usize>() {
                    graph
                        .schedule_change(ParameterChange::now(node0, freq as f32).l("freq"))
                        .unwrap();
                    graph.update();
                } else if input == "q" {
                    break;
                }
            }
            Err(error) => println!("error: {}", error),
        }
        input.clear();
    }
}
