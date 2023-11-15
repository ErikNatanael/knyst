use anyhow::Result;
use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions},
    controller::{print_error_handler, KnystCommands},
    graph::Mult,
    prelude::*,
};
use std::time::Duration;

fn main() -> Result<()> {
    let mut backend = CpalBackend::new(CpalBackendOptions::default())?;

    let sample_rate = backend.sample_rate() as Sample;
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
            Box::new(print_error_handler),
        )
        .unwrap();
    let node0 = k.push(
        WavetableOscillatorOwned::new(Wavetable::sine()),
        inputs!(("freq" : 440.)),
    );
    let modulator = k.push(
        WavetableOscillatorOwned::new(Wavetable::sine()),
        inputs!(("freq" : 5.)),
    );
    let mod_amp = k.push(Mult, inputs!((0 ; modulator.out(0)), (1 : 0.25)));
    let amp = k.push(
        Mult,
        inputs!((0 ; node0.out(0)), (1 : 0.5 ; mod_amp.out(0))),
    );
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
                    k.schedule_change(ParameterChange::now(node0.input("freq"), freq as Sample));
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
