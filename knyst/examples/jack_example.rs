use anyhow::Result;
use knyst::{audio_backend::JackBackend, controller, graph::Mult, prelude::*};

fn main() -> Result<()> {
    let mut backend = JackBackend::new("Knyst<3JACK")?;

    let sample_rate = backend.sample_rate() as f32;
    let block_size = backend.block_size().unwrap_or(64);
    println!("sr: {sample_rate}, block: {block_size}");
    let resources = Resources::new(ResourcesSettings::default());
    let graph: Graph = Graph::new(GraphSettings {
        block_size,
        sample_rate,
        // In JACK we can decide ourselves how many outputs and inputs we want
        num_inputs: 1,
        num_outputs: 2,
        ..Default::default()
    });
    // `start_processing` is starting a Controller on a separate thread by
    // default. If you want to handle when the Controller updates manually you
    // can use `start_processing_retyrn_controller` instead
    let mut k = backend.start_processing(
        graph,
        resources,
        RunGraphSettings::default(),
        Box::new(controller::print_error_handler),
    )?;
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
    let node1 = k.push(
        WavetableOscillatorOwned::new(Wavetable::sine()),
        inputs!(("freq" : 220.)),
    );
    let modulator = k.push(
        WavetableOscillatorOwned::new(Wavetable::sine()),
        inputs!(("freq" : 3.)),
    );
    let mod_amp = k.push(Mult, inputs!((0 ; modulator.out(0)), (1 : 0.25)));
    let amp = k.push(
        Mult,
        inputs!((0 ; node1.out(0)), (1 : 0.5 ; mod_amp.out(0))),
    );
    k.connect(amp.to_graph_out().to_index(1));
    let mut input = String::new();
    loop {
        match std::io::stdin().read_line(&mut input) {
            Ok(_n) => {
                let input = input.trim();
                if let Ok(freq) = input.parse::<usize>() {
                    k.schedule_change(ParameterChange::now(node0.input("freq"), freq as f32));
                    println!("Setting freq to {freq}");
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
