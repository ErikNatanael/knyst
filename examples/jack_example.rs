use anyhow::Result;
use knyst::{
    audio_backend::JackBackend,
    controller,
    graph::Mult,
    prelude::*,
    wavetable::{Wavetable, WavetableOscillatorOwned},
};

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
        controller::print_error_handler,
    )?;
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
    let node1 = k.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    k.connect(constant(220.).to(&node1).to_label("freq"));
    let modulator = k.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    k.connect(constant(3.).to(&modulator).to_label("freq"));
    let mod_amp = k.push(Mult);
    k.connect(modulator.to(&mod_amp));
    k.connect(constant(0.25).to(&mod_amp).to_index(1));
    let amp = k.push(Mult);
    k.connect(node1.to(&amp));
    k.connect(constant(0.5).to(&amp).to_index(1));
    k.connect(mod_amp.to(&amp).to_index(1));
    k.connect(amp.to_graph_out().to_index(1));
    let mut input = String::new();
    loop {
        match std::io::stdin().read_line(&mut input) {
            Ok(_n) => {
                let input = input.trim();
                if let Ok(freq) = input.parse::<usize>() {
                    k.schedule_change(ParameterChange::now(node0.clone(), freq as f32).l("freq"));
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
