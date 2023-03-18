use anyhow::Result;
use knyst::{
    audio_backend::JackBackend,
    controller,
    filter::{hiir::Downsampler2X, HalfbandFilter},
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
        num_inputs: 2,
        num_outputs: 3,
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

    let inner_graph = Graph::new(GraphSettings {
        block_size,
        sample_rate,
        oversampling: knyst::graph::Oversampling::X2,
        num_inputs: 0,
        num_outputs: 1,
        ..Default::default()
    });
    let inner_graph_id = inner_graph.id();
    let inner_graph = k.push(inner_graph, inputs!());
    let freq_saw = k.push_to_graph(
        WavetableOscillatorOwned::new(Wavetable::saw()),
        inner_graph_id,
        inputs!(("freq" : 0.1)),
    );
    let freq_amp = k.push_to_graph(
        Mult,
        inner_graph_id,
        inputs!((0 ; freq_saw.out(0)), (1 : 10000.)),
    );
    let saw = k.push_to_graph(
        WavetableOscillatorOwned::new(Wavetable::saw()),
        inner_graph_id,
        inputs!(("freq" : 10300. ; freq_amp.out(0))),
    );
    let amp = k.push_to_graph(Mult, inner_graph_id, inputs!((0 ; saw.out(0)), (1 : 0.5)));
    k.connect(amp.to_graph_out());
    k.connect(inner_graph.to_graph_out());
    let freq_saw = k.push(
        WavetableOscillatorOwned::new(Wavetable::saw()),
        inputs!(("freq" : 0.1)),
    );
    let freq_amp = k.push(Mult, inputs!((0 ; freq_saw.out(0)), (1 : 10000.)));
    let saw = k.push(
        WavetableOscillatorOwned::new(Wavetable::saw()),
        inputs!(("freq" : 10300. ; freq_amp.out(0))),
    );
    let amp = k.push(Mult, inputs!((0 ; saw.out(0)), (1 : 0.5)));
    k.connect(amp.to_graph_out().to_channel(1));

    // Test filter input from 1, output on 3
    let mut downsampler_filter = Downsampler2X::new(12);
    let coefs = vec![
        0.017347915108876406,
        0.067150480426919179,
        0.14330738338179819,
        0.23745131944299824,
        0.34085550201503761,
        0.44601111310335906,
        0.54753112652956148,
        0.6423859124721446,
        0.72968928615804163,
        0.81029959388029904,
        0.88644514917318362,
        0.96150605146543733,
    ];
    downsampler_filter.set_coefs(coefs);
    let filter = k.push(downsampler_filter, inputs!());
    k.connect(Connection::graph_input(&filter));
    k.connect(filter.to_graph_out().to_channel(2));

    let mut input = String::new();
    loop {
        match std::io::stdin().read_line(&mut input) {
            Ok(_n) => {
                let input = input.trim();
                if input == "q" {
                    break;
                }
            }
            Err(error) => println!("error: {}", error),
        }
        input.clear();
    }
    Ok(())
}
