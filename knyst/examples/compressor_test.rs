use anyhow::Result;
use knyst::audio_backend::{CpalBackend, CpalBackendOptions};
use knyst::{
    audio_backend::JackBackend, controller::print_error_handler,
    gen::dynamics::randja_compressor::RandjaCompressor, prelude::*,
};

fn main() -> Result<()> {
    let mut backend =
        CpalBackend::new(CpalBackendOptions::default()).expect("Unable to connect to CPAL backend");
    // Uncomment the line below and comment the line above to use the JACK backend instead
    // let mut backend = JackBackend::new("Knyst<3JACK").expect("Unable to start JACK backend");

    let sample_rate = backend.sample_rate() as Sample;
    let block_size = backend.block_size().unwrap_or(64);
    println!("sr: {sample_rate}, block: {block_size}");

    let _sphere = KnystSphere::start(
        &mut backend,
        SphereSettings {
            num_inputs: 2,
            num_outputs: 2,
            ..Default::default()
        },
        print_error_handler,
    )?;
    let threshold = 0.1;
    let attack = 0.01;
    let release = 0.03;

    let mut com = RandjaCompressor::new();
    com.set_threshold(threshold);
    com.set_attack(attack * sample_rate);
    com.set_release(release * sample_rate);
    com.set_ratio(1.0 / 100.);
    com.set_output(1.0);

    let com = com
        .upload()
        .input_left(graph_input(0, 1))
        .input_right(graph_input(1, 1));

    graph_output(0, com);
    let mut compressor_on = true;

    let mut input = String::new();
    loop {
        println!("Enter t to toggle compressor bypass, q to exit");
        match std::io::stdin().read_line(&mut input) {
            Ok(_n) => {
                let input = input.trim();
                if let Ok(_freq) = input.parse::<usize>() {
                } else if input == "t" {
                    if compressor_on {
                        println!("Turning compressor off");
                        com.clear_graph_output_connections();
                        graph_output(0, graph_input(0, 2));
                    } else {
                        println!("Turning compressor on");
                        graph_input(0, 2).clear_graph_output_connections();
                        graph_output(0, com);
                    }
                    compressor_on = !compressor_on;
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
