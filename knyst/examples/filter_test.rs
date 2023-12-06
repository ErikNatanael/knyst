
use anyhow::Result;
use knyst::gen::filter::svf::{SvfFilterType, general_svf_filter};
#[allow(unused)]
use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions, JackBackend},
    controller::print_error_handler,
    envelope::Envelope,
    handles::{graph_output, handle, Handle},
    modal_interface::knyst,
    prelude::{delay::static_sample_delay, *},
    sphere::{KnystSphere, SphereSettings},
};
fn main() -> Result<()> {
    // let mut backend = CpalBackend::new(CpalBackendOptions::default())?;
    let mut backend = JackBackend::new("Knyst<3JACK")?;
    let _sphere = KnystSphere::start(
        &mut backend,
        SphereSettings {
            num_inputs: 1,
            num_outputs: 1,
            ..Default::default()
        },
        print_error_handler,
    );

    let filter = general_svf_filter(SvfFilterType::Band, 1000., 1.0 ,-20.).input(graph_input(0, 1));
    graph_output(0, filter);

    // graph_output(0, (sine(wt).freq(200.)).repeat_outputs(1));

    let mut input = String::new();
    loop {
        match std::io::stdin().read_line(&mut input) {
            Ok(n) => {
                println!("{} bytes read", n);
                println!("{}", input.trim());
                let input = input.trim();
                if let Ok(freq) = input.parse::<usize>() {
                    // node0.freq(freq as f32);
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

fn sine() -> Handle<OscillatorHandle> {
    oscillator(WavetableId::cos())
}
