use anyhow::Result;
use knyst::gen::filter::svf::{svf_filter, SvfFilterType};
#[allow(unused)]
use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions, JackBackend},
    controller::print_error_handler,
    envelope::Envelope,
    handles::{graph_output, handle, Handle},
    modal_interface::knyst_commands,
    prelude::{delay::static_sample_delay, *},
    sphere::{KnystSphere, SphereSettings},
};

fn main() -> Result<()> {
    let _backend = setup();

    let filter = svf_filter(SvfFilterType::Band, 1000., 10000.0, -20.)
        .input(graph_input(0, 1) + pink_noise());
    graph_output(0, filter * 0.01);
    // graph_output(0, white_noise());

    // Quick and dirty analysis of the amplitude of the input signal
    let mut average = 0.0;
    let mut peak = 0.0;
    let mut peak_env = 0.0;
    let analyser = handle(
        gen(move |ctx, _| {
            let noise = ctx.inputs.get_channel(0);
            let rms: f32 = (noise.iter().map(|v| *v * *v).sum::<f32>() / noise.len() as f32).sqrt();
            average += (rms.abs() - average) * 0.01;
            let peak_this_block = noise
                .iter()
                .map(|v| v.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            peak = peak_this_block.max(peak);
            peak_env = peak_this_block.max(peak_env);
            peak_env += (peak_this_block - peak_env) * 0.002;
            println!(
                "avg: {average:.3}, peak: {peak:.3}, peak_env: {peak_env:.3}, crest factor: {:.3}",
                peak / average
            );
            GenState::Continue
        })
        .input("noise")
        .output("output"),
    );
    analyser.set("noise", pink_noise());

    // graph_output(0, (sine(wt).freq(200.)).repeat_outputs(1));

    // Wait for ENTER
    println!("Press ENTER to exit");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(())
}

/// Initializes the audio backend and starts a `KnystSphere` for audio processing.
/// Start with an automatic helper thread for scheduling changes and managing resources.
/// If you want to manage the `Controller` yourself, use `start_return_controller`.
///
/// The backend is returned here because it would otherwise be dropped at the end of setup()
fn setup() -> impl AudioBackend {
    let mut backend =
        CpalBackend::new(CpalBackendOptions::default()).expect("Unable to connect to CPAL backend");
    // Uncomment the line below and comment the line above to use the JACK backend instead
    // let mut backend = JackBackend::new("Knyst<3JACK").expect("Unable to start JACK backend");

    let _sphere_id = KnystSphere::start(
        &mut backend,
        SphereSettings {
            num_inputs: 1,
            num_outputs: 1,
            ..Default::default()
        },
        print_error_handler,
    )
    .ok();
    backend
}
