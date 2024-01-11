use std::time::Duration;

use anyhow::Result;
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
use knyst::{
    envelope::envelope_gen,
    gen::filter::svf::{svf_dynamic, svf_filter, SvfFilterType},
};
use rand::{thread_rng, Rng};
fn main() -> Result<()> {
    let mut backend = CpalBackend::new(CpalBackendOptions::default())?;
    // let mut backend = JackBackend::new("Knyst<3JACK")?;
    let _sphere = KnystSphere::start(
        &mut backend,
        SphereSettings {
            num_inputs: 1,
            num_outputs: 1,
            ..Default::default()
        },
        print_error_handler,
    );

    let mut rng = thread_rng();
    // loop {
    //     let freq = rng.gen_range(100.0..1000.0);
    //     let length = rng.gen_range(0.5..4.0);
    //     dbg!(freq, length);
    //     spawn_filtered_noise(freq, length);
    //     std::thread::sleep(Duration::from_secs_f32(length));
    // }
    loop {
        let mut chord = [1.0, 5. / 4., 3. / 2., 17. / 8., 7. / 4.];
        let freq = rng.gen_range(100.0..200.0);
        for f in &mut chord {
            *f *= freq;
        }
        changed_harmony_chord(&chord);

        std::thread::sleep(Duration::from_secs_f32(10.));
    }

    // graph_output(0, white_noise());

    // graph_output(0, (sine(wt).freq(200.)).repeat_outputs(1));

    // Wait for ENTER
    println!("Press ENTER to exit");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(())
}

fn spawn_filtered_noise(freq: f32, length: f32) {
    let mut rng = thread_rng();
    let filtered_noise = upload_graph(knyst_commands().default_graph_settings(), || {
        let env = envelope_gen(
            0.0,
            vec![
                (1.0, rng.gen_range(0.7..1.9)),
                (0.5, length * 0.34),
                (0.0, length * 0.66),
            ],
            knyst::envelope::SustainMode::NoSustain,
            StopAction::FreeGraph,
        );
        let source = pink_noise();
        let sig = svf_filter(
            SvfFilterType::Band,
            freq,
            rng.gen_range(2000.0..10000.),
            0.0,
        )
        .input(source);
        let sig = sig * env * 0.01;
        graph_output(0, sig.channels(2));
    });
    graph_output(0, filtered_noise);
}

fn changed_harmony_chord(new_chord: &[f32]) {
    let mut rng = thread_rng();
    if rng.gen::<f32>() > 0.4 {
        for f in new_chord {
            let length = rng.gen_range(6.0..14.0);
            let speaker = rng.gen_range(0..4);
            let filtered_noise = upload_graph(knyst_commands().default_graph_settings(), || {
                let env = envelope_gen(
                    0.0,
                    vec![(1.0, 3.), (1.0, length - 5.), (0.0, 2.)],
                    knyst::envelope::SustainMode::NoSustain,
                    StopAction::FreeGraph,
                );
                let source = white_noise();
                let mut sigs = vec![];
                for i in 0..5 {
                    let freq_detune = [1.0, 1.001, 0.999, 1.002, 0.998][i];
                    let q_env = envelope_gen(
                        1.0 / rng.gen_range(0.001..0.008),
                        vec![(1. / 0.0003, length)],
                        knyst::envelope::SustainMode::NoSustain,
                        StopAction::Continue,
                    );

                    let sig = svf_dynamic(SvfFilterType::Band)
                        .cutoff_freq(f * freq_detune)
                        .q(q_env)
                        .gain(0.0)
                        .input(source);
                    sigs.push(sig);
                }
                let sig = sigs[0] + sigs[1] + sigs[2] + sigs[3] + sigs[4];
                let sig = sig * env * 0.01;
                graph_output(speaker, sig);
            });
            graph_output(0, filtered_noise);
        }
    }
}
