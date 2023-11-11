use anyhow::Result;
use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions, JackBackend},
    controller::print_error_handler,
    envelope::Envelope,
    graph,
    handles::{graph_output, handle, Handle},
    modal_interface::commands,
    prelude::{delay::static_sample_delay, *},
    sphere::{KnystSphere, SphereSettings},
};
use rand::{thread_rng, Rng};
fn main() -> Result<()> {
    // let mut backend = CpalBackend::new(CpalBackendOptions::default())?;
    let mut backend = JackBackend::new("Knyst<3JACK")?;
    let _sphere = KnystSphere::start(
        &mut backend,
        SphereSettings {
            num_inputs: 0,
            num_outputs: 2,
            ..Default::default()
        },
        print_error_handler,
    );
    let sin_wt = Wavetable::sine();
    let wt = commands().insert_wavetable(sin_wt);

    // let root_freq = var(440.); // todo: add nodes which are a constant value, could be a Bus(1) or a special Gen // todo: add nodes which are a constant value, could be a Bus(1) or a special Gen
    // let mut rng = thread_rng();
    // for _ in 0..10 {
    //     let freq = (sine().freq(
    //         sine()
    //             .freq(sine().freq(0.01).range(0.02, rng.gen_range(0.05..0.3_f32)))
    //             .range(0.0, 400.),
    //     ) * 100.0)
    //         + 440.;
    //     // let freq = sine().freq(0.5).range(200.0, 200.0 * 9.0 / 8.0);
    //     let node0 = sine();
    //     node0.freq(freq);
    //     let modulator = sine();
    //     modulator.freq(sine().freq(0.09) * -5.0 + 6.0);
    //     graph_output(0, (node0 * modulator * 0.0025).repeat_outputs(1));
    // }

    // let mut rng = thread_rng();
    // for _ in 0..10 {
    //     let freq = (sine().freq(
    //         sine(wt)
    //             .freq(
    //                 sine(wt)
    //                     .freq(0.01)
    //                     .range(0.02, rng.gen_range(0.05..0.3 as Sample)),
    //             )
    //             .range(0.0, 400.),
    //     ) * 100.0)
    //         + 440.;
    //     // let freq = sine().freq(0.5).range(200.0, 200.0 * 9.0 / 8.0);
    //     let node0 = sine();
    //     node0.freq(freq);
    //     let modulator = sine();
    //     modulator.freq(sine().freq(0.09) * -5.0 + 6.0);
    //     graph_output(0, (node0 * modulator * 0.025).repeat_outputs(1));
    // }

    for &freq in [400, 600, 500].iter().cycle() {
        // new graph
        commands().init_local_graph(commands().default_graph_settings());
        let sig = sine().freq(freq as f32).out("sig") * 0.25;
        let env = Envelope {
            points: vec![(1.0, 0.005), (0.0, 0.5)],
            stop_action: StopAction::FreeGraph,
            ..Default::default()
        };
        let sig = sig * handle(env.to_gen());
        // let sig = sig * handle(env.to_gen());

        graph_output(0, sig.repeat_outputs(1));
        // push graph to sphere
        let graph = commands().upload_local_graph();
        let sig = graph + static_sample_delay(48 * 500).input(graph);

        graph_output(0, sig.repeat_outputs(1));
        std::thread::sleep(std::time::Duration::from_millis(2500));
    }

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

// fn sine() -> NodeHandle<WavetableOscillatorOwnedHandle> {
//     wavetable_oscillator_owned(Wavetable::sine())
// }
fn sine() -> Handle<OscillatorHandle> {
    oscillator(WavetableId::cos())
}
