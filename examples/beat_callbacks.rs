use anyhow::Result;
use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions},
    controller::print_error_handler,
    graph::Mult,
    prelude::*,
    time::Superbeats,
    wavetable::{Wavetable, WavetableOscillatorOwned},
};
use rand::{thread_rng, Rng};
use std::time::Duration;

fn main() -> Result<()> {
    let mut backend = CpalBackend::new(CpalBackendOptions::default())?;

    let sample_rate = backend.sample_rate() as f32;
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
            print_error_handler,
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

    let mut i = 0;
    // Start a callback on the controller thread. The callback is called before
    // the time when the next events should be scheduled to give the scheduler
    // some time to send the changes to the audio thread in time. The callback
    // is given the timestamp for the next beat that things should be scheduled
    // at. The time between calls to the callback is determined by the return
    // value of the previous callback. If you return None the callback will not
    // be called again.
    let callback = k.schedule_beat_callback(
        move |time, k| {
            println!("Callback called {i} for time {time:?}");
            let mut rng = thread_rng();
            let freq = rng.gen_range(200..600);
            k.schedule_change(ParameterChange::beats(node0.clone(), freq as f32, time).l("freq"));
            i += 1;
            if time > Superbeats::from_beats(32) {
                None
            } else {
                if i % 2 == 0 {
                    Some(Superbeats::from_beats(2))
                } else {
                    Some(Superbeats::from_beats(3))
                }
            }
        },
        Superbeats::from_beats(4), // Time for the first time the
    );
    let mut input = String::new();
    loop {
        match std::io::stdin().read_line(&mut input) {
            Ok(n) => {
                println!("{} bytes read", n);
                println!("{}", input.trim());
                let input = input.trim();
                if let Ok(_freq) = input.parse::<usize>() {
                    // k.schedule_change(ParameterChange::now(node0.clone(), freq as f32).l("freq"));
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
