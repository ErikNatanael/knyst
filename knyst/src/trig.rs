//! # Triggers
//! Triggers in Knyst are simply a sample with a value above 0. This module contains some tools to work with triggers.
//!
//! Triggers are useful for determining when things happen in a precise way e.g.
//! triggering an envelope to restart, starting a new grain in a granular
//! synthesizer Gen or signalling that it is time for a new value.
//!
use crate as knyst;
use knyst::{gen::GenState, Sample, SampleRate};
use knyst_macro::impl_gen;

use crate::time::Seconds;

/// Returns true is `sample` is a trigger, otherwise false.
#[inline(always)]
pub fn is_trigger(sample: Sample) -> bool {
    sample > 0.
}

/// For each Sample in `inputs`, set the corresponding `outputs` to true if the
/// Sample is a trigger, otherwise to false.
#[inline]
pub fn is_trigger_in_place(inputs: &[Sample], outputs: &mut [bool]) {
    for (inp, out) in inputs.iter().zip(outputs.iter_mut()) {
        *out = *inp > 0.
    }
}

/// Sends one trigger immediately and then frees itself.
/// *outputs*
/// 0. "trig": The trigger
pub struct OnceTrig(bool);

#[impl_gen]
impl OnceTrig {
    #[new]
    #[allow(missing_docs)]
    pub fn new() -> Self {
        OnceTrig(false)
    }
    #[process]
    /// Process one block
    pub fn process(&mut self, trig: &mut [Sample]) -> GenState {
        let out = trig;
        if self.0 {
            for o in out.iter_mut() {
                *o = 0.
            }
        } else {
            // If we haven't triggered yet, send a trigger on the first sample and then nothing.
            out[0] = 1.0;
            self.0 = true;
            for o in out.iter_mut().skip(1) {
                *o = 0.
            }
        }
        GenState::FreeSelf
    }
}

/// Send a trigger at a constant interval.
/// *inputs*
/// 0. "interval": The interval at which the trigger is being sent in seconds.
/// *outputs*
/// 0. "trig": A trigger sent at the interval.
pub struct IntervalTrig {
    // counter: Vec<Sample>,
    counter: Seconds,
}

#[impl_gen]
impl IntervalTrig {
    #[new]
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {
            counter: Seconds::ZERO,
        }
    }
    #[process]
    fn process(
        &mut self,
        sample_rate: SampleRate,
        interval: &[Sample],
        trig: &mut [Sample],
    ) -> GenState {
        let one_sample = Seconds::from_samples(1, *sample_rate as u64);
        for (interval, trig_out) in interval.iter().zip(trig.iter_mut()) {
            // Adding first makes the time until the first trigger the same as
            // the time between subsequent triggers so it is more consistent.
            self.counter += one_sample;
            let interval_as_seconds = Seconds::from_seconds_f64(*interval as f64);
            *trig_out = if self.counter >= interval_as_seconds {
                self.counter = self
                    .counter
                    .checked_sub(interval_as_seconds)
                    .expect("Counter was checked to be bigger than or equal to the interval so the subtraction should always work.");
                1.0
            } else {
                0.0
            };
        }
        GenState::Continue
    }
    #[init]
    fn init(&mut self) {
        self.counter = Seconds::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::resources::{Resources, ResourcesSettings};

    use crate::prelude::*;
    use crate::time::Seconds;
    use crate::trig::IntervalTrig;
    use crate::*;
    #[test]
    fn regular_interval_trig() {
        const SR: u64 = 44100;
        const BLOCK_SIZE: usize = 128 as usize;
        let graph_settings = GraphSettings {
            block_size: BLOCK_SIZE,
            sample_rate: SR as Sample,
            num_outputs: 2,
            ..Default::default()
        };
        let mut graph = Graph::new(graph_settings);
        let node = graph.push(IntervalTrig::new());
        graph.connect(node.to_graph_out()).unwrap();
        let every_8_samples = Seconds::from_samples(8, SR).to_seconds_f64();
        graph
            .connect(constant(every_8_samples as Sample).to(node))
            .unwrap();
        let (mut run_graph, _, _) = RunGraph::new(
            &mut graph,
            Resources::new(ResourcesSettings::default()),
            RunGraphSettings {
                scheduling_latency: Duration::new(0, 0),
            },
        )
        .unwrap();
        graph.update();
        run_graph.process_block();
        // The 8th sample should be 1.0 and then every 8th sample after that
        for (i, out) in run_graph
            .graph_output_buffers()
            .get_channel(0)
            .iter()
            .skip(7)
            .enumerate()
        {
            assert_eq!(
                *out,
                if i % 8 == 0 { 1.0 } else { 0.0 },
                "Failed at sample {i}"
            );
        }
    }
}
