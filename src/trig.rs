use crate::{
    graph::{self, Gen, GenState},
    Sample,
};

/// # Triggers
/// Triggers in Knyst are simply a sample with a value above 0. This module contains some tools to work with triggers.
///
/// Triggers are useful for determining when things happen in a precise way e.g.
/// triggering an envelope to restart, starting a new grain in a granular
/// synthesizer Gen or signalling that it is time for a new value.
///

pub fn is_trigger(sample: Sample) -> bool {
    sample > 0.
}

pub fn is_trigger_in_place(inputs: &[Sample], outputs: &mut [bool]) {
    for (inp, out) in inputs.iter().zip(outputs.iter_mut()) {
        *out = *inp > 0.
    }
}

pub struct OnceTrig(bool);

impl OnceTrig {
    pub fn new() -> Self {
        OnceTrig(false)
    }
}

impl Gen for OnceTrig {
    fn process(
        &mut self,
        ctx: graph::GenContext,
        _resources: &mut crate::Resources,
    ) -> graph::GenState {
        let out = ctx.outputs.split_mut().next().unwrap();
        if self.0 {
            for o in out.iter_mut() {
                *o = 0.
            }
        } else {
            // If we haven't triggered yet, send a trigger on the first sample and then nothing.
            out[0] = 1.0;
            for o in out.iter_mut().skip(1) {
                *o = 0.
            }
        }
        GenState::FreeSelf
    }
    fn num_inputs(&self) -> usize {
        0
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_desc(&self, _output: usize) -> &'static str {
        "trig"
    }
}

pub struct IntervalTrig {
    counter: u64,
}

impl Gen for IntervalTrig {
    fn process(
        &mut self,
        ctx: graph::GenContext,
        _resources: &mut crate::Resources,
    ) -> graph::GenState {
        let intervals_in_seconds = ctx.inputs.get_channel(0);
        let output = ctx.outputs.split_mut().next().unwrap();
        for (interval, trig_out) in intervals_in_seconds.iter().zip(output.iter_mut()) {
            let interval_sample = *interval * ctx.sample_rate;
            *trig_out = if self.counter >= interval_sample as u64 {
                self.counter = 0;
                1.0
            } else {
                0.0
            };
        }
        GenState::Continue
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn input_desc(&self, _input: usize) -> &'static str {
        "interval"
    }
    fn output_desc(&self, _output: usize) -> &'static str {
        "trig"
    }
}
