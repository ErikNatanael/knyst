//! Includes basic `Gen`s such as `Mul` and `Range`

use crate::{
    self as knyst,
    graph::NodeId,
    handles::{Handle, HandleData, Input, NodeIdIter, SinkChannelIter, SourceChannelIter},
    prelude::{KnystCommands, Seconds},
    SampleRate,
};
use knyst_macro::impl_gen;

use crate::{
    gen::{Gen, GenContext, GenState},
    BlockSize, Resources, Sample,
};

/// SubGen(num out channels, or number of pairs of inputs) - subtraction
///
/// Variable number channel Sub Gen. Every pair of inputs are subtracted input0 - input1 into one output.
pub struct SubGen(pub usize);
impl Gen for SubGen {
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        let block_size = ctx.block_size();
        let mut out_bufs = ctx.outputs.iter_mut();

        for i in 0..self.0 {
            let product = out_bufs.next().unwrap();
            let value0 = ctx.inputs.get_channel(i * 2);
            let value1 = ctx.inputs.get_channel(i * 2 + 1);

            // fallback
            #[cfg(not(feature = "unstable"))]
            {
                for i in 0..block_size {
                    product[i] = value0[i] - value1[i];
                }
            }
            #[cfg(feature = "unstable")]
            {
                use std::simd::f32x2;
                let simd_width = 2;
                let mut out_buf = product;
                let mut value0 = value0;
                let mut value1 = value1;
                for _ in 0..block_size / simd_width {
                    let s_in0 = f32x2::from_slice(&value0[..simd_width]);
                    let s_in1 = f32x2::from_slice(&value1[..simd_width]);
                    let product = s_in0 - s_in1;
                    product.copy_to_slice(out_buf);
                    value0 = &value0[simd_width..];
                    value1 = &value1[simd_width..];
                    out_buf = &mut out_buf[simd_width..];
                }
            }
        }
        GenState::Continue
    }

    fn num_inputs(&self) -> usize {
        self.0 * 2
    }

    fn num_outputs(&self) -> usize {
        self.0
    }

    fn name(&self) -> &'static str {
        "SubGen"
    }
}

/// PowGen(num out channels)
///
/// Variable number of channels at creation. The first input is the exponent, remaining inputs are taken
pub struct PowfGen(pub usize);
impl Gen for PowfGen {
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        let block_size = ctx.block_size();
        let mut out_bufs = ctx.outputs.iter_mut();
        let exponent = ctx.inputs.get_channel(0);

        for i in 0..self.0 {
            let out = out_bufs.next().unwrap();
            let value = ctx.inputs.get_channel(i + 1);

            // fallback
            // #[cfg(not(feature = "unstable"))]
            {
                for i in 0..block_size {
                    out[i] = fastapprox::fast::pow(value[i] as f32, exponent[i] as f32) as Sample;
                }
            }
            // #[cfg(feature = "unstable")]
            // {
            //     use std::simd::f32x2;
            //     let simd_width = 2;
            //     for _ in 0..block_size / simd_width {
            //         let s_in0 = f32x2::from_slice(&value0[..simd_width]);
            //         let s_in1 = f32x2::from_slice(&value1[..simd_width]);
            //         let product = s_in0 * s_in1;
            //         product.copy_to_slice(out_buf);
            //         in0 = &value0[simd_width..];
            //         in1 = &value1[simd_width..];
            //         out_buf = &mut out_buf[simd_width..];
            //     }
            // }
        }
        GenState::Continue
    }

    fn num_inputs(&self) -> usize {
        self.0 + 1
    }

    fn num_outputs(&self) -> usize {
        self.0
    }

    fn name(&self) -> &'static str {
        "PowfGen"
    }
}
/// base to the power of exponent, where base and exponent can be constants or handles
pub fn powf_gen(base: impl Into<Input>, exponent: impl Into<Input>) -> Handle<PowfHandle> {
    let base = base.into();
    let (node_id, num_channels) = match base {
        Input::Constant(c) => {
            let node_id = knyst::knyst_commands().push_without_inputs(PowfGen(1));
            knyst::knyst_commands().connect(
                knyst::graph::connection::constant(c)
                    .to(node_id)
                    .to_channel(1),
            );
            (node_id, 1)
        }
        Input::Handle { output_channels } => {
            let connecting_channels: Vec<_> = output_channels.collect();
            let num_channels = connecting_channels.len();
            let node_id =
                knyst::knyst_commands().push_without_inputs(PowfGen(connecting_channels.len()));
            for (i, (source, chan)) in connecting_channels.into_iter().enumerate() {
                knyst::knyst_commands()
                    .connect(source.to(node_id).from_channel(chan).to_channel(i + 1));
            }

            (node_id, num_channels)
        }
    };
    Handle::new(PowfHandle {
        node_id,
        num_channels,
    })
    .exponent(exponent)
}
/// Handle to a PowfGen
#[derive(Copy, Clone, Debug)]
pub struct PowfHandle {
    pub(crate) node_id: NodeId,
    pub(crate) num_channels: usize,
}
impl PowfHandle {
    /// Set the exponent
    pub fn exponent(self, exponent: impl Into<Input>) -> Handle<Self> {
        let inp = exponent.into();
        match inp {
            Input::Constant(v) => {
                knyst::knyst_commands().connect(
                    crate::graph::connection::constant(v)
                        .to(self.node_id)
                        .to_channel(0),
                );
            }
            Input::Handle { output_channels } => {
                for (node_id, chan) in output_channels {
                    crate::modal_interface::knyst_commands()
                        .connect(node_id.to(self.node_id).from_channel(chan).to_channel(0));
                }
            }
        }
        Handle::new(self)
    }
}
impl HandleData for PowfHandle {
    fn out_channels(&self) -> SourceChannelIter {
        SourceChannelIter::single_node_id(self.node_id, self.num_channels)
    }

    fn in_channels(&self) -> SinkChannelIter {
        SinkChannelIter::single_node_id(self.node_id, self.num_channels + 1)
    }

    fn node_ids(&self) -> NodeIdIter {
        NodeIdIter::Single(self.node_id)
    }
}

/// Mul(num out channels)
///
/// Variable number channel Mul Gen. Every pair of inputs are multiplied together into one output.
pub struct MulGen(pub usize);
impl Gen for MulGen {
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        let block_size = ctx.block_size();
        let mut out_bufs = ctx.outputs.iter_mut();

        for i in 0..self.0 {
            let product = out_bufs.next().unwrap();
            let value0 = ctx.inputs.get_channel(i * 2);
            let value1 = ctx.inputs.get_channel(i * 2 + 1);

            // fallback
            #[cfg(not(feature = "unstable"))]
            {
                for i in 0..block_size {
                    product[i] = value0[i] * value1[i];
                }
            }
            #[cfg(feature = "unstable")]
            {
                use std::simd::f32x2;
                let simd_width = 2;
                let mut out_buf = product;
                let mut value0 = value0;
                let mut value1 = value1;
                for _ in 0..block_size / simd_width {
                    let s_in0 = f32x2::from_slice(&value0[..simd_width]);
                    let s_in1 = f32x2::from_slice(&value1[..simd_width]);
                    let product = s_in0 * s_in1;
                    product.copy_to_slice(out_buf);
                    value0 = &value0[simd_width..];
                    value1 = &value1[simd_width..];
                    out_buf = &mut out_buf[simd_width..];
                }
            }
        }
        GenState::Continue
    }

    fn num_inputs(&self) -> usize {
        self.0 * 2
    }

    fn num_outputs(&self) -> usize {
        self.0
    }

    fn name(&self) -> &'static str {
        "MulGen"
    }
}
/// Bus(channels)
///
/// Convenience Gen for collecting many signals to one node address. Inputs will
/// be copied to the corresponding outputs.
pub struct Bus(pub usize);
impl Gen for Bus {
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        let mut out_bufs = ctx.outputs.iter_mut();
        for channel in 0..self.0 {
            let in_buf = ctx.inputs.get_channel(channel);
            let out_buf = out_bufs.next().unwrap();
            out_buf.copy_from_slice(in_buf);
        }
        GenState::Continue
    }

    fn num_inputs(&self) -> usize {
        self.0
    }

    fn num_outputs(&self) -> usize {
        self.0
    }

    fn name(&self) -> &'static str {
        "Bus"
    }
}

/// RangeGen is used when calling `.range(min, max)` on a Handle. Remaps each input from 0..=1 to min..=max. Unusually, the first input is not the signal processed, but the min and max values.
///
/// The number of channels of this Gen is determined at instantiation, i.e. when calling `.range`.
///
/// *inputs*
/// 0. "min": The output min value
/// 1. "max": The output max value
/// 2..N: The input channels
pub struct RangeGen(pub usize);
impl Gen for RangeGen {
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        let block_size = ctx.block_size();
        let mut out_bufs = ctx.outputs.iter_mut();

        let min = ctx.inputs.get_channel(0);
        let max = ctx.inputs.get_channel(1);

        for i in 0..self.0 {
            let out = out_bufs.next().unwrap();
            let value0 = ctx.inputs.get_channel(i + 2);
            for f in 0..block_size {
                let width = max[f] - min[f];
                out[f] = value0[f] * width + min[f];
            }
        }
        GenState::Continue
    }

    fn num_inputs(&self) -> usize {
        self.0 + 2
    }

    fn num_outputs(&self) -> usize {
        self.0
    }

    fn name(&self) -> &'static str {
        "RangeGen"
    }
}

/// Pan a mono signal to stereo using the cos/sine pan law. Pan value should be
/// between -1 and 1, 0 being in the center.
///
/// ```rust
/// use knyst::prelude::*;
/// use knyst::graph::RunGraph;
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let sample_rate = 44100.;
///     let block_size = 8;
///     let resources = Resources::new(ResourcesSettings::default());
///     let graph_settings = GraphSettings {
///         block_size,
///         sample_rate,
///         num_outputs: 2,
///         ..Default::default()
///     };
///     let mut graph: Graph = Graph::new(graph_settings);
///     let pan = graph.push(PanMonoToStereo);
///     // The signal is a constant 1.0
///     graph.connect(constant(1.).to(pan).to_label("signal"))?;
///     // Pan to the left
///     graph.connect(constant(-1.).to(pan).to_label("pan"))?;
///     graph.connect(pan.to_graph_out().channels(2))?;
///     graph.commit_changes();
///     graph.update();
///     let (mut run_graph, _, _) = RunGraph::new(&mut graph, resources, RunGraphSettings::default())?;
///     run_graph.process_block();
///     assert!(run_graph.graph_output_buffers().read(0, 0) > 0.9999);
///     assert!(run_graph.graph_output_buffers().read(1, 0) < 0.0001);
///     // Pan to the right
///     graph.connect(constant(1.).to(pan).to_label("pan"))?;
///     graph.commit_changes();
///     graph.update();
///     run_graph.process_block();
///     assert!(run_graph.graph_output_buffers().read(0, 0) < 0.0001);
///     assert!(run_graph.graph_output_buffers().read(1, 0) > 0.9999);
///     // Pan to center
///     graph.connect(constant(0.).to(pan).to_label("pan"))?;
///     graph.commit_changes();
///     graph.update();
///     run_graph.process_block();
///     assert_eq!(run_graph.graph_output_buffers().read(0, 0), 0.7070929);
///     assert_eq!(run_graph.graph_output_buffers().read(1, 0), 0.7070929);
///     assert_eq!(
///         run_graph.graph_output_buffers().read(0, 0),
///         run_graph.graph_output_buffers().read(1, 0)
///     );
///     Ok(())
/// }
/// ```
// TODO: Implement multiple different pan laws, maybe as a generic.
pub struct PanMonoToStereo;
#[impl_gen]
impl PanMonoToStereo {
    #[new]
    #[must_use]
    fn new() -> Self {
        Self
    }
    #[process]
    fn process(
        #[allow(unused)] &mut self,
        signal: &[Sample],
        pan: &[Sample],
        left: &mut [Sample],
        right: &mut [Sample],
        block_size: BlockSize,
    ) -> GenState {
        for i in 0..*block_size {
            let signal = signal[i];
            // The equation needs pan to be in the range [0, 1]
            let pan = pan[i] * 0.5 + 0.5;
            let pan_pos_radians = pan * std::f64::consts::FRAC_PI_2 as Sample;
            let left_gain = fastapprox::fast::cos(pan_pos_radians as f32) as Sample;
            let right_gain = fastapprox::fast::sin(pan_pos_radians as f32) as Sample;
            left[i] = signal * left_gain;
            right[i] = signal * right_gain;
        }
        GenState::Continue
    }
}

/// A linear interpolation between two numbers over some time
pub struct LineSegment {
    start: Sample,
    end: Sample,
    dur: Seconds,
    num_samples_left: usize,
    current_value: f64,
    step: f64,
}
#[impl_gen]
impl LineSegment {
    #[allow(missing_docs)]
    #[new]
    #[must_use]
    pub fn new(start: Sample, end: Sample, dur: Seconds) -> Self {
        Self {
            start,
            end,
            dur,
            num_samples_left: 0,
            current_value: start as f64,
            step: 0.,
        }
    }
    fn init(&mut self, sample_rate: SampleRate) {
        self.num_samples_left = self.dur.to_samples(*sample_rate as u64) as usize;
        self.step = (self.end as f64 - self.start as f64) / self.num_samples_left as f64;
        self.current_value = self.start as f64;
    }
    #[allow(missing_docs)]
    #[process]
    pub fn process(&mut self, output: &mut [Sample], block_size: BlockSize) -> GenState {
        if self.num_samples_left == 0 {
            output.fill(self.end);
        } else if self.num_samples_left < block_size.0 {
            for out in output {
                *out = self.current_value as Sample;
                if self.num_samples_left == 0 {
                    self.step = 0.;
                    self.current_value = self.end as f64;
                } else {
                    self.current_value += self.step;
                    self.num_samples_left -= 1;
                }
            }
        } else {
            for out in output.iter_mut() {
                *out = self.current_value as Sample;
                self.current_value += self.step;
            }
            self.num_samples_left -= output.len();
        }
        GenState::Continue
    }
}

/// A linear interpolation between two numbers over some time
pub struct ExpLineSegment {
    start: Sample,
    end: Sample,
    dur: Seconds,
    num_samples_left: usize,
    current_value: f64,
    coeff: f64,
}
#[impl_gen]
impl ExpLineSegment {
    #[allow(missing_docs)]
    #[new]
    #[must_use]
    pub fn new(start: Sample, end: Sample, dur: Seconds) -> Self {
        Self {
            start,
            end,
            dur,
            num_samples_left: 0,
            current_value: start as f64,
            coeff: 0.,
        }
    }
    fn init(&mut self, sample_rate: SampleRate) {
        self.num_samples_left = self.dur.to_samples(*sample_rate as u64) as usize;
        self.coeff = ((self.end / self.start) as f64).powf(1. / self.num_samples_left as f64);
        self.current_value = self.start as f64;
    }
    #[allow(missing_docs)]
    #[process]
    pub fn process(&mut self, output: &mut [Sample], block_size: BlockSize) -> GenState {
        if self.num_samples_left == 0 {
            output.fill(self.end);
        } else if self.num_samples_left < block_size.0 {
            for out in output {
                *out = self.current_value as Sample;
                if self.num_samples_left == 0 {
                    self.coeff = 1.;
                    self.current_value = self.end as f64;
                } else {
                    self.current_value *= self.coeff;
                    self.num_samples_left -= 1;
                }
            }
        } else {
            for out in output.iter_mut() {
                *out = self.current_value as Sample;
                self.current_value *= self.coeff;
            }
            self.num_samples_left -= output.len();
        }
        GenState::Continue
    }
}

#[cfg(test)]
mod tests {
    use crate::{handles::graph_output, offline::KnystOffline, prelude::Seconds};

    use super::{exp_line_segment, line_segment};

    #[test]
    fn line_segment_test() {
        let mut kt = KnystOffline::new(8, 8, 0, 1);
        graph_output(0, line_segment(1.0, 2.0, Seconds::from_seconds_f64(1.0)));
        kt.process_block();
        let output = kt.output_channel(0).unwrap();
        for i in 0..8 {
            assert_eq!(output[i], 1.0 + (i as f32 / 8.));
        }
        kt.process_block();
        let output = kt.output_channel(0).unwrap();
        for i in 0..8 {
            assert_eq!(output[i], 2.);
        }
    }
    #[test]
    fn exp_line_segment_test() {
        let sr = 192000;
        let mut kt = KnystOffline::new(sr, sr, 0, 1);
        graph_output(
            0,
            exp_line_segment(1.0, 2.0, Seconds::from_seconds_f64(1.0)),
        );
        kt.process_block();
        let output = kt.output_channel(0).unwrap();
        assert_eq!(output[0], 1.0);
        dbg!(output);
        let mut diff = output[1] - output[0];
        for i in 1..sr - 1 {
            let new_diff = output[i + 1] - output[0];
            assert!(new_diff > diff);
            diff = new_diff;
            // assert_eq!(output[i], 1.0 + (i as f32 / 8.));
        }
        assert!(output[sr - 1] < 2.0);
        kt.process_block();
        let output = kt.output_channel(0).unwrap();
        for i in 0..32 {
            assert_eq!(output[i], 2.);
        }
    }
}
