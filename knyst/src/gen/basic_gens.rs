//! Includes basic `Gen`s such as `Mul` and `Range`

use crate::{
    gen::{Gen, GenContext, GenState},
    Resources,
};

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
                for _ in 0..block_size / simd_width {
                    let s_in0 = f32x2::from_slice(&value0[..simd_width]);
                    let s_in1 = f32x2::from_slice(&value1[..simd_width]);
                    let product = s_in0 * s_in1;
                    product.copy_to_slice(out_buf);
                    in0 = &value0[simd_width..];
                    in1 = &value1[simd_width..];
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

/// RampGen is used when calling `.ramp` on a Handle. Remaps each input from 0..=1 to min..=max. Unusually, the first input is not the signal processed, but the min and max values.
///
/// The number of channels of this Gen is determined at instantiation, i.e. when calling `.ramp`.
///
/// *inputs*
/// 0. "min": The output min value
/// 1. "max": The output max value
/// 2..N: The input channels
pub struct RampGen(pub usize);
impl Gen for RampGen {
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
        "RampGen"
    }
}
