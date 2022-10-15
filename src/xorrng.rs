//! Simple pseudorandom number generator
//!
//! license: Public Domain
//!
//! [https://github.com/BillyDM/Fast-DSP-Approximations/blob/main/rng_and_noise.md](https://github.com/BillyDM/Fast-DSP-Approximations/blob/main/rng_and_noise.md)

#[derive(Clone, Copy)]
pub struct XOrShift32Rng {
    fpd: u32,
}

impl Default for XOrShift32Rng {
    fn default() -> XOrShift32Rng {
        XOrShift32Rng { fpd: 17 }
    }
}

impl XOrShift32Rng {
    pub fn new(mut seed: u32) -> XOrShift32Rng {
        // seed cannot be zero
        if seed == 0 {
            seed = 17;
        }
        XOrShift32Rng { fpd: seed }
    }

    #[inline]
    pub fn gen_u32(&mut self) -> u32 {
        self.fpd ^= self.fpd << 13;
        self.fpd ^= self.fpd >> 17;
        self.fpd ^= self.fpd << 5;
        self.fpd
    }

    #[inline]
    pub fn gen_f32(&mut self) -> f32 {
        self.gen_u32() as f32 / std::u32::MAX as f32
    }

    #[inline]
    pub fn gen_f64(&mut self) -> f64 {
        self.gen_u32() as f64 / std::u32::MAX as f64
    }
}
