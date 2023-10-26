/// This is ported and adapted from Laurent de Soras' HIIR library which
/// includes filters especially useful for oversampling.
/// http://ldesoras.free.fr/prod.html#src_hiir
///
/// Original license: DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE Version 2
///
/// TODO:
/// - Report the delay of the oversampling and automatically correct for it
use crate::Sample;

#[derive(Copy, Clone, Default, Debug)]
struct StageData {
    coef: f64,
    mem: f64,
}

#[derive(Clone)]
struct Downsampler2X {
    filter: Vec<StageData>,
    num_coefs: usize,
}
impl Downsampler2X {
    pub fn new(coefs: Vec<f64>) -> Self {
        let mut filter = vec![StageData::default(); coefs.len() + 2];
        for i in 2..(filter.len()) {
            filter[i].coef = coefs[i - 2];
        }

        Self {
            filter,
            num_coefs: coefs.len(),
        }
    }
    pub fn process_sample(&mut self, input: &mut [f64]) -> f64 {
        assert!(input.len() == 2);
        process_sample_pos(self.num_coefs, self.num_coefs, input, &mut self.filter);
        // DataType       spl_0 (in_ptr [1]);
        // DataType       spl_1 (in_ptr [0]);

        // StageProcTpl <NBR_COEFS, DataType>::process_sample_pos (
        // 	NBR_COEFS, spl_0, spl_1, _filter.data ()
        // );

        // return DataType (0.5f) * (spl_0 + spl_1);
        0.5 * (input[0] + input[1])
    }
    pub fn process_block(&mut self, input: &[Sample], output: &mut [Sample]) {
        // Input buffer twice as large because it is upsampled
        assert!(output.len() * 2 == input.len());
        for (inp, out) in input.chunks(2).zip(output.iter_mut()) {
            let mut inp_array = [inp[0] as f64, inp[1] as f64];
            *out = self.process_sample(&mut inp_array) as Sample;
        }
    }
    fn _clear_buffers(&mut self) {
        for d in &mut self.filter {
            d.mem = 0.0;
        }
    }
}

fn process_sample_pos(
    coefs_remaining: usize,
    num_coefs: usize,
    input: &mut [f64],
    filter: &mut Vec<StageData>,
) {
    match coefs_remaining {
        0 => {
            let i = num_coefs + 2;
            filter[i - 2].mem = input[0];
            filter[i - 1].mem = input[1];
        }
        1 => {
            let i = num_coefs + 2 - 1;
            let mut tmp0 = input[0];
            tmp0 -= filter[i].mem;
            tmp0 *= filter[i].coef;
            tmp0 += filter[i - 2].mem;

            filter[i - 2].mem = input[0];
            filter[i - 1].mem = input[1];
            filter[i].mem = tmp0;

            input[0] = tmp0;
        }
        _ => {
            let i = num_coefs + 2 - coefs_remaining;
            let mut tmp0 = input[0];
            tmp0 -= filter[i].mem;
            tmp0 *= filter[i].coef;
            tmp0 += filter[i - 2].mem;

            let mut tmp1 = input[1];
            tmp1 -= filter[i + 1].mem;
            tmp1 *= filter[i + 1].coef;
            tmp1 += filter[i - 1].mem;

            filter[i - 2].mem = input[0];
            filter[i - 1].mem = input[1];

            input[0] = tmp0;
            input[1] = tmp1;
            process_sample_pos(coefs_remaining - 2, num_coefs, input, filter);
        }
    }
}

#[derive(Clone)]
pub struct StandardDownsampler2X {
    downsampler: Downsampler2X,
}

impl StandardDownsampler2X {
    pub fn new() -> Self {
        // coefficients from the hiir oversampling.txt list
        let num_coefs = 12;
        // let coefficients = compute_coefs_spec_order_tbw(num_coefs, 0.04);
        // dbg!(&coefficients);
        let (coefficients, _delay_in_samples) = match num_coefs {
            6 => (
                vec![
                    0.086928900551398763,
                    0.29505822040137708,
                    0.52489392936346835,
                    0.7137336652558357,
                    0.85080135560651127,
                    0.95333447720743869,
                ],
                2,
            ),
            12 => (
                vec![
                    0.017347915108876406,
                    0.067150480426919179,
                    0.14330738338179819,
                    0.23745131944299824,
                    0.34085550201503761,
                    0.44601111310335906,
                    0.54753112652956148,
                    0.6423859124721446,
                    0.72968928615804163,
                    0.81029959388029904,
                    0.88644514917318362,
                    0.96150605146543733,
                ],
                5,
            ),
            _ => {
                panic!("Incorrect order in Oversampler")
            }
        };
        let downsampler = Downsampler2X::new(coefficients);
        Self { downsampler }
    }

    /// Downsample one channel by 2x from input to output
    pub fn process_block(&mut self, input: &[Sample], output: &mut [Sample]) {
        self.downsampler.process_block(input, output)
    }
}

#[allow(dead_code)]
mod coefficient_design {
    pub fn compute_coefs_spec_order_tbw(nbr_coefs: usize, transition: f64) -> Vec<f64> {
        assert!(nbr_coefs > 0);
        assert!(transition > 0.0);
        assert!(transition < 0.5);
        let (k, q) = compute_transition_param(transition);
        dbg!(k, q);
        let mut coefs = Vec::with_capacity(nbr_coefs);
        let order = nbr_coefs * 2 + 1;
        for i in 0..nbr_coefs {
            coefs.push(compute_coef(i, k, q, order));
            println!();
        }
        coefs
    }
    fn compute_transition_param(transition: f64) -> (f64, f64) {
        assert!(transition > 0.);
        assert!(transition < 0.5);
        let mut k = ((1. - transition * 2.) * std::f64::consts::PI / 4.).tan();
        k *= k;
        assert!(k < 1.);
        assert!(k > 0.);
        let kksqrt: f64 = (1. - k * k).powf(0.25);
        let e = 0.5 * (1. - kksqrt) / (1. + kksqrt);
        let e2 = e * e;
        let e4 = e2 * e2;
        let q = e * (1. + e4 * (2. + e4 * (15. + 150. * e4)));
        assert!(q > 0.);
        (k, q)
    }

    fn compute_coef(index: usize, k: f64, q: f64, order: usize) -> f64 {
        assert!(index * 2 < order);

        let c = (index + 1) as u64;
        let num: f64 = compute_acc_num(q, order, c) * q.powf(0.25);
        let den: f64 = compute_acc_den(q, order, c) + 0.5;
        let ww = num / den;
        let wwsq = ww * ww;
        dbg!(num, den, ww, wwsq);
        dbg!((1. - wwsq * k) * (1. - wwsq / k));
        let x = ((1. - wwsq * k) * (1. - wwsq / k)).sqrt() / (1. + wwsq);
        assert!(!x.is_nan());
        let coef = (1. - x) / (1. + x);

        coef
    }
    use std::f64::consts::PI;
    fn compute_acc_num(q: f64, order: usize, c: u64) -> f64 {
        assert!(c >= 1);
        assert!((c as usize) < order * 2);

        let mut i = 0;
        let mut j = 1;
        let mut acc: f64 = 0.;
        let mut q_ii1;
        loop {
            q_ii1 = ipowp(q, i * (i + 1));
            q_ii1 *= ((((i * 2 + 1) * c) as f64 * PI) / order as f64).sin() * j as f64;
            acc += q_ii1;

            j = -j;
            i += 1;
            if !(q_ii1.abs() > 1e-100) {
                break;
            }
        }

        assert!(!acc.is_nan());
        return acc;
    }
    fn compute_acc_den(q: f64, order: usize, c: u64) -> f64 {
        assert!(c >= 1);
        assert!((c as usize) < order * 2);
        let mut i = 1;
        let mut j = -1;
        let mut acc: f64 = 0.;
        let mut q_i2;
        loop {
            q_i2 = ipowp(q, i * i);
            q_i2 *= (((i * 2 * c) as f64 * PI) / order as f64).cos() * j as f64;
            acc += q_i2;

            j = -j;
            i += 1;
            if !(q_i2.abs() > 1e-100) {
                break;
            }
        }

        assert!(!acc.is_nan());
        return acc;
    }

    fn ipowp(mut x: f64, mut n: u64) -> f64 {
        let mut z = 1.0;
        while n != 0 {
            if (n & 1) != 0 {
                z *= x;
            }
            n >>= 1;
            x *= x;
        }
        z
    }
}
