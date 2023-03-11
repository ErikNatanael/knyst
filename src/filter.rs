/*
 *
 * void ButterworthLowpassFilter0100SixthOrder(const double src[], double dest[], int size)
{
    const int NZEROS = 6;
    const int NPOLES = 6;
    const double GAIN = 2.936532839e+03;

    double xv[NZEROS+1] = {0.0}, yv[NPOLES+1] = {0.0};

    for (int i = 0; i < size; i++)
      {
        xv[0] = xv[1]; xv[1] = xv[2]; xv[2] = xv[3]; xv[3] = xv[4]; xv[4] = xv[5]; xv[5] = xv[6];
        xv[6] = src[i] / GAIN;
        yv[0] = yv[1]; yv[1] = yv[2]; yv[2] = yv[3]; yv[3] = yv[4]; yv[4] = yv[5]; yv[5] = yv[6];
        yv[6] =   (xv[0] + xv[6]) + 6.0 * (xv[1] + xv[5]) + 15.0 * (xv[2] + xv[4])
                     + 20.0 * xv[3]
                     + ( -0.0837564796 * yv[0]) + (  0.7052741145 * yv[1])
                     + ( -2.5294949058 * yv[2]) + (  4.9654152288 * yv[3])
                     + ( -5.6586671659 * yv[4]) + (  3.5794347983 * yv[5]);
        dest[i] = yv[6];
    }
}
 */

// Bra resurs? https://www.musicdsp.org/en/latest/Filters/276-biquad-butterworth-chebyshev-n-order-m-channel-optimized-filters.html
use crate::Sample;

///
/// Uses f64 internally for filter stability.
struct ButterworthLowpass<const ORDER: usize> {
    x: Vec<f64>,
    y: Vec<f64>,
    coefficients: Vec<f64>,
}
impl<const ORDER: usize> ButterworthLowpass<ORDER> {
    const MAKE_UP_GAIN: f64 = 2.936532839e+03;
    pub fn new(cutoff_freq: Sample, sample_freq: Sample) -> Self {
        let mut coefficients = vec![0.0; ORDER];
        let gamma = std::f64::consts::PI as / (ORDER as f64 * 2.);
        // First coefficient is always 1
        coefficients[0] = 1.0;
        for i in 0..ORDER {
            let last = if i == 0 { 1.0 } else { coefficients[i - 1] };
            coefficients[i + 1] =
                ((i as Sample * gamma).cos() / ((i + 1) as Sample * gamma).sin()) * last;
        }
        // Adjust for
        Self {
            x: vec![0.0 as Sample; ORDER + 1],
            y: vec![0.0 as Sample; ORDER + 1],
            coefficients,
        }
    }
    pub fn tick(&mut self, input: Sample) -> Sample {
        self.x.as_mut_slice().rotate_left(1);
        self.x[6] = input / ButterworthLowpass::<ORDER>::MAKE_UP_GAIN;
        self.y.as_mut_slice().rotate_left(1);
        let x = &mut self.x;
        let y = &mut self.y;
        y[6] = x[0]
            + x[6]
            + 6.0 * (x[1] + x[5])
            + 15. * (x[2] + x[4])
            + 20. * x[3]
            + (-0.0837564796 * y[0])
            + (0.7052741145 * y[1])
            + (-2.5294949058 * y[2])
            + (4.9654152288 * y[3])
            + (-5.6586671659 * y[4])
            + (3.5794347983 * y[5]);
        y[6]
    }
}
