use std::{ops, time::Duration};

pub static SUBSAMPLE_TESIMALS_PER_SECOND: u32 = 282_240_000;

/// A description of time well suited for sample based wall clock time with
/// lossless converstion between all common sample rates.
///
/// "tesimal" is a made up word to refer to a very short amount of time.
///
/// Inspired by BillyDM's blog post https://billydm.github.io/blog/time-keeping/
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde-derive", derive(serde::Serialize, serde::Deserialize))]
pub struct Superseconds {
    seconds: u32,
    subsample_tesimals: u32,
}
impl Superseconds {
    pub fn new(seconds: u32, subsample_tesimals: u32) -> Self {
        Self {
            seconds,
            subsample_tesimals,
        }
    }
    pub fn zero() -> Self {
        Self::new(0, 0)
    }
    pub fn from_subsample_tesimals_u64(subsample_tesimals: u64) -> Self {
        let seconds = (subsample_tesimals / SUBSAMPLE_TESIMALS_PER_SECOND as u64) as u32;
        let subsample_tesimals =
            (subsample_tesimals - (seconds as u64 * SUBSAMPLE_TESIMALS_PER_SECOND as u64)) as u32;
        Self::new(seconds, subsample_tesimals)
    }
    pub fn to_subsample_tesimals_u64(&self) -> u64 {
        self.seconds as u64 * SUBSAMPLE_TESIMALS_PER_SECOND as u64 + self.subsample_tesimals as u64
    }
    pub fn from_seconds_f64(seconds_f64: f64) -> Self {
        let seconds = seconds_f64.floor() as u32;
        let subsample_tesimals =
            (seconds_f64.fract() * SUBSAMPLE_TESIMALS_PER_SECOND as f64) as u32;
        Self::new(seconds, subsample_tesimals)
    }
    pub fn from_seconds_f64_return_precision_loss(seconds_f64: f64) -> (Self, f64) {
        let ts = Self::from_seconds_f64(seconds_f64);
        (ts, seconds_f64 - ts.to_seconds_f64())
    }
    pub fn to_seconds_f64(&self) -> f64 {
        self.seconds as f64
            + (self.subsample_tesimals as f64 / SUBSAMPLE_TESIMALS_PER_SECOND as f64)
    }
    /// Convert a number of samples at a given sample rate to a `Superseconds`
    pub fn from_samples(samples: u64, sample_rate: u64) -> Self {
        let seconds = (samples / sample_rate) as u32;
        let subsample_tesimals =
            ((samples % sample_rate) * SUBSAMPLE_TESIMALS_PER_SECOND as u64 / sample_rate) as u32;
        Self {
            seconds,
            subsample_tesimals,
        }
    }
    /// Convert to samples at a specific sample rate.
    pub fn to_samples(&self, sample_rate: u64) -> u64 {
        self.seconds as u64 * sample_rate
            + ((self.subsample_tesimals as u64 * sample_rate)
                / SUBSAMPLE_TESIMALS_PER_SECOND as u64) as u64
    }

    /// Returns self - other if self is bigger than or equal to other, otherwise a SubsampleTime at 0
    pub fn checked_sub(self, rhs: Superseconds) -> Option<Self> {
        if self < rhs {
            None
        } else {
            Some(if self.subsample_tesimals >= rhs.subsample_tesimals {
                Self::new(
                    self.seconds - rhs.seconds,
                    self.subsample_tesimals - rhs.subsample_tesimals,
                )
            } else {
                Self::new(
                    self.seconds - rhs.seconds - 1,
                    SUBSAMPLE_TESIMALS_PER_SECOND
                        - (rhs.subsample_tesimals - self.subsample_tesimals),
                )
            })
        }
    }
}
impl From<Duration> for Superseconds {
    fn from(value: Duration) -> Self {
        let seconds = value.as_secs();
        let nanos = value.subsec_nanos();
        let conversion_factor = SUBSAMPLE_TESIMALS_PER_SECOND as f64 / 1_000_000_000_f64;
        let subsample_tesimals = nanos as f64 * conversion_factor;
        Self::new(seconds as u32, subsample_tesimals as u32)
    }
}

impl PartialOrd for Superseconds {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.seconds == other.seconds {
            self.subsample_tesimals
                .partial_cmp(&other.subsample_tesimals)
        } else {
            self.seconds.partial_cmp(&other.seconds)
        }
    }
}
impl Ord for Superseconds {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.seconds == other.seconds {
            self.subsample_tesimals.cmp(&other.subsample_tesimals)
        } else {
            self.seconds.cmp(&other.seconds)
        }
    }
}

impl ops::Add<Superseconds> for Superseconds {
    type Output = Self;

    fn add(self, rhs: Superseconds) -> Self::Output {
        let mut seconds = self.seconds + rhs.seconds;
        let mut subsample_tesimals = self.subsample_tesimals + rhs.subsample_tesimals;
        while subsample_tesimals > SUBSAMPLE_TESIMALS_PER_SECOND {
            seconds += 1;
            subsample_tesimals -= SUBSAMPLE_TESIMALS_PER_SECOND;
        }

        Superseconds::new(seconds, subsample_tesimals)
    }
}
impl ops::AddAssign<Superseconds> for Superseconds {
    fn add_assign(&mut self, rhs: Superseconds) {
        let result = *self + rhs;
        *self = result;
    }
}
impl ops::Mul<Superseconds> for Superseconds {
    type Output = Self;

    fn mul(self, rhs: Superseconds) -> Self::Output {
        let mut seconds = self.seconds * rhs.seconds;
        let mut subsample_tesimals = self.subsample_tesimals as u64 * rhs.subsample_tesimals as u64;
        if subsample_tesimals > SUBSAMPLE_TESIMALS_PER_SECOND as u64 {
            seconds += (subsample_tesimals / SUBSAMPLE_TESIMALS_PER_SECOND as u64) as u32;
            subsample_tesimals %= SUBSAMPLE_TESIMALS_PER_SECOND as u64;
        }
        Superseconds::new(seconds, subsample_tesimals as u32)
    }
}
impl ops::MulAssign<Superseconds> for Superseconds {
    fn mul_assign(&mut self, rhs: Superseconds) {
        *self = *self * rhs;
    }
}

mod tests {
    use std::time::Duration;

    use super::{Superseconds, SUBSAMPLE_TESIMALS_PER_SECOND};

    #[test]
    fn convert_to_u64_and_back() {
        let original_ts = Superseconds::new(8347, SUBSAMPLE_TESIMALS_PER_SECOND - 5);
        let as_u64 = original_ts.to_subsample_tesimals_u64();
        assert_eq!(
            original_ts,
            Superseconds::from_subsample_tesimals_u64(as_u64)
        );
    }
    #[test]
    fn duration_to_subsample_time() {
        for seconds in [73.73, 10.832, 10000.25, 84923.399] {
            let superseconds = Superseconds::from_seconds_f64(seconds);
            let duration = Duration::from_secs_f64(seconds);
            assert_eq!(superseconds, duration.into())
        }
    }
    #[test]
    fn sample_conversion() {
        assert_eq!(Superseconds::from_samples(1, 44100).to_samples(88200), 2);
        assert_eq!(Superseconds::from_samples(1, 44100).to_samples(44100), 1);
        assert_eq!(Superseconds::from_samples(2, 44100).to_samples(44100), 2);
        assert_eq!(Superseconds::from_samples(3, 44100).to_samples(44100), 3);
        assert_eq!(Superseconds::from_samples(4, 44100).to_samples(44100), 4);
        assert_eq!(
            Superseconds::from_samples(44100, 44100).to_samples(88200),
            88200
        );
        assert_eq!(
            Superseconds::from_samples(44100 * 3 + 1, 44100).to_samples(88200),
            3 * 88200 + 2
        );
        assert_eq!(
            Superseconds::from_samples(96000 * 3 + 8, 96000).to_samples(88200),
            3 * 88200 + 7
        );
    }
}
