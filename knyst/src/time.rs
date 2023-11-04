//! Structs for dealing with time with determinism and high accuracy in seconds and beats.
//!
//! Heavily inspired by BillyDM's blog post: https://billydm.github.io/blog/time-keeping/

use std::{ops, time::Duration};

/// How many subsample tesimals fit in one second
pub static SUBSAMPLE_TESIMALS_PER_SECOND: u32 = 282_240_000;
/// How many beat tesimals fit in one beat
pub static SUBBEAT_TESIMALS_PER_BEAT: u32 = 1_476_034_560;

/// A description of time well suited for sample based wall clock time with
/// lossless converstion between all common sample rates. Can only represent a
/// positive time value.
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
    /// 0 seconds 0 tesmials
    pub const ZERO: Self = Self {
        seconds: 0,
        subsample_tesimals: 0,
    };
    #[allow(missing_docs)]
    pub fn new(seconds: u32, subsample_tesimals: u32) -> Self {
        Self {
            seconds,
            subsample_tesimals,
        }
    }
    /// Convert from subsample tesimals to a Supersecond.
    pub fn from_subsample_tesimals_u64(subsample_tesimals: u64) -> Self {
        let seconds = (subsample_tesimals / SUBSAMPLE_TESIMALS_PER_SECOND as u64) as u32;
        let subsample_tesimals =
            (subsample_tesimals - (seconds as u64 * SUBSAMPLE_TESIMALS_PER_SECOND as u64)) as u32;
        Self::new(seconds, subsample_tesimals)
    }
    /// Convert seconds and tesimals to tesimals
    pub fn to_subsample_tesimals_u64(&self) -> u64 {
        self.seconds as u64 * SUBSAMPLE_TESIMALS_PER_SECOND as u64 + self.subsample_tesimals as u64
    }
    #[allow(missing_docs)]
    pub fn from_seconds_f64(seconds_f64: f64) -> Self {
        let seconds = seconds_f64.floor() as u32;
        let subsample_tesimals =
            (seconds_f64.fract() * SUBSAMPLE_TESIMALS_PER_SECOND as f64) as u32;
        Self::new(seconds, subsample_tesimals)
    }
    /// Convert from seconds in f64 and return any precision loss incurred in the conversion
    pub fn from_seconds_f64_return_precision_loss(seconds_f64: f64) -> (Self, f64) {
        let ts = Self::from_seconds_f64(seconds_f64);
        (ts, seconds_f64 - ts.to_seconds_f64())
    }
    /// Convert to seconds in an f64. May be lossy depending on the value.
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
    #[must_use]
    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
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
        Some(self.cmp(other))
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
        while subsample_tesimals >= SUBSAMPLE_TESIMALS_PER_SECOND {
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

/// A description of time well suited for musical beat time with
/// lossless converstion between all common subdivisions of beats.
///
/// "tesimal" is a made up word to refer to a very short amount of time.
///
/// Inspired by BillyDM's blog post https://billydm.github.io/blog/time-keeping/
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde-derive", derive(serde::Serialize, serde::Deserialize))]
pub struct Superbeats {
    beats: u32,
    beat_tesimals: u32,
}
impl Superbeats {
    /// Zero beats
    pub const ZERO: Self = Self {
        beats: 0,
        beat_tesimals: 0,
    };
    #[allow(missing_docs)]
    pub fn new(beats: u32, beat_tesimals: u32) -> Self {
        Self {
            beat_tesimals,
            beats,
        }
    }
    /// Returns Superbeats with `beats` beats and 0 tesimals
    pub fn from_beats(beats: u32) -> Self {
        Self::new(beats, 0)
    }
    /// Construct from a number of fractional beats. This will be exact for any
    /// fraction that [`SUBBEAT_TESIMALS_PER_BEAT`] is evenly divisible by
    /// including 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 24, 32,
    /// 64, 128, 256, 512, and 1920.
    ///
    /// Fractional_beats must be in the range [0, FRACTION-1]
    pub fn from_fractional_beats<const FRACTION: u32>(beats: u32, fractional_beats: u32) -> Self {
        assert!(fractional_beats < FRACTION);
        let tesimals = (SUBBEAT_TESIMALS_PER_BEAT / FRACTION) * fractional_beats;
        Self::new(beats, tesimals)
    }
    /// Convert from number of beats in an f32. May be lossy depending on the value
    pub fn from_beats_f32(beats: f32) -> Self {
        let fract = beats.fract();
        Self {
            beats: beats as u32,
            beat_tesimals: (fract * SUBBEAT_TESIMALS_PER_BEAT as f32) as u32,
        }
    }
    /// Convert from number of beats in an f64. May be lossy depending on the value
    pub fn from_beats_f64(beats: f64) -> Self {
        let fract = beats.fract();
        Self {
            beats: beats as u32,
            beat_tesimals: (fract * SUBBEAT_TESIMALS_PER_BEAT as f64) as u32,
        }
    }
    /// Convert to number of beats in an f32. May be lossy depending on the value
    pub fn as_beats_f32(&self) -> f32 {
        self.beats as f32 + (self.beat_tesimals as f32 / SUBBEAT_TESIMALS_PER_BEAT as f32)
    }
    /// Convert to number of beats in an f64. May be lossy depending on the value
    pub fn as_beats_f64(&self) -> f64 {
        self.beats as f64 + (self.beat_tesimals as f64 / SUBBEAT_TESIMALS_PER_BEAT as f64)
    }
    /// Returns `Some(self-v)` if v >= self, otherwise `None`
    pub fn checked_sub(&self, v: Self) -> Option<Self> {
        if self < &v {
            None
        } else {
            if self.beat_tesimals < v.beat_tesimals {
                Some(Self {
                    beats: self.beats - v.beats - 1,
                    beat_tesimals: SUBBEAT_TESIMALS_PER_BEAT
                        - (v.beat_tesimals - self.beat_tesimals),
                })
            } else {
                Some(Self {
                    beats: self.beats - v.beats,
                    beat_tesimals: self.beat_tesimals - v.beat_tesimals,
                })
            }
        }
    }
}
impl std::iter::Sum for Superbeats {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Superbeats::ZERO, |acc, elem| acc + elem)
    }
}
impl PartialOrd for Superbeats {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Superbeats {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.beats == other.beats {
            self.beat_tesimals.cmp(&other.beat_tesimals)
        } else {
            self.beats.cmp(&other.beats)
        }
    }
}
impl ops::Add<Superbeats> for Superbeats {
    type Output = Self;

    fn add(self, rhs: Superbeats) -> Self::Output {
        let mut beats = self.beats + rhs.beats;
        let mut beat_tesimals = self.beat_tesimals + rhs.beat_tesimals;
        while beat_tesimals >= SUBBEAT_TESIMALS_PER_BEAT {
            beats += 1;
            beat_tesimals -= SUBBEAT_TESIMALS_PER_BEAT;
        }

        Superbeats::new(beats, beat_tesimals)
    }
}
impl ops::AddAssign<Superbeats> for Superbeats {
    fn add_assign(&mut self, rhs: Superbeats) {
        let result = *self + rhs;
        *self = result;
    }
}
impl ops::Mul<Superbeats> for Superbeats {
    type Output = Self;

    fn mul(self, rhs: Superbeats) -> Self::Output {
        let mut beats = self.beats * rhs.beats
            + ((self.beats as u64 * rhs.beat_tesimals as u64) / SUBBEAT_TESIMALS_PER_BEAT as u64)
                as u32;
        let mut beat_tesimals = (self.beat_tesimals as u64 * rhs.beat_tesimals as u64)
            / SUBBEAT_TESIMALS_PER_BEAT as u64
            + self.beat_tesimals as u64 * rhs.beats as u64;
        if beat_tesimals > SUBBEAT_TESIMALS_PER_BEAT as u64 {
            beats += (beat_tesimals / SUBBEAT_TESIMALS_PER_BEAT as u64) as u32;
            beat_tesimals %= SUBBEAT_TESIMALS_PER_BEAT as u64;
        }
        Superbeats::new(beats, beat_tesimals as u32)
    }
}
impl ops::MulAssign<Superbeats> for Superbeats {
    fn mul_assign(&mut self, rhs: Superbeats) {
        *self = *self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::{Superseconds, SUBSAMPLE_TESIMALS_PER_SECOND};
    use std::time::Duration;

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
    #[test]
    fn arithmetic() {
        assert_eq!(
            Superseconds::new(0, SUBSAMPLE_TESIMALS_PER_SECOND - 1) + Superseconds::new(1, 1),
            Superseconds::new(2, 0)
        );
    }
}
