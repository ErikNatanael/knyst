//! A pretty barebones Envelope Gen
//!
//! [`EnvelopeGen`] can be constructed directly, but it is more convenient to
//! create an [`Envelope`] and then call [`Envelope::to_gen`] on it.
//!
//! ```rust
//! use knyst::envelope::*;
//! use knyst::prelude::*;
//! let amplitude = 0.5;
//! let attack_time = 0.2;
//! let release_time = 2.0;
//! let env = Envelope {
//!     start_value: 0.0,
//!     points: vec![(amplitude, attack_time), (0.0, release_time)],
//!     curves: vec![Curve::Linear, Curve::Exponential(2.0)],
//!     stop_action: StopAction::FreeGraph,
//!     ..Default::default()
//! };
//! let mut env = env.to_gen();
//! ```

// level, duration
type Point = (Sample, Sample);
// Storing time as samples in an f64 is fine, there's adequate range and avoids type casting.

// Benefits compared to Spline implementation:
// - can store time as relative instead of absolute
// - can manipulate the data structure in place
// - can introduce new interpolation methods
// - because of relative time, complex behaviour of jumping around inside the envelope can be implemented (e.g. looping envelope or random/markov chain envelop movement)

use crate::trig::is_trigger;
use crate::{self as knyst, Trig};
use crate::{
    controller::KnystCommands,
    gen::{GenState, StopAction},
    Sample, SampleRate,
};
use knyst_macro::impl_gen;

// TODO:
// [ ] Different curve types:
// - [ ] Bezier
// - [ ] Sine
// [ ] Custom envelope type constructors
// [ ] Different envelope speeds (useful for non-audio thread stuff)
// [ ] Fast forward to specific time (useful for things that aren't sample by sample)
// [ ] Test sustaining and looping variants in practice
// [ ] StopAction for when it is finished playing
//
//
/// Convenience struct to provide a more ergonomic interface to creating an [`EnvelopeGen`]
pub struct Envelope {
    /// The start value of the envelope. Default: 0.0
    pub start_value: Sample,
    /// Points after the start value in the format (next_value, time_to_reach_it_in_seconds)
    pub points: Vec<Point>,
    /// Curves per point
    pub curves: Vec<Curve>,
    /// Sustain mode of the envelope
    pub sustain: SustainMode,
    /// If the envelope is not looping, what StopAction to emit when finished
    pub stop_action: StopAction,
}
impl Envelope {
    /// Converts an [`Envelope`] to an [`EnvelopeGen`] and starts it.
    pub fn to_gen(&self) -> EnvelopeGen {
        let mut e = EnvelopeGen::new(
            self.start_value,
            self.points.clone(),
            self.sustain,
            self.stop_action,
        )
        .curves(self.curves.clone());
        e.start();
        e
    }
}
impl Default for Envelope {
    fn default() -> Self {
        Self {
            start_value: 0.0,
            points: vec![(1.0, 0.5), (0., 0.5)],
            curves: vec![Curve::Linear],
            sustain: SustainMode::NoSustain,
            stop_action: StopAction::Continue,
        }
    }
}
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy)]
pub enum SustainMode {
    NoSustain,
    SustainAtPoint(usize),
    Loop { start: usize, end: usize },
}

/// The curve type/slope of an envelope segment
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy)]
pub enum Curve {
    Linear,
    Exponential(Sample),
}

impl Curve {
    /// Convert from linear 0<=a<1.0 to this curve
    #[inline]
    pub fn transform(&self, a: Sample) -> Sample {
        match self {
            Curve::Linear => a,
            // Using the fastapprox::faster variant is significantly faster, but too inaccurate
            Curve::Exponential(exponent) => {
                fastapprox::fast::pow(a as f32, *exponent as f32) as Sample
            }
        }
    }
}

/// An envelope Gen. Must be started by calling [`EnvelopeGen::start`] since that
/// will initialise it based on the settings of its segments. If you want to use
/// it offline, it can be turned into an iterator by calling [`EnvelopeGen::iter_mut`].
#[derive(Debug, Clone)]
pub struct EnvelopeGen {
    #[allow(missing_docs)]
    pub start_value: Sample,

    // Points with their time value as seconds. This enables setting the sample rate when the Gen is initiated.
    points_secs: Vec<Point>,
    points: Vec<Point>,
    curves: Vec<Curve>,
    source_value: Sample,
    target_value: Sample,
    /// The difference between source_value and target_value
    source_target_diff: Sample,
    fade_out_duration: Sample,
    segment_duration: Sample,
    current_curve: Curve,
    current_timestep: f64,
    /// Goes from 0 to 1 over a segment
    duration_passed: f64,
    next_index: usize,
    #[allow(missing_docs)]
    pub playing: bool,
    sample_rate: Sample,
    sustain: SustainMode,
    stop_action: StopAction,
    // pub sustaining: bool, // if the envelope should stop at a certain point before release
    // pub sustaining_point: usize, // which point the envelope should sustain at, if any
    // pub release_point: Option<usize>,
    // pub looping: bool,
    waiting_for_release: bool,
}

impl EnvelopeGen {
    /// Convenience method for an ADSR envelope
    pub fn adsr(
        attack_time: Sample,
        decay_time: Sample,
        sustain_level: Sample,
        release_time: Sample,
    ) -> Self {
        let points = vec![
            (1.0, attack_time),
            (sustain_level, decay_time),
            (0.0, release_time),
        ];
        Self::new(
            0.0,
            points,
            SustainMode::SustainAtPoint(1),
            StopAction::Continue,
        )
    }
    /// Set the [`SustainMode`]
    pub fn sustain(mut self, sustain: SustainMode) -> Self {
        self.sustain = sustain;
        self
    }
    /// Set the [`StopAction`]
    pub fn stop_action(mut self, stop_action: StopAction) -> Self {
        self.stop_action = stop_action;
        self
    }

    /// Set the points of the envelope. Points are in the format (next_leve, time_to_reach_that_level_in_seconds)
    pub fn set_points(&mut self, points: Vec<Point>) {
        self.points_secs = points.clone();
        // Convert to points where the time is in samples
        let points = points
            .into_iter()
            .map(|mut p| {
                p.1 *= self.sample_rate;
                p
            })
            .collect();
        self.points = points;
        if self.curves.len() != self.points.len() {
            self.curves.resize(self.points.len(), Curve::Linear);
        }
    }
    /// If the curves are differently many from the points, fill with Curve::Linear for the later segments.
    pub fn curves(mut self, curves: Vec<Curve>) -> Self {
        if curves.len() == self.points.len() {
            self.curves = curves;
        } else {
            let default_curve = Curve::Linear;
            self.curves = curves
                .into_iter()
                .chain(std::iter::repeat(default_curve))
                .take(self.points.len())
                .collect();
        }
        self
    }
    /// Set a specific segment curve
    pub fn set_curve(&mut self, curve: Curve, index: usize) {
        self.curves[index] = curve;
    }
    #[allow(missing_docs)]
    pub fn get_point(&mut self, index: usize) -> Point {
        self.points[index]
    }
    /// Returns if the envelope is playing
    pub fn playing(&self) -> bool {
        self.playing
    }
    /// Updates the values of the this Envelope to match those of the other
    /// Envelope if it can be done without allocation. The Envelopes are assumed
    /// to match.
    pub fn update_from_envelope(&mut self, other: &EnvelopeGen) {
        self.start_value = other.start_value;
        for (i, p) in other.points.iter().enumerate() {
            self.points[i] = *p;
            if i == self.next_index - 1 {
                if self.playing {
                    self.target_value = self.points[i].0;
                    self.source_target_diff = self.target_value - self.source_value;
                    self.segment_duration = self.points[i].1;
                } else {
                    self.source_value = self.points[i].0;
                }
            }
        }
    }
    /// Initialises the envelope based on the start_value and the first point
    pub fn start(&mut self) {
        self.playing = true;
        self.waiting_for_release = false;
        self.source_value = self.start_value;
        self.target_value = self.points[0].0;
        self.source_target_diff = self.target_value - self.source_value;
        self.segment_duration = self.points[0].1;
        self.current_curve = self.curves[0];
        self.current_timestep = (self.segment_duration as f64).recip();
        self.duration_passed = 0.;
        self.next_index = 1;
    }
    /// Initialises the envelope based on the current value and the first point
    pub fn restart_from_current(&mut self) {
        // If we're playing, grab the current value. If not the end value is
        // already stored in self.source_value
        if self.playing {
            self.source_value = self.current_value();
        }
        self.playing = true;
        self.waiting_for_release = false;
        self.target_value = self.points[0].0;
        self.source_target_diff = self.target_value - self.source_value;
        self.segment_duration = self.points[0].1;
        self.current_curve = self.curves[0];
        self.current_timestep = (self.segment_duration as f64).recip();
        self.duration_passed = 0.;
        self.next_index = 1;
    }
    /// Releases the envelope if it is sustaining or immediately fades out if it is not.
    pub fn release(&mut self) {
        match self.sustain {
            SustainMode::NoSustain => self.fade_out(),
            SustainMode::SustainAtPoint(sustain_point) => {
                if self.waiting_for_release {
                    self.next_segment();
                    self.waiting_for_release = false;
                }
                // We're jumping from somewhere to the end segment
                self.jump_to_segment(sustain_point + 1);
            }
            SustainMode::Loop { start: _, end } => {
                self.jump_to_segment(end + 1);
            }
        }
    }
    /// Immediately fade the output to 0.0. This is called if
    /// [`EnvelopeGen::release`] is called on a non sustaining and non looping
    /// envelope.
    pub fn fade_out(&mut self) {
        self.source_value = self.current_value();
        self.next_index = self.points.len();
        self.target_value = 0.0;
        self.source_target_diff = self.target_value - self.source_value;
        self.duration_passed = 0.0;
        self.segment_duration = self.fade_out_duration * self.sample_rate;
        self.current_timestep = (self.segment_duration as f64).recip();
        self.current_curve = Curve::Linear;
    }
    /// Change the value of a specific point.
    pub fn set_value(&mut self, value: Sample, index: usize) {
        self.points[index].0 = value;
        // Also update the value if it's currently playing
        if index == self.next_index - 1 {
            if self.playing && !self.waiting_for_release {
                self.target_value = value;
                self.source_target_diff = self.target_value - self.source_value;
            } else {
                self.source_value = value;
                self.target_value = value;
                self.source_target_diff = self.target_value - self.source_value;
            }
        }
    }
    /// Set the duration of a segment in seconds (the duration to reach the value with the same index). Will not affect the currently playing segment.
    pub fn set_duration(&mut self, duration: Sample, index: usize) {
        // Convert seconds to samples
        self.points[index].1 = duration * self.sample_rate;
    }
    fn next_segment(&mut self) {
        // Skip over 0 and negative duration points by looping until the
        // duration is valid
        loop {
            self.source_value = self.target_value;

            if !match self.sustain {
                SustainMode::NoSustain => false,
                SustainMode::SustainAtPoint(sustain_index) => {
                    if self.next_index == sustain_index + 1 {
                        self.waiting_for_release = true;
                        self.source_target_diff = self.target_value - self.source_value;
                        true
                    } else {
                        false
                    }
                }
                SustainMode::Loop { start, end } => {
                    if self.next_index == end + 1 {
                        self.jump_to_segment(start);
                        true
                    } else {
                        false
                    }
                }
            } {
                if self.next_index < self.points.len() {
                    self.target_value = self.points[self.next_index].0;
                    self.source_target_diff = self.target_value - self.source_value;
                    self.segment_duration = self.points[self.next_index].1;
                    self.current_timestep = (self.segment_duration as f64).recip();
                    self.current_curve = self.curves[self.next_index];
                    self.duration_passed = 0.0;
                    self.next_index += 1;
                    // Don't break here since the segment duration may be 0 in which case we should immediately go to the next segment
                } else {
                    self.playing = false;
                    break;
                }
            }
            if self.segment_duration > 0.0 || !self.playing {
                break;
            }
        }
    }
    fn jump_to_segment(&mut self, destination_index: usize) {
        self.source_value = self.current_value();
        if destination_index < self.points.len() {
            self.target_value = self.points[destination_index].0;
            self.segment_duration = self.points[destination_index].1;
            self.current_timestep = (self.segment_duration as f64).recip();
            self.current_curve = self.curves[destination_index];
            self.duration_passed = 0.;
            self.next_index = destination_index + 1;
        } else {
            self.playing = false;
        }
        self.source_target_diff = self.target_value - self.source_value;
    }
    #[inline(always)]
    fn current_value(&mut self) -> Sample {
        // note: t goes from 1 to just above 0 over the duration of a segment
        let t = self.current_curve.transform(self.duration_passed as Sample);
        // linear interpolation
        self.source_value + (t * self.source_target_diff)
    }
    /// Get the next sample of the envelope
    #[inline(always)]
    pub fn next_sample(&mut self) -> Sample {
        if self.playing && !self.waiting_for_release {
            let value = self.current_value();
            self.duration_passed += self.current_timestep;
            if self.duration_passed >= 1.0 {
                // Since self.next_index points at the next index and we want to know if we're at self.sustaining_point+1
                self.next_segment();
            }
            value
        } else {
            self.source_value // if we're not playing the envelope, the final value is saved here. If pausing is implemented, a current_value field may be needed
        }
    }
    /// Evaluate [`Self`] as an iterator. Evaluating the envelope means changing
    /// the state which is why it is mut.
    pub fn iter_mut(&mut self) -> EnvelopeIterator {
        EnvelopeIterator { envelope: self }
    }
}

#[impl_gen]
impl EnvelopeGen {
    /// Create a new Envelope. points are in the format (level, duration) where the duration is given in seconds, and later converted to samples internally.
    pub fn new(
        start_value: Sample,
        points: Vec<Point>,
        sustain: SustainMode,
        stop_action: StopAction,
    ) -> Self {
        let mut points = points;
        let sample_rate = 41000.;
        let points_secs = points.clone();
        // Convert durations from seconds to samples
        for point in &mut points {
            point.1 *= sample_rate;
        }
        let target_value = points[0].0;
        let segment_duration = points[0].1;
        let curves = vec![Curve::Linear; points.len()];
        let mut s = Self {
            points_secs,
            points,
            curves,
            start_value,
            source_value: start_value,
            target_value,
            source_target_diff: target_value - start_value,
            current_curve: Curve::Linear,
            current_timestep: 0.0,
            fade_out_duration: 0.5,
            segment_duration,
            duration_passed: 0.,
            stop_action,
            next_index: 1,
            playing: true,
            sample_rate,
            sustain,
            // sustaining: false,
            // sustaining_point: 0,
            // release_point: None,
            // looping: false,
            waiting_for_release: false,
        };
        s.start();
        s
    }
    // TODO: Add more input options for runtime changes e.g. the values and durations of points
    /// Process block
    pub fn process(
        &mut self,
        release: &[Trig],
        restart: &[Trig],
        amplitude: &mut [Sample],
    ) -> GenState {
        let release_trigger_in = release;
        let restart_trigger_in = restart;
        let mut stop_sample = None;
        for (((i, out), &release_trig), &restart_trig) in amplitude
            .iter_mut()
            .enumerate()
            .zip(release_trigger_in.iter())
            .zip(restart_trigger_in.iter())
        {
            if is_trigger(release_trig) {
                self.release();
            }
            if is_trigger(restart_trig) {
                self.restart_from_current();
            }
            *out = self.next_sample();
            if !self.playing() && stop_sample.is_none() {
                stop_sample = Some(i)
            }
        }
        if self.playing {
            GenState::Continue
        } else {
            self.stop_action.to_gen_state(stop_sample.unwrap())
        }
    }
    fn init(&mut self, sample_rate: SampleRate) {
        if self.sample_rate != *sample_rate {
            self.points = self
                .points_secs
                .iter()
                .copied()
                .map(|(x, y)| (x, y * *sample_rate))
                .collect();
        }
        self.sample_rate = *sample_rate;
    }
}

/// Evaluate an [`EnvelopeGen`] as an iterator.
pub struct EnvelopeIterator<'a> {
    envelope: &'a mut EnvelopeGen,
}

impl<'a> Iterator for EnvelopeIterator<'a> {
    type Item = Sample;

    fn next(&mut self) -> Option<Self::Item> {
        if self.envelope.playing() {
            Some(self.envelope.next_sample())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_simple_envelope() {
        let sample_rate = 44100.;
        let mut env = EnvelopeGen::new(
            0.0,
            vec![(1.0, 1.0), (0.75, 0.5), (0.1, 3.0)],
            SustainMode::NoSustain,
            StopAction::Continue,
        );
        env.init(SampleRate(sample_rate));
        env.start();
        assert_eq!(env.next_sample(), 0.);
        assert!(env.next_sample() > 0.);
        // fast forward 0.5 seconds minus the samples we've already extracted
        for _i in 0..(sample_rate * 0.5 - 2.0) as i32 {
            env.next_sample();
        }
        assert_eq!(
            env.next_sample(),
            0.5,
            "Envelope value was expected to be 0.5 halfway between 0.0 and 1.0. {:?}",
            env
        );
        // fast forward another 0.5 seconds minus the samples we've already extracted
        for _i in 0..(sample_rate * 0.5 - 1.0) as i32 {
            env.next_sample();
        }
        assert_eq!(env.next_sample(), 1.0);
        // fast forward to the point where we arrive at the second level
        for _i in 0..(sample_rate * 0.5 - 1.0) as i32 {
            env.next_sample();
        }
        assert_eq!(
            env.next_sample(),
            0.75,
            "Envelope value was expected to be 0.75 right at the second point. {:?}",
            env
        );
        // fast forward past the end
        for _i in 0..(sample_rate * 3.01) as i32 {
            env.next_sample();
        }
        assert_eq!(
            env.next_sample(),
            0.1,
            "Envelope did not keep its last value after finishing the envelope {:?}",
            env
        );
        assert!(
            !env.playing,
            "Envelope is not supposed to be playing after it is done {:?}",
            env
        );
    }
}
