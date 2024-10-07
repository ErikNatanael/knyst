//! This module contains things related to scheduling that are more generic than
//! graph internals.

use crate::time::Beats;
use crate::time::Seconds;
use std::sync::Arc;
use std::sync::RwLock;

/// A change in musical tempo for use in a [`MusicalTimeMap`]
pub enum TempoChange {
    /// New BPM value
    #[allow(missing_docs)]
    NewTempo { bpm: f64 },
}

impl TempoChange {
    /// Give the duration in seconds of the TempoChange
    pub fn to_secs_f64(&self, duration: Beats) -> f64 {
        match self {
            TempoChange::NewTempo { bpm } => (duration.as_beats_f64() * 60.) / bpm,
        }
    }
    /// Converts a duration in seconds within this TempoChange to Beats
    pub fn secs_f64_to_beats(&self, section_duration: f64) -> Beats {
        match self {
            TempoChange::NewTempo { bpm } => Beats::from_beats_f64(*bpm * (section_duration / 60.)),
        }
    }
}
/// A map detailing tempo changes such that a [`Beats`] value can be
/// mapped to a deterministic time in seconds (wall clock time). The timestamps
/// are stored as [`Beats`] in absolute time from the start (0 beats), not
/// as relative section lengths.
///
/// Must always have a first [`TempoChange`] at the start, otherwise it wouldn't
/// be possible to map [`Beats`] to seconds. The tempo_changes must be
/// sorted in ascending order.
pub struct MusicalTimeMap {
    tempo_changes: Vec<(TempoChange, Beats)>,
}
impl MusicalTimeMap {
    /// Make a new [`MusicalTimeMap`] with a single BPM tempo value of 60 bpm at time 0
    pub fn new() -> Self {
        Default::default()
    }
    /// Insert a new [`TempoChange`]. If there is a time stamp collision the
    /// existing [`TempoChange`] will be replaced. This will also sort the list
    /// of tempo changes.
    ///
    /// # Example
    /// ```
    /// use knyst::scheduling::{MusicalTimeMap, TempoChange};
    /// use knyst::time::Beats;
    /// let mut map = MusicalTimeMap::new();
    /// map.insert(TempoChange::NewTempo { bpm: 60.0 }, Beats::new(999, 200000));
    /// map.insert(TempoChange::NewTempo { bpm: 50.0 }, Beats::new(504, 200));
    /// map.insert(TempoChange::NewTempo { bpm: 34.1 }, Beats::new(5, 200));
    /// map.insert(TempoChange::NewTempo { bpm: 642.999 }, Beats::new(5, 201));
    /// map.insert(TempoChange::NewTempo { bpm: 60.0 }, Beats::new(5, 200));
    /// map.insert(TempoChange::NewTempo { bpm: 60.0 }, Beats::new(2000, 0));
    /// map.insert(TempoChange::NewTempo { bpm: 80.0 }, Beats::new(0, 0));
    /// assert!(map.is_sorted());
    /// assert_eq!(map.len(), 6);
    /// ```
    pub fn insert(&mut self, tempo_change: TempoChange, time_stamp: Beats) {
        let mut same_timestamp_index = None;
        for (i, (_change, time)) in self.tempo_changes.iter().enumerate() {
            if *time == time_stamp {
                same_timestamp_index = Some(i);
                break;
            }
        }
        if let Some(same_ts_index) = same_timestamp_index {
            self.tempo_changes[same_ts_index] = (tempo_change, time_stamp);
        } else {
            self.tempo_changes.push((tempo_change, time_stamp));
            self.tempo_changes
                .sort_by_key(|(_tc, timestamp)| *timestamp);
        }
    }
    /// Replace a [`TempoChange`]. Will insert the default 60 bpm tempo change
    /// at the start if the first tempo change is removed.
    pub fn remove(&mut self, index: usize) {
        self.tempo_changes.remove(index);
        // Uphold the promise that there is a first tempo change that starts at zero
        if self.tempo_changes.is_empty() {
            self.tempo_changes
                .push((TempoChange::NewTempo { bpm: 60.0 }, Beats::new(0, 0)));
        }
        if self.tempo_changes[0].1 != Beats::new(0, 0) {
            self.insert(TempoChange::NewTempo { bpm: 60.0 }, Beats::new(0, 0));
        }
    }
    /// Replace a [`TempoChange`]
    pub fn replace(&mut self, index: usize, tempo_change: TempoChange) {
        if index <= self.tempo_changes.len() {
            self.tempo_changes[index].0 = tempo_change;
        }
    }
    /// Move a [`TempoChange`] to a new position in [`Beats`]. If the
    /// first tempo change is moved a 60 bpm tempo change will be inserted at
    /// the start.
    pub fn move_tempo_change(&mut self, index: usize, time_stamp: Beats) {
        if index <= self.tempo_changes.len() {
            self.tempo_changes[index].1 = time_stamp;
        }
        if index == 0 && time_stamp != Beats::new(0, 0) {
            self.insert(TempoChange::NewTempo { bpm: 60.0 }, Beats::new(0, 0));
        }
    }
    /// Convert a [`Beats`] timestamp to seconds using this map.
    ///
    /// # Example
    /// ```
    /// use knyst::scheduling::{MusicalTimeMap, TempoChange};
    /// use knyst::time::Beats;
    /// let mut map = MusicalTimeMap::new();
    /// assert_eq!(map.musical_time_to_secs_f64(Beats::new(0, 0)), 0.0);
    /// // Starts with a TempoChange for a constant 60 bpm per default
    /// assert_eq!(map.musical_time_to_secs_f64(Beats::new(1, 0)), 1.0);
    /// map.replace(0, TempoChange::NewTempo{bpm: 120.0}); // Double the speed
    /// assert_eq!(map.musical_time_to_secs_f64(Beats::new(1, 0)), 0.5);
    /// // With multiple tempo changes the accumulated time passed will be returned
    /// map.insert(
    ///     TempoChange::NewTempo { bpm: 60.0 },
    ///     Beats::from_beats(16),
    /// );
    /// map.insert(
    ///     TempoChange::NewTempo { bpm: 6000.0 },
    ///     Beats::from_beats(32),
    /// );
    ///
    /// assert_eq!(
    ///     map.musical_time_to_secs_f64(Beats::from_beats(32 + 1000)),
    ///     16.0 * 0.5 + 16.0 * 1.0 + (1000. * 0.01)
    /// );
    /// ```
    pub fn musical_time_to_secs_f64(&self, ts: Beats) -> f64 {
        // If we have not upheld our promise about the state of the MusicalTimeMap there is a bug
        assert!(self.tempo_changes.len() > 0);
        assert_eq!(self.tempo_changes[0].1, Beats::new(0, 0));
        let mut accumulated_seconds: f64 = 0.0;
        let mut duration_remaining = ts;
        // Accumulate the entire tempo changes ranges up to the one we are in
        for (tempo_change_pair0, tempo_change_pair1) in self
            .tempo_changes
            .iter()
            .zip(self.tempo_changes.iter().skip(1))
        {
            // The subtraction should always work since the tempo changes are
            // supposed to always be sorted. If they aren't, that's a bug.
            let section_duration = tempo_change_pair1
                .1
                .checked_sub(tempo_change_pair0.1)
                .unwrap();
            if duration_remaining > section_duration {
                accumulated_seconds += tempo_change_pair0.0.to_secs_f64(section_duration);
                duration_remaining = duration_remaining.checked_sub(section_duration).unwrap();
            } else {
                accumulated_seconds += tempo_change_pair0.0.to_secs_f64(duration_remaining);
                duration_remaining = Beats::new(0, 0);
                break;
            }
        }

        if duration_remaining > Beats::new(0, 0) {
            // The time stamp given is after the last tempo change, simply
            // calculate the remaining duration using the last tempo change.
            accumulated_seconds += self
                .tempo_changes
                .last()
                .unwrap()
                .0
                .to_secs_f64(duration_remaining);
        }

        accumulated_seconds
    }
    /// Convert a timestamp in seconds to beats using self
    pub fn seconds_to_beats(&self, ts: Seconds) -> Beats {
        // If we have not upheld our promise about the state of the MusicalTimeMap there is a bug
        assert!(self.tempo_changes.len() > 0);
        assert_eq!(self.tempo_changes[0].1, Beats::ZERO);
        let mut accumulated_beats = Beats::ZERO;
        let mut duration_remaining = ts.to_seconds_f64();
        // Accumulate the entire tempo changes ranges up to the one we are in
        for (tempo_change_pair0, tempo_change_pair1) in self
            .tempo_changes
            .iter()
            .zip(self.tempo_changes.iter().skip(1))
        {
            // The subtraction should always work since the tempo changes are
            // supposed to always be sorted. If they aren't, that's a bug.
            let section_duration = tempo_change_pair1
                .1
                .checked_sub(tempo_change_pair0.1)
                .unwrap();
            let section_duration_secs = tempo_change_pair0.0.to_secs_f64(section_duration);
            if duration_remaining >= section_duration_secs {
                accumulated_beats += section_duration;
                duration_remaining -= section_duration_secs;
            } else {
                accumulated_beats += tempo_change_pair0.0.secs_f64_to_beats(duration_remaining);
                duration_remaining = 0.0;
                break;
            }
        }

        if duration_remaining > 0.0 {
            // The time stamp given is after the last tempo change, simply
            // calculate the remaining duration using the last tempo change.
            accumulated_beats += self
                .tempo_changes
                .last()
                .unwrap()
                .0
                .secs_f64_to_beats(duration_remaining);
        }

        accumulated_beats
    }
    /// Returns the number of tempo changes
    pub fn len(&self) -> usize {
        self.tempo_changes.len()
    }
    /// Returns true if the tempo changes are in order, false if not. For testing purposes.
    pub fn is_sorted(&self) -> bool {
        let mut last_musical_time = Beats::new(0, 0);
        for &(_, musical_time) in &self.tempo_changes {
            if musical_time < last_musical_time {
                return false;
            }
            last_musical_time = musical_time
        }

        true
    }
}

impl Default for MusicalTimeMap {
    fn default() -> Self {
        Self {
            tempo_changes: vec![(TempoChange::NewTempo { bpm: 60.0 }, Beats::new(0, 0))],
        }
    }
}

/// Exposes the shared MusicalTimeMap in a read only Sync container.
pub struct MusicalTimeMapRef(#[allow(unused)] Arc<RwLock<MusicalTimeMap>>);

#[cfg(test)]
mod tests {
    use crate::time::Seconds;

    #[test]
    fn musical_time_test() {
        use crate::scheduling::{Beats, MusicalTimeMap, TempoChange};
        let mut map = MusicalTimeMap::new();
        assert_eq!(map.musical_time_to_secs_f64(Beats::new(0, 0)), 0.0);
        // Starts with a TempoChange for a constant 60 bpm per default
        assert_eq!(map.musical_time_to_secs_f64(Beats::new(1, 0)), 1.0);
        assert_eq!(
            map.musical_time_to_secs_f64(Beats::from_fractional_beats::<4>(5, 3)),
            5.75
        );
        assert_eq!(
            map.seconds_to_beats(Seconds::from_seconds_f64(2.)),
            Beats::from_beats(2)
        );
        map.replace(0, TempoChange::NewTempo { bpm: 120.0 });
        assert_eq!(map.musical_time_to_secs_f64(Beats::new(1, 0)), 0.5);
        assert_eq!(
            map.musical_time_to_secs_f64(Beats::from_fractional_beats::<4>(5, 3)),
            5.75 * 0.5
        );
        map.insert(TempoChange::NewTempo { bpm: 60.0 }, Beats::from_beats(16));
        map.insert(TempoChange::NewTempo { bpm: 6000.0 }, Beats::from_beats(32));
        assert_eq!(
            map.musical_time_to_secs_f64(Beats::from_beats(17)),
            16.0 * 0.5 + 1.0
        );
        assert_eq!(
            map.musical_time_to_secs_f64(Beats::from_beats(32)),
            16.0 * 0.5 + 16.0
        );
        assert_eq!(
            map.musical_time_to_secs_f64(Beats::from_beats(33)),
            16.0 * 0.5 + 16.0 + 0.01
        );
        assert_eq!(
            map.musical_time_to_secs_f64(Beats::from_beats(32 + 1000)),
            16.0 * 0.5 + 16.0 + 10.
        );
        assert_eq!(
            map.seconds_to_beats(Seconds::from_seconds_f64(2.)),
            Beats::from_beats(4)
        );
    }
}
