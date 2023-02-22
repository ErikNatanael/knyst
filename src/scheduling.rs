use std::sync::Arc;
use std::sync::RwLock;

pub use meadowlark_core_types::time::MusicalTime;

pub enum TempoChange {
    NewTempo { bpm: f64 },
}

impl TempoChange {
    /// Give the duration in seconds of the TempoChange
    pub fn to_secs_f64(&self, duration: MusicalTime) -> f64 {
        match self {
            TempoChange::NewTempo { bpm } => (duration.as_beats_f64() * 60.) / bpm,
        }
    }
}
/// A map detailing tempo changes such that a [`MusicalTime`] value can be
/// mapped to a deterministic time in seconds (wall clock time). The timestamps
/// are stored as [`MusicalTime`] in absolute time from the start (0 beats), not
/// as relative section lengths.
///
/// Must always have a first [`TempoChange`] at the start, otherwise it wouldn't
/// be possible to map [`MusicalTime`] to seconds. The tempo_changes must be
/// sorted in ascending order.
pub struct MusicalTimeMap {
    tempo_changes: Vec<(TempoChange, MusicalTime)>,
}
impl MusicalTimeMap {
    pub fn new() -> Self {
        Default::default()
    }
    /// Insert a new [`TempoChange`]. If there is a time stamp collision the
    /// existing [`TempoChange`] will be replaced. This will also sort the list
    /// of tempo changes.
    ///
    /// # Example
    /// ```
    /// use knyst::scheduling::{MusicalTimeMap, MusicalTime, TempoChange};
    /// let mut map = MusicalTimeMap::new();
    /// map.insert(TempoChange::NewTempo { bpm: 60.0 }, MusicalTime::new(999, 200000));
    /// map.insert(TempoChange::NewTempo { bpm: 50.0 }, MusicalTime::new(504, 200));
    /// map.insert(TempoChange::NewTempo { bpm: 34.1 }, MusicalTime::new(5, 200));
    /// map.insert(TempoChange::NewTempo { bpm: 642.999 }, MusicalTime::new(5, 201));
    /// map.insert(TempoChange::NewTempo { bpm: 60.0 }, MusicalTime::new(5, 200));
    /// map.insert(TempoChange::NewTempo { bpm: 60.0 }, MusicalTime::new(2000, 0));
    /// map.insert(TempoChange::NewTempo { bpm: 80.0 }, MusicalTime::new(0, 0));
    /// assert!(map.is_sorted());
    /// assert_eq!(map.len(), 6);
    /// ```
    pub fn insert(&mut self, tempo_change: TempoChange, time_stamp: MusicalTime) {
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
                .push((TempoChange::NewTempo { bpm: 60.0 }, MusicalTime::new(0, 0)));
        }
        if self.tempo_changes[0].1 != MusicalTime::new(0, 0) {
            self.insert(TempoChange::NewTempo { bpm: 60.0 }, MusicalTime::new(0, 0));
        }
    }
    /// Replace a [`TempoChange`]
    pub fn replace(&mut self, index: usize, tempo_change: TempoChange) {
        if index <= self.tempo_changes.len() {
            self.tempo_changes[index].0 = tempo_change;
        }
    }
    /// Move a [`TempoChange`] to a new position in [`MusicalTime`]. If the
    /// first tempo change is moved a 60 bpm tempo change will be inserted at
    /// the start.
    pub fn move_tempo_change(&mut self, index: usize, time_stamp: MusicalTime) {
        if index <= self.tempo_changes.len() {
            self.tempo_changes[index].1 = time_stamp;
        }
        if index == 0 && time_stamp != MusicalTime::new(0, 0) {
            self.insert(TempoChange::NewTempo { bpm: 60.0 }, MusicalTime::new(0, 0));
        }
    }
    /// Convert a [`MusicalTime`] timestamp to seconds using this map.
    ///
    /// # Example
    /// ```
    /// use knyst::scheduling::{MusicalTimeMap, MusicalTime, TempoChange};
    /// let mut map = MusicalTimeMap::new();
    /// assert_eq!(map.musical_time_to_secs_f64(MusicalTime::new(0, 0)), 0.0);
    /// // Starts with a TempoChange for a constant 60 bpm per default
    /// assert_eq!(map.musical_time_to_secs_f64(MusicalTime::new(1, 0)), 1.0);
    /// map.replace(0, TempoChange::NewTempo{bpm: 120.0}); // Double the speed
    /// assert_eq!(map.musical_time_to_secs_f64(MusicalTime::new(1, 0)), 0.5);
    /// // With multiple tempo changes the accumulated time passed will be returned
    /// map.insert(
    ///     TempoChange::NewTempo { bpm: 60.0 },
    ///     MusicalTime::from_beats(16),
    /// );
    /// map.insert(
    ///     TempoChange::NewTempo { bpm: 6000.0 },
    ///     MusicalTime::from_beats(32),
    /// );
    ///
    /// assert_eq!(
    ///     map.musical_time_to_secs_f64(MusicalTime::from_beats(32 + 1000)),
    ///     16.0 * 0.5 + 16.0 * 1.0 + (1000. * 0.01)
    /// );
    /// ```
    pub fn musical_time_to_secs_f64(&self, ts: MusicalTime) -> f64 {
        // If we have not upheld our promise about the state of the MusicalTimeMap there is a bug
        assert!(self.tempo_changes.len() > 0);
        assert_eq!(self.tempo_changes[0].1, MusicalTime::new(0, 0));
        let mut accumulated_seconds: f64 = 0.0;
        let mut duration_remaining = ts;
        println!("duration: {duration_remaining:?}");
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
                println!("section_duration: {section_duration:?}, duration_remaining: {duration_remaining:?}, accumulated_seconds: {accumulated_seconds}");
            } else {
                accumulated_seconds += tempo_change_pair0.0.to_secs_f64(duration_remaining);
                duration_remaining = MusicalTime::new(0, 0);
                break;
            }
        }

        if duration_remaining > MusicalTime::new(0, 0) {
            // The time stamp given is after the last tempo change, simply
            // calculate the remaining duration using the last tempo change.
            println!("Thereis duration remaining: {duration_remaining:?}");
            accumulated_seconds += self
                .tempo_changes
                .last()
                .unwrap()
                .0
                .to_secs_f64(duration_remaining);
        }

        accumulated_seconds
    }
    pub fn len(&self) -> usize {
        self.tempo_changes.len()
    }
    /// Returns true if the tempo changes are in order, false if not. For testing purposes.
    pub fn is_sorted(&self) -> bool {
        let mut last_musical_time = MusicalTime::new(0, 0);
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
            tempo_changes: vec![(TempoChange::NewTempo { bpm: 60.0 }, MusicalTime::new(0, 0))],
        }
    }
}

/// Exposes the shared MusicalTimeMap in a read only Sync container.
pub struct MusicalTimeMapRef(Arc<RwLock<MusicalTimeMap>>);

mod tests {
    #[test]
    fn musical_time_test() {
        use crate::scheduling::{MusicalTime, MusicalTimeMap, TempoChange};
        let mut map = MusicalTimeMap::new();
        assert_eq!(map.musical_time_to_secs_f64(MusicalTime::new(0, 0)), 0.0);
        // Starts with a TempoChange for a constant 60 bpm per default
        assert_eq!(map.musical_time_to_secs_f64(MusicalTime::new(1, 0)), 1.0);
        assert_eq!(
            map.musical_time_to_secs_f64(MusicalTime::from_quarter_beats(5, 3)),
            5.75
        );
        map.replace(0, TempoChange::NewTempo { bpm: 120.0 });
        assert_eq!(map.musical_time_to_secs_f64(MusicalTime::new(1, 0)), 0.5);
        assert_eq!(
            map.musical_time_to_secs_f64(MusicalTime::from_quarter_beats(5, 3)),
            5.75 * 0.5
        );
        map.insert(
            TempoChange::NewTempo { bpm: 60.0 },
            MusicalTime::from_beats(16),
        );
        map.insert(
            TempoChange::NewTempo { bpm: 6000.0 },
            MusicalTime::from_beats(32),
        );
        assert_eq!(
            map.musical_time_to_secs_f64(MusicalTime::from_beats(17)),
            16.0 * 0.5 + 1.0
        );
        assert_eq!(
            map.musical_time_to_secs_f64(MusicalTime::from_beats(32)),
            16.0 * 0.5 + 16.0
        );
        assert_eq!(
            map.musical_time_to_secs_f64(MusicalTime::from_beats(33)),
            16.0 * 0.5 + 16.0 + 0.01
        );
        assert_eq!(
            map.musical_time_to_secs_f64(MusicalTime::from_beats(32 + 1000)),
            16.0 * 0.5 + 16.0 + 10.
        );
    }
}
