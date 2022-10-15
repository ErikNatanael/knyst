//! # Knyst - audio graph and synthesis library
//!
//! Knyst is a real time audio synthesis framework focusing on flexibility and
//! performance. It's main target use case is desktop multi-threaded real time
//! environments, but it can also do single threaded and/or non real time
//! synthesis. Embedded platforms are currently not supported, but on the
//! roadmap.
//!
//! The main selling point of Knyst is that the graph can be modified while it's
//! running: nodes and connections between nodes can be added/removed. It also
//! supports shared resources such as wavetables and buffers.
//!
//! ## Status
//!
//! Knyst is in its early stages. Expect large breaking API changes between
//! versions.
//!
//! ## The name
//!
//! "Knyst" is a Swedish word meaning _very faint sound_.
//!
//! ## Architecture
//!
//! The core of Knyst is the [`Graph`] struct and the [`Gen`] trait. [`Graph`]s
//! can have nodes containing anything that implements [`Gen`]. [`Graph`]s
//! can also themselves be added as a node.
//!
//! Nodes in a running [`Graph`] can be freed or signal to the [`Graph`]
//! that they or the entire [`Graph`] should be freed. [`Connection`]s between
//! Nodes and the inputs and outputs of a [`Graph`] can also be changed
//! while the [`Graph`] is running. This way, Knyst acheives a similar
//! flexibility to SuperCollider.
//!
//! It is easy to get things wrong when using a [`Graph`] as a [`Gen`] directly
//! so that functionality is encapsulated. For the highest level [`Graph`] of
//! your program you may want to use [`RunGraph`] to get a node which
//! you can run in a real time thread or non real time to generate samples.
//! Using the [`audio_backend`]s this process is automated for you.
//!

#![deny(rustdoc::broken_intra_doc_links)] // error if there are broken intra-doc links

#[warn(missing_docs)]
use buffer::{Buffer, BufferKey};
use core::fmt::Debug;
use downcast_rs::{impl_downcast, Downcast};
use graph::GenState;
// Import these for docs
#[allow(unused_imports)]
use graph::{Connection, Gen, Graph, RunGraph};
use slotmap::SlotMap;
use std::collections::HashMap;
use wavetable::{Wavetable, WavetableKey};

use crate::wavetable::{FRACTIONAL_PART, TABLE_SIZE};

pub mod audio_backend;
pub mod buffer;
pub mod envelope;
pub mod graph;
pub mod prelude;
pub mod wavetable;
pub mod xorrng;

pub type Sample = f32;
pub trait AnyData: Downcast + Send + Debug {}
impl_downcast!(AnyData);

/// Specify what happens when a [`Gen`] is done with its processing. This translates to a [`GenState`] being returned from the [`Gen`], but without additional parameters.
#[derive(Debug, Clone, Copy)]
pub enum StopAction {
    /// Continue running
    Continue,
    /// Free the node containing the Gen
    FreeSelf,
    /// Free the node containing the Gen, bridging its input node(s) to its output node(s).
    FreeSelfMendConnections,
    /// Free the graph containing the node containing the Gen.
    FreeGraph,
    /// Free the graph containing the node containing the Gen, bridging its input node(s) to its output node(s).
    FreeGraphMendConnections,
}
impl StopAction {
    #[must_use]
    pub fn to_gen_state(&self, stop_sample: usize) -> GenState {
        match self {
            StopAction::Continue => GenState::Continue,
            StopAction::FreeSelf => GenState::FreeSelf,
            StopAction::FreeSelfMendConnections => GenState::FreeSelfMendConnections,
            StopAction::FreeGraph => GenState::FreeGraph(stop_sample),
            StopAction::FreeGraphMendConnections => GenState::FreeGraphMendConnections(stop_sample),
        }
    }
}

#[derive(Copy, Clone, Debug)]
/// Settings used to initialise [`Resources`].
pub struct ResourcesSettings {
    pub sample_rate: Sample,
    /// The maximum number of wavetables that can be added to the Resources. The standard wavetables will always be available regardless.
    pub max_wavetables: usize,
    /// The maximum number of buffers that can be added to the Resources
    pub max_buffers: usize,
    pub max_user_data: usize,
}
impl Default for ResourcesSettings {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            max_wavetables: 10,
            max_buffers: 10,
            max_user_data: 0,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ResourcesError {
    #[error("There is not enough space to insert the given Wavetable. You can create a Resources with more space or remove old Wavetables")]
    WavetablesFull(Wavetable),
    #[error("There is not enough space to insert the given Buffer. You can create a Resources with more space or remove old Buffers")]
    BuffersFull(Buffer),
}

/// Common resources for all Nodes in a Graph and all its sub Graphs:
/// - [`Wavetable`]
/// - [`Buffer`]
/// - [`fastrand::Rng`]
///
/// You can also add any resource you need to be shared between nodes using [`AnyData`].
pub struct Resources {
    pub buffers: SlotMap<BufferKey, Buffer>,
    pub wavetables: SlotMap<WavetableKey, Wavetable>,
    /// A precalculated value based on the sample rate and the table size. The
    /// frequency * this number is the amount that the phase should increase one
    /// sample. It is stored here so that it doesn't need to be stored in every
    /// wavetable oscillator.
    pub freq_to_phase_inc: f64,
    /// UserData is meant for data that needs to be read by many nodes and
    /// updated for all of them simultaneously. Strings are used as keys for
    /// simplicity. A HopSlotMap could be used, but it would require sending and
    /// matching keys back and forth.
    ///
    /// This is a temporary solution. If you have a suggestion for a better way
    /// to make Resources user extendable, plese get in touch.
    pub user_data: HashMap<String, Box<dyn AnyData>>,

    /// The sample rate of the audio process
    pub sample_rate: Sample,
    pub rng: fastrand::Rng,
}

impl Resources {
    #[must_use]
    pub fn new(settings: ResourcesSettings) -> Self {
        // let user_data = HopSlotMap::with_capacity_and_key(1000);
        let user_data = HashMap::with_capacity(1000);
        let rng = fastrand::Rng::new();
        // Add standard wavetables to the arena
        let wavetables = SlotMap::with_capacity_and_key(settings.max_wavetables);
        let buffers = SlotMap::with_capacity_and_key(settings.max_buffers);

        let freq_to_phase_inc =
            TABLE_SIZE as f64 * FRACTIONAL_PART as f64 * (1.0 / settings.sample_rate as f64);

        Resources {
            buffers,
            wavetables,
            freq_to_phase_inc,
            user_data,
            sample_rate: settings.sample_rate,
            rng,
        }
    }
    /// Insert any kind of data using [`AnyData`].
    ///
    /// # Errors
    /// Returns the `data` in an error if there is not enough space for the data.
    pub fn insert_user_data(
        &mut self,
        key: String,
        data: Box<dyn AnyData>,
    ) -> Result<(), Box<dyn AnyData>> {
        if self.user_data.len() < self.user_data.capacity() {
            self.user_data.insert(key, data);
            Ok(())
        } else {
            Err(data)
        }
    }
    pub fn get_user_data(&mut self, key: &String) -> Option<&mut Box<dyn AnyData>> {
        self.user_data.get_mut(key)
    }

    pub fn insert_wavetable(
        &mut self,
        wavetable: Wavetable,
    ) -> Result<WavetableKey, ResourcesError> {
        if self.wavetables.len() < self.wavetables.capacity() {
            Ok(self.wavetables.insert(wavetable))
        } else {
            Err(ResourcesError::WavetablesFull(wavetable))
        }
    }
    pub fn remove_wavetable(&mut self, wavetable_key: WavetableKey) -> Option<Wavetable> {
        self.wavetables.remove(wavetable_key)
    }
    pub fn insert_buffer(&mut self, buf: Buffer) -> Result<BufferKey, ResourcesError> {
        if self.buffers.len() < self.buffers.capacity() {
            Ok(self.buffers.insert(buf))
        } else {
            Err(ResourcesError::BuffersFull(buf))
        }
    }
    /// Returns the rate with which a buffer needs to be played to sound at its original speed at the current sample rate.
    ///
    /// # Example:
    /// ```
    /// # use knyst::prelude::*;
    /// // Create a buffer with a sample rate of 44100.0
    /// let buffer = Buffer::new(1024, 1, 44100.0);
    /// let mut resources = Resources::new(ResourcesSettings{
    ///     sample_rate: 48000.,
    ///     ..Default::default()
    /// });
    /// let key = resources.insert_buffer(buffer).unwrap();
    /// assert_eq!(resources.buf_rate_scale(key), 44100.0 / 48000.0);
    /// ```
    pub fn buf_rate_scale(&self, buffer_key: BufferKey) -> f64 {
        if let Some(buf) = self.buffers.get(buffer_key) {
            buf.buf_rate_scale(self.sample_rate)
        } else {
            1.0
        }
    }
    /// Removes the buffer and returns it if the key is valid. Don't do this on
    /// the audio thread unless you have a way of sending the buffer to a
    /// different thread for deallocation.
    pub fn remove_buffer(&mut self, buffer_key: BufferKey) -> Option<Buffer> {
        self.buffers.remove(buffer_key)
    }
}

#[must_use]
pub fn db_to_amplitude(db: f32) -> f32 {
    10.0_f32.powf(db / 20.)
}
#[must_use]
pub fn amplitude_to_db(amplitude: f32) -> f32 {
    20.0 * amplitude.log10()
}
