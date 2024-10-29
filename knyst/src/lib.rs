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
//! ## Features
//!
//! - *unstable*: Enables unstable optimisations in some cases that currently requires nightly.
//! - *assert_no_alloc*: (default) Panics in debug builds if an allocation is detected on the audio thread.
//! - *debug-warn-on-alloc*: Print a warning instead of panicing when allocating on the audio thread (debug build only).
//! - *serde-derive*: Enables some data structures to be serialized/deserialized using serde.
//! - *cpal*: (default) Enables the cpal AudioBackend
//! - *jack*: (default) Enables the JACK AudioBackend
//!
#![deny(rustdoc::broken_intra_doc_links)] // error if there are broken intra-doc links
#![warn(missing_docs)]
// #![warn(clippy::pedantic)]
#![cfg_attr(feature = "unstable", feature(portable_simd))]

#[allow(unused)]
use crate::audio_backend::AudioBackend;
#[allow(unused)]
use crate::gen::Gen;
#[allow(unused)]
use crate::sphere::KnystSphere;
use audio_backend::AudioBackendError;
use core::fmt::Debug;
use modal_interface::SphereError;
use resources::ResourcesError;
use std::ops::{Deref, DerefMut};
// Import these for docs
#[allow(unused_imports)]
use graph::{Connection, Graph, RunGraph};
pub use modal_interface::knyst_commands;
pub use resources::Resources;

// assert_no_alloc to make sure we are not allocating on the audio thread. The
// assertion is put in AudioBackend.
#[allow(unused_imports)]
#[cfg(all(debug_assertions, feature = "assert_no_alloc"))]
use assert_no_alloc::*;

#[cfg(all(debug_assertions, feature = "assert_no_alloc"))]
#[global_allocator]
static A: AllocDisabler = AllocDisabler;

pub mod audio_backend;
pub mod buffer;
pub mod controller;
pub mod envelope;
pub mod gen;
pub mod graph;
pub mod handles;
pub mod inspection;
mod internal_filter;
pub mod modal_interface;
pub mod node_buffer;
pub mod offline;
pub mod prelude;
pub mod resources;
pub mod scheduling;
pub mod sphere;
pub mod time;
pub mod trig;
pub mod wavetable;
pub mod wavetable_aa;
pub mod xorrng;

/// Combined error type for Knyst, containing any other error in the library.
#[derive(thiserror::Error, Debug)]
pub enum KnystError {
    /// Error making a connection inside a [`Graph`]
    #[error("Error adding or removing connections: {0}")]
    ConnectionError(#[from] graph::connection::ConnectionError),
    /// Error freeing a node from a [`Graph`]
    #[error("Error freeing a node: {0}")]
    FreeError(#[from] graph::FreeError),
    /// Error pushing a node to a [`Graph`]
    #[error("Error pushing a node: {0}]")]
    PushError(#[from] graph::PushError),
    /// Error scheduling a change
    #[error("Error scheduling a change: {0}")]
    ScheduleError(#[from] graph::ScheduleError),
    /// Error from creating a RunGraph
    #[error("Error with the RunGraph: {0}")]
    RunGraphError(#[from] graph::run_graph::RunGraphError),
    /// Error from interacting with [`Resources`]
    #[error("Resources error : {0}")]
    ResourcesError(#[from] ResourcesError),
    /// Error from interacting with [`KnystSphere`] or any of the modal command functions
    #[error("Sphere error : {0}")]
    SphereError(#[from] SphereError),
    /// Error from interacting with an [`AudioBackend`].
    #[error("Audio backend error : {0}")]
    AudioBackendError(#[from] AudioBackendError),
}

/// Convert db to amplitude
#[inline]
#[must_use]
pub fn db_to_amplitude(db: Sample) -> Sample {
    (10.0 as Sample).powf(db / 20.0)
}
/// Convert amplitude to db
#[inline]
#[must_use]
pub fn amplitude_to_db(amplitude: Sample) -> Sample {
    20.0 * amplitude.log10()
}

/// The current sample type used throughout Knyst
pub type Sample = f32;

/// Marker for inputs that are trigs. This makes it possible to set that value correctly through a Handle.
pub type Trig = Sample;

/// Newtype for a sample rate to identify it in function signatures. Derefs to a `Sample` for easy use on the audio thread.
#[derive(Copy, Clone, Debug)]
pub struct SampleRate(pub Sample);

impl SampleRate {
    #[inline(always)]
    #[allow(missing_docs)]
    pub fn to_f64(self) -> f64 {
        self.0 as f64
    }
    #[allow(missing_docs)]
    #[inline(always)]
    pub fn to_usize(self) -> usize {
        self.0 as usize
    }
}

impl Deref for SampleRate {
    type Target = Sample;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SampleRate {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<SampleRate> for f64 {
    fn from(value: SampleRate) -> Self {
        value.0 as f64
    }
}

impl From<f32> for SampleRate {
    fn from(value: f32) -> Self {
        Self(value as Sample)
    }
}
impl From<f64> for SampleRate {
    fn from(value: f64) -> Self {
        Self(value as Sample)
    }
}

#[derive(Copy, Clone, Debug)]
/// BlockSize.
///
/// Can be an unorthodox block size value in the event of a partial block at the beginning of a node's existence in the graph.
pub struct BlockSize(pub usize);

impl From<BlockSize> for usize {
    #[inline(always)]
    fn from(value: BlockSize) -> Self {
        value.0
    }
}
impl From<usize> for BlockSize {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl Deref for BlockSize {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BlockSize {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
