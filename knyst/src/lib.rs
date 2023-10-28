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
//! - *debug-warn-on-alloc*: Print a warning instead of panicing when allocating on the audio thread (debug build only).
//! - *serde-derive*: Enables some data structures to be serialized/deserialized using serde.
//! - *cpal*: Enables the cpal AudioBackend
//! - *jack*: Enables the JACK AudioBackend
//!
#![deny(rustdoc::broken_intra_doc_links)] // error if there are broken intra-doc links
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![cfg_attr(feature = "unstable", feature(portable_simd))]

use core::fmt::Debug;
use knyst_core::resources::ResourcesError;
// Import these for docs
#[allow(unused_imports)]
use graph::{Connection, Graph, RunGraph};
pub use knyst_core::Resources;
pub use knyst_core::Sample;

// assert_no_alloc to make sure we are not allocating on the audio thread. The
// assertion is put in AudioBackend.
#[allow(unused_imports)]
use assert_no_alloc::*;

#[cfg(debug_assertions)] // required when disable_release is set (default)
#[global_allocator]
static A: AllocDisabler = AllocDisabler;

pub mod audio_backend;
pub mod controller;
pub mod delay;
pub mod envelope;
mod filter;
pub mod graph;
pub mod handles;
pub mod inspection;
pub mod modal_interface;
pub mod osc;
pub mod prelude;
pub mod scheduling;
pub mod sphere;
pub mod time;
pub mod trig;

/// Combined error type for Knyst, containing any other error in the library.
#[derive(thiserror::Error, Debug)]
pub enum KnystError {
    /// Error making a connection inside a [`Graph`]
    #[error("There was an error adding or removing connections between nodes: {0}")]
    ConnectionError(#[from] graph::connection::ConnectionError),
    /// Error freeing a node from a [`Graph`]
    #[error("There was an error freeing a node: {0}")]
    FreeError(#[from] graph::FreeError),
    /// Error pushing a node to a [`Graph`]
    #[error("There was an error pushing a node: {0}]")]
    PushError(#[from] graph::PushError),
    /// Error scheduling a change
    #[error("There was an error scheduling a change: {0}")]
    ScheduleError(#[from] graph::ScheduleError),
    /// Error from creating a RunGraph
    #[error("There was an error with the RunGraph: {0}")]
    RunGraphError(#[from] graph::run_graph::RunGraphError),
    /// Error from interacting with [`Resources`]
    #[error("Resources error : {0}")]
    ResourcesError(#[from] ResourcesError),
}

/// Convert db to amplitude
#[inline]
#[must_use]
pub fn db_to_amplitude(db: f32) -> f32 {
    (10.0_f32).powf(db / 20.0)
}
/// Convert amplitude to db
#[inline]
#[must_use]
pub fn amplitude_to_db(amplitude: f32) -> f32 {
    20.0 * amplitude.log10()
}
