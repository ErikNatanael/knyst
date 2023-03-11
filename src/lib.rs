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
use slotmap::{SecondaryMap, SlotMap};
use std::{collections::HashMap, hash::Hash, sync::atomic::AtomicU64};
use wavetable::{Wavetable, WavetableKey};

// assert_no_alloc to make sure we are not allocating on the audio thread. The
// assertion is put in AudioBackend.
#[allow(unused_imports)]
use assert_no_alloc::*;

#[cfg(debug_assertions)] // required when disable_release is set (default)
#[global_allocator]
static A: AllocDisabler = AllocDisabler;

pub mod audio_backend;
pub mod buffer;
pub mod controller;
pub mod envelope;
mod filter;
pub mod graph;
pub mod prelude;
pub mod scheduling;
pub mod time;
pub mod trig;
pub mod wavetable;
pub mod xorrng;

#[derive(thiserror::Error, Debug)]
pub enum KnystError {
    #[error("There was an error adding or removing connections between nodes: {0}")]
    ConnectionError(#[from] graph::connection::ConnectionError),
    #[error("There was an error freeing a node: {0}")]
    FreeError(#[from] graph::FreeError),
    #[error("There was an error pushing a node: {0}]")]
    PushError(#[from] graph::PushError),
    #[error("There was an error scheduling a change: {0}")]
    ScheduleError(#[from] graph::ScheduleError),
    #[error("There was an error with the RunGraph: {0}")]
    RunGraphError(#[from] graph::run_graph::RunGraphError),
    #[error("Resources error : {0}")]
    ResourcesError(#[from] ResourcesError),
}

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
    /// The maximum number of wavetables that can be added to the Resources. The standard wavetables will always be available regardless.
    pub max_wavetables: usize,
    /// The maximum number of buffers that can be added to the Resources
    pub max_buffers: usize,
    pub max_user_data: usize,
}
impl Default for ResourcesSettings {
    fn default() -> Self {
        Self {
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
    #[error("The key for replacement did not exist.")]
    ReplaceBufferKeyInvalid(Buffer),
    #[error("The id supplied does not match any buffer.")]
    BufferIdNotFound(BufferId),
    #[error("The id supplied does not match any wavetable.")]
    WavetableIdNotFound(WavetableId),
    #[error("The key for replacement did not exist.")]
    ReplaceWavetableKeyInvalid(Wavetable),
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum IdOrKey<I, K>
where
    I: Clone + Copy + Debug + Hash + Eq,
    K: Clone + Copy + Debug + Hash + Eq,
{
    Id(I),
    Key(K),
}

type IdType = u64;
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferId(IdType);

impl BufferId {
    pub fn new() -> Self {
        Self(NEXT_BUFFER_ID.fetch_add(1, std::sync::atomic::Ordering::Release))
    }
}

/// Get a unique id for a Graph from this by using `fetch_add`
pub(crate) static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(0);
pub(crate) static NEXT_WAVETABLE_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct WavetableId(IdType);

impl WavetableId {
    pub fn new() -> Self {
        Self(NEXT_WAVETABLE_ID.fetch_add(1, std::sync::atomic::Ordering::Release))
    }
}

impl From<WavetableId> for IdOrKey<WavetableId, WavetableKey> {
    fn from(value: WavetableId) -> Self {
        IdOrKey::Id(value)
    }
}

/// Common resources for all Nodes in a Graph and all its sub Graphs:
/// - [`Wavetable`]
/// - [`Buffer`]
/// - [`fastrand::Rng`]
///
/// You can also add any resource you need to be shared between nodes using [`AnyData`].
pub struct Resources {
    pub buffers: SlotMap<BufferKey, Buffer>,
    pub buffer_ids: SecondaryMap<BufferKey, BufferId>,
    pub wavetables: SlotMap<WavetableKey, Wavetable>,
    pub wavetable_ids: SecondaryMap<WavetableKey, WavetableId>,
    /// UserData is meant for data that needs to be read by many nodes and
    /// updated for all of them simultaneously. Strings are used as keys for
    /// simplicity. A HopSlotMap could be used, but it would require sending and
    /// matching keys back and forth.
    ///
    /// This is a temporary solution. If you have a suggestion for a better way
    /// to make Resources user extendable, plese get in touch.
    pub user_data: HashMap<String, Box<dyn AnyData>>,

    pub rng: fastrand::Rng,
}

pub enum ResourcesCommand {
    InsertBuffer {
        id: BufferId,
        buffer: Buffer,
    },
    RemoveBuffer {
        id: BufferId,
    },
    ReplaceBuffer {
        id: BufferId,
        buffer: Buffer,
    },
    InsertWavetable {
        id: WavetableId,
        wavetable: Wavetable,
    },
    RemoveWavetable {
        id: WavetableId,
    },
    ReplaceWavetable {
        id: WavetableId,
        wavetable: Wavetable,
    },
}
pub enum ResourcesResponse {
    InsertBuffer(Result<BufferKey, ResourcesError>),
    RemoveBuffer(Result<Option<Buffer>, ResourcesError>),
    ReplaceBuffer(Result<Buffer, ResourcesError>),
    InsertWavetable(Result<WavetableKey, ResourcesError>),
    RemoveWavetable(Result<Option<Wavetable>, ResourcesError>),
    ReplaceWavetable(Result<Wavetable, ResourcesError>),
}

impl Resources {
    #[must_use]
    pub fn new(settings: ResourcesSettings) -> Self {
        // let user_data = HopSlotMap::with_capacity_and_key(1000);
        let user_data = HashMap::with_capacity(1000);
        let rng = fastrand::Rng::new();
        // Add standard wavetables to the arena
        let wavetables = SlotMap::with_capacity_and_key(settings.max_wavetables);
        let wavetable_ids = SecondaryMap::with_capacity(settings.max_wavetables);
        let buffers = SlotMap::with_capacity_and_key(settings.max_buffers);
        let buffer_ids = SecondaryMap::with_capacity(settings.max_buffers);

        // let freq_to_phase_inc =
        //     TABLE_SIZE as f64 * FRACTIONAL_PART as f64 * (1.0 / settings.sample_rate as f64);

        Resources {
            buffers,
            buffer_ids,
            wavetables,
            wavetable_ids,
            user_data,
            rng,
        }
    }
    fn apply_command(&mut self, command: ResourcesCommand) -> ResourcesResponse {
        match command {
            ResourcesCommand::InsertBuffer { id, buffer } => {
                ResourcesResponse::InsertBuffer(self.insert_buffer_with_id(buffer, id))
            }
            ResourcesCommand::RemoveBuffer { id } => match self.buffer_key_from_id(id) {
                Some(key) => ResourcesResponse::RemoveBuffer(Ok(self.remove_buffer(key))),
                None => ResourcesResponse::RemoveBuffer(Err(ResourcesError::BufferIdNotFound(id))),
            },
            ResourcesCommand::ReplaceBuffer { id, buffer } => match self.buffer_key_from_id(id) {
                Some(key) => ResourcesResponse::ReplaceBuffer(self.replace_buffer(key, buffer)),
                None => ResourcesResponse::RemoveBuffer(Err(ResourcesError::BufferIdNotFound(id))),
            },
            ResourcesCommand::InsertWavetable { id, wavetable } => {
                ResourcesResponse::InsertWavetable(self.insert_wavetable_with_id(wavetable, id))
            }
            ResourcesCommand::RemoveWavetable { id } => match self.wavetable_key_from_id(id) {
                Some(key) => ResourcesResponse::RemoveWavetable(Ok(self.remove_wavetable(key))),
                None => {
                    ResourcesResponse::RemoveWavetable(Err(ResourcesError::WavetableIdNotFound(id)))
                }
            },

            ResourcesCommand::ReplaceWavetable { id, wavetable } => {
                match self.wavetable_key_from_id(id) {
                    Some(key) => {
                        ResourcesResponse::ReplaceWavetable(self.replace_wavetable(key, wavetable))
                    }
                    None => ResourcesResponse::RemoveWavetable(Err(
                        ResourcesError::WavetableIdNotFound(id),
                    )),
                }
            }
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
    pub fn insert_wavetable_with_id(
        &mut self,
        wavetable: Wavetable,
        wavetable_id: WavetableId,
    ) -> Result<WavetableKey, ResourcesError> {
        if self.wavetables.len() < self.wavetables.capacity() {
            let wavetable_key = self.wavetables.insert(wavetable);
            self.wavetable_ids.insert(wavetable_key, wavetable_id);
            Ok(wavetable_key)
        } else {
            Err(ResourcesError::WavetablesFull(wavetable))
        }
    }
    pub fn insert_wavetable(
        &mut self,
        wavetable: Wavetable,
    ) -> Result<WavetableKey, ResourcesError> {
        self.insert_wavetable_with_id(wavetable, WavetableId::new())
    }
    pub fn remove_wavetable(&mut self, wavetable_key: WavetableKey) -> Option<Wavetable> {
        self.wavetables.remove(wavetable_key)
    }
    pub fn replace_wavetable(
        &mut self,
        wavetable_key: WavetableKey,
        wavetable: Wavetable,
    ) -> Result<Wavetable, ResourcesError> {
        match self.wavetables.get_mut(wavetable_key) {
            Some(map_wavetable) => {
                let old_wavetable = std::mem::replace(map_wavetable, wavetable);
                Ok(old_wavetable)
            }
            None => Err(ResourcesError::ReplaceWavetableKeyInvalid(wavetable)),
        }
    }
    pub fn insert_buffer(&mut self, buf: Buffer) -> Result<BufferKey, ResourcesError> {
        self.insert_buffer_with_id(buf, BufferId::new())
    }
    pub fn insert_buffer_with_id(
        &mut self,
        buf: Buffer,
        buf_id: BufferId,
    ) -> Result<BufferKey, ResourcesError> {
        if self.buffers.len() < self.buffers.capacity() {
            let buf_key = self.buffers.insert(buf);
            self.buffer_ids.insert(buf_key, buf_id);
            Ok(buf_key)
        } else {
            Err(ResourcesError::BuffersFull(buf))
        }
    }
    pub fn replace_buffer(
        &mut self,
        buffer_key: BufferKey,
        buf: Buffer,
    ) -> Result<Buffer, ResourcesError> {
        match self.buffers.get_mut(buffer_key) {
            Some(map_buf) => {
                let old_buffer = std::mem::replace(map_buf, buf);
                Ok(old_buffer)
            }
            None => Err(ResourcesError::ReplaceBufferKeyInvalid(buf)),
        }
    }
    /// Removes the buffer and returns it if the key is valid. Don't do this on
    /// the audio thread unless you have a way of sending the buffer to a
    /// different thread for deallocation.
    pub fn remove_buffer(&mut self, buffer_key: BufferKey) -> Option<Buffer> {
        self.buffers.remove(buffer_key)
    }
    pub fn buffer_key_from_id(&self, buf_id: BufferId) -> Option<BufferKey> {
        for (key, &id) in self.buffer_ids.iter() {
            if id == buf_id {
                return Some(key);
            }
        }
        None
    }
    pub fn wavetable_key_from_id(&self, wavetable_id: WavetableId) -> Option<WavetableKey> {
        for (key, &id) in self.wavetable_ids.iter() {
            if id == wavetable_id {
                return Some(key);
            }
        }
        None
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
