//! Some [`Gen`]s benefit from shared resources: [`Buffer`]s for example [`Wavetable`]s.
//! [`Resources`] provides an interface to such shared resources.

#[allow(unused)]
use crate::controller::KnystCommands;
#[allow(unused)]
use crate::gen::Gen;
use core::fmt::Debug;
use downcast_rs::{impl_downcast, Downcast};
use slotmap::{new_key_type, SecondaryMap, SlotMap};
use std::{collections::HashMap, hash::Hash, sync::atomic::AtomicU64};

use crate::{
    buffer::{Buffer, BufferKey},
    prelude::Seconds,
    wavetable_aa::Wavetable,
};

#[derive(Copy, Clone, Debug)]
/// Settings used to initialise [`Resources`].
pub struct ResourcesSettings {
    /// The maximum number of wavetables that can be added to the Resources. The standard wavetables will always be available regardless.
    pub max_wavetables: usize,
    /// The maximum number of buffers that can be added to the Resources.
    pub max_buffers: usize,
    /// The maximum number of user data objects that can be added.
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

/// Trait for any data
pub trait AnyData: Downcast + Send + Debug {}
impl_downcast!(AnyData);

/// Error from changing a [`Resources`]
#[derive(thiserror::Error, Debug)]
pub enum ResourcesError {
    /// No space for more wavetables. Increase the `max_wavetables` setting if you need to hold more.
    #[error(
        "There is not enough space to insert the given Wavetable. You can create a Resources with more space or remove old Wavetables"
    )]
    WavetablesFull(Wavetable),
    /// No space for more buffers. Increase the `max_buffers` setting if you need to hold more.
    #[error(
        "There is not enough space to insert the given Buffer. You can create a Resources with more space or remove old Buffers"
    )]
    BuffersFull(Buffer),
    /// Tried to replace a buffer, but that buffer doesn't exist.
    #[error("The key for replacement did not exist.")]
    ReplaceBufferKeyInvalid(Buffer),
    /// The `BufferId` doesn't match any buffer in the `Resources`.
    #[error("The id supplied does not match any buffer.")]
    BufferIdNotFound(BufferId),
    /// The `WavetableId` doesn't match any wavetable in the `Resources`.
    #[error("The id supplied does not match any wavetable.")]
    WavetableIdNotFound(WavetableId),
    /// Tried to replace a wavetable, but the WavetableKey supplied doesn't exist.
    #[error("The key for replacement did not exist.")]
    ReplaceWavetableKeyInvalid(Wavetable),
}

/// Used for holding either an Id (user facing identifier) or Key (internal
/// identifier) for certain APIs.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum IdOrKey<I, K>
where
    I: Clone + Copy + Debug + Hash + Eq,
    K: Clone + Copy + Debug + Hash + Eq,
{
    #[allow(missing_docs)]
    Id(I),
    #[allow(missing_docs)]
    Key(K),
}

type IdType = u64;
/// A unique id for a Buffer. Can be converted to a [`BufferKey`] by the [`Resources`]. Also contains data about the number of channels in the buffer, which is necessary to know for many operations and cannot change once the Buffer has been uploaded.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferId {
    id: IdType,
    channels: usize,
    duration: Seconds,
}

impl BufferId {
    /// Generate a new unique `BufferId`
    pub fn new(buf: &Buffer) -> Self {
        Self {
            id: NEXT_BUFFER_ID.fetch_add(1, std::sync::atomic::Ordering::Release),
            channels: buf.num_channels(),
            duration: Seconds::from_seconds_f64(buf.length_seconds()),
        }
    }
    /// Number of channels in the Buffer this id points to
    pub fn num_channels(&self) -> usize {
        self.channels
    }
    /// The duration of this buffer at its native sample rate
    pub fn duration(&self) -> Seconds {
        self.duration
    }
}

/// Get a unique id for a Buffer from this by using `fetch_add`
pub(crate) static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(0);
/// Get a unique id for a wavetable. Starts after all the default wavetables, see WavetableId impl block
pub(crate) static NEXT_WAVETABLE_ID: AtomicU64 = AtomicU64::new(1);

/// A unique id for a Wavetable. Can be converted to a [`WavetableKey`] by the [`Resources`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct WavetableId(IdType);

impl WavetableId {
    /// Generate a new unique `WavetableId`
    pub fn new() -> Self {
        Self(NEXT_WAVETABLE_ID.fetch_add(1, std::sync::atomic::Ordering::Release))
    }
    /// Return the WavetableId for the static cosine table.
    pub fn cos() -> Self {
        Self(0)
    }
}
impl Default for WavetableId {
    fn default() -> Self {
        Self::new()
    }
}

new_key_type! {
    /// Key for selecting a wavetable that has been added to Resources.
    pub struct WavetableKey;
}

impl From<WavetableId> for IdOrKey<WavetableId, WavetableKey> {
    fn from(value: WavetableId) -> Self {
        IdOrKey::Id(value)
    }
}
impl From<BufferId> for IdOrKey<BufferId, BufferKey> {
    fn from(value: BufferId) -> Self {
        IdOrKey::Id(value)
    }
}
impl From<BufferKey> for IdOrKey<BufferId, BufferKey> {
    fn from(value: BufferKey) -> Self {
        IdOrKey::Key(value)
    }
}

/// Common resources for all Nodes in a Graph and all its sub Graphs:
/// - [`Wavetable`]
/// - [`Buffer`]
/// - [`fastrand::Rng`]
///
/// You can also add any resource you need to be shared between nodes using [`AnyData`].
pub struct Resources {
    buffers: SlotMap<BufferKey, Buffer>,
    buffer_ids: SecondaryMap<BufferKey, BufferId>,
    wavetables: SlotMap<WavetableKey, Wavetable>,
    wavetable_ids: SecondaryMap<WavetableKey, WavetableId>,
    /// UserData is meant for data that needs to be read by many nodes and
    /// updated for all of them simultaneously. Strings are used as keys for
    /// simplicity. A HopSlotMap could be used, but it would require sending and
    /// matching keys back and forth.
    ///
    /// This is a temporary solution. If you have a suggestion for a better way
    /// to make Resources user extendable, plese get in touch.
    pub user_data: HashMap<String, Box<dyn AnyData>>,

    /// A realtime safe shared random number generator
    pub rng: fastrand::Rng,
}

/// Command to modify the [`Resources`] instance while it is being used on the
/// audio thread. Prefer the methods on [`KnystCommands`] to making these
/// directly.
#[allow(missing_docs)]
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

/// Response to a [`ResourcesCommand`]. Usually used to send anything that
/// deallocates away from the audio thread, but also reports any errors.
#[allow(missing_docs)]
pub enum ResourcesResponse {
    InsertBuffer(Result<BufferKey, ResourcesError>),
    RemoveBuffer(Result<Option<Buffer>, ResourcesError>),
    ReplaceBuffer(Result<Buffer, ResourcesError>),
    InsertWavetable(Result<WavetableKey, ResourcesError>),
    RemoveWavetable(Result<Option<Wavetable>, ResourcesError>),
    ReplaceWavetable(Result<Wavetable, ResourcesError>),
}

impl Resources {
    /// Create a new `Resources` using `settings`
    #[must_use]
    pub fn new(settings: ResourcesSettings) -> Self {
        const NUM_DEFAULT_WAVETABLES: usize = 1;
        // let user_data = HopSlotMap::with_capacity_and_key(1000);
        let user_data = HashMap::with_capacity(1000);
        let rng = fastrand::Rng::new();
        // Add standard wavetables to the arena
        let wavetables =
            SlotMap::with_capacity_and_key(settings.max_wavetables + NUM_DEFAULT_WAVETABLES);
        let wavetable_ids =
            SecondaryMap::with_capacity(settings.max_wavetables + NUM_DEFAULT_WAVETABLES);
        let buffers = SlotMap::with_capacity_and_key(settings.max_buffers);
        let buffer_ids = SecondaryMap::with_capacity(settings.max_buffers);

        // let freq_to_phase_inc =
        //     TABLE_SIZE as f64 * FRACTIONAL_PART as f64 * (1.0 / settings.sample_rate as f64);

        let mut r = Resources {
            buffers,
            buffer_ids,
            wavetables,
            wavetable_ids,
            user_data,
            rng,
        };

        // Insert default wavetables
        r.insert_wavetable_with_id(Wavetable::cosine(), WavetableId::cos())
            .expect("No space in Resources for default wavetables");

        r
    }
    /// Apply the command sent from the user to change the Resources.
    pub(crate) fn apply_command(&mut self, command: ResourcesCommand) -> ResourcesResponse {
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
    /// Return a Buffer if the key is valid.
    pub fn buffer(&self, key: BufferKey) -> Option<&Buffer> {
        self.buffers.get(key)
    }
    /// Return a Wavetable if the key is valid.
    pub fn wavetable(&self, key: WavetableKey) -> Option<&Wavetable> {
        self.wavetables.get(key)
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
    /// Returns the user data if the key exists. You have to cast it to the
    /// original type yourself (see the documentation for `downcast`)
    pub fn get_user_data(&mut self, key: &String) -> Option<&mut Box<dyn AnyData>> {
        self.user_data.get_mut(key)
    }
    /// Inserts a [`Wavetable`] and returns the [`WavetableKey`] if successful, mapping the
    /// key to the given [`WavetableId`].
    ///
    /// # Errors
    /// May fail if there is not room for any more wavetables in the `Resources`
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
    /// Inserts a wavetable and returns the `WavetableKey` if successful.
    ///
    /// # Errors
    /// May fail if there is not room for any more wavetables in the `Resources`
    pub fn insert_wavetable(
        &mut self,
        wavetable: Wavetable,
    ) -> Result<WavetableKey, ResourcesError> {
        self.insert_wavetable_with_id(wavetable, WavetableId::new())
    }
    /// Removes a wavetable if the `wavetable_key` exists.
    ///
    /// NB: If this is called on the audio thread you have to send the Wavetable
    /// to a different thread for deallocation.
    pub fn remove_wavetable(&mut self, wavetable_key: WavetableKey) -> Option<Wavetable> {
        self.wavetables.remove(wavetable_key)
    }
    /// Replace the wavetable at the `wavetable_key` with a new [`Wavetable`].
    /// The key is still valid. Returns the old [`Wavetable`] if successful,
    /// otherwise returns the new one wrapped in an error.
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
    /// Inserts a buffer and returns the `BufferKey` if successful.
    ///
    /// # Errors
    /// May fail if there is not room for any more buffers in the `Resources`
    pub fn insert_buffer(&mut self, buf: Buffer) -> Result<BufferKey, ResourcesError> {
        let id = BufferId::new(&buf);
        self.insert_buffer_with_id(buf, id)
    }
    /// Inserts a buffer and returns the `BufferKey` if successful, mapping the
    /// key to the given [`BufferId`].
    ///
    /// # Errors
    /// May fail if there is not room for any more buffers in the `Resources`
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
    /// Replace the buffer at the `buffer_key` with a new [`Buffer`].
    /// The key is still valid. Returns the old [`Buffer`] if successful,
    /// otherwise returns the new one wrapped in an error.
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
    /// Returns the [`BufferKey`] corresponding to the given [`BufferId`] if
    /// there is one registered.
    pub fn buffer_key_from_id(&self, buf_id: BufferId) -> Option<BufferKey> {
        for (key, &id) in &self.buffer_ids {
            if id == buf_id {
                return Some(key);
            }
        }
        None
    }
    /// Returns the [`WavetableKey`] corresponding to the given [`WavetableId`] if
    /// there is one registered.
    pub fn wavetable_key_from_id(&self, wavetable_id: WavetableId) -> Option<WavetableKey> {
        for (key, &id) in &self.wavetable_ids {
            if id == wavetable_id {
                return Some(key);
            }
        }
        None
    }
}
