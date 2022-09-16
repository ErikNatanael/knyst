use buffer::{Buffer, BufferIndex};
use core::fmt::Debug;
use downcast_rs::{impl_downcast, Downcast};
use graph::NodeAddress;
use std::collections::HashMap;
use wavetable::{Wavetable, WavetableArena, SINE_WAVETABLE};

pub mod audio_backend;
pub mod buffer;
pub mod envelope;
pub mod graph;
pub mod prelude;
pub mod wavetable;
pub mod xorrng;

pub type Sample = f32;
/// The number of samples per wavetable, and also the number of high bits used
/// for the phase indexing into the wavetable. With the current u32 phase, this can be maximum 16.
pub const TABLE_POWER: u32 = 16;
pub const TABLE_SIZE: usize = 2_usize.pow(TABLE_POWER);
/// The high mask is used to 0 everything above the table size so that adding
/// further would have the same effect as wrapping.
pub const TABLE_HIGH_MASK: u32 = TABLE_SIZE as u32 - 1;
pub const FRACTIONAL_PART: u32 = 65536;
pub trait AnyData: Downcast + Send + Debug {}
impl_downcast!(AnyData);

#[derive(Debug, Clone, Copy)]
pub enum StopAction {
    Continue,
    FreeSelf,
    FreeSelfMendConnections,
    FreeGraph,
    FreeGraphMendConnections,
}

/// Resources contains common resources for all Nodes in some structure.
pub struct Resources {
    // TODO: Replace with a HopSlotMap
    pub buffers: Vec<Option<Buffer>>,
    // TODO: Replace by HopSlotMap
    pub wavetable_arena: WavetableArena,
    // TODO: Merge with wavetable_arena and other wavetable things
    pub lookup_tables: Vec<Vec<Sample>>,
    /// A precalculated value based on the sample rate and the table size. The
    /// frequency * this number is the amount that the phase should increase one
    /// sample. It is stored here so that it doesn't need to be stored in every
    /// wavetable oscillator.
    // TODO: Merge with wavetable_arena and other wavetable things
    pub freq_to_phase_inc: Sample,
    // pub user_data: HopSlotMap<UserDataKey,Box<dyn UserData>>,
    /// UserData is meant for data that needs to be read by many nodes and
    /// updated for all of them simultaneously. Strings are used as keys for
    /// simplicity. A HopSlotMap could be used, but it would require sending and
    /// matching keys back and forth.
    pub user_data: HashMap<String, Box<dyn AnyData>>,

    /// The sample rate of the audio process
    pub sample_rate: Sample,
    pub rng: fastrand::Rng,
}

impl Resources {
    pub fn new(sample_rate: Sample) -> Self {
        // let user_data = HopSlotMap::with_capacity_and_key(1000);
        let user_data = HashMap::with_capacity(1000);
        let rng = fastrand::Rng::new();
        let mut wavetable_arena = WavetableArena::new();
        // Add standard wavetables to the arena
        // Calculate the two FastSine tables
        let mut fast_sine_table = Vec::with_capacity(TABLE_SIZE);
        for k in 0..TABLE_SIZE {
            fast_sine_table
                .push((std::f64::consts::TAU * (k as f64 / TABLE_SIZE as f64)).sin() as f32);
        }
        let mut fast_sine_diff_table = Vec::with_capacity(TABLE_SIZE as usize);
        for k in 0..TABLE_SIZE {
            let v0 = fast_sine_table[k];
            let v1 = fast_sine_table[(k + 1) % TABLE_SIZE];
            fast_sine_diff_table.push(v1 - v0);
        }
        let mut lookup_tables = vec![];
        let num_standard_wavetables = StandardWt::Last as u32;

        let freq_to_phase_inc =
            (TABLE_SIZE as f64 * FRACTIONAL_PART as f64 * (1.0 / sample_rate as f64)) as Sample;
        for i in 0..num_standard_wavetables {
            // Safety: We previously checked that i is within bounds
            let y: StandardWt = unsafe { std::mem::transmute(i) };
            match y {
                StandardWt::FastSine => {
                    lookup_tables.push(fast_sine_table.clone());
                }
                StandardWt::FastSineDiff => {
                    lookup_tables.push(fast_sine_diff_table.clone());
                }
                _ => (),
            }
        }
        for i in 0..1 {
            match i {
                SINE_WAVETABLE => {
                    wavetable_arena.add(Wavetable::sine());
                }
                _ => (),
            }
        }

        Resources {
            buffers: vec![None; 1000],
            wavetable_arena,
            lookup_tables,
            freq_to_phase_inc,
            user_data,
            sample_rate,
            rng,
        }
    }
    pub fn push_user_data(&mut self, key: String, data: Box<dyn AnyData>) {
        self.user_data.insert(key, data);
    }
    pub fn get_user_data(&mut self, key: &String) -> Option<&mut Box<dyn AnyData>> {
        self.user_data.get_mut(key)
    }
    pub fn push_buffer(&mut self, buf: Buffer) -> Result<BufferIndex, Buffer> {
        let mut found_free_index = false;
        let mut index: BufferIndex = 0;
        for i in 0..self.buffers.len() {
            let is_free = self.buffers[i].is_none();
            if is_free {
                index = i;
                found_free_index = true;
                break;
            }
        }
        if found_free_index {
            // The old value should always be None, but we're being extra careful here if we'd allow for using an occupied slot in the future
            let _old_value = self.buffers[index].replace(buf);
            Ok(index)
        } else {
            // If we were unable to push the buffer we have to deallocate it
            // self.dealloc_buf_sender.send(buf).unwrap();
            Err(buf)
        }
    }
    pub fn buf_rate_scale(&self, index: BufferIndex, sample_rate: f64) -> f64 {
        if let Some(buf) = &self.buffers[index] {
            buf.buf_rate_scale(sample_rate)
        } else {
            1.0
        }
    }
    pub fn remove_buffer(&mut self, index: BufferIndex) -> Option<Buffer> {
        let mut dealloc = false;
        if let Some(_buf) = &self.buffers[index] {
            dealloc = true;
        }
        if dealloc {
            let buf = self.buffers[index].take().expect("Resources::deallocate_buffers: Already checked that there is a Buffer here, something is wrong");
            return Some(buf);
        }
        None
    }
}
// The repr(C) guarantees the order of the enum variants will be conserved
#[repr(u32)]
#[allow(dead_code)]
/// Used for indexing the standard precalculated wavetables
pub enum StandardWt {
    FastSine = 0,
    FastSineDiff,
    Last,
}

/// Interface for setting up and interacting with a running audio graph.
///
/// You can just as well manually use the AudioBackend and together with Graph,
/// but using this interface instead saves you from having to think about
/// regularly updating the scheduler
struct Knyst {}

// Action: push a gen, add connections to its inputs and to its outputs
// 1. push_gen -> NodeHandle
// 2. connect input NodeHandles to NodeHandle
// 3. connect NodeHandle outputs to other NodeHandles
// 4. connect NodeHandle to the graph outputs
//
// If you're creating a whole graph all at once, you probably want to do it in a
// function that returns said Graph. Then you end up with NodeAddresses to the
// nodes within the graph anyways.

/// Asynchronous NodeAddress. Returned from the helper thread interface when
/// using it to schedule an event that returns a NodeAddress
enum NodeHandle {
    Waiting(rtrb::Consumer<NodeAddress>),
    Received(NodeAddress),
}

impl NodeHandle {
    pub fn get(&mut self) -> Option<NodeAddress> {
        let n = match self {
            NodeHandle::Waiting(consumer) => {
                if let Ok(node_address) = consumer.pop() {
                    Some(node_address)
                } else {
                    None
                }
            }
            NodeHandle::Received(n) => Some(*n),
        };
        if let Some(n) = n {
            *self = NodeHandle::Received(n);
        }
        n
    }
}
