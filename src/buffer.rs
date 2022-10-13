//! Loading sound files and other data and reading from them.
//! Module containing buffer functionality:
//! - [`Buffer`] for storing sound and other data
//! - [`BufferReader`] for reading a single channel [`Buffer`] or only the first channel from a multi channel buffer
//! - [`BufferReaderMulti`] for reading multiple channels from a [`Buffer`]. The number of channels is fixed once it has been added to a [`Graph`]

use std::{fs::File, path::PathBuf};

use slotmap::new_key_type;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::{
    audio::SampleBuffer,
    codecs::{DecoderOptions, CODEC_TYPE_NULL},
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

use crate::graph::GenContext;
#[allow(unused)]
use crate::{
    graph::{Gen, GenState, Graph},
    StopAction,
};

use super::Sample;

new_key_type! {
    pub struct BufferKey;
}
/// A buffer containing sound or data. Channels are stored interleaved in a 1-dimensional list.
#[derive(Clone, Debug)]
pub struct Buffer {
    buffer: Vec<Sample>,
    num_channels: usize,
    size: f64,
    /// The sample rate of the buffer, can be different from the sample rate of the audio server
    sample_rate: f64,
}

impl Buffer {
    pub fn new(size: usize, num_channels: usize, sample_rate: f64) -> Self {
        Buffer {
            buffer: vec![0.0; size],
            num_channels,
            size: size as f64,
            sample_rate,
        }
    }
    /// Create a [`Buffer`] from a single channel buffer.
    pub fn from_vec(buffer: Vec<Sample>, sample_rate: f64) -> Self {
        let size = buffer.len() as f64;
        Buffer {
            buffer,
            num_channels: 1,
            size,
            sample_rate,
        }
    }
    /// Create a [`Buffer`] from a multi channel buffer. Channels should be
    /// interleaved e.g. [sample0_channel0, sample0_channel1, sample1_channel0,
    /// sample1_channel1, ..] etc
    pub fn from_vec_interleaved(
        buffer: Vec<Sample>,
        num_channels: usize,
        sample_rate: f64,
    ) -> Self {
        let size = buffer.len() as f64;
        Buffer {
            buffer,
            num_channels,
            size,
            sample_rate,
        }
    }

    /// Create a [`Buffer`] by loading a sound file from disk. Currently
    /// supported file formats: Wave, Ogg Vorbis, FLAC, MP3
    pub fn from_sound_file(path: impl Into<PathBuf>) -> Result<Self, SymphoniaError> {
        let path = path.into();
        let mut buffer = Vec::new();
        let mut codec_params = None;
        let inp_file = File::open(&path).expect("Buffer: failed to open file!");
        // hint to the format registry of the decoder what file format it might be
        let mut hint = Hint::new();
        // Provide the file extension as a hint.
        if let Some(extension) = path.extension() {
            if let Some(extension_str) = extension.to_str() {
                hint.with_extension(extension_str);
            }
        }
        let mss = MediaSourceStream::new(Box::new(inp_file), Default::default());
        // Use the default options for metadata and format readers.
        let format_opts: FormatOptions = Default::default();
        let metadata_opts: MetadataOptions = Default::default();
        let mut sample_buf = None;

        // Probe the media source stream for metadata and get the format reader.
        match symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts) {
            Ok(probed) => {
                let mut reader = probed.format;
                // Find the first audio track with a known (decodeable) codec.
                let track = reader
                    .tracks()
                    .iter()
                    .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
                    .expect("no supported audio tracks");
                // Set the decoder options.
                let decode_options = DecoderOptions {
                    ..Default::default()
                };

                // Create a decoder for the stream.
                let mut decoder = symphonia::default::get_codecs()
                    .make(&track.codec_params, &decode_options)
                    .expect("unsupported codec");
                codec_params = Some(track.codec_params.clone());
                // Store the track identifier, it will be used to filter packets.
                let track_id = track.id;

                // The decode loop.
                loop {
                    // Get the next packet from the media format.
                    let packet = match reader.next_packet() {
                        Ok(packet) => packet,
                        Err(SymphoniaError::ResetRequired) => {
                            // The track list has been changed. Re-examine it and create a new set of decoders,
                            // then restart the decode loop. This is an advanced feature and it is not
                            // unreasonable to consider this "the end." As of v0.5.0, the only usage of this is
                            // for chained OGG physical streams.
                            unimplemented!();
                        }
                        Err(err) => match err {
                            SymphoniaError::IoError(e) => {
                                println!("{e}");
                                break;
                            }
                            SymphoniaError::DecodeError(_) => todo!(),
                            SymphoniaError::SeekError(_) => todo!(),
                            SymphoniaError::Unsupported(_) => todo!(),
                            SymphoniaError::LimitError(_) => todo!(),
                            SymphoniaError::ResetRequired => todo!(),
                        },
                    };

                    // Consume any new metadata that has been read since the last packet.
                    while !reader.metadata().is_latest() {
                        // Pop the old head of the metadata queue.
                        reader.metadata().pop();

                        // Consume the new metadata at the head of the metadata queue.
                    }

                    // If the packet does not belong to the selected track, skip over it.
                    if packet.track_id() != track_id {
                        continue;
                    }

                    // Decode the packet into audio samples.
                    match decoder.decode(&packet) {
                        Ok(audio_buf) => {
                            // Consume the decoded audio samples
                            // If this is the *first* decoded packet, create a sample buffer matching the
                            // decoded audio buffer format.
                            if sample_buf.is_none() {
                                // Get the audio buffer specification.
                                let spec = *audio_buf.spec();

                                // Get the capacity of the decoded buffer. Note: This is capacity, not length!
                                let duration = audio_buf.capacity() as u64;

                                // Create the f32 sample buffer.
                                sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                            }

                            // Copy the decoded audio buffer into the sample buffer in an interleaved format.
                            if let Some(buf) = &mut sample_buf {
                                buf.copy_interleaved_ref(audio_buf);

                                // TODO: Get only one channel
                                for sample in buf.samples() {
                                    buffer.push(*sample);
                                }
                            }
                        }
                        Err(SymphoniaError::IoError(_)) => {
                            // The packet failed to decode due to an IO error, skip the packet.
                            continue;
                        }
                        Err(SymphoniaError::DecodeError(_)) => {
                            // The packet failed to decode due to invalid data, skip the packet.
                            continue;
                        }
                        Err(err) => {
                            // An unrecoverable error occured, halt decoding.
                            match err {
                                SymphoniaError::SeekError(_) => todo!(),
                                SymphoniaError::Unsupported(_) => todo!(),
                                SymphoniaError::LimitError(_) => todo!(),
                                SymphoniaError::ResetRequired => todo!(),
                                _ => (),
                            }
                            panic!("{}", err);
                        }
                    }
                }
            }
            Err(err) => {
                // The input was not supported by any format reader.
                eprintln!("file not supported: {}", err);
            }
        }

        let (sampling_rate, num_channels) = if let Some(cp) = codec_params {
            println!(
                "channels: {}, rate: {}, num samples: {}",
                cp.channels.unwrap(),
                cp.sample_rate.unwrap(),
                buffer.len()
            );
            // The channels are stored as a bit field
            // https://docs.rs/symphonia-core/0.5.1/src/symphonia_core/audio.rs.html#29-90
            // The number of bits set to 1 is the number of channels in the buffer.
            (
                cp.sample_rate.unwrap() as f64,
                cp.channels.unwrap().bits().count_ones() as usize,
            )
        } else {
            (0.0, 1)
        };
        // TODO: Return Err if there's no audio data
        Ok(Self::from_vec_interleaved(
            buffer,
            num_channels,
            sampling_rate,
        ))
    }
    /// Returns the step size in samples for playing this buffer with the correct speed
    pub fn buf_rate_scale(&self, server_sample_rate: f32) -> f64 {
        self.sample_rate / server_sample_rate as f64
    }
    /// Linearly interpolate between the value in between to samples
    #[inline]
    pub fn get_linear_interp(&self, index: Sample) -> Sample {
        let mix = index.fract();
        let index_u = index as usize;
        unsafe {
            *self.buffer.get_unchecked(index_u) * (1.0 - mix)
                + *self.buffer.get_unchecked((index_u + 1) % self.buffer.len()) * mix
        }
    }
    /// Get the samples for all channels at the index.
    #[inline]
    pub fn get_interleaved(&self, index: usize) -> &[Sample] {
        let index = index * self.num_channels;
        &self.buffer[index..index + self.num_channels]
        // unsafe{ *self.buffer.get_unchecked(index) }
    }
    pub fn size(&self) -> f64 {
        self.size
    }
}

/// Reads a sample from a buffer and outputs it. In a multi channel [`Buffer`] only the first channel will be read.
/// TODO: Support rate through an argument with a default constant of 1
#[derive(Clone, Debug)]
pub struct BufferReader {
    buffer_key: BufferKey,
    read_pointer: f64,
    rate: f64,
    base_rate: f64, // The basic rate for playing the buffer at normal speed
    pub finished: bool,
    pub looping: bool,
    stop_action: StopAction,
}

impl BufferReader {
    pub fn new(buffer_key: BufferKey, rate: f64, stop_action: StopAction) -> Self {
        BufferReader {
            buffer_key,
            read_pointer: 0.0,
            base_rate: 0.0, // initialise to the correct value the first time next() is called
            rate,
            finished: false,
            looping: true,
            stop_action,
        }
    }
    pub fn reset(&mut self) {
        self.jump_to(0.0);
    }
    pub fn jump_to(&mut self, new_pointer_pos: f64) {
        self.read_pointer = new_pointer_pos;
        self.finished = false;
    }
}

impl Gen for BufferReader {
    fn process(
        &mut self,
        ctx: GenContext,
        resources: &mut crate::Resources,
    ) -> crate::graph::GenState {
        let mut stop_sample = None;
        if !self.finished {
            if let Some(buffer) = &mut resources.buffers.get(self.buffer_key) {
                // Initialise the base rate if it hasn't been set
                if self.base_rate == 0.0 {
                    self.base_rate = buffer.buf_rate_scale(resources.sample_rate);
                }

                for (i, out) in ctx.outputs.get_channel_mut(0).iter_mut().enumerate() {
                    let samples = buffer.get_interleaved((self.read_pointer) as usize);
                    *out = samples[0];
                    // println!("out: {}", sample);
                    self.read_pointer += self.base_rate * self.rate;
                    if self.read_pointer >= buffer.size() {
                        self.finished = true;
                        if self.looping {
                            self.reset();
                        }
                    }
                    if self.finished {
                        stop_sample = Some(i + 1);
                        break;
                    }
                }
            } else {
                // Output zeroes if the buffer doesn't exist.
                // TODO: Send error back to the user that the buffer doesn't exist without interrupting the audio thread.
                eprintln!("Error: BufferReader: buffer doesn't exist in Resources");
                stop_sample = Some(0);
            }
        } else {
            stop_sample = Some(0);
        }
        if let Some(stop_sample) = stop_sample {
            let output = ctx.outputs.get_channel_mut(0);
            for out in output[stop_sample..].iter_mut() {
                *out = 0.;
            }
            self.stop_action.to_gen_state(stop_sample)
        } else {
            GenState::Continue
        }
    }

    fn num_inputs(&self) -> usize {
        0
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn init(&mut self, _sample_rate: crate::graph::Sample) {}

    fn input_desc(&self, _input: usize) -> &'static str {
        ""
    }

    fn output_desc(&self, _output: usize) -> &'static str {
        "out"
    }

    fn name(&self) -> &'static str {
        "BufferReader"
    }
}

/// Play back a buffer with multiple channels. You cannot change the number of
/// channels after pushing this to a graph. If the buffer has fewer channels
/// than `num_channels`, the remaining outputs will be left at their current
/// value, not zeroed.
#[derive(Clone, Debug)]
pub struct BufferReaderMulti {
    buffer_key: BufferKey,
    read_pointer: f64,
    rate: f64,
    num_channels: usize,
    base_rate: f64, // The basic rate for playing the buffer at normal speed
    pub finished: bool,
    pub looping: bool,
    stop_action: StopAction,
}

impl BufferReaderMulti {
    pub fn new(buffer_key: BufferKey, rate: f64, stop_action: StopAction) -> Self {
        Self {
            buffer_key,
            read_pointer: 0.0,
            base_rate: 0.0, // initialise to the correct value the first time next() is called
            rate,
            num_channels: 1,
            finished: false,
            looping: true,
            stop_action,
        }
    }
    pub fn channels(mut self, num_channels: usize) -> Self {
        self.num_channels = num_channels;
        self
    }
    pub fn reset(&mut self) {
        self.jump_to(0.0);
    }
    pub fn jump_to(&mut self, new_pointer_pos: f64) {
        self.read_pointer = new_pointer_pos;
        self.finished = false;
    }
}

impl Gen for BufferReaderMulti {
    fn process(
        &mut self,
        ctx: GenContext,
        resources: &mut crate::Resources,
    ) -> crate::graph::GenState {
        let mut stop_sample = None;
        if !self.finished {
            if let Some(buffer) = &mut resources.buffers.get(self.buffer_key) {
                // Initialise the base rate if it hasn't been set
                if self.base_rate == 0.0 {
                    self.base_rate = buffer.buf_rate_scale(resources.sample_rate);
                }
                for i in 0..ctx.block_size() {
                    let samples = buffer.get_interleaved((self.read_pointer) as usize);
                    for (out_num, sample) in samples.iter().take(self.num_channels).enumerate() {
                        ctx.outputs.write(*sample, out_num, i);
                    }
                    self.read_pointer += self.base_rate * self.rate;
                    if self.read_pointer >= buffer.size() {
                        self.finished = true;
                        if self.looping {
                            self.reset();
                        }
                    }
                    if self.finished {
                        stop_sample = Some(i + 1);
                        break;
                    }
                }
            } else {
                // Output zeroes if the buffer doesn't exist.
                // TODO: Send error back to the user that the buffer doesn't exist without interrupting the audio thread.
                eprintln!("Error: BufferReader: buffer doesn't exist in Resources");
                stop_sample = Some(0);
            }
        } else {
            stop_sample = Some(0);
        }
        if let Some(stop_sample) = stop_sample {
            let output = ctx.outputs.get_channel_mut(0);
            for out in output[stop_sample..].iter_mut() {
                *out = 0.;
            }
            self.stop_action.to_gen_state(stop_sample)
        } else {
            GenState::Continue
        }
    }

    fn num_inputs(&self) -> usize {
        0
    }

    fn num_outputs(&self) -> usize {
        self.num_channels
    }

    fn output_desc(&self, output: usize) -> &'static str {
        if output < self.num_channels {
            output_str(output)
        } else {
            ""
        }
    }

    fn name(&self) -> &'static str {
        "BufferReader"
    }
}

fn output_str(num: usize) -> &'static str {
    match num {
        0 => "output0",
        1 => "output1",
        2 => "output2",
        3 => "output3",
        4 => "output4",
        5 => "output5",
        6 => "output6",
        7 => "output7",
        8 => "output8",
        9 => "output9",
        10 => "output10",
        _ => "",
    }
}
