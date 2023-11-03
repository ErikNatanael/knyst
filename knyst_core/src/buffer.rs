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

use crate::SampleRate;

use super::Sample;

#[allow(missing_docs)]
#[derive(thiserror::Error, Debug)]
pub enum BufferError {
    #[error("Tried to load a file in an unsupported format: {0}")]
    FileFormatNotSupported(PathBuf),
    #[error("Symphonia error: {0}")]
    SymphoniaError(#[from] SymphoniaError),
}

new_key_type! {
    /// Refer to a specific buffer in a SlotMap
    pub struct BufferKey;
}
/// A buffer containing sound or data. Channels are stored interleaved in a 1-dimensional list.
#[derive(Clone, Debug)]
pub struct Buffer {
    buffer: Vec<Sample>,
    num_channels: usize,
    /// Size in number of frames independent on the number of channels.
    num_frames: f64,
    /// The sample rate of the buffer, can be different from the sample rate of the audio server
    sample_rate: f64,
}

impl Buffer {
    /// Create an empty buffer with the specified options
    pub fn new(size: usize, num_channels: usize, sample_rate: f64) -> Self {
        Buffer {
            buffer: vec![0.0; size],
            num_channels,
            num_frames: size as f64,
            sample_rate,
        }
    }
    /// Create a [`Buffer`] from a single channel buffer.
    pub fn from_vec(buffer: Vec<Sample>, sample_rate: f64) -> Self {
        let size = buffer.len() as f64;
        Buffer {
            buffer,
            num_channels: 1,
            num_frames: size,
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
        let size = (buffer.len() / num_channels) as f64;
        Buffer {
            buffer,
            num_channels,
            num_frames: size,
            sample_rate,
        }
    }

    /// Create a [`Buffer`] by loading a sound file from disk. Currently
    /// supported file formats: Wave, Ogg Vorbis, FLAC, MP3
    pub fn from_sound_file(path: impl Into<PathBuf>) -> Result<Self, BufferError> {
        let path = path.into();
        let mut buffer = Vec::new();
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
        let codec_params = match symphonia::default::get_probe().format(
            &hint,
            mss,
            &format_opts,
            &metadata_opts,
        ) {
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
                let codec_params = track.codec_params.clone();
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
                            SymphoniaError::IoError(_e) => {
                                // println!("{e}");
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

                                // Create the Sample sample buffer.
                                sample_buf = Some(SampleBuffer::<Sample>::new(duration, spec));
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
                            return Err(From::from(err));
                        }
                    }
                }
                codec_params
            }
            Err(_err) => {
                // The input was not supported by any format reader.
                return Err(BufferError::FileFormatNotSupported(path));
            }
        };

        let (sampling_rate, num_channels) = {
            let cp = codec_params;
            // println!(
            //     "channels: {}, rate: {}, num samples: {}",
            //     cp.channels.unwrap(),
            //     cp.sample_rate.unwrap(),
            //     buffer.len()
            // );
            // The channels are stored as a bit field
            // https://docs.rs/symphonia-core/0.5.1/src/symphonia_core/audio.rs.html#29-90
            // The number of bits set to 1 is the number of channels in the buffer.
            (
                cp.sample_rate.unwrap() as f64,
                cp.channels.unwrap().bits().count_ones() as usize,
            )
        };
        let buffer = buffer.into_iter().map(|v| v as Sample).collect();
        // TODO: Return Err if there's no audio data
        Ok(Self::from_vec_interleaved(
            buffer,
            num_channels,
            sampling_rate,
        ))
    }
    /// Returns the step size in samples for playing this buffer with the correct speed
    pub fn buf_rate_scale(&self, server_sample_rate: SampleRate) -> f64 {
        self.sample_rate / f64::from(server_sample_rate)
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
    pub fn save_to_disk(&self, path: impl Into<PathBuf>) -> Result<(), hound::Error> {
        let spec = hound::WavSpec {
            channels: self.num_channels as u16,
            sample_rate: self.sample_rate as u32,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path.into(), spec)?;

        for sample in &self.buffer {
            let amplitude = i16::MAX as Sample;
            writer.write_sample((sample * amplitude) as i16)?;
        }
        Ok(())
    }
    /// Size in number of frames regardless of the number of samples
    pub fn num_frames(&self) -> f64 {
        self.num_frames
    }
    /// The number of channels in the Buffer. Mustn't change once uploaded/inserted in a Resources or anything else on the audio thread.
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }
    pub fn length_seconds(&self) -> f64 {
        self.num_frames / self.sample_rate
    }
    pub fn remove_dc(&mut self) {
        let mut prev = vec![0.0; self.num_channels];
        let mut lpf_sample = vec![0.0; self.num_channels];
        for (i, sample) in self.buffer.iter_mut().enumerate() {
            let c = i % self.num_channels;
            lpf_sample[c] = lpf_sample[c] * 0.999 + *sample - prev[c];
            prev[c] = *sample;
            *sample = lpf_sample[c];
        }
    }
}
