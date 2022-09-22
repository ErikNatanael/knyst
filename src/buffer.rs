use std::{fs::File, path::Path};

use slotmap::new_key_type;
use symphonia::core::{
    audio::{AudioBufferRef, Signal},
    codecs::{DecoderOptions, CODEC_TYPE_NULL},
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

use super::Sample;

new_key_type! {
    pub struct BufferKey;
}
/// The Buffer is currently very similar to Wavetable, but they may evolve differently
#[derive(Clone, Debug)]
pub struct Buffer {
    buffer: Vec<Sample>,
    size: f64,
    /// The sample rate of the buffer, can be different from the sample rate of the audio server
    sample_rate: f64,
}

impl Buffer {
    pub fn new(size: usize, sample_rate: f64) -> Self {
        Buffer {
            buffer: vec![0.0; size],
            size: size as f64,
            sample_rate,
        }
    }
    pub fn from_vec(buffer: Vec<Sample>, sample_rate: f64) -> Self {
        let size = buffer.len() as f64;
        Buffer {
            buffer,
            size,
            sample_rate,
        }
    }

    // TODO: Return whatever error is produced
    pub fn from_file(path: &Path) -> Self {
        use symphonia::core::errors::Error;
        let mut buffer = Vec::new();
        let mut codec_params = None;
        let inp_file = File::open(path).expect("Buffer: failed to open file!");
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
                        Err(Error::ResetRequired) => {
                            // The track list has been changed. Re-examine it and create a new set of decoders,
                            // then restart the decode loop. This is an advanced feature and it is not
                            // unreasonable to consider this "the end." As of v0.5.0, the only usage of this is
                            // for chained OGG physical streams.
                            unimplemented!();
                        }
                        Err(err) => match err {
                            Error::IoError(_) => todo!(),
                            Error::DecodeError(_) => todo!(),
                            Error::SeekError(_) => todo!(),
                            Error::Unsupported(_) => todo!(),
                            Error::LimitError(_) => todo!(),
                            Error::ResetRequired => todo!(),
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
                        Ok(decoded) => {
                            // Consume the decoded audio samples
                            match decoded {
                                AudioBufferRef::F32(buf) => {
                                    for &sample in buf.chan(0) {
                                        buffer.push(sample);
                                    }
                                }
                                _ => {
                                    // Repeat for the different sample formats.
                                    unimplemented!()
                                }
                            }
                        }
                        Err(Error::IoError(_)) => {
                            // The packet failed to decode due to an IO error, skip the packet.
                            continue;
                        }
                        Err(Error::DecodeError(_)) => {
                            // The packet failed to decode due to invalid data, skip the packet.
                            continue;
                        }
                        Err(err) => {
                            // An unrecoverable error occured, halt decoding.
                            match err {
                                Error::SeekError(_) => todo!(),
                                Error::Unsupported(_) => todo!(),
                                Error::LimitError(_) => todo!(),
                                Error::ResetRequired => todo!(),
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

        let sampling_rate = if let Some(cp) = codec_params {
            println!(
                "channels: {}, rate: {}, num samples: {}",
                cp.channels.unwrap(),
                cp.sample_rate.unwrap(),
                buffer.len()
            );
            cp.sample_rate.unwrap() as f64
        } else {
            0.0
        };
        // TODO: Return Err if there's no audio data
        Self::from_vec(buffer, sampling_rate)
    }
    /// Returns the rate parameter for playing this buffer with the correct speed given that the playhead moves between 0 and 1
    pub fn buf_rate_scale(&self, server_sample_rate: f64) -> f64 {
        let sample_rate_conversion = server_sample_rate / self.sample_rate;
        1.0 / (self.size as f64 * sample_rate_conversion)
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
    /// Get the sample at the index discarding the fraction with no interpolation
    #[inline]
    pub fn get(&self, index: usize) -> Sample {
        self.buffer[index]
        // unsafe{ *self.buffer.get_unchecked(index) }
    }
    pub fn size(&self) -> f64 {
        self.size
    }
}
