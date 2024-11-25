//! Audio backends for getting up and running quickly.
//! To use the backends in this module you need to enable either the jack or the cpal feature.
//!
//! [`JackBackend`] currently has better support including a duplex client with
//! the same number of inputs and outputs as the [`Graph`].
//!
//! To use an [`AudioBackend`], first create it to get the parameters of the
//! system. When you have created your main graph, call
//! [`AudioBackend::start_processing`]. This will do something similar to
//! creating a [`RunGraph`] from a `&mut Graph` and a `Resources` and populating
//! the backend output buffer with the output of the [`Graph`]. From this point,
//! the [`Graph`] is considered to be running, meaning changes to the [`Graph`]
//! may take longer to perform since they involve the audio thread.

use crate::{
    controller::Controller, graph::RunGraphSettings, prelude::MultiThreadedKnystCommands,
    KnystError,
};
#[allow(unused)]
use crate::{
    graph::{Graph, RunGraph},
    Resources,
};

#[cfg(feature = "cpal")]
pub use cpal_backend::{CpalBackend, CpalBackendOptions};
#[cfg(feature = "jack")]
pub use jack_backend::JackBackend;

/// Unified API for different backends.
pub trait AudioBackend {
    /// Starts processing and returns a [`Controller`]. This is the easiest
    /// option and will run the [`Controller`] in a loop on a new thread.
    fn start_processing(
        &mut self,
        graph: Graph,
        resources: Resources,
        run_graph_settings: RunGraphSettings,
        error_handler: Box<dyn FnMut(KnystError) + Send + 'static>,
    ) -> Result<MultiThreadedKnystCommands, AudioBackendError> {
        let controller = self.start_processing_return_controller(
            graph,
            resources,
            run_graph_settings,
            error_handler,
        )?;
        Ok(controller.start_on_new_thread())
    }
    /// Starts processing and returns a [`Controller`]. This is suitable if you
    /// want to run single threaded or handle running the [`Controller`]
    /// manually.
    fn start_processing_return_controller(
        &mut self,
        graph: Graph,
        resources: Resources,
        run_graph_settings: RunGraphSettings,
        error_handler: Box<dyn FnMut(KnystError) + Send + 'static>,
    ) -> Result<Controller, AudioBackendError>;
    /// Stop the backend
    fn stop(&mut self) -> Result<(), AudioBackendError>;
    /// Get the native sample rate of the backend
    fn sample_rate(&self) -> usize;
    /// Get the native block size of the backend if there is one
    fn block_size(&self) -> Option<usize>;
    /// Get the native number of output channels for this backend, if any
    fn native_output_channels(&self) -> Option<usize>;
    /// Get the native number of input channels for this backend, if any
    fn native_input_channels(&self) -> Option<usize>;
}

#[allow(missing_docs)]
#[derive(thiserror::Error, Debug)]
pub enum AudioBackendError {
    #[error("You tried to start a backend that was already running. A backend can only be started once.")]
    BackendAlreadyRunning,
    #[error("You tried to stop a backend that was already stopped.")]
    BackendNotRunning,
    #[error("Unable to create a node from the Graph: {0}")]
    CouldNotCreateNode(String),
    #[error(transparent)]
    RunGraphError(#[from] crate::graph::run_graph::RunGraphError),
    #[cfg(feature = "jack")]
    #[error(transparent)]
    JackError(#[from] jack::Error),
    #[cfg(feature = "cpal")]
    #[error(transparent)]
    CpalDevicesError(#[from] cpal::DevicesError),
    #[cfg(feature = "cpal")]
    #[error(transparent)]
    CpalDeviceNameError(#[from] cpal::DeviceNameError),
    #[cfg(feature = "cpal")]
    #[error(transparent)]
    CpalStreamError(#[from] cpal::StreamError),
    #[cfg(feature = "cpal")]
    #[error(transparent)]
    CpalBuildStreamError(#[from] cpal::BuildStreamError),
    #[cfg(feature = "cpal")]
    #[error(transparent)]
    CpalPlayStreamError(#[from] cpal::PlayStreamError),
}

#[cfg(feature = "jack")]
mod jack_backend {
    use crate::audio_backend::{AudioBackend, AudioBackendError};
    use crate::controller::Controller;
    use crate::graph::{RunGraph, RunGraphSettings};
    use crate::{graph::Graph, Resources};
    use crate::{KnystError, Sample};
    #[cfg(all(debug_assertions, feature = "assert_no_alloc"))]
    use assert_no_alloc::*;
    enum JackClient {
        Passive(jack::Client),
        Active(jack::AsyncClient<JackNotifications, JackProcess>),
    }

    /// A backend using JACK
    pub struct JackBackend {
        client: Option<JackClient>,
        sample_rate: usize,
        block_size: usize,
    }

    impl JackBackend {
        /// Create a new JACK client using the given name
        pub fn new<S: AsRef<str>>(name: S) -> Result<Self, jack::Error> {
            // Create client
            let (client, _status) =
                jack::Client::new(name.as_ref(), jack::ClientOptions::NO_START_SERVER).unwrap();
            let sample_rate = client.sample_rate();
            let block_size = client.buffer_size() as usize;
            Ok(Self {
                client: Some(JackClient::Passive(client)),
                sample_rate,
                block_size,
            })
        }
    }

    impl AudioBackend for JackBackend {
        fn start_processing_return_controller(
            &mut self,
            mut graph: Graph,
            resources: Resources,
            run_graph_settings: RunGraphSettings,
            error_handler: Box<dyn FnMut(KnystError) + Send + 'static>,
        ) -> Result<Controller, AudioBackendError> {
            if let Some(JackClient::Passive(client)) = self.client.take() {
                let mut in_ports = vec![];
                let mut out_ports = vec![];
                let num_inputs = graph.num_inputs();
                let num_outputs = graph.num_outputs();
                for i in 0..num_inputs {
                    in_ports
                        .push(client.register_port(&format!("in_{i}"), jack::AudioIn::default())?);
                }
                for i in 0..num_outputs {
                    out_ports.push(
                        client.register_port(&format!("out_{i}"), jack::AudioOut::default())?,
                    );
                }
                let (run_graph, resources_command_sender, resources_command_receiver) =
                    RunGraph::new(&mut graph, resources, run_graph_settings)?;
                let jack_process = JackProcess {
                    run_graph,
                    in_ports,
                    out_ports,
                };
                // Activate the client, which starts the processing.
                let active_client = client
                    .activate_async(JackNotifications::default(), jack_process)
                    .unwrap();
                self.client = Some(JackClient::Active(active_client));
                let controller = Controller::new(
                    graph,
                    error_handler,
                    resources_command_sender,
                    resources_command_receiver,
                );
                Ok(controller)
            } else {
                Err(AudioBackendError::BackendAlreadyRunning)
            }
        }

        fn stop(&mut self) -> Result<(), AudioBackendError> {
            if let Some(JackClient::Active(active_client)) = self.client.take() {
                active_client.deactivate().unwrap();
                Ok(())
            } else {
                return Err(AudioBackendError::BackendNotRunning);
            }
        }

        fn sample_rate(&self) -> usize {
            self.sample_rate
        }

        fn block_size(&self) -> Option<usize> {
            Some(self.block_size)
        }

        fn native_output_channels(&self) -> Option<usize> {
            None
        }

        fn native_input_channels(&self) -> Option<usize> {
            None
        }
    }

    struct JackProcess {
        run_graph: RunGraph,
        in_ports: Vec<jack::Port<jack::AudioIn>>,
        out_ports: Vec<jack::Port<jack::AudioOut>>,
    }

    impl jack::ProcessHandler for JackProcess {
        fn process(&mut self, _: &jack::Client, ps: &jack::ProcessScope) -> jack::Control {
            // Duplication due to conditional compilation
            #[cfg(all(debug_assertions, feature = "assert_no_alloc"))]
            {
                assert_no_alloc(|| {
                    let graph_input_buffers = self.run_graph.graph_input_buffers();
                    for (i, in_port) in self.in_ports.iter().enumerate() {
                        let in_port_slice = in_port.as_slice(ps);
                        let in_buffer = unsafe { graph_input_buffers.get_channel_mut(i) };
                        // in_buffer.clone_from_slice(in_port_slice);
                        for (from_jack, graph_in) in in_port_slice.iter().zip(in_buffer.iter_mut())
                        {
                            *graph_in = *from_jack as Sample;
                        }
                    }
                    self.run_graph.run_resources_communication(50);
                    self.run_graph.process_block();

                    let graph_output_buffers = self.run_graph.graph_output_buffers_mut();
                    for (i, out_port) in self.out_ports.iter_mut().enumerate() {
                        let out_buffer = unsafe { graph_output_buffers.get_channel_mut(i) };
                        for sample in out_buffer.iter_mut() {
                            *sample = sample.clamp(-1.0, 1.0);
                            if sample.is_nan() {
                                *sample = 0.0;
                            }
                        }
                        let out_port_slice = out_port.as_mut_slice(ps);
                        // out_port_slice.clone_from_slice(out_buffer);
                        for (to_jack, graph_out) in out_port_slice.iter_mut().zip(out_buffer.iter())
                        {
                            *to_jack = *graph_out as f32;
                        }
                    }
                    jack::Control::Continue
                })
            }
            #[cfg(not(all(debug_assertions, feature = "assert_no_alloc")))]
            {
                let graph_input_buffers = self.run_graph.graph_input_buffers();
                for (i, in_port) in self.in_ports.iter().enumerate() {
                    let in_port_slice = in_port.as_slice(ps);
                    let in_buffer = unsafe { graph_input_buffers.get_channel_mut(i) };
                    // in_buffer.clone_from_slice(in_port_slice);
                    for (from_jack, graph_in) in in_port_slice.iter().zip(in_buffer.iter_mut()) {
                        *graph_in = *from_jack as Sample;
                    }
                }
                self.run_graph.run_resources_communication(50);
                self.run_graph.process_block();

                let graph_output_buffers = self.run_graph.graph_output_buffers_mut();
                for (i, out_port) in self.out_ports.iter_mut().enumerate() {
                    let out_buffer = unsafe { graph_output_buffers.get_channel_mut(i) };
                    for sample in out_buffer.iter_mut() {
                        *sample = sample.clamp(-1.0, 1.0);
                        if sample.is_nan() {
                            *sample = 0.0;
                        }
                    }
                    let out_port_slice = out_port.as_mut_slice(ps);
                    // out_port_slice.clone_from_slice(out_buffer);
                    for (to_jack, graph_out) in out_port_slice.iter_mut().zip(out_buffer.iter()) {
                        *to_jack = *graph_out as f32;
                    }
                }
                jack::Control::Continue
            }
        }
    }

    struct JackNotifications;
    impl Default for JackNotifications {
        fn default() -> Self {
            Self
        }
    }

    impl jack::NotificationHandler for JackNotifications {
        fn thread_init(&self, _: &jack::Client) {}

        unsafe fn shutdown(&mut self, _status: jack::ClientStatus, _reason: &str) {}

        fn freewheel(&mut self, _: &jack::Client, _is_enabled: bool) {}

        fn sample_rate(&mut self, _: &jack::Client, _srate: jack::Frames) -> jack::Control {
            // println!("JACK: sample rate changed to {}", srate);
            jack::Control::Continue
        }

        fn client_registration(&mut self, _: &jack::Client, _name: &str, _is_reg: bool) {
            // println!(
            //     "JACK: {} client with name \"{}\"",
            //     if is_reg { "registered" } else { "unregistered" },
            //     name
            // );
        }

        fn port_registration(&mut self, _: &jack::Client, _port_id: jack::PortId, _is_reg: bool) {
            // println!(
            //     "JACK: {} port with id {}",
            //     if is_reg { "registered" } else { "unregistered" },
            //     port_id
            // );
        }

        fn port_rename(
            &mut self,
            _: &jack::Client,
            _port_id: jack::PortId,
            _old_name: &str,
            _new_name: &str,
        ) -> jack::Control {
            // println!(
            //     "JACK: port with id {} renamed from {} to {}",
            //     port_id, old_name, new_name
            // );
            jack::Control::Continue
        }

        fn ports_connected(
            &mut self,
            _: &jack::Client,
            _port_id_a: jack::PortId,
            _port_id_b: jack::PortId,
            _are_connected: bool,
        ) {
            // println!(
            //     "JACK: ports with id {} and {} are {}",
            //     port_id_a,
            //     port_id_b,
            //     if are_connected {
            //         "connected"
            //     } else {
            //         "disconnected"
            //     }
            // );
        }

        fn graph_reorder(&mut self, _: &jack::Client) -> jack::Control {
            // println!("JACK: graph reordered");
            jack::Control::Continue
        }

        fn xrun(&mut self, _: &jack::Client) -> jack::Control {
            // println!("JACK: xrun occurred");
            jack::Control::Continue
        }
    }
}

/// [`AudioBackend`] implementation for CPAL
#[cfg(feature = "cpal")]
pub mod cpal_backend {
    use crate::audio_backend::{AudioBackend, AudioBackendError};
    use crate::controller::Controller;
    use crate::graph::{RunGraph, RunGraphSettings};
    use crate::KnystError;
    use crate::Sample;
    use crate::{graph::Graph, Resources};
    #[cfg(all(debug_assertions, feature = "assert_no_alloc"))]
    use assert_no_alloc::*;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    #[allow(missing_docs)]
    pub struct CpalBackendOptions {
        pub device: String,
        pub verbose: bool,
    }
    impl Default for CpalBackendOptions {
        fn default() -> Self {
            Self {
                device: "default".into(),
                verbose: false,
            }
        }
    }
    /// CPAL backend for convenience. The CPAL backend currently does not support passing on audio inputs from outside the program.
    pub struct CpalBackend {
        stream: Option<cpal::Stream>,
        sample_rate: usize,
        config: cpal::SupportedStreamConfig,
        device: cpal::Device,
    }

    impl CpalBackend {
        /// Create a new CpalBackend using the default host, getting a device, but not a stream.
        pub fn new(options: CpalBackendOptions) -> Result<Self, AudioBackendError> {
            let host = cpal::default_host();

            let device = if options.device == "default" {
                host.default_output_device()
            } else {
                host.output_devices()?
                    .find(|x| x.name().map(|y| y == options.device).unwrap_or(false))
            }
            .expect("failed to find output device");
            if options.verbose {
                println!("Output device: {}", device.name()?);
            }

            let config = device.default_output_config().unwrap();
            if options.verbose {
                println!("Default output config: {:?}", config);
            }
            Ok(Self {
                stream: None,
                sample_rate: config.sample_rate().0 as usize,
                config,
                device,
            })
        }
        /// The number of outputs for the device's default output config
        pub fn num_outputs(&self) -> usize {
            self.config.channels() as usize
        }
    }

    impl AudioBackend for CpalBackend {
        fn start_processing_return_controller(
            &mut self,
            mut graph: Graph,
            resources: Resources,
            run_graph_settings: RunGraphSettings,
            error_handler: Box<dyn FnMut(KnystError) + Send + 'static>,
        ) -> Result<crate::controller::Controller, AudioBackendError> {
            if self.stream.is_some() {
                return Err(AudioBackendError::BackendAlreadyRunning);
            }
            if graph.num_outputs() != self.config.channels() as usize {
                panic!("CpalBackend expects a graph with the same number of outputs as the device. Check CpalBackend::channels().")
            }
            if graph.num_inputs() > 0 {
                eprintln!("Warning: CpalBackend currently does not support inputs into the top level Graph.")
            }
            let (run_graph, resources_command_sender, resources_command_receiver) =
                RunGraph::new(&mut graph, resources, run_graph_settings)?;
            let config = self.config.clone();
            let stream = match self.config.sample_format() {
                cpal::SampleFormat::F32 => run::<f32>(&self.device, &config.into(), run_graph),
                cpal::SampleFormat::I16 => run::<i16>(&self.device, &config.into(), run_graph),
                cpal::SampleFormat::U16 => run::<u16>(&self.device, &config.into(), run_graph),
                cpal::SampleFormat::I8 => run::<i8>(&self.device, &config.into(), run_graph),
                cpal::SampleFormat::I32 => run::<i32>(&self.device, &config.into(), run_graph),
                cpal::SampleFormat::I64 => run::<i64>(&self.device, &config.into(), run_graph),
                cpal::SampleFormat::U8 => run::<u8>(&self.device, &config.into(), run_graph),
                cpal::SampleFormat::U32 => run::<u32>(&self.device, &config.into(), run_graph),
                cpal::SampleFormat::U64 => run::<u64>(&self.device, &config.into(), run_graph),
                cpal::SampleFormat::F64 => run::<f64>(&self.device, &config.into(), run_graph),
                _ => todo!(),
            }?;
            self.stream = Some(stream);
            let controller = Controller::new(
                graph,
                error_handler,
                resources_command_sender,
                resources_command_receiver,
            );
            Ok(controller)
        }

        fn stop(&mut self) -> Result<(), AudioBackendError> {
            todo!()
        }

        fn sample_rate(&self) -> usize {
            self.sample_rate
        }

        fn block_size(&self) -> Option<usize> {
            None
        }

        fn native_output_channels(&self) -> Option<usize> {
            Some(self.num_outputs())
        }

        fn native_input_channels(&self) -> Option<usize> {
            // TODO: support duplex streams
            Some(0)
        }
    }

    fn run<T>(
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        mut run_graph: RunGraph,
    ) -> Result<cpal::Stream, AudioBackendError>
    where
        T: cpal::Sample + cpal::FromSample<Sample> + cpal::SizedSample + std::fmt::Display,
    {
        let channels = config.channels as usize;

        // TODO: Send error back from the audio thread in a unified way.
        let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

        let mut sample_counter = 0;
        let graph_block_size = run_graph.block_size();
        run_graph.run_resources_communication(50);
        run_graph.process_block();
        let stream = device.build_output_stream(
            config,
            move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
                // TODO: When CPAL support duplex streams, copy inputs to graph inputs here.
                #[cfg(all(debug_assertions, feature = "assert_no_alloc"))]
                {
                    assert_no_alloc(|| {
                        for frame in output.chunks_mut(channels) {
                            if sample_counter >= graph_block_size {
                                run_graph.run_resources_communication(50);
                                run_graph.process_block();
                                sample_counter = 0;
                            }
                            let buffer = run_graph.graph_output_buffers();
                            // println!("{}", T::from_sample(buffer.read(0, sample_counter)));
                            for (channel_i, out) in frame.iter_mut().enumerate() {
                                let value: T =
                                    T::from_sample(buffer.read(channel_i, sample_counter));
                                *out = value;
                            }
                            sample_counter += 1;
                        }
                    })
                }
                #[cfg(not(all(debug_assertions, feature = "assert_no_alloc")))]
                {
                    for frame in output.chunks_mut(channels) {
                        if sample_counter >= graph_block_size {
                            run_graph.run_resources_communication(50);
                            run_graph.process_block();
                            sample_counter = 0;
                        }
                        let buffer = run_graph.graph_output_buffers();
                        // println!("{}", T::from_sample(buffer.read(0, sample_counter)));
                        for (channel_i, out) in frame.iter_mut().enumerate() {
                            let value: T = T::from_sample(buffer.read(channel_i, sample_counter));
                            *out = value;
                        }
                        sample_counter += 1;
                    }
                }
            },
            err_fn,
            None,
        )?;

        stream.play()?;
        Ok(stream)
    }
}
