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

#[allow(unused)]
use crate::{
    graph::{Graph, RunGraph},
    Resources,
};

#[cfg(feature = "cpal")]
pub use cpal_backend::{CpalBackend, CpalBackendOptions};
#[cfg(feature = "jack")]
pub use jack_backend::JackBackend;

pub trait AudioBackend {
    fn start_processing(
        &mut self,
        graph: &mut Graph,
        resources: Resources,
    ) -> Result<(), AudioBackendError>;
    fn stop(&mut self) -> Result<(), AudioBackendError>;
    fn sample_rate(&self) -> usize;
    fn block_size(&self) -> Option<usize>;
}

#[derive(thiserror::Error, Debug)]
pub enum AudioBackendError {
    #[error("You tried to start a backend that was already running. A backend can only be started once.")]
    BackendAlreadyRunning,
    #[error("You tried to stop a backend that was already stopped.")]
    BackendNotRunning,
    #[error("Unable to create a node from the Graph: {0}")]
    CouldNotCreateNode(String),
    #[error(transparent)]
    RunGraphError(#[from] crate::graph::RunGraphError),
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
    use crate::graph::RunGraph;
    use crate::{graph::Graph, Resources};
    enum JackClient {
        Passive(jack::Client),
        Active(jack::AsyncClient<JackNotifications, JackProcess>),
    }

    pub struct JackBackend {
        client: Option<JackClient>,
        sample_rate: usize,
        block_size: usize,
    }

    impl JackBackend {
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
        fn start_processing(
            &mut self,
            graph: &mut Graph,
            resources: Resources,
        ) -> Result<(), AudioBackendError> {
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
                let run_graph = RunGraph::new(graph, resources)?;
                let jack_process = JackProcess {
                    run_graph,
                    in_ports,
                    out_ports,
                };
                // Activate the client, which starts the processing.
                let active_client = client
                    .activate_async(JackNotifications, jack_process)
                    .unwrap();
                self.client = Some(JackClient::Active(active_client));
            } else {
                return Err(AudioBackendError::BackendAlreadyRunning);
            }
            Ok(())
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
    }

    struct JackProcess {
        run_graph: RunGraph,
        in_ports: Vec<jack::Port<jack::AudioIn>>,
        out_ports: Vec<jack::Port<jack::AudioOut>>,
    }

    impl jack::ProcessHandler for JackProcess {
        fn process(&mut self, _: &jack::Client, ps: &jack::ProcessScope) -> jack::Control {
            let graph_input_buffers = self.run_graph.graph_input_buffers();
            for (i, in_port) in self.in_ports.iter().enumerate() {
                let in_port_slice = in_port.as_slice(ps);
                let in_buffer = graph_input_buffers.get_channel_mut(i);
                in_buffer.clone_from_slice(in_port_slice);
            }
            self.run_graph.process_block();

            let graph_output_buffers = self.run_graph.graph_output_buffers();
            for (i, out_port) in self.out_ports.iter_mut().enumerate() {
                let out_buffer = graph_output_buffers.get_channel(i);
                let out_port_slice = out_port.as_mut_slice(ps);
                out_port_slice.clone_from_slice(out_buffer);
            }
            jack::Control::Continue
        }
    }

    struct JackNotifications;

    impl jack::NotificationHandler for JackNotifications {
        fn thread_init(&self, _: &jack::Client) {
            println!("JACK: thread init");
        }

        fn shutdown(&mut self, status: jack::ClientStatus, reason: &str) {
            println!(
                "JACK: shutdown with status {:?} because \"{}\"",
                status, reason
            );
        }

        fn freewheel(&mut self, _: &jack::Client, is_enabled: bool) {
            println!(
                "JACK: freewheel mode is {}",
                if is_enabled { "on" } else { "off" }
            );
        }

        fn sample_rate(&mut self, _: &jack::Client, srate: jack::Frames) -> jack::Control {
            println!("JACK: sample rate changed to {}", srate);
            jack::Control::Continue
        }

        fn client_registration(&mut self, _: &jack::Client, name: &str, is_reg: bool) {
            println!(
                "JACK: {} client with name \"{}\"",
                if is_reg { "registered" } else { "unregistered" },
                name
            );
        }

        fn port_registration(&mut self, _: &jack::Client, port_id: jack::PortId, is_reg: bool) {
            println!(
                "JACK: {} port with id {}",
                if is_reg { "registered" } else { "unregistered" },
                port_id
            );
        }

        fn port_rename(
            &mut self,
            _: &jack::Client,
            port_id: jack::PortId,
            old_name: &str,
            new_name: &str,
        ) -> jack::Control {
            println!(
                "JACK: port with id {} renamed from {} to {}",
                port_id, old_name, new_name
            );
            jack::Control::Continue
        }

        fn ports_connected(
            &mut self,
            _: &jack::Client,
            port_id_a: jack::PortId,
            port_id_b: jack::PortId,
            are_connected: bool,
        ) {
            println!(
                "JACK: ports with id {} and {} are {}",
                port_id_a,
                port_id_b,
                if are_connected {
                    "connected"
                } else {
                    "disconnected"
                }
            );
        }

        fn graph_reorder(&mut self, _: &jack::Client) -> jack::Control {
            println!("JACK: graph reordered");
            jack::Control::Continue
        }

        fn xrun(&mut self, _: &jack::Client) -> jack::Control {
            println!("JACK: xrun occurred");
            jack::Control::Continue
        }
    }
}

#[cfg(feature = "cpal")]
pub mod cpal_backend {
    use crate::audio_backend::{AudioBackend, AudioBackendError};
    use crate::graph::RunGraph;
    use crate::{graph::Graph, Resources};
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    pub struct CpalBackendOptions {
        device: String,
        verbose: bool,
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
        pub fn num_outputs(&self) -> usize {
            self.config.channels() as usize
        }
    }

    impl AudioBackend for CpalBackend {
        fn start_processing(
            &mut self,
            graph: &mut Graph,
            resources: Resources,
        ) -> Result<(), AudioBackendError> {
            if self.stream.is_some() {
                return Err(AudioBackendError::BackendAlreadyRunning);
            }
            if graph.num_outputs() != self.config.channels() as usize {
                panic!("CpalBackend expects a graph with the same number of outputs as the device. Check CpalBackend::channels().")
            }
            if graph.num_inputs() > 0 {
                eprintln!("Warning: CpalBackend currently does not support inputs into Graphs.")
            }
            let run_graph = RunGraph::new(graph, resources)?;
            let config = self.config.clone();
            let stream = match self.config.sample_format() {
                cpal::SampleFormat::F32 => run::<f32>(&self.device, &config.into(), run_graph),
                cpal::SampleFormat::I16 => run::<i16>(&self.device, &config.into(), run_graph),
                cpal::SampleFormat::U16 => run::<u16>(&self.device, &config.into(), run_graph),
            }?;
            self.stream = Some(stream);
            Ok(())
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
    }

    fn run<T>(
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        mut run_graph: RunGraph,
    ) -> Result<cpal::Stream, AudioBackendError>
    where
        T: cpal::Sample,
    {
        let channels = config.channels as usize;

        // TODO: Send error back from the audio thread in a unified way.
        let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

        let mut sample_counter = 0;
        let graph_block_size = run_graph.block_size();
        run_graph.process_block();
        let stream = device.build_output_stream(
            config,
            move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
                // TODO: When CPAL support duplex streams, copy inputs to graph inputs here.
                for frame in output.chunks_mut(channels) {
                    if sample_counter >= graph_block_size {
                        run_graph.process_block();
                        sample_counter = 0;
                    }
                    let buffer = run_graph.graph_output_buffers();
                    for (channel_i, out) in frame.iter_mut().enumerate() {
                        let value: T =
                            cpal::Sample::from::<f32>(&buffer.read(channel_i, sample_counter));
                        *out = value;
                    }
                    sample_counter += 1;
                }
            },
            err_fn,
        )?;

        stream.play()?;
        Ok(stream)
    }
}
