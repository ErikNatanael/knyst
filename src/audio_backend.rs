use crate::{
    graph::{Graph, Node},
    Resources, Sample,
};
pub trait AudioBackend {
    fn start_processing(
        &mut self,
        graph: &mut Graph,
        resources: Resources,
    ) -> Result<(), AudioBackendError>;
    fn stop(&mut self) -> Result<(), AudioBackendError>;
    fn sample_rate(&self) -> usize;
    fn block_size(&self) -> usize;
}

#[derive(thiserror::Error, Debug, PartialEq)]
pub enum AudioBackendError {
    #[error("You tried to start a backend that was already running. A backend can only be started once.")]
    BackendAlreadyRunning,
    #[error("You tried to stop a backend that was already stopped.")]
    BackendNotRunning,
    #[error(transparent)]
    JackError(#[from] jack::Error),
}

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
        let node = graph.to_node().unwrap();
        if let Some(JackClient::Passive(client)) = self.client.take() {
            let mut in_ports = vec![];
            let mut out_ports = vec![];
            let num_inputs = node.num_inputs();
            let num_outputs = node.num_outputs();
            for i in 0..num_inputs {
                in_ports.push(client.register_port(&format!("in_{i}"), jack::AudioIn::default())?);
            }
            for i in 0..num_outputs {
                out_ports
                    .push(client.register_port(&format!("out_{i}"), jack::AudioOut::default())?);
            }
            let mut input_buffers = vec![];
            for _ in 0..num_inputs {
                input_buffers.push(vec![0.0; graph.block_size()].into_boxed_slice());
            }
            let input_buffers = input_buffers.into_boxed_slice();
            let jack_process = JackProcess {
                main_node: node,
                input_buffers,
                resources,
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

    fn block_size(&self) -> usize {
        self.block_size
    }
}

struct JackProcess {
    main_node: Node,
    in_ports: Vec<jack::Port<jack::AudioIn>>,
    input_buffers: Box<[Box<[Sample]>]>,
    resources: Resources,
    out_ports: Vec<jack::Port<jack::AudioOut>>,
}

impl jack::ProcessHandler for JackProcess {
    fn process(&mut self, _: &jack::Client, ps: &jack::ProcessScope) -> jack::Control {
        for (in_port, in_buffer) in self.in_ports.iter().zip(self.input_buffers.iter_mut()) {
            let in_port_slice = in_port.as_slice(ps);
            in_buffer.clone_from_slice(in_port_slice);
        }
        self.main_node
            .process(&self.input_buffers, &mut self.resources);

        for (out_port, out_buffer) in self
            .out_ports
            .iter_mut()
            .zip(self.main_node.output_buffers().iter())
        {
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

pub struct CpalBackend;
