//! For running and inspecting the output of Knyst offline i.e. generating buffers of samples without automatically outputing them anywhere e.g. to a sound card.
//!
//! Useful for tests and non-realtime processing.
use crate::{
    audio_backend::AudioBackend,
    controller::{print_error_handler, Controller},
    graph::RunGraph,
    modal_interface::{remove_sphere, set_active_sphere, SphereId},
    prelude::{KnystSphere, SphereSettings},
    Sample,
};

/// For running and inspecting the output of Knyst offline i.e. generating buffers of samples without outputing them anywhere. Removes the associated KnystSphere when dropped.
pub struct KnystOffline {
    test_backend: OfflineBackend,
    controller: Controller,
    sphere_id: SphereId,
}
impl KnystOffline {
    /// Creates an offline Knyst sphere and activates it. You can then use the modal knyst API as
    /// usual, buf you have to manually process blocks of audio samples and store them if you want.
    pub fn new(
        sample_rate: usize,
        block_size: usize,
        num_inputs: usize,
        num_outputs: usize,
    ) -> KnystOffline {
        let mut backend = OfflineBackend {
            sample_rate,
            block_size,
            num_inputs,
            num_outputs,
            run_graph: None,
        };
        let (sphere_id, controller) = KnystSphere::start_return_controller(
            &mut backend,
            SphereSettings {
                num_inputs: 0,
                num_outputs: 2,
                ..Default::default()
            },
            print_error_handler,
        )
        .unwrap();
        set_active_sphere(sphere_id).unwrap();
        KnystOffline {
            test_backend: backend,
            controller,
            sphere_id,
        }
    }
    /// Returns the [`SphereId`] of the offline sphere
    pub fn sphere_id(&self) -> SphereId {
        self.sphere_id
    }
    /// Process one block of audio, incl controller
    pub fn process_block(&mut self) {
        self.controller.run(10000);
        if let Some(run_graph) = &mut self.test_backend.run_graph {
            run_graph.run_resources_communication(10000);
            run_graph.process_block();
        }
    }
    /// Fill one graph input channel for this block
    pub fn set_input(&mut self, index: usize, input: &[Sample]) {
        if let Some(run_graph) = &mut self.test_backend.run_graph {
            let input_buffers = run_graph.graph_input_buffers();
            assert!(index < self.test_backend.num_inputs);
            assert_eq!(self.test_backend.block_size, input.len());
            let chan = unsafe { input_buffers.get_channel_mut(index) };
            chan.copy_from_slice(input);
        }
    }
    /// Assert the output of one output channel
    pub fn assert_eq_output_channel(&self, channel: usize, expected_output: &[Sample]) {
        if let Some(run_graph) = &self.test_backend.run_graph {
            let output_buffers = run_graph.graph_output_buffers();
            assert!(channel < self.test_backend.num_outputs);
            assert_eq!(self.test_backend.block_size, expected_output.len());
            let chan = output_buffers.get_channel(channel);
            for (a, b) in expected_output.iter().zip(chan.iter()) {
                assert_eq!(*a, *b);
            }
        }
    }
    /// Returns a channel of audio if that channel exists
    pub fn output_channel(&self, channel: usize) -> Option<&[Sample]> {
        if let Some(run_graph) = &self.test_backend.run_graph {
            let output_buffers = run_graph.graph_output_buffers();
            if channel < self.test_backend.num_outputs {
                let chan = output_buffers.get_channel(channel);
                Some(chan)
            } else {
                None
            }
        } else {
            None
        }
    }
}
impl Drop for KnystOffline {
    fn drop(&mut self) {
        remove_sphere(self.sphere_id).unwrap();
    }
}

struct OfflineBackend {
    sample_rate: usize,
    block_size: usize,
    num_outputs: usize,
    num_inputs: usize,
    run_graph: Option<RunGraph>,
}
impl AudioBackend for OfflineBackend {
    fn start_processing_return_controller(
        &mut self,
        mut graph: crate::graph::Graph,
        resources: crate::Resources,
        run_graph_settings: crate::graph::RunGraphSettings,
        error_handler: Box<dyn FnMut(crate::KnystError) + Send + 'static>,
    ) -> Result<crate::controller::Controller, crate::audio_backend::AudioBackendError> {
        let (run_graph, resources_command_sender, resources_command_receiver) =
            RunGraph::new(&mut graph, resources, run_graph_settings)?;
        let controller = Controller::new(
            graph,
            error_handler,
            resources_command_sender,
            resources_command_receiver,
        );
        self.run_graph = Some(run_graph);
        Ok(controller)
    }

    fn stop(&mut self) -> Result<(), crate::audio_backend::AudioBackendError> {
        todo!()
    }

    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn block_size(&self) -> Option<usize> {
        Some(self.block_size)
    }

    fn native_output_channels(&self) -> Option<usize> {
        Some(self.num_outputs)
    }

    fn native_input_channels(&self) -> Option<usize> {
        Some(self.num_inputs)
    }
}
