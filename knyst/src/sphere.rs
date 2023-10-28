use std::time::Duration;

use knyst_core::{resources::ResourcesSettings, Resources};

use crate::{
    controller::print_error_handler,
    graph::{Graph, GraphSettings, RunGraphSettings},
    modal_interface::{register_sphere, set_active_sphere, SphereId},
    prelude::{AudioBackend, MultiThreadedKnystCommands},
};

/// One instance of Knyst, responsible for its overall context.
pub struct KnystSphere {
    name: String,
    knyst_commands: MultiThreadedKnystCommands,
}

impl KnystSphere {
    pub fn start<B: AudioBackend>(
        mut backend: &mut B,
        settings: SphereSettings,
    ) -> Result<SphereId, Box<dyn std::error::Error>> {
        let resources = Resources::new(settings.resources_settings);
        let graph_settings = GraphSettings {
            name: settings.name.clone(),
            num_inputs: backend
                .native_input_channels()
                .unwrap_or(settings.num_inputs),
            num_outputs: backend
                .native_output_channels()
                .unwrap_or(settings.num_outputs),
            block_size: backend.block_size().unwrap_or(64),
            sample_rate: backend.sample_rate() as f32,
            ..Default::default()
        };
        let graph: Graph = Graph::new(graph_settings);
        let mut k = backend
            .start_processing(
                graph,
                resources,
                RunGraphSettings {
                    scheduling_latency: settings.scheduling_latency,
                },
                Box::new(print_error_handler),
            )
            .unwrap();
        let s = Self {
            name: settings.name,
            knyst_commands: k,
        };
        // Add the sphere to the global list of spheres
        let sphere_id = register_sphere(s)?;
        // Set the sphere to be active
        set_active_sphere(sphere_id)?;
        Ok(sphere_id)
    }
    #[must_use]
    pub fn commands(&self) -> MultiThreadedKnystCommands {
        self.knyst_commands.clone()
    }
}

pub struct SphereSettings {
    name: String,
    resources_settings: ResourcesSettings,
    /// num inputs if determinable by the user (e.g. in the JACK backend)
    num_inputs: usize,
    /// num outputs if determinable by the user (e.g. in the JACK backend)
    num_outputs: usize,
    scheduling_latency: Duration,
}

impl Default for SphereSettings {
    fn default() -> Self {
        Self {
            name: Default::default(),
            resources_settings: Default::default(),
            scheduling_latency: Duration::from_millis(100),
            num_inputs: 2,
            num_outputs: 2,
        }
    }
}
