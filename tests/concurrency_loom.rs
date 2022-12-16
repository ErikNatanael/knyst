use knyst::graph::Gen;
use knyst::graph::RunGraph;
use knyst::prelude::*;
#[cfg(loom)]
use loom::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
};
#[cfg(not(loom))]
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
};

// Outputs its input value + 1
struct OneGen {}
impl Gen for OneGen {
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        for i in 0..ctx.block_size() {
            ctx.outputs.write(ctx.inputs.read(0, i) + 1.0, 0, i);
        }
        GenState::Continue
    }
    fn input_desc(&self, input: usize) -> &'static str {
        match input {
            0 => "passthrough",
            _ => "",
        }
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}
struct DummyGen {
    counter: f32,
}
impl Gen for DummyGen {
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        for i in 0..ctx.block_size() {
            self.counter += 1.;
            ctx.outputs
                .write(ctx.inputs.read(0, i) + self.counter, 0, i);
        }
        GenState::Continue
    }

    fn input_desc(&self, input: usize) -> &'static str {
        match input {
            0 => "counter",
            _ => "",
        }
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

#[test]
fn parallel_mutation() {
    fn add_remove_nodes() {
        // Here we just want to check that the program doesn't crash/segfault when changing the graph while running the GraphGen.
        const BLOCK: usize = 1;
        let mut graph: Graph = Graph::new(GraphSettings {
            block_size: BLOCK,
            ..Default::default()
        });
        let resources = Resources::new(ResourcesSettings::default());
        let mut run_graph = RunGraph::new(&mut graph, resources).unwrap();
        let mut nodes = vec![];
        let mut last_node = None;
        let done_flag = Arc::new(AtomicBool::new(false));
        let audio_done_flag = done_flag.clone();
        let audio_thread = thread::spawn(move || {
            while !audio_done_flag.load(Ordering::Relaxed) {
                run_graph.process_block();
                #[cfg(loom)]
                loom::thread::yield_now();
            }
        });
        for _ in 0..10 {
            let node = graph.push(OneGen {});
            if let Some(last) = last_node.take() {
                graph.connect(node.to(last)).unwrap();
            } else {
                graph.connect(Connection::graph_output(node)).unwrap();
            }
            last_node = Some(node);
            nodes.push(node);
            graph.commit_changes();
            #[cfg(loom)]
            loom::thread::yield_now();
        }
        for node in nodes.into_iter().rev() {
            graph.free_node(node).unwrap();
            graph.commit_changes();
            #[cfg(loom)]
            loom::thread::yield_now();
        }
        let mut nodes = vec![];
        last_node = None;
        for _ in 0..10 {
            let node = graph.push(DummyGen { counter: 0. });
            if let Some(last) = last_node.take() {
                graph.connect(node.to(last)).unwrap();
            } else {
                graph.connect(Connection::graph_output(node)).unwrap();
            }
            last_node = Some(node);
            nodes.push(node);
            graph.commit_changes();
            #[cfg(loom)]
            loom::thread::yield_now();
        }
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        nodes.shuffle(&mut rng);
        for node in nodes.into_iter() {
            graph.free_node(node).unwrap();
            graph.commit_changes();
            graph.update();
            #[cfg(loom)]
            loom::thread::yield_now();
        }
        drop(graph);
        done_flag.store(true, Ordering::SeqCst);
        audio_thread.join().unwrap();
    }
    #[cfg(loom)]
    loom::model(|| {
        add_remove_nodes();
    });
    #[cfg(not(loom))]
    add_remove_nodes();
}
