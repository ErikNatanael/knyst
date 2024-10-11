use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use knyst::graph::Time;
use knyst::resources::{IdOrKey, ResourcesSettings};
use knyst::Resources;

use super::{Gen, RunGraph};
use crate as knyst;
use crate::controller::KnystCommands;
use crate::gen::{BufferReader, WavetableOscillatorOwned};
use crate::graph::{FreeError, Oversampling, ScheduleError};
use crate::prelude::*;
use crate::time::{Beats, Seconds};
use crate::{controller::Controller, graph::connection::constant};

// Outputs its input value + 1
struct OneGen {}
#[impl_gen]
impl OneGen {
    #[process]
    fn process(&mut self, passthrough: &[Sample], out: &mut [Sample]) -> GenState {
        for (i, o) in passthrough.iter().zip(out.iter_mut()) {
            *o = *i + 1.0;
        }
        GenState::Continue
    }
}
struct DummyGen {
    counter: Sample,
}
#[impl_gen]
impl DummyGen {
    fn new(counter_start: Sample) -> Self {
        DummyGen {
            counter: counter_start,
        }
    }
    #[process]
    fn process(&mut self, counter: &[Sample], output: &mut [Sample]) -> GenState {
        for (count, out) in counter.iter().zip(output.iter_mut()) {
            self.counter += 1.;
            *out = count + self.counter;
        }
        GenState::Continue
    }
}
fn test_resources_settings() -> ResourcesSettings {
    ResourcesSettings::default()
}
fn test_run_graph(graph: &mut Graph, settings: RunGraphSettings) -> RunGraph {
    let resources = Resources::new(test_resources_settings());
    let (run_graph, _, _) = RunGraph::new(graph, resources, settings).unwrap();
    run_graph
}

#[test]
fn create_graph() {
    let mut graph: Graph = Graph::new(GraphSettings::default());
    let node_id = graph.push(DummyGen { counter: 0.0 });
    graph.connect(Connection::graph_output(node_id)).unwrap();
    graph.init();
    let mut run_graph = test_run_graph(&mut graph, RunGraphSettings::default());
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 31), 32.0);
}
#[test]
fn multiple_nodes() {
    let mut graph: Graph = Graph::new(GraphSettings {
        block_size: 4,
        ..Default::default()
    });
    let node1 = DummyGen { counter: 0.0 };
    let node2 = DummyGen { counter: 0.0 };
    let node3 = DummyGen { counter: 0.0 };
    let node_id1 = graph.push(node1);
    let node_id2 = graph.push(node2);
    let node_id3 = graph.push(node3);
    graph.connect(node_id1.to(node_id3)).unwrap();
    graph.connect(node_id2.to(node_id3)).unwrap();
    graph.connect(Connection::graph_output(node_id3)).unwrap();
    let mut run_graph = test_run_graph(&mut graph, RunGraphSettings::default());
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 2), 9.0);
}
#[test]
fn node_mortality() {
    let mut graph: Graph = Graph::new(GraphSettings {
        block_size: 4,
        ..Default::default()
    });
    let node1 = DummyGen { counter: 0.0 };
    let node2 = DummyGen { counter: 0.0 };
    let node3 = DummyGen { counter: 0.0 };
    let node_id1 = graph.push(node1);
    graph.set_node_mortality(node_id1, false).unwrap();
    let node_id2 = graph.push(node2);
    let node_id3 = graph.push(node3);
    graph.connect(node_id2.to(node_id3)).unwrap();
    graph.connect(Connection::graph_output(node_id3)).unwrap();
    graph.free_disconnected_nodes().unwrap();
    assert!(matches!(
        graph.free_node(node_id1),
        Err(FreeError::ImmortalNode)
    ));
    graph.commit_changes();
    graph.connect(node_id1.to(node_id3)).unwrap();
    let mut run_graph = test_run_graph(&mut graph, RunGraphSettings::default());
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 2), 9.0);
}
#[test]
fn constant_inputs() {
    const BLOCK: usize = 16;
    let mut graph: Graph = Graph::new(GraphSettings {
        block_size: BLOCK,
        ..Default::default()
    });
    let node = DummyGen { counter: 0.0 };
    let node_id = graph.push(node);
    graph.connect(constant(0.5).to(node_id)).unwrap();
    graph.connect(Connection::graph_output(node_id)).unwrap();
    let mut run_graph = test_run_graph(&mut graph, RunGraphSettings::default());
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, BLOCK - 1), 16.5);
}
#[test]
fn graph_in_a_graph() {
    const BLOCK: usize = 16;
    const CONSTANT_INPUT_TO_NODE: Sample = 0.25;
    let mut inner_graph: Graph = Graph::new(GraphSettings {
        block_size: BLOCK,
        ..Default::default()
    });
    let node = DummyGen { counter: 0.0 };
    let node_id = inner_graph.push(node);
    inner_graph
        .connect(Connection::graph_output(node_id))
        .unwrap();
    let mut top_level_graph: Graph = Graph::new(GraphSettings {
        block_size: BLOCK,
        ..Default::default()
    });
    let inner_graph_node_id = top_level_graph.push(inner_graph);
    top_level_graph
        .connect(Connection::graph_output(inner_graph_node_id).channels(2))
        .unwrap();
    // Parent graphs should propagate a connection to its child graphs
    // without issue, but it will be scheduled instead of applied directly
    top_level_graph
        .connect(constant(CONSTANT_INPUT_TO_NODE).to(node_id))
        .unwrap();
    let mut run_graph = test_run_graph(&mut top_level_graph, RunGraphSettings::default());
    run_graph.process_block();
    assert_eq!(
        run_graph.graph_output_buffers().read(0, BLOCK - 2),
        (BLOCK - 1) as Sample + CONSTANT_INPUT_TO_NODE
    );
}
#[test]
fn feedback_in_graph() {
    // v-------------<
    // |   2 -> 3 -> 4
    // |   |    |
    // fbv fbv fbv
    //   0     -> 1 -> out
    const BLOCK: usize = 4;
    let mut graph: Graph = Graph::new(GraphSettings {
        block_size: BLOCK,
        ..Default::default()
    });
    let n0 = graph.push(DummyGen { counter: 0.0 });
    let n1 = graph.push(DummyGen { counter: 0.0 });
    let n2 = graph.push(DummyGen { counter: 4.0 });
    let n3 = graph.push(DummyGen { counter: 5.0 });
    let n4 = graph.push(DummyGen { counter: 1.0 });
    graph.connect(Connection::graph_output(n1)).unwrap();
    graph
        .connect(Connection::graph_output(n0).to_index(1))
        .unwrap();
    graph.connect(n0.to(n1)).unwrap();
    graph.connect(n2.to(n3)).unwrap();
    graph.connect(n3.to(n4)).unwrap();
    graph.connect(n2.to(n0).feedback(true)).unwrap();
    graph.connect(n3.to(n1).feedback(true)).unwrap();
    graph.connect(n4.to(n0).feedback(true)).unwrap();
    let mut run_graph = test_run_graph(&mut graph, RunGraphSettings::default());
    run_graph.process_block();
    let block1 = vec![2.0 as Sample, 4., 6., 8.];
    let block2 = vec![39 as Sample, 47., 55., 63.];
    for (&output, expected) in run_graph
        .graph_output_buffers()
        .get_channel(0)
        .iter()
        .zip(block1)
    {
        assert_eq!(
            output,
            expected,
            "block1 failed with o: {}, expected: {}, n1: {:#?}, n0: {:#?}",
            output,
            expected,
            run_graph.graph_output_buffers().get_channel(0),
            run_graph.graph_output_buffers().get_channel(1)
        );
    }
    run_graph.process_block();
    for (&output, expected) in run_graph
        .graph_output_buffers()
        .get_channel(0)
        .iter()
        .zip(block2)
    {
        assert_eq!(
            output,
            expected,
            "block2 failed with o: {}, expected: {}, n1: {:#?}, n0: {:#?}",
            output,
            expected,
            run_graph.graph_output_buffers().get_channel(0),
            run_graph.graph_output_buffers().get_channel(1)
        );
    }
}

#[test]
fn changing_the_graph() {
    const BLOCK: usize = 1;
    let mut graph: Graph = Graph::new(GraphSettings {
        block_size: BLOCK,
        ..Default::default()
    });
    let mut run_graph = test_run_graph(&mut graph, RunGraphSettings::default());
    let triangular_sequence = |n| (n * (n + 1)) / 2;
    for i in 0..10 {
        let node = DummyGen { counter: 0.0 };
        let node_id = graph.push(node);
        graph.connect(constant(0.5).to(node_id)).unwrap();
        graph.connect(Connection::graph_output(node_id)).unwrap();
        graph.update();
        run_graph.process_block();
        assert_eq!(
            run_graph.graph_output_buffers().read(0, 0),
            (i + 1) as Sample * 0.5 + triangular_sequence(i + 1) as Sample,
            "i: {}, output: {}, expected: {}",
            i,
            run_graph.graph_output_buffers().read(0, 0),
            (i + 1) as Sample * 0.5 + triangular_sequence(i + 1) as Sample,
        );
    }
}

#[test]
fn graph_inputs() {
    const BLOCK: usize = 1;
    let mut graph: Graph = Graph::new(GraphSettings {
        block_size: BLOCK,
        num_inputs: 3,
        ..Default::default()
    });
    let mut run_graph = test_run_graph(&mut graph, RunGraphSettings::default());
    let node = DummyGen { counter: 0.0 };
    let node_id = graph.push(node);
    graph
        .connect(
            Connection::graph_input(node_id)
                .from_index(2)
                .to_label("counter"),
        )
        .unwrap();
    graph.connect(Connection::graph_output(node_id)).unwrap();
    graph.update();
    run_graph.graph_input_buffers().fill_channel(10.0, 2);
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 11.0);
}

#[test]
fn remove_nodes() {
    const BLOCK: usize = 1;
    let mut graph: Graph = Graph::new(GraphSettings {
        block_size: BLOCK,
        ..Default::default()
    });
    let mut run_graph = test_run_graph(&mut graph, RunGraphSettings::default());
    let mut nodes = vec![];
    let mut last_node = None;
    for _ in 0..10 {
        let node = graph.push(OneGen {});
        if let Some(last) = last_node.take() {
            graph.connect(node.to(last)).unwrap();
        } else {
            graph.connect(Connection::graph_output(node)).unwrap();
        }
        last_node = Some(node.clone());
        nodes.push(node);
    }
    graph.update();
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 10.0);
    // Convert NodeAddress to RawNodeAddress for removal, which we know will work because we pushed it straight to the graph
    graph.free_node(nodes[9]).unwrap();
    graph.update();
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 9.0);
    graph.free_node(nodes[4]).unwrap();
    graph.update();
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 4.0);
    graph.connect(nodes[5].to(nodes[3])).unwrap();
    graph.update();
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 8.0);
    for node in &nodes[1..4] {
        graph.free_node(*node).unwrap();
    }
    graph.update();
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 1.0);
    assert_eq!(graph.free_node(nodes[4]), Err(FreeError::NodeNotFound));
}

#[test]
fn parallel_mutation() {
    // Here we just want to check that the program doesn't crash/segfault when changing the graph while running the GraphGen.
    const BLOCK: usize = 1;
    let mut graph: Graph = Graph::new(GraphSettings {
        block_size: BLOCK,
        ..Default::default()
    });
    let mut nodes = vec![];
    let mut run_graph = test_run_graph(&mut graph, RunGraphSettings::default());
    let mut last_node = None;
    let done_flag = Arc::new(AtomicBool::new(false));
    let audio_done_flag = done_flag.clone();
    let audio_thread = std::thread::spawn(move || {
        while !audio_done_flag.load(Ordering::SeqCst) {
            run_graph.process_block();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });
    for _ in 0..10 {
        let node = graph.push(OneGen {});
        if let Some(last) = last_node.take() {
            graph.connect(node.to(last)).unwrap();
        } else {
            graph.connect(Connection::graph_output(node)).unwrap();
        }
        last_node = Some(node.clone());
        nodes.push(node);
        graph.update();
        std::thread::sleep(std::time::Duration::from_millis(3));
    }
    for node in nodes.into_iter().rev() {
        graph.free_node(node).unwrap();
        graph.commit_changes();
        std::thread::sleep(std::time::Duration::from_millis(2));
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
        last_node = Some(node.clone());
        nodes.push(node);
        graph.update();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    nodes.shuffle(&mut rng);
    for node in nodes.into_iter() {
        graph.free_node(node).unwrap();
        graph.update();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    drop(graph);
    done_flag.store(true, Ordering::SeqCst);
    audio_thread.join().unwrap();
}
struct SelfFreeing {
    samples_countdown: usize,
    value: Sample,
    mend: bool,
}
impl Gen for SelfFreeing {
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        for i in 0..ctx.block_size() {
            if self.samples_countdown == 0 {
                if self.mend {
                    ctx.outputs.write(ctx.inputs.read(0, i), 0, i);
                } else {
                    ctx.outputs.write(0., 0, i);
                }
            } else {
                ctx.outputs.write(ctx.inputs.read(0, i) + self.value, 0, i);
                self.samples_countdown -= 1;
            }
        }
        if self.samples_countdown == 0 {
            if self.mend {
                GenState::FreeSelfMendConnections
            } else {
                GenState::FreeSelf
            }
        } else {
            GenState::Continue
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
fn self_freeing_nodes() {
    const BLOCK: usize = 1;
    let mut graph: Graph = Graph::new(GraphSettings {
        block_size: BLOCK,
        ..Default::default()
    });
    let mut run_graph = test_run_graph(&mut graph, RunGraphSettings::default());
    let n0 = graph.push(SelfFreeing {
        samples_countdown: 2,
        value: 1.,
        mend: true,
    });
    let n1 = graph.push(SelfFreeing {
        samples_countdown: 1,
        value: 1.,
        mend: true,
    });
    let n2 = graph.push(SelfFreeing {
        samples_countdown: 4,
        value: 2.,
        mend: false,
    });
    let n3 = graph.push(SelfFreeing {
        samples_countdown: 5,
        value: 3.,
        mend: false,
    });
    graph.connect(Connection::graph_output(n0)).unwrap();
    graph.connect(n1.to(n0)).unwrap();
    graph.connect(n2.to(n1)).unwrap();
    graph.connect(n3.to(n2)).unwrap();

    graph.update();
    assert_eq!(graph.num_nodes(), 4);
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 7.0);
    graph.update();
    // Still 4 since the node has been added to the free node queue, but there
    // hasn't been a new generation in the GraphGen yet.
    assert_eq!(graph.num_nodes(), 4);
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 6.0);
    graph.update();
    assert_eq!(graph.num_nodes(), 3);
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 5.0);
    graph.update();
    assert_eq!(graph.num_nodes(), 2);
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 5.0);
    graph.update();
    assert_eq!(graph.num_nodes(), 2);
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 0.0);
    graph.update();
    assert_eq!(graph.num_nodes(), 1);
    run_graph.process_block();
    graph.update();
    assert_eq!(graph.num_nodes(), 0);
}
#[test]
fn scheduling() {
    const BLOCK: usize = 4;
    const SR: u64 = 44100;
    let mut graph: Graph = Graph::new(GraphSettings {
        block_size: BLOCK,
        sample_rate: SR as Sample,
        ..Default::default()
    });
    // let mut graph_node = graph_node(&mut graph);
    let mut run_graph = test_run_graph(
        &mut graph,
        RunGraphSettings {
            scheduling_latency: Duration::from_millis(0),
        },
    );
    let node0 = graph.push(OneGen {});
    let node1 = graph.push(OneGen {});
    graph.connect(Connection::graph_output(node0)).unwrap();
    graph.connect(node1.to(node0)).unwrap();
    // This gets applied only one sample late. Why?
    graph.connect(constant(2.).to(node1)).unwrap();
    graph.update();
    graph
        .schedule_change(ParameterChange::seconds(
            node0.input(0),
            1.0,
            Seconds::from_samples(3, SR),
        ))
        .unwrap();
    graph
        .schedule_change(ParameterChange::seconds(
            node0.input("passthrough"),
            2.0,
            Seconds::from_samples(2, SR),
        ))
        .unwrap();
    graph
        .schedule_change(ParameterChange::seconds(
            node1.input(0),
            10.0,
            Seconds::from_samples(7, SR),
        ))
        .unwrap();
    // Schedule far into the future, this should not show up in the test output
    graph
        .schedule_change(ParameterChange::duration_from_now(
            node1.input(0),
            3000.0,
            Duration::from_secs(100),
        ))
        .unwrap();
    graph.update();
    run_graph.process_block();
    dbg!(run_graph.graph_output_buffers().get_channel(0));
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 4.0);
    assert_eq!(run_graph.graph_output_buffers().read(0, 1), 4.0);
    assert_eq!(run_graph.graph_output_buffers().read(0, 2), 6.0);
    assert_eq!(run_graph.graph_output_buffers().read(0, 3), 5.0);
    graph
        .schedule_change(ParameterChange::seconds(
            node0.input(0),
            0.0,
            Seconds::from_samples(5, SR),
        ))
        .unwrap();
    graph
        .schedule_change(ParameterChange::seconds(
            node1.input(0),
            0.0,
            Seconds::from_samples(6, SR),
        ))
        .unwrap();
    assert_eq!(
        graph.schedule_change(ParameterChange::seconds(
            node1.input("pasta"),
            100.0,
            Seconds::from_samples(6, SR)
        )),
        Err(ScheduleError::InputLabelNotFound("pasta"))
    );
    graph.update();
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 5.0);
    assert_eq!(run_graph.graph_output_buffers().read(0, 1), 4.0);
    assert_eq!(run_graph.graph_output_buffers().read(0, 2), 2.0);
    assert_eq!(run_graph.graph_output_buffers().read(0, 3), 12.0);
    // Schedule "now", but this will be a few samples into the
    // future depending on the time it takes to run this test.
    graph
        .schedule_change(ParameterChange::now(node1.input(0), 1000.0))
        .unwrap();
    graph.update();
    // Move a few hundred samples into the future
    for _ in 0..600 {
        run_graph.process_block();
        graph.update();
    }
    // If this fails, it may be because the machine is too slow. Try
    // increasing the number of iterations above.
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 1002.0);
}
#[test]
fn index_routing() {
    let graph_settings = GraphSettings {
        block_size: 4,
        sample_rate: 44100.,
        num_outputs: 2,
        ..Default::default()
    };
    let mut graph = Graph::new(graph_settings);
    let mut run_graph = test_run_graph(&mut graph, RunGraphSettings::default());
    let mult = graph.push(Mult);
    let five = graph.push(
        gen(move |ctx, _resources| {
            let out_chan = ctx.outputs.iter_mut().next().unwrap();
            for o0 in out_chan {
                *o0 = 5.;
            }
            GenState::Continue
        })
        .output("o"),
    );
    let nine = graph.push(
        gen(move |ctx, _resources| {
            let out_chan = ctx.outputs.iter_mut().next().unwrap();
            for o0 in out_chan {
                *o0 = 9.;
            }
            GenState::Continue
        })
        .output("o"),
    );
    // Connecting the node to the graph output
    graph.connect(mult.to_graph_out()).unwrap();
    // Multiply 5 by 9
    graph.connect(five.to(mult).to_index(0)).unwrap();
    graph.connect(nine.to(mult).to_index(1)).unwrap();
    // You need to commit changes and update if the graph is running.
    graph.commit_changes();
    graph.update();
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 45.0);
}
#[test]
fn index_routing_advanced() {
    let graph_settings = GraphSettings {
        block_size: 4,
        sample_rate: 44100.,
        num_outputs: 2,
        ..Default::default()
    };
    let mut graph = Graph::new(graph_settings);
    let mut run_graph = test_run_graph(&mut graph, RunGraphSettings::default());
    let multiplier = graph.push(
        gen(move |ctx, _resources| {
            for i in 0..ctx.block_size() {
                let i0 = ctx.inputs.read(0, i) * 1.;
                let i1 = ctx.inputs.read(1, i) * 2.;
                let i2 = ctx.inputs.read(2, i) * 3.;
                let i3 = ctx.inputs.read(3, i) * 4.;
                let i4 = ctx.inputs.read(4, i) * 5.;
                let i5 = ctx.inputs.read(5, i) * 6.;
                let mut value = i0;
                for i in [i1, i2, i3, i4, i5] {
                    if i != 0.0 {
                        value *= i;
                    }
                }
                ctx.outputs.write(value, 0, i);
            }
            GenState::Continue
        })
        .output("o")
        .input("i0")
        .input("i1")
        .input("i2")
        .input("i3")
        .input("i4")
        .input("i5"),
    );
    let five = graph.push(
        gen(move |ctx, _resources| {
            for o0 in ctx.outputs.iter_mut().next().unwrap() {
                *o0 = 5.;
            }
            GenState::Continue
        })
        .output("o"),
    );
    let nine = graph.push(
        gen(move |ctx, _resources| {
            for o0 in ctx.outputs.iter_mut().next().unwrap() {
                *o0 = 9.;
            }
            GenState::Continue
        })
        .output("o"),
    );
    let numberer = graph.push(
        gen(move |ctx, _resources| {
            for i in 0..ctx.block_size() {
                ctx.outputs.write(0.0, 0, i);
                ctx.outputs.write(1.0, 1, i);
                ctx.outputs.write(2.0, 2, i);
                ctx.outputs.write(3.0, 3, i);
                ctx.outputs.write(4.0, 4, i);
                ctx.outputs.write(5.0, 5, i);
                ctx.outputs.write(6.0, 6, i);
                ctx.outputs.write(7.0, 7, i);
                ctx.outputs.write(8.0, 8, i);
                ctx.outputs.write(9.0, 9, i);
            }
            GenState::Continue
        })
        .output("o0")
        .output("o1")
        .output("o2")
        .output("o3")
        .output("o4")
        .output("o5")
        .output("o6")
        .output("o7")
        .output("o8")
        .output("o9"),
    );
    // Connecting the node to the graph output
    graph.connect(multiplier.to_graph_out()).unwrap();
    // Multiply 5 by 9
    graph.connect(nine.to(multiplier).to_index(0)).unwrap();
    graph.connect(five.to(multiplier).to_index(1)).unwrap();
    // You need to commit changes and update if the graph is running.
    graph.update();
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 90.0);

    graph.disconnect(five.to(multiplier).to_index(1)).unwrap();
    graph.connect(five.to(multiplier).to_index(3)).unwrap();
    graph.update();
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 180.0);

    graph.connect(five.to(multiplier).to_index(4)).unwrap();
    graph.connect(five.to(multiplier).to_label("i5")).unwrap();
    graph.update();
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 135000.0);

    graph.connect(Connection::clear_to_nodes(five)).unwrap();
    graph.update();
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 9.0);

    graph
        .connect(numberer.to(multiplier).from_index(2).to_index(4))
        .unwrap();
    graph
        .connect(numberer.to(multiplier).from_index(7).to_index(2))
        .unwrap();
    graph.update();
    run_graph.process_block();
    assert_eq!(run_graph.graph_output_buffers().read(0, 0), 1890.0);
}
#[test]
fn sending_buffer_to_resources() {
    const BLOCK_SIZE: usize = 16;
    let graph_settings = GraphSettings {
        block_size: BLOCK_SIZE,
        sample_rate: 44100.,
        num_outputs: 2,
        ..Default::default()
    };
    let mut graph = Graph::new(graph_settings);
    let resources = Resources::new(ResourcesSettings {
        max_wavetables: 0,
        max_buffers: 3,
        max_user_data: 0,
    });
    let (mut run_graph, resources_command_sender, resources_response_receiver) = RunGraph::new(
        &mut graph,
        resources,
        RunGraphSettings {
            scheduling_latency: Duration::from_secs(0),
        },
    )
    .unwrap();

    let mut controller = Controller::new(
        graph,
        |e| println!("{e}"),
        resources_command_sender,
        resources_response_receiver,
    );
    let mut k = controller.get_knyst_commands();

    // Send a buffer to the RunGraph
    let buffer_vec = vec![10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.];
    let buffer = Buffer::from_vec(buffer_vec.clone(), 44100.);
    let buffer_id = k.insert_buffer(buffer);
    // Push a buffer reader that can read the Buffer
    let buffer_reader = BufferReader::new(IdOrKey::Id(buffer_id), 1.0, true, StopAction::FreeSelf);
    let br = k.push(buffer_reader, inputs!());
    k.connect(br.to_graph_out());
    // Process the Controller and RunGraph in order
    controller.run(300);
    run_graph.run_resources_communication(50);
    run_graph.process_block();
    // We should be able to read the buffer
    let mut cycling_vec = buffer_vec.iter().cycle();
    for i in 0..BLOCK_SIZE {
        assert_eq!(
            run_graph.graph_output_buffers().read(0, i),
            *cycling_vec.next().unwrap()
        );
    }
}
#[test]
fn start_nodes_with_sample_precision() {
    const SR: u64 = 44100;
    const BLOCK_SIZE: usize = 8;
    let graph_settings = GraphSettings {
        block_size: BLOCK_SIZE,
        sample_rate: SR as Sample,
        num_outputs: 2,
        ..Default::default()
    };
    let mut graph = Graph::new(graph_settings);
    let mut counter0 = 0.;
    // push before starting the graph
    let n0 = graph.push_at_time(
        gen(move |ctx, _resources| {
            let out_chan = ctx.outputs.iter_mut().next().unwrap();
            for sample in out_chan {
                *sample = counter0;
                counter0 += 1.0;
            }
            GenState::Continue
        })
        .output("out"),
        Time::Seconds(Seconds::from_samples(1, SR)),
    );
    graph.connect(n0.to_graph_out()).unwrap();
    let mut counter1 = 0.;
    let n1 = graph.push_at_time(
        gen(move |ctx, _resources| {
            for i in 0..ctx.outputs.block_size() {
                ctx.outputs.write(counter1 * -1. - 1., 0, i);
                counter1 += 1.0;
            }
            GenState::Continue
        })
        .output("out"),
        Time::Seconds(Seconds::from_samples(2, SR)),
    );
    graph.connect(n1.to_graph_out()).unwrap();
    let mut run_graph = test_run_graph(&mut graph, RunGraphSettings::default());
    run_graph.process_block();

    for i in 0..BLOCK_SIZE {
        assert_eq!(run_graph.graph_output_buffers().get_channel(0)[i], 0.)
    }
    // push after starting the graph
    let mut counter2 = 0.;
    let n2 = graph.push_at_time(
        gen(move |ctx, _resources| {
            for i in 0..ctx.outputs.block_size() {
                ctx.outputs.write(counter2, 0, i);
                counter2 += 1.0;
            }
            GenState::Continue
        })
        .output("out"),
        Time::Seconds(Seconds::from_samples(3 + BLOCK_SIZE as u64, SR)),
    );
    let mut counter3 = 0.;
    let n3 = graph.push_at_time(
        gen(move |ctx, _resources| {
            for i in 0..ctx.outputs.block_size() {
                ctx.outputs.write(counter3 * -1. - 1., 0, i);
                counter3 += 1.0;
            }
            GenState::Continue
        })
        .output("out"),
        Time::Seconds(Seconds::from_samples(4 + BLOCK_SIZE as u64, SR)),
    );
    graph.connect(n2.to_graph_out()).unwrap();
    graph.connect(n3.to_graph_out()).unwrap();
    graph.update();
    for _ in 0..2 {
        run_graph.process_block();
        for i in 0..BLOCK_SIZE {
            assert_eq!(run_graph.graph_output_buffers().get_channel(0)[i], 0.)
        }
    }
}

#[test]
fn beat_scheduling() {
    const SR: u64 = 16;
    const BLOCK_SIZE: usize = SR as usize;
    let graph_settings = GraphSettings {
        block_size: BLOCK_SIZE,
        sample_rate: SR as Sample,
        num_outputs: 2,
        ..Default::default()
    };
    let mut graph = Graph::new(graph_settings);
    let node = graph.push(OneGen {});
    graph.connect(node.to_graph_out()).unwrap();
    let (mut run_graph, _, _) = RunGraph::new(
        &mut graph,
        Resources::new(ResourcesSettings::default()),
        RunGraphSettings {
            scheduling_latency: Duration::new(0, 0),
        },
    )
    .unwrap();
    graph
        .change_musical_time_map(|mtm| {
            mtm.insert(
                crate::scheduling::TempoChange::NewTempo { bpm: 120.0 },
                Beats::from_beats(1),
            );
        })
        .unwrap();
    graph
        .schedule_change(ParameterChange::beats(
            node.input(0),
            1.0,
            Beats::from_fractional_beats::<4>(0, 1),
        ))
        .unwrap();
    graph
        .schedule_change(ParameterChange::beats(
            node.input(0),
            2.0,
            Beats::from_fractional_beats::<4>(0, 2),
        ))
        .unwrap();
    graph
        .schedule_change(ParameterChange::beats(
            node.input(0),
            3.0,
            Beats::from_fractional_beats::<4>(1, 0),
        ))
        .unwrap();
    graph
        .schedule_change(ParameterChange::beats(
            node.input(0),
            4.0,
            Beats::from_fractional_beats::<2>(1, 1),
        ))
        .unwrap();
    graph.update();
    run_graph.process_block();
    let out = run_graph.graph_output_buffers().get_channel(0);
    assert_eq!(out[0], 1.0);
    assert_eq!(out[(SR / 4) as usize], 2.0);
    assert_eq!(out[(SR / 2) as usize], 3.0);
    graph.update();
    run_graph.process_block();
    let out = run_graph.graph_output_buffers().get_channel(0);
    assert_eq!(out[0], 4.0);
    // Tempo is twice as fast so half as many samples to get to the half beat mark.
    assert_eq!(out[SR as usize / 4], 5.0);
}

#[test]
fn inner_graph_different_block_size() {
    // An inner graph should get to have any valid block size and be converted
    // to the block size of the outer graph. An inner graph with a larger block
    // size cannot have inputs though.
    const SR: u64 = 44100;
    const BLOCK_SIZE: usize = 2_usize.pow(5);
    let graph_settings = GraphSettings {
        block_size: BLOCK_SIZE,
        sample_rate: SR as Sample,
        num_outputs: 10,
        ..Default::default()
    };
    let freq = 442.0;
    let mut graph = Graph::new(graph_settings);
    let node = graph.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    graph
        .connect(constant(freq).to(node).to_label("freq"))
        .unwrap();
    graph.connect(node.to_graph_out().to_index(0)).unwrap();
    for i in 1..10 {
        let graph_settings = GraphSettings {
            block_size: 2_usize.pow(i),
            sample_rate: SR as Sample,
            num_outputs: 1,
            ..Default::default()
        };
        let mut inner_graph = Graph::new(graph_settings);
        let node = inner_graph.push(WavetableOscillatorOwned::new(Wavetable::sine()));
        inner_graph
            .connect(node.to_graph_out().to_index(0))
            .unwrap();
        inner_graph
            .connect(constant(freq).to(node).to_label("freq"))
            .unwrap();
        let inner_graph = graph.push(inner_graph);
        graph
            .connect(inner_graph.to_graph_out().to_index(i as usize))
            .unwrap();
    }

    graph.update();
    let (mut run_graph, _, _) = RunGraph::new(
        &mut graph,
        Resources::new(ResourcesSettings::default()),
        RunGraphSettings {
            scheduling_latency: Duration::new(0, 0),
        },
    )
    .unwrap();
    for _block_num in 0..10 {
        run_graph.process_block();
        for i in 1..10 {
            assert_eq!(
                std::cmp::Ordering::Equal,
                compare(
                    run_graph.graph_output_buffers().get_channel(0),
                    run_graph.graph_output_buffers().get_channel(i),
                ),
            );
        }
    }
}
fn compare<T: PartialOrd>(a: &[T], b: &[T]) -> std::cmp::Ordering {
    for (v, w) in a.iter().zip(b.iter()) {
        match v.partial_cmp(w) {
            Some(std::cmp::Ordering::Equal) => continue,
            ord => return ord.unwrap(),
        }
    }
    return a.len().cmp(&b.len());
}

#[test]
fn inner_graph_different_oversampling() {
    // An inner graph should get to have any valid block size and be converted
    // to the block size of the outer graph. An inner graph with a larger block
    // size cannot have inputs though.
    const SR: u64 = 44100;
    const BLOCK_SIZE: usize = 16;
    let graph_settings = GraphSettings {
        block_size: BLOCK_SIZE,
        sample_rate: SR as Sample,
        oversampling: Oversampling::X1,
        num_outputs: 10,
        ..Default::default()
    };
    let freq = 442.0;
    let mut graph = Graph::new(graph_settings);
    let node = graph.push(WavetableOscillatorOwned::new(Wavetable::sine()));
    graph
        .connect(constant(freq).to(node).to_label("freq"))
        .unwrap();
    graph.connect(node.to_graph_out().to_index(0)).unwrap();
    for i in 1..=1 {
        let graph_settings = GraphSettings {
            block_size: BLOCK_SIZE,
            oversampling: Oversampling::from_usize(2_usize.pow(i)).unwrap(),
            sample_rate: SR as Sample,
            num_outputs: 1,
            ..Default::default()
        };
        let mut inner_graph = Graph::new(graph_settings);
        let node = inner_graph.push(WavetableOscillatorOwned::new(Wavetable::sine()));
        let amp = inner_graph.push(Mult);

        inner_graph.connect(node.to(amp).to_index(0)).unwrap();
        inner_graph
            .connect(constant(1.0).to(amp).to_index(1))
            .unwrap();
        inner_graph.connect(amp.to_graph_out().to_index(0)).unwrap();
        inner_graph
            .connect(constant(freq).to(node).to_label("freq"))
            .unwrap();
        let inner_graph = graph.push(inner_graph);
        graph
            .connect(inner_graph.to_graph_out().to_index(i as usize))
            .unwrap();
    }

    graph.update();
    let (mut run_graph, _, _) = RunGraph::new(
        &mut graph,
        Resources::new(ResourcesSettings::default()),
        RunGraphSettings {
            scheduling_latency: Duration::new(0, 0),
        },
    )
    .unwrap();
    for _block_num in 0..5 {
        run_graph.process_block();
        for i in 1..=1 {
            // dbg!(run_graph.graph_output_buffers().get_channel(0));
            // dbg!(run_graph.graph_output_buffers().get_channel(i));
            let org = run_graph.graph_output_buffers().get_channel(0);
            let over = run_graph.graph_output_buffers().get_channel(i);
            // The downsampling filter adds a small amount of latency and
            // also changes the amplitude a small amount.
            for frame in 2..run_graph.block_size() {
                assert!((org[frame - 2] - over[frame]).abs() < 0.02);
            }
        }
    }
}
use crate::offline::KnystOffline;

struct DummyGenStereo {
    counter: Sample,
}
#[impl_gen]
impl DummyGenStereo {
    fn new() -> Self {
        DummyGenStereo { counter: 0.0 }
    }
    #[process]
    fn process(
        &mut self,
        counter: &[Sample],
        output_l: &mut [Sample],
        output_r: &mut [Sample],
    ) -> GenState {
        for ((count, out_l), out_r) in counter
            .iter()
            .zip(output_l.iter_mut())
            .zip(output_r.iter_mut())
        {
            self.counter += 1.;
            *out_l = count + self.counter;
            *out_r = count + self.counter + 10.;
        }
        GenState::Continue
    }
}

#[test]
fn stereo_timing() {
    let mut kt = KnystOffline::new(128, 64, 0, 2);
    let dummy = dummy_gen_stereo();
    graph_output(0, dummy.out(0));
    graph_output(1, dummy.out(1));
    kt.process_block();

    let out_left = kt.output_channel(0).unwrap();
    let out_right = kt.output_channel(1).unwrap();
    for (&l, &r) in out_left.into_iter().zip(out_right) {
        assert!(l == r - 10.);
    }
}

#[test]
fn stereo_timing_mult() {
    let mut kt = KnystOffline::new(128, 64, 0, 2);
    let dummy = dummy_gen_stereo();
    let left = dummy.out(0);
    let left_mult = Mult.upload().set(0, left).set(1, 0.5);
    let right = dummy.out(1);
    let right_mult = Mult.upload().set(0, right).set(1, 0.5);

    graph_output(0, left_mult);
    graph_output(1, right_mult);
    kt.process_block();

    let out_left = kt.output_channel(0).unwrap();
    let out_right = kt.output_channel(1).unwrap();
    for (&l, &r) in out_left.into_iter().zip(out_right) {
        assert!(l == r - 5.);
    }
}
#[test]
fn stereo_timing_mult_inner_graph() {
    let sample_rate = 44100;
    let block_size = 64;
    let mut ko = KnystOffline::new(sample_rate, block_size, 0, 2);
    let mut k = knyst_commands();
    let mut settings = k.default_graph_settings();
    settings.sample_rate = sample_rate as Sample;
    settings.block_size = block_size * 4;
    settings.num_outputs = 2;
    settings.num_inputs = 0;
    let mut graph = Graph::new(settings);

    let dummy_l = graph.push(DummyGen::new(0.));
    let amp_l = graph.push(Mult);
    graph
        .connect(amp_l.to_graph_out().from_index(0).to_index(0))
        .unwrap();
    graph.connect(constant(2.0).to(amp_l).to_index(0)).unwrap();
    graph.connect(dummy_l.to(amp_l).to_index(1)).unwrap();
    let dummy_r = graph.push(DummyGen::new(100.));
    let amp_r = graph.push(Mult);
    graph
        .connect(amp_r.to_graph_out().from_index(0).to_index(1))
        .unwrap();
    graph.connect(constant(2.0).to(amp_r).to_index(0)).unwrap();
    graph.connect(dummy_r.to(amp_r).to_index(1)).unwrap();

    let graph_id = k.push(graph, inputs!());
    k.connect(graph_id.to_graph_out().channels(2));
    ko.process_block();

    let out_left = ko.output_channel(0).unwrap();
    let out_right = ko.output_channel(1).unwrap();
    for (&l, &r) in out_left.into_iter().zip(out_right) {
        assert!(l == r - 200.);
    }
}
