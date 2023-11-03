use criterion::{black_box, criterion_group, criterion_main, Criterion};
use knyst::{graph::RunGraph, prelude::*};

pub fn empty_graph(c: &mut Criterion) {
    let graph_settings = GraphSettings {
        block_size: 64,
        sample_rate: 44100.,
        num_outputs: 2,
        ..Default::default()
    };
    let mut graph = Graph::new(graph_settings);
    let resources = Resources::new(ResourcesSettings::default());
    let (mut run_graph, _, _) =
        RunGraph::new(&mut graph, resources, RunGraphSettings::default()).unwrap();
    c.bench_function("0 nodes, block size 64, sr 44100", |b| {
        b.iter(|| {
            run_graph.process_block();
            black_box(run_graph.graph_output_buffers().get_channel(0));
        });
    });
}
pub fn large_sine_graph(c: &mut Criterion) {
    let graph_settings = GraphSettings {
        block_size: 64,
        sample_rate: 44100.,
        num_outputs: 2,
        ..Default::default()
    };
    let mut graph = Graph::new(graph_settings);
    for _ in 0..1000 {
        let sine = graph.push(WavetableOscillatorOwned::new(Wavetable::sine()));
        graph.connect(sine.to_graph_out()).unwrap();
    }
    graph.update();
    let resources = Resources::new(ResourcesSettings::default());
    let (mut run_graph, _, _) =
        RunGraph::new(&mut graph, resources, RunGraphSettings::default()).unwrap();
    c.bench_function("1000 nodes, block size 64, sr 44100", |b| {
        b.iter(|| {
            run_graph.process_block();
            black_box(run_graph.graph_output_buffers().get_channel(0));
        });
    });
    let graph_settings = GraphSettings {
        block_size: 16,
        sample_rate: 44100.,
        num_outputs: 2,
        ..Default::default()
    };
    let mut graph = Graph::new(graph_settings);
    for _ in 0..1000 {
        let sine = graph.push(WavetableOscillatorOwned::new(Wavetable::sine()));
        graph.connect(sine.to_graph_out()).unwrap();
    }
    graph.update();
    let resources = Resources::new(ResourcesSettings::default());
    let (mut run_graph, _, _) =
        RunGraph::new(&mut graph, resources, RunGraphSettings::default()).unwrap();
    c.bench_function("1000 nodes, block size 16, sr 44100", |b| {
        b.iter(|| {
            run_graph.process_block();
            black_box(run_graph.graph_output_buffers().get_channel(0));
        });
    });
    let graph_settings = GraphSettings {
        block_size: 512,
        sample_rate: 44100.,
        num_outputs: 2,
        ..Default::default()
    };
    let mut graph = Graph::new(graph_settings);
    for _ in 0..1000 {
        let sine = graph.push(WavetableOscillatorOwned::new(Wavetable::sine()));
        graph.connect(sine.to_graph_out()).unwrap();
    }
    graph.update();
    let resources = Resources::new(ResourcesSettings::default());
    let (mut run_graph, _, _) =
        RunGraph::new(&mut graph, resources, RunGraphSettings::default()).unwrap();
    c.bench_function("1000 nodes, block size 512, sr 44100", |b| {
        b.iter(|| {
            run_graph.process_block();
            black_box(run_graph.graph_output_buffers().get_channel(0));
        });
    });
}

fn create_pyramid_graph(power_of_two: usize, mut graph_settings: GraphSettings) -> Graph {
    graph_settings.num_nodes = 2_usize.pow(power_of_two as u32);

    let mut graph = Graph::new(graph_settings);
    let mut last_layer_node_ids = Vec::new();
    let mut this_layer_node_ids = Vec::new();
    for i in 0..power_of_two {
        this_layer_node_ids.clear();
        let num_nodes = 2_usize.pow(i as u32);
        for j in 0..num_nodes {
            let gen = WavetableOscillatorOwned::new(Wavetable::sine());
            let node_id = graph.push(gen);
            if last_layer_node_ids.len() == 0 {
                graph.connect(node_id.to_graph_out()).unwrap();
            } else {
                let index = j / 2;
                graph
                    .connect(node_id.to(last_layer_node_ids[index]))
                    .unwrap();
            }
            this_layer_node_ids.push(node_id);
        }
        std::mem::swap(&mut last_layer_node_ids, &mut this_layer_node_ids);
    }
    graph
}
pub fn large_complex_sine_graph(c: &mut Criterion) {
    let graph_settings = GraphSettings {
        block_size: 64,
        sample_rate: 44100.,
        num_outputs: 2,
        ..Default::default()
    };
    let mut graph = create_pyramid_graph(10, graph_settings);
    graph.update();
    let resources = Resources::new(ResourcesSettings::default());
    let (mut run_graph, _, _) =
        RunGraph::new(&mut graph, resources, RunGraphSettings::default()).unwrap();
    c.bench_function(
        &format!("{} nodes, block size 64, sr 44100", 2_i32.pow(10)),
        |b| {
            b.iter(|| {
                run_graph.process_block();
                black_box(run_graph.graph_output_buffers().get_channel(0));
            });
        },
    );
}
criterion_group!(
    benches,
    empty_graph,
    large_sine_graph,
    large_complex_sine_graph
);

criterion_main!(benches);
