use std::f32::consts::TAU;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use knyst::envelope::{Curve, Envelope};
use knyst::prelude::*;
use knyst::wavetable::{PhaseF32, WavetablePhase, FRACTIONAL_PART};

// Test if integer phase is in fact faster than floating point phase
pub fn phase_float_or_uint(c: &mut Criterion) {
    let sample_rate = 44100.0;

    let freq_to_phase_inc =
        (TABLE_SIZE as f64 * FRACTIONAL_PART as f64 * (1.0 / sample_rate as f64)) as Sample;
    let freq_to_f32_phase_inc = 1.0 / sample_rate as f32;
    let freq = 824.3;
    let table: Vec<f32> = (0..TABLE_SIZE)
        .map(|i| ((i as f32) / TABLE_SIZE as f32).sin())
        .collect();
    let differences: Vec<f32> = table
        .iter()
        .zip(table.iter().skip(1).cycle())
        .map(|(&a, &b)| b - a)
        .collect();
    c.bench_function("u32 phase", |b| {
        b.iter(|| {
            let mut phase = WavetablePhase(0);
            for _ in 0..44100 {
                let index = phase.integer_component();
                let mix = phase.fractional_component_f32();
                let value = table[index] + differences[index] * mix;
                black_box(value);
                phase.increase((freq * freq_to_phase_inc) as u32);
            }
        })
    });
    c.bench_function("f32 phase", |b| {
        b.iter(|| {
            let mut phase = PhaseF32(0.);
            for _ in 0..44100 {
                let (index, mix) = phase.index_mix();
                let value = table[index] + differences[index] * mix;
                black_box(value);
                phase.increase(freq * freq_to_f32_phase_inc);
            }
        })
    });
}

pub fn envelope_segments(c: &mut Criterion) {
    let sample_rate = 44100.0;
    c.bench_function("envelope linear", |b| {
        b.iter(|| {
            let envelope = Envelope {
                start_value: 0.0,
                points: vec![(1.0, 0.1), (0.5, 0.5), (0.1, 1.2)],
                ..Default::default()
            };
            let mut envelope = envelope.to_gen();
            let mut all_values = Vec::with_capacity(44100);
            for _ in 0..sample_rate as usize {
                let value = envelope.next_sample();
                all_values.push(value);
                black_box(value);
            }
        })
    });
    c.bench_function("envelope exponential", |b| {
        b.iter(|| {
            let envelope = Envelope {
                start_value: 0.0,
                points: vec![(1.0, 0.1), (0.5, 0.5), (0.1, 1.2)],
                curves: vec![
                    Curve::Exponential(2.0),
                    Curve::Exponential(4.0),
                    Curve::Exponential(0.125),
                ],
                ..Default::default()
            };
            let mut envelope = envelope.to_gen();
            let mut all_values = Vec::with_capacity(44100);
            for _ in 0..sample_rate as usize {
                let value = envelope.next_sample();
                all_values.push(value);
                black_box(value);
            }
        })
    });
}

pub fn wavetable_vs_sin(c: &mut Criterion) {
    let frequency = 440.0;
    let sample_rate = 48000.0;
    let integer_phase_increase =
        ((frequency / sample_rate) * TABLE_SIZE as f32 * FRACTIONAL_PART as f32) as u32;
    let w = Wavetable::sine();
    let mut phase = WavetablePhase(0);
    c.bench_function("wavetable sin linear", |b| {
        b.iter(|| {
            black_box(w.get_linear_interp(phase, frequency));
            phase.increase(integer_phase_increase);
        });
    });
    let mut phase = WavetablePhase(0);
    c.bench_function("wavetable sin no interpolation", |b| {
        b.iter(|| {
            black_box(w.get(phase, frequency));
            phase.increase(integer_phase_increase);
        });
    });
    let mut phase: f32 = 0.0;
    let float_phase_increase = (frequency / sample_rate) * TAU;

    c.bench_function("raw sin no phase overflow", |b| {
        b.iter(|| {
            black_box(phase.sin());
            phase += float_phase_increase;
        });
    });
    c.bench_function("raw sin", |b| {
        b.iter(|| {
            black_box(phase.sin());
            phase += float_phase_increase;
            while phase > TAU {
                phase -= TAU;
            }
        });
    });
    c.bench_function("raw tanh", |b| {
        b.iter(|| {
            black_box(phase.tanh());
            phase += float_phase_increase;
            while phase > TAU {
                phase -= TAU;
            }
        });
    });
}

// criterion_group!(benches, phase_float_or_uint);
// criterion_group!(benches, envelope_segments);
criterion_group!(benches, wavetable_vs_sin);

criterion_main!(benches);
