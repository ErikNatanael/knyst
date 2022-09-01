use criterion::{black_box, criterion_group, criterion_main, Criterion};
use knyst::wavetable::{Phase, PhaseF32};
use knyst::{prelude::*, FRACTIONAL_PART};

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
            let mut phase = Phase(0);
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

criterion_group!(benches, phase_float_or_uint);
criterion_main!(benches);
