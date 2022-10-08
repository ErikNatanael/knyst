use knyst::envelope::Envelope;

fn main() {
    let sample_rate = 44100.0;
    let envelope = Envelope {
        start_value: 0.0,
        points: vec![(1.0, 0.1), (0.5, 0.5), (0.1, 1.2)],
        sample_rate,
        ..Default::default()
    };
    let mut envelope = envelope.to_gen();
    let mut all_values = Vec::with_capacity(44100);
    for _ in 0..(sample_rate as usize * 2) {
        let value = envelope.next_sample();
        all_values.push(value);
    }
    println!("{all_values:?}");
}
