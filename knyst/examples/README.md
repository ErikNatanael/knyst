# Knyst examples

List all available examples using:

```shell
cargo run --example
```

## Basic examples

### Tone

This example plays a single tone

```shell
cargo run --example tone
```

### Tones

This example plays two tones. One in the left channel and one in the right channel.

```shell
cargo run --example tones
```

### Modulation

This example plays a tone modulated by a oscillator.

```shell
cargo run --example modulation
```

### Adjust frequency

This example plays a different tone in each channel, both with different modulators. You can change the frequency of the tone in the left channel by entering a numerical value.

```shell
cargo run --example adjust_frequency
```

### Sound file playback

Plays back 10 seconds of an audio file chosen by the user.

```shell
cargo run --example sound_file_playback
```

### Schedule a tone

Plays back a tone after 3 seconds of silence.

```shell
cargo run --example scheduling
```

## Envelopes

### Volume envelope

This example plays a tone with a volume envelope.

```shell
cargo run --example volume_envelope
```

### Frequency envelope

This example plays a tone with a frequency envelope.

```shell
cargo run --example frequency_envelope
```

## Advanced

### Beat Callbacks

The main function initializes and starts the audio processing system with the default settings. It sets up a graph with wavetables, modulators, and amplitude modulators, and schedules beat-accurate parameter changes. The function reads user input to allow interaction with the callback and offers options to stop the callback or quit the program.

```shell
cargo run --example beat_callbacks
```

### Filter

```shell
cargo run --example filter_test
```

### Interactive

This example aims to provide an overview of different ways that Knyst can be used. The example currently demonstrates:

- starting an audio backend
- pushing nodes
- making connections
- inner graphs
- async and multi threaded usage of KnystCommands
- scheduling changes
- interactivity
- wrapping other dsp libraries (fundsp in this case)
- writing a custom error handler

```shell
cargo run --example interactive
```

### More advanced example

```shell
cargo run --example more_advanced_example
```

## Using JACK

All the examples currently use either the JACK or the CPAL backend. If you want to use JACK, add that as a feature flag. Also uncomment the JACK backend line in the example and comment out the CPAL backend line.

```sh
cargo run --example filter_test --features jack
```
