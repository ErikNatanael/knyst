# Knyst examples

All the examples currently use either the JACK or the CPAL backend. That means that to run them you need to enable that feature, e.g.

```sh
cargo run --release --example interactive --features cpal
```

## Interactive

This example aims to provide an overview of different ways that Knyst can be used. The example currently demonstrates:

- using fundsp within a Gen and passing Knyst audio through it
- async using tokio
- interactivity using the keyboard to play pitches monophonically
- sound file playback
- replacing a wavetable in Resources while everything is running
