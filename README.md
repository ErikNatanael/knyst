# Knyst

Knyst is a real time audio synthesis framework focusing on flexibility and performance. It's main target use case is desktop multi-threaded environments, but it can also do single threaded and/or non real time synthesis. Embedded platforms are currently not supported, but on the roadmap.

> [!IMPORTANT]  
> Knyst is not stable. Knyst's API is still developing and can vary wildly between releases.


# Examples

## Play a sine wave

```rust
// Set up knyst
let mut backend = CpalBackend::new(CpalBackendOptions::default())?;
let _sphere = KnystSphere::start(
    &mut backend,
    SphereSettings {
        num_inputs: 1,
        num_outputs: 2,
        ..Default::default()
    },
    print_error_handler,
);
// Add a sine wave to the top level graph
let sine_osc_handle = oscillator(WavetableId::cos()).freq(200.);
// Output it to the graph output, which for the top level graph is the
// application output.
graph_output(0, sine_osc_handle);
```

## Implement your own `Gen`

Using the `impl_gen` macro takes care of most of the boiler plate. However, if you need a variable number of inputs or outputs
you have to implement the `Gen` trait yourself.

```rust 
struct DummyGen {
    counter: Sample,
}
#[impl_gen]
impl DummyGen {
    // Having a `new` method enabled generation of a function `dummy_gen() -> Handle<DummyGenHandle>`
    pub fn new() -> Self {
      Self {
        counter: 0.,
      }
    }

    /// The process method, or any method with the `#[process]` attribute is what will be run every block
    /// when processing audio. It must not allocate or wait (unless you are doing non-realtime stuff).
    /// 
    /// Inputs are extracted from any argument with the type &[Sample]
    /// Outputs are extracted from any argument with the type &mut [Sample]
    /// 
    /// Additionally, &[Trig] is an input channel of triggers and enables the `inputname_trig()` method on the handle
    /// to conveniently schedule a trigger. `SampleRate` gives you the current sample rate and `BlockSize` gives you
    /// the block size.
    #[process]
    fn process(&mut self, counter: &[Sample], output: &mut [Sample]) -> GenState {
        for (count, out) in counter.iter().zip(output.iter_mut()) {
            self.counter += 1.;
            *out = count + self.counter;
        }
        GenState::Continue
    }
    /// The init method will be called before adding the `Gen` to a running graph. Here is where you can
    /// allocate buffers and initialise state based on sample rate and block size.
    fn init(&mut self, _sample_rate: SampleRate, _block_size: BlockSize, _node_id: NodeId) {
      self.counter = 0.;
    }
}
```

# Features

- good runtime performance
- real time changes to the audio graph via an async compatible interface
- sample accurate node and parameter change scheduling to the graph
- interopability with static Rust DSP libraries e.g. dasp and fundsp (they can be encapsulated in a node and added to the graph)
- graphs can be nodes and changes can be applied to graphs within graphs at any depth
- feedback connections to get a 1 block delayed output of a node to an earlier node in the chain
- any number of inputs/outputs from a node
- allows inner graphs with different block sizes using automatic buffering where necessary
- choose your level of flexibility and optimisation: create a graph with lots of interconnected nodes or hard code it all into one node when you need more performance

# Safety

Knyst uses a little bit of unsafe under the hood to improve performance in the most sensitive parts of the library, as do some of its dependencies. The user, however, never _needs_ to write any unsafe code.

## The name

"Knyst" is a Swedish word meaning _very faint sound_. It is normally almost exclusively used in the negative e.g. "Det hörs inte ett knyst" (eng. "You cannot hear a sound"), but I think it's well worth some affirmative use.

# Roadmap

Vision for the future

- automatic parameter change interpolation
- automatic GUI generation from the graph incl. interactively changing connections and parameters
- tools for musical time scheduling incl phrasing options e.g. rubato, accel./rit. and asymmetric time signatures
- parallel processing of large graphs
- automatic sample rate conversion
- support no_std for embedded platforms

# License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in Knyst by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
