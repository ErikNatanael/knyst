# Knyst

Knyst is a real time audio synthesis framework focusing on flexibility and performance. It's main target use case is desktop multi-threaded environments, but it can also do single threaded and/or non real time synthesis. Embedded platforms are currently not supported, but on the roadmap.

## The name

"Knyst" is a Swedish word meaning _very faint sound_. It is normally almost exclusively used in the negative e.g. "Det hörs inte ett knyst" (eng. "You cannot hear a sound"), but I think it's well worth some affirmative use.

# Features

- good runtime performance
- interopability with static Rust DSP libraries e.g. dasp and fundsp (they can be encapsulated in a node and added to the graph)
- real time changes to the audio graph
- sample accurate parameter changes
- graphs can be nodes and changes can be applied to graphs within graphs at any depth
- feedback connections to get a 1 block delayed output of a node
- any number of inputs/outputs from a node
- choose your level of flexibility and optimisation: create a graph with lots of interconnected nodes or hard code it all into one node when you need more performance

# Safety

Knyst uses a little bit of unsafe under the hood to improve performance in the most sensitive parts of the library, as do some of its dependencies. The user, however, never _needs_ to write any unsafe code.

# Roadmap

Vision for the future

- sample accurate node addition to the graph
- sample accurate connection change to the graph
- automatic parameter change interpolation
- automatic GUI generation from the graph incl. interactively changing connections and parameters
- parallel processing of large graphs
- automatic sample rate conversion
- allowing graphs with differenc block sizes using automatic buffering
- support no_std for embedded platforms

# License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in Knyst by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
