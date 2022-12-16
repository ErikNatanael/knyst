# Change Log

## 0.3.1

- Fixed a data race bug potentially resulting in a segfault by replacing the generation counting synchronisation with atomic flags.
- Improved tests for catching new data race bugs in the future.

## 0.3.0

- Fixed serious bug in signal routing due to the incorrect index being set when generating Tasks (i.e. the graph representation running on the audio thread).
- Introduces a unified way of pushing both `Gen`s and `Graph`s to the `Graph` using the `Graph::push` method, which relies on the `GenOrGraph` trait.

## 0.2.0

"if some code contains UB but also plays a sine wave, is that code then sound?" - orlp

Shortly after the release of v0.1.0, harudagondi discovered some Undefined Behaviour using [miri](https://github.com/rust-lang/miri). The fixes required breaking changes to the Gen trait, but I feel like the result is an improvement to ergonomics when writing Gens. Thanks harudagondi for bringing the UB to my attention and also for the other improvement suggestions! Another big thanks goes out to the #dark-arts discord channel for helping me understand the language of miri.

- Fixed undefined behaviour and data races discovered through miri (#3). This brought about some necessary breaking changes to the API:
  - `Gen::process` now provides a `GenContext` parameter instead of `inputs` and `outputs` slices.
  - A new `RunGraph` interface can be used to run the main `Graph` either on the audio thread or for non real time applications.
  - `Node` and `Graph::to_node` are removed from the public API because the `Node` is too unsafe to expose.
- Added and refined documentation and documentation examples.
- Renamed some methods to agree with Rust conventions.
