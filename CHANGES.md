# Change Log

## Current changes

## v0.5.1

- New examples (thank you Tuurlijk)
- Fixed a bug in calculating the node order when connecting a multiple output node to different nodes per output.

## v0.5.0

Large breaking changes in this version

- New Handle API for ergonomically building and interacting with the graph!
- A modal thread local interface to interact with knyst, making the Handle system possible, accessed through `knyst_commands()`.
- NodeAddress was replaced by NodeId which is Copy.
- Enabled routing a graph's inputs to its outputs directly, mostly for completeness of the API.
- Fixed bug where input from graph inputs would overwrite other inputs to a node.
- Updated all dependencies to the latest version.
- New `impl_gen` macro for implementing the Gen trait and adding a Handle with input dependent methods and with much less error prone boilerplate.
- Fixed bug in `Wavetable` which produced artefacts.
- Refactored the codebase to something more logical. Many paths have changed, but importing the prelude should give you most of what you need.

## v0.4.3

- Allow clearing all connections for a specific channel by calling `to_channel`, `from_channel` and similar on a `Connection::Clear`.
- Allow scheduling multiple changes at the same time using `SimultaneousChanges`.
- Allow scheduling triggers the same way input constant changes are scheduled.
- Fix a bug in sustaining envelopes.
- Add the _unstable_ feature, providing some SIMD support in nightly.

## v0.4.2

- Fix arithmetics bug for `Superbeats` and `Superseconds`.
- Update fundsp to the latest version which allows us to allocate fundsp internal buffers in advance instead of on the audio thread.

## v0.4.1

- Disable the JACK AudioBackend notifications from printing to stdout since this interferes with tui applications.

## v0.4.0

- Convenient and unified way of running knyst in single or multi threaded contexts through the `Controller` and `KnystCommands` (name inspired from Bevy). This was incorporated in the AudioBackend API for convenience.
- Schedule nodes to start with sample accuracy (see `Graph::push_at_time` and similar).
- Oversampling of graphs. Currently limited to 2x oversampling for graphs with no inputs. Inner graphs cannot have a lower oversampling than their parent graph.
- Introduce a definition of a trigger which is currently a value above 0.0 for one sample. Multiple samples in a row with values > 0.0 mean multiple triggers. Triggers can currently be used to trigger envelopes, but will be expanded in the future.
- Inner graphs with smaller or larger block sizes are automatically converted to their parent graph's block size. Inner graphs with a larger block size cannot have inputs, since the inputs for the block would not exist yet when the larger block is run. This feature enables sample by sample processing for only a small part of a node tree, while the rest of the tree (in an outer graph) can benefit from block processing.
- Fixed a panic that occurred when adding an edge to a node that was pending removal.
- Introduces the ability to push a node while reusing a preexisting NodeAddress. This is useful mainly for the KnystCommands API to create the NodeAddress before the node is pushed.
- Changed NodeAddress to be async compatible i.e. it can be created before the Gen/Graph has been pushed to a Graph. This required making it not Copy and methods taking a reference to it instead.
- Renamed functions/methods for clarity.
  - ParameterChange::relative_duration -> ParameterChange::duration_from_now
  - Connection::clear_output_nodes -> Connection::clear_to_nodes
  - Connection::clear_inputs -> Connection::clear_from_nodes
  - Connection::clear_graph_outputs -> Connection::clear_to_graph_outputs
  - Connection::clear_graph_inputs -> Connection::clear_from_graph_inputs
- Removed the deprecated methods (see previous version log notes).
- Internal changes to make scheduling changes more flexible and exact.
- Move latency setting from GraphSettings to RunGraphSettings so that the latency is the same for every Graph.
- Replace absolute sample scheduling by a type that can handle all common sample rates transparently, inspired by BillyDM's blogpost: https://billydm.github.io/blog/time-keeping/
- Also implement a musical beat scheduling time primitive along the ideas of BillyDM expressed in that same blog post.
- Barebones scheduling of events in musical time.
- Streamlined examples to decrease maintenance time.
- Enable assertions for no allocation on the audio thread when compiling in debug mode.

## 0.3.1

- Fixed a data race bug potentially resulting in a segfault by replacing the generation counting synchronisation with atomic flags.
- Improved tests for catching new data race bugs in the future.
- Deprecated `Graph::push_graph` and `Graph::push_gen` methods

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
