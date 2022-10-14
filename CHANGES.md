# Change Log

## Unreleased

- Fixed undefined behaviour and data races discovered through miri (#3). Thanks harudagondi for brining it to my attention! This brought about some necessary breaking changes to the API:
  - `Gen::process` now provides a `GenContext` parameter instead of `inputs` and `outputs` slices.
  - A new `RunGraph` interface can be used to run the main `Graph` either on the audio thread or for non real time applications.
  - `Node` is removed from the public API because it is too unsafe to expose.
- Added and refined documentation
- Renamed some methods to agree with Rust conventions.
