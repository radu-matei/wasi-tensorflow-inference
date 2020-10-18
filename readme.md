# TensorFlow inferencing in WebAssembly outside the browser

This project is a demonstration of running the [MobileNet V2 TensorFlow
model][mobilenet] in WebAssembly System Interface (WASI) runtimes outside the
browser. The project uses [the Sonos Tract crate][sonos-tract] to build an
inference program in Rust, which is then compiled to Rust's `wasm32-wasi`
WebAssembly target.

This is an educational project, which has a few goals:

- build a more complex WebAssembly module that does _not_ use any code
  generation for bindings (such as [`wasm-bindgen`][wasm-bindgen]).
- run the module in Wasmtime, exemplifying writing arbitrary data (such as
  images) in the guest's linear memory using Rust and `Wasmtime::Memory`.
- build a simple runtime on top of a web server that accepts incoming
  connections, reads an image URL from their request body, execute the inference
  on the fetched image, and return the model's prediction
- execute the same WebAssembly module in Node's WASI runtime and exemplify
  writing into the module's linear memory using JavaScript.

### Implementation Notes

- the project uses a pre-trained convolutional neural network model - [MobileNet
  V2][mobilenet] with around 6M parameters, used for computer vision tasks such
  as classification or object detection.
- this project starts from [the Sonos Tract's example][sonos-example-mobilenet]
  for execution the model locally, and makes the necessary changes to compile it
  to WebAssembly.
- while the _approach_ used by this project can be used to execute inferences
  using different neural network models, the implementation is specialized for
  performing inferences using MobileNet model. Changing the model architecture,
  as well as its inputs and outputs, requires significant changes in both the
  WebAssembly module, as well as how it is instantiated in Wasmtime.
- because a `Wasmtime::Instance` [cannot be safely sent between
  threads][instance-send], a new instance of the module is created for each
  request, which adds to the overall latency.

### Building and running from source

When executing `cargo build`, the following are executed:

- build and optimize a WebAssembly module based on [the
  `wasi-mobilenet-inferencing` crate][crate] (see [`build.rs`][build])
- build a server that listens for HTTP requests and get the model prediction for
  the image URL in the request body using Wasmtime.

```
$ cargo run --release
Listening on http://127.0.0.1:3000

module instantiation time: 774.715145ms
inference time: 723.531083ms
```

In another terminal instance (or from a HTTP request builder, such as Postman):

```
$ curl --request GET 'localhost:3000' \
--header 'Content-Type: text/plain' \
--data-raw 'https://upload.wikimedia.org/wikipedia/commons/3/33/GoldenRetrieverSnow.jpg'
golden retriever
```

Prerequisites (required in the path):

- `cargo`
- [`wasm-opt` from Binaryen][binaryen]

### Testing the module in Node's WASI runtime

The repository contains an already built and optimized module, which can be
found in `model/optimized-wasi.wasm`. It can be tested without any compilation
(and without any Node dependencies) using a recent NodeJS installation:

```
$ node -v
v14.5.0
$ node --experimental-wasi-unstable-preview1 --experimental-wasm-bigint test.js

predicting on file  golden-retriever.jpeg
inference time:  953 ms
prediction:  golden retriever

predicting on file  husky.jpeg
inference time:  625 ms
prediction:  Eskimo dog, husky
```

[binaryen]: https://github.com/WebAssembly/binaryen#tools
[mobilenet]:
  https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
[sonos-example-mobilenet]:
  https://github.com/sonos/tract/tree/main/examples/tensorflow-mobilenet-v2
[sonos-tract]: https://github.com/sonos/tract
[wasm-bindgen]: https://github.com/rustwasm/wasm-bindgen
[instance-send]: https://github.com/bytecodealliance/wasmtime/issues/793
[crate]: ./crates/wasi-mobilenet-inference/src/lib.rs
[build]: ./build.rs
