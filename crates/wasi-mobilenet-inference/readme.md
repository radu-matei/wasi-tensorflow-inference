# `wasi-mobilenet-inference`

### Building

This crate is automatically built when compiling the top-level project. However,
to manually compile and optimize the module:

```
$ cargo build --target wasm32-wasi --release
$ wasm-opt target/wasm32-wasi/release/wasi_mobilenet_inference.wasm -O -o ../../model/optimized-wasi.wasm
```
