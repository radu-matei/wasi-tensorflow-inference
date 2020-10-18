fn main() {
    println!("cargo:rerun-if-changed=crates/wasi-mobilenet-inference/src/lib.rs");

    build_inference_crate();
}

fn build_inference_crate() {
    let mut cmd = std::process::Command::new("cargo");
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    cmd.arg("build")
        .arg("--manifest-path")
        .arg("crates/wasi-mobilenet-inference/Cargo.toml")
        .arg("--release")
        .arg("--target")
        .arg("wasm32-wasi");

    cmd.output().unwrap();
    run_wasm_opt();
}

fn run_wasm_opt() {
    let mut cmd = std::process::Command::new("wasm-opt");
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    cmd.arg(
        "crates/wasi-mobilenet-inference/target/wasm32-wasi/release/wasi_mobilenet_inference.wasm",
    )
    .arg("-O")
    .arg("-o")
    .arg("model/optimized-wasi.wasm");
    cmd.output().unwrap();
    println!("executed wasm-opt");
}
