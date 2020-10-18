use std::{
    fs::{metadata, File},
    io::{BufRead, Read},
    time::Instant,
};

use hyper::service::{make_service_fn, service_fn};
use hyper::{body::HttpBody as _, Client};
use hyper::{Body, Request, Response, Server};
use hyper_tls::HttpsConnector;

use wasmtime::*;
use wasmtime_wasi::{Wasi, WasiCtxBuilder};

const MOBILENET_V2: &str = "./model/mobilenet_v2_1.4_224_frozen.pb";
const LABELS: &str = "./model/labels.txt";
const WASM: &str = "./model/optimized-wasi.wasm";

const ALLOC_FN: &str = "alloc";
const MEMORY: &str = "memory";
const INFER_FN: &str = "infer_from_ptrs";

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let make_svc = make_service_fn(|_conn| async { Ok::<_, anyhow::Error>(service_fn(predict)) });

    let addr = ([127, 0, 0, 1], 3000).into();
    let server = Server::bind(&addr).serve(make_svc);
    println!("Listening on http://{}", addr);
    server.await?;
    Ok(())
}

/// Respond to a request containing the URL of an image with the result of
/// running the Mobilenet V2 model on the image.
async fn predict(req: Request<Body>) -> Result<Response<Body>, anyhow::Error> {
    let (_, body) = req.into_parts();

    // The current assumption is that the request body contains a
    // single URL pointing to an image.
    let data = hyper::body::to_bytes(body).await?.to_vec();

    let url = std::str::from_utf8(&data)?;
    match get_prediction(url).await {
        Ok(label) => {
            return Ok(Response::new(Body::from(label)));
        }
        Err(_) => return Err(anyhow::Error::msg("cannot get prediction")),
    }
}

/// Download an image from a given URL and run the Mobilenet V2 model.
async fn get_prediction(url: &str) -> Result<String, anyhow::Error> {
    let img_bytes = fetch_url_to_bytes(url).await?;
    let model_bytes = read_file_bytes(MOBILENET_V2.to_string())?;

    // Unfortunately, we have to create a new module instance for every prediction,
    // since a Wasmtime::Instance cannot be safely sent between threads.
    // See https://github.com/bytecodealliance/wasmtime/issues/793
    let instance = create_instance(WASM.to_string())?;

    let start = Instant::now();

    // Write the Mobilenet model and the image contents to
    // the module's linear memory, and get their pointers.
    let model_bytes_ptr = write_guest_memory(&model_bytes, &instance)?;
    let img_bytes_ptr = write_guest_memory(&img_bytes, &instance)?;

    // Get the module's "infer_from_ptrs" function, which is the
    // entrypoint for executing the inference.
    // If the function is not found, the execution cannot continue.
    let infer = instance
        .get_func(INFER_FN)
        .expect("expected inference function not found");

    // Call the inference function with the pointer and length of the
    // model contents and image.
    let results = infer.call(&vec![
        Val::from(model_bytes_ptr as i32),
        Val::from(model_bytes.len() as i32),
        Val::from(img_bytes_ptr as i32),
        Val::from(img_bytes.len() as i32),
    ])?;
    let duration = start.elapsed();
    println!("inference time: {:#?}", duration);

    // The inference function has one return argument, the index of the
    // predicted class.
    match results
        .get(0)
        .expect("expected the result of the inference to have one value")
    {
        Val::I32(val) => {
            let label = get_label(*val as usize)?;
            return Ok(label);
        }
        _ => return Err(anyhow::Error::msg("cannot get prediction")),
    }
}

/// Return a buffer with the contents of an image from a given URL.
/// Note that this will download the contents of a random URL,
/// which will later be copied into the module's linear memory.
async fn fetch_url_to_bytes(url: &str) -> Result<Vec<u8>, anyhow::Error> {
    let mut buf: Vec<u8> = Vec::new();
    let https = HttpsConnector::new();
    let client = Client::builder().build::<_, hyper::Body>(https);
    let uri = url.parse::<hyper::Uri>()?;
    let mut res = client.get(uri).await?;
    while let Some(next) = res.data().await {
        let chunk = next?;
        std::io::Write::write(&mut buf, &mut chunk.to_vec())?;
    }
    Ok(buf)
}

/// Get the human-readable label of a prediction
/// from the Mobilenet V2 labels file
fn get_label(num: usize) -> Result<String, anyhow::Error> {
    // The result of executing the inference is the predicted class,
    // which also indicates the line number in the (1-indexed) labels file.
    let labels = File::open(LABELS.to_string())?;
    let content = std::io::BufReader::new(&labels);
    content
        .lines()
        .nth(num - 1)
        .expect("cannot get prediction label")
        .map_err(|err| anyhow::Error::new(err))
}

/// Write a bytes array into the instance's linear memory
/// and return the offset relative to the module's memory.
fn write_guest_memory(bytes: &Vec<u8>, instance: &Instance) -> Result<isize, anyhow::Error> {
    // Get the "memory" export of the module.
    // If the module does not export it, just panic,
    // since we are not going to be able to copy the model and image.
    let memory = instance
        .get_memory(MEMORY)
        .expect("expected memory not found");

    // The module is not using any bindgen libraries, so it should export
    // its own alloc function.
    //
    // Get the guest's exported alloc function, and call it with the
    // length of the byte array we are trying to copy.
    // The result is an offset relative to the module's linear memory, which is
    // used to copy the bytes into the module's memory.
    // Then, return the offset.
    unsafe {
        let alloc = instance
            .get_func(ALLOC_FN)
            .expect("expected alloc function not found");
        let alloc_result = alloc.call(&vec![Val::from(bytes.len() as i32)])?;

        let guest_ptr_offset = match alloc_result
            .get(0)
            .expect("expected the result of the allocation to have one value")
        {
            Val::I32(val) => *val as isize,
            _ => return Err(anyhow::Error::msg("guest pointer must be Val::I32")),
        };

        let raw = memory.data_ptr().offset(guest_ptr_offset);
        raw.copy_from(bytes.as_ptr(), bytes.len());
        return Ok(guest_ptr_offset);
    }
}

/// Create a Wasmtime::Instance from a compiled module and
/// link the WASI imports.
fn create_instance(filename: String) -> Result<Instance, anyhow::Error> {
    let start = Instant::now();
    let store = Store::default();
    let mut linker = Linker::new(&store);

    let ctx = WasiCtxBuilder::new()
        .inherit_stdin()
        .inherit_stdout()
        .inherit_stderr()
        .build()?;

    let wasi = Wasi::new(&store, ctx);
    wasi.add_to_linker(&mut linker)?;
    let module = wasmtime::Module::from_file(store.engine(), filename)?;

    let instance = linker.instantiate(&module)?;
    let duration = start.elapsed();
    println!("module instantiation time: {:#?}", duration);
    return Ok(instance);
}

/// Return the contents of a file.
fn read_file_bytes(filename: String) -> Result<Vec<u8>, std::io::Error> {
    let mut file = File::open(&filename)?;
    let meta = metadata(&filename)?;
    let mut buf = vec![0; meta.len() as usize];
    file.read(&mut buf)?;

    Ok(buf)
}
