use tract_tensorflow::prelude::*;

/// Allocate memory into the module's linear memory
/// and return the offset to the start of the block.
#[no_mangle]
pub unsafe extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::with_capacity(len);
    let ptr = buf.as_mut_ptr();

    std::mem::forget(buf);
    return ptr;
}

/// This is the module's entry point for executing inferences.
/// It takes as arguments pointers to the start of the module's memory blocks
/// where the model and the image were copied, as well as their lengths,
/// meaning that callers of this function must first copy the model and
/// image into the module's linear memory using the module's `alloc` function.
///
/// It retrieves the contents of the model and image, then calls
/// the `wasmtime_infer` function, which performs the prediction.
#[no_mangle]
pub unsafe extern "C" fn infer_from_ptrs(
    model_ptr: *mut u8,
    model_len: usize,
    img_ptr: *mut u8,
    img_len: usize,
) -> i32 {
    let model_bytes = Vec::from_raw_parts(model_ptr, model_len, model_len);
    let img_bytes = Vec::from_raw_parts(img_ptr, img_len, img_len);

    return wasmtime_infer(&model_bytes, &img_bytes);
}

/// Perform the inference given the contents of the model and the image, and
/// return the index of the predicted class.
///
/// Adapted from https://github.com/sonos/tract/tree/main/examples/tensorflow-mobilenet-v2 and
/// using the TensorFlow Mobilenet V2 model.
/// See https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
fn wasmtime_infer(model_bytes: &[u8], image_bytes: &[u8]) -> i32 {
    let mut model = std::io::Cursor::new(model_bytes);
    let model = tract_tensorflow::tensorflow()
        .model_for_read(&mut model)
        .unwrap()
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 224, 224, 3)),
        )
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();

    let image = image::load_from_memory(image_bytes).unwrap().to_rgb();
    // The model was trained on 224 x 224 RGB images, so we are resizing the input image to this dimension.
    let resized =
        image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    })
    .into();

    let result = model.run(tvec!(image)).unwrap();
    let best = result[0]
        .to_array_view::<f32>()
        .unwrap()
        .iter()
        .cloned()
        .zip(1..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    return best.unwrap().1;
}

/// If running in Node's WASI runtime, a `_start` function
/// is required for instantiating the module.
#[no_mangle]
pub unsafe extern "C" fn _start() {}
