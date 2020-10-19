const fs = require("fs");
const { WASI } = require("wasi");
const path = require("path");

const module_bytes = fs.readFileSync("./model/optimized-wasi.wasm");
const model_bytes = fs.readFileSync("./model/mobilenet_v2_1.4_224_frozen.pb");
const label_bytes = fs.readFileSync("./model/labels.txt", "utf-8");
const testdata_dir = "./testdata";

const mod = new WebAssembly.Module(module_bytes);
const wasi = new WASI();

(async () => {
  const instance = await WebAssembly.instantiate(mod, {
    wasi_snapshot_preview1: wasi.wasiImport,
  });
  wasi.start(instance);

  const files = fs.readdirSync(testdata_dir);
  for (const f of files) {
    const img_bytes = fs.readFileSync(path.join(testdata_dir, f));

    console.log("\npredicting on file ", f);
    console.log(
      "prediction: ",
      getPrediction(model_bytes, img_bytes, instance) + "\n"
    );
  }
})();

function getPrediction(model_bytes, img_bytes, instance) {
  var start = new Date();
  var mptr = writeGuestMemory(model_bytes, instance);
  var iptr = writeGuestMemory(img_bytes, instance);

  let pred = instance.exports.infer_from_ptrs(
    mptr,
    model_bytes.length,
    iptr,
    model_bytes.length
  );
  console.log("inference time: ", new Date() - start + " ms");

  return getLabel(pred);
}

function writeGuestMemory(bytes, instance) {
  var len = bytes.byteLength;
  var ptr = instance.exports.alloc(len);
  var m = new Uint8Array(instance.exports.memory.buffer, ptr, len);
  m.set(new Uint8Array(bytes.buffer));

  return ptr;
}

function getLabel(pred) {
  var lines = label_bytes.split("\n");

  if (pred > lines.length) {
    throw new Error("cannot get predicted label");
  }

  return lines[pred - 1];
}
