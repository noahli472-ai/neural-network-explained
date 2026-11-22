// Forward pass of networks exported by scratchnet (Python). Inference only:
// all training happens in Python, the browser just evaluates the stored weights.

/** Decode {shape, data: base64 little-endian float32} into a Float32Array. */
export function decodeArray(encoded) {
  const binary = atob(encoded.data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

/** Turn the JSON written by scratchnet.export.export_network into a model object. */
export function loadModel(json) {
  return json.layers.map((layer) => {
    if (layer.type !== "dense") return { type: layer.type };
    const [nIn, nOut] = layer.W.shape;
    return { type: "dense", nIn, nOut, W: decodeArray(layer.W), b: decodeArray(layer.b) };
  });
}

/**
 * Build a model from one flat parameter list (W1 row-major, b1, W2, b2, ...),
 * the format of the recorded playground frames.
 */
export function modelFromFlat(sizes, activation, params) {
  const layers = [];
  let offset = 0;
  for (let i = 0; i < sizes.length - 1; i++) {
    const nIn = sizes[i];
    const nOut = sizes[i + 1];
    const W = Float64Array.from(params.slice(offset, offset + nIn * nOut));
    offset += nIn * nOut;
    const b = Float64Array.from(params.slice(offset, offset + nOut));
    offset += nOut;
    layers.push({ type: "dense", nIn, nOut, W, b });
    if (i < sizes.length - 2) layers.push({ type: activation });
  }
  return layers;
}

function dense(layer, x) {
  const { nIn, nOut, W, b } = layer;
  const out = new Float64Array(nOut);
  for (let j = 0; j < nOut; j++) out[j] = b[j];
  for (let i = 0; i < nIn; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = i * nOut;
    for (let j = 0; j < nOut; j++) out[j] += xi * W[row + j];
  }
  return out;
}

export const sigmoid = (z) => 0.5 * (1 + Math.tanh(0.5 * z));

const ACTIVATIONS = {
  relu: (v) => v.map((z) => (z > 0 ? z : 0)),
  tanh: (v) => v.map(Math.tanh),
  sigmoid: (v) => v.map(sigmoid),
};

/** Return [input, output of layer 1, output of layer 2, ...]; the last entry are the logits. */
export function forwardAll(model, input) {
  const outputs = [Float64Array.from(input)];
  for (const layer of model) {
    const x = outputs[outputs.length - 1];
    if (layer.type === "dense") outputs.push(dense(layer, x));
    else if (ACTIVATIONS[layer.type]) outputs.push(ACTIVATIONS[layer.type](x));
    else throw new Error(`unknown layer type ${layer.type}`);
  }
  return outputs;
}

/** Logits of the network for one input vector. */
export function forward(model, input) {
  const outputs = forwardAll(model, input);
  return outputs[outputs.length - 1];
}

/** Numerically stable softmax. */
export function softmax(logits) {
  const max = Math.max(...logits);
  const exps = Array.from(logits, (z) => Math.exp(z - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}
