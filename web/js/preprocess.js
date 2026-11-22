// MNIST-style normalisation of a drawing. Mirrors src/scratchnet/preprocessing.py
// step by step; tests/js/parity.test.mjs checks that both give the same output.

export const IMAGE_SIZE = 28;
export const BOX_SIZE = 20;
export const INK_THRESHOLD = 0.05;

const roundHalfUp = (x) => Math.floor(x + 0.5);

/** 1-D area-averaging weights as a dense (nOut x nIn) row-major array. */
function areaWeights(nIn, nOut) {
  const scale = nIn / nOut;
  const w = new Float64Array(nOut * nIn);
  for (let o = 0; o < nOut; o++) {
    const start = o * scale;
    const end = (o + 1) * scale;
    for (let k = Math.floor(start); k < Math.min(Math.ceil(end), nIn); k++) {
      const overlap = Math.min(end, k + 1) - Math.max(start, k);
      if (overlap > 0) w[o * nIn + k] = overlap / scale;
    }
  }
  return w;
}

/** Resize a (h x w) image to (outH x outW) by area averaging: Wy * img * Wx^T. */
export function resizeArea(img, h, w, outH, outW) {
  const wy = areaWeights(h, outH);
  const wx = areaWeights(w, outW);
  const rows = new Float64Array(outH * w); // Wy * img
  for (let o = 0; o < outH; o++) {
    for (let k = 0; k < h; k++) {
      const weight = wy[o * h + k];
      if (weight === 0) continue;
      for (let c = 0; c < w; c++) rows[o * w + c] += weight * img[k * w + c];
    }
  }
  const out = new Float64Array(outH * outW); // rows * Wx^T
  for (let r = 0; r < outH; r++) {
    for (let o = 0; o < outW; o++) {
      let sum = 0;
      for (let c = 0; c < w; c++) {
        const weight = wx[o * w + c];
        if (weight !== 0) sum += weight * rows[r * w + c];
      }
      out[r * outW + o] = sum;
    }
  }
  return out;
}

/**
 * Turn a grey-scale drawing (values in [0, 1], 1 = ink, row-major h x w) into a
 * 28x28 MNIST-like image: crop to the ink, scale the longer side to 20 px and put
 * the centre of mass in the middle. Returns a Float32Array of length 784.
 */
export function preprocessDrawing(img, h, w, threshold = INK_THRESHOLD) {
  const out = new Float32Array(IMAGE_SIZE * IMAGE_SIZE);
  let top = h;
  let bottom = -1;
  let left = w;
  let right = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (Math.min(Math.max(img[y * w + x], 0), 1) > threshold) {
        if (y < top) top = y;
        if (y > bottom) bottom = y;
        if (x < left) left = x;
        if (x > right) right = x;
      }
    }
  }
  if (bottom < 0) return out;

  const ch = bottom - top + 1;
  const cw = right - left + 1;
  const crop = new Float64Array(ch * cw);
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      crop[y * cw + x] = Math.min(Math.max(img[(top + y) * w + left + x], 0), 1);
    }
  }
  const scale = BOX_SIZE / Math.max(ch, cw);
  const nh = Math.max(1, roundHalfUp(ch * scale));
  const nw = Math.max(1, roundHalfUp(cw * scale));
  const small = resizeArea(crop, ch, cw, nh, nw);

  let total = 0;
  let sy = 0;
  let sx = 0;
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const v = small[y * nw + x];
      total += v;
      sy += v * y;
      sx += v * x;
    }
  }
  const oy = roundHalfUp((IMAGE_SIZE - 1) / 2 - sy / total);
  const ox = roundHalfUp((IMAGE_SIZE - 1) / 2 - sx / total);
  for (let y = 0; y < nh; y++) {
    const ty = y + oy;
    if (ty < 0 || ty >= IMAGE_SIZE) continue;
    for (let x = 0; x < nw; x++) {
      const tx = x + ox;
      if (tx < 0 || tx >= IMAGE_SIZE) continue;
      out[ty * IMAGE_SIZE + tx] = Math.min(Math.max(small[y * nw + x], 0), 1);
    }
  }
  return out;
}
