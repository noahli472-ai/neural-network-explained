// Drawing helpers shared by the 2-D stations: the coloured decision field and the points.

export const COLORS = {
  ink: "#1f3fb5",
  red: "#d3263a",
  paper: "#f8fafb",
  graphite: "#262a31",
  marker: "#ffe45c",
};

const INK_RGB = [31, 63, 181];
const RED_RGB = [211, 38, 58];

/** Half-width of the plotted square: x and y run from -DOMAIN to +DOMAIN. */
export const DOMAIN = 1.15;

/**
 * Size a canvas to its CSS box on high-density screens and return the drawing context
 * plus the CSS size. The context is scaled so that drawing code works in CSS pixels.
 */
export function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const size = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height || rect.width));
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  if (canvas.width !== Math.round(size * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: size, height };
}

export const toPx = (v, size) => ((v + DOMAIN) / (2 * DOMAIN)) * size;
export const toPy = (v, size) => ((DOMAIN - v) / (2 * DOMAIN)) * size;

const fieldCanvas = typeof document !== "undefined" ? document.createElement("canvas") : null;

/**
 * Tint the plane by the network's opinion: blue where it says class 0, red where it says
 * class 1, stronger colour = more confident. probability(x, y) returns P(class 1).
 */
export function drawField(ctx, size, probability, resolution = 90) {
  fieldCanvas.width = resolution;
  fieldCanvas.height = resolution;
  const fctx = fieldCanvas.getContext("2d");
  const image = fctx.createImageData(resolution, resolution);
  for (let r = 0; r < resolution; r++) {
    const y = DOMAIN - ((r + 0.5) / resolution) * 2 * DOMAIN;
    for (let c = 0; c < resolution; c++) {
      const x = -DOMAIN + ((c + 0.5) / resolution) * 2 * DOMAIN;
      const p = probability(x, y);
      const rgb = p >= 0.5 ? RED_RGB : INK_RGB;
      const strength = Math.abs(p - 0.5) * 2;
      const i = (r * resolution + c) * 4;
      image.data[i] = rgb[0];
      image.data[i + 1] = rgb[1];
      image.data[i + 2] = rgb[2];
      image.data[i + 3] = Math.round(255 * (0.05 + 0.27 * strength));
    }
  }
  fctx.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(fieldCanvas, 0, 0, size, size);
}

/** Draw class 0 as filled blue dots and class 1 as red crosses (shape and colour differ). */
export function drawPoints(ctx, size, xs, ys) {
  const r = Math.max(3, size / 95);
  for (let i = 0; i < xs.length; i++) {
    const px = toPx(xs[i][0], size);
    const py = toPy(xs[i][1], size);
    if (ys[i] === 0) {
      ctx.beginPath();
      ctx.arc(px, py, r, 0, 2 * Math.PI);
      ctx.fillStyle = COLORS.ink;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = COLORS.paper;
      ctx.stroke();
    } else {
      ctx.lineCap = "round";
      ctx.strokeStyle = COLORS.paper;
      ctx.lineWidth = 4.5;
      cross(ctx, px, py, r);
      ctx.strokeStyle = COLORS.red;
      ctx.lineWidth = 2.2;
      cross(ctx, px, py, r);
    }
  }
}

function cross(ctx, x, y, r) {
  ctx.beginPath();
  ctx.moveTo(x - r, y - r);
  ctx.lineTo(x + r, y + r);
  ctx.moveTo(x + r, y - r);
  ctx.lineTo(x - r, y + r);
  ctx.stroke();
}

/** Frame index -> fraction helper for smooth replays: ease nothing, just clamp. */
export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** True if the visitor asked the system for less motion. */
export const prefersReducedMotion = () =>
  typeof window !== "undefined" &&
  window.matchMedia &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;
