// Station 2 and the opening figure: replay recorded training runs of small networks.

import { COLORS, drawField, drawPoints, fitCanvas, prefersReducedMotion } from "./plot.js";
import { forward, modelFromFlat, sigmoid } from "./nn.js";

const DATASET_LABELS = { circles: "Circle", xor: "Checkerboard", moons: "Moons", spiral: "Spiral" };

function probabilityFn(arch, params) {
  const model = modelFromFlat(arch.sizes, arch.activation, params);
  return (x, y) => sigmoid(forward(model, [x, y])[0]);
}

/** Draw one replay frame (field + points) into a square canvas. */
export function drawRun(canvas, data, datasetName, archName, frameIndex) {
  const arch = data.architectures[archName];
  const frames = data.runs[`${datasetName}/${archName}`].frames;
  const frame = frames[frameIndex];
  const dataset = data.datasets[datasetName];
  const { ctx, width: size } = fitCanvas(canvas);
  ctx.clearRect(0, 0, size, size);
  drawField(ctx, size, probabilityFn(arch, frame.params));
  drawPoints(ctx, size, dataset.x, dataset.y);
  return frame;
}

/** Nodes in columns, edges coloured by sign (blue +, red -) and sized by magnitude. */
function drawDiagram(canvas, arch, params) {
  const { ctx, width, height } = fitCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const sizes = arch.sizes;
  const colX = (i) => 28 + (i * (width - 56)) / (sizes.length - 1);
  const rowY = (n, j) => {
    const gap = Math.min(30, (height - 30) / Math.max(n - 1, 1));
    return height / 2 + (j - (n - 1) / 2) * gap;
  };
  let offset = 0;
  for (let l = 0; l < sizes.length - 1; l++) {
    const nIn = sizes[l];
    const nOut = sizes[l + 1];
    let maxAbs = 1e-9;
    for (let k = 0; k < nIn * nOut; k++) maxAbs = Math.max(maxAbs, Math.abs(params[offset + k]));
    for (let i = 0; i < nIn; i++) {
      for (let j = 0; j < nOut; j++) {
        const w = params[offset + i * nOut + j];
        ctx.beginPath();
        ctx.moveTo(colX(l), rowY(nIn, i));
        ctx.lineTo(colX(l + 1), rowY(nOut, j));
        ctx.strokeStyle = w >= 0 ? COLORS.ink : COLORS.red;
        ctx.globalAlpha = 0.25 + 0.75 * (Math.abs(w) / maxAbs);
        ctx.lineWidth = 0.6 + 3.4 * (Math.abs(w) / maxAbs);
        ctx.stroke();
      }
    }
    offset += nIn * nOut + nOut;
  }
  ctx.globalAlpha = 1;
  sizes.forEach((n, l) => {
    for (let j = 0; j < n; j++) {
      ctx.beginPath();
      ctx.arc(colX(l), rowY(n, j), 7, 0, 2 * Math.PI);
      ctx.fillStyle = COLORS.paper;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = COLORS.graphite;
      ctx.stroke();
    }
  });
  ctx.fillStyle = COLORS.graphite;
  ctx.font = "13px 'Atkinson Hyperlegible', system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("input", colX(0), height - 4);
  ctx.fillText("output", colX(sizes.length - 1), height - 4);
}

/** Play frames 0..n-1 over `duration` ms; onFrame(i) is called for each new index. */
function play(nFrames, duration, onFrame, onDone) {
  const start = performance.now();
  let last = -1;
  let handle = 0;
  const tick = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const i = Math.min(nFrames - 1, Math.floor(t * nFrames));
    if (i !== last) {
      onFrame(i);
      last = i;
    }
    if (t < 1) handle = requestAnimationFrame(tick);
    else onDone();
  };
  handle = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(handle);
}

export function initHero(root, data) {
  const canvas = root.querySelector("canvas");
  const button = root.querySelector("button");
  const frames = data.runs["spiral/large"].frames;
  let current = frames.length - 1;
  let cancel = null;
  const show = (i) => {
    current = i;
    drawRun(canvas, data, "spiral", "large", i);
  };
  const run = () => {
    if (cancel) cancel();
    if (prefersReducedMotion()) return show(frames.length - 1);
    cancel = play(frames.length, 6000, show, () => (cancel = null));
  };
  button.addEventListener("click", run);
  new ResizeObserver(() => show(current)).observe(canvas);
  run();
}

export function initPlayground(root, data) {
  const plot = root.querySelector("[data-plot]");
  const diagram = root.querySelector("[data-diagram]");
  const scrub = root.querySelector("[data-scrub]");
  const playButton = root.querySelector("[data-play]");
  const readout = root.querySelector("[data-readout]");
  const verdict = root.querySelector("[data-verdict]");
  let dataset = "circles";
  let arch = "neuron";
  let cancel = null;

  const frames = () => data.runs[`${dataset}/${arch}`].frames;

  function show(i) {
    scrub.value = i;
    const frame = drawRun(plot, data, dataset, arch, i);
    drawDiagram(diagram, data.architectures[arch], frame.params);
    const pct = Math.round(frame.accuracy * 100);
    readout.textContent = `Step ${frame.step.toLocaleString("en")}: ${pct} % of points classified correctly, error ${frame.loss.toFixed(3)}`;
    const last = i === frames().length - 1;
    if (!last) verdict.textContent = "";
    else if (pct === 100) verdict.textContent = "Solved. Every point is on the right side.";
    else if (arch === "neuron") verdict.textContent = "Stuck. A single neuron can only draw a straight line.";
    else verdict.textContent = "Not enough neurons to bend the boundary into this shape.";
  }

  function stop() {
    if (cancel) cancel();
    cancel = null;
    playButton.textContent = "Train";
  }

  function select() {
    stop();
    scrub.max = frames().length - 1;
    show(0);
  }

  function setupChoices(selector, onPick) {
    const buttons = root.querySelectorAll(selector);
    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        buttons.forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
        onPick(btn.value);
        select();
      });
    });
  }

  setupChoices("[data-dataset]", (v) => (dataset = v));
  setupChoices("[data-arch]", (v) => (arch = v));
  root.querySelectorAll("[data-dataset]").forEach((b) => {
    b.textContent = DATASET_LABELS[b.value];
  });

  playButton.addEventListener("click", () => {
    if (cancel) return stop();
    if (prefersReducedMotion()) return show(frames().length - 1);
    playButton.textContent = "Pause";
    const n = frames().length;
    const startAt = Number(scrub.value) >= n - 1 ? 0 : Number(scrub.value);
    const remaining = n - startAt;
    cancel = play(remaining, (remaining / n) * 5000, (i) => show(startAt + i), stop);
  });
  scrub.addEventListener("input", () => {
    stop();
    show(Number(scrub.value));
  });
  new ResizeObserver(() => show(Number(scrub.value))).observe(plot);
  select();
}
