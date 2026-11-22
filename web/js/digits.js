// Station 3: draw a digit, watch it pass through the network layer by layer.

import { COLORS, fitCanvas } from "./plot.js";
import { forwardAll, loadModel, softmax } from "./nn.js";
import { IMAGE_SIZE, preprocessDrawing } from "./preprocess.js";

const PAD = 280; // internal resolution of the drawing pad
const STROKE = 22; // pen width in pad pixels (about MNIST's stroke-to-size ratio)
const GRID = 8; // first hidden layer shown as an 8 x 8 grid

/** Render one first-layer neuron's 784 weights as a 28x28 picture (blue +, red -). */
function weightTemplate(layer, j) {
  const canvas = document.createElement("canvas");
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE);
  let maxAbs = 1e-9;
  for (let i = 0; i < layer.nIn; i++) maxAbs = Math.max(maxAbs, Math.abs(layer.W[i * layer.nOut + j]));
  const ink = [31, 63, 181];
  const red = [211, 38, 58];
  for (let i = 0; i < layer.nIn; i++) {
    const w = layer.W[i * layer.nOut + j] / maxAbs;
    const rgb = w >= 0 ? ink : red;
    const a = Math.min(1, Math.abs(w) * 1.3);
    // blend onto paper so the tiles are opaque
    img.data[i * 4] = Math.round(248 + (rgb[0] - 248) * a);
    img.data[i * 4 + 1] = Math.round(250 + (rgb[1] - 250) * a);
    img.data[i * 4 + 2] = Math.round(251 + (rgb[2] - 251) * a);
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

export function initDigits(root, json) {
  const model = loadModel(json);
  const first = model[0];
  const templates = Array.from({ length: first.nOut }, (_, j) => weightTemplate(first, j));

  const pad = root.querySelector("[data-pad]");
  const seen = root.querySelector("[data-seen]");
  const hidden = root.querySelector("[data-hidden]");
  const second = root.querySelector("[data-second]");
  const bars = root.querySelector("[data-bars]");
  const guess = root.querySelector("[data-guess]");
  const detail = root.querySelector("[data-detail]");
  const detailCanvas = detail.querySelector("canvas");
  const detailText = detail.querySelector("p");
  const clearButton = root.querySelector("[data-clear]");

  pad.width = PAD;
  pad.height = PAD;
  const pctx = pad.getContext("2d", { willReadFrequently: true });
  let input = new Float32Array(IMAGE_SIZE * IMAGE_SIZE);
  let acts = forwardAll(model, input);
  let selected = null;
  let drawing = false;
  let last = null;
  let pending = false;

  for (let d = 0; d < 10; d++) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="digit">${d}</span><span class="bar"><span></span></span><span class="pct"></span>`;
    bars.appendChild(li);
  }

  function padPoint(event) {
    const rect = pad.getBoundingClientRect();
    return [((event.clientX - rect.left) / rect.width) * PAD, ((event.clientY - rect.top) / rect.height) * PAD];
  }

  function stroke(from, to) {
    pctx.strokeStyle = COLORS.ink;
    pctx.lineWidth = STROKE;
    pctx.lineCap = "round";
    pctx.lineJoin = "round";
    pctx.beginPath();
    pctx.moveTo(from[0], from[1]);
    pctx.lineTo(to[0], to[1]);
    pctx.stroke();
    schedule();
  }

  pad.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    pad.setPointerCapture(e.pointerId);
    drawing = true;
    last = padPoint(e);
    stroke(last, last);
  });
  pad.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    const p = padPoint(e);
    stroke(last, p);
    last = p;
  });
  const end = () => {
    drawing = false;
    last = null;
  };
  pad.addEventListener("pointerup", end);
  pad.addEventListener("pointercancel", end);

  clearButton.addEventListener("click", () => {
    pctx.clearRect(0, 0, PAD, PAD);
    schedule();
  });

  function schedule() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      predict();
    });
  }

  function predict() {
    const rgba = pctx.getImageData(0, 0, PAD, PAD).data;
    const ink = new Float32Array(PAD * PAD);
    for (let i = 0; i < ink.length; i++) ink[i] = rgba[i * 4 + 3] / 255;
    input = preprocessDrawing(ink, PAD, PAD);
    acts = forwardAll(model, input);
    render();
  }

  function render() {
    renderSeen();
    renderHidden();
    renderSecond();
    renderOutput();
    renderDetail();
  }

  function renderSeen() {
    const ctx = seen.getContext("2d");
    seen.width = IMAGE_SIZE;
    seen.height = IMAGE_SIZE;
    const img = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE);
    for (let i = 0; i < input.length; i++) {
      const v = input[i];
      img.data[i * 4] = Math.round(248 + (31 - 248) * v);
      img.data[i * 4 + 1] = Math.round(250 + (63 - 250) * v);
      img.data[i * 4 + 2] = Math.round(251 + (181 - 251) * v);
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  function tileGeometry(width) {
    const gap = Math.max(3, width / 70);
    const tile = (width - gap * (GRID - 1)) / GRID;
    return { gap, tile };
  }

  function renderHidden() {
    const { ctx, width } = fitCanvas(hidden);
    const { gap, tile } = tileGeometry(width);
    const a = acts[2];
    const maxA = Math.max(1e-9, ...a);
    const empty = isEmpty();
    ctx.clearRect(0, 0, width, width);
    ctx.imageSmoothingEnabled = true;
    for (let j = 0; j < a.length; j++) {
      const x = (j % GRID) * (tile + gap);
      const y = Math.floor(j / GRID) * (tile + gap);
      const level = empty ? 0 : a[j] / maxA;
      ctx.drawImage(templates[j], x, y, tile, tile);
      if (empty) {
        // nothing drawn yet: show the weights only, no activity
      } else if (level < 1) {
        ctx.fillStyle = `rgba(248, 250, 251, ${0.8 * (1 - level)})`;
        ctx.fillRect(x, y, tile, tile);
      }
      if (level > 0.05) {
        ctx.lineWidth = 1 + 3 * level;
        ctx.strokeStyle = COLORS.marker;
        ctx.strokeRect(x + ctx.lineWidth / 2, y + ctx.lineWidth / 2, tile - ctx.lineWidth, tile - ctx.lineWidth);
      }
      if (j === selected) {
        ctx.lineWidth = 2;
        ctx.strokeStyle = COLORS.graphite;
        ctx.strokeRect(x - 1, y - 1, tile + 2, tile + 2);
      }
    }
  }

  function renderSecond() {
    const { ctx, width, height } = fitCanvas(second);
    const a = acts[4];
    const maxA = Math.max(1e-9, ...a);
    const cols = 16;
    const rows = Math.ceil(a.length / cols);
    const cell = Math.min(width / cols, height / rows);
    const r = cell * 0.36;
    ctx.clearRect(0, 0, width, height);
    for (let j = 0; j < a.length; j++) {
      const cx = (j % cols) * cell + cell / 2;
      const cy = Math.floor(j / cols) * cell + cell / 2;
      const level = isEmpty() ? 0 : a[j] / maxA;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, 2 * Math.PI);
      ctx.fillStyle = level > 0.02 ? `rgba(255, 228, 92, ${0.25 + 0.75 * level})` : COLORS.paper;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = level > 0.02 ? COLORS.graphite : "#aab3bf";
      ctx.stroke();
    }
  }

  function isEmpty() {
    return input.every((v) => v === 0);
  }

  function renderOutput() {
    const empty = isEmpty();
    const probs = softmax(acts[acts.length - 1]);
    let best = 0;
    for (let d = 1; d < 10; d++) if (probs[d] > probs[best]) best = d;
    bars.querySelectorAll("li").forEach((li, d) => {
      const pct = empty ? 0 : probs[d] * 100;
      li.querySelector(".bar span").style.width = `${pct}%`;
      li.querySelector(".pct").textContent = empty ? "" : `${Math.round(pct)} %`;
      li.classList.toggle("best", !empty && d === best);
    });
    const p = Math.round(probs[best] * 100);
    const a = best === 8 ? "an" : "a";
    if (empty) guess.textContent = "Draw a digit from 0 to 9 in the box.";
    else if (p >= 80) guess.textContent = `That's ${a} ${best}. I'm ${p} % sure.`;
    else if (p >= 50) guess.textContent = `Probably ${a} ${best} (${p} %).`;
    else guess.textContent = `Hard to say. Maybe ${a} ${best}? Only ${p} % sure.`;
  }

  function renderDetail() {
    if (selected === null) {
      detail.hidden = true;
      return;
    }
    detail.hidden = false;
    const ctx = detailCanvas.getContext("2d");
    detailCanvas.width = IMAGE_SIZE * 4;
    detailCanvas.height = IMAGE_SIZE * 4;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(templates[selected], 0, 0, detailCanvas.width, detailCanvas.height);
    const a = acts[2][selected];
    let state = "Draw something to see how it reacts.";
    if (!isEmpty()) state = a > 0 ? `Right now it fires with strength ${a.toFixed(2)}.` : "Right now it is silent.";
    detailText.textContent =
      `Neuron ${selected + 1} of ${templates.length}. ` +
      `Ink on its blue pixels pushes it up, ink on its red pixels pushes it down. ${state}`;
  }

  hidden.addEventListener("click", (e) => {
    const rect = hidden.getBoundingClientRect();
    const { gap, tile } = tileGeometry(rect.width);
    const col = Math.floor((e.clientX - rect.left) / (tile + gap));
    const row = Math.floor((e.clientY - rect.top) / (tile + gap));
    const j = row * GRID + col;
    selected = j === selected || col < 0 || col >= GRID || row < 0 || row >= GRID ? null : j;
    render();
  });

  new ResizeObserver(render).observe(hidden);
  render();
}
