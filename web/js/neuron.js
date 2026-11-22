// Station 1: a single neuron with three sliders, and a replay of gradient descent.

import { COLORS, DOMAIN, drawField, drawPoints, fitCanvas, prefersReducedMotion, toPx, toPy } from "./plot.js";
import { sigmoid } from "./nn.js";

const fmt = (v) => (v < 0 ? `−${Math.abs(v).toFixed(1)}` : v.toFixed(1));
const signed = (v) => (v < 0 ? `− ${Math.abs(v).toFixed(1)}` : `+ ${v.toFixed(1)}`);

export function initNeuron(root, data) {
  const canvas = root.querySelector("canvas");
  const sliders = {
    w1: root.querySelector("#w1"),
    w2: root.querySelector("#w2"),
    b: root.querySelector("#bias"),
  };
  const formula = root.querySelector("[data-formula]");
  const score = root.querySelector("[data-score]");
  const status = root.querySelector("[data-status]");
  const learnButton = root.querySelector("[data-learn]");
  const resetButton = root.querySelector("[data-reset]");
  const { x: xs, y: ys, frames } = data;
  let animation = null;

  const weights = () => ({
    w1: Number(sliders.w1.value),
    w2: Number(sliders.w2.value),
    b: Number(sliders.b.value),
  });

  function setWeights(w1, w2, b) {
    sliders.w1.value = w1;
    sliders.w2.value = w2;
    sliders.b.value = b;
  }

  function render() {
    const { w1, w2, b } = weights();
    const { ctx, width: size } = fitCanvas(canvas);
    ctx.clearRect(0, 0, size, size);
    drawField(ctx, size, (x, y) => sigmoid(w1 * x + w2 * y + b), 60);
    drawLine(ctx, size, w1, w2, b);
    drawPoints(ctx, size, xs, ys);

    let correct = 0;
    for (let i = 0; i < xs.length; i++) {
      const red = w1 * xs[i][0] + w2 * xs[i][1] + b > 0;
      if (red === (ys[i] === 1)) correct++;
    }
    for (const key of Object.keys(sliders)) {
      root.querySelector(`output[for="${sliders[key].id}"]`).textContent = fmt(Number(sliders[key].value));
    }
    formula.innerHTML =
      `z = ${fmt(w1)}·x<sub>1</sub> ${signed(w2)}·x<sub>2</sub> ${signed(b)}` +
      `<br>output = σ(z)`;
    score.textContent =
      correct === xs.length
        ? `All ${xs.length} points on the right side.`
        : `${correct} of ${xs.length} points on the right side.`;
  }

  function stop() {
    if (animation) cancelAnimationFrame(animation);
    animation = null;
    learnButton.disabled = false;
  }

  function showFrame(frame) {
    const [w1, w2, b] = frame.params;
    setWeights(w1, w2, b);
    status.textContent = `Training step ${frame.step} of ${frames[frames.length - 1].step}, error ${frame.loss.toFixed(3)}`;
    render();
  }

  function learn() {
    stop();
    learnButton.disabled = true;
    if (prefersReducedMotion()) {
      showFrame(frames[frames.length - 1]);
      learnButton.disabled = false;
      return;
    }
    const duration = 4500;
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      showFrame(frames[Math.min(frames.length - 1, Math.floor(t * frames.length))]);
      if (t < 1) animation = requestAnimationFrame(tick);
      else stop();
    };
    animation = requestAnimationFrame(tick);
  }

  for (const slider of Object.values(sliders)) {
    slider.addEventListener("input", () => {
      stop();
      status.textContent = "You are the training algorithm now.";
      render();
    });
  }
  learnButton.addEventListener("click", learn);
  resetButton.addEventListener("click", () => {
    stop();
    showFrame(frames[0]);
    status.textContent = "Back to the starting weights.";
  });
  new ResizeObserver(render).observe(canvas);
  showFrame(frames[0]);
  status.textContent = "Move the sliders, or press “Let it learn”.";
}

/** The neuron's decision line w1*x + w2*y + b = 0, clipped to the plot. */
function drawLine(ctx, size, w1, w2, b) {
  const pts = [];
  if (Math.abs(w2) > 1e-9) {
    for (const x of [-DOMAIN, DOMAIN]) pts.push([x, -(w1 * x + b) / w2]);
  } else if (Math.abs(w1) > 1e-9) {
    for (const y of [-DOMAIN, DOMAIN]) pts.push([-b / w1, y]);
  } else {
    return;
  }
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, size, size);
  ctx.clip();
  ctx.beginPath();
  ctx.moveTo(toPx(pts[0][0], size), toPy(pts[0][1], size));
  ctx.lineTo(toPx(pts[1][0], size), toPy(pts[1][1], size));
  ctx.strokeStyle = COLORS.graphite;
  ctx.lineWidth = 2;
  ctx.setLineDash([7, 5]);
  ctx.stroke();
  ctx.restore();
}
