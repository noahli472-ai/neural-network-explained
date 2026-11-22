// Entry point: load the exported training data, then start the three stations.

import { initDigits } from "./digits.js";
import { initNeuron } from "./neuron.js";
import { initHero, initPlayground } from "./playground.js";

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

function showError(error) {
  const box = document.querySelector("[data-load-error]");
  box.hidden = false;
  box.querySelector("code").textContent = String(error.message || error);
}

async function main() {
  try {
    const [playground, digits] = await Promise.all([
      loadJson("data/playground.json"),
      loadJson("data/digits.json"),
    ]);
    initHero(document.querySelector("#hero-figure"), playground);
    initNeuron(document.querySelector("#neuron"), playground.neuron);
    initPlayground(document.querySelector("#playground"), playground);
    initDigits(document.querySelector("#digits"), digits);

    const info = digits.info;
    const pct = (info.test_accuracy * 100).toFixed(1);
    document.querySelectorAll("[data-test-accuracy]").forEach((el) => (el.textContent = `${pct} %`));
    document.querySelectorAll("[data-n-params]").forEach((el) => {
      el.textContent = info.n_params.toLocaleString("en");
    });
  } catch (error) {
    showError(error);
  }
}

main();
