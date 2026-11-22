// JavaScript side of the Python/JavaScript parity check: the website's forward pass and
// preprocessing must reproduce the outputs Python wrote to tests/fixtures/parity.json.
// Run with: npm test  (or: node --test tests/js/*.test.mjs)

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { decodeArray, forward, loadModel, modelFromFlat, softmax } from "../../web/js/nn.js";
import { preprocessDrawing } from "../../web/js/preprocess.js";

const root = new URL("../../", import.meta.url);
const fixtures = JSON.parse(readFileSync(new URL("tests/fixtures/parity.json", root), "utf8"));

function assertClose(actual, expected, tol, label) {
  assert.equal(actual.length, expected.length, `${label}: length`);
  for (let i = 0; i < expected.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    assert.ok(diff <= tol * (1 + Math.abs(expected[i])), `${label}[${i}]: ${actual[i]} vs ${expected[i]}`);
  }
}

test("network with relu, tanh and sigmoid layers", () => {
  const { model, inputs, expected } = fixtures.mixed_network;
  const m = loadModel(model);
  inputs.forEach((x, i) => assertClose(forward(m, x), expected[i], 1e-9, `row ${i}`));
});

test("flat-parameter network used by the playground", () => {
  const { sizes, activation, params, inputs, expected } = fixtures.flat_network;
  const m = modelFromFlat(sizes, activation, params);
  inputs.forEach((x, i) => assertClose(forward(m, x), expected[i], 1e-9, `row ${i}`));
});

test("shipped digit model", () => {
  const model = loadModel(JSON.parse(readFileSync(new URL("web/data/digits.json", root), "utf8")));
  const { inputs, expected } = fixtures.digits_model;
  inputs.forEach((enc, i) => assertClose(forward(model, decodeArray(enc)), expected[i], 1e-6, `digit ${i}`));
});

test("preprocessing matches Python", () => {
  for (const c of fixtures.preprocessing) {
    const [h, w] = c.image.shape;
    const out = preprocessDrawing(decodeArray(c.image), h, w);
    assertClose(out, decodeArray(c.expected), 1e-5, c.name);
  }
});

test("softmax is stable and normalised", () => {
  const p = softmax([1000, 0, -1000]);
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(Math.abs(p[0] - 1) < 1e-12);
});
