# Code walkthrough

How the pieces fit together, in the order the data flows: Python trains and exports,
the browser loads the JSON and only runs forward passes.

```
scratchnet (Python)                      web/ (browser)
───────────────────                      ──────────────
datasets.py ─┐                           main.js  loads both JSON files
             ├─ playground.py ──► data/playground.json ──► neuron.js      (station 1)
layers.py    │   (record frames)                       └─► playground.js  (opening + station 2)
losses.py    │
network.py ──┤
optim.py     │
training.py  │
             └─ digits.py ──────► data/digits.json ─────► digits.js       (station 3)
mnist.py ────┘   (train MLP)                                 ├─ preprocess.js
augment.py, preprocessing.py                                 └─ nn.js (forward pass)
export.py  (JSON format for both files)
```

## The library: `src/scratchnet/`

**`layers.py`.** Every layer has `forward(x)` and `backward(grad_out)`. `forward` caches what
the backward pass needs (`Dense` keeps the input `x`, `ReLU` the mask `x > 0`, `Tanh` and
`Sigmoid` their output). `backward` turns dL/d(output) into dL/d(input) and, for `Dense`,
stores dL/dW and dL/db in `self.grads`.

**`losses.py`.** Losses take *logits*, not probabilities. `SoftmaxCrossEntropy` and
`BinaryCrossEntropy` both have the gradient (p − y)/N. `log_softmax` subtracts the row
maximum and `BinaryCrossEntropy` uses `np.logaddexp(0, z)`, so large logits never overflow.

**`network.py`.** `Sequential.forward` runs the layers in order, `backward` in reverse: this
is the chain rule. `mlp([784, 64, 32, 10], "relu")` builds Dense/activation pairs with He
initialisation for ReLU and Glorot for tanh/sigmoid. `forward_all` returns every
intermediate output; the website does the same in `nn.js` to show activations.

**`optim.py`.** Optimisers read `layer.grads` and update `layer.params` in place.
`Adam` keeps first and second moment estimates per parameter, applies bias correction and
optional decoupled weight decay (weights only, not biases).

**`training.py`.** `fit` shuffles once per epoch, runs `train_step` (forward, loss,
backward, update) per mini-batch, and optionally applies an augmentation function to each
batch.

**`gradcheck.py`.** `numerical_gradient` perturbs one entry at a time by ±ε and uses the
central difference. The tests compare it with backprop for every layer and loss.

## Data and preprocessing

**`datasets.py`.** Deterministic 2-D datasets in [-1, 1]². `make_linear` is separable by a
known line with a margin (station 1); circle, checkerboard (XOR), moons and spiral are not.

**`mnist.py`.** Downloads the four original files, checks SHA-256 and parses the IDX
format (magic bytes, dimensions, raw uint8 data).

**`preprocessing.py`.** `preprocess_drawing` reproduces MNIST's normalisation: bounding box
of the ink, area-averaging resize so the longer side is 20 px, then a shift that puts the
centre of mass at (13.5, 13.5). `round_half_up` replaces Python's banker's rounding so the
result matches JavaScript's `Math.round` exactly.

**`augment.py`.** Random rotation, shear and aspect change (`scipy.ndimage.affine_transform`),
random dilation/erosion for stroke thickness, rescaling to full ink, then
`preprocess_drawing`. The network therefore trains on exactly the kind of input the website
produces.

## Producing the website data

**`playground.py`.** `record_training` trains with full-batch updates and saves a frame
(step, loss, accuracy, all parameters rounded to 4 decimals) at the steps from
`snapshot_steps`. `build_neuron_station` starts a single neuron from deliberately wrong
weights and records plain gradient descent.

**`digits.py`.** `train_digit_model` trains the 784-64-32-10 network with Adam, halves the
learning rate twice and reports accuracy on the original and on the renormalised MNIST
test set.

**`export.py`.** Weight matrices become base64-encoded float32 (`encode_array`), which the
browser decodes with `new Float32Array(...)`. The small playground networks use one flat
list per frame (`flat_parameters`: W1 row-major, b1, W2, b2, ...).

## The website: `web/js/`

- **`nn.js`**: `loadModel`, `modelFromFlat`, `forwardAll`, `softmax`. Inference only.
- **`preprocess.js`**: line-by-line port of `preprocessing.py`.
- **`plot.js`**: decision-field rendering (evaluates the network on a 90 x 90 grid and
  scales the image up) and the dot/cross markers.
- **`neuron.js`**: sliders → weights → field, line and score; "Let it learn" replays the
  recorded frames.
- **`playground.js`**: dataset/network choice, replay with play/pause and a scrubber,
  network diagram with weight-coloured edges.
- **`digits.js`**: drawing pad (pointer events, 280 x 280 internal resolution), then
  preprocessing → forward pass → render input, 64 weight templates with activity,
  32 second-layer neurons and the probability bars.

## Tests

`tests/fixtures/parity.json` is written by `scripts/make_fixtures.py`. The Python test
`test_parity_fixtures.py` checks that Python still produces those outputs;
`tests/js/parity.test.mjs` checks that the JavaScript produces the same ones. Together they
guarantee that the website computes what the Python code computes.
