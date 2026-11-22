# Running the project in VS Code (Windows)

A cheat sheet: one-time setup, the commands to run, and fixes for the usual errors.

---

## Quick start (after the one-time setup)

1. Open VS Code → *File → Open Folder* → the `neural-network-explained` folder
   (the one that contains `README.md` and the `web` folder).
2. *Terminal → New Terminal*. The prompt must start with `(.venv)`.
   If not: `.venv\Scripts\Activate.ps1`
3. Start the website:

```powershell
python -m http.server --directory web
```

4. Open <http://localhost:8000> in the browser. Stop the server with `Ctrl+C`.

To show the page on your phone in the same Wi-Fi, use your laptop's IP address instead of
`localhost` (find it with `ipconfig`, e.g. `http://192.168.0.23:8000`). Windows may ask
whether Python may accept connections in private networks: allow it.

---

## All commands

| Command | What it does |
|---|---|
| `python -m http.server --directory web` | Serves the website on <http://localhost:8000> |
| `python examples/quickstart.py` | Gradient check, then one neuron vs. a deeper network on the spiral |
| `python examples/classify_drawing.py` | Classifies a synthetic 7 with the shipped digit model |
| `python scripts/build_playground.py` | Re-records the training runs for stations 1 and 2 |
| `python scripts/train_digits.py` | Retrains the digit model (downloads MNIST on first run, a few minutes) |
| `python scripts/make_fixtures.py` | Refreshes the test fixtures after retraining |
| `pytest` | Runs the Python tests |
| `npm test` | Runs the JavaScript tests (needs Node.js 22) |
| `ruff check .` | Linting |

Opening `web/index.html` by double-click does **not** work: the page loads its data files
with `fetch`, which browsers block for files opened from disk. Always use the server.

---

## One-time setup

### 1. Allow PowerShell scripts (once per Windows user)

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Confirm with `J` (German Windows) or `Y` (English Windows).

### 2. Get the code

```powershell
cd "$HOME\Desktop\Coding Projects"
git clone https://github.com/noahli472-ai/neural-network-explained.git
cd neural-network-explained
```

Without git: on GitHub *Code → Download ZIP*, unzip, and `cd` into the folder that directly
contains `README.md` (ZIP downloads often create a nested folder of the same name).

### 3. Create the virtual environment and install

Run these **inside the project folder**:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
```

Check: `.venv\Scripts\python.exe examples\quickstart.py` prints a gradient check and two
accuracies.

### 4. Connect VS Code to the environment

1. *File → Open Folder* → the project folder.
2. Install the **Python** extension (Microsoft) if VS Code asks or it is missing.
3. `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the entry with `.venv`
   (`.\.venv\Scripts\python.exe`). If it is not listed: *Enter interpreter path…* →
   `.venv\Scripts\python.exe`.
4. Close old terminals (trash icon) and open a new one. It should now start with `(.venv)`.

### 5. Optional: Node.js for the JavaScript tests

Install Node.js 22 LTS from <https://nodejs.org>, open a new terminal, run `npm test`.
The website itself does not need Node.

---

## Troubleshooting

| Error | Cause → fix |
|---|---|
| Page shows "The page could not load its data" | The page was opened from disk. Start the server (quick start) and open <http://localhost:8000>. |
| `OSError: [WinError 10048]` when starting the server | Port 8000 is in use. Use another port: `python -m http.server 8001 --directory web`. |
| `[WinError 5] Access denied: 'C:\WINDOWS\System32\.venv'` | The terminal is in the Windows system folder. Open the project folder in VS Code (step 4) or `cd` into it. Do not run VS Code as administrator. |
| `...Activate.ps1 cannot be loaded because running scripts is disabled` | Run step 1 (`Set-ExecutionPolicy ...`). |
| `ModuleNotFoundError: No module named 'numpy'` / `'scipy'` | Packages are not installed in the Python that runs. Run step 3 in the project folder and select the `.venv` interpreter (step 4). |
| `ModuleNotFoundError: No module named 'scratchnet'` | Wrong folder (often the outer folder of a nested ZIP) or `pip install -e .` missing. |
| `RuntimeError: could not download train-images-idx3-ubyte.gz` | No internet or all MNIST mirrors unreachable. Try again later, or put the four `.gz` files into `%USERPROFILE%\.cache\scratchnet\mnist`. |
| Changes in `web/` do not show up | The browser cached the old files. Reload with `Ctrl+F5`. |

Which Python is running?

```powershell
python -c "import sys; print(sys.executable)"
```

It must end with `...\neural-network-explained\.venv\Scripts\python.exe`.

---

## macOS / Linux

Same steps with these differences: `python3 -m venv .venv`, activate with
`source .venv/bin/activate`, the interpreter path is `.venv/bin/python`, and the MNIST cache
is `~/.cache/scratchnet/mnist`.
