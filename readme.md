# Testing Google Colab (T4 GPU) from VS Code

A short guide for running a quick GPU sanity-check script — an MLP that
fits `y = e^x` over `x ∈ [0, 200]` — on a Google Colab T4 runtime, driven
from inside VS Code.

---

## 1. Why this setup

Colab gives you a free T4 GPU. VS Code gives you a real editor instead of
the browser notebook UI. The Colab extension bridges them: your notebook
stays local, the execution happens on a remote Colab VM with the T4.

Downside to know up front:

- The VM is **ephemeral**. Reconnect = fresh machine, all in-memory
  variables gone, all `pip install` gone. Files written to `/content/`
  are gone too unless you saved them to Drive.
- Idle timeout is roughly 90 minutes; hard cap is ~12 hours.
- "Runtime disconnected" is a normal state, not a bug.

---

## 2. Prerequisites

- VS Code (recent version)
- Python extension
- Jupyter extension
- Google Colab extension (`google.colab`), installed from the marketplace
- A Google account

---

## 3. Setup

1. In VS Code, open the Extensions panel and install **Colab**.
2. Sign in when prompted (a browser tab will open for OAuth).
3. Open or create a `.ipynb` file. VS Code will treat it as a notebook.
4. In the top-right of the notebook editor, click the kernel picker.
5. Choose **Colab** → **New Colab Server** → **GPU**.
   - If you don't see the GPU option, the account may already be using
     its GPU quota; try again later or use a different account.

Once connected, the status bar shows the Colab runtime. You can verify
the GPU from a cell:

```python
!nvidia-smi
```

You should see `Tesla T4` and ~15 GB of memory.

---

## 4. What the test script does

The script is a minimal but meaningful GPU exercise:

1. Confirms the runtime has a CUDA device.
2. Generates 200,000 samples of `y = e^x` for `x ∈ [0, 200]`.
3. Trains a 4-layer MLP (256 hidden units, SiLU) to predict `log y`,
   which is the linear function `x`.
4. Reports training throughput (steps/sec) and peak VRAM.
5. Plots the fit and the relative error.

The reason it fits `log y` instead of `y` directly: `e^200 ≈ 7.2e86`
overflows float32 (max ≈ 3.4e38), and MSE on the raw values would be
completely dominated by the large-`x` end. Working in log space turns
the problem into a well-conditioned linear regression.

---

## 5. Running it

Run the cells top to bottom. Expect:

- `!nvidia-smi` prints a T4.
- Data cell prints `y ∈ [1.00e+00, 7.23e+86]`.
- Training loss falls from `~8e-2` to `~1e-7` within a few dozen epochs.
- A log-scale plot where the predicted curve sits on top of `e^x`.
- A relative-error plot that rises with `x` (expected: 1e-4 → a few %).

To watch GPU utilization while training, open a terminal
(`Terminal → New Terminal`) and run:

```bash
watch -n 1 nvidia-smi
```

Do **not** put `!nvidia-smi -l 1` in a notebook cell — it blocks the
cell forever and the training cell will never run.

Note that this MLP is small enough that `nvidia-smi` will often show
0–10% utilization: the GPU finishes each step in ~2 ms and then waits
for Python. That is not a malfunction. If you want to actually saturate
the T4, scale the model up or move to a CNN.

---

## 6. Common issues

### `Runtime disconnected`
The VM was reclaimed (idle, quota, or browser tab suspended). Fix:
reconnect from the Colab status bar, re-run `!nvidia-smi`, and re-run
every cell from the top — nothing persists.

### `RuntimeError: mat1 and mat2 shapes cannot be multiplied`
Shape mismatch. The classic cause here is feeding a 1-D tensor
`(2000,)` into a `Linear(1, 256)` layer. `nn.Linear` treats it as
`(1, 2000)`. Fix: `.unsqueeze(-1)` on the input.

### Loss stuck at ~8.33e-2 and never moves
`8.33e-2 ≈ 1/12`, which is the variance of `U(0, 1)` — the loss of a
network that outputs a constant 0.5. The cause is a broadcast bug:
model output is `(B,)` but target is `(B, 1)`, so `MSELoss` computes a
`(B, B)` matrix of pairwise differences. The gradient only pulls every
prediction toward the global mean; no learning happens. Fix: make sure
`pred.shape == target.shape` before calling the loss.

Guard against this by asserting shapes once before the loop:

```python
assert model(u[:4]).shape == t[:4].shape
```

### `Glyph XXXXX missing from font(s) DejaVu Sans`
Matplotlib has no CJK glyphs by default, so Chinese labels render as
boxes. Either switch labels to English, or register a CJK font:

```python
import matplotlib, matplotlib.font_manager as fm
fm.fontManager.addfont('/tmp/NotoSansCJK.ttf')
matplotlib.rcParams['font.family'] = 'Noto Sans CJK SC'
matplotlib.rcParams['axes.unicode_minus'] = False
```

The registration is per-session; you'll need to redo it after every
reconnect.

### `!nvidia: event not found` in the terminal
You are in a bash shell, not a notebook cell. `!` is history expansion
in bash. Drop the `!` and run `nvidia-smi` (or `watch -n 1 nvidia-smi`).

---

## 7. Keeping the session alive

Free-tier Colab reclaims idle VMs. A common workaround is a snippet in
the browser devtools console that clicks the connect button periodically:

```javascript
function keepAlive() {
  document.querySelector("colab-connect-button")?.click();
}
setInterval(keepAlive, 60000);
```

This reduces — but does not eliminate — disconnects. It does nothing
about the 12-hour cap.

---

## 8. Cleanup

When finished: disconnect the runtime from the Colab status bar, or let
it idle out. Nothing on the VM needs to be cleaned up; it is destroyed
either way.
```
