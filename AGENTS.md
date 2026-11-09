# AGENTS.md

## Environment

- Python and all Python dependencies are managed by `uv`; there is no Nix flake, direnv, or dev shell. Run `uv`/`python` directly.
- `uv` must already be on `PATH` (installed by the user, e.g. via the standalone installer); do not add Nix or direnv wrappers back.
- The machine is low-spec: do not run the full training pipeline here. Use Google Colab/Kaggle or another machine.
- Setup after cloning or changing dependencies: `uv sync` (creates `.venv`, uses Python 3.12 from `.python-version`, installs from `uv.lock`).
- The dataset is the 144 MB Kaggle `creditcard.csv` at the repo root. It is gitignored; don't commit it or rewrite it.

## Commands

- Install/sync dependencies:
  ```bash
  uv sync
  ```
- Run the full pipeline from the repo root:
  ```bash
  uv run python -m fraud_detection --data creditcard.csv --output-dir outputs
  ```
- Add or update dependencies:
  ```bash
  uv add nama-package
  uv lock
  ```
  Commit `uv.lock`; don't hand-edit it.
- Build the report PDF (Typst via tinymist; there is no standalone `typst` binary):
  ```bash
  tinymist compile LAPORAN.typ LAPORAN.pdf
  ```
- There is no test, lint, typecheck, or CI configuration. Don't assume pytest/ruff/mypy exist; add config explicitly if introducing them.

## Layout

- `fraud_detection/` is the package; `uv run python -m fraud_detection` is the entrypoint (`__main__.py`). Modules: `data.py` (load/validate), `preprocessing.py` (split + `Amount` scaling), `modeling.py` (baseline vs SMOTE), `evaluation.py` (metrics/plots).
- The distribution name is `fraud-detection` while the import package is `fraud_detection`. Hatchling packages `fraud_detection` explicitly; update `[tool.hatch.build.targets.wheel]` when adding packages.
- `Time` is intentionally dropped: it is seconds since the first transaction, not a stable calendar feature. `V1..V28` are already PCA-transformed and are not rescaled.
- Primary metric is **AUPRC / average precision**, not accuracy, because fraud is 0.172% of the data. Keep accuracy and confusion matrix for the assignment, but interpret them alongside precision/recall/F1.
- Outputs are written to `outputs/` (`metrics.json`, `classification_report.txt`, `results_table.md`, PNG plots). The report source is `LAPORAN.typ` (Typst); its results table and figures come from `outputs/results_table.md` and the PNG plots after a real run. Build the PDF with `tinymist compile LAPORAN.typ LAPORAN.pdf` (there is no standalone `typst` binary; keep the document free of network packages so it compiles offline).
