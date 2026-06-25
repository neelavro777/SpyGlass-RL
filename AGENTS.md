# AGENTS.md

Guidance for AI coding agents working in this repo. Read this before doing anything.

## Working with the user

This project is a **learning sandbox** for Reinforcement Learning (A2C, PPO) applied to finance and stock trading. The user is learning both RL algorithms and their application to trading.

- The user is **new to finance** but **genuinely interested** in the domain. Do not shy away from real financial terms (Sharpe ratio, drawdown, non-stationarity, long/short, volatility, momentum, etc.). Use the real vocabulary — that is the point.
- When a meaningful finance or RL concept comes up, give a **short 2–3 sentence explainer** in plain English. For minor terms, a one-line gloss is enough. Don't over-dumb it down; teach the term, don't avoid it.
- Use a **teaching tone**: explain *why* a step matters, not just *what* to do. Connect RL concepts to finance concepts where they overlap (reward = trade P&L, observation = market state the agent sees, policy = the trading strategy, action space = buy/sell/hold).
- Reference `findings/` and `data/description.md` — the user has already read these. They are teaching material, not code.

## Project shape

- The real entrypoint is **`notebooks/notebook.ipynb`** — a single end-to-end notebook (load → EDA → train → eval → feature-engineer → retrain). All work happens here.
- `scripts/` is an **empty stub**. Don't look for code there.
- `README.md` describes an aspirational microservices architecture (MLflow / FastAPI / Streamlit / Docker) that is **not implemented**. Trust the notebook over the README.
- `findings/` holds dated learning notes (see below). `data/description.md` is a plain-English glossary of the SPY dataset and finance terms.

## Environment setup

- Python **3.11.9** (`.python-version`, pyenv-style). Use the existing local `.venv/`.
- **There is no `requirements.txt` / `pyproject.toml` / lockfile.** Deps live only in `.venv`. To recreate an env from scratch, install the exact pinned versions:
  ```
  pip install gym-anytrading==2.0.0 gymnasium==1.3.0 stable-baselines3==2.9.0 pandas==3.0.3 plotly==6.8.0 numpy==2.4.6
  ```
- `yfinance` is referenced in the README for data regeneration but is **not** installed in the current venv — install it separately if regenerating the dataset.
- If you add or change a dependency, **also write a `requirements.txt`** so the env becomes reproducible.

## Running the notebook

- Run with **CWD = `notebooks/`**. The notebook uses relative paths: `../data/raw/SPY_2020_2025_daily.csv`, `./models/`, `./logs/`. Running from repo root breaks path resolution.
- Outputs are written into `notebooks/models/` and `notebooks/logs/`. These directories are **tracked in git** (`.gitignore` only excludes `.venv/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`) — be deliberate about committing regenerated artifacts.

## Modeling constraints

These are hard-earned. Read `findings/01_raw-price-trap.md` for the full story.

- The raw `gym-anytrading` `stocks-v0` env exposes raw `Close` prices and **known-fails** on the 2024–2025 test set: the agent freezes on Action 0 (sell/stay out) for all 189 days because test prices (>$500) are out-of-distribution vs training (~$250–$450). This is the "raw price trap." **Don't ship a model trained on raw prices.**
- Use the **`AdvancedSPYEnv`** subclass (Phase 8 of the notebook): raw prices still drive the profit calculation, but the **observation space is stationary features only** — `Return`, `Dist_SMA_20`, `Volatility_20`, `Relative_Volume`. Never feed raw `Close` to the policy.
- Split is **chronological, not shuffled**: 70/15/15 train/val/test, `window_size=15`. Shuffling leaks the future.
- Defaults: A2C + `MlpPolicy`, ~25k timesteps, `EvalCallback` saves best model by validation reward every 2000 steps to `notebooks/models/adv/best_model.zip`.

## Verification

- **No tests, linter, typechecker, or CI.** Verification = re-run the notebook's final test cell and confirm `total_profit > 1.0` on the held-out 2024–2025 test bound. A flat 1.0x with zero trades means the OOD-price regression returned.

## Findings folder

`findings/` is the user's learning journal. As the user explores RL, the dataset, or model behavior, write down discoveries here so the learning journey is reviewable at a bird's-eye level later.

- **Location:** `findings/`
- **Naming:** `NN_kebab-case-title.md` — zero-padded number for journey order, short kebab-case slug for the topic. Example: `01_raw-price-trap.md`, `02_reward-shaping-vs-stationary-features.md`.
- **When to write one:** when the user discovers something non-obvious — a model behavior, a data insight, an RL/finance concept that clicked, a bug and its fix, a comparison of approaches. If the user says "note this down" or "write a finding," do it naturally.
- **Structure each file:**
  - `#` title + one-line summary
  - dated or numbered so order is clear
  - short sections: observation → why it happens → fix/insight → what it means for the project
  - keep it self-contained and readable on its own
- Don't overwrite existing findings; add new ones with the next number. Keep prose tight — these are notes, not essays.
