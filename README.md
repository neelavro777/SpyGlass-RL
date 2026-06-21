# RL Trading Agent Project

This repository contains the structure for building an end-to-end Reinforcement Learning (RL) trading agent pipeline, integrating Stable-Baselines3, FastAPI, Streamlit, MLflow, and Docker.

## Directory Structure

```text
rl_project/
├── data/
│   └── raw/             # Raw dataset files (e.g., SPY_2020_2025_daily.csv)
├── notebooks/           # Jupyter notebooks for prototyping and EDA
├── scripts/             # Python scripts for training, utility functions, etc.
└── README.md            # Project overview and execution plan
```

## Getting Started

### 1. Dataset Generation
Generate the volatile 5-year S&P 500 (SPY) daily dataset using `yfinance` and save it to `data/raw/SPY_2020_2025_daily.csv`.

Here is the data retrieval script for your reference:
```python
import yfinance as yf
import pandas as pd

# Fetch the data for the volatile 5-year period
ticker = "SPY"
data = yf.download(ticker, start="2020-01-01", end="2025-01-01")

# Clean up and save to a CSV file
data.reset_index(inplace=True)
data.to_csv("data/raw/SPY_2020_2025_daily.csv", index=False)

print("Dataset saved to data/raw/SPY_2020_2025_daily.csv")
```

### 2. Architecture & Microservices
* **Service A: MLflow (Lab Notebook)** - Experiment tracking for Actor-Critic hyperparameters, logging returns, Sharpe ratios, and storing the trained model artifacts.
* **Service B: Training Environment (Gym)** - A custom or wrapper gym environment (e.g., `gym-anytrading` or `FinRL`) to train the Stable-Baselines3 model.
* **Service C: FastAPI Backend (Broker)** - Simple REST API hosting the trained agent model for serving buy/sell predictions.
* **Service D: Streamlit Frontend (Dashboard)** - Interactive data visualization using Plotly for displaying performance, trade signals, and historical stock trends.
