"""Train the main PPO trading agent with MLflow logging.

This script reproduces the successful notebook experiment:
stationary SPY features + transaction-cost-aware reward.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from gym_anytrading.envs import StocksEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]


class CostAwareSPYEnv(StocksEnv):
    """SPY trading env with stationary observations and a trade cost penalty."""

    trade_cost = 0.001

    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        self.processed_index = self.df.index[start:end]

        prices = self.df["Close"].to_numpy()[start:end]
        features = self.df[
            ["Return", "Dist_SMA_20", "Volatility_20", "Rel_Volume"]
        ].to_numpy()[start:end]
        return prices, features

    def _calculate_reward(self, action):
        step_reward = super()._calculate_reward(action)
        new_position = 1 if int(action) == 1 else 0
        if self._position.value != new_position:
            step_reward -= self.trade_cost * self.prices[self._current_tick]
        return step_reward


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_spy_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.ffill(inplace=True)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the stationary feature set from notebook_ppo.ipynb."""
    features = df.copy()
    features["Return"] = features["Close"].pct_change()
    features["SMA_20"] = features["Close"].rolling(20).mean()
    features["Dist_SMA_20"] = (features["Close"] - features["SMA_20"]) / features["SMA_20"]
    features["Volatility_20"] = features["Return"].rolling(20).std()
    features["Rel_Volume"] = features["Volume"] / features["Volume"].rolling(20).mean()
    features.dropna(inplace=True)
    return features


def chronological_bounds(df: pd.DataFrame, window_size: int, train_ratio: float, val_ratio: float):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if not (window_size < train_end < val_end < n):
        raise ValueError("Invalid split settings for dataset length/window size.")
    return {
        "train": (window_size, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n),
    }


def make_env(df: pd.DataFrame, frame_bound: tuple[int, int], window_size: int, monitor_path: Path | None = None):
    env = CostAwareSPYEnv(df=df, frame_bound=frame_bound, window_size=window_size)
    if monitor_path is not None:
        monitor_path.parent.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, str(monitor_path))
    return env


def evaluate_agent(model: PPO, env: CostAwareSPYEnv, seed: int) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    trades = []
    positions = []
    prices = []
    dates = []
    last_position = 0
    step_count = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        tick = env._current_tick
        date = env.processed_index[tick] if tick < len(env.processed_index) else None
        price = float(env.prices[tick])
        position_info = info.get("position", 0)
        position = position_info.value if hasattr(position_info, "value") else int(position_info)

        if position != last_position:
            trades.append(
                {
                    "date": date,
                    "price": price,
                    "action": "BUY" if position == 1 else "SELL",
                }
            )

        total_reward += float(reward)
        positions.append(position)
        prices.append(price)
        dates.append(date)
        last_position = position
        step_count += 1

        if terminated or truncated:
            break

    prices_arr = np.asarray(prices, dtype=float)
    positions_arr = np.asarray(positions, dtype=float)
    price_returns = pd.Series(prices_arr).pct_change().fillna(0.0).to_numpy()
    strategy_returns = price_returns * positions_arr
    equity_curve = np.cumprod(1.0 + strategy_returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve / running_max) - 1.0
    std = float(strategy_returns.std())
    sharpe = 0.0 if std == 0.0 else float(strategy_returns.mean() / std * np.sqrt(252))

    return {
        "total_reward": float(total_reward),
        "total_profit": float(info.get("total_profit", 1.0)),
        "return_pct": float((info.get("total_profit", 1.0) - 1.0) * 100.0),
        "trade_count": len(trades),
        "step_count": step_count,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "buy_hold_return_pct": float((prices_arr[-1] / prices_arr[0] - 1.0) * 100.0),
        "dates": dates,
        "prices": prices,
        "positions": positions,
        "trades": trades,
    }


def save_trade_plot(result: dict[str, Any], title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(result["dates"], result["prices"], color="#2563eb", linewidth=1.8, label="SPY Close")

    buys = [t for t in result["trades"] if t["action"] == "BUY"]
    sells = [t for t in result["trades"] if t["action"] == "SELL"]
    if buys:
        ax.scatter(
            [t["date"] for t in buys],
            [t["price"] for t in buys],
            marker="^",
            s=90,
            color="#16a34a",
            edgecolor="white",
            linewidth=0.8,
            label="Buy",
            zorder=3,
        )
    if sells:
        ax.scatter(
            [t["date"] for t in sells],
            [t["price"] for t in sells],
            marker="v",
            s=90,
            color="#dc2626",
            edgecolor="white",
            linewidth=0.8,
            label="Sell",
            zorder=3,
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("SPY close price ($)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def log_eval_history(log_dir: Path, output_path: Path) -> None:
    """Log EvalCallback history to MLflow and save a validation reward curve."""
    eval_path = log_dir / "evaluations.npz"
    if not eval_path.exists():
        print(f"No evaluation history found at {eval_path}")
        return

    evaluations = np.load(eval_path)
    timesteps = evaluations["timesteps"]
    results = evaluations["results"]
    ep_lengths = evaluations.get("ep_lengths")

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    for idx, timestep in enumerate(timesteps):
        step = int(timestep)
        mlflow.log_metric("val_mean_reward", float(mean_rewards[idx]), step=step)
        mlflow.log_metric("val_std_reward", float(std_rewards[idx]), step=step)
        if ep_lengths is not None:
            mlflow.log_metric("val_mean_episode_length", float(ep_lengths[idx].mean()), step=step)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timesteps, mean_rewards, color="#2563eb", marker="o", linewidth=2)
    ax.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color="#93c5fd",
        alpha=0.35,
        label="Reward std. dev.",
    )
    best_idx = int(np.argmax(mean_rewards))
    ax.scatter(
        [timesteps[best_idx]],
        [mean_rewards[best_idx]],
        color="#16a34a",
        s=90,
        zorder=3,
        label="Best validation reward",
    )
    ax.set_title("Validation Reward During PPO Training")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean validation reward")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    mlflow.log_artifact(str(output_path), artifact_path="plots")


def flatten_params(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    params = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            params.update(flatten_params(value, full_key))
        else:
            params[full_key] = value
    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the main PPO SPY trading agent.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "train_config.yaml",
        help="Path to training config.",
    )
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override PPO timesteps.")
    parser.add_argument("--run-name", type=str, default=None, help="Override MLflow run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.total_timesteps is not None:
        config["training"]["total_timesteps"] = args.total_timesteps
    if args.run_name is not None:
        config["mlflow"]["run_name"] = args.run_name

    seed = int(config["training"]["seed"])
    set_seed(seed)

    data_path = resolve_path(config["data"]["path"])
    model_dir = resolve_path(config["paths"]["model_dir"])
    log_dir = resolve_path(config["paths"]["log_dir"])
    artifact_dir = resolve_path(config["paths"]["artifact_dir"])
    mlruns_dir = resolve_path(config["mlflow"]["tracking_dir"])

    for path in [model_dir, log_dir, artifact_dir, mlruns_dir]:
        path.mkdir(parents=True, exist_ok=True)

    df_raw = load_spy_data(data_path)
    df = prepare_features(df_raw)
    CostAwareSPYEnv.trade_cost = float(config["environment"]["trade_cost"])

    window_size = int(config["environment"]["window_size"])
    splits = chronological_bounds(
        df=df,
        window_size=window_size,
        train_ratio=float(config["data"]["train_ratio"]),
        val_ratio=float(config["data"]["val_ratio"]),
    )

    train_env = DummyVecEnv(
        [
            lambda: make_env(
                df,
                splits["train"],
                window_size,
                log_dir / "train.monitor.csv",
            )
        ]
    )
    val_env = DummyVecEnv(
        [
            lambda: make_env(
                df,
                splits["val"],
                window_size,
                log_dir / "val.monitor.csv",
            )
        ]
    )
    test_env = make_env(df, splits["test"], window_size)

    best_model_dir = model_dir / "best"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(log_dir),
        eval_freq=int(config["training"]["eval_freq"]),
        deterministic=True,
        render=False,
    )

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(mlruns_dir.as_uri())
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
        mlflow.log_params(flatten_params(config))
        mlflow.log_params(
            {
                "dataset_rows_raw": len(df_raw),
                "dataset_rows_after_features": len(df),
                "train_start": str(df.index[splits["train"][0]].date()),
                "train_end": str(df.index[splits["train"][1] - 1].date()),
                "val_start": str(df.index[splits["val"][0]].date()),
                "val_end": str(df.index[splits["val"][1] - 1].date()),
                "test_start": str(df.index[splits["test"][0]].date()),
                "test_end": str(df.index[splits["test"][1] - 1].date()),
            }
        )

        model = PPO(
            policy=config["model"]["policy"],
            env=train_env,
            verbose=int(config["training"]["verbose"]),
            device=config["training"]["device"],
            seed=seed,
            learning_rate=float(config["model"]["learning_rate"]),
            n_steps=int(config["model"]["n_steps"]),
            batch_size=int(config["model"]["batch_size"]),
            gamma=float(config["model"]["gamma"]),
            gae_lambda=float(config["model"]["gae_lambda"]),
            clip_range=float(config["model"]["clip_range"]),
            ent_coef=float(config["model"]["ent_coef"]),
        )

        total_timesteps = int(config["training"]["total_timesteps"])
        print(f"Training PPO informed_trader for {total_timesteps} timesteps...")
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)

        best_model_path = best_model_dir / "best_model.zip"
        if not best_model_path.exists():
            best_model_path = model_dir / "final_model.zip"
            model.save(best_model_path)

        export_model_path = model_dir / "best_model.zip"
        shutil.copy2(best_model_path, export_model_path)
        best_model = PPO.load(export_model_path, device=config["training"]["device"])

        metrics = evaluate_agent(best_model, test_env, seed=seed)
        plot_path = artifact_dir / "training_result.png"
        save_trade_plot(metrics, "PPO Informed Trader: Held-Out Test Decisions", plot_path)
        val_curve_path = artifact_dir / "validation_reward_curve.png"
        log_eval_history(log_dir, val_curve_path)

        mlflow.log_metrics(
            {
                "test_total_reward": metrics["total_reward"],
                "test_total_profit": metrics["total_profit"],
                "test_return_pct": metrics["return_pct"],
                "test_trade_count": metrics["trade_count"],
                "test_step_count": metrics["step_count"],
                "test_sharpe": metrics["sharpe"],
                "test_max_drawdown": metrics["max_drawdown"],
                "test_buy_hold_return_pct": metrics["buy_hold_return_pct"],
            }
        )
        mlflow.log_artifact(str(export_model_path), artifact_path="model")
        mlflow.log_artifact(str(plot_path), artifact_path="plots")
        mlflow.log_artifact(str(args.config), artifact_path="config")

        metrics_path = artifact_dir / "test_metrics.json"
        serializable_metrics = {
            key: value
            for key, value in metrics.items()
            if key not in {"dates", "prices", "positions", "trades"}
        }
        metrics_path.write_text(json.dumps(serializable_metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

    print("\nTraining complete.")
    print(f"Best model: {display_path(export_model_path)}")
    print(f"MLflow runs: {display_path(mlruns_dir)}")
    print(
        "Test metrics: "
        f"profit={metrics['total_profit']:.4f}x, "
        f"return={metrics['return_pct']:+.1f}%, "
        f"trades={metrics['trade_count']}, "
        f"reward={metrics['total_reward']:+.1f}"
    )


if __name__ == "__main__":
    main()
