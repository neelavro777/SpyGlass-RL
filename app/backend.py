"""
SpyGlass-RL Backend: data loading, PPO environment classes, and test-period runner.
Importable standalone without Streamlit.
"""

import pandas as pd
import numpy as np
import os, warnings
from gym_anytrading.envs import StocksEnv
from stable_baselines3 import PPO

warnings.filterwarnings('ignore')

# ── Paths ──
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BACKEND_DIR)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'SPY_2020_2025_daily.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'notebooks', 'models')

# ── Data Loading ──

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.ffill(inplace=True)
    return df


def prepare_features(df):
    """Compute three feature variants for the three experiments."""
    df_raw = df.copy()

    df_r = df.copy()
    df_r['Return'] = df_r['Close'].pct_change()
    df_r.dropna(inplace=True)

    df_f = df.copy()
    df_f['Return'] = df_f['Close'].pct_change()
    df_f['SMA_20'] = df_f['Close'].rolling(20).mean()
    df_f['Dist_SMA_20'] = (df_f['Close'] - df_f['SMA_20']) / df_f['SMA_20']
    df_f['Volatility_20'] = df_f['Return'].rolling(20).std()
    df_f['Rel_Volume'] = df_f['Volume'] / df_f['Volume'].rolling(20).mean()
    df_f.dropna(inplace=True)

    return df_raw, df_r, df_f


# ── Custom Environment Classes ──

class ReturnsOnlyEnv(StocksEnv):
    def _process_data(self):
        prices = self.df['Close'].to_numpy()
        features = self.df[['Return']].to_numpy()
        return prices, features


COST = 0.001


class CostAwareEnv(StocksEnv):
    def _process_data(self):
        prices = self.df['Close'].to_numpy()
        features = self.df[['Return', 'Dist_SMA_20', 'Volatility_20', 'Rel_Volume']].to_numpy()
        return prices, features

    def _calculate_reward(self, action):
        step_reward = super()._calculate_reward(action)
        new_pos = 1 if action == 1 else 0
        if self._position.value != new_pos:
            step_reward -= COST * self.prices[self._current_tick]
        return step_reward


# ── Model Loading ──

def load_models():
    models = {}
    configs = [
        ('exp1_raw_prices', 'frozen_agent', StocksEnv),
        ('exp2_returns_no_cost', 'reckless_trader', ReturnsOnlyEnv),
        ('exp3_features_cost', 'informed_trader', CostAwareEnv),
    ]
    for key, folder, env_class in configs:
        path = os.path.join(MODELS_DIR, folder, 'best_model.zip')
        if os.path.exists(path):
            models[key] = {'model': PPO.load(path, device='cpu'), 'env_class': env_class}
        else:
            models[key] = None
    return models


# ── Agent Runner ──

def run_agent(model, env_class, use_df, start_idx, end_idx, window_size=15):
    """Run a trained PPO policy on a selected date range.

    The environment receives a 15-day lookback before the selected period so
    the first visible decision has enough observation history. Returned rows are
    clipped to the selected range only.
    """
    if end_idx - start_idx < window_size:
        raise ValueError(f'Select at least {window_size} trading days.')
    if start_idx < window_size:
        raise ValueError(f'Start date needs at least {window_size} prior trading days.')

    frame_start = start_idx - window_size
    eval_df = use_df.iloc[frame_start:end_idx].copy()
    frame_bound = (window_size, len(eval_df))

    env = env_class(df=eval_df, frame_bound=frame_bound, window_size=window_size)
    obs, info = env.reset(seed=42)

    total_reward = 0.0
    trades = []
    positions = []
    actions = []
    rewards_per_step = []
    prices_seen = []
    dates_seen = []
    in_position = 0

    step = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        action_val = int(action)
        obs, reward, terminated, truncated, info = env.step(action)

        tick = env._current_tick
        if 0 <= tick < len(env.prices):
            prices_seen.append(float(env.prices[tick]))
        if tick < len(eval_df.index):
            dates_seen.append(eval_df.index[tick])
        else:
            dates_seen.append(None)

        pos_info = info.get('position', 0)
        pos_val = pos_info.value if hasattr(pos_info, 'value') else int(pos_info)

        is_visible = dates_seen[-1] is not None and dates_seen[-1] >= use_df.index[start_idx]
        if is_visible and pos_val != in_position:
            trades.append({
                'step': len(positions),
                'date': dates_seen[-1] if dates_seen else None,
                'price': prices_seen[-1] if prices_seen else 0,
                'action': 'BUY' if pos_val == 1 else 'SELL'
            })

        in_position = pos_val
        if is_visible:
            positions.append(pos_val)
            actions.append(action_val)
            rewards_per_step.append(float(reward))
            total_reward += float(reward)
        step += 1
        if terminated or truncated:
            break

    total_profit = float(info.get('total_profit', 1.0))
    visible_dates = [d for d in dates_seen if d is not None and d >= use_df.index[start_idx]]
    visible_prices = prices_seen[-len(visible_dates):] if visible_dates else []

    return {
        'prices': visible_prices,
        'dates': visible_dates,
        'positions': positions,
        'actions': actions,
        'rewards': rewards_per_step,
        'trades': trades,
        'total_reward': total_reward,
        'total_profit': total_profit,
        'trade_count': len(trades),
        'step_count': len(visible_dates),
        'days_in_market': int(sum(positions)),
    }


# ── Date Helper ──

def date_to_indices(df, start_date, end_date):
    """Convert date range to integer indices using the DataFrame's dates."""
    if isinstance(df.index, pd.DatetimeIndex):
        date_values = df.index
    elif 'Date' in df.columns:
        date_values = pd.to_datetime(df['Date'])
    else:
        date_values = pd.to_datetime(df.index, errors='coerce')

    if pd.isna(date_values).any():
        raise ValueError('Could not read dates from the selected data.')

    date_values = np.asarray(date_values, dtype='datetime64[ns]')
    start_idx = int(np.searchsorted(date_values, np.datetime64(pd.Timestamp(start_date))))
    end_idx = int(np.searchsorted(date_values, np.datetime64(pd.Timestamp(end_date)), side='right'))
    if start_idx >= end_idx:
        raise ValueError('Start date must be before end date')
    return start_idx, end_idx
