"""
SpyGlass-RL: Streamlit dashboard for PPO test-time trading decisions.
"""

import os
import sys

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend import (  # noqa: E402
    CostAwareEnv,
    ReturnsOnlyEnv,
    StocksEnv,
    date_to_indices,
    load_data,
    load_models,
    prepare_features,
    run_agent,
)

WINDOW_SIZE = 15

COLORS = {
    'exp1_raw_prices': '#8a8fd8',
    'exp2_returns_no_cost': '#ff6b5f',
    'exp3_features_cost': '#00c2ff',
}

EXPERIMENTS = [
    {
        'key': 'exp1_raw_prices',
        'label': '1. Frozen Agent: Raw Prices',
        'short': 'Raw prices',
        'description': (
            'The policy sees raw dollar prices, which are non-stationary. '
            'When test prices move outside the training distribution, it often '
            'chooses cash instead of taking market exposure.'
        ),
        'env': StocksEnv,
    },
    {
        'key': 'exp2_returns_no_cost',
        'label': '2. Reckless Trader: Returns, No Cost',
        'short': 'Returns, no cost',
        'description': (
            'The policy sees percentage returns, so the observation is more '
            'stationary. Without a transaction-cost penalty, it can still churn '
            'between positions too often.'
        ),
        'env': ReturnsOnlyEnv,
    },
    {
        'key': 'exp3_features_cost',
        'label': '3. Informed Trader: Features + Cost',
        'short': 'Features + cost',
        'description': (
            'The policy sees stationary features: return, trend distance, '
            'volatility, and relative volume. The cost penalty encourages fewer, '
            'more selective trades.'
        ),
        'env': CostAwareEnv,
    },
]


st.set_page_config(
    page_title='SpyGlass-RL: PPO Test Decisions',
    layout='wide',
    initial_sidebar_state='expanded',
)


@st.cache_data(show_spinner=False)
def cached_data():
    base_df = load_data()
    return (base_df, *prepare_features(base_df))


@st.cache_resource(show_spinner=False)
def cached_models():
    return load_models()


def format_pct(multiplier):
    return f'{(multiplier - 1.0) * 100:+.1f}%'


def build_agent_figure(result, selected_df, color):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=selected_df.index,
            y=selected_df['Close'],
            mode='lines',
            name='SPY Close',
            line=dict(color=color, width=2),
            showlegend=False,
        ),
    )

    buy_trades = [t for t in result['trades'] if t['action'] == 'BUY']
    sell_trades = [t for t in result['trades'] if t['action'] == 'SELL']

    if buy_trades:
        fig.add_trace(
            go.Scatter(
                x=[t['date'] for t in buy_trades],
                y=[t['price'] for t in buy_trades],
                mode='markers',
                name='Buy / enter SPY',
                marker=dict(
                    symbol='triangle-up',
                    size=13,
                    color='#3ddc97',
                    line=dict(width=1.5, color='white'),
                ),
                hovertemplate='BUY<br>%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>',
            ),
        )

    if sell_trades:
        fig.add_trace(
            go.Scatter(
                x=[t['date'] for t in sell_trades],
                y=[t['price'] for t in sell_trades],
                mode='markers',
                name='Sell / move to cash',
                marker=dict(
                    symbol='triangle-down',
                    size=13,
                    color='#ff4d6d',
                    line=dict(width=1.5, color='white'),
                ),
                hovertemplate='SELL<br>%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>',
            ),
        )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor='#0b132b',
        plot_bgcolor='#0b132b',
        font=dict(color='#e0e1dd', size=12),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    )
    fig.update_yaxes(title_text='SPY close price ($)', gridcolor='#1c2541')
    fig.update_xaxes(range=[selected_df.index[0], selected_df.index[-1]], gridcolor='#1c2541')

    return fig


df, df_raw, df_r, df_f = cached_data()
all_models = cached_models()
data_by_key = {
    'exp1_raw_prices': df_raw,
    'exp2_returns_no_cost': df_r,
    'exp3_features_cost': df_f,
}

st.sidebar.title('SpyGlass-RL')
st.sidebar.markdown('---')
st.sidebar.header('Test Period')

min_start = max(data.index[WINDOW_SIZE].date() for data in data_by_key.values())
max_end = min(data.index[-1].date() for data in data_by_key.values())
default_start = df.index[int(len(df) * 0.85)].date()
default_start = max(default_start, min_start)

start_date = st.sidebar.date_input(
    'Start date',
    value=default_start,
    min_value=min_start,
    max_value=max_end,
)
end_date = st.sidebar.date_input(
    'End date',
    value=max_end,
    min_value=min_start,
    max_value=max_end,
)

if start_date >= end_date:
    st.sidebar.error('End date must be after start date.')
    st.stop()

st.sidebar.markdown('---')
st.sidebar.header('PPO Models')
for exp in EXPERIMENTS:
    model_info = all_models.get(exp['key'])
    color = COLORS[exp['key']]
    status = 'loaded' if model_info is not None else 'not found'
    dot = '●' if model_info is not None else '○'
    st.sidebar.markdown(
        f'<span style="color:{color}">{dot}</span> {exp["short"]}: {status}',
        unsafe_allow_html=True,
    )

st.sidebar.markdown('---')
st.sidebar.caption('PPO + Stable-Baselines3 | SPY daily data 2020-2024')

st.title('PPO Agent Test-Time Decisions')
st.markdown(
    'Select a test period in the sidebar to see only the decisions each PPO '
    'agent made during that range. Green arrows mark buys, and red arrows mark sells.'
)

with st.spinner('Running PPO agents on the selected period...'):
    agent_results = {}
    for exp in EXPERIMENTS:
        key = exp['key']
        model_info = all_models.get(key)
        if model_info is None:
            st.warning(f'Model for {exp["label"]} was not found.')
            continue

        use_df = data_by_key[key]
        try:
            start_idx, end_idx = date_to_indices(use_df, start_date, end_date)
            if end_idx - start_idx < WINDOW_SIZE:
                raise ValueError(f'Select at least {WINDOW_SIZE} trading days.')
            result = run_agent(
                model_info['model'],
                exp['env'],
                use_df,
                start_idx,
                end_idx,
                window_size=WINDOW_SIZE,
            )
        except ValueError as exc:
            st.sidebar.error(str(exc))
            st.stop()

        selected_df = use_df.iloc[start_idx:end_idx]
        agent_results[key] = {
            **result,
            'label': exp['label'],
            'description': exp['description'],
            'selected_df': selected_df,
        }

if not agent_results:
    st.error('No PPO models are available to evaluate.')
    st.stop()

first_result = next(iter(agent_results.values()))
period_days = len(first_result['selected_df'])
st.caption(
    f'Selected range: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d} '
    f'({period_days} trading days)'
)

summary_rows = []
for exp in EXPERIMENTS:
    result = agent_results.get(exp['key'])
    if result is None:
        continue

    profit = result['total_profit']
    summary_rows.append(
        {
            'Agent': exp['short'],
            'Trades': result['trade_count'],
            'Profit': f'{profit:.4f}x',
            'Return': format_pct(profit),
            'Reward': f'{result["total_reward"]:+.1f}',
        }
    )

st.subheader('Selected-Period Summary')
st.dataframe(summary_rows, use_container_width=True, hide_index=True)

st.markdown('---')
st.subheader('Agent Decisions')

for exp in EXPERIMENTS:
    key = exp['key']
    result = agent_results.get(key)
    if result is None:
        continue

    profit = result['total_profit']

    st.markdown(f'### {result["label"]}')
    m1, m2, m3 = st.columns(3)
    m1.metric('Profit', f'{profit:.4f}x', format_pct(profit))
    m2.metric('Trades', result['trade_count'])
    m3.metric('Reward', f'{result["total_reward"]:+.1f}')

    st.plotly_chart(
        build_agent_figure(result, result['selected_df'], COLORS[key]),
        use_container_width=True,
    )
    st.caption(result['description'])

st.markdown('---')
st.caption(
    'Buy means the policy entered SPY; sell means it exited back to cash.'
)
