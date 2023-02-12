import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import pyfolio
from pyfolio import timeseries
from stable_baselines3.common.base_class import BaseAlgorithm

from environment.SingleStockTradingEnv import SingleStockTradingEnv

# Dow 30 constituents in 2021/10
DOW_30_TICKER = [
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "CRM",
    "VZ",
    "V",
    "WBA",
    "WMT",
    "DIS",
    "DOW",
]

def check_and_make_directories(directories: Union[str, List[str]]):
    if not isinstance(directories, List):
        directories = [directories]
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)

def calculate_sharpe_ratio(asset: List) -> float:
    df = pd.DataFrame(asset, columns=['asset'])
    df['daily_return'] = df['asset'].pct_change(1)
    if df['daily_return'].std() !=0:
        # Multiply (252 ** 0.5) to convert daily return sharpe ratio to annually return sharpe ratio
        sharpe = (252**0.5) * df['daily_return'].mean() / df['daily_return'].std()
        return sharpe
    else:
        return np.nan

def simulate_trading(env, model: BaseAlgorithm) -> Tuple[List[float], List[Any], List[float]]:
        """Simulate trading with model in env."""
        action_memory = []
        reward_memory = []

        dones = False
        obs = env.reset()

        while not dones:
            action, _ = model.predict(obs, deterministic=True)
            action_memory.append(action)
            obs, r, dones, _ = env.step(action)
            reward_memory.append(r)
        return env.asset_memory, action_memory, reward_memory

def simulate_trading_masked(env: SingleStockTradingEnv, model: BaseAlgorithm) -> Tuple[List[float], List[int]]:
        """Simulate trading with model in env."""
        actions_memory = []
        obs = env.reset()
        dones = False
        while not dones:
            masks = np.array(env.action_masks())
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            actions_memory.append(action)
            obs, r, dones, _ = env.step(action)
        return env.asset_memory, actions_memory


def get_daily_return(asset: pd.Series) -> pd.Series:
    return asset.pct_change(1)

def backtest_stats(sr_return: pd.Series, verbose: bool = True) -> pd.DataFrame:
    '''Get backtest stats with pyfolio.timeseries.'''
    perf_stats_all = timeseries.perf_stats(
        returns=sr_return,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    if verbose:
        print(perf_stats_all)
    return perf_stats_all

def backtest_plot(test_returns: pd.Series, baseline_returns: pd.Series) -> None:
    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(
            returns=test_returns, benchmark_rets=baseline_returns, set_context=False
        )

def split_data(
        df_dict: Dict[str, pd.DataFrame], 
        test_start: str, 
        trade_start: str
        ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    df_dict_train = dict()
    df_dict_test = dict()
    df_dict_trade = dict()

    test_start = pd.to_datetime(test_start, format='%Y-%m-%d')
    trade_start = pd.to_datetime(trade_start, format='%Y-%m-%d')

    for tic, df in df_dict.items():
        df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
        df_dict_train[tic] = df.loc[df.date < test_start].sort_index(ascending=True).copy()
        df_dict_test[tic] = df.loc[(df.date >= test_start) & (df.date < trade_start)].sort_index(ascending=True).copy()
        df_dict_trade[tic] = df.loc[df.date >= trade_start].sort_index(ascending=True).copy()

    return df_dict_train, df_dict_test, df_dict_trade

class PruneCallback:
    '''
    Callback class for hyperparameter tuning with optuna. Prune while objective stop increasing.

    :param threshhold: tolerance for increase in sharpe ratio
    :param trail_number: minimum number of trials before pruning
    :param patience for the threshold
    '''
    def __init__(self, threshold: float, trial_number: int, patience: int):
        self.threshold = threshold
        self.trial_number  = trial_number
        self.patience = patience
        
        # Trials list for which threshold is reached
        self.cb_list = [] 

    def __call__(self, study: optuna.study, frozen_trial: optuna.Trial):
        # Store the best value in current trial
        study.set_user_attr("previous_best_value", study.best_value)
        
        # Minimum number of trials
        if frozen_trial.number > self.trial_number:
            previous_best_value = study.user_attrs.get("previous_best_value", None)
            # Check whether previous objective values have the same sign as the current one
            if previous_best_value * study.best_value >= 0:
                # Check for the threshold condition
                if abs(previous_best_value - study.best_value) < self.threshold: 
                    self.cb_list.append(frozen_trial.number)
                    # If threshold is achieved for the patience amount of time
                    if len(self.cb_list) > self.patience:
                        print(f'Study stops After {frozen_trial.number} trails, current objective value: {frozen_trial.value}')
                        print(f'Previous best values: {previous_best_value}, current best values: {study.best_value}')
                        study.stop()