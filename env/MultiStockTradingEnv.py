from typing import Dict, Tuple

import gym
from gym import spaces
import numpy as np
import pandas as pd
import scipy


class MultiStockTradingEnv(gym.Env):
    '''Multiple stock trading environment with continuous action space. Also a
    portfolio allocation environment.

    :param dfs: Data frame of tickers. Should have multi-index with "date" and "tic"
        and a "change" column.
    :param transaction_cost: cost of transaction
    :param stack_frame: how many days for data present in observation space.
    '''

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        transaction_cost: float = 1e-3,
        stack_frame: int = 1,
        ) -> None:

        assert isinstance(df.index, pd.MultiIndex) and \
            'date' in df.index.names and 'tic' in df.index.names, \
                "Data frame should have multi-index with date and tic."
        assert 'change' in df.columns, "Data frame should have column 'change'."
        assert df.isna().sum().sum() == 0, f"{df.isna().sum().sum()} NaN found."

        self.df = df
        self.transaction_cost = transaction_cost
        self.stack_frame = stack_frame

        self.df.reset_index(inplace=True)
        self.n_tickers = len(self.df['tic'].unique())
        self.feature_dims = len(self.df.columns) - 2    # exclude date and tic
        self.date = pd.Series(self.df.date.unique())
        self.df.set_index(['date', 'tic'], inplace=True)

        # Observation space includes n features for each day and previous action.
        self.observation_shape = (self.stack_frame * self.feature_dims + self.n_tickers + 1,)
        # Action space includes cash(index 0) and n tickers.
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_tickers + 1,)) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape)

        self.reset()

    def reset(self) -> np.array:
        self.terminal = False
        # skip n days for frame stacking
        self.day = self.stack_frame - 1
        self.asset = 1.
        self.portfolio = np.zeros((self.n_tickers + 1,))
        self.portfolio[0] = 1.
        self.asset_memory = [self.asset]
        self.portfolio_memory = [self.portfolio]

        # update state
        self._update_data()
        self._update_state()

        return self.state

    def step(self, action: np.array) -> Tuple[np.array, float, bool, Dict]:
        assert len(action) == self.feature_dims + 1, f"Invalid action {action}."
        action = scipy.special.softmax(action)

        self.terminal = self.day >= self.df.shape[0] - 2

        transaction = np.abs(self.portfolio - action).sum() * self.transaction_cost
        self.asset *= 1 - transaction
        self.portfolio = action

        # state: s -> s+1
        self.day += 1
        self._update_data()
        self._update_state()

        self.asset *= 1 + np.dot(self.changes, self.portfolio)
        self.asset_memory.append(self.asset)

        reward = np.log2(self.asset_memory[-1] / self.asset_memory[-2])
        assert not np.isnan(reward), f'Nan reward with assets: {self.asset_memory}'

        return self.state, reward, self.terminal, {}

    def _update_data(self) -> None:
        # bound checking
        assert self.day < self.df.shape[0]
        cur_dates = self.date[self.day-self.stack_frame+1 : self.day+1]
        self.data = self.df.loc[cur_dates]

    def _update_state(self) -> None:
        self.state = self.data.to_numpy()
        assert self.state.shape[1] == self.feature_dims
        self.state = np.concatenate([self.state.flatten(), self.portfolio], axis=0)

    @property
    def changes(self) -> np.array:
        cur_dates = self.date[self.day]
        changes = self.df.loc[cur_dates]['change'].to_numpy()
        return np.insert(changes, 0, 1)