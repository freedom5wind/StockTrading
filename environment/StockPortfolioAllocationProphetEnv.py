from typing import Dict, Tuple

import gym
from gym import spaces
import numpy as np
import pandas as pd
import scipy


class StockPortfolioManagementProphetEnv(gym.Env):
    '''Stock portfolio management environment with 
    prophet(state with future data under noise).

    :param dfs: Data frame of tickers. Should have multi-index with "date" and "tic"
        and a "change" column.
    :param transaction_cost: cost of transaction
    :param look_back: how many days to look back in observation space.
    :param look_ahead: how many days to look ahead in observation space.
    :param noise_amp: amplitude of Gaussian noise 
    :param noise_scale: standard deviation of Gaussian noise 
    '''

    metadata = {"render.modes": ["human"]}


    def __init__(
        self,
        df: pd.DataFrame,
        transaction_cost: float = 1e-3,
        look_back: int = 4,
        look_ahead: int = 1,
        noise_amp: float = 1.,
        noise_scale: float = 1.
    ) -> None:

        assert isinstance(df.index, pd.MultiIndex) and \
            'date' in df.index.names and 'tic' in df.index.names, \
                "Data frame should have multi-index with date and tic."
        assert 'change' in df.columns, "Data frame should have column 'change'."
        assert df.isna().sum().sum() == 0, f"{df.isna().sum().sum()} NaN found."
        assert look_ahead >= 1, "look_ahead should not be 0."

        self.df = df
        self.transaction_cost = transaction_cost
        self.look_back = look_back
        self.look_ahead = look_ahead
        self.noise_amp = noise_amp
        self.noise_scale = noise_scale

        self.df.reset_index(inplace=True)
        self.n_tickers = len(self.df['tic'].unique())
        self.feature_dims = len(self.df.columns) - 2    # exclude date and tic
        self.date = pd.Series(self.df.date.unique())
        self.df.set_index(['date', 'tic'], inplace=True)

        # Observation space includes n features for today and each previous day,
        # changes with noise for each day ahead and previous action.
        self.observation_shape = \
            ((self.look_back + 1) * self.feature_dims \
                + self.look_ahead \
                + self.n_tickers + 1,)
        # Action space includes cash(index 0) and n tickers.
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_tickers + 1,)) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape)

        self.reset()

    def reset(self) -> np.array:
        self.terminal = False
        # skip n days for look back
        self.day = self.look_back
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

        self.terminal = self.day + self.look_ahead >= self.df.shape[0] - 1

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
        assert self.day + self.look_ahead < self.df.shape[0]
        cur_dates = self.date[self.day-self.look_back : self.day+1]
        self.data = self.df.loc[cur_dates]
        future_dates = self.date[self.day+1 : self.day+1+self.look_ahead]
        self.prophecy = self.df.loc[future_dates]['change']

    def _update_state(self) -> None:
        self.state = self.data.to_numpy()
        assert self.state.shape[1] == self.feature_dims
        # Add Gaussian noise.
        prophecy = self.prophecy.to_numpy()
        assert len(prophecy.shape) == 1, f"prophecy.shape: {prophecy.shape}"
        prophecy += self.noise_amp * np.random.normal(scale=self.noise_scale)
        self.state = np.concatenate([self.state.flatten(), prophecy, self.portfolio], axis=0)

    @property
    def changes(self) -> np.array:
        cur_dates = self.date[self.day]
        changes = self.df.loc[cur_dates]['change'].to_numpy()
        return np.insert(changes, 0, 1)