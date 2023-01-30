import random
from typing import Dict, List, Tuple

import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch as th


class SingleStockTradingEnv(gym.Env):
    """A single stock trading environment for OpenAI gym with discrete action space
    
    :param dfs: Data frames of tickers. First column should be "date" and 
        should have column "close". All data frames should have the same columns
    :param initial_cash: initial amount of cash
    :param buy_cost_pct: cost for buying shares
    :param sell_cost_pct: cost for selling shares
    :param stack_frame: how many days for data present in observation space
    :param reward_bias: whether to adjust reward based on ticker's average daily return
    :param ignore_close: whether to return 'close' column to state
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        dfs: List[pd.DataFrame],
        initial_cash: float = 1000_000,
        buy_cost_pct: float = 1e-3,
        sell_cost_pct: float = 1e-3,
        stack_frame: int = 14,
        reward_bias: bool = True,
        ignore_close: bool = True,
    ):
        self.dfs = dfs
        self.initial_cash = initial_cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.stack_frame = max(1, stack_frame)
        self.reward_bias = 1 if reward_bias else 0
        self.ignore_close = ignore_close

        self.feature_dims = self.dfs[0].shape[1] - 1
        self.feature_dims -= 1 if self.ignore_close else 0
        self.observation_shape = ((self.stack_frame) * (self.feature_dims) + 1,) # append position state
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape)
        self.episode = 0
        self.reset()

    def reset(self) -> pd.DataFrame:
        # select data frame randomly and check its columns.
        self.df_index = random.randint(0, (len(self.dfs)) - 1)
        assert 'close' in self.df.columns, \
            f"{self.df_index}th data frame doesn't have column 'close'."
        assert self.df.columns[0] == 'date', \
            f"First column of {self.df_index}th data frame is not 'date'"

        self.terminal = False
        # skip n days for frame stacking
        self.day = self.stack_frame - 1
        self.share_num = 0
        self.account = self.initial_cash
        self.asset_memory = []
        self.episode += 1

        # update state
        self._update_data()
        self._update_state()
        self.log_cumulative_return = \
            np.log2(self.df['close'].iloc[-1] / self.price) / (self.df.shape[0] - 1 - self.day)

        return self.state

    def step(self, actions: int) -> Tuple[pd.DataFrame, float, bool, Dict]:
        # validate input action
        assert 0 <= actions <= 2, f"Invalid action {actions}."

        self.terminal = self.day >= self.df.shape[0] - 2

        assert self.action_masks()[actions], \
            f'Invalid action {actions} with {self.share_num} shares of stocks.'

        if actions == 0:
            self._sell(self.share_num)
        elif actions == 2:
            buying_share = np.floor(self.account / ( self.price * (1 + self.buy_cost_pct)))
            self._buy(buying_share)

        # state: s -> s+1
        self.day += 1
        self._update_data()
        self._update_state()
        self.asset_memory.append(self.asset)

        if self.day == self.stack_frame:
            # return 0 for the first day.
            reward = 0
        else:
            reward = np.log2(self.asset_memory[-1] / self.asset_memory[-2])
        reward -= self.reward_bias * self.log_cumulative_return

        assert not np.isnan(reward), f'Nan reward with assets: {self.asset_memory}'
        
        return self.state, reward, self.terminal, {}

    def render(self, mode="human", close=False) -> pd.DataFrame:
        return self.state

    def action_masks(self) -> List[bool]:
        if self.share_num > 0:
            return [True, True, False]
        else:
            return [False, True, True]

    def _update_data(self) -> None:
        # bound checking
        assert self.day < self.df.shape[0]

        self.data = self.df.iloc[self.day-self.stack_frame+1 : self.day+1, :]
        if self.ignore_close:
            self.data = self.data.drop(columns=['close'])
        assert self.data.shape[0] == self.stack_frame

    def _update_state(self) -> None:
        self.state = self.data.iloc[:, 1:].astype(dtype='float32').values
        self.state = np.concatenate([self.state.flatten(), self.position], axis=0)
        assert np.isnan(self.state).sum() == 0, f"Invalid state {self.state}"

    def _sell(self, share_num: int) -> None:
        self.share_num -= share_num
        self.account += self.price * share_num * (1 - self.sell_cost_pct)
    
    def _buy(self, share_num: int) -> None:
        self.share_num += share_num
        self.account -= share_num * self.price * (1 + self.buy_cost_pct)

    @property
    def df(self) -> pd.DataFrame:
        return self.dfs[self.df_index]

    @property
    def price(self) -> float:
        return self.df['close'].iloc[self.day]

    @property
    def asset(self) -> float:
        return self.account + self.price * self.share_num

    @property
    def position(self) -> np.array:
        if self.share_num > 0:
            return np.array([1])
        else:
            return np.array([0])

    @staticmethod
    def action_mask_func(obs: th.Tensor) -> th.Tensor:
        if obs[-1] == 1:
            return th.Tensor([0, 0, -th.inf])
        elif obs[-1] == 0:
            return th.Tensor([-th.inf, 0, 0])
        else:
            raise ValueError(f'Invalid obs {obs}')