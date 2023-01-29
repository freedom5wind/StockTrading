from typing import Callable, Dict, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import is_vectorized_observation
from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy
import torch as th
from torch.nn import functional as F


class MaskableDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_mask_func: Callable[[th.Tensor], th.Tensor] = None

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self.q_net(obs)
        # Apply action mask
        action_mask = self.action_mask_func(obs)
        q_values += action_mask.expand(q_values.shape[0], -1)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)

        return action


SelfMaskableDQN = TypeVar("SelfMaskableDQN", bound="MaskableDQN")


class MaskableDQN(DQN):
    def __init__(
        self,
        policy: Union[str, Type[MaskableDQNPolicy]],
        env: Union[GymEnv, str],
        action_mask_func: Callable[[th.Tensor], th.Tensor],
        **kwargs
    ):
        '''Action_mask_func should return -inf for invalid actions.'''
        super().__init__(policy, env, **kwargs)
        assert self.action_space.n == \
            action_mask_func(th.zeros(self.observation_space.sample().shape)).shape[0], \
            'Invalid action mask function.'
        self.action_mask_func = action_mask_func
        self.policy.action_mask_func = action_mask_func

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(
                    replay_data.next_observations)
                # Apply action mask
                action_mask = self.action_mask_func(replay_data.next_observations)
                next_q_values += action_mask.expand(batch_size, -1)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + \
                    (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates",
                           self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        def sample_action(obs: np.ndarray) -> int:
            action_mask = self.action_mask_func(th.tensor(obs))
            action=self.action_space.sample()
            while(not action_mask[action]):
                action=self.action_space.sample()
            return action
    
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(observation, dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([sample_action(observation) for _ in range(n_batch)])
            else:
                action = np.array(sample_action(observation))
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    def learn(self: SelfMaskableDQN, *args, **kwargs) -> SelfMaskableDQN:
        super().learn(*args, **kwargs)
