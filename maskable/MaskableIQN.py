from typing import Callable, Dict, Type, TypeVar, Union, Tuple, Optional

import numpy as np
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import is_vectorized_observation
import torch as th

from iqn.policies import IQNPolicy
from iqn.iqn import IQN, quantile_huber_loss


class MaskableIQNPolicy(IQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_mask_func: Callable[[th.Tensor], th.Tensor] = None

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values, _ = self.quantile_net(obs)
        q_values = q_values.mean(dim=1)
        # Apply action mask
        action_mask = self.action_mask_func(obs)
        q_values += action_mask.expand(q_values.shape[0], -1)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action


SelfMaskableIQN = TypeVar("SelfMaskableIQN", bound="MaskableIQN")


class MaskableIQN(IQN):
    def __init__(
        self,
        policy: Union[str, Type[MaskableIQNPolicy]],
        env: Union[GymEnv, str],
        action_mask_func: Callable[[th.Tensor], th.Tensor],
        **kwargs
    ):
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
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Sample the qa of next observation
                next_qa_sampled, _ = self.quantile_net_target(replay_data.next_observations, self.K)
                assert next_qa_sampled.shape == (batch_size, self.K, self.action_space.n)
                # Apply action mask
                action_mask = self.action_mask_func(replay_data.observations)
                masked_next_qa_sampled = next_qa_sampled + action_mask.expand(batch_size, self.K, -1)
                # Compute the greedy actions which maximize the next Q values
                next_greedy_actions = masked_next_qa_sampled.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
                # Make "self.N2" copies of actions, and reshape to (batch_size, self.N2, 1)
                next_greedy_actions = next_greedy_actions.expand(batch_size, self.N2, 1)
                if self.K < self.N2:
                    # Sample the qa again for self.N2 times and then
                    # follow greedy policy: use the one with the highest Q values
                    next_qa_sampled, _ = self.quantile_net_target(replay_data.next_observations, self.N2)
                next_qa_sampled = next_qa_sampled.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
                # 1-step TD target
                target_qa_sampled = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_qa_sampled

            # Get current quantile estimates
            current_qa_sampled, taus = self.quantile_net(replay_data.observations, self.N1)

            # Make "self.N1" copies of actions, and reshape to (batch_size, self.N1, 1).
            actions = replay_data.actions[..., None].long().expand(batch_size, self.N1, 1)
            # Retrieve the sampled q-values for the actions from the replay buffer
            current_qa_sampled = th.gather(current_qa_sampled, dim=2, index=actions).squeeze(dim=2)

            # Compute Quantile Huber loss, summing over a quantile dimension as in the paper.
            loss = quantile_huber_loss(current_qa_sampled, target_qa_sampled, taus, sum_over_quantiles=True)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
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


    def learn(self: SelfMaskableIQN, *args, **kwargs) -> SelfMaskableIQN:
        return super().learn(*args, **kwargs)
