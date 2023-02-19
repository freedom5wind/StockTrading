from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv
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

        if deterministic:
            # Greedy action
            action = q_values.argmax(dim=1).reshape(-1)
        else:
            action =  q_values.exp().multinomial(1).reshape(-1)

        return action


SelfMaskableDQN = TypeVar("SelfMaskableDQN", bound="MaskableDQN")


class MaskableDQN(DQN):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MaskableDQNPolicy,
    }
    
    def __init__(
        self,
        policy: Union[str, Type[MaskableDQNPolicy]],
        env: Union[GymEnv, str],
        **kwargs
    ):
        super().__init__(policy, env, **kwargs)
        self.action_mask_func = env.action_mask_func
        assert self.action_space.n == \
            self.action_mask_func(th.zeros_like(th.tensor(self.observation_space.sample()).unsqueeze(0))).shape[1], \
            'Invalid action mask function.'
        self.policy.action_mask_func = self.action_mask_func

        # Don't try this at home.
        # Inject sample() to self.action_space.
        self.action_space.parent_agent = self
        def masked_sample(obj):
            action_masks = np.array(obj.parent_agent.env.env_method('action_masks'))
            valid_action = obj.np_random.randint(action_masks.sum())
            idx = np.searchsorted(action_masks.cumsum(), valid_action + 1)
            return idx
        from types import MethodType
        self.action_space.sample = MethodType(masked_sample, self.action_space)

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

    def predict(self, *args, action_masks: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self._action_masks = action_masks
        return super().predict(*args, **kwargs)

    def learn(self: SelfMaskableDQN, *args, **kwargs) -> SelfMaskableDQN:
        super().learn(*args, **kwargs)

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["action_space"]

