from typing import Callable, Dict, Type, TypeVar, Union

import numpy as np
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv
from sb3_contrib.common.utils import quantile_huber_loss
from sb3_contrib.qrdqn import QRDQN
from sb3_contrib.qrdqn.policies import QRDQNPolicy
import torch as th


class MaskableQRDQNPolicy(QRDQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_mask_func: Callable[[th.Tensor], th.Tensor] = None

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self.quantile_net(obs).mean(dim=1)
        # Apply action mask
        action_mask = self.action_mask_func(obs)
        q_values += action_mask.expand(q_values.shape[0], -1)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action


SelfMaskableQRDQN = TypeVar("SelfMaskableQRDQN", bound="MaskableQRDQN")


class MaskableQRDQN(QRDQN):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MaskableQRDQNPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[MaskableQRDQNPolicy]],
        env: Union[GymEnv, str],
        **kwargs
    ):
        super().__init__(policy, env, **kwargs)
        self.action_mask_func = env.action_mask_func
        assert self.action_space.n == \
            self.action_mask_func(th.zeros_like(th.tensor(self.observation_space.sample()))).shape[0], \
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
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the quantiles of next observation
                next_quantiles = self.quantile_net_target(replay_data.next_observations)
                # Apply action mask
                action_mask = self.action_mask_func(replay_data.observations)
                masked_next_quantiles = next_quantiles + action_mask.expand(batch_size, self.n_quantiles, -1)
                # Compute the greedy actions which maximize the next Q values
                next_greedy_actions = masked_next_quantiles.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
                # Make "n_quantiles" copies of actions, and reshape to (batch_size, n_quantiles, 1)
                next_greedy_actions = next_greedy_actions.expand(batch_size, self.n_quantiles, 1)
                # Follow greedy policy: use the one with the highest Q values
                next_quantiles = next_quantiles.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
                # 1-step TD target
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_quantiles

            # Get current quantile estimates
            current_quantiles = self.quantile_net(replay_data.observations)

            # Make "n_quantiles" copies of actions, and reshape to (batch_size, n_quantiles, 1).
            actions = replay_data.actions[..., None].long().expand(batch_size, self.n_quantiles, 1)
            # Retrieve the quantiles for the actions from the replay buffer
            current_quantiles = th.gather(current_quantiles, dim=2, index=actions).squeeze(dim=2)

            # Compute Quantile Huber loss, summing over a quantile dimension as in the paper.
            loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=True)
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
    
    def learn(self: SelfMaskableQRDQN, *args, **kwargs) -> SelfMaskableQRDQN:
        return super().learn(*args, **kwargs)
