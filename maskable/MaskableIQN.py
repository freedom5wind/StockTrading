from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv
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

        if deterministic:
            # Greedy action
            action = q_values.argmax(dim=1).reshape(-1)
        else:
            action =  q_values.exp().multinomial(1).reshape(-1)

        return action


SelfMaskableIQN = TypeVar("SelfMaskableIQN", bound="MaskableIQN")


class MaskableIQN(IQN):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MaskableIQNPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[MaskableIQNPolicy]],
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
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Sample the qa of next observation
                next_qa_sampled, _ = self.quantile_net_target(replay_data.next_observations, self.K)
                assert next_qa_sampled.shape == (batch_size, self.K, self.action_space.n)
                # Apply action mask
                action_mask = self.action_mask_func(replay_data.observations)
                masked_next_qa_sampled = next_qa_sampled + action_mask.unsqueeze(-2)
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

    def predict(self, *args, action_masks: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self._action_masks = action_masks
        return super().predict(*args, **kwargs)


    def learn(self: SelfMaskableIQN, *args, **kwargs) -> SelfMaskableIQN:
        return super().learn(*args, **kwargs)
        
    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["action_space"]
