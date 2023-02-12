from typing import Callable, Dict, Type, TypeVar

import numpy as np
import torch as th
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import polyak_update

from sb3_contrib.common.utils import quantile_huber_loss
from sb3_contrib.tqc import TQC
from sb3_contrib.tqc.policies import TQCPolicy


SelfTQCR = TypeVar("SelfTQCR", bound="TQCR")


class TQCR(TQC):
    """
    TQC with risk distortion measure as a replacement for quantile truncating.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient update after each step
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param top_quantiles_to_drop_per_net: Number of quantiles to drop per network
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": TQCPolicy,
    }

    def __init__(
            self, 
            *args, 
            utility_func: Callable[[th.Tensor], th.Tensor] = None, 
            **kwargs):
        super().__init__(*args, **kwargs)
        self.utility_func = utility_func

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())
            self.replay_buffer.ent_coef = ent_coef.item()

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute and cut quantiles at the next state
                # batch x nets x quantiles
                next_quantiles = self.critic_target(replay_data.next_observations, next_actions)

                # Quantile truncating in TQC
                # # Sort and drop top k quantiles to control overestimation.
                # n_target_quantiles = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
                # next_quantiles, _ = th.sort(next_quantiles.reshape(batch_size, -1))
                # next_quantiles = next_quantiles[:, :n_target_quantiles]
                next_quantiles, _ = th.sort(next_quantiles.reshape(batch_size, -1))
                next_quantiles = self.utility_func(next_quantiles)
                assert len(next_quantiles.shape) == 2 and next_quantiles.shape[0] == batch_size, \
                    f'Tensor returned by risk distortion measure has invalid shape {next_quantiles.shape}'

                # td error + entropy term
                target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_quantiles
                # Make target_quantiles broadcastable to (batch_size, n_critics, n_target_quantiles).
                target_quantiles.unsqueeze_(dim=1)

            # Get current Quantile estimates using action from the replay buffer
            current_quantiles = self.critic(replay_data.observations, replay_data.actions)
            # Compute critic loss, not summing over the quantile dimension as in the paper.
            critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            qf_pi = self.critic(replay_data.observations, actions_pi).mean(dim=2).mean(dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see https://github.com/DLR-RM/stable-baselines3/issues/996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfTQCR,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TQCR",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTQCR:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


