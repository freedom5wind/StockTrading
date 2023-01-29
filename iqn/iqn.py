from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, is_vectorized_observation, polyak_update

from iqn.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, RnnPolicy, IQNPolicy


def quantile_huber_loss(
    sampled_current: th.Tensor,
    sampled_target: th.Tensor,
    taus: th.Tensor,
    sum_over_quantiles: bool = True,
) -> th.Tensor:
    """
    The quantile-regression loss.
    Code borrowed from sb3_contrib.common.utils.quantile_huber_loss

    :param sampled_current: sampled current q-values, must be (batch_size, N1),
        where N1 is the number of current samples.
    :param sampled_target: sampled target q-values, must be (batch_size, N2),
        where N2 is the number of target samples.
    :param taus: taus used to sample current q-values, must be (batch_size, N1)
    :param sum_over_quantiles: if summing over the quantile dimension or not
    :return: the loss
    """
    if sampled_current.ndim != sampled_target.ndim:
        raise ValueError(
            f"Error: The dimension of sampled_current ({sampled_current.ndim}) needs to match "
            f"the dimension of sampled_target ({sampled_target.ndim})."
        )
    if sampled_current.shape[0] != sampled_target.shape[0]:
        raise ValueError(
            f"Error: The batch size of sampled_current ({sampled_current.shape[0]}) needs to match "
            f"the batch size of sampled_target ({sampled_target.shape[0]})."
        )

    # QR-DQN
    # target_quantiles: (batch_size, n_target_quantiles) -> (batch_size, 1, n_target_quantiles)
    # current_quantiles: (batch_size, n_quantiles) -> (batch_size, n_quantiles, 1)
    # pairwise_delta: (batch_size, n_target_quantiles, n_quantiles)
    pairwise_delta = sampled_target.unsqueeze(-2) - sampled_current.unsqueeze(-1)
    abs_pairwise_delta = th.abs(pairwise_delta)
    huber_loss = th.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)
    loss = th.abs(taus.unsqueeze(-2) - (pairwise_delta.detach() < 0).float()) * huber_loss
    if sum_over_quantiles:
        loss = loss.sum(dim=-1).mean()
    else:
        loss = loss.mean()
    return loss

SelfIQN = TypeVar("SelfIQN", bound="IQN")


class IQN(OffPolicyAlgorithm):
    """
    Implicit Quantile Network (IQN)
    Paper: https://arxiv.org/abs/1806.06923
    Default hyperparameters are taken from the paper and are tuned for Atari games. 
    (most of them correspond to the one used in QR-DQN)
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping (if None, no clipping)
    :param N1: number of samples used to evaluate current q-values
    :param N2: number of samples used for target q-values
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[IQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 5e-5,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.005,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.01,
        max_grad_norm: Optional[float] = None,
        N1: int = 32,
        N2: int = 32,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.N1 = N1
        self.N2 = N2
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.quantile_net, self.quantile_net_target = None, None

        if "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.Adam
            # Proposed in the QR-DQN paper where `batch_size = 32`
            self.policy_kwargs["optimizer_kwargs"] = dict(eps=0.01 / batch_size)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see https://github.com/DLR-RM/stable-baselines3/issues/996
        self.batch_norm_stats = get_parameters_by_name(self.quantile_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.quantile_net_target, ["running_"])
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )

    def _create_aliases(self) -> None:
        self.quantile_net = self.policy.quantile_net
        self.quantile_net_target = self.policy.quantile_net_target
        self.K = self.policy.K

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        if self.num_timesteps % self.target_update_interval == 0:
            polyak_update(self.quantile_net.parameters(), self.quantile_net_target.parameters(), self.tau)
            # Copy running stats, see https://github.com/DLR-RM/stable-baselines3/issues/996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

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
                if isinstance(self.policy, RnnPolicy):
                    # TODO: might need to initialize lstm hidden state if RnnPolicy is used
                    pass
                else:
                    # Sample the qa of next observation
                    next_qa_sampled, _ = self.quantile_net_target(replay_data.next_observations, self.K)
                    assert next_qa_sampled.shape == (batch_size, self.K, self.action_space.n)
                    # Compute the greedy actions which maximize the next Q values
                    next_greedy_actions = next_qa_sampled.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)

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
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).
        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(self.observation_space, spaces.Dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    def learn(
        self: SelfIQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "IQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfIQN:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["quantile_net", "quantile_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []