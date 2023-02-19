from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from gym import spaces
import numpy as np
import torch as th
from sb3_contrib.tqc import TQC
from sb3_contrib.tqc.policies import Actor
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.policies import BasePolicy, BaseModel
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from torch import nn as nn


from iqn.iqn import quantile_huber_loss
from iqn.policies import CosineEmbeddingNetwork


class QuantileNetwork(nn.Module):
    '''
    :param risk_distortion_measure: feed taus to it, if its type is Callable[[th.Tensor], th.Tensor],
        otherwise, use taus._apply with it.
    '''
    def __init__(
            self, 
            features_dim: int, 
            net_arch: List[int], 
            cos_embedding_dims: int, 
            n_samples: int, 
            risk_distortion_measure: Callable[[th.Tensor], th.Tensor] = None,
            activation_fn: Type[nn.Module] = nn.ReLU, 
    ):
        super().__init__()

        self.features_dim = features_dim
        self.net_arch = net_arch
        self.cos_embedding_dims = cos_embedding_dims,
        self.n_samples = n_samples
        self.risk_distortion_measure = risk_distortion_measure
        self.activation_fn = activation_fn

        self.cosine_net = CosineEmbeddingNetwork(features_dim, cos_embedding_dims)
        quantile_net = create_mlp(features_dim, 1, self.net_arch, self.activation_fn)
        self.quantile_net = nn.Sequential(*quantile_net)

    def forward(self, obs: th.Tensor, n_samples: Optional[int] = None) -> Tuple[th.Tensor, th.Tensor]:
        """
        Sample q-values for each action for n_samples times.
        :param obs: Observation
        :param n_samples: sampling times
        :return: The sampled q-values for each action, shape: (batch_size, n_samples, action_space.n)
            and sampled taus, shape: (batch_size, n_samples)
        """
        # sample taus
        batch_size = obs.shape[0]

        if n_samples is None:
            n_samples = self.n_samples

        taus = th.rand(
            batch_size, n_samples,
            dtype=obs.dtype,
            device=obs.device
        )

        if self.risk_distortion_measure is not None:
            taus = self.risk_distortion_measure(taus)

        cosine_embedding = self.cosine_net(taus)

        assert cosine_embedding.shape[0] == obs.shape[0]
        assert cosine_embedding.shape[2] == obs.shape[1]

        # feature_embedding:
        # (batch_size, feature_dim) -> (batch_size, 1, feature_dim)
        feature_embedding = obs.view(batch_size, 1, self.features_dim)
        embedding = feature_embedding * cosine_embedding
        embedding = embedding.view(batch_size * n_samples, self.features_dim)
        q_values = self.quantile_net(embedding)   # sampled q-values for each action

        return q_values.view(batch_size, n_samples), taus


class IQNCritic(BaseModel):
    """
    Critic network (q-value function) for TQCI.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_samples: int = 64,
        cos_embedding_dims: int = 64,
        risk_distortion_measures: Union[Callable[[th.Tensor], th.Tensor], List[Callable[[th.Tensor], th.Tensor]]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.q_networks = []
        self.n_samples = n_samples
        self.n_critics = n_critics

        for i in range(n_critics):
            rdm = risk_distortion_measures[i] if isinstance(risk_distortion_measures, List) \
                else risk_distortion_measures
            qf_net = QuantileNetwork(
                features_dim = features_dim + action_dim, 
                n_samples = n_samples, 
                net_arch = net_arch, 
                activation_fn = activation_fn,
                cos_embedding_dims = cos_embedding_dims,
                risk_distortion_measure = rdm,
                )
            self.add_module(f"qf{i}", qf_net)
            self.q_networks.append(qf_net)

    def forward(self, obs: th.Tensor, action: th.Tensor) -> List[th.Tensor]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, action], dim=1)
        list_qs = []
        list_taus = []
        for qf in self.q_networks:
            q, tau = qf(qvalue_input)
            list_qs.append(q)
            list_taus.append(tau)
        samples = th.stack(list_qs, dim=1)
        taus = th.stack(list_taus, dim=1)
        return samples, taus


class TQCIPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for TQCI.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the feature extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_samples_critics: Number of samples for critic networks
    :param n_samples_target_critcs: Number of samples for target critic networks
    :param cos_embedding_dims: cosine embedding dimensions
    :param risk_distortion_measures: risk distortion measures
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_samples_critics: int = 64,
        n_samples_target_critcs: int = 64,
        cos_embedding_dims: int = 64,
        risk_distortion_measures: Union[Callable[[th.Tensor], th.Tensor], \
                                        List[Callable[[th.Tensor], th.Tensor]], \
                                        Callable[[float], float], \
                                        List[Callable[[float], float]]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
            squash_output=True,
        )

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)

        self.critic_kwargs = self.net_args.copy()
        tqc_kwargs = {
            "n_samples_critics": n_samples_critics,
            "n_samples_target_critcs": n_samples_target_critcs,
            "cos_embedding_dims": cos_embedding_dims,
            "risk_distortion_measures": risk_distortion_measures,
            "n_critics": n_critics,
            "net_arch": critic_arch,
            "share_features_extractor": share_features_extractor,
        }
        self.critic_kwargs.update(tqc_kwargs)
        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)
        
    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None, is_target_critic=False)
            critic_parameters = self.critic.parameters()

        # Critic target should not share the feature extactor with critic
        self.critic_target = self.make_critic(features_extractor=None, is_target_critic=True)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                n_samples_critics=self.critic_kwargs["n_samples_critics"],
                n_samples_target_critcs=self.critic_kwargs["n_samples_target_critcs"],
                cos_embedding_dims=self.critic_kwargs["cos_embedding_dims"],
                risk_distortion_measures=self.critic_kwargs["risk_distortion_measures"],
                n_critics=self.critic_kwargs["n_critics"],
            )
        )
        return data

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None, is_target_critic: bool = False) -> IQNCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        n_samples_critics = critic_kwargs.pop("n_samples_critics")
        n_samples_target_critcs = critic_kwargs.pop("n_samples_target_critcs")
        if is_target_critic:
            critic_kwargs["n_samples"] = n_samples_target_critcs
        else:
            critic_kwargs["n_samples"] = n_samples_critics
        return IQNCritic(**critic_kwargs).to(self.device)
    
    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        This affects certain modules, such as batch normalisation and dropout.
        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


SelfTQCI = TypeVar("SelfTQCI", bound="TQCI")


class TQCI(TQC):
    """
    TQC with IQN critics.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param n_samples_current_q: number of samples used to evaluate current q-values in critics
    :param n_samples_target_q: number of samples used for target q-values in critics
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
        "MlpPolicy": TQCIPolicy,
    }

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
                # batch x nets x target_samples
                next_q_samples, _ = self.critic_target(replay_data.next_observations, next_actions)
                next_q_samples = next_q_samples.reshape(batch_size, -1)

                # td error + entropy term
                target_q_samples = next_q_samples - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_samples = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q_samples

            # Get current samples estimates using action from the replay buffer
            current_q_samples, taus = self.critic(replay_data.observations, replay_data.actions)
            current_q_samples = current_q_samples.reshape(batch_size, -1)
            taus = taus.reshape(batch_size, -1)
            critic_loss = quantile_huber_loss(current_q_samples, target_q_samples, taus)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            qf_pi = self.critic(replay_data.observations, actions_pi)[0].mean(dim=2).mean(dim=1, keepdim=True)
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
        self: SelfTQCI,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TQCI",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTQCI:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


