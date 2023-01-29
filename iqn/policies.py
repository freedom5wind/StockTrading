from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn


class CosineEmbeddingNetwork(nn.Module):
    '''Borrow from https://github.com/toshikwa/fqf-iqn-qrdqn.pytorch/fqf_iqn_qrdqn/network.py'''
    def __init__(self, feature_dim, cosine_embedding=64, ):
        super(CosineEmbeddingNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(cosine_embedding, feature_dim),
            nn.ReLU()
        )
        self.cosine_embedding = cosine_embedding
        self.feature_dim = feature_dim

    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * th.arange(
            start=0, end=self.cosine_embedding, dtype=taus.dtype,
            device=taus.device).view(1, 1, self.cosine_embedding)

        # Calculate cos(i * \pi * \tau).
        cosines = th.cos(
            taus.view(batch_size, N, 1) * i_pi
        ).view(batch_size * N, self.cosine_embedding)

        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(
            batch_size, N, self.feature_dim)

        return tau_embeddings


class QuantileNetwork(BasePolicy):
    """
    Quantile network for IQN
    :param observation_space: Observation space
    :param action_space: Action space
    :param risk_distortion_measure: risk distortion measure
    :param K: number of samples used for the policy
    :param cosine_embedding_dim: cosine embedding dimension
    :param net_arch: The specification of the network architecture.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        risk_distortion_measure: Callable[[float], float] = None,
        K: int = 32,
        cosine_embedding_dim: int = 64,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.risk_distortion_measure = risk_distortion_measure
        self.K = K
        self.cosine_embedding_dim = cosine_embedding_dim
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self._build()

    def _build(self) -> None:
        '''Create the network.'''
        self.cosine_net = CosineEmbeddingNetwork(self.features_dim, self.cosine_embedding_dim)

        action_dim = self.action_space.n  # number of actions
        quantile_net = create_mlp(
            self.features_dim, action_dim, self.net_arch, self.activation_fn)
        self.quantile_net = nn.Sequential(*quantile_net)

    def forward(self, obs: th.Tensor, n_samples: Optional[int] = None) -> Tuple[th.Tensor, th.Tensor]:
        """
        Sample q-values for each action for n_samples times.
        :param obs: Observation
        :param n_samples: sampling times
        :return: The sampled q-values for each action, shape: (batch_size, n_samples, action_space.n)
            and sampled taus, shape: (batch_size, n_samples)
        """
        feature_embedding = self.extract_features(obs, self.features_extractor)

        # sample taus
        batch_size = feature_embedding.shape[0]
        if n_samples is None:
            n_samples = self.K
        taus = th.rand(
            batch_size, n_samples,
            dtype=feature_embedding.dtype,
            device=feature_embedding.device
        )
        if self.risk_distortion_measure is not None:
            taus = taus.apply_(self.risk_distortion_measure)

        cosine_embedding = self.cosine_net(taus)

        assert cosine_embedding.shape[0] == feature_embedding.shape[0]
        assert cosine_embedding.shape[2] == feature_embedding.shape[1]

        # feature_embedding:
        # (batch_size, feature_dim) -> (batch_size, 1, feature_dim)
        feature_embedding = feature_embedding.view(
            batch_size, 1, self.features_dim)
        embedding = feature_embedding * cosine_embedding
        embedding = embedding.view(batch_size * n_samples, self.features_dim)
        qa = self.quantile_net(embedding)   # sampled q-values for each action

        return qa.view(batch_size, n_samples, self.action_space.n), taus

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values, _ = self(observation)
        q_values = q_values.mean(dim=1)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                K=self.K,
                cosine_embedding_dim=self.cosine_embedding_dim,
                risk_distortion_measure=self.risk_distortion_measure,
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class IQNPolicy(BasePolicy):
    """
    Policy class with quantile and target networks for QR-DQN.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param K: number of samples used for the policy
    :param cosine_embedding_dim: cosine embedding dimension
    :param risk_distortion_measure: risk ditortion measure
    :param net_arch: The specification of the network architecture.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        risk_distortion_measure: Callable[[float], float] = None,
        K: int = 32,
        cosine_embedding_dim: int = 64,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        self.K = K
        self.cosine_embedding_dim = cosine_embedding_dim
        self.risk_distortion_measure = risk_distortion_measure
        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "K": self.K,
            "cosine_embedding_dim": self.cosine_embedding_dim,
            "risk_distortion_measure": self.risk_distortion_measure,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.quantile_net, self.quantile_net_target = None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.quantile_net = self.make_quantile_net()
        self.quantile_net_target = self.make_quantile_net()
        self.quantile_net_target.load_state_dict(
            self.quantile_net.state_dict())
        self.quantile_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_quantile_net(self) -> QuantileNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None)
        return QuantileNetwork(**net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self.quantile_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                K=self.K,
                cosine_embedding_dim=self.cosine_embedding_dim,
                risk_distortion_measure=self.risk_distortion_measure,
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                # dummy lr schedule, not needed for loading policy alone
                lr_schedule=self._dummy_schedule,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        This affects certain modules, such as batch normalisation and dropout.
        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.quantile_net.set_training_mode(mode)
        self.training = mode


MlpPolicy = IQNPolicy


class CnnPolicy(IQNPolicy):
    pass


class MultiInputPolicy(IQNPolicy):
    pass


class RnnPolicy(IQNPolicy):
    pass
