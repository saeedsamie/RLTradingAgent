import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces

class MixedActionPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.Tanh, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, net_arch=net_arch, activation_fn=activation_fn, *args, **kwargs)
        input_dim = observation_space.shape[0] * observation_space.shape[1]
        self.flatten = nn.Flatten()
        self.net_arch = net_arch or [64, 64]
        self.activation_fn = activation_fn
        layers = []
        last_dim = input_dim
        for hidden in self.net_arch:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(self.activation_fn())
            last_dim = hidden
        self.mlp = nn.Sequential(*layers)
        # Action head: output 2 values (action_type, confidence)
        self.action_head = nn.Linear(last_dim, 2)
        # Value head
        self.value_head = nn.Linear(last_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, deterministic=False):
        x = self.flatten(obs)
        x = self.mlp(x)
        action_out = self.action_head(x)
        value = self.value_head(x)
        std = torch.ones_like(action_out) * 0.1
        dist = torch.distributions.Normal(action_out, std)
        if deterministic:
            actions = torch.stack([
                torch.round(action_out[..., 0]),
                torch.clamp(action_out[..., 1], 0, 1)
            ], dim=-1)
        else:
            sampled = dist.rsample()
            action_type = torch.round(sampled[..., 0])
            confidence = torch.clamp(sampled[..., 1], 0, 1)
            actions = torch.stack([action_type, confidence], dim=-1)
        log_prob = dist.log_prob(actions).sum(-1)
        return actions, value, log_prob

    def _predict(self, observation, deterministic=False):
        action_out, _, _ = self.forward(observation, deterministic)
        if deterministic:
            action_type = torch.round(action_out[..., 0])
            confidence = torch.clamp(action_out[..., 1], 0, 1)
        else:
            action_type = torch.round(action_out[..., 0] + torch.randn_like(action_out[..., 0]) * 0.1)
            confidence = torch.clamp(action_out[..., 1] + torch.randn_like(action_out[..., 1]) * 0.1, 0, 1)
        return torch.stack([action_type, confidence], dim=-1)

    def evaluate_actions(self, obs, actions):
        action_out, value, _ = self.forward(obs)
        # Assume Gaussian for both action_type and confidence
        mean = action_out
        std = torch.ones_like(mean) * 0.1
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return value, log_prob, entropy 