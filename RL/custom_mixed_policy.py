import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import BasePolicy
from gymnasium import spaces

class MixedActionPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.Tanh, *args, **kwargs):
        super().__init__(observation_space, action_space, *args, **kwargs)
        self.n_actions = action_space[0].n  # Discrete actions
        self.net_arch = net_arch or [64, 64]
        self.activation_fn = activation_fn
        input_dim = observation_space.shape[0] * observation_space.shape[1]
        # Flatten input
        self.flatten = nn.Flatten()
        # Shared MLP
        layers = []
        last_dim = input_dim
        for hidden in self.net_arch:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(self.activation_fn())
            last_dim = hidden
        self.mlp = nn.Sequential(*layers)
        # Discrete action head
        self.action_head = nn.Linear(last_dim, self.n_actions)
        # Continuous confidence head
        self.confidence_head = nn.Linear(last_dim, 1)
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
        logits = self.action_head(x)
        confidence = torch.sigmoid(self.confidence_head(x))  # (0,1)
        value = self.value_head(x)
        return logits, confidence, value

    def _predict(self, observation, deterministic=False):
        logits, confidence, _ = self.forward(observation, deterministic)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            conf = confidence.squeeze(-1)
        else:
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            conf_dist = torch.distributions.Normal(confidence, 0.1)
            conf = torch.clamp(conf_dist.sample(), 0, 1).squeeze(-1)
        # Return as tuple for env
        return torch.stack([action.float(), conf], dim=-1)

    def evaluate_actions(self, obs, actions):
        logits, confidence, value = self.forward(obs)
        action_logits = logits
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = actions[:, 0].long()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        # Confidence log prob (treat as Normal for simplicity)
        conf = actions[:, 1].unsqueeze(-1)
        conf_dist = torch.distributions.Normal(confidence, 0.1)
        log_prob_conf = conf_dist.log_prob(conf).sum(-1)
        total_log_prob = log_prob + log_prob_conf
        total_entropy = entropy + conf_dist.entropy().sum(-1)
        return value, total_log_prob, total_entropy 