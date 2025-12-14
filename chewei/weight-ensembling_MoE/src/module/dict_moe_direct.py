import logging
from copy import deepcopy
from typing import List, Optional

import torch
import torch.func
from torch import Tensor, nn
from torch.nn import functional as F

log = logging.getLogger(__name__)


def join_list(list_of_list: List[List]):
    ans = []
    for l in list_of_list:
        ans.extend(l)
    return ans


class DictMoEGate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        init_lambda: float,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        assert num_hidden_layers <= 2
        self.input_dim = hidden_size
        self.num_experts = num_experts
        self.num_hidden_layers = num_hidden_layers

        if num_hidden_layers == 2:
            self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
            nn.init.normal_(self.fc1.weight, std=0.01)
            nn.init.zeros_(self.fc1.bias)
        elif num_hidden_layers == 1:
            self.fc1 = nn.Identity()

        if num_hidden_layers >= 1:
            self.fc2 = nn.Linear(hidden_size, num_experts, bias=True)
            nn.init.normal_(self.fc2.weight, std=0.01)
            nn.init.constant_(self.fc2.bias, init_lambda)

        if num_hidden_layers == 0:
            self.weight = nn.Parameter(torch.ones(num_experts) * init_lambda, requires_grad=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.num_hidden_layers == 0:
            return self.weight

        if self.num_hidden_layers == 2:
            hidden_states = F.relu(self.fc1(hidden_states))
        gate_weights = self.fc2(hidden_states)
        return gate_weights


class DictMoEDirect(nn.Module):
    """
    DictMoE variant that directly uses expert weights without task vectors.
    Output = Σ(router_weight[i] × expert[i](x))

    No pretrained base weight is added back.
    """
    def __init__(
        self,
        hidden_size: int,
        expert_models: List[nn.Module],
        init_lambda: float = 0.2,
        fix_experts: bool = True,
        batch_first: bool = False,
        router_hidden_layers: int = 2,
    ):
        super().__init__()
        self.num_experts = len(expert_models)
        self.input_dim = hidden_size
        self.batch_first = batch_first

        self.gate = DictMoEGate(
            hidden_size,
            self.num_experts,
            init_lambda=init_lambda,
            num_hidden_layers=router_hidden_layers,
        )

        # Store one base model for architecture
        self.base_model = deepcopy(expert_models[0])

        # Pre-extract expert parameters to avoid repeated state_dict() calls
        experts_params = []
        experts_sd = [deepcopy(e).state_dict() for e in expert_models]
        base_sd_keys = list(experts_sd[0].keys())

        for name in base_sd_keys:
            expert_weights = []
            for e_sd in experts_sd:
                expert_weights.append(e_sd[name])
            expert_weights = torch.stack(expert_weights)  # [num_experts, ...]
            experts_params.append(nn.Parameter(expert_weights, requires_grad=not fix_experts))

        self.expert_parms = nn.ParameterList(experts_params)

        if fix_experts:
            for p in self.base_model.parameters():
                p.requires_grad_(False)
            for p in self.expert_parms.parameters():
                p.requires_grad_(False)

    def forward(self, hidden_states: Tensor):
        if not self.batch_first:
            hidden_states = hidden_states.permute(1, 0, 2)

        batch_size, seq_len, hidden_size = hidden_states.shape
        gate_weights: Tensor = self.gate(hidden_states)

        if self.gate.num_hidden_layers == 0:
            # Static routing (same weights for all samples)
            base_sd = self.base_model.state_dict(keep_vars=True)
            combined_sd = {}
            for param_idx, (name, param) in enumerate(base_sd.items()):
                expert_params: nn.Parameter = self.expert_parms[param_idx]  # [num_experts, ...]
                # Weight each expert's parameter by gate_weights
                weighted_param = expert_params * gate_weights.view([-1] + [1] * (expert_params.dim() - 1))
                combined_sd[name] = weighted_param.sum(dim=0)

            # Use combined weights to compute output
            final_hidden_states = torch.func.functional_call(self.base_model, combined_sd, hidden_states)
        else:
            # Dynamic routing (different weights per sample)
            gate_weights = gate_weights.mean(dim=1)  # [batch_size, num_experts]
            final_hidden_states = []

            base_sd = self.base_model.state_dict(keep_vars=True)
            for sample_idx in range(batch_size):
                # Weighted combination of expert weights for this sample
                combined_sd = {}
                for param_idx, (name, param) in enumerate(base_sd.items()):
                    expert_params: nn.Parameter = self.expert_parms[param_idx]  # [num_experts, ...]
                    # Weight each expert's parameter by this sample's gate_weights
                    weighted_param = expert_params * gate_weights[sample_idx].view([-1] + [1] * (expert_params.dim() - 1))
                    combined_sd[name] = weighted_param.sum(dim=0)

                # Use combined weights to compute output for this sample
                _final_hidden_states = torch.func.functional_call(
                    self.base_model,
                    combined_sd,
                    hidden_states[sample_idx : sample_idx + 1]
                )
                final_hidden_states.append(_final_hidden_states)

            final_hidden_states = torch.cat(final_hidden_states, dim=0)

        if not self.batch_first:
            final_hidden_states = final_hidden_states.permute(1, 0, 2)

        return final_hidden_states
