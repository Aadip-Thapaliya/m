"""Optional PyTorch graph-conditioned diagonal selective SSM research model.

Install the research extra before importing this module. The sequential loop is
an interpretable reference implementation; it is not a fused Mamba kernel.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GraphConditionedSelectiveSSM(nn.Module):
    """Condition stable diagonal state transitions on a lagged adjacency graph.

    Inputs have shape ``[batch, length, variables]``; graphs have shape
    ``[batch, variables, variables]`` with the ``[target, parent]`` convention.
    The graph propagation is dense ``O(D²)``; the diagonal state update is
    ``O(D * N)``. Runtime is linear in sequence length for fixed ``D`` and ``N``.
    """

    def __init__(self, n_variables: int, state_size: int = 16, graph_size: int = 32) -> None:
        super().__init__()
        self.n_variables = n_variables
        self.state_size = state_size
        self.graph_encoder = nn.Sequential(
            nn.Linear(n_variables, graph_size), nn.SiLU(), nn.Linear(graph_size, graph_size)
        )
        self.input_projection = nn.Linear(2, state_size)
        self.delta_projection = nn.Linear(graph_size + 1, state_size)
        self.b_projection = nn.Linear(graph_size + 1, state_size)
        self.c_projection = nn.Linear(graph_size + 1, state_size)
        self.log_decay = nn.Parameter(torch.zeros(n_variables, state_size))
        self.output_projection = nn.Linear(state_size, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, observations: Tensor, adjacency: Tensor) -> Tensor:
        if observations.ndim != 3:
            raise ValueError("Observations must have shape [batch, length, variables]")
        batch, length, variables = observations.shape
        if variables != self.n_variables:
            raise ValueError("Observation variable count does not match the model")
        if adjacency.shape != (batch, variables, variables):
            raise ValueError("Adjacency must have shape [batch, variables, variables]")

        # Each row encodes the incoming parent weights of one target variable.
        graph_embedding = self.graph_encoder(adjacency)
        hidden = observations.new_zeros(batch, variables, self.state_size)
        outputs = []
        stable_decay = -torch.exp(self.log_decay).unsqueeze(0)

        for step in range(length):
            current = observations[:, step]
            graph_message = torch.einsum("bij,bj->bi", adjacency, current)
            local_input = torch.stack((current, graph_message), dim=-1)
            projected = self.input_projection(local_input)
            selective_input = torch.cat((graph_embedding, current.unsqueeze(-1)), dim=-1)
            delta = F.softplus(self.delta_projection(selective_input)) + 1e-4
            input_gate = torch.sigmoid(self.b_projection(selective_input))
            readout_gate = torch.sigmoid(self.c_projection(selective_input))
            transition = torch.exp(delta * stable_decay)
            hidden = transition * hidden + delta * input_gate * projected
            prediction = self.output_projection(readout_gate * hidden).squeeze(-1)
            outputs.append(prediction + self.residual_scale * graph_message)

        return torch.stack(outputs, dim=1)
