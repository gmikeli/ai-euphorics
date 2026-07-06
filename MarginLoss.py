from torch import nn


class MarginLoss(nn.Module):
    """Maximizes the gap between the candidate logit and the strongest competitor."""

    def __init__(self):
        super().__init__()

    def forward(self, candidate, other_logit):
        return -(candidate - other_logit)