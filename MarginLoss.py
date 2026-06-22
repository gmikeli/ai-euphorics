from torch import nn


class MarginLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, candidate, other_logit):
        return -(candidate - other_logit)