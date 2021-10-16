from torch import nn


class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)