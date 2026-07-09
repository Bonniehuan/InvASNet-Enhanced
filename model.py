import torch.optim
import torch.nn as nn
import config as c
from hinet import Hinet


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = Hinet()

    def forward(self, x, rev=False):

        if not rev:
            out = self.model(x)

        else:
            out = self.model(x, rev=True)

        return out


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            # ✅ 用 param.device，不要硬 cuda
            param.data = c.init_scale * torch.randn_like(param.data, device=param.device)
            if split[-2] == 'conv5':
                param.data.fill_(0.)
