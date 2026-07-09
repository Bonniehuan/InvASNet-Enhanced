import torch
import torch.nn as nn


class ResidualDenseBlock_out_1D(nn.Module):
    """
    1D version of ResidualDenseBlock_out:
    Conv2d -> Conv1d, kernel=3, padding=1 to keep length.
    """
    def __init__(self, input_ch, output_ch, bias=True):
        super().__init__()
        self.conv1 = nn.Conv1d(input_ch, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv1d(input_ch + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv1d(input_ch + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv1d(input_ch + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv1d(input_ch + 4 * 32, output_ch, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)

        # mimic "initialize_weights([conv5], 0.)" effect: set last conv near zero
        nn.init.zeros_(self.conv5.weight)
        if self.conv5.bias is not None:
            nn.init.zeros_(self.conv5.bias)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5

