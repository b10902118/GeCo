from torch import nn


class UpsamplingLayer(nn.Module):

    def __init__(self, in_channels, out_channels, leaky=True):

        super(UpsamplingLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU() if leaky else nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

    def forward(self, x):
        return self.layer(x)
