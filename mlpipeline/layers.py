import torch
import torch.nn.functional as f


class ResidualLayer(torch.nn.Module):
    def __init__(self, channel_sizes: tuple, is_increase: bool = False, strides: int = 0):
        super(ResidualLayer, self).__init__()
        self.is_increase = is_increase

        # (w - f + 2p) / s + 1
        self.conv_layer_1 = torch.nn.Conv2d(channel_sizes[0], channel_sizes[0] // 4, kernel_size=(1, 1),
                                            stride=(strides, strides))
        self.conv_layer_2 = torch.nn.Conv2d(channel_sizes[0] // 4, channel_sizes[0] // 4, kernel_size=(3, 3),
                                            stride=(1, 1), padding=(1, 1))
        self.conv_layer_3 = torch.nn.Conv2d(channel_sizes[0] // 4, channel_sizes[1], kernel_size=(1, 1), stride=(1, 1))

        if self.is_increase:
            # to increase the channel size and halve the image sizes
            self.conv_layer_4 = torch.nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size=(1, 1),
                                                stride=(strides, strides))
            self.bn_4 = torch.nn.BatchNorm2d(channel_sizes[1])

        self.bn_1 = torch.nn.BatchNorm2d(channel_sizes[0] // 4)
        self.bn_2 = torch.nn.BatchNorm2d(channel_sizes[0] // 4)
        self.bn_3 = torch.nn.BatchNorm2d(channel_sizes[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_identity = x
        x = self.conv_layer_1(x)
        x = self.bn_1(x)
        x = f.gelu(x)

        x = self.conv_layer_2(x)
        x = self.bn_2(x)
        x = f.gelu(x)

        x = self.conv_layer_3(x)
        x = self.bn_3(x)

        if self.is_increase:
            x_identity = self.conv_layer_4(x_identity)
            x_identity = self.bn_4(x_identity)

        return f.gelu(x + x_identity)