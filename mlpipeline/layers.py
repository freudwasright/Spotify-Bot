import torch


class ResidualLayer(torch.nn.Module):
    def __init__(self, input_channels: tuple, output_channels: tuple, strides: int):
        super(ResidualLayer, self).__init__()
        self.conv_layer_1 = torch.nn.Conv2d(input_channels[0], output_channels[0], kernel_size=(1, 1),
                                            stride=(strides, strides), padding=(1, 1))
        self.conv_layer_2 = torch.nn.Conv2d(output_channels[0] // 2, output_channels[0], kernel_size=(3, 3),
                                            stride=(strides, strides), padding=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layer_1(x)
        return x
