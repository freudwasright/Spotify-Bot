import torch
import torch.nn.functional as f
from mlpipeline.layers import ResidualLayer


class ResNet(torch.nn.Module):
    def __init__(self, input_channel: int, output_channel: int, out_dim: int):
        super(ResNet, self).__init__()
        self.conv_layer_1 = torch.nn.Conv2d(input_channel, output_channel, kernel_size=(7, 7), stride=(2, 2))
        self.max_pool_layer_1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn_1 = torch.nn.BatchNorm2d(output_channel)

        self.res_layer_1 = torch.nn.Sequential(
            *(3 * [ResidualLayer((output_channel, output_channel), strides=1)])
        )
        self.res_layer_2 = torch.nn.Sequential(
            ResidualLayer((output_channel, output_channel * 2), is_increase=True, strides=2),
            *(3 * [ResidualLayer((output_channel * 2, output_channel * 2), strides=1)])
        )
        self.res_layer_3 = torch.nn.Sequential(
            ResidualLayer((output_channel * 2, output_channel * 4), is_increase=True, strides=2),
            *(3 * [ResidualLayer((output_channel * 4, output_channel * 4), strides=1)])
        )
        self.res_layer_4 = torch.nn.Sequential(
            ResidualLayer((output_channel * 4, output_channel * 6), is_increase=True, strides=2),
            *(3 * [ResidualLayer((output_channel * 6, output_channel * 6), strides=1)])
        )

        # classifier part
        self.dense_layer = torch.nn.Linear(192*9*14, out_dim)
        self.fl = torch.nn.Flatten()

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.bn_1(x)
        x = self.max_pool_layer_1(f.gelu(x))

        x = self.res_layer_1(x)
        x = self.res_layer_2(x)
        x = self.res_layer_3(x)
        x = self.res_layer_4(x)
        x = self.fl(x)
        x = self.dense_layer(x)

        return x