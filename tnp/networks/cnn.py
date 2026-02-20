from typing import List, Optional, Tuple, Union

import torch
from check_shapes import check_shapes
from torch import nn

CONV = {
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d,
}

TRANSPOSE_CONV = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}

POOL = {
    1: nn.MaxPool1d,
    2: nn.MaxPool2d,
    3: nn.MaxPool3d,
}

UPSAMPLE_MODE = {
    1: "linear",
    2: "bilinear",
    3: "trilinear",
}


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        Conv: nn.Module,
        kernel_size: Union[int, Tuple[int, ...]] = 5,
        activation: nn.Module = nn.ReLU(),
        **kwargs,
    ):
        super().__init__()

        self.activation = activation

        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            kernel_size = tuple(kernel_size)
            padding = [k // 2 for k in kernel_size]

        # Conv = make_depth_sep_conv(Conv)
        self.conv = Conv(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs
        )

    @check_shapes("x: [m, c, ...]")
    def forward(self, x: torch.Tensor):
        return self.conv(self.activation(x))


class ResConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        Conv: nn.Module,
        kernel_size: Union[int, Tuple[int, ...]] = 5,
        activation: nn.Module = nn.ReLU(),
        bias: bool = True,
        num_conv_layers: int = 1,
    ):
        super().__init__()

        self.activation = activation
        self.num_conv_layers = num_conv_layers
        assert num_conv_layers in [1, 2]

        if isinstance(kernel_size, int):
            if kernel_size % 2 == 0:
                raise ValueError(f"kernel_size={kernel_size}, but should be odd.")

            padding = kernel_size // 2
        else:
            kernel_size = tuple(kernel_size)
            for k in kernel_size:
                if k % 2 == 0:
                    raise ValueError(f"kernel_size={k}, but should be odd.")

            padding = [k // 2 for k in kernel_size]

        if num_conv_layers == 2:
            # self.conv1 = make_depth_sep_conv(Conv)(
            #     in_channels, in_channels, kernel_size, padding=padding, bias=bias
            # )
            self.conv1 = Conv(
                in_channels, in_channels, kernel_size, padding=padding, bias=bias
            )

        self.conv2_depthwise = Conv(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.conv2_pointwise = Conv(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_conv_layers == 2:
            out = self.conv1(self.activation(x))
        else:
            out = x

        out = self.conv2_depthwise(self.activation(out))

        # Adds residual before point wise, output can change number of channels.
        out = out + x
        out = self.conv2_pointwise(out)
        return out


class CNN(nn.Module):
    """Simple multilayer CNN.

    Args:
        dim (int): Input dimensionality.

        num_channels : int or list
            Number of channels, same for input and output. If list then needs to be
            of size `n_blocks - 1`, e.g. [16, 32, 64] means that you will have a
            `[ConvBlock(16,32), ConvBlock(32, 64)]`.

        num_blocks : int, optional
            Number of convolutional blocks.

        kwargs :
            Additional arguments to `ConvBlock`.
    """

    def __init__(
        self,
        dim: int,
        num_channels: Union[List[int], int],
        num_blocks: Optional[int] = None,
        conv_module: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__()

        if num_blocks is None:
            assert isinstance(num_channels, list)
            num_blocks = len(num_channels) - 1

        self.dim = dim
        self.num_blocks = num_blocks
        self.in_out_channels = self._get_in_out_channels(num_channels, num_blocks)
        if conv_module is None:
            conv_module = CONV[dim]

        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_c, out_c, conv_module, **kwargs)
                for in_c, out_c in self.in_out_channels
            ]
        )

    @check_shapes("x: [m, ..., c]", "return: [m, ...]")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move channels to after batch dimension.
        x = torch.movedim(x, -1, 1)

        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Move channels to final dimension.
        x = torch.movedim(x, 1, -1)

        return x

    def _get_in_out_channels(
        self, num_channels: Union[List[int], int], num_blocks: int
    ) -> List[Tuple[int, int]]:
        """Return a list of tuple of input and output channels."""
        if isinstance(num_channels, int):
            channel_list = [num_channels] * (num_blocks + 1)
        else:
            channel_list = list(num_channels)

        assert len(channel_list) == (
            num_blocks + 1
        ), f"{len(channel_list)} != {num_blocks}."

        return list(zip(channel_list, channel_list[1:]))


class UNet(CNN):
    def __init__(
        self,
        dim: int,
        num_channels: Union[int, List[int]],
        num_blocks: Optional[int] = None,
        max_num_channels: int = 256,
        pooling_size: Union[int, Tuple[int, ...]] = 2,
        factor_chan: int = 2,
        **kwargs,
    ):
        self.max_num_channels = max_num_channels
        self.factor_chan = factor_chan
        super().__init__(dim, num_channels, num_blocks, **kwargs)

        if not isinstance(pooling_size, int):
            pooling_size = tuple(pooling_size)

        self.pooling_size = pooling_size
        self.pooling = POOL[dim](pooling_size)
        self.upsample_mode = UPSAMPLE_MODE[dim]

    @check_shapes("x: [m, ..., c]", "return: [m, ..., c]")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move channels to after batch dimension.
        x = torch.movedim(x, -1, 1)

        num_down_blocks = self.num_blocks // 2
        residuals = []

        # Downwards convolutions.
        for i in range(num_down_blocks):
            x = self.conv_blocks[i](x)
            residuals.append(x)
            x = self.pooling(x)

        # Bottleneck.
        x = self.conv_blocks[num_down_blocks](x)

        # Upwards convolutions.
        for i in range(num_down_blocks + 1, self.num_blocks):
            x = nn.functional.interpolate(
                x,
                size=residuals[num_down_blocks - i].shape[-self.dim :],
                mode=self.upsample_mode,
                align_corners=True,
            )

            x = torch.cat((x, residuals[num_down_blocks - i]), dim=1)
            x = self.conv_blocks[i](x)

        x = torch.movedim(x, 1, -1)
        return x

    def _get_in_out_channels(
        self, num_channels: Union[List[int], int], num_blocks: int
    ) -> List[Tuple[int, int]]:
        # Doubles at every down layer, as in vanila UNet.
        factor_chan = self.factor_chan

        assert num_blocks % 2 == 1, f"n_blocks={num_blocks} not odd."

        if isinstance(num_channels, int):
            # e.g. if n_channels=16, n_blocks=5: [16, 32, 64].
            channel_list = [
                factor_chan**i * num_channels for i in range(num_blocks // 2 + 1)
            ]
        else:
            channel_list = list(num_channels)

        # e.g.: [16, 32, 64, 64, 32, 16].
        channel_list = channel_list + channel_list[::-1]

        # Bound max number of channels by self.max_nchannels (besides first and
        # last dim as this is input / output should not be changed).
        channel_list = (
            channel_list[:1]
            + [min(c, self.max_num_channels) for c in channel_list[1:-1]]
            + channel_list[-1:]
        )

        # e.g.: [(16, 32), (32, 64), (64, 64), (64, 32), (32, 16)].
        in_out_channels = super()._get_in_out_channels(channel_list, num_blocks)
        # e.g.: [(16, 32), (32, 64), (64, 64), (128, 32), (64, 16)] due to concat.
        idcs = slice(len(in_out_channels) // 2 + 1, len(in_out_channels))
        in_out_channels[idcs] = [
            (in_chan * 2, out_chan) for in_chan, out_chan in in_out_channels[idcs]
        ]
        return in_out_channels
