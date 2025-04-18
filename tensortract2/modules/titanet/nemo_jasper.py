"""
   Copyright 2025 Altavo GmbH

   “Commons Clause” License Condition v1.0

   The Software is provided to you by Altavo GmbH under the License, 
   as defined below, subject to the following condition.

   Without limiting other conditions in the License, the grant of rights 
   under the License will not include, and the License does not grant to
   you, the right to Sell the Software.

   For purposes of the foregoing, “Sell” means practicing any or all of the 
   rights granted to you under the License to provide to third parties, 
   for a fee or other consideration (including without limitation fees for
   hosting or consulting/ support services related to the Software), 
   a product or service whose value derives, entirely or substantially, 
   from the functionality of the Software. 

   Any license notice or attribution required by the License must also 
   include this Commons Clause License Condition notice.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

"""
This file is adapted from:
https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/jasper.py

under the following license:


Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""



import torch
from typing import (
    List,
    Tuple,
    Optional,
    Iterable,
    Callable,
)
import logging
import math

""" ML Module port of the Jasper Modules and functionalities
    from the NeMo toolkit without dependencies on the
    following classes: AdapterModuleMixin, AccessMixin
    Removed the following features:
        - quantization (via the PYTORCH_QUANTIZATION_AVAILABLE flag)
        - input type checking
    Source:
        https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/jasper.py
    """

jasper_activations = {
    "identity": torch.nn.Identity,
    "hardtanh": torch.nn.Hardtanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": torch.nn.SiLU,
    "silu": torch.nn.SiLU,
    "gelu": torch.nn.GELU,
}


class MaskedConv1d(torch.nn.Module):
    """
    Calculates masked convolutions for a stack of unequal
    length sequences is use_mask is True.
    """

    __constants__ = [
        "use_conv_mask",
        "real_out_channels",
        "heads",
    ]  # TODO; use?

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        heads=-1,
        bias=False,
        use_mask=True,
        quantize=False,  # kept for config compatibility
    ):
        super(
            MaskedConv1d,
            self,
        ).__init__()

        if not (heads == -1 or groups == in_channels):
            raise ValueError("Only use heads for depthwise convolutions")

        self.real_out_channels = out_channels
        if heads != -1:
            in_channels = heads
            out_channels = heads
            groups = heads

        # preserve original padding
        self._padding = padding

        # if padding is a tuple/list, it is considered as asymmetric padding
        if type(padding) in (
            tuple,
            list,
        ):
            self.pad_layer = torch.nn.ConstantPad1d(
                padding,
                value=0.0,
            )
            # reset padding for conv since pad_layer will handle this
            padding = 0
        else:
            self.pad_layer = None

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.use_mask = use_mask
        self.heads = heads

        # Calculations for "same" padding cache
        self.same_padding = (self.conv.stride[0] == 1) and (
            2 * self.conv.padding[0]
            == self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
        )
        if self.pad_layer is None:
            self.same_padding_asymmetric = False
        else:
            self.same_padding_asymmetric = (self.conv.stride[0] == 1) and (
                sum(self._padding)
                == self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
            )

        # `self.lens` caches consecutive integers from 0 to `self.max_len` that are used to compute the mask for a
        # batch. Recomputed to bigger size as needed. Stored on a device of the latest batch lens.
        if self.use_mask:
            self.max_len = torch.tensor(0)
            self.lens = torch.tensor(0)

    def get_seq_len(
        self,
        lens,
    ):
        if self.same_padding or self.same_padding_asymmetric:
            return lens

        if self.pad_layer is None:
            return (
                torch.div(
                    lens
                    + 2 * self.conv.padding[0]
                    - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
                    - 1,
                    self.conv.stride[0],
                    rounding_mode="trunc",
                )
                + 1
            )
        else:
            return (
                torch.div(
                    lens
                    + sum(self._padding)
                    - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
                    - 1,
                    self.conv.stride[0],
                    rounding_mode="trunc",
                )
                + 1
            )

    def forward(
        self,
        x,
        lens,
    ):
        """Calculate the 1d conv on the input batch.
            Note: masking is applied to the *input* of the conv layer, not the output after the convolution.
            Thus, the output has non-zero elements at samples (kernel_size-1)//2 to the left and right if padding
            allows it.
        Args:
            x (torch.Tensor): Input tensor of size (B, C, T).
            lens (torch.Tensor): Tensor of size (B,) containing the lengths of each sequence in the batch.
        Returns:
            torch.Tensor: Output tensor of size (B, C', T').
            torch.Tensor: Tensor of size (B,) containing the lengths of each sequence in the batch.
        """

        if self.use_mask:
            # Generally will be called by ConvASREncoder, but kept as single gpu backup.
            if x.size(2) > self.max_len:
                self.update_masked_length(
                    x.size(2),
                    device=lens.device,
                )
            x = self.mask_input(
                x,
                lens,
            )

        # Update lengths
        lens = self.get_seq_len(lens)

        # asymmtric pad if necessary
        if self.pad_layer is not None:
            x = self.pad_layer(x)

        sh = x.shape
        if self.heads != -1:
            x = x.view(
                -1,
                self.heads,
                sh[-1],
            )

        out = self.conv(x)

        if self.heads != -1:
            out = out.view(
                sh[0],
                self.real_out_channels,
                -1,
            )

        return (
            out,
            lens,
        )

    def update_masked_length(
        self,
        max_len,
        seq_range=None,
        device=None,
    ):
        if seq_range is None:
            (
                self.lens,
                self.max_len,
            ) = _masked_conv_init_lens(
                self.lens,
                max_len,
                self.max_len,
            )
            self.lens = self.lens.to(device)
        else:
            self.lens = seq_range
            self.max_len = torch.tensor(max_len)

    def mask_input(
        self,
        x,
        lens,
    ):
        max_len = x.size(2)
        mask = self.lens[:max_len].unsqueeze(0).to(lens.device) < lens.unsqueeze(1)
        x = x * mask.unsqueeze(1).to(device=x.device)
        return x


class SqueezeExcite(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int,
        context_window: int = -1,
        interpolation_mode: str = "nearest",
        activation: Optional[Callable] = None,
        quantize: bool = False,
    ):
        """
        Squeeze-and-Excitation sub-module.

        Args:
            channels: Input number of channels.
            reduction_ratio: Reduction ratio for "squeeze" layer.
            context_window: Integer number of timesteps that the context
                should be computed over, using stride 1 average pooling.
                If value < 1, then global context is computed.
            interpolation_mode: Interpolation mode of timestep dimension.
                Used only if context window is > 1.
                The modes available for resizing are: `nearest`, `linear` (3D-only),
                `bilinear`, `area`
            activation: Intermediate activation function used. Must be a
                callable activation function.
        """
        super(
            SqueezeExcite,
            self,
        ).__init__()
        self.interpolation_mode = interpolation_mode
        self._quantize = quantize

        self.pool = None  # prepare a placeholder which will be updated

        if activation is None:
            activation = torch.nn.ReLU(inplace=True)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(
                channels,
                channels // reduction_ratio,
                bias=False,
            ),
            activation,
            torch.nn.Linear(
                channels // reduction_ratio,
                channels,
                bias=False,
            ),
        )
        self.gap = torch.nn.AdaptiveAvgPool1d(1)

        # Set default context window
        self.change_context_window(context_window=context_window)

        # Set default max sequence length
        self.set_max_len(16)

    def forward(
        self,
        x,
        lengths,
    ):
        return self.forward_for_export(
            x,
            lengths,
        )

    def forward_for_export(
        self,
        x,
        lengths,
    ):
        # The use of negative indices on the transpose allow for expanded SqueezeExcite
        max_len = x.shape[-1]
        if max_len > self.max_len:
            self.set_max_len(max_len)
        dtype = x.dtype
        # Computes in float32 to avoid instabilities during training with AMP.
        with torch.cuda.amp.autocast(enabled=False):
            # Create sample mask - 1 represents value, 0 represents pad
            mask = self.make_pad_mask(
                lengths,
                max_audio_length=max_len,
                device=x.device,
            )
            mask = ~mask  # 0 represents value, 1 represents pad
            x = x.float()  # For stable AMP, SE must be computed at fp32.
            x.masked_fill_(
                mask,
                0.0,
            )  # mask padded values explicitly to 0
            y = self._se_pool_step(
                x,
                mask,
            )  # [B, C, 1]
            y = y.transpose(
                1,
                -1,
            )  # [B, 1, C]
            y = self.fc(y)  # [B, 1, C]
            y = y.transpose(
                1,
                -1,
            )  # [B, C, 1]

            # Note: Keep for future, in case we improve WER from doing so.
            # if self.context_window >= 0:
            #     y = F.interpolate(y, size=x.shape[-1], mode=self.interpolation_mode)

            y = torch.sigmoid(y)
            y = x * y
        return (
            y,
            lengths,
        )

    def _se_pool_step(
        self,
        x,
        mask,
    ):
        # Negate mask back to represent 1 for signal and 0 for padded timestep.
        mask = ~mask

        if self.context_window < 0:
            # [B, C, 1] - Masked Average over value + padding.
            y = torch.sum(
                x,
                dim=-1,
                keepdim=True,
            ) / mask.sum(
                dim=-1,
                keepdim=True,
            ).type(x.dtype)
        else:
            # [B, C, 1] - Masked Average over value + padding with limited context.
            # During training randomly subsegments a context_window chunk of timesteps.
            # During inference selects only the first context_window chunk of timesteps.
            if self.training:
                y = _se_pool_step_script_train(
                    x,
                    self.context_window,
                    mask,
                )
            else:
                y = _se_pool_step_script_infer(
                    x,
                    self.context_window,
                    mask,
                )
        return y

    def set_max_len(
        self,
        max_len,
        seq_range=None,
    ):
        """Sets maximum input length.
        Pre-calculates internal seq_range mask.
        """
        self.max_len = max_len
        if seq_range is None:
            device = next(self.parameters()).device
            seq_range = torch.arange(
                0,
                self.max_len,
                device=device,
            )
        if hasattr(
            self,
            "seq_range",
        ):
            self.seq_range = seq_range
        else:
            self.register_buffer(
                "seq_range",
                seq_range,
                persistent=False,
            )

    def make_pad_mask(
        self,
        seq_lens,
        max_audio_length,
        device=None,
    ):
        """Make masking for padding."""
        if device and self.seq_range.device != device:
            self.seq_range = self.seq_range.to(device)
        if self.seq_range.device != seq_lens.device:
            seq_lens = seq_lens.to(self.seq_range.device)

        mask = self.seq_range[:max_audio_length].expand(
            seq_lens.size(0),
            -1,
        ) < seq_lens.unsqueeze(
            -1
        )  # [B, T]; bool
        mask = mask.unsqueeze(1)  # [B, 1, T]

        return mask

    def change_context_window(
        self,
        context_window: int,
    ):
        """
        Update the context window of the SqueezeExcitation module, in-place if possible.

        Will update the pooling layer to either nn.AdaptiveAvgPool1d() (for global SE) or nn.AvgPool1d()
        (for limited context SE).

        If only the context window is changing but still a limited SE context block - then
        the earlier instance of nn.AvgPool1d() will be updated.

        Args:
            context_window: An integer representing the number of input timeframes that will be used
                to compute the context. Each timeframe corresponds to a single window stride of the
                STFT features.

                Say the window_stride = 0.01s, then a context window of 128 represents 128 * 0.01 s
                of context to compute the Squeeze step.
        """
        if hasattr(
            self,
            "context_window",
        ):
            logging.info(
                f"Changing Squeeze-Excitation context window from {self.context_window} to {context_window}"
            )

        self.context_window = context_window


class JasperBlock(torch.nn.Module):
    """
    ML Modules port of the JasperBlock from NeMo.

    Constructs a single "Jasper" block. With modified parameters, also constructs other blocks for models
    such as `QuartzNet` and `Citrinet`.

    - For `Jasper`    : `separable` flag should be False
    - For `QuartzNet` : `separable` flag should be True
    - For `Citrinet`  : `separable` flag and `se` flag should be True

    Note that above are general distinctions, each model has intricate differences that expand over
    multiple such blocks.

    For further information about the differences between models which use JasperBlock, please review
    the configs for ASR models found in the ASR examples directory.

    Args:
        inplanes: Number of input channels.
        planes: Number of output channels.
        repeat: Number of repeated sub-blocks (R) for this block.
        kernel_size (int | tuple): Convolution kernel size across all repeated sub-blocks.
        kernel_size_factor: Floating point scale value that is multiplied with kernel size,
            then rounded down to nearest odd integer to compose the kernel size. Defaults to 1.0.
        stride (int | tuple): Stride of the convolutional layers.
        dilation (int | tuple): Integer which defined dilation factor of kernel. Note that when dilation > 1, stride must
            be equal to 1.
        padding: String representing type of padding. Currently only supports "same" padding,
            which symmetrically pads the input tensor with zeros.
        dropout: Floating point value, determins percentage of output that is zeroed out.
        activation: String representing activation functions. Valid activation functions are :
            {"hardtanh": nn.Hardtanh, "relu": nn.ReLU, "selu": nn.SELU, "swish": Swish}.
            Defaults to "relu".
        residual: Bool that determined whether a residual branch should be added or not.
            All residual branches are constructed using a pointwise convolution kernel, that may or may not
            perform strided convolution depending on the parameter `residual_mode`.
        groups: Number of groups for Grouped Convolutions. Defaults to 1.
        separable: Bool flag that describes whether Time-Channel depthwise separable convolution should be
            constructed, or ordinary convolution should be constructed.
        heads: Number of "heads" for the masked convolution. Defaults to -1, which disables it.
        normalization: String that represents type of normalization performed. Can be one of
            "batch", "group", "instance" or "layer" to compute BatchNorm1D, GroupNorm1D, InstanceNorm or
            LayerNorm (which are special cases of GroupNorm1D).
        norm_groups: Number of groups used for GroupNorm (if `normalization` == "group").
        residual_mode: String argument which describes whether the residual branch should be simply
            added ("add") or should first stride, then add ("stride_add"). Required when performing stride on
            parallel branch as well as utilizing residual add.
        residual_panes: Number of residual panes, used for Jasper-DR models. Please refer to the paper.
        conv_mask: Bool flag which determines whether to utilize masked convolutions or not. In general,
            it should be set to True.
        se: Bool flag that determines whether Squeeze-and-Excitation layer should be used.
        se_reduction_ratio: Integer value, which determines to what extend the hidden dimension of the SE
            intermediate step should be reduced. Larger values reduce number of parameters, but also limit
            the effectiveness of SE layers.
        se_context_window: Integer value determining the number of timesteps that should be utilized in order
            to compute the averaged context window. Defaults to -1, which means it uses global context - such
            that all timesteps are averaged. If any positive integer is used, it will utilize limited context
            window of that size.
        se_interpolation_mode: String used for interpolation mode of timestep dimension for SE blocks.
            Used only if context window is > 1.
            The modes available for resizing are: `nearest`, `linear` (3D-only),
            `bilinear`, `area`.
        stride_last: Bool flag that determines whether all repeated blocks should stride at once,
            (stride of S^R when this flag is False) or just the last repeated block should stride
            (stride of S when this flag is True).
        future_context: Int value that determins how many "right" / "future" context frames will be utilized
            when calculating the output of the conv kernel. All calculations are done for odd kernel sizes only.

            By default, this is -1, which is recomputed as the symmetric padding case.

            When future_context >= 0, will compute the asymmetric padding as follows :
            (left context, right context) = [K - 1 - future_context, future_context]

            Determining an exact formula to limit future context is dependent on global layout of the model.
            As such, we provide both "local" and "global" guidelines below.

            Local context limit (should always be enforced)
            - future context should be <= half the kernel size for any given layer
            - future context > kernel size defaults to symmetric kernel
            - future context of layer = number of future frames * width of each frame (dependent on stride)

            Global context limit (should be carefully considered)
            - future context should be layed out in an ever reducing pattern. Initial layers should restrict
            future context less than later layers, since shallow depth (and reduced stride) means each frame uses
            less amounts of future context.
            - Beyond a certain point, future context should remain static for a given stride level. This is
            the upper bound of the amount of future context that can be provided to the model on a global scale.
            - future context is calculated (roughly) as - (2 ^ stride) * (K // 2) number of future frames.
            This resultant value should be bound to some global maximum number of future seconds of audio (in ms).

            Note: In the special case where K < future_context, it is assumed that the kernel is too small to limit
            its future context, so symmetric padding is used instead.

            Note: There is no explicit limitation on the amount of future context used, as long as
            K > future_context constraint is maintained. This might lead to cases where future_context is
            more than half the actual kernel size K! In such cases, the conv layer is utilizing more of the future
            context than its current and past context to compute the output. While this is possible to do,
            it is not recommended and the layer will raise a warning to notify the user of such cases.
            It is advised to simply use symmetric padding for such cases.

            Example:
            Say we have a model that performs 8x stride and receives spectrogram frames with stride of 0.01s.
            Say we wish to upper bound future context to 80 ms.

            Layer ID, Kernel Size, Stride, Future Context, Global Context
            0, K=5,  S=1, FC=8, GC= 2 * (2^0) = 2 * 0.01 ms  (special case, K < FC so use symmetric pad)
            1, K=7,  S=1, FC=3, GC= 3 * (2^0) = 3 * 0.01 ms  (note that symmetric pad here uses 3 FC frames!)
            2, K=11, S=2, FC=4, GC= 4 * (2^1) = 8 * 0.01 ms  (note that symmetric pad here uses 5 FC frames!)
            3, K=15, S=1, FC=4, GC= 4 * (2^1) = 8 * 0.01 ms  (note that symmetric pad here uses 7 FC frames!)
            4, K=21, S=2, FC=2, GC= 2 * (2^2) = 8 * 0.01 ms  (note that symmetric pad here uses 10 FC frames!)
            5, K=25, S=2, FC=1, GC= 1 * (2^3) = 8 * 0.01 ms  (note that symmetric pad here uses 14 FC frames!)
            6, K=29, S=1, FC=1, GC= 1 * (2^3) = 8 * 0.01 ms ...
        quantize: Bool flag whether to quantize the Convolutional blocks.
        layer_idx (int, optional): can be specified to allow layer output capture for InterCTC loss. Defaults to -1.
    """

    __constants__ = [
        "conv_mask",
        "separable",
        "residual_mode",
        "res",
        "mconv",
    ]

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int | tuple = 11,
        repeat: int = 1,
        kernel_size_factor: int = 1,
        stride: int | tuple = 1,
        dilation: int | tuple = 1,
        padding: str = "same",
        dropout: float = 0.2,
        activation: str = None,
        residual: bool = True,
        groups: int = 1,
        separable: bool = False,
        heads: int = -1,
        normalization: str = "batch",
        norm_groups: int = 1,
        residual_mode: str = "add",
        residual_panes: list = [],
        conv_mask: bool = False,
        se: bool = False,
        se_reduction_ratio: int = 16,
        se_context_window: int = -1,
        se_interpolation_mode: str = "nearest",
        stride_last: bool = False,
        future_context: int = -1,
        quantize: bool = False,
        layer_idx: int = -1,  # only used for capturing tensors for interctc loss
    ):
        super(
            JasperBlock,
            self,
        ).__init__()

        if quantize:
            raise ValueError("Quantization is not supported")

        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")

        # If kernel_size, stride and/or dilation are integers, convert them to iterable lists
        # to comply with the rest of the code.
        if not isinstance(
            kernel_size,
            Iterable,
        ):
            kernel_size = [kernel_size]
        if not isinstance(
            stride,
            Iterable,
        ):
            stride = [stride]
        if not isinstance(
            dilation,
            Iterable,
        ):
            dilation = [dilation]

        kernel_size_factor = float(kernel_size_factor)
        # Note: kernel_size can be a tuple of the kernel dimensions
        if isinstance(
            kernel_size,
            Iterable,
        ):
            kernel_size = [
                compute_new_kernel_size(
                    k,
                    kernel_size_factor,
                )
                for k in kernel_size
            ]
        else:
            kernel_size = [
                compute_new_kernel_size(
                    kernel_size,
                    kernel_size_factor,
                )
            ]

        if future_context < 0:
            padding_val = get_same_padding(
                kernel_size[0],
                stride[0],
                dilation[0],
            )
        else:
            padding_val = get_asymtric_padding(
                kernel_size[0],
                stride[0],
                dilation[0],
                future_context,
            )

        self.inplanes = inplanes
        self.planes = planes
        self.conv_mask = conv_mask
        self.separable = separable
        self.residual_mode = residual_mode
        self.se = se
        self.quantize = quantize
        self.layer_idx = layer_idx
        # will be set in self.forward() if defined in AccessMixin config
        self.interctc_should_capture = None

        inplanes_loop = inplanes
        conv = torch.nn.ModuleList()

        for _ in range(repeat - 1):
            # Stride last means only the last convolution in block will have stride
            if stride_last:
                stride_val = [1]
            else:
                stride_val = stride

            conv.extend(
                self._get_conv_bn_layer(
                    inplanes_loop,
                    planes,
                    kernel_size=kernel_size,
                    stride=stride_val,
                    dilation=dilation,
                    padding=padding_val,
                    groups=groups,
                    heads=heads,
                    separable=separable,
                    normalization=normalization,
                    norm_groups=norm_groups,
                    quantize=quantize,
                )
            )

            conv.extend(
                self._get_act_dropout_layer(
                    drop_prob=dropout,
                    activation=activation,
                )
            )

            inplanes_loop = planes

        conv.extend(
            self._get_conv_bn_layer(
                inplanes_loop,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding_val,
                groups=groups,
                heads=heads,
                separable=separable,
                normalization=normalization,
                norm_groups=norm_groups,
                quantize=quantize,
            )
        )

        if se:
            conv.append(
                SqueezeExcite(
                    planes,
                    reduction_ratio=se_reduction_ratio,
                    context_window=se_context_window,
                    interpolation_mode=se_interpolation_mode,
                    activation=activation,
                    quantize=quantize,
                )
            )

        self.mconv = conv

        res_panes = residual_panes.copy()
        self.dense_residual = residual

        if residual:
            res_list = torch.nn.ModuleList()

            if residual_mode == "stride_add":
                stride_val = stride
            else:
                stride_val = [1]

            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                res = torch.nn.ModuleList(
                    self._get_conv_bn_layer(
                        ip,
                        planes,
                        kernel_size=1,
                        normalization=normalization,
                        norm_groups=norm_groups,
                        stride=stride_val,
                        quantize=quantize,
                    )
                )

                res_list.append(res)

            self.res = res_list
        else:
            self.res = None

        self.mout = torch.nn.Sequential(
            *self._get_act_dropout_layer(
                drop_prob=dropout,
                activation=activation,
            )
        )

    def _get_conv(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        stride=1,
        dilation=1,
        padding=0,
        bias=False,
        groups=1,
        heads=-1,
        separable=False,
        quantize=False,
    ):
        use_mask = self.conv_mask
        if use_mask:
            return MaskedConv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias,
                groups=groups,
                heads=heads,
                use_mask=use_mask,
                quantize=quantize,
            )
        else:
            return torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias,
                groups=groups,
            )

    def _get_conv_bn_layer(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        stride=1,
        dilation=1,
        padding=0,
        bias=False,
        groups=1,
        heads=-1,
        separable=False,
        normalization="batch",
        norm_groups=1,
        quantize=False,
    ):
        if norm_groups == -1:
            norm_groups = out_channels

        if separable:
            layers = [
                self._get_conv(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=in_channels,
                    heads=heads,
                    quantize=quantize,
                ),
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    padding=0,
                    bias=bias,
                    groups=groups,
                    quantize=quantize,
                ),
            ]
        else:
            layers = [
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=groups,
                    quantize=quantize,
                )
            ]

        if normalization == "group":
            layers.append(
                torch.nn.GroupNorm(
                    num_groups=norm_groups,
                    num_channels=out_channels,
                )
            )
        elif normalization == "instance":
            layers.append(
                torch.nn.GroupNorm(
                    num_groups=out_channels,
                    num_channels=out_channels,
                )
            )
        elif normalization == "layer":
            layers.append(
                torch.nn.GroupNorm(
                    num_groups=1,
                    num_channels=out_channels,
                )
            )
        elif normalization == "batch":
            layers.append(
                torch.nn.BatchNorm1d(
                    out_channels,
                    eps=1e-3,
                    momentum=0.1,
                )
            )
        else:
            raise ValueError(
                f"Normalization method ({normalization}) does not match"
                f" one of [batch, layer, group, instance]."
            )

        if groups > 1:
            layers.append(
                GroupShuffle(
                    groups,
                    out_channels,
                )
            )
        return layers

    def _get_act_dropout_layer(
        self,
        drop_prob=0.2,
        activation=None,
    ):
        if activation is None:
            activation = torch.nn.Hardtanh(
                min_val=0.0,
                max_val=20.0,
            )
        layers = [
            activation,
            torch.nn.Dropout(p=drop_prob),
        ]
        return layers

    def forward(
        self,
        input_: Tuple[
            List[torch.Tensor],
            Optional[torch.Tensor],
        ],
    ) -> Tuple[
        List[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Forward pass of the module.

        Args:
            input_: The input is a tuple of two values - the preprocessed audio signal as well as the lengths
                of the audio signal. The audio signal is padded to the shape [B, D, T] and the lengths are
                a torch vector of length B.

        Returns:
            The output of the block after processing the input through `repeat` number of sub-blocks,
            as well as the lengths of the encoded audio after padding/striding.
        """
        lens_orig = None
        xs = input_[0]
        if len(input_) == 2:
            (
                xs,
                lens_orig,
            ) = input_

        # compute forward convolutions
        out = xs[-1]

        lens = lens_orig
        for (
            i,
            l,
        ) in enumerate(self.mconv):
            # if we're doing masked convolutions, we need to pass in and
            # possibly update the sequence lengths
            # if (i % 4) == 0 and self.conv_mask:
            if isinstance(
                l,
                (
                    MaskedConv1d,
                    SqueezeExcite,
                ),
            ):
                (
                    out,
                    lens,
                ) = l(
                    out,
                    lens,
                )
            else:
                out = l(out)

        # compute the residuals
        if self.res is not None:
            for (
                i,
                layer,
            ) in enumerate(self.res):
                res_out = xs[i]
                for (
                    j,
                    res_layer,
                ) in enumerate(layer):
                    if isinstance(
                        res_layer,
                        MaskedConv1d,
                    ):
                        (
                            res_out,
                            _,
                        ) = res_layer(
                            res_out,
                            lens_orig,
                        )
                    else:
                        res_out = res_layer(res_out)

                if self.residual_mode == "add" or self.residual_mode == "stride_add":
                    out = out + res_out
                else:
                    out = torch.max(
                        out,
                        res_out,
                    )

        # compute the output
        out = self.mout(out)

        if self.res is not None and self.dense_residual:
            return (
                xs + [out],
                lens,
            )

        return (
            [out],
            lens,
        )


class GroupShuffle(torch.nn.Module):
    def __init__(
        self,
        groups,
        channels,
    ):
        super(
            GroupShuffle,
            self,
        ).__init__()

        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(
        self,
        x,
    ):
        sh = x.shape

        x = x.view(
            -1,
            self.groups,
            self.channels_per_group,
            sh[-1],
        )

        x = torch.transpose(
            x,
            1,
            2,
        ).contiguous()

        x = x.view(
            -1,
            self.groups * self.channels_per_group,
            sh[-1],
        )

        return x


def tds_uniform_(
    tensor,
    mode="fan_in",
):
    """
    Uniform Initialization from the paper [Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2460.pdf)
    Normalized to -

    .. math::
        \\text{bound} = \\text{2} \\times \\sqrt{\\frac{1}{\\text{fan\\_mode}}}

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    """
    fan = torch.nn.init._calculate_correct_fan(
        tensor,
        mode,
    )
    gain = 2.0  # sqrt(4.0) = 2
    std = gain / math.sqrt(fan)  # sqrt(4.0 / fan_in)
    bound = std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(
            -bound,
            bound,
        )


def tds_normal_(
    tensor,
    mode="fan_in",
):
    """
    Normal Initialization from the paper [Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2460.pdf)
    Normalized to -

    .. math::
        \\text{bound} = \\text{2} \\times \\sqrt{\\frac{1}{\\text{fan\\_mode}}}

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    """
    fan = torch.nn.init._calculate_correct_fan(
        tensor,
        mode,
    )
    gain = 2.0
    std = gain / math.sqrt(fan)  # sqrt(4.0 / fan_in)
    bound = std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.normal_(
            0.0,
            bound,
        )


def init_weights(
    m,
    mode: Optional[str] = "xavier_uniform",
):
    if isinstance(
        m,
        MaskedConv1d,
    ):
        init_weights(
            m.conv,
            mode,
        )
    if isinstance(
        m,
        (
            torch.nn.Conv1d,
            torch.nn.Linear,
        ),
    ):
        if mode is not None:
            if mode == "xavier_uniform":
                torch.nn.init.xavier_uniform_(
                    m.weight,
                    gain=1.0,
                )
            elif mode == "xavier_normal":
                torch.nn.init.xavier_normal_(
                    m.weight,
                    gain=1.0,
                )
            elif mode == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(
                    m.weight,
                    nonlinearity="relu",
                )
            elif mode == "kaiming_normal":
                torch.nn.init.kaiming_normal_(
                    m.weight,
                    nonlinearity="relu",
                )
            elif mode == "tds_uniform":
                tds_uniform_(m.weight)
            elif mode == "tds_normal":
                tds_normal_(m.weight)
            else:
                raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif isinstance(
        m,
        torch.nn.BatchNorm1d,
    ):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)


def compute_new_kernel_size(
    kernel_size,
    kernel_width,
):
    new_kernel_size = max(
        int(kernel_size * kernel_width),
        1,
    )
    # If kernel is even shape, round up to make it odd
    if new_kernel_size % 2 == 0:
        new_kernel_size += 1
    return new_kernel_size


def get_same_padding(
    kernel_size,
    stride,
    dilation,
) -> int:
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (dilation * (kernel_size - 1)) // 2


def get_asymtric_padding(
    kernel_size,
    stride,
    dilation,
    future_context,
):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")

    left_context = kernel_size - 1 - future_context
    right_context = future_context

    symmetric_padding = get_same_padding(
        kernel_size,
        stride,
        dilation,
    )

    if kernel_size <= future_context:
        # kernel size is smaller than future context, equivalent to using entire context of kernel
        # simply return symmetric padding for this scenario
        logging.warning(
            f"Future context window is larger than the kernel size!\n"
            f"Left context = {left_context} | Right context = greater than {right_context} | "
            f"Kernel size = {kernel_size}\n"
            f"Switching to symmetric padding (left context = right context = {symmetric_padding})"
        )
        return symmetric_padding

    if left_context < symmetric_padding:
        logging.warning(
            f"Future context window is larger than half the kernel size!\n"
            f"Conv layer therefore uses more future information than past to compute its output!\n"
            f"Left context = {left_context} | Right context = {right_context} | "
            f"Kernel size = {kernel_size}"
        )

    if dilation > 1:
        left_context = dilation * kernel_size - 1 - dilation * future_context
        right_context = dilation * future_context
        return (
            left_context,
            right_context,
        )

    return (
        left_context,
        right_context,
    )


@torch.jit.script
def _se_pool_step_script_train(
    x: torch.Tensor,
    context_window: int,
    mask: torch.Tensor,
):
    """
    Calculates the masked average over padded limited context segment during training mode.
    Randomly slices a segment of length `context_window` from signal+padded input tensor across all channels and
    uses it for computing masked limited context.

    Args:
        x: Input tensor. Shape = [B, C, T]
        context_window: Integer context window, must be 0 or greater.
        mask: Mask tensor, 1 represents value index, 0 represents padded index. Shape = [B, 1, T].

    Returns:
        A tensor reduced via masked average pool over some limited context. Shape = [B, C, 1]
    """
    timesteps = x.shape[-1]
    if timesteps < context_window:
        y = torch.sum(
            x,
            dim=-1,
            keepdim=True,
        ) / mask.sum(
            dim=-1,
            keepdim=True,
        ).to(x.dtype)
    else:
        start_idx = torch.randint(
            0,
            timesteps - context_window,
            size=[1],
            dtype=torch.int32,
        )[0]
        x = x[
            :,
            :,
            start_idx : (start_idx + context_window),
        ]  # [B, C, context_window]
        mask = mask[
            :,
            :,
            start_idx : (start_idx + context_window),
        ]  # [B, 1, context_window]

        mask = mask.sum(
            dim=-1,
            keepdim=True,
        ).to(
            x.dtype
        )  # [B, C, 1]
        y = x.sum(
            dim=-1,
            keepdim=True,
        )  # [B, 1, 1]
        y = y / (mask + 1e-8)  # [B, C, 1]

    return y


@torch.jit.script
def _se_pool_step_script_infer(
    x: torch.Tensor,
    context_window: int,
    mask: torch.Tensor,
):
    """
    Calculates the masked average over padded limited context segment during inference mode.

    Args:
        x: Input tensor. Shape = [B, C, T]
        context_window: Integer context window, must be 0 or greater.
        mask: Mask tensor, 1 represents value index, 0 represents padded index. Shape = [B, 1, T].

    Returns:
        A tensor reduced via masked average pool over some limited context. Shape = [B, C, 1]
    """
    timesteps = x.shape[-1]
    if timesteps < context_window:
        y = torch.sum(
            x,
            dim=-1,
            keepdim=True,
        ) / mask.sum(
            dim=-1,
            keepdim=True,
        ).to(x.dtype)
    else:
        # << During inference prefer to use entire context >>
        # x = x[:, :, :context_window]  # [B, C, context_window]
        # mask = mask[:, :, :context_window]  # [B, 1, context_window]
        #
        # mask = mask.sum(dim=-1, keepdim=True).to(x.dtype)  # [B, C, 1]
        # y = x.sum(dim=-1, keepdim=True)  # [B, 1, 1]
        # y = y / (mask + 1e-8)  # [B, C, 1]
        y = torch.sum(
            x,
            dim=-1,
            keepdim=True,
        ) / mask.sum(
            dim=-1,
            keepdim=True,
        ).to(x.dtype)

    return y


@torch.jit.script
def _masked_conv_init_lens(
    lens: torch.Tensor,
    current_maxlen: int,
    original_maxlen: torch.Tensor,
):
    if current_maxlen > original_maxlen:
        new_lens = torch.arange(current_maxlen)
        new_max_lens = torch.tensor(current_maxlen)
    else:
        new_lens = lens
        new_max_lens = original_maxlen
    return (
        new_lens,
        new_max_lens,
    )
