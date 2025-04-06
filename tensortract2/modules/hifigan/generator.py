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
This implementation is based on the following repository:
https://github.com/jik876/hifi-gan

The original code is licensed under the MIT License:

MIT License

Copyright (c) 2020 Jungil Kong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import warnings
import torch
import torch.nn.functional as F
import torch.nn as nn

# Try to import new weight_norm from torch.nn.utils.parametrizations
# But also keep the deprecated version for compatibility
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm
    warnings.warn(
        """
        The deprecated weight_norm from torch.nn.utils.weight_norm was imported.
        Update your PyTorch version to get rid of this warning.
        """
        )

from pytorch_tcn.tcn import BaseTCN
from pytorch_tcn import TemporalConv1d as Conv1d
from pytorch_tcn import TemporalConvTranspose1d as ConvTranspose1d

from .resblock import MultiReceptiveFieldFusion
from .resblock import LRELU_SLOPE
from .condition import Condition

from typing import Optional
from typing import Union
from typing import List


  
class HifiGenerator( BaseTCN ):
    def __init__(
            self,
            in_channels: int,
            out_channels: int = 1,
            pre_conv_kernel_size: int = 7,
            post_conv_kernel_size: int = 7,
            upsample_initial_channel: int = 512,
            upsample_rates: List[ int ] = [10,8,2,2],
            upsample_kernel_sizes: List[ int ] = [20,16,4,4],
            resblock_kernel_sizes: List[ int ] = [3,7,11],
            resblock_dilation_sizes: List[ List[ int ] ] = [
                [1,3,5],
                [1,3,5],
                [1,3,5],
                ],
            resblock_type = 1,
            dim_cond: Optional[ int ] = None,
            mode_cond: str = 'add',
            padding_mode: str = 'zeros',
            causal: bool = False,
            ):
        super(HifiGenerator, self).__init__()

        self.pre_conv_kernel_size = pre_conv_kernel_size
        self.post_conv_kernel_size = post_conv_kernel_size
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.resblock_type = resblock_type
        self.dim_cond = dim_cond
        self.mode_cond = mode_cond
        self.padding_mode = padding_mode
        self.causal = causal

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        
        self.conv_pre = weight_norm(
            Conv1d(
                in_channels=in_channels,
                out_channels=upsample_initial_channel,
                kernel_size=pre_conv_kernel_size,
                stride=1,
                padding_mode=padding_mode,
                causal=causal,
                )
        )

        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i, (s, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch = upsample_initial_channel // (2 ** (i + 1))

            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        in_channels = upsample_initial_channel // (2 ** i),
                        #out_channels = upsample_initial_channel // (2 ** (i + 1)),
                        out_channels = ch,
                        kernel_size = k,
                        stride = s,
                        padding_mode = padding_mode,
                        causal=causal,
                    )
                )
            )

            self.blocks.append(
                MultiReceptiveFieldFusion(
                    channels = ch,
                    kernel_sizes = resblock_kernel_sizes,
                    dilations = resblock_dilation_sizes,
                    resblock_type = resblock_type,
                    padding_mode = padding_mode,
                    causal = causal,
                )
            )

    
        self.conv_post = weight_norm(
            Conv1d(
                in_channels=ch,
                out_channels=out_channels,
                kernel_size=post_conv_kernel_size,
                stride=1,
                padding_mode=padding_mode,
                causal=causal,
                )
            )
        
        if dim_cond is not None:
            self.cond = Condition(
                dim=upsample_initial_channel,
                dim_cond=dim_cond,
                mode=mode_cond,
                )
        
        self.init_weights()
        self.reset_buffers()
        return
    
    def _condition(
            self,
            x: torch.Tensor,
            condition: Optional[ torch.Tensor ] = None,
            ):
        if condition is not None:
            x = self.cond( x, condition )
        return x
        
    def _forward_blocks(
            self,
            x,
            **kwargs,
            ):
        
        for i in range(self.num_upsamples):
            # Activation
            x = F.leaky_relu(x, LRELU_SLOPE)
            # Upsample
            x = self.ups[i]( x, **kwargs )
            # MRF
            x = self.blocks[i]( x, **kwargs )

        x = F.leaky_relu(x)

        return x

    def _forward_post(
            self,
            x: torch.Tensor,
            **kwargs,
            ):
        
        x = self.conv_post(
            x=x,
            **kwargs,
            )
        x = torch.tanh(x)

        return x

    def forward(
            self,
            x: torch.Tensor,
            cond: Optional[
                Union[
                    torch.Tensor,
                    List[ torch.Tensor ],
                    ]
                ] = None,
            **kwargs,
            ):

        x = self.conv_pre(
            x=x,
            **kwargs,
            )
        
        x = self._condition(
            x=x,
            condition=cond,
            )
        
        x = self._forward_blocks(
            x=x,
            **kwargs,
            )

        x = self._forward_post(
            x=x,
            **kwargs,
            )

        return x