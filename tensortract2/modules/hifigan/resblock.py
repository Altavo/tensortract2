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

from typing import Any, Mapping



LRELU_SLOPE = 0.1


class ResBlock( BaseTCN ):
    def __init__(
            self,
            channels,
            kernel_size,
            dilation,
            resblock_type,
            padding_mode,
            causal,
            ):
        super(ResBlock, self).__init__()

        self.resblock_type = resblock_type
        if resblock_type not in [1,2]:
            raise ValueError()
        
        self.convs1 = nn.ModuleList(
            [
            weight_norm(
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=d,
                    padding_mode=padding_mode,
                    causal=causal,
                    )
                )
            for d in dilation
            ]
        )

        if resblock_type == 1:
            self.convs2 = nn.ModuleList([
                weight_norm(
                    Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=1,
                        padding_mode=padding_mode,
                        causal=causal,
                        )
                    )
                for _ in dilation
            ]
        )

        self.init_weights()
        return
    
    def forward(
            self,
            x,
            **kwargs,
            ):
        if self.resblock_type == 1:
            for c1, c2 in zip(self.convs1, self.convs2):
                xt = F.leaky_relu(x, LRELU_SLOPE)
                xt = c1( x=xt, **kwargs )
                xt = F.leaky_relu(xt, LRELU_SLOPE)
                xt = c2( x=xt, **kwargs )
                x = xt + x

        elif self.resblock_type == 2:
            for c in self.convs1:
                xt = F.leaky_relu(x, LRELU_SLOPE)
                xt = c( x=xt, **kwargs )
                x = xt + x

        else:
            raise ValueError(
                f"Invalid resblock_type: {self.resblock_type}"
                )
        return x
    

class MultiReceptiveFieldFusion( BaseTCN ):
    def __init__(
            self,
            channels,
            kernel_sizes,
            dilations,
            resblock_type,
            padding_mode,
            causal,
            ):
        super(MultiReceptiveFieldFusion, self).__init__()

        self.num_blocks = len(kernel_sizes)

        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            self.blocks += [
                ResBlock(
                    channels = channels,
                    kernel_size = kernel_sizes[i],
                    dilation = dilations[i],
                    resblock_type = resblock_type,
                    padding_mode = padding_mode,
                    causal = causal,
                    )
                ]
            
        return

    def forward(
            self,
            x,
            **kwargs,
            ):
        xs = 0.0
        for i in range(self.num_blocks):
            xs += self.blocks[i](
                x = x,
                **kwargs,
                )
        x = xs / self.num_blocks
        return x