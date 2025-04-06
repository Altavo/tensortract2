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

import torch

from .conformer import Conformer
from .utils import create_mask


class ConformerEncoder( torch.nn.Module ):
    def __init__(
            self,
            in_channels: int = 1024,
            hidden_channels: int = 192,
            out_channels: int = 20,
            use_rnn: bool = True,
            num_layers: int = 8,
            num_heads: int = 4,
            dropout: float = 0.1,
            use_tanh: bool = True,
            **kwargs,
            ):
        super().__init__( **kwargs )

        self.input_projection = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            )
        
        self.conformer_block = Conformer(
            input_dim=hidden_channels,
            num_heads=num_heads,
            ffn_dim=4*hidden_channels,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout,
            use_group_norm=False,
            convolution_first=False,
            )

        self.lstm = torch.nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            )
        
        D = 2 if use_rnn else 1
        
        self.linear = torch.nn.Conv1d(
            in_channels=D*hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            ) # for bidirectional LSTM
        


        self.output_activation = torch.nn.Tanh()

        self.use_rnn = use_rnn
        self.use_tanh = use_tanh

        self.init_weights()
        return
    
    def forward(
            self,
            x,
            x_len,
            ):
        # clip length of x to the maximum length provided in x_len
        x = x[ ..., :x_len.max() ]
        mask = create_mask( x, x_len )

        x = self.input_projection( x )
        x = x.transpose(1, 2)
        x, _ = self.conformer_block( x, x_len )
        if self.use_rnn:
            x, _ = self.lstm( x )
        x = x.transpose(1, 2)
        x = self.linear( x )
        if self.use_tanh:
            x = self.output_activation( x )

        # multiply y by mask
        x = x * mask

        return x, x_len

    def init_weights(self):
        # init all weights of self
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                # use standard normal distribution
                torch.nn.init.normal_(m.weight, 0.0, 0.01)
        return