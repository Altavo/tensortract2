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
from transformers import WavLMModel

from .utils import create_mask_same_shape


class SSL(torch.nn.Module):
    def __init__(
            self,
            ssl_model,
            ):
        super(SSL,self).__init__()

        self.model = ssl_model

        if self.model.config.feat_extract_norm != "layer":
            raise ValueError("Model must use layer norm.")

        return

    def forward(
            self,
            x: torch.Tensor,
            x_len: torch.Tensor,
            layer: int = -1,
            ):
        # Expected shape: (B, T)
        assert x.dim() == 2

        x_mask = create_mask_same_shape(
                x,
                x_len,
            )

        outputs = self.model(
            x,
            x_mask,
            output_hidden_states=True,
        )
        x = outputs.hidden_states[layer]

        x = x.transpose(1, 2)

        # convert xlen from 16000 hz to 50 hz as integer
        x_len = x_len // 320

        # clip x_len to max length
        x_len = torch.clamp(x_len, max=x.size(2))

        return x, x_len


class WavLM( SSL ):
    def __init__(
            self,
            **kwargs,
            ):
        super(WavLM,self).__init__(
            ssl_model = WavLMModel.from_pretrained(
                pretrained_model_name_or_path = "microsoft/wavlm-large",
                **kwargs,
                )
            )
        return
    