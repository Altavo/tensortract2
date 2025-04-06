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
https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/modules/conv_asr.py

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
    Optional,
    Union,
)
from .nemo_tdnn import (
    AttentivePoolLayer,
    StatsPoolLayer,
)
from .nemo_jasper import (
    init_weights,
)

""" ML Module port of the SpeakerDecoder from the NeMo toolkit without dependencies on the
    following classes: NeuralModule, Exportable
    Removed the following features:
        - quantization (via the PYTORCH_QUANTIZATION_AVAILABLE flag)
        - input type checking
    Source:
        https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/modules/conv_asr.py
    """


class SpeakerDecoder(torch.nn.Module):
    """
    Speaker Decoder creates the final neural layers that maps from the outputs
    of Jasper Encoder to the embedding layer followed by speaker based softmax loss.
    Args:
        feat_in (int): Number of channels being input to this module
        num_classes (int): Number of unique speakers in dataset
        emb_sizes (list) : shapes of intermediate embedding layers (we consider speaker embbeddings from 1st of this layers)
                Defaults to [1024,1024]
        pool_mode (str) : Pooling strategy type. options are 'xvector','tap', 'attention'
                Defaults to 'xvector (mean and variance)'
                tap (temporal average pooling: just mean)
                attention (attention based pooling)
        angular (bool): whether to normalize the final_layer weights and embeddings to 1.
        attention_channels (int): Number of channels in the attention layer.
        init_mode (str): Describes how neural network parameters are
            initialized. Options are ['xavier_uniform', 'xavier_normal',
            'kaiming_uniform','kaiming_normal'].
            Defaults to "xavier_uniform".
    """

    def input_example(
        self,
        max_batch=1,
        max_dim=256,
    ):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        input_example = torch.randn(
            max_batch,
            self.feat_in,
            max_dim,
        ).to(next(self.parameters()).device)
        return tuple([input_example])

    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        emb_sizes: Optional[
            Union[
                int,
                list,
            ]
        ] = 256,
        pool_mode: str = "xvector",
        angular: bool = False,
        attention_channels: int = 128,
        init_mode: str = "xavier_uniform",
    ):
        super().__init__()

        self.feat_in = feat_in
        self.angular = angular
        self.emb_id = 2
        bias = False if self.angular else True
        emb_sizes = [emb_sizes] if type(emb_sizes) is int else emb_sizes

        self._num_classes = num_classes
        self.pool_mode = pool_mode.lower()
        if self.pool_mode == "xvector" or self.pool_mode == "tap":
            self._pooling = StatsPoolLayer(
                feat_in=feat_in,
                pool_mode=self.pool_mode,
            )
            affine_type = "linear"
        elif self.pool_mode == "attention":
            self._pooling = AttentivePoolLayer(
                inp_filters=feat_in,
                attention_channels=attention_channels,
            )
            affine_type = "conv"

        shapes = [self._pooling.feat_in]
        for size in emb_sizes:
            shapes.append(int(size))

        emb_layers = []
        for (
            shape_in,
            shape_out,
        ) in zip(
            shapes[:-1],
            shapes[1:],
        ):
            layer = self.affine_layer(
                shape_in,
                shape_out,
                learn_mean=False,
                affine_type=affine_type,
            )
            emb_layers.append(layer)

        self.emb_layers = torch.nn.ModuleList(emb_layers)

        self.final = torch.nn.Linear(
            shapes[-1],
            self._num_classes,
            bias=bias,
        )

        self.apply(
            lambda x: init_weights(
                x,
                mode=init_mode,
            )
        )

    def affine_layer(
        self,
        inp_shape,
        out_shape,
        learn_mean=True,
        affine_type="conv",
    ):
        if affine_type == "conv":
            layer = torch.nn.Sequential(
                torch.nn.BatchNorm1d(
                    inp_shape,
                    affine=True,
                    track_running_stats=True,
                ),
                torch.nn.Conv1d(
                    inp_shape,
                    out_shape,
                    kernel_size=1,
                ),
            )

        else:
            layer = torch.nn.Sequential(
                torch.nn.Linear(
                    inp_shape,
                    out_shape,
                ),
                torch.nn.BatchNorm1d(
                    out_shape,
                    affine=learn_mean,
                    track_running_stats=True,
                ),
                torch.nn.ReLU(),
            )

        return layer

    def forward(
        self,
        encoder_output,
        lengths=None,
    ):
        """Returns the output of the decoder.
        Args:
            encoder_output (torch.Tensor): Output of the encoder. Contains
                the encoder output, which are the raw logits (batch_size, num_classes)
                and the embeddings (batch_size, emb_sizes).
            lengths (torch.Tensor): Length of the input (batch_size,)
        Returns:
            out (torch.Tensor): Output of the decoder. Contains the raw logits (batch_size, num_classes).
            embs (torch.Tensor): Embeddings of the last emb layer stack of the decoder (batch_size, emb_sizes),
                before the final linear layer.
        """
        pool = self._pooling(
            encoder_output,
            lengths,
        )  # pool unequal length sequences to fixed size vectors
        embs = []

        # Note: afaik, the decoder can have stacked emb_layers (Attn, BN, linear - blocks),
        # but the Titanet SpeakerDecoder only has one.
        for layer in self.emb_layers:
            (
                pool,
                emb,
            ) = layer(
                pool
            ), layer[: self.emb_id](
                pool
            )  # pool == emb for emb_id=2, but pool is used for logits in final layer
            embs.append(emb)

        pool = pool.squeeze(-1)
        if self.angular:
            for W in self.final.parameters():
                W = torch.nn.functional.normalize(
                    W,
                    p=2,
                    dim=1,
                )
            pool = torch.nn.functional.normalize(
                pool,
                p=2,
                dim=1,
            )

        logits = self.final(pool)
        last_emb_layer = embs[-1].squeeze(-1)

        return (
            logits,
            last_emb_layer,
        )
