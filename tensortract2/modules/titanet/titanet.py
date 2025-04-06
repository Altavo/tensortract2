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

from .nemo_audioprocessing import AudioToMelSpectrogramPreprocessor as Wav2Mel
from .nemo_encoder import ConvASREncoder
from .nemo_decoder import SpeakerDecoder


class TitaNet( torch.nn.Module ):
    def __init__(
            self,
            wav2mel_kwargs: dict,
            encoder_kwargs: dict,
            decoder_kwargs: dict,
            ):
        super(TitaNet, self).__init__()
        self.preprocess = Wav2Mel( **wav2mel_kwargs )
        self.encoder = ConvASREncoder( **encoder_kwargs )
        self.decoder = SpeakerDecoder( **decoder_kwargs )
        return

    def forward(
            self,
            x: torch.Tensor,
            x_len: torch.Tensor,
            ):
        """
        Args:
            x: torch.Tensor (B, T)
            x_len: torch.Tensor (B,)

        Returns:
            logits: torch.Tensor (B, num_classes)
            embedding: torch.Tensor (B, emb_sizes)
        """
        
        mel, mel_len = self.preprocess(
            input_signal = x,
            length = x_len,
            )
        z, z_len = self.encoder(
            audio_features = mel,
            lengths = mel_len,
            )
        logits, embedding = self.decoder(
            encoder_output = z,
            lengths = z_len,
            )
        
        return logits, embedding