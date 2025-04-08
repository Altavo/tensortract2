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

import os
import torch
import torchaudio
import yaml
import warnings

from typing import Optional
from typing import Union
from typing import List

from .modules.hifigan import HifiGenerator
from .modules.titanet import TitaNet
from .modules.wavlm import WavLM
from .modules.motor_encoder import ConformerEncoder
from .modules.utils import handle_audio_input, get_user_cache_dir
from .modules.utils import download_file_from_google_drive, verify_checksum
from .modules.vtl import MotorProcessor


    
class TensorTract2( torch.nn.Module ):
    def __init__(
            self,
            cfg_path: str = 'tensortract2_version_uc81_am100',
            auto_load_weights: bool = True,
            hf_token: Optional[str] = None,
            ):
        super(TensorTract2,self).__init__()

        # if cfg_path is not an explicit path, treat it as a file name
        # and look for it in the cfg directory.
        if not os.path.exists(cfg_path):
            if not cfg_path.endswith('.yaml'):
                cfg_path = f'{cfg_path}.yaml'
            cfg_path = os.path.join(
                os.path.dirname(__file__),
                'cfg',
                cfg_path,
                )

        # load cfg from yaml file
        with open( cfg_path, 'r' ) as f:
            cfg = yaml.load( f, Loader = yaml.FullLoader )
        self.cfg = cfg

        self.wavlm = WavLM( token = hf_token )
        self.encoder = ConformerEncoder( **cfg[ 'encoder' ] )
        self.generator = HifiGenerator( **cfg[ 'generator' ] )
        self.titanet = TitaNet(
            wav2mel_kwargs = cfg[ 'titanet_wav2mel' ],
            encoder_kwargs = cfg[ 'titanet_encoder' ],
            decoder_kwargs = cfg[ 'titanet_decoder' ],
            )

        self.motor_processor = MotorProcessor()

        if auto_load_weights:
            self.auto_load_weights()

        # Set everything to eval mode
        self.wavlm.eval()
        self.encoder.eval()
        self.generator.eval()
        self.titanet.eval()

        return
        
    def auto_load_weights(self):
        """
        Load the pretrained weights from the cache directory.
        """
    
        user_cache_dir = get_user_cache_dir()
        p = os.path.join(
            user_cache_dir,
            'tensortract2',
            self.cfg[ 'weights' ]['file_name'],
            )
        # check if the weights file exists in user cache directory
        if not os.path.exists(p):
            tt2_file_url = self.cfg[ 'weights' ]['url']
            tt2_file_id = tt2_file_url.split('/')[-2]

            warnings.warn(
                f"""
                Weights file not found in the cache directory: {p}.
                Downloading weights file from {tt2_file_url}.
                """
                )
            # download the weights file from 
            download_file_from_google_drive(
                file_id=tt2_file_id,
                destination=p,
                )
            
        # verify the checksum of the weights file
        if not verify_checksum(p, self.cfg[ 'weights' ]['expected_sha256_checksum']):
            raise ValueError(
                f"""
                The checksum of weights file {p} does not match the expected checksum.
                The file might have been corrupted during the download. Please remove the
                corrupted file and try to download it again.
                """
                )
        else:
            print(f"Checksum of {p} verified successfully.")

        self.load_weights(p)
        return

    def load_weights(
            self,
            p: str,
            ):
        """
        Load the pretrained weights from a file.
        """
        weights = torch.load(p, map_location='cpu', weights_only=True)
        self.encoder.load_state_dict(weights['encoder'], strict=True)
        self.generator.load_state_dict(weights['generator'], strict=True)
        self.titanet.load_state_dict(weights['titanet'], strict=True)
        return

    def encode(
            self,
            x: torch.Tensor,
            x_len: torch.Tensor,
            ):
        """
        Encode input x into a latent representation m.
        
        Args:
            x: torch.Tensor, (B, T). Audio waveform tensor
                at a sample rate of 16000 Hz.
            x_len: torch.Tensor, (B,). Length of each waveform
                tensor in samples.

        Returns:
            m: torch.Tensor, (B,D,T)
            m_len: torch.Tensor, (B,)
        """
        w, w_len = self.wavlm(x,x_len)
        m, m_len = self.encoder(w,w_len)
        return m, m_len
    
    def decode(
            self,
            m: torch.Tensor,
            m_len: torch.Tensor,
            target: torch.tensor,
            target_len: torch.tensor,
            ):
        """
        Decode latent representation m into an audio waveform x.

        Args:
            m: torch.Tensor, (B,D,T). Latent representation tensor, 50Hz.
            m_len: torch.Tensor, (B,). Length of each latent tensor.
            target: torch.Tensor, (B,T). Audio waveform tensor containing 
                the target speaker voice to condition on.
            target_len: torch.Tensor, (B,). Length of each target tensor.

        Returns:
            x: torch.Tensor, (B,T). Audio waveform tensor, 16kHz.
            x_len: torch.Tensor, (B,). Length of each output tensor.
        """
        _, s = self.titanet(target,target_len)
        x = self.generator(m,cond = s)
        # go from m_len (50Hz) to x_len (16000Hz)
        x_len = m_len * 320
        # clip x_len to max length
        x_len = torch.clamp(x_len, max=x.size(2))
        return x, x_len
    
    def forward(
            self,
            x: torch.Tensor,
            x_len: torch.Tensor,
            target: Optional[ torch.Tensor ] = None,
            target_len: Optional[ torch.Tensor ] = None,
            ):
        if target is None:
            target = x
            target_len = x_len
        else:
            assert target_len is not None
        m, m_len = self.encode(x,x_len)
        x, x_len = self.decode(m,m_len,target,target_len)
        return x, x_len
    
    def motor_to_speech(
            self,
            msrs,
            target,
            output: Optional[ Union[ str, List[str] ] ] = None,
            time_stretch: Optional[ float ] = None,
            pitch_shift: Optional[ float ] = None,
            msrs_type: str = 'tt2',
            ):
        """
        Convert motor series objects to speech.

        Args:
            msrs: List[
                Union[
                    target_approximation.vocaltractlab.MotorSeries,
                    target_approximation.tensortract.MotorSeries,
                    ],
                ], List of motor series objects.
            target: Union[str,List[str]], Path to audio file or list of paths.
            output: Union[str,List[str]], Path to output file or list of paths.
            time_stretch: float, Latent time-stretching factor. E.g. to double
                the speed, set to 2. Default is None, which means no time-stretching
                will be applied.
            pitch_shift: float, Latent pitch shift value in semitones. E.g. to shift
                pitch up/down an octave, set to +/- 12. Default is None, which
                means no pitch shift will be applied.
            msrs_type: str, Expected type of motor series input. Must be 'tt2' or 'vtl'.
                'vtl' means a VTL-Python standard (30 articulatory features at 441 Hz).
                'tt2' means a TensorTract2 standard (20 articulatory features at 50 Hz).

        Returns:
            y: List[ torch.Tensor ], (B,T). List of audio waveform tensors, 16kHz.
        """
        if output is not None:
            if isinstance(output,str):
                output = [output]
            assert len(output) == len(msrs)
        with torch.inference_mode():
            target, target_len = handle_audio_input(target)
            m, m_len = self.motor_processor.series_to_tensor(
                msrs=msrs,
                time_stretch=time_stretch,
                pitch_shift=pitch_shift,
                in_type=msrs_type,
                )
            x, x_len = self.decode(m,m_len,target,target_len)
        if output is not None:
            for p, wav, wav_len in zip(output,x,x_len):
                torchaudio.save(p, wav[:wav_len], 16000)
        y = []
        for wav, wav_len in zip(x,x_len):
            y.append(wav[:wav_len])
        return y
    
    def speech_to_motor(
            self,
            x: Union[
                str,
                List[str],
                ],
            msrs_type: str = 'tt2',
            ):
        """
        Convert speech to TT2- or VTL-compatible motor series objects.

        Args:
            x: Union[str,List[str]], Path to audio file or list of paths.
            msrs_type: str, Type of motor series to return. Must be 'tt2' or 'vtl'.
                'vtl' means a VTL-Python compatible motor-series object
                will be returned, which has 30 articulatory features at a sample rate of 441 Hz.
                'tt2' will return a motor series with 20 articulatory features at a sample rate
                of 50 Hz, as used by TensorTract2. Default is 'tt2'.

        Returns:
            msrs: List[
                Union[
                    target_approximation.vocaltractlab.MotorSeries,
                    target_approximation.tensortract.MotorSeries,
                    ],
                ], List of motor series objects.
        """
        with torch.inference_mode():
            x, x_len = handle_audio_input(x)
            m, m_len = self.encode(x,x_len)
            msrs = self.motor_processor.tensor_to_series(
                m,
                m_len,
                out_type=msrs_type,
                )
        return msrs
    
    def speech_to_speech(
            self,
            x: Union[
                str,
                List[str],
                ],
            target: Optional[ Union[ str, List[str] ] ] = None,
            output: Optional[ Union[ str, List[str] ] ] = None,
            time_stretch: Optional[ float ] = None,
            pitch_shift: Optional[ float ] = None,
            ):
        """
        Convert speech to speech, i.e. re-synthesis or voice conversion.

        Args:
            x: Union[str,List[str]], Path to audio file or list of paths.
            target: Union[str,List[str]], Path to audio file or list of paths.
                Default is None, which means the target speaker voice is the
                same as in x.
            output: Union[str,List[str]], Path to output file or list of paths.
                Default is None, which means no output will be saved.
            time_stretch: float, Latent time-stretching factor. E.g. to double
                the speed, set to 2. Default is None, which means no time-stretching
                will be applied.
            pitch_shift: float, Latent pitch shift value in semitones. E.g. to shift
                pitch up/down an octave, set to +/- 12. Default is None, which
                means no pitch shift will be applied.
            
        Returns:
            y: List[ torch.Tensor ], (B,T). List of audio waveform tensors, 16kHz.
        """
        msrs = self.speech_to_motor(x)
        if target is None:
            target = x
        y = self.motor_to_speech(
            msrs,
            target,
            output=output,
            time_stretch=time_stretch,
            pitch_shift=pitch_shift,
            )
        return y


        



