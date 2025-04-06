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
import torchaudio.functional as F
import numpy as np

from typing import (
    Union,
)
from typing import (
    Optional,
)
from typing import (
    Tuple,
)
from numpy.typing import (
    ArrayLike,
)


def to_float(
    waveform: ArrayLike,
) -> torch.Tensor:
    """
    Convert any audio array to Torch float tensor.
    :param waveform: Audio to convert.
    :return: Audio as float.
    """
    # Convert to torch tensor
    if not isinstance(
        waveform,
        torch.Tensor,
    ):
        waveform = torch.tensor(waveform)
    # Convert to float
    if not waveform.dtype == torch.float32:
        input_dtype = waveform.dtype
        waveform = waveform.float()
        if input_dtype == torch.int16:
            waveform = waveform / 32768.0
    return waveform


def to_int(
    waveform,
):
    """
    Convert any audio array to Torch int tensor.
    :param waveform: Audio to convert.
    :return: Audio as int16.
    """
    # Convert to torch tensor
    if not isinstance(
        waveform,
        torch.Tensor,
    ):
        waveform = torch.tensor(waveform)
    # Conver to int
    if not waveform.dtype == torch.int16:
        waveform = waveform * 32768
        waveform = waveform.short()
    return waveform


def normalize_audio_amplitude(
    waveform: ArrayLike,
    target_dBFS: float = -1,
    axis: int = -1,
) -> torch.Tensor:
    """
    Normalizes audio amplitude, so that the maximum of the waveform corresponds
    to a given target value specified in dBFS.

    :param waveform: Audio to normalize.
    :param target_dBFS: Target dBFS.
    :param axis: Axis to normalize, default: -1, which assumes shape: (..., time).
    :return: Normalized audio.
    """
    waveform = to_float(waveform)
    norm_factor = 10 ** (-1 * target_dBFS * 0.05) - 1
    norm_max = torch.max(
        torch.abs(waveform),
        axis=axis,
        keepdim=True,
    ).values
    waveform_normalized = waveform / (norm_max + (norm_max * norm_factor))
    return waveform_normalized


def resample_like_librosa(
    waveform: ArrayLike,
    input_samplerate: int,
    output_samplerate: int,
) -> torch.Tensor:
    """
    Resample audio with results similar librosa
    with 'kaiser best' setting.
    :param waveform: Audio to resample.
    :param input_samplerate: Input sample rate.
    :param output_samplerate: Output sample rate.
    :return: Resampled audio.
    """
    waveform = to_float(waveform)
    if input_samplerate != output_samplerate:
        waveform = F.resample(
            waveform=waveform,
            orig_freq=input_samplerate,
            new_freq=output_samplerate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
    return waveform

def hz_to_st(
        frequency_hz,
        reference = 1.0,
    ):
    return 12.0*np.log( frequency_hz / reference ) / np.log(2.0)

def st_to_hz(
        frequency_st,
        reference = 1.0,
    ):
    return reference*pow( 2, frequency_st / 12.0 )