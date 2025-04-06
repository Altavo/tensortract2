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
https://github.com/NVIDIA/NeMo

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
    Tuple,
    Union,
)
import math
import random
import logging
from abc import (
    ABC,
)

try:
    import torchaudio

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False

CONSTANT = 1e-5


def splice_frames(
    x,
    frame_splicing,
):
    """Stacks frames together across feature dim

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim*frame_splicing, num_frames

    """
    seq = [x]
    for n in range(
        1,
        frame_splicing,
    ):
        seq.append(
            torch.cat(
                [
                    x[
                        :,
                        :,
                        :n,
                    ],
                    x[
                        :,
                        :,
                        n:,
                    ],
                ],
                dim=2,
            )
        )
    return torch.cat(
        seq,
        dim=1,
    )


def normalize_batch(
    x,
    seq_len,
    normalize_type,
):
    x_mean = None
    x_std = None
    if normalize_type == "per_feature":
        x_mean = torch.zeros(
            (
                seq_len.shape[0],
                x.shape[1],
            ),
            dtype=x.dtype,
            device=x.device,
        )
        x_std = torch.zeros(
            (
                seq_len.shape[0],
                x.shape[1],
            ),
            dtype=x.dtype,
            device=x.device,
        )
        for i in range(x.shape[0]):
            if (
                x[
                    i,
                    :,
                    : seq_len[i],
                ].shape[1]
                == 1
            ):
                raise ValueError(
                    "normalize_batch with `per_feature` normalize_type received a tensor of length 1. This will result "
                    "in torch.std() returning nan. Make sure your audio length has enough samples for a single "
                    "feature (ex. at least `hop_length` for Mel Spectrograms)."
                )
            x_mean[
                i,
                :,
            ] = x[
                i,
                :,
                : seq_len[i],
            ].mean(dim=1)
            x_std[
                i,
                :,
            ] = x[
                i,
                :,
                : seq_len[i],
            ].std(dim=1)
        # make sure x_std is not zero
        x_std += CONSTANT
        return (
            (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2),
            x_mean,
            x_std,
        )
    elif normalize_type == "all_features":
        x_mean = torch.zeros(
            seq_len.shape,
            dtype=x.dtype,
            device=x.device,
        )
        x_std = torch.zeros(
            seq_len.shape,
            dtype=x.dtype,
            device=x.device,
        )
        for i in range(x.shape[0]):
            x_mean[i] = x[
                i,
                :,
                : seq_len[i].item(),
            ].mean()
            x_std[i] = x[
                i,
                :,
                : seq_len[i].item(),
            ].std()
        # make sure x_std is not zero
        x_std += CONSTANT
        return (
            (
                x
                - x_mean.view(
                    -1,
                    1,
                    1,
                )
            )
            / x_std.view(
                -1,
                1,
                1,
            ),
            x_mean,
            x_std,
        )
    elif "fixed_mean" in normalize_type and "fixed_std" in normalize_type:
        x_mean = torch.tensor(
            normalize_type["fixed_mean"],
            device=x.device,
        )
        x_std = torch.tensor(
            normalize_type["fixed_std"],
            device=x.device,
        )
        return (
            (
                x
                - x_mean.view(
                    x.shape[0],
                    x.shape[1],
                ).unsqueeze(2)
            )
            / x_std.view(
                x.shape[0],
                x.shape[1],
            ).unsqueeze(2),
            x_mean,
            x_std,
        )
    else:
        return (
            x,
            x_mean,
            x_std,
        )


@torch.jit.script_if_tracing
def make_seq_mask_like(
    lengths: torch.Tensor,
    like: torch.Tensor,
    time_dim: int = -1,
    valid_ones: bool = True,
) -> torch.Tensor:
    """

    Args:
        lengths: Tensor with shape [B] containing the sequence length of each batch element
        like: The mask will contain the same number of dimensions as this Tensor, and will have the same max
            length in the time dimension of this Tensor.
        time_dim: Time dimension of the `shape_tensor` and the resulting mask. Zero-based.
        valid_ones: If True, valid tokens will contain value `1` and padding will be `0`. Else, invert.

    Returns:
        A :class:`torch.Tensor` containing 1's and 0's for valid and invalid tokens, respectively, if `valid_ones`, else
        vice-versa. Mask will have the same number of dimensions as `like`. Batch and time dimensions will match
        the `like`. All other dimensions will be singletons. E.g., if `like.shape == [3, 4, 5]` and
        `time_dim == -1', mask will have shape `[3, 1, 5]`.
    """
    # Mask with shape [B, T]
    mask = (
        torch.arange(
            like.shape[time_dim],
            device=like.device,
        )
        .repeat(
            lengths.shape[0],
            1,
        )
        .lt(
            lengths.view(
                -1,
                1,
            )
        )
    )
    # [B, T] -> [B, *, T] where * is any number of singleton dimensions to expand to like tensor
    for _ in range(like.dim() - mask.dim()):
        mask = mask.unsqueeze(1)
    # If needed, transpose time dim
    if time_dim != -1 and time_dim != mask.dim() - 1:
        mask = mask.transpose(
            -1,
            time_dim,
        )
    # Maybe invert the padded vs. valid token values
    if not valid_ones:
        mask = ~mask
    return mask


class FilterbankFeatures(torch.nn.Module):
    """Featurizer that converts wavs to Mel Spectrograms.
    See AudioToMelSpectrogramPreprocessor for args.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_window_size=320,
        n_window_stride=160,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        nfilt=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        dither=CONSTANT,
        pad_to=16,
        max_duration=16.7,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        use_grads=False,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        mel_norm="slaney",
        stft_exact_pad=False,  # Deprecated arguments; kept for config compatibility
        stft_conv=False,  # Deprecated arguments; kept for config compatibility
    ):
        super().__init__()
        if stft_conv or stft_exact_pad:
            logging.warning(
                "Using torch_stft is deprecated and has been removed. The values have been forcibly set to False "
                "for FilterbankFeatures and AudioToMelSpectrogramPreprocessor. Please set exact_pad to True "
                "as needed."
            )
        if exact_pad and n_window_stride % 2 == 1:
            raise NotImplementedError(
                f"{self} received exact_pad == True, but hop_size was odd. If audio_length % hop_size == 0. Then the "
                "returned spectrogram would not be of length audio_length // hop_size. Please use an even hop_size."
            )
        self.log_zero_guard_value = log_zero_guard_value
        if (
            n_window_size is None
            or n_window_stride is None
            or not isinstance(
                n_window_size,
                int,
            )
            or not isinstance(
                n_window_stride,
                int,
            )
            or n_window_size <= 0
            or n_window_stride <= 0
        ):
            raise ValueError(
                f"{self} got an invalid value for either n_window_size or "
                f"n_window_stride. Both must be positive ints."
            )
        logging.info(f"PADDING: {pad_to}")

        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_pad_amount = (
            (self.n_fft - self.hop_length) // 2 if exact_pad else None
        )

        if exact_pad:
            logging.info("STFT using exact pad")
        torch_windows = {
            "hann": torch.hann_window,
            "hamming": torch.hamming_window,
            "blackman": torch.blackman_window,
            "bartlett": torch.bartlett_window,
            "none": None,
        }
        window_fn = torch_windows.get(
            window,
            None,
        )
        window_tensor = (
            window_fn(
                self.win_length,
                periodic=False,
            )
            if window_fn
            else None
        )
        self.register_buffer(
            "window",
            window_tensor,
        )
        self.stft = lambda x: torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=(False if exact_pad else True),
            window=self.window.to(dtype=torch.float),
            return_complex=True,
        )

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2

        # filterbanks_ = torch.tensor(
        #     librosa.filters.mel(
        #         sr=sample_rate,
        #         n_fft=self.n_fft,
        #         n_mels=nfilt,
        #         fmin=lowfreq,
        #         fmax=highfreq,
        #         norm=mel_norm
        #     ),
        #     dtype=torch.float,
        # ).unsqueeze(0) # old librosa implementation

        filterbanks = (
            torchaudio.functional.melscale_fbanks(
                sample_rate=sample_rate,
                n_freqs=self.n_fft // 2 + 1,
                n_mels=nfilt,
                f_min=lowfreq,
                f_max=highfreq,
                norm=mel_norm,
                mel_scale=mel_norm,
            )
            .T.to(torch.float)
            .unsqueeze(0)
        )

        # assert filterbanks_.shape == filterbanks.shape, f"filterbanks_.shape: {filterbanks_.shape}, filterbanks.shape: {filterbanks.shape}"
        # assert torch.allclose(filterbanks_, filterbanks, atol=1e-7)

        self.register_buffer(
            "fb",
            filterbanks,
        )

        # Calculate maximum sequence length
        max_length = self.get_seq_len(
            torch.tensor(
                max_duration * sample_rate,
                dtype=torch.float,
            )
        )
        max_pad = pad_to - (max_length % pad_to) if pad_to > 0 else 0
        self.max_length = max_length + max_pad
        self.pad_value = pad_value
        self.mag_power = mag_power

        # We want to avoid taking the log of zero
        # There are two options: either adding or clamping to a small value
        if log_zero_guard_type not in [
            "add",
            "clamp",
        ]:
            raise ValueError(
                f"{self} received {log_zero_guard_type} for the "
                f"log_zero_guard_type parameter. It must be either 'add' or "
                f"'clamp'."
            )

        self.use_grads = use_grads
        if not use_grads:
            self.forward = torch.no_grad()(self.forward)
        self._rng = random.Random() if rng is None else rng
        self.nb_augmentation_prob = nb_augmentation_prob
        if self.nb_augmentation_prob > 0.0:
            if nb_max_freq >= sample_rate / 2:
                self.nb_augmentation_prob = 0.0
            else:
                self._nb_max_fft_bin = int((nb_max_freq / sample_rate) * n_fft)

        # log_zero_guard_value is the the small we want to use, we support
        # an actual number, or "tiny", or "eps"
        self.log_zero_guard_type = log_zero_guard_type
        logging.debug(f"sr: {sample_rate}")
        logging.debug(f"n_fft: {self.n_fft}")
        logging.debug(f"win_length: {self.win_length}")
        logging.debug(f"hop_length: {self.hop_length}")
        logging.debug(f"n_mels: {nfilt}")
        logging.debug(f"fmin: {lowfreq}")
        logging.debug(f"fmax: {highfreq}")
        logging.debug(f"using grads: {use_grads}")
        logging.debug(f"nb_augmentation_prob: {nb_augmentation_prob}")

    def log_zero_guard_value_fn(
        self,
        x,
    ):
        if isinstance(
            self.log_zero_guard_value,
            str,
        ):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            elif self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            else:
                raise ValueError(
                    f"{self} received {self.log_zero_guard_value} for the "
                    f"log_zero_guard_type parameter. It must be either a "
                    f"number, 'tiny', or 'eps'"
                )
        else:
            return self.log_zero_guard_value

    def get_seq_len(
        self,
        seq_len,
    ):
        # Assuming that center is True is stft_pad_amount = 0
        pad_amount = (
            self.stft_pad_amount * 2
            if self.stft_pad_amount is not None
            else self.n_fft // 2 * 2
        )
        seq_len = (
            torch.floor_divide(
                (seq_len + pad_amount - self.n_fft),
                self.hop_length,
            )
            + 1
        )
        return seq_len.to(dtype=torch.long)

    @property
    def filter_banks(
        self,
    ):
        return self.fb

    def forward(
        self,
        x,
        seq_len,
        linear_spec=False,
    ):
        seq_len = self.get_seq_len(seq_len)

        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1),
                (
                    self.stft_pad_amount,
                    self.stft_pad_amount,
                ),
                "reflect",
            ).squeeze(1)

        # dither (only in training mode for eval determinism)
        if self.training and self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat(
                (
                    x[
                        :,
                        0,
                    ].unsqueeze(1),
                    x[
                        :,
                        1:,
                    ]
                    - self.preemph
                    * x[
                        :,
                        :-1,
                    ],
                ),
                dim=1,
            )

        # disable autocast to get full range of stft values
        with torch.cuda.amp.autocast(enabled=False):
            x = self.stft(x)

        # torch stft returns complex tensor (of shape [B,N,T]); so convert to magnitude
        # guard is needed for sqrt if grads are passed through
        guard = 0 if not self.use_grads else CONSTANT
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1) + guard)

        if self.training and self.nb_augmentation_prob > 0.0:
            for idx in range(x.shape[0]):
                if self._rng.random() < self.nb_augmentation_prob:
                    x[
                        idx,
                        self._nb_max_fft_bin :,
                        :,
                    ] = 0.0

        # get power spectrum
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        # return plain spectrogram if required
        if linear_spec:
            return (
                x,
                seq_len,
            )

        # dot with filterbank energies
        x = torch.matmul(
            self.fb.to(x.dtype),
            x,
        )
        # log features if required
        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(
                    torch.clamp(
                        x,
                        min=self.log_zero_guard_value_fn(x),
                    )
                )
            else:
                raise ValueError("log_zero_guard_type was not understood")

        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(
                x,
                self.frame_splicing,
            )

        # normalize if required
        if self.normalize:
            (
                x,
                _,
                _,
            ) = normalize_batch(
                x,
                seq_len,
                normalize_type=self.normalize,
            )

        # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len).to(x.device)
        mask = mask.repeat(
            x.size(0),
            1,
        ) >= seq_len.unsqueeze(1)
        x = x.masked_fill(
            mask.unsqueeze(1).type(torch.bool).to(device=x.device),
            self.pad_value,
        )
        del mask
        pad_to = self.pad_to
        if pad_to == "max":
            x = torch.nn.functional.pad(
                x,
                (
                    0,
                    self.max_length - x.size(-1),
                ),
                value=self.pad_value,
            )
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = torch.nn.functional.pad(
                    x,
                    (
                        0,
                        pad_to - pad_amt,
                    ),
                    value=self.pad_value,
                )
        return (
            x,
            seq_len,
        )


class FilterbankFeaturesTA(torch.nn.Module):
    """
    Exportable, `torchaudio`-based implementation of Mel Spectrogram extraction.

    See `AudioToMelSpectrogramPreprocessor` for args.

    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_window_size: int = 320,
        n_window_stride: int = 160,
        normalize: Optional[str] = "per_feature",
        nfilt: int = 64,
        n_fft: Optional[int] = None,
        preemph: float = 0.97,
        lowfreq: float = 0,
        highfreq: Optional[float] = None,
        log: bool = True,
        log_zero_guard_type: str = "add",
        log_zero_guard_value: Union[
            float,
            str,
        ] = 2
        ** -24,
        dither: float = 1e-5,
        window: str = "hann",
        pad_to: int = 0,
        pad_value: float = 0.0,
        mel_norm="slaney",
        # Seems like no one uses these options anymore. Don't convolute the code by supporting thm.
        use_grads: bool = False,  # Deprecated arguments; kept for config compatibility
        max_duration: float = 16.7,  # Deprecated arguments; kept for config compatibility
        frame_splicing: int = 1,  # Deprecated arguments; kept for config compatibility
        exact_pad: bool = False,  # Deprecated arguments; kept for config compatibility
        nb_augmentation_prob: float = 0.0,  # Deprecated arguments; kept for config compatibility
        nb_max_freq: int = 4000,  # Deprecated arguments; kept for config compatibility
        mag_power: float = 2.0,  # Deprecated arguments; kept for config compatibility
        rng: Optional[
            random.Random
        ] = None,  # Deprecated arguments; kept for config compatibility
        stft_exact_pad: bool = False,  # Deprecated arguments; kept for config compatibility
        stft_conv: bool = False,  # Deprecated arguments; kept for config compatibility
    ):
        super().__init__()
        if not HAVE_TORCHAUDIO:
            raise ValueError(
                f"Need to install torchaudio to instantiate a {self.__class__.__name__}"
            )

        # Make sure log zero guard is supported, if given as a string
        supported_log_zero_guard_strings = {
            "eps",
            "tiny",
        }
        if (
            isinstance(
                log_zero_guard_value,
                str,
            )
            and log_zero_guard_value not in supported_log_zero_guard_strings
        ):
            raise ValueError(
                f"Log zero guard value must either be a float or a member of {supported_log_zero_guard_strings}"
            )

        # Copied from `AudioPreprocessor` due to the ad-hoc structuring of the Mel Spec extractor class
        self.torch_windows = {
            "hann": torch.hann_window,
            "hamming": torch.hamming_window,
            "blackman": torch.blackman_window,
            "bartlett": torch.bartlett_window,
            "ones": torch.ones,
            None: torch.ones,
        }

        # Ensure we can look up the window function
        if window not in self.torch_windows:
            raise ValueError(
                f"Got window value '{window}' but expected a member of {self.torch_windows.keys()}"
            )

        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self._sample_rate = sample_rate
        self._normalize_strategy = normalize
        self._use_log = log
        self._preemphasis_value = preemph
        self.log_zero_guard_type = log_zero_guard_type
        self.log_zero_guard_value: Union[
            str,
            float,
        ] = log_zero_guard_value
        self.dither = dither
        self.pad_to = pad_to
        self.pad_value = pad_value
        self.n_fft = n_fft
        self._mel_spec_extractor: torchaudio.transforms.MelSpectrogram = (
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self._sample_rate,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=nfilt,
                window_fn=self.torch_windows[window],
                mel_scale="slaney",
                norm=mel_norm,
                n_fft=n_fft,
                f_max=highfreq,
                f_min=lowfreq,
                wkwargs={"periodic": False},
            )
        )

    @property
    def filter_banks(
        self,
    ):
        """Matches the analogous class"""
        return self._mel_spec_extractor.mel_scale.fb

    def _resolve_log_zero_guard_value(
        self,
        dtype: torch.dtype,
    ) -> float:
        if isinstance(
            self.log_zero_guard_value,
            float,
        ):
            return self.log_zero_guard_value
        return getattr(
            torch.finfo(dtype),
            self.log_zero_guard_value,
        )

    def _apply_dithering(
        self,
        signals: torch.Tensor,
    ) -> torch.Tensor:
        if self.training and self.dither > 0.0:
            noise = torch.randn_like(signals) * self.dither
            signals = signals + noise
        return signals

    def _apply_preemphasis(
        self,
        signals: torch.Tensor,
    ) -> torch.Tensor:
        if self._preemphasis_value is not None:
            padded = torch.nn.functional.pad(
                signals,
                (
                    1,
                    0,
                ),
            )
            signals = (
                signals
                - self._preemphasis_value
                * padded[
                    :,
                    :-1,
                ]
            )
        return signals

    def _compute_output_lengths(
        self,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        out_lengths = (
            input_lengths.div(
                self.hop_length,
                rounding_mode="floor",
            )
            .add(1)
            .long()
        )
        return out_lengths

    def _apply_pad_to(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        # Only apply during training; else need to capture dynamic shape for exported models
        if (
            not self.training
            or self.pad_to == 0
            or features.shape[-1] % self.pad_to == 0
        ):
            return features
        pad_length = self.pad_to - (features.shape[-1] % self.pad_to)
        return torch.nn.functional.pad(
            features,
            pad=(
                0,
                pad_length,
            ),
            value=self.pad_value,
        )

    def _apply_log(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        if self._use_log:
            zero_guard = self._resolve_log_zero_guard_value(features.dtype)
            if self.log_zero_guard_type == "add":
                features = features + zero_guard
            elif self.log_zero_guard_type == "clamp":
                features = features.clamp(min=zero_guard)
            else:
                raise ValueError(
                    f"Unsupported log zero guard type: '{self.log_zero_guard_type}'"
                )
            features = features.log()
        return features

    def _extract_spectrograms(
        self,
        signals: torch.Tensor,
    ) -> torch.Tensor:
        # Complex FFT needs to be done in single precision
        with torch.cuda.amp.autocast(enabled=False):
            features = self._mel_spec_extractor(waveform=signals)
        return features

    def _apply_normalization(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        # For consistency, this function always does a masked fill even if not normalizing.
        mask: torch.Tensor = make_seq_mask_like(
            lengths=lengths,
            like=features,
            time_dim=-1,
            valid_ones=False,
        )
        features = features.masked_fill(
            mask,
            0.0,
        )
        # Maybe don't normalize
        if self._normalize_strategy is None:
            return features
        # Use the log zero guard for the sqrt zero guard
        guard_value = self._resolve_log_zero_guard_value(features.dtype)
        if (
            self._normalize_strategy == "per_feature"
            or self._normalize_strategy == "all_features"
        ):
            # 'all_features' reduces over each sample; 'per_feature' reduces over each channel
            reduce_dim = 2
            if self._normalize_strategy == "all_features":
                reduce_dim = [
                    1,
                    2,
                ]
            # [B, D, T] -> [B, D, 1] or [B, 1, 1]
            means = features.sum(
                dim=reduce_dim,
                keepdim=True,
            ).div(
                lengths.view(
                    -1,
                    1,
                    1,
                )
            )
            stds = (
                features.sub(means)
                .masked_fill(
                    mask,
                    0.0,
                )
                .pow(2.0)
                .sum(
                    dim=reduce_dim,
                    keepdim=True,
                )  # [B, D, T] -> [B, D, 1] or [B, 1, 1]
                .div(
                    lengths.view(
                        -1,
                        1,
                        1,
                    )
                    - 1
                )  # assume biased estimator
                .clamp(min=guard_value)  # avoid sqrt(0)
                .sqrt()
            )
            features = (features - means) / (stds + eps)
        else:
            # Deprecating constant std/mean
            raise ValueError(f"Unsupported norm type: '{self._normalize_strategy}")
        features = features.masked_fill(
            mask,
            0.0,
        )
        return features

    def forward(
        self,
        input_signal: torch.Tensor,
        length: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        feature_lengths = self._compute_output_lengths(input_lengths=length)
        signals = self._apply_dithering(signals=input_signal)
        signals = self._apply_preemphasis(signals=signals)
        features = self._extract_spectrograms(signals=signals)
        features = self._apply_log(features=features)
        features = self._apply_normalization(
            features=features,
            lengths=feature_lengths,
        )
        features = self._apply_pad_to(features=features)
        return (
            features,
            feature_lengths,
        )


class AudioToMelSpectrogramPreprocessor(torch.nn.Module):
    """ML Modules port of the featurizer module that converts wavs to mel spectrograms.

    Args:
        sample_rate (int): Sample rate of the input audio data.
            Defaults to 16000
        window_size (float): Size of window for fft in seconds
            Defaults to 0.02
        window_stride (float): Stride of window for fft in seconds
            Defaults to 0.01
        n_window_size (int): Size of window for fft in samples
            Defaults to None. Use one of window_size or n_window_size.
        n_window_stride (int): Stride of window for fft in samples
            Defaults to None. Use one of window_stride or n_window_stride.
        window (str): Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett']
            Defaults to "hann"
        normalize (str): Can be one of ['per_feature', 'all_features']; all
            other options disable feature normalization. 'all_features'
            normalizes the entire spectrogram to be mean 0 with std 1.
            'pre_features' normalizes per channel / freq instead.
            Defaults to "per_feature"
        n_fft (int): Length of FT window. If None, it uses the smallest power
            of 2 that is larger than n_window_size.
            Defaults to None
        preemph (float): Amount of pre emphasis to add to audio. Can be
            disabled by passing None.
            Defaults to 0.97
        features (int): Number of mel spectrogram freq bins to output.
            Defaults to 64
        lowfreq (int): Lower bound on mel basis in Hz.
            Defaults to 0
        highfreq  (int): Lower bound on mel basis in Hz.
            Defaults to None
        log (bool): Log features.
            Defaults to True
        log_zero_guard_type(str): Need to avoid taking the log of zero. There
            are two options: "add" or "clamp".
            Defaults to "add".
        log_zero_guard_value(float, or str): Add or clamp requires the number
            to add with or clamp to. log_zero_guard_value can either be a float
            or "tiny" or "eps". torch.finfo is used if "tiny" or "eps" is
            passed.
            Defaults to 2**-24.
        dither (float): Amount of white-noise dithering.
            Defaults to 1e-5
        pad_to (int): Ensures that the output size of the time dimension is
            a multiple of pad_to.
            Defaults to 16
        frame_splicing (int): Defaults to 1
        exact_pad (bool): If True, sets stft center to False and adds padding, such that num_frames = audio_length
            // hop_length. Defaults to False.
        pad_value (float): The value that shorter mels are padded with.
            Defaults to 0
        mag_power (float): The power that the linear spectrogram is raised to
            prior to multiplication with mel basis.
            Defaults to 2 for a power spec
        rng : Random number generator
        nb_augmentation_prob (float) : Probability with which narrowband augmentation would be applied to
            samples in the batch.
            Defaults to 0.0
        nb_max_freq (int) : Frequency above which all frequencies will be masked for narrowband augmentation.
            Defaults to 4000
        use_torchaudio: Whether to use the `torchaudio` implementation.
        mel_norm: Normalization used for mel filterbank weights.
            Defaults to 'slaney' (area normalization)
        stft_exact_pad: Deprecated argument, kept for compatibility with older checkpoints.
        stft_conv: Deprecated argument, kept for compatibility with older checkpoints.
    """

    def __init__(
        self,
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        n_window_size=None,
        n_window_stride=None,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        features=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        dither=1e-5,
        pad_to=16,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        use_torchaudio: bool = False,
        mel_norm="slaney",
        stft_exact_pad=False,  # Deprecated arguments; kept for config compatibility
        stft_conv=False,  # Deprecated arguments; kept for config compatibility
    ):
        super().__init__()
        self.win_length = n_window_size
        self.hop_length = n_window_stride

        self.torch_windows = {
            "hann": torch.hann_window,
            "hamming": torch.hamming_window,
            "blackman": torch.blackman_window,
            "bartlett": torch.bartlett_window,
            "ones": torch.ones,
            None: torch.ones,
        }

        self._sample_rate = sample_rate
        if window_size and n_window_size:
            raise ValueError(
                f"{self} received both window_size and "
                f"n_window_size. Only one should be specified."
            )
        if window_stride and n_window_stride:
            raise ValueError(
                f"{self} received both window_stride and "
                f"n_window_stride. Only one should be specified."
            )
        if window_size:
            n_window_size = int(window_size * self._sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * self._sample_rate)

        # Given the long and similar argument list, point to the class and instantiate it by reference
        if not use_torchaudio:
            featurizer_class = FilterbankFeatures
        else:
            featurizer_class = FilterbankFeaturesTA
        self.featurizer = featurizer_class(
            sample_rate=self._sample_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            window=window,
            normalize=normalize,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=features,
            lowfreq=lowfreq,
            highfreq=highfreq,
            log=log,
            log_zero_guard_type=log_zero_guard_type,
            log_zero_guard_value=log_zero_guard_value,
            dither=dither,
            pad_to=pad_to,
            frame_splicing=frame_splicing,
            exact_pad=exact_pad,
            pad_value=pad_value,
            mag_power=mag_power,
            rng=rng,
            nb_augmentation_prob=nb_augmentation_prob,
            nb_max_freq=nb_max_freq,
            mel_norm=mel_norm,
            stft_exact_pad=stft_exact_pad,  # Deprecated arguments; kept for config compatibility
            stft_conv=stft_conv,  # Deprecated arguments; kept for config compatibility
        )

    def input_example(
        self,
        max_batch: int = 8,
        max_dim: int = 32000,
        min_length: int = 200,
    ):
        batch_size = torch.randint(
            low=1,
            high=max_batch,
            size=[1],
        ).item()
        max_length = torch.randint(
            low=min_length,
            high=max_dim,
            size=[1],
        ).item()
        signals = (
            torch.rand(
                size=[
                    batch_size,
                    max_length,
                ]
            )
            * 2
            - 1
        )
        lengths = torch.randint(
            low=min_length,
            high=max_dim,
            size=[batch_size],
        )
        lengths[0] = max_length
        return (
            signals,
            lengths,
        )

    def get_features(
        self,
        input_signal,
        length,
    ):
        return self.featurizer(
            input_signal,
            length,
        )

    def forward(
        self,
        input_signal,
        length,
    ):
        (
            processed_signal,
            processed_length,
        ) = self.get_features(
            input_signal,
            length,
        )

        return (
            processed_signal,
            processed_length,
        )

    @property
    def filter_banks(
        self,
    ):
        return self.featurizer.filter_banks
