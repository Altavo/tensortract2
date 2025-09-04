# Official TensorTract2 Implementation

This repository contains the official implementation of the TensorTract2 model described in the paper: "[Precisely Controllable Neural Speech Synthesis](https://ieeexplore.ieee.org/abstract/document/10890772)", by Krug et al., published in Proc. ICASSP 2025.

TensorTract2 is a multi-task model that features:
- State-of-the-art acoustic-to-articulatory inversion
- A neural audio codec with a fully interpretable and disentangled speech latent representation that offers precise control over the speech production process.
- High-performance voice conversion
- Speech denoising, pitch-detection and more

## NOTE
This is a stand-alone implementation of the TensorTract2 model, so there is no articulatory synthesizer included. If you wish to do **articulatory synthesis** from the latent representation or if you wish to create **2D vocaltract visualizations and videos**, please use [TensorTractLab](https://github.com/tensortract-dev/tensortractlab), which integrates both TensorTract2 and VocalTractLab-Python.

## Installation
TensorTract2 can be installed via pip:
```bash
pip install tensortract2
```

# Usage

## Loading the Model
```python
from tensortract2 import TensorTract2

# Load the model
tt2 = TensorTract2()
```
Per default the model will download the weights from [google drive](https://drive.google.com/file/d/1UIPlLKXCuVHPj3_nmJflw48TzXI4BtUw/view?usp=sharing) on its first initialization and put it into the users cache directory (wavlm-large will be downloaded from Huggingface). If the model is used again, no download will be necessary anymore. If you wish to load the weights manually, you could do it like this:
```python
tt2 = TensorTract2(auto_load_weights = False)
tt2.load_weights( 'path/to/weights' )
```
Note that the model will always be initialiazed in `eval mode` automatically, so you don't need to set it manually.

## Acoustic-to-Articulatory Inversion
You can convert any speech audio files to articulatory parameters using the `speech_to_motor` method. The input `x` can be a string or a list of strings. The output is a list of `MotorSeries` objects, for more info on these objects, see package [target-approximation](https://github.com/paul-krug/target-approximation).
```python
# Load speech from an audio file and process it
motor_data = tt2.speech_to_motor(
    x='path/to/audio.wav',
    # Optional parameters:
    msrs_type='tt2',
    )

# motor_data is a list of MotorSeries objects
# Each MotorSeries object contains the articulatory parameters
# you can plot them like this:
motor_data[0].plot()

# get the articulatory parameters as numpy array:
array = motor_data[0].to_numpy()
```
The parameter `msrs_type` describes the type of returned motor-series data. `tt2`means 20 articulatory features at a sampling rate of 50 Hz (TensorTract2 standard), `vtl` means 30 articulatory features at a sampling rate of 441 Hz (VocalTractLab-Python standard).  Use `vtl` type iif you want compatibility with the articulatory synthesizer VocalTractLab-Python.

## Articulatory Synthesis
This is a stand-alone implementation of the TensorTract2 model, so there is no articulatory synthesizer included.
However, you can generate articulatory data that is directly compatible for articulatory synthesis with [VocalTractLab-Python](https://github.com/paul-krug/VocalTractLab-Python).
```python
# Load speech from an audio file and process it
motor_data = tt2.speech_to_motor(
    x="path/to/audio.wav",
    msrs_type='vtl',
    )

# continue to process motor_data with VocalTractLab-Python
```

## Neural Re-synthesis and Voice-Conversion
```python
wavs = tt2.speech_to_speech(
    x="path/to/audio.wav",
    # Optional parameters:
    target="path/to/target.wav",
    output="path/to/output.wav",
    time_stretch=None,  # time stretch factor
    pitch_shift=None,  # pitch shift in semitones
)
wavs # is a list of audio tensors (16kHz, mono)
```
The parameter `target` is optional. If you provide a target audio file, the model will perform voice conversion using the voice characteristic from the target speech file. If you don't provide a target, the model will perform neural re-synthesis. The output will be saved to the specified path.
The parameters `x`, `target` and `output` can be a string or a list of strings. If you provide a list of strings, the model will process each file and save the resulting audio to the paths provided in `output`.

## Fine-grained Speech Manipulation
At the moment you can only manipulate the articulatory parameters manually like this:
```python
motor_data = tt2.speech_to_motor(
    x="path/to/audio.wav",
    msrs_type='tt2',
    )

m = motor_data[0]  # get the first MotorSeries object

# Manipulate the articulatory parameters (for example TCX):
m[ 'TCX' ] *= 1.5  # increase TCX by 50%

# or directly access the numpy array:
m_np = m.to_numpy()
m_np[ :, v:w ] = .... # any manipulation

from target_approximation.tensortract import MotorSeries
m = MotorSeries( m_np, sr=50 ) # back to motor-series 

# re-synthesize the audio
wavs = tt2.motor_to_speech(
    msrs=m,
    target='path/to/target.wav',  # Get a voice for synthesis
    # Optional parameters:
    output: Optional[ Union[ str, List[str] ] ] = None,
    time_stretch: Optional[ float ] = None,
    pitch_shift: Optional[ float ] = None,
    msrs_type: str = 'tt2',
    )
```


## Speech Denoising
Speech denoising will happen automatically if the input audio is noisy. The model will automatically detect the noise and remove it.


# How to cite
If you use this code in your research, please cite the following paper:
```bibtex
@inproceedings{krug2025precisely,
  title={Precisely Controllable Neural Speech Synthesis},
  author={Krug, Paul Konstantin and Wagner, Christoph and Birkholz, Peter and Stich, Timo},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```
