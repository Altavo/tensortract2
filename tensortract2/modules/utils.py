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
import requests
import re
import hashlib
from tqdm import tqdm
import warnings
import logging

from typing import Union
from typing import List

from .audioprocessing_functional import resample_like_librosa


def create_mask(x, x_len):
    with torch.no_grad():
        S = []
        N = x.shape[-1]
        for n in x_len:
            s = torch.zeros( [1, N,] )
            s[..., :n,] = 1.0
            S.append(s)
        return torch.stack(S).to(x.device)
    

def create_mask_same_shape(x, x_len):
    # should be like create_mask but output same shape as x
    # Expects input of shape (B, ..., L), B: batch size, L: length of sequence
    if len(x.shape) < 2:
        raise ValueError("Input must have at least 2 dimensions (B, L).")
    if x.shape[0] != len(x_len):
        raise ValueError(
            "len(x_len) must be same as first dimension of input (batch_size)."
        )

    with torch.no_grad():
        S = torch.zeros_like(x)
        for (
            index,
            length,
        ) in enumerate(x_len):
            S[
                index,
                ...,
                :length,
            ] = 1.0
    return S.to(x.device)


def pad_list(l, pad_value=0):
    """"
    Pad list of tensors to the same length.
    
    Args:
        l: List[torch.Tensor], Expected shape (..., L)
        pad_value: int, value to pad with
        
    Returns:
        x: torch.Tensor, (B, ..., L)
        x_len: torch.Tensor, (B,)
    """
    max_len = max([y.shape[-1] for y in l])
    x = []
    x_len = []
    for y in l:
        x.append(
            torch.nn.functional.pad(
                y,
                pad=[0, max_len - y.shape[-1]],
                value=pad_value,
            )
        )
        x_len.append(y.shape[-1])
    x = torch.stack(x, dim=0)
    x_len = torch.tensor(x_len)
    return x, x_len

def handle_audio_input(
            x: Union[
                str,
                List[str],
                ],
            ):
        """
        Handle input x as a string or list of strings.
        """
        if isinstance(x,str):
            x = [x]
        
        wavs = []
        for p in x:
            assert isinstance(p,str)
            wav, sr = torchaudio.load(p, channels_first=True)
            # if wav has more than one channel, make it mono
            if wav.size(0) > 1:
                wav = wav.mean(0,keepdim=True)
            # normalize shape to (T,)
            wav = wav.squeeze(0)
            assert wav.dim() == 1
            # resample to 16kHz
            wav = resample_like_librosa(wav, sr, 16000)
            wavs.append(wav)

        x, x_len = pad_list(wavs)
        return x, x_len

def load_weights_initial(p, prefix):
    x = torch.load(
        p,
        map_location=torch.device('cpu'), 
        weights_only=False,
        )
    sd = x[ 'state_dict' ]
    weights = { replace_prefix(k, prefix): v for k, v in sd.items() if prefix in k }
    return weights

def replace_prefix( k, prefix ):
    try:
        if k[ :len(prefix)+1] == f'{prefix}.':
            k = k[ len(prefix)+1: ]
            return k
        else:
            raise ValueError(
                f"""
                Prefix: {prefix} not compatible with key from state_dict:
                {k}
                """
            )
    except:
        raise ValueError( 'fail' )
    

# SPDX-License-Id: MIT
# Credit: Jack Giffin and platformdirs: github.com/tox-dev/platformdirs
# Source: https://stackoverflow.com/a/79403791/5601591
def get_user_cache_dir():
    from sys import platform, path
    from os import getenv, path
    if platform == "darwin":
        return os.path.expanduser("~/Library/Caches")
    elif platform == "win32":
        try:
            from ctypes import windll, wintypes, create_unicode_buffer, c_int
            buf, gfpW = create_unicode_buffer(1024), windll.shell32.SHGetFolderPathW
            gfpW.argtypes = [wintypes.HWND,c_int,wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR]
            gfpW.restype = wintypes.HRESULT
            if 0 == gfpW(None, 28, None, 0, buf) and buf[0] != 0:
                return buf.value # CSIDL_LOCAL_APPDATA = 28
        except Exception:
            pass
        if getenv("LOCALAPPDATA") and path.isdir(getenv("LOCALAPPDATA")):
            return getenv("LOCALAPPDATA")
        from winreg import OpenKey, QueryValueEx, HKEY_CURRENT_USER
        key = OpenKey(HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
        return str( QueryValueEx(key, "Local AppData")[1] )
    else:
        # For all Linux and *nix including Haiku, OpenIndiana, and the BSDs:
        return getenv("XDG_CACHE_HOME","").strip() or path.expanduser("~/.cache")


def download_file_from_google_drive(file_id, destination):
    """Download a large file from Google Drive, handling confirmation prompts."""
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    # Step 1: Initial request to get the confirmation token
    response = session.get(URL, params={'id': file_id}, stream=True)
    confirm_token = get_confirm_token(response.text)

    # Step 2: If there's a confirmation token, request the file again with it
    if confirm_token:
        print("⚠️ Google Drive requires confirmation... retrying with token.")
        response = session.get(
            "https://drive.usercontent.google.com/download",
            params={'id': file_id, 'export': 'download', 'confirm': confirm_token},
            stream=True
        )

    # Step 3: Verify the response before saving
    if not response.ok or 'text/html' in response.headers.get('Content-Type', ''):
        print("❌ Failed to download the file. Google Drive is blocking the request.")
        print("Response headers:", response.headers)
        print("First 500 bytes of response:", response.content[:500].decode(errors='ignore'))
        return

    save_response_content(response, destination)
    print(f"✅ File downloaded successfully to: {destination}")

def get_confirm_token(response_text):
    """Extract the confirmation token from the warning page."""
    match = re.search(r'name="confirm" value="([^"]+)"', response_text)
    return match.group(1) if match else None

def save_response_content(response, destination):
    """Save file to disk in chunks with a progress bar."""
    CHUNK_SIZE = 32768  # 32 KB
    total_size = int(response.headers.get('content-length', 0))  # Get total file size if available

    os.makedirs(
        os.path.dirname(destination),
        exist_ok=True,
        )

    with open(destination, "wb") as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))  # Update progress bar


def verify_checksum(file_path, expected_checksum, hash_algorithm="sha256"):
    """Compute the file checksum and compare it with the expected checksum."""
    hash_func = getattr(hashlib, hash_algorithm, None)
    if hash_func is None:
        print(f"Error: Unsupported hash algorithm '{hash_algorithm}'")
        return False

    computed_hash = hash_func()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            computed_hash.update(chunk)

    computed_checksum = computed_hash.hexdigest()

    if computed_checksum.lower() == expected_checksum.lower():
        #print(f"✅ Checksum match: {computed_checksum}")
        logging.info(f"✅ Checksum match: {computed_checksum}")
        return True
    else:
        #print(f"❌ Checksum mismatch!")
        #print(f"Expected: {expected_checksum}")
        #print(f"Got: {computed_checksum}")
        warnings.warn(
            f"❌ Checksum mismatch! Expected: {expected_checksum}, Got: {computed_checksum}"
        )
        return False