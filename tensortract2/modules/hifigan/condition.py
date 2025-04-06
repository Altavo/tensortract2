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
import torch.nn as nn
from collections.abc import Iterable

from .conditional_layer import Additive, Concatenative, FiLM

from typing import Union, List

MODES = {
    'add': Additive,
    'concat': Concatenative,
    'film': FiLM,
    }

class Condition(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_cond: Union[int, List[int]],
            mode: str,
            ):
        super().__init__()
        self.mode = mode
        self.dim = dim
        if not isinstance( dim_cond, Iterable ):
            dim_cond = [ dim_cond ]
        self.dim_cond = dim_cond
        self.condition = MODES[ mode ](
            dim = self.dim,
            dim_cond = sum( self.dim_cond ),
            )
        return
    
    def forward(
            self,
            x: torch.Tensor,
            cond: Union[ torch.Tensor, List[ torch.Tensor ] ],
            ):
        # Expects x of shape ( batch, dim, time )
        # and cond of shape ( batch, dim_cond ) or ( batch, dim_cond, time )
        # or a list of such tensors

        if isinstance( cond, torch.Tensor ):
            cond = [ cond ]
        elif not isinstance( cond, Iterable ):
            cond = [ cond ]

        e = []
        for c, expected_shape in zip( cond, self.dim_cond ):
            c = c.to( x.device )
            # check if condition is 1D
            if c.dim() == 1:
                # unsqueeze the batch dimension and repeat it to match x
                c = c.unsqueeze(0).repeat(x.shape[0], 1)
            if c.shape[1] != expected_shape:
                raise ValueError(
                    f"""
                    Condition shape {c.shape} passed to 'forward' does
                    not match the expected shape {expected_shape} provided
                    as input to argument 'dim_cond'.
                    """
                    )
            if c.dim() == 2:
                # unsqueeze time dimension of e and repeat it to match x
                e.append( c.unsqueeze(2).repeat(1, 1, x.shape[2]) )
            elif c.dim() == 3:
                # check if time dimension of c matches x
                if c.shape[2] != x.shape[2]:
                    raise ValueError(
                        f"""
                        Condition time dimension {c.shape[2]} does not
                        match the input time dimension {x.shape[2]}.
                        """
                        )
                e.append( c )
            else:
                raise ValueError(
                    f"""
                    Condition tensor must have shape ( dim_cond, ) or
                    ( batch, dim_cond ) or ( batch, dim_cond, time ).
                    """
                    )
            
        e = torch.cat( e, dim = 1 )
        x = self.condition( x, e )
        return x