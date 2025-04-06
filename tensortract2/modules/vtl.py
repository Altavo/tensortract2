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
import torch.nn.functional as F
import numpy as np

from target_approximation.vocaltractlab import MotorSeries as VTL_MSRS
from target_approximation.tensortract import MotorSeries as TT_MSRS


class MotorProcessor():
    def __init__(self):
        super().__init__()
        self.norm_values = [
            dict(name = 'HX', min = 0.0, max = 1.0),
            dict(name = 'HY', min = -6.0, max = -3.5),
            dict(name = 'JX', min = -0.5, max = 0.0),
            dict(name = 'JA', min = -7.0, max = 0.0),
            dict(name = 'LP', min = -1.0, max = 1.0),
            dict(name = 'LD', min = -0.5, max = 2.0),
            dict(name = 'VS', min = 0.0, max = 1.0),
            dict(name = 'VO', min = -0.1, max = 1.0),
            dict(name = 'TCX', min = -1.5, max = 3.01),
            dict(name = 'TCY', min = -3.3, max = 0.3),
            dict(name = 'TTX', min = 1.38, max = 5.01),
            dict(name = 'TTY', min = -3.15, max = 5.01),
            dict(name = 'TBX', min = -1.5, max = 4.0),
            dict(name = 'TBY', min = -3.0, max = 5.0),
            dict(name = 'TS1', min = 0.0, max = 1.0),
            dict(name = 'TS2', min = 0.0, max = 1.0),
            dict(name = 'TS3', min = -1.0, max = 1.0),
            dict(name = 'F0', min = 50.0, max = 400.0),
            dict(name = 'PR', min = 0.0, max = 8000.0),
            dict(name = 'XB', min = 0.01, max = 0.1),
            #dict(name = 'CA', min = 0.05, max = 0.1),
            #dict(name = 'RA', min = 0.0, max = 1.0),
        ]
        
        a_shape = np.array([
            0.1667, -3.9392,  0., -4.1498,  0.0718,  0.9937,  0.8,
            -0.1,  0.1524, -1.8333,  4.2474, -1.694,  2.5488, -0.675,
            -2.8371, -2.9034,  0.2064,  0.0384,  0.1488,
            ])
        modal_shape = np.array([
            1.200000e+02,  8.000000e+03,  1.020000e-02,  2.035000e-02,
            5.000000e-02,  1.222044e+00,  1.000000e+00,  5.000000e-02,
            0.000000e+00,  2.500000e+01, -1.000000e+01,
            ])
        self.proto_shape = np.concatenate((a_shape, modal_shape), axis=0)
        return
    
    def norm(
            self,
            x: torch.Tensor,
            ):
        """
        Normalizes a TT2 motor tensor with 20 feature dimensions from VTL ranges
        to range [ -1, 1 ] using min-max normalization. For F0 (feature no. 17)
        a log scaling is applied before min-max normalization.

        Args:
            x: torch.Tensor, (C, T), Motor tensor.

        Returns:
            x: torch.Tensor, (C, T), Normalized motor tensor.
        """

        for i in range( 20 ):
            if i == 17:
                min_val = np.log( self.norm_values[i]['min'] )
                max_val = np.log( self.norm_values[i]['max'] )
                x[i] = 2.0 * ( torch.log( x[i] ) - min_val ) / ( max_val - min_val ) - 1.0
            else:
                min_val = self.norm_values[i]['min']
                max_val = self.norm_values[i]['max']
                x[i] = 2.0 * ( x[i] - min_val ) / ( max_val - min_val ) - 1.0

        return x
    
    def denorm(
            self,
            x: torch.Tensor,
            ):
        """
        De-normalizes a normalized TT2 motor tensor with 20 feature dimensions
        from ranges [-1, 1] back to VTL ranges.
        
        Args:
            x: torch.Tensor, (C, T), Normaized Motor tensor.
            
        Returns:
            x: torch.Tensor, (C, T), De-normalized motor tensor.
        """

        for i in range( 20 ):
            if i == 17:
                min_val = np.log( self.norm_values[i]['min'] )
                max_val = np.log( self.norm_values[i]['max'] )
                x[i] = torch.exp( 0.5 * ( x[i] + 1.0 ) * ( max_val - min_val ) + min_val )
            else:
                min_val = self.norm_values[i]['min']
                max_val = self.norm_values[i]['max']
                x[i] = 0.5 * ( x[i] + 1.0 ) * ( max_val - min_val ) + min_val

        return x
    
    def tensor_to_series(
            self,
            x,
            x_len,
            out_type = 'vtl',
            ):
        """
        Convert the raw TensorTract2 motor tensor outputs to motor series objects,
        that can be plotted, alterd or used for articulatory synthesis with VocalTractLab.

        Args:
            x: torch.Tensor, (B, C, T)
            x_len: torch.Tensor, (B,)
            out_type: str, 'vtl' or 'tt2'. 'vtl' means a VTL-Python compatible motor-series object
                will be returned, which has 30 articulatory features at a sample rate of 441 Hz.
                'tt2' will return a motor series with 20 articulatory features at a sample rate
                of 50 Hz, as used by TensorTract2. Default is 'vtl'.

        Returns:
            motor_data: List[MotorSeries], List of motor series objects.
        """
        
        # x has shape (batch, features, time)
        motor_data = []
        for motor, l in zip(x, x_len):
            motor = self.denorm( motor ).cpu().numpy()
            motor = motor[:, :l].T
            ms_tt = TT_MSRS(
                series = motor,
                sr = 50,
                #sg_set='jd3',
                )
            if out_type == 'tt2':
                motor_data.append( ms_tt )
            elif out_type == 'vtl':
                ms_tt.resample(441)

                ms_proto = VTL_MSRS(
                    series = np.repeat(
                        self.proto_shape[np.newaxis, :],
                        #motor.shape[0],
                        len( ms_tt ),
                        axis=0,
                        ),
                    sr=441,
                    )
            
                ms_vtl = ms_proto & ms_tt

                # post process the motor series:
                ms_vtl[ 'XT' ] = ms_vtl[ 'XB' ]
                # CA = XB but rescale to 0.05 - 0.1
                #ms_vtl[ 'CA' ] = ms_vtl[ 'XB' ] * 0.05/0.1
                # CA = XB but rescale from [ 0.01, 0.1 ] to [ 0.05, 0.1 ]
                ms_vtl[ 'CA' ] = 0.05 + ( ms_vtl[ 'XB' ] - 0.01 ) * 0.05/0.09


                # RA = XB but invert and rescale to [ -0.1, 1.1 ] instead of [ 0.01, 0.1 ]
                ms_vtl[ 'RA' ] = 1.0 - ( ms_vtl[ 'XB' ] - 0.01 ) * 1.2/0.09
                # set RA to 1.0
                #ms_vtl[ 'RA' ] = 1.0
                # RA = XB but invert and rescale to [ 0.0, 1.0 ] instead of [ 0.01, 0.1 ]
                #ms_vtl[ 'RA' ] = 1.0 - ( ms_vtl[ 'XB' ] - 0.01 ) * 1.0/0.09

                motor_data.append( ms_vtl )
            else:
                raise ValueError(
                    f"""
                    Invalid out_type: {out_type}. Type must be 'vtl' or 'tt2'.
                    """
                    )

        return motor_data
    

    def series_to_tensor(
            self,
            msrs,
            time_stretch = None,
            pitch_shift = None,
            in_type = 'tt2',
            ):
        """
        Convert motor series objects to TensorTract2 motor tensor inputs.

        Args:
            msrs: List[MotorSeries], List of motor series objects.
            time_stretch: float, Time stretch factor. Default is None.
            pitch_shift: float, Pitch shift factor. Default is None.

        Returns:
            m: torch.Tensor, (B, C, T), Motor tensor.
            m_len: torch.Tensor, (B,), Length of motor tensor.
        """
        if not isinstance( msrs, list ):
            msrs = [ msrs ]

        msrs_lens = []
        tt2_msrs = []
        for ms in msrs:
            if in_type == 'vtl':
                ms = ms.to_tt( target_sr=50 )
            if time_stretch is not None:
                ms.time_stretch( time_stretch )
            if pitch_shift is not None:
                ms.pitch_shift( pitch_shift )
            msrs_lens.append( len( ms ) )
            tt2_msrs.append( ms )

        motor_data = []
        # pad the motor data to the same length
        max_len = max( msrs_lens )
        for ms in tt2_msrs:
            t = torch.tensor( ms.to_numpy(transpose=True) ).float()
            t = F.pad( t, ( 0, max_len - len( ms ) ) )
            t = self.norm( t )
            motor_data.append( t )


        m = torch.stack( motor_data, dim=0 )
        m_len = torch.tensor( msrs_lens )

        return m, m_len
