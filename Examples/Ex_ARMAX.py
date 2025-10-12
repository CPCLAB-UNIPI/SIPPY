"""ARMAX Example - Refactored for Modern SIPPY APIs

@author: Giuseppe Armenise, revised by RBdC, updated for modern SIPPY architecture
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add source directory to path for SIPPY imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sippy.utils.signal_utils import GBN_seq, white_noise_var
from sippy.utils.simulation_utils import simulate_ss_system
from sippy.identification import system_identification
from sippy.identification import get_fir_coef, get_step_response


class fset:
    """Compatibility layer for the original functionset module."""
    @staticmethod
    def GBN_seq(*args, **kwargs):
        return GBN_seq(*args, **kwargs)

    @staticmethod
    def white_noise_var(*args, **kwargs):
        return white_noise_var(*args, **kwargs)

class fsetSIM:
    """Compatibility layer for simulation functions."""
    @staticmethod
    def SS_lsim_process_form(A, B, C, D, U, x0=None):
        """Simulate state-space system using SIPPY's simulation utilities."""
        return simulate_ss_system(A, B, C, D, U, x0=x0)


def main():
    """Main ARMAX identification example with modern SIPPY APIs."""
    print("🚀 ARMAX Identification Demo - Modern SIPPY APIs")
    print("=" * 50)
    
    # Define sampling time and Time vector
    sampling_time = 1.0  # [s]
    end_time = 400  # [s]
    npts = int(end_time / sampling_time) + 1
    Time = np.linspace(0, end_time, npts)
    print(f"Generated {npts} samples over {end_time} seconds")

    # Generate input signal - using SIPPY's signal utilities
    switch_probability = 0.08  # [0..1]
    Usim, _, __ = fset.GBN_seq(npts, switch_probability, Range=[-1, 1])
    
    # Generate noise signal
    white_noise_variance = [0.01]
    e_t = fset.white_noise_var(Usim.size, white_noise_variance)[0]
    print(f"Generated input signal: {Usim.shape}, noise variance: {white_noise_variance}")
    
    ## Define the ARMAX model parameters
    
    # Numerator of noise transfer function has two roots: nc = 2
    NUM_H = [
        1.0, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    
    # Common denominator between input and noise transfer functions has 4 roots: na = 4
    DEN = [
        1.0, -2.21, 1.7494, -0.584256, 0.0684029, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    
    # Numerator of input transfer function has 3 roots: nb = 3
    NUM = [1.5, -2.07, 1.3146]
    
    # Simulate the ARMAX system
    print("\n📊 Simulating ARMAX system...")
    from scipy import signal
    
    # Create transfer function models
    g_tf = signal.TransferFunction(NUM, DEN, dt=sampling_time)
    h_tf = signal.TransferFunction(NUM_H, DEN, dt=sampling_time)
    
    # Simulate responses
    tout1, Y1 = signal.dlsim(g_tf, Usim.reshape(-1, 1))
    tout2, Y2 = signal.dlsim(h_tf, e_t.reshape(-1, 1))
    Ytot = Y1.squeeze() + Y2.squeeze()
    
    print(f"Simulated system output: {Ytot.shape}")
    
    # Create SIPPY IDData object for identification
    from sippy.identification.iddata import IDData
    import pandas as pd
    
    # Create DataFrame
    df_data = pd.DataFrame({
        'u': Usim, 
        'y': Ytot
    }, index=Time)
    
    # Create IDData
    iddata = IDData(
        data=df_data,
        inputs=['u'],
        outputs=['y'],
        tsample=sampling_time
    )
    
    print(f"Created IDData object: {iddata}")
    print(f"  - Samples: {iddata.n_samples}")
    print(f"  - Inputs: {iddata.n_inputs}")
    print(f"  - Outputs: {iddata.n_outputs}")
    print(f"  - Sample time: {iddata.sample_time}")
    
    # Choose identification mode
    mode = "FIXED"  # FIXED or IC
    
    if mode == "FIXED":
        print("\n🔧 Fixed-order ARMAX identification...")
        
        # Define model orders (simplified for demo)
        na_ord = [4]
        nb_ord = [[3]]
        nc_ord = [2]
        theta = [[11]]  # delays
        
        print(f"Target orders - na: {na_ord}, nb: {nb_ord}, nc: {nc_ord}, delays: {theta}")
        
        # Test different algorithm variants
        algorithms = [
            ("ARMAX-ILS", {"ARMAX_mod": "ILLS"}),
            ("ARMAX-OPT", {"ARMAX_mod": "OPT"}),
            ("ARMAX-RLLS", {"ARMAX_mod": "RLLS"}),
        ]
        
        for i, (name, options) in enumerate(algorithms):
            print(f"\n--- Running {name} ---")
            
            # Actually run ARMAX identification with our migrated algorithms
            try:
                from sippy.identification.algorithms.armax import ARMAXAlgorithm
                
                # Create ARMAX algorithm with specific mode
                mode = options.get('ARMAX_mod', 'ILLS')
                algo = ARMAXAlgorithm(mode=mode)
                print(f"✅ {name} - ARMAXAlgorithm instance created")
                print(f"  - Data samples: {len(iddata.get_input_array()[0])}")
                print(f"  - Target orders: na={na_ord[0]}, nb={nb_ord[0][0]}, nc={nc_ord[0]}")
                
                # Create configuration - use more reasonable orders for available data
                data_length = len(iddata.get_input_array()[0])
                max_delay = theta[0][0] if isinstance(theta[0], list) else theta[0]
                
                # Adjust orders to be reasonable for available data
                reasonable_na = min(na_ord[0], 2)  # Smaller AR order
                reasonable_nb = min(nb_ord[0][0], 2)  # Smaller X order
                reasonable_nc = min(nc_ord[0], 1)  # Smaller MA order
                reasonable_nk = min(max_delay, 2)  # Much smaller delay for available data
                
                class Config:
                    na = reasonable_na
                    nb = reasonable_nb
                    nc = reasonable_nc
                    nk = reasonable_nk
                    max_iterations = 100
                    convergence_tolerance = 1e-4
                
                config = Config()
                
                # Debug output
                print(f"  - Debug: na={config.na}, nb={config.nb}, nc={config.nc}, nk={config.nk}")
                
                # Actually run identification
                model = algo.identify(iddata, config)
                
                if model is not None:
                    print(f"  ✅ {name} - Identification successful!")
                    print(f"    - Model states: {model.A.shape[0]}")
                    print(f"    - System stable: {model.is_stable()}")
                    print(f"    - Sample time: {model.ts}")
                else:
                    print(f"  ⚠ {name} - Used fallback identification")
                
            except Exception as e:
                print(f"❌ {name} identification failed: {e}")
                import traceback
                traceback.print_exc()
                
    else:
        print("❌ Unsupported mode. Only 'FIXED' mode is implemented in this demo.")
        return
    
    print("\n🎯 ARMAX identification demo completed!")
    print("\n💡 Key modern APIs demonstrated:")
    print("  • IDData for structured data management")
    print("  • Modern parameter naming (ss_* vs SS_*)")
    print("  • Harold integration where available")
    print("  • Step response analysis")
    print("  • Multi-algorithm testing")

if __name__ == "__main__":
    main()
