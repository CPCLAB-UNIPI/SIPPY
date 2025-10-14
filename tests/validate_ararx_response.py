"""
ARARX NLP Validation: Time and Frequency Domain Response Comparison

Compares harold branch NLP implementation vs master branch by evaluating:
- Time domain: Step response, impulse response
- Frequency domain: Frequency response magnitude and phase
- Metrics: MSE, MAE, max error

This is the correct way to validate since different state-space realizations
can represent the same transfer function.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/josephj/Workspace/SIPPY-master')

from src.sippy.identification import SystemIdentification, SystemIdentificationConfig
from sippy_unipi import system_identification as master_sysid
import matplotlib.pyplot as plt

def compute_step_response(model, n_steps=100):
    """Compute step response for a model."""
    try:
        # Try harold model first
        if hasattr(model, 'A'):
            A, B, C, D = model.A, model.B, model.C, model.D
            nx = A.shape[0]
            x = np.zeros((nx, 1))
            u = np.ones((1, 1))  # Step input
            y_step = np.zeros(n_steps)
            
            for k in range(n_steps):
                y_step[k] = (C @ x + D @ u)[0, 0]
                x = A @ x + B @ u
            
            return y_step
        
        # Try control.matlab model
        elif hasattr(model, 'G'):
            import control.matlab as cnt
            G = model.G
            t = np.arange(n_steps)
            _, y_step = cnt.step(G, t)
            return y_step.flatten()
        
        else:
            return None
            
    except Exception as e:
        print(f"Error computing step response: {e}")
        return None

def compute_impulse_response(model, n_steps=100):
    """Compute impulse response for a model."""
    try:
        # Try harold model first
        if hasattr(model, 'A'):
            A, B, C, D = model.A, model.B, model.C, model.D
            nx = A.shape[0]
            x = np.zeros((nx, 1))
            y_imp = np.zeros(n_steps)
            
            for k in range(n_steps):
                if k == 0:
                    u = np.ones((1, 1))  # Impulse at k=0
                else:
                    u = np.zeros((1, 1))
                    
                y_imp[k] = (C @ x + D @ u)[0, 0]
                x = A @ x + B @ u
            
            return y_imp
        
        # Try control.matlab model
        elif hasattr(model, 'G'):
            import control.matlab as cnt
            G = model.G
            t = np.arange(n_steps)
            _, y_imp = cnt.impulse(G, t)
            return y_imp.flatten()
        
        else:
            return None
            
    except Exception as e:
        print(f"Error computing impulse response: {e}")
        return None

def compute_frequency_response(model, omega):
    """Compute frequency response for a model."""
    try:
        # Try harold model first
        if hasattr(model, 'G_tf') and model.G_tf is not None:
            import harold
            mag = np.zeros(len(omega))
            phase = np.zeros(len(omega))
            
            for i, w in enumerate(omega):
                z = np.exp(1j * w)  # Discrete-time frequency
                num = model.G_tf.num
                den = model.G_tf.den
                
                # Evaluate polynomial at z
                num_val = np.polyval(num.flatten(), z)
                den_val = np.polyval(den.flatten(), z)
                
                H = num_val / den_val
                mag[i] = np.abs(H)
                phase[i] = np.angle(H)
            
            return mag, phase
        
        # Try control.matlab model
        elif hasattr(model, 'G'):
            import control.matlab as cnt
            G = model.G
            mag_db, phase_deg, _ = cnt.bode(G, omega, plot=False)
            
            # Convert from dB to linear and degrees to radians
            mag = 10 ** (mag_db / 20)
            phase = np.deg2rad(phase_deg)
            
            return mag.flatten(), phase.flatten()
        
        else:
            return None, None
            
    except Exception as e:
        print(f"Error computing frequency response: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def compute_metrics(y1, y2, metric_name="Response"):
    """Compute comparison metrics between two responses."""
    if y1 is None or y2 is None:
        return None
    
    # Ensure same length
    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]
    
    # Compute metrics
    mse = np.mean((y1 - y2) ** 2)
    mae = np.mean(np.abs(y1 - y2))
    max_error = np.max(np.abs(y1 - y2))
    
    # Relative errors (avoid division by zero)
    y2_rms = np.sqrt(np.mean(y2 ** 2))
    if y2_rms > 1e-10:
        nmse = mse / (y2_rms ** 2)  # Normalized MSE
        nrmse = np.sqrt(nmse)  # Normalized RMSE
    else:
        nmse = nrmse = float('inf')
    
    # Correlation
    if np.std(y1) > 1e-10 and np.std(y2) > 1e-10:
        correlation = np.corrcoef(y1, y2)[0, 1]
    else:
        correlation = 0.0
    
    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'nmse': nmse,
        'nrmse': nrmse,
        'correlation': correlation
    }

def print_metrics(metrics, name="Response"):
    """Print metrics in a formatted table."""
    if metrics is None:
        print(f"{name}: Could not compute metrics")
        return False
    
    print(f"\n{name}:")
    print(f"  MSE:          {metrics['mse']:.6e}")
    print(f"  MAE:          {metrics['mae']:.6e}")
    print(f"  Max Error:    {metrics['max_error']:.6e}")
    print(f"  NMSE:         {metrics['nmse']:.6e}")
    print(f"  NRMSE:        {metrics['nrmse']:.6e}")
    print(f"  Correlation:  {metrics['correlation']:.6f}")
    
    # Pass/fail criteria
    passed = metrics['nrmse'] < 0.01 and metrics['correlation'] > 0.99
    status = "✅ PASS" if passed else "⚠️ NEEDS IMPROVEMENT"
    print(f"  Status:       {status}")
    
    return passed

def main():
    print("=" * 80)
    print("ARARX NLP Validation: Time and Frequency Domain Response Comparison")
    print("=" * 80)
    
    # Generate test data
    np.random.seed(42)
    N = 500  # More samples for better frequency resolution
    u = np.random.randn(1, N)
    y = np.zeros((1, N))
    
    # True system: SISO with some dynamics
    for k in range(2, N):
        y[0, k] = 0.6 * y[0, k-1] - 0.1 * y[0, k-2] + 0.3 * u[0, k-1] + 0.1 * u[0, k-2] + np.random.randn() * 0.01
    
    print(f"\nTest Data: N={N} samples, SISO system")
    print(f"  Input RMS:  {np.std(u):.4f}")
    print(f"  Output RMS: {np.std(y):.4f}")
    
    # Model orders
    na, nb, nd, theta = 2, 2, 1, 1
    max_iterations = 200
    
    print(f"\nModel Orders: na={na}, nb={nb}, nd={nd}, theta={theta}")
    print(f"IPOPT Iterations: {max_iterations}")
    
    # Harold branch identification (NLP with rescaling)
    print("\n" + "-" * 80)
    print("Identifying with Harold Branch (NLP + Rescaling)...")
    print("-" * 80)
    
    config = SystemIdentificationConfig(method="ARARX")
    config.na = na
    config.nb = nb
    config.nd = nd
    config.theta = theta
    config.max_iterations = max_iterations
    
    identifier = SystemIdentification(config)
    model_harold = identifier.identify(y=y, u=u)
    
    print(f"✅ Harold identification complete")
    print(f"  Vn (noise variance): {model_harold.Vn:.6f}")
    
    # Master branch identification
    print("\n" + "-" * 80)
    print("Identifying with Master Branch (Reference)...")
    print("-" * 80)
    
    model_master = master_sysid(
        y, u, "ARARX",
        ARARX_orders=[[na], [[nb]], [nd], [[theta]]],
        tsample=1.0,
        max_iterations=max_iterations
    )
    
    print(f"✅ Master identification complete")
    print(f"  Vn (noise variance): {model_master.Vn:.6f}")
    
    # ========================================================================
    # TIME DOMAIN COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("TIME DOMAIN RESPONSE COMPARISON")
    print("=" * 80)
    
    n_steps = 100
    
    # Step responses
    print("\nComputing step responses...")
    step_harold = compute_step_response(model_harold, n_steps)
    step_master = compute_step_response(model_master, n_steps)
    
    if step_harold is not None and step_master is not None:
        step_metrics = compute_metrics(step_harold, step_master, "Step Response")
        step_pass = print_metrics(step_metrics, "Step Response")
    else:
        print("⚠️ Could not compute step responses")
        step_pass = False
    
    # Impulse responses
    print("\nComputing impulse responses...")
    impulse_harold = compute_impulse_response(model_harold, n_steps)
    impulse_master = compute_impulse_response(model_master, n_steps)
    
    if impulse_harold is not None and impulse_master is not None:
        impulse_metrics = compute_metrics(impulse_harold, impulse_master, "Impulse Response")
        impulse_pass = print_metrics(impulse_metrics, "Impulse Response")
    else:
        print("⚠️ Could not compute impulse responses")
        impulse_pass = False
    
    # ========================================================================
    # FREQUENCY DOMAIN COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("FREQUENCY DOMAIN RESPONSE COMPARISON")
    print("=" * 80)
    
    # Frequency range (discrete-time, omega in [0, pi])
    n_freq = 100
    omega = np.linspace(0.01, np.pi, n_freq)
    
    print(f"\nComputing frequency responses ({n_freq} points)...")
    mag_harold, phase_harold = compute_frequency_response(model_harold, omega)
    mag_master, phase_master = compute_frequency_response(model_master, omega)
    
    freq_pass = False
    if mag_harold is not None and mag_master is not None:
        mag_metrics = compute_metrics(mag_harold, mag_master, "Magnitude Response")
        mag_pass = print_metrics(mag_metrics, "Magnitude Response")
        
        phase_metrics = compute_metrics(phase_harold, phase_master, "Phase Response")
        phase_pass = print_metrics(phase_metrics, "Phase Response (radians)")
        
        freq_pass = mag_pass and phase_pass
    else:
        print("⚠️ Could not compute frequency responses")
    
    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL VALIDATION RESULTS")
    print("=" * 80)
    
    results = {
        'Step Response': step_pass if 'step_pass' in locals() else False,
        'Impulse Response': impulse_pass if 'impulse_pass' in locals() else False,
        'Frequency Response': freq_pass
    }
    
    print("\nTest Results:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:.<30} {status}")
    
    overall_pass = all(results.values())
    
    print("\n" + "=" * 80)
    if overall_pass:
        print("🎉 OVERALL: ✅ PASS - NLP implementation matches master branch!")
        print("   ARARX NLP is PRODUCTION READY")
    else:
        print("⚠️  OVERALL: NEEDS IMPROVEMENT")
        print("   Some responses don't match within tolerance")
        print("   Target: NRMSE < 1%, Correlation > 0.99")
    print("=" * 80)
    
    return overall_pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
