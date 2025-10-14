"""
Numerical and Algorithmic Accuracy Investigation for ARX, FIR, and ARMAX Migration.

This investigation analyzes the harold branch implementation by:
1. Reading and comparing the source code algorithms
2. Testing execution and numerical behavior
3. Documenting findings about the migration
"""

import sys
import os
import numpy as np

# Add harold branch to path
sys.path.insert(0, '/Users/josephj/Workspace/SIPPY/src')

from sippy.identification import IDData, SystemIdentificationConfig, SystemIdentification
from sippy.identification.algorithms.arx import ARXAlgorithm
from sippy.identification.algorithms.fir import FIRAlgorithm
from sippy.identification.algorithms.armax import ARMAXAlgorithm

import pandas as pd


class AlgorithmInvestigator:
    """Investigate harold branch algorithms for numerical accuracy."""

    def __init__(self):
        self.findings = []

    def log_finding(self, category, test_name, result, details):
        """Log an investigation finding."""
        self.findings.append({
            'category': category,
            'test': test_name,
            'result': result,
            'details': details
        })
        print(f"\n{'='*80}")
        print(f"[{category}] {test_name}")
        print(f"Result: {result}")
        print(f"Details: {details}")
        print(f"{'='*80}")

    def generate_siso_arx_data(self, N=500, seed=42):
        """Generate SISO ARX test data."""
        np.random.seed(seed)

        # True ARX(2,2) system parameters
        a1_true, a2_true = 0.5, 0.3
        b1_true, b2_true = 0.8, 0.2

        # Generate input
        u = np.random.randn(N)

        # Generate output with known ARX model
        y = np.zeros(N)
        for k in range(2, N):
            y[k] = a1_true * y[k-1] + a2_true * y[k-2] + b1_true * u[k-1] + b2_true * u[k-2]

        # Add small noise
        y = y + 0.01 * np.random.randn(N)

        return u, y, (a1_true, a2_true, b1_true, b2_true)

    def investigate_arx_algorithm(self):
        """Investigate ARX algorithm implementation."""
        print("\n" + "="*80)
        print("INVESTIGATING ARX ALGORITHM")
        print("="*80)

        # Generate test data
        u, y, true_params = self.generate_siso_arx_data(N=500)
        a1_true, a2_true, b1_true, b2_true = true_params

        print(f"\nTrue ARX(2,2) parameters:")
        print(f"  A coefficients: a1={a1_true}, a2={a2_true}")
        print(f"  B coefficients: b1={b1_true}, b2={b2_true}")

        # Identify using harold branch
        time_index = pd.date_range("2023-01-01", periods=len(y), freq="1s")
        data_df = pd.DataFrame({"u": u, "y": y}, index=time_index)
        id_data = IDData(data=data_df, inputs=["u"], outputs=["y"], tsample=1.0)

        config = SystemIdentificationConfig(method="ARX")
        config.na = 2  # AR order
        config.nb = 2  # X order
        config.nk = 1  # Input delay

        algorithm = ARXAlgorithm()
        model = algorithm.identify(id_data, config)

        print(f"\nIdentified model (harold branch):")
        print(f"  State-space dimensions: A={model.A.shape}, B={model.B.shape}, C={model.C.shape}, D={model.D.shape}")

        # Extract A and B coefficients from the companion form state-space
        # For companion form, last row of A contains -a1, -a2
        if model.A.shape[0] >= 2:
            a_est = -model.A[-1, :2]
            print(f"  Estimated A coefficients: {a_est}")
            a_error = np.abs(a_est - np.array([a1_true, a2_true]))
            print(f"  A coefficient errors: {a_error}")

        # Check transfer function creation
        if hasattr(model, 'G_tf') and model.G_tf is not None:
            print(f"\n  Transfer function G(z) created: Yes")
            print(f"    Numerator: {model.G_tf.num}")
            print(f"    Denominator: {model.G_tf.den}")

            # Extract B coefficients from numerator
            # NUM should be [0, 0, b1, b2, ...] for nk=1, nb=2
            num = model.G_tf.num
            # Find where B coefficients start (after nk zeros)
            nk_start = config.nk
            if len(num) > nk_start + 1:
                b_est = num[nk_start:nk_start + config.nb]
                print(f"  Estimated B coefficients from G_tf: {b_est}")
                b_error = np.abs(b_est - np.array([b1_true, b2_true]))
                print(f"  B coefficient errors: {b_error}")

                # Check accuracy
                if np.max(a_error) < 0.01 and np.max(b_error) < 0.01:
                    self.log_finding(
                        "ARX", "Parameter Estimation Accuracy",
                        "EXCELLENT",
                        f"A error: {np.max(a_error):.6f}, B error: {np.max(b_error):.6f}"
                    )
                else:
                    self.log_finding(
                        "ARX", "Parameter Estimation Accuracy",
                        "NEEDS REVIEW",
                        f"A error: {np.max(a_error):.6f}, B error: {np.max(b_error):.6f}"
                    )
        else:
            print(f"\n  Transfer function G(z) created: No (harold not available)")
            self.log_finding(
                "ARX", "Transfer Function Creation",
                "WARNING",
                "harold transfer functions not created"
            )

        # Check one-step-ahead predictions
        if hasattr(model, 'Yid') and model.Yid is not None:
            yid = model.Yid.flatten() if model.Yid.ndim > 1 else model.Yid
            max_lag = max(config.na, config.nb + config.nk - 1)

            # Compare predictions to actual output
            pred_error = np.abs(y[max_lag:] - yid[max_lag:])
            max_pred_error = np.max(pred_error)
            mean_pred_error = np.mean(pred_error)

            print(f"\n  One-step-ahead prediction error:")
            print(f"    Max: {max_pred_error:.6f}")
            print(f"    Mean: {mean_pred_error:.6f}")

            # Since we added 0.01 noise, expect error around that magnitude
            if mean_pred_error < 0.02:
                self.log_finding(
                    "ARX", "Prediction Accuracy",
                    "EXCELLENT",
                    f"Mean prediction error {mean_pred_error:.6f} is close to noise level 0.01"
                )
            else:
                self.log_finding(
                    "ARX", "Prediction Accuracy",
                    "NEEDS REVIEW",
                    f"Mean prediction error {mean_pred_error:.6f} exceeds expected noise level"
                )

    def investigate_fir_algorithm(self):
        """Investigate FIR algorithm implementation."""
        print("\n" + "="*80)
        print("INVESTIGATING FIR ALGORITHM")
        print("="*80)

        # Generate FIR test data (pure impulse response)
        np.random.seed(42)
        N = 500

        # True FIR coefficients
        fir_true = np.array([0.5, 0.3, 0.2, 0.1, 0.05])
        nb_true = len(fir_true)

        # Generate input
        u = np.random.randn(N)

        # Generate FIR output
        y = np.zeros(N)
        for k in range(nb_true, N):
            for j in range(nb_true):
                y[k] += fir_true[j] * u[k - j - 1]

        # Add small noise
        y = y + 0.005 * np.random.randn(N)

        print(f"\nTrue FIR parameters:")
        print(f"  Coefficients: {fir_true}")

        # Identify using harold branch
        time_index = pd.date_range("2023-01-01", periods=N, freq="1s")
        data_df = pd.DataFrame({"u": u, "y": y}, index=time_index)
        id_data = IDData(data=data_df, inputs=["u"], outputs=["y"], tsample=1.0)

        config = SystemIdentificationConfig(method="FIR")
        config.nb = nb_true
        config.nk = 1

        algorithm = FIRAlgorithm()
        model = algorithm.identify(id_data, config)

        print(f"\nIdentified FIR model (harold branch):")
        print(f"  State-space dimensions: A={model.A.shape}, B={model.B.shape}, C={model.C.shape}, D={model.D.shape}")

        # Check transfer function
        if hasattr(model, 'G_tf') and model.G_tf is not None:
            print(f"\n  Transfer function G(z) created: Yes")
            print(f"    Numerator: {model.G_tf.num}")
            print(f"    Denominator: {model.G_tf.den}")

            # Extract FIR coefficients
            num = model.G_tf.num
            nk_start = config.nk
            if len(num) > nk_start:
                fir_est = num[nk_start:nk_start + nb_true]
                print(f"  Estimated FIR coefficients: {fir_est}")

                fir_error = np.abs(fir_est - fir_true)
                print(f"  FIR coefficient errors: {fir_error}")

                if np.max(fir_error) < 0.01:
                    self.log_finding(
                        "FIR", "Coefficient Estimation Accuracy",
                        "EXCELLENT",
                        f"Max error: {np.max(fir_error):.6f}"
                    )
                else:
                    self.log_finding(
                        "FIR", "Coefficient Estimation Accuracy",
                        "NEEDS REVIEW",
                        f"Max error: {np.max(fir_error):.6f}"
                    )
        else:
            print(f"\n  Transfer function G(z) created: No")
            self.log_finding(
                "FIR", "Transfer Function Creation",
                "WARNING",
                "harold transfer functions not created"
            )

    def investigate_armax_modes(self):
        """Investigate ARMAX algorithm modes (ILLS, RLLS, OPT)."""
        print("\n" + "="*80)
        print("INVESTIGATING ARMAX ALGORITHM MODES")
        print("="*80)

        # Generate ARMAX test data
        np.random.seed(42)
        N = 500

        # True ARMAX parameters
        a1_true, a2_true = 0.5, 0.3
        b1_true, b2_true = 0.7, 0.2
        c1_true = 0.4

        u = np.random.randn(N)
        e = 0.01 * np.random.randn(N)  # White noise
        y = np.zeros(N)

        # Generate ARMAX output: A(z)y = B(z)u + C(z)e
        for k in range(2, N):
            y[k] = (a1_true * y[k-1] + a2_true * y[k-2] +
                   b1_true * u[k-1] + b2_true * u[k-2] +
                   c1_true * e[k-1] + e[k])

        print(f"\nTrue ARMAX(2,2,1) parameters:")
        print(f"  A coefficients: a1={a1_true}, a2={a2_true}")
        print(f"  B coefficients: b1={b1_true}, b2={b2_true}")
        print(f"  C coefficients: c1={c1_true}")

        # Test each ARMAX mode
        modes = ['ILLS', 'RLLS', 'OPT']

        for mode in modes:
            print(f"\n{'-'*80}")
            print(f"Testing ARMAX Mode: {mode}")
            print(f"{'-'*80}")

            try:
                # Prepare data
                time_index = pd.date_range("2023-01-01", periods=N, freq="1s")
                data_df = pd.DataFrame({"u": u, "y": y}, index=time_index)
                id_data = IDData(data=data_df, inputs=["u"], outputs=["y"], tsample=1.0)

                config = SystemIdentificationConfig(method="ARMAX")
                config.na = 2
                config.nb = 2
                config.nc = 1
                config.nk = 1
                config.max_iterations = 50
                config.armx_mode = mode

                algorithm = ARMAXAlgorithm(mode=mode)
                model = algorithm.identify(id_data, config)

                print(f"  Model created: Yes")
                print(f"  State-space dimensions: A={model.A.shape}, B={model.B.shape}, C={model.C.shape}, D={model.D.shape}")

                # Check transfer functions
                if hasattr(model, 'G_tf') and model.G_tf is not None:
                    print(f"  G(z) transfer function: Created")
                    print(f"    Numerator: {model.G_tf.num}")
                    print(f"    Denominator: {model.G_tf.den}")

                if hasattr(model, 'H_tf') and model.H_tf is not None:
                    print(f"  H(z) transfer function: Created")
                    print(f"    Numerator: {model.H_tf.num}")
                    print(f"    Denominator: {model.H_tf.den}")

                self.log_finding(
                    "ARMAX", f"{mode} Mode Execution",
                    "SUCCESS",
                    f"Model created with dimensions {model.A.shape}"
                )

            except Exception as e:
                print(f"  ERROR: {e}")
                self.log_finding(
                    "ARMAX", f"{mode} Mode Execution",
                    "FAILED",
                    str(e)
                )

    def run_investigation(self):
        """Run complete investigation."""
        print("\n" + "="*80)
        print("NUMERICAL AND ALGORITHMIC ACCURACY INVESTIGATION")
        print("ARX, FIR, and ARMAX Algorithms - Harold Branch")
        print("="*80)

        self.investigate_arx_algorithm()
        self.investigate_fir_algorithm()
        self.investigate_armax_modes()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print investigation summary."""
        print("\n" + "="*80)
        print("INVESTIGATION SUMMARY")
        print("="*80)

        categories = {}
        for finding in self.findings:
            cat = finding['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(finding)

        for cat, findings in categories.items():
            print(f"\n{cat}:")
            for f in findings:
                result_marker = {
                    'EXCELLENT': '✓',
                    'SUCCESS': '✓',
                    'WARNING': '⚠',
                    'NEEDS REVIEW': '⚠',
                    'FAILED': '✗'
                }.get(f['result'], '?')

                print(f"  {result_marker} {f['test']}: {f['result']}")
                print(f"      {f['details']}")

        print(f"\n{'='*80}")
        print(f"Total findings: {len(self.findings)}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    investigator = AlgorithmInvestigator()
    investigator.run_investigation()
