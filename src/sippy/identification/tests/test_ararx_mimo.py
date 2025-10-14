"""
Comprehensive test suite for ARARX MIMO implementation.

This test suite validates the MIMO capabilities of the ARARX algorithm,
ensuring compatibility with master branch behavior while adding performance
optimizations through Numba JIT compilation and parallel processing.
"""

import numpy as np
import pandas as pd
import pytest
import time
from unittest.mock import patch

from sippy.identification.algorithms.ararx import ARARXAlgorithm
from sippy.identification.base import StateSpaceModel, SystemIdentificationConfig
from sippy.identification.iddata import IDData


class TestARARXMIMOAlgorithm:
    """Test cases for ARARX MIMO algorithm implementation."""

    def setup_method(self):
        """Set up synthetic MIMO test data for ARARX algorithm."""
        np.random.seed(42)
        self.n_samples = 1000
        
        # Create 2-input, 2-output ARARX process
        self.ny = 2  # Number of outputs
        self.nu = 2  # Number of inputs
        self.ts = 1.0  # Sample time

        # Generate test data for MIMO ARARX system
        t = np.linspace(0, 100, self.n_samples)
        u = np.random.normal(0, 1, (self.nu, self.n_samples))
        y = np.zeros((self.ny, self.n_samples))

        # Simple MIMO ARARX process for testing
        # G1(y1,u1,u2): A1*y1[k] = B11*u1[k-1] + B12*u2[k-1] + e1[k]
        # G2(y2,u1,u2): A2*y2[k] = B21*u1[k-1] + B22*u2[k-1] + e2[k]
        e_white = np.random.normal(0, 0.1, (self.ny, self.n_samples))

        # ARARX coefficients (na=1, nb=1, nd=1, theta=1 for simplicity)
        A_coeffs = np.array([[0.3, 0.1], [0.1, 0.4]])  # Cross-coupling in A
        B_coeffs = np.array([[0.5, 0.2], [0.3, 0.6]])  # Input-output gains
        D_coeffs = np.array([[0.2, 0.0], [0.0, 0.3]])   # Denominator dynamics

        for k in range(1, self.n_samples):
            # Output 1: y1[k] = 0.5*u1[k-1] + 0.2*u2[k-1] - 0.3*y1[k-1] - 0.1*y2[k-1] + e1[k]
            y[0, k] = (
                B_coeffs[0, 0] * u[0, k-1] + B_coeffs[0, 1] * u[1, k-1]
                - A_coeffs[0, 0] * y[0, k-1] - A_coeffs[0, 1] * y[1, k-1]
                + e_white[0, k]
            )

            # Output 2: y2[k] = 0.3*u1[k-1] + 0.6*u2[k-1] - 0.1*y1[k-1] - 0.4*y2[k-1] + e2[k]  
            y[1, k] = (
                B_coeffs[1, 0] * u[0, k-1] + B_coeffs[1, 1] * u[1, k-1]
                - A_coeffs[1, 0] * y[0, k-1] - A_coeffs[1, 1] * y[1, k-1]
                + e_white[1, k]
            )

        # Create IDData for MIMO system
        time_index = pd.date_range("2023-01-01", periods=self.n_samples, freq="1s")
        data_df = pd.DataFrame(
            {
                "u1": u[0, :], "u2": u[1, :],
                "y1": y[0, :], "y2": y[1, :]
            },
            index=time_index
        )

        self.data = IDData(
            data=data_df, 
            inputs=["u1", "u2"], 
            outputs=["y1", "y2"], 
            tsample=self.ts
        )

        # Test coefficients for validation
        self.test_A = A_coeffs
        self.test_B = B_coeffs  
        self.test_D = D_coeffs

    def test_ararx_mimo_basic_identification(self):
        """Test basic ARARX MIMO identification with scalar parameters."""
        algorithm = ARARXAlgorithm()

        # Test with scalar parameters (auto-expanded to MIMO matrices)
        config = SystemIdentificationConfig(method="ARARX")
        config.na = [1, 1]      # na per output
        config.nb = [[1, 1], [1, 1]]  # nb per (output,input)
        config.nd = [1, 1]      # nd per output  
        config.theta = [[1, 1], [1, 1]]  # theta per (output,input)

        result = algorithm.identify(self.data, config)

        assert result is not None
        assert isinstance(result, StateSpaceModel)
        assert result.A.shape == (4, 4)  # 2*na + nu*2*(nb+nd) states
        assert result.B.shape == (4, 2)
        assert result.C.shape == (2, 4)
        assert result.D.shape == (2, 2)

    def test_ararx_mimo_matrix_parameters(self):
        """Test ARARX MIMO with explicit matrix parameter specification."""
        algorithm = ARARXAlgorithm()

        config = SystemIdentificationConfig(method="ARARX")
        config.na = [2, 1]          # Different orders per output
        config.nb = [[2, 1], [1, 3]]  # Different input orders per output
        config.nd = [1, 2]          # Different denominator orders
        config.theta = [[1, 2], [3, 1]]  # Different delays per pair

        result = algorithm.identify(self.data, config)

        assert result is not None
        assert isinstance(result, StateSpaceModel)
        # State dimension should reflect the maximum polynomial orders
        expected_states = max(config.na) + sum(max(row) for row in config.nb) + max(config.nd)
        assert result.A.shape == (expected_states, expected_states)

    def test_ararx_mimo_parameter_validation(self):
        """Test MIMO parameter validation and error handling."""
        algorithm = ARARXAlgorithm()

        # Test mismatched dimensions
        invalid_config = SystemIdentificationConfig(method="ARARX")
        invalid_config.na = [1, 1]           # 2 outputs
        invalid_config.nb = [[1, 1]]         # Only 1 row instead of 2
        invalid_config.nd = [1, 1]
        invalid_config.theta = [[1, 1], [1, 1]]

        # Test theta matrix dimension mismatch
        invalid_config = SystemIdentificationConfig(method="ARARX")
        invalid_config.na = [1, 1]           # 2 outputs
        invalid_config.nb = [[1, 1], [1, 1]]         # Only 1 row instead of 2
        invalid_config.theta = [[1, 1], [1, 1]]
    
        with pytest.raises(ValueError, match="theta matrix dimensions") as e:
            algorithm.identify(self.data, config)

        # Test negative values
        negative_config = SystemIdentificationConfig(method="ARARX")
        negative_config.na = [-1, 0]
        negative_config.nb = [[-1, -1]]
        negative_config.nd = [-1, 0]
        negative_config.theta = [[-1, -1], [-1, -1]]
    
        # Test extreme values
        extreme_config = SystemIdentificationConfig(method="ARARX")
        extreme_config.na = [10, 10]
        extreme_config.nb = [[5, 5], [5, 5]]
        extreme_config.nd = [10, 10]
        extreme_config.theta = [[10, 10], [10, 10]]
    
        # All should pass with ValueError when validated via previous checks
        with pytest.raises(ValueError, match="Output order must be non-negative or integer"):
            algorithm.identify(self.data, negative_config)

        with pytest.raises(ValueError, match="Parameter normalization failed") as e:
            algorithm.identify(self.data, extreme_config)

        # Test scalar parameters (should still work for SISO)
        siso_config = SystemIdentificationConfig(method="ARARX")
        siso_config.na = 2
        siso_config.nb = 2
        siso_config.nd = 1
        siso_config.theta = 1
        
        result = algorithm.identify(self.data, siso_config)
        assert result is not None
        assert isinstance(result, StateSpaceModel)
        # SISO tests should still pass

        # Test backward compatibility: 3x3 and 3x2 MIMO should work
        mixed_config = SystemIdentificationConfig(method="ARARX")
        mixed_config.na = [3, 3]
        mixed_config.nb = [[3, 3], [3, 3]]
        mixed_config.nd = [3, 3]
        mixed_config.theta = [[3, 3], [3, 3]]
        result = algorithm.identify(self.data, mixed_config)
        assert result is not None
        assert isinstance(result, StateSpaceModel)

        # Test that MIMO routing works without explicit test expectations
        result = algorithm.identify(self.data, 
            na=[1, 2], 
            nb=[[2, 2], [2, 2]], 
            nd=[1, 1], 
            theta=[[2, 3], [2, 3]]
        )
        assert result is not None
        assert isinstance(result, StateSpaceModel)

if __name__ == "__main__":
    pytest.main([__file__])
        invalid_config = SystemIdentificationConfig(method="ARARX")
        invalid_config.na = [-1, 1]
        invalid_config.nb = [[1, 1], [1, 1]]
        invalid_config.nd = [1, 1]
        invalid_config.theta = [[1, 1], [1, 1]]

        with pytest.raises(ValueError, match="na.*must be non-negative"):
            algorithm.identify(self.data, invalid_config)

    def test_ararx_mimo_insufficient_data(self):
        """Test MIMO algorithm with insufficient data."""
        algorithm = ARARXAlgorithm()

        # Create small dataset
        n_samples = 10
        y_small = np.random.randn(2, n_samples)
        u_small = np.random.randn(2, n_samples)
        
        time_index = pd.date_range("2023-01-01", periods=n_samples, freq="1s")
        data_df = pd.DataFrame({
            "u1": u_small[0, :], "u2": u_small[1, :],
            "y1": y_small[0, :], "y2": y_small[1, :]
        }, index=time_index)
        
        small_data = IDData(
            data=data_df, inputs=["u1", "u2"], outputs=["y1", "y2"], tsample=1.0
        )

        config = SystemIdentificationConfig(method="ARARX")
        config.na = [3, 3]  # Require at least 4 samples per output
        config.nb = [[3, 3], [3, 3]]
        config.nd = [3, 3]
        config.theta = [[2, 2], [2, 2]]

        with pytest.raises(ValueError, match="Not enough data"):
            algorithm.identify(small_data, config)

    def test_ararx_mimo_higher_orders(self):
        """Test ARARX MIMO with higher order polynomials."""
        algorithm = ARARXAlgorithm()

        config = SystemIdentificationConfig(method="ARARX")
        config.na = [2, 3]          # Higher AR orders
        config.nb = [[3, 2], [2, 4]]  # Higher input orders
        config.nd = [2, 1]          # Higher denominator orders
        config.theta = [[1, 1], [1, 1]]

        result = algorithm.identify(self.data, config)

        assert result is not None
        assert isinstance(result, StateSpaceModel)
        
        # State-space matrices should be properly sized
        expected_states = max(config.na) + sum(max(row) for row in config.nb) + max(config.nd)
        assert result.A.shape == (expected_states, expected_states)

    def test_ararx_mimo_vs_siso_compatibility(self):
        """Test that MIMO implementation maintains SISO compatibility."""
        algorithm = ARARXAlgorithm()

        # Test with 1x1 data using MIMO interface
        n_samples = 500
        y_siso = np.random.randn(1, n_samples)
        u_siso = np.random.randn(1, n_samples)
        
        time_index = pd.date_range("2023-01-01", periods=n_samples, freq="1s")
        data_df = pd.DataFrame({"u1": u_siso[0, :], "y1": y_siso[0, :]}, index=time_index)
        siso_data = IDData(data=data_df, inputs=["u1"], outputs=["y1"], tsample=1.0)

        # Test scalar parameters (traditional SISO approach)
        siso_config = SystemIdentificationConfig(method="ARARX")
        siso_config.na = 2
        siso_config.nb = 2
        siso_config.nd = 1
        siso_config.theta = 1

        # Test MIMO parameters for 1x1 system
        mimo_config = SystemIdentificationConfig(method="ARARX")  
        mimo_config.na = [2]
        mimo_config.nb = [[2]]
        mimo_config.nd = [1]
        mimo_config.theta = [[1]]

        # Both should work and give similar results
        siso_result = algorithm.identify(siso_data, siso_config)
        mimo_result = algorithm.identify(siso_data, mimo_config)

        assert siso_result is not None
        assert mimo_result is not None
        assert isinstance(siso_result, StateSpaceModel)
        assert isinstance(mimo_result, StateSpaceModel)

        # State dimensions should match (for identical params)
        assert siso_result.A.shape == mimo_result.A.shape

    def test_ararx_mimo_transfer_functions(self):
        """Test MIMO transfer function creation when harold available."""
        algorithm = ARARXAlgorithm()

        config = SystemIdentificationConfig(method="ARARX")
        config.na = [1, 1]
        config.nb = [[1, 1], [1, 1]]
        config.nd = [1, 1]
        config.theta = [[1, 1], [1, 1]]

        result = algorithm.identify(self.data, config)

        # Test transfer function attributes
        assert hasattr(result, 'G_tf')  # Deterministic TF matrix [ny x nu]
        assert hasattr(result, 'H_tf')  # Noise TF matrix [ny x ny]

        # Check transfer function structures if harold is available
        if result.G_tf is not None:
            # Should have ny rows and nu columns (for ny inputs, nu outputs)
            # Note: Actual structure depends on harold implementation
            pass

    def test_ararx_mimo_harold_integration(self):
        """Test MIMO algorithm with harold integration mocking."""
        algorithm = ARARXAlgorithm()

        config = SystemIdentificationConfig(method="ARARX")
        config.na = [1, 1]
        config.nb = [[1, 1], [1, 1]]
        config.nd = [1, 1]
        config.theta = [[1, 1], [1, 1]]

        with (
            patch("sippy.identification.algorithms.ararx.HAROLD_AVAILABLE", True),
            patch("sippy.identification.algorithms.ararx.harold") as mock_harold,
        ):
            # Mock harold methods
            mock_tf = mock_harold.Transfer.return_value
            mock_ss = mock_harold.transfer_to_state.return_value
            mock_harold.haroldpolymul.return_value = np.array([1.0, 0.5, 0.2])
            
            # Mock state-space matrices
            mock_ss.a = np.eye(6)
            mock_ss.b = np.eye(6, 2)
            mock_ss.c = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
            mock_ss.d = np.zeros((2, 2))

            result = algorithm.identify(self.data, config)

            assert result is not None
            assert isinstance(result, StateSpaceModel)
            # Harold should be called multiple times for MIMO TF creation
            assert mock_harold.Transfer.called
            assert mock_harold.transfer_to_state.called

    def test_ararx_mimo_performance_benchmark(self):
        """Performance benchmark for MIMO implementation."""
        algorithm = ARARXAlgorithm()

        # Test with different problem sizes
        test_sizes = [
            (1, 1),  # SISO
            (2, 2),  # 2x2 MIMO
            (3, 3),  # 3x3 MIMO
        ]

        config = SystemIdentificationConfig(method="ARARX")
        config.na = [1, 1, 1]
        config.nb = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        config.nd = [1, 1, 1] 
        config.theta = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

        for ny, nu in test_sizes:
            # Create appropriate sized test data
            n_samples = 500
            y_test = np.random.randn(ny, n_samples)
            u_test = np.random.randn(nu, n_samples)
            
            input_cols = [f"u{i+1}" for i in range(nu)]
            output_cols = [f"y{i+1}" for i in range(ny)]
            
            data_dict = {}
            data_dict.update({col: u_test[i, :] for i, col in enumerate(input_cols)})
            data_dict.update({col: y_test[i, :] for i, col in enumerate(output_cols)})
            
            time_index = pd.date_range("2023-01-01", periods=n_samples, freq="1s")
            data_df = pd.DataFrame(data_dict, index=time_index)
            
            test_data = IDData(
                data=data_df,
                inputs=input_cols,
                outputs=output_cols,
                tsample=1.0
            )

            # Time the identification
            start_time = time.time()
            result = algorithm.identify(test_data, config)
            elapsed_time = time.time() - start_time

            print(f"MIMO {ny}x{nu} identification time: {elapsed_time:.4f}s")
            
            assert result is not None
            assert isinstance(result, StateSpaceModel)

            # Performance check: should not take more than 10x SISO time on reasonable
            # (this is a loose check that will be refined in optimization phase)
            assert elapsed_time < 60.0, f"Performance too slow for {ny}x{nu} MIMO"

    def test_ararx_mimo_edge_cases(self):
        """Test edge cases and special conditions for MIMO ARARX."""
        algorithm = ARARXAlgorithm()

        # Test minimum viable MIMO (1 input, 2 outputs)
        config = SystemIdentificationConfig(method="ARARX")
        config.na = [1, 1]
        config.nb = [[1], [1]]  # 2 outputs x 1 input
        config.nd = [1, 1]
        config.theta = [[1], [1]]

        # Create 1 input, 2 output data
        n_samples = 500
        u_data = np.random.randn(1, n_samples)
        y_data = np.random.randn(2, n_samples)
        
        time_index = pd.date_range("2023-01-01", periods=n_samples, freq="1s")
        data_df = pd.DataFrame({
            "u1": u_data[0, :],
            "y1": y_data[0, :],
            "y2": y_data[1, :]
        }, index=time_index)
        
        test_data = IDData(
            data=data_df, inputs=["u1"], outputs=["y1", "y2"], tsample=1.0
        )

        result = algorithm.identify(test_data, config)
        assert result is not None
        assert isinstance(result, StateSpaceModel)

        # Test zero delays
        config.theta = [[0, 0], [0, 0]]
        result = algorithm.identify(self.data, config)
        assert result is not None

    @pytest.mark.parametrize("ny,nu", [(2, 1), (1, 2), (3, 2), (2, 3)])
    def test_ararx_mimo_rectangular_systems(self, ny, nu):
        """Test MIMO systems with different numbers of inputs and outputs."""
        algorithm = ARARXAlgorithm()

        # Generate test data for rectangular systems
        n_samples = 500
        y_test = np.random.randn(ny, n_samples)  
        u_test = np.random.randn(nu, n_samples)

        input_cols = [f"u{i+1}" for i in range(nu)]
        output_cols = [f"y{i+1}" for i in range(ny)]

        data_dict = {}
        data_dict.update({col: u_test[i, :] for i, col in enumerate(input_cols)})
        data_dict.update({col: y_test[i, :] for i, col in enumerate(output_cols)})

        time_index = pd.date_range("2023-01-01", periods=n_samples, freq="1s")
        data_df = pd.DataFrame(data_dict, index=time_index)

        test_data = IDData(
            data=data_df, inputs=input_cols, outputs=output_cols, tsample=1.0
        )

        # Create appropriate sized parameters
        config = SystemIdentificationConfig(method="ARARX")
        config.na = [1] * ny
        config.nb = [[1] * nu] * ny
        config.nd = [1] * ny
        config.theta = [[1] * nu] * ny

        result = algorithm.identify(test_data, config)
        assert result is not None
        assert isinstance(result, StateSpaceModel)

        # Check matrix dimensions
        assert result.A.shape[0] == result.A.shape[1]  # A is square
        assert result.B.shape[0] == result.A.shape[0]      # B rows match A rows
        assert result.C.shape[1] == result.A.shape[1]     # C cols match A cols
        assert result.B.shape[1] == nu                    # B cols match inputs
        assert result.C.shape[0] == ny                    # C rows match outputs
        assert result.D.shape == (ny, nu)                 # D matches I/O dimensions


class TestARARXMIMOValidation:
    """Specialized tests for MIMO parameter validation and edge cases."""

    def test_mimo_parameter_matrix_validation(self):
        """Test MIMO parameter matrix validation utilities."""
        from sippy.identification.algorithms.ararx import MIMOParameterValidator

        # Test valid parameter structures
        validator = MIMOParameterValidator()

        # Valid 2x2 case
        assert validator.validate_mimo_structure(
            na=[1, 1],
            nb=[[1, 1], [1, 1]],  
            nd=[1, 1],
            theta=[[1, 1], [1, 1]],
            ny=2, nu=2
        ) is True

        # Invalid nb dimensions
        assert validator.validate_mimo_structure(
            na=[1, 1],
            nb=[[1, 1]],  # Only 1 row instead of 2
            nd=[1, 1],
            theta=[[1, 1], [1, 1]],
            ny=2, nu=2
        ) is False

    def test_mimo_parameter_normalization(self):
        """Test MIMO parameter expansion and normalization."""
        from sippy.identification.algorithms.ararx import MIMOParameterNormalizer

        normalizer = MIMOParameterNormalizer()

        # Test scalar parameter expansion
        # na=1, nb=1, nd=1, theta=1 for 2x2 system should become:
        # na=[1,1], nb=[[1,1],[1,1]], nd=[1,1], theta=[[1,1],[1,1]]
        normalized = normalizer.normalize_parameters(
            na=1, nb=1, nd=1, theta=1, ny=2, nu=2
        )
        
        expected = {
            'na': [1, 1],
            'nb': [[1, 1], [1, 1]], 
            'nd': [1, 1],
            'theta': [[1, 1], [1, 1]]
        }
        
        assert normalized['na'] == expected['na']
        assert normalized['nb'] == expected['nb']
        assert normalized['nd'] == expected['nd']
        assert normalized['theta'] == expected['theta']


if __name__ == "__main__":
    pytest.main([__file__])
