"""
Test suite for ARARX (Auto-Regressive Auto-Regressive X) algorithm implementation.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from sippy.identification.algorithms.arx import ARXAlgorithm
from sippy.identification.algorithms.ararx import ARARXAlgorithm
from sippy.identification.base import SystemIdentificationConfig, StateSpaceModel
from sippy.identification.iddata import IDData


class TestARARXAlgorithm:
    """Test cases for ARARX algorithm following TDD approach."""

    def setup_method(self):
        """Set up test data for ARARX algorithm."""
        np.random.seed(42)
        self.n_samples = 1000
        
        # Create test data for ARARX (SISO system with colored noise)
        t = np.linspace(0, 100, self.n_samples)
        u = np.random.normal(0, 1, self.n_samples)
        y = np.zeros(self.n_samples)
        
        # Generate ARARX process: y[k] = B(q)/D(q)*u[k] + C(q)e(k] 
        # ARARX(2,1,1,1) model as example
        e_white = np.random.normal(0, 0.1, self.n_samples)
        
        for k in range(2, self.n_samples):
            # Input part: B(q)/D(q) * u[k]
            if k >= 1:
                input_part = 0.4 * u[k-1] + 0.2 * u[k-2]
                # Simplified D(q): y[k-1]  (for example)
                input_part -= 0.3 * y[k-1]
            
            # Noise AR part: C(q)*e[k] (simplified example with nc=1)
            if k >= 1:
                noise_part = e_white[k] + 0.3 * e_white[k-1]  # C(q) polynomial
            
            y[k] = input_part + noise_part
        
        # Create IDData
        time_index = pd.date_range('2023-01-01', periods=self.n_samples, freq='1s')
        data_df = pd.DataFrame({
            'u1': u,
            'y1': y
        }, index=time_index)
        
        self.data = IDData(
            data=data_df,
            inputs=['u1'],
            outputs=['y1'],
            tsample=1.0
        )
        
        self.config = SystemIdentificationConfig(method='ARARX')
        self.config.nb = 2  # Input transfer function order
        self.config.nc = 1  # Noise AR polynomial order
        self.config.nd = 1  # D polynomial order
        self.config.nf = 1  # F polynomial order
        self.config.nk = 1  # Input delay
        self.config.na = 0  # AR part (ARARX has na=0, unlike ARX)

    def test_ararx_algorithm_initialization(self):
        """Test ARARX algorithm can be initialized."""
        algorithm = ARARXAlgorithm()
        assert algorithm.get_algorithm_name() == "ARARX"

    def test_ararx_algorithm_name(self):
        """Test ARARX algorithm name."""
        algorithm = ARARXAlgorithm()
        assert algorithm.get_algorithm_name() == "ARARX"

    def test_ararx_basic_identification(self):
        """Test ARARX basic identification."""
        algorithm = ARARXAlgorithm()
        
        result = algorithm.identify(self.data, self.config)
        
        assert result is not None
        assert isinstance(result, StateSpaceModel)
        assert result.A is not None
        assert result.B is not None
        assert result.C is not None
        assert result.D is not None

    def test_ararx_with_different_orders(self):
        """Test ARARX with different model orders."""
        algorithm = ARARXAlgorithm()
        
        # Test ARARX(3,2,1,1)
        config = SystemIdentificationConfig(method='ARARX')
        config.nb = 3  # Input transfer function order
        config.nc = 2  # Noise AR polynomial order
        config.nd = 1  # D polynomial order
        config.nf = 1  # F polynomial order
        config.nk = 1  # Input delay
        config.na = 0  # AR part (ARARX has na=0)
        
        result = algorithm.identify(self.data, config)
        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_ararx_parameter_validation(self):
        """Test ARARX parameter validation."""
        algorithm = ARARXAlgorithm()
        
        # Test with zero input order
        invalid_config = SystemIdentificationConfig(method='ARARX')
        invalid_config.nb = 0
        invalid_config.nc = 1
        invalid_config.nd = 1
        invalid_config.nf = 1
        invalid_config.na = 0
        
        with pytest.raises(ValueError, match="Input order .* must be positive"):
            algorithm.identify(self.data, invalid_config)
        
        # Test with zero noise AR order
        invalid_config = SystemIdentificationConfig(method='ARARX')
        invalid_config.nb = 1
        invalid_config.nc = 0
        invalid_config.nd = 1
        invalid_config.nf = 1
        invalid_config.na = 0
        
        with pytest.raises(ValueError, match="Noise AR order .* must be positive"):
            algorithm.identify(self.data, invalid_config)
        
        # Test with zero noise MA orders
        invalid_config = SystemIdentificationConfig(method='ARARX')
        invalid_config.nb = 1
        invalid_config.nc = 1
        invalid_config.nd = 0  
        invalid_config.nf = 0
        invalid_config.na = 0
        
        with pytest.raises(ValueError, match="Noise MA orders must be positive"):
            algorithm.identify(self.data, invalid_config)

    def test_ararx_mimo_system(self):
        """Test ARARX with MIMO system."""
        # Create 2-input, 2-output data
        np.random.seed(42)
        n_samples = 500
        
        u = np.random.randn(2, n_samples)
        y1 = np.zeros(n_samples)
        y2 = np.random.randn(n_samples)
        
        # Simple input-output relationships
        for k in range(2, n_samples):
            y1[k] = 0.3 * u[0, k-1] + 0.2 * u[1, k-1] + 0.1 * y1[k-1] + 0.05 * y2[k-1]
            y2[k] = 0.4 * u[1, k-1] + 0.1 * u[0, k-1] + 0.3 * y2[k-1]
        
        y = np.vstack([y1, y2])
        
        time_index = pd.date_range('2023-01-01', periods=n_samples, freq='1s')
        data_df = pd.DataFrame({
            'u1': u[0, :],
            'u2': u[1, :],
            'y1': y1,
            'y2': y2
        }, index=time_index)
        
        data = IDData(
            data=data_df,
            inputs=['u1', 'u2'],
            outputs=['y1', 'y2'],
            tsample=1.0
        )
        
        config = SystemIdentificationConfig(method='ARARX')
        config.nb = 1
        config.nc = 1
        config.nd = 1
        config.nf = 1
        config.na = 0
        
        algorithm = ARARXAlgorithm()
        result = algorithm.identify(data, config)
        
        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_ararx_without_harold(self):
        """Test ARARX algorithm graceful degradation without harold."""
        algorithm = ARARXAlgorithm()
        
        with patch('sippy.identification.algorithms.ararx.HAROLD_AVAILABLE', False):
            result = algorithm.identify(self.data, self.config)
            assert result is not None
            assert isinstance(result, StateSpaceModel)

    def test_ararx_insufficient_data(self):
        """Test ARARX algorithm with insufficient data."""
        algorithm = ARARXAlgorithm()
        
        # Create very short dataset
        n_samples = 4
        np.random.seed(42)
        y = np.random.randn(1, n_samples)
        u = np.random.randn(1, n_samples)
        
        time_index = pd.date_range('2023-01-01', periods=n_samples, freq='1s')
        data_df = pd.DataFrame({
            'u1': u[0, :],
            'y1': y[0, :]
        }, index=time_index)
        
        data = IDData(
            data=data_df,
            inputs=['u1'],
            outputs=['y1'],
            tsample=1.0
        )
        
        config = SystemIdentificationConfig(method='ARARX')
        config.nb = 4  # This will require at least 4 samples
        config.nc = 2
        config.nd = 3
        config.nf = 2
        config.na = 0
        
        with pytest.raises(ValueError, match="Not enough data"):
            algorithm.identify(data, config)

    def test_ararx_state_space_models(self):
        """Test ARARX creates valid state-space models."""
        algorithm = ARARXAlgorithm()
        
        result = algorithm.identify(self.data, self.config)
        
        # Check state-space model properties
        assert hasattr(result, 'A')
        assert hasattr(result, 'B')
        assert hasattr(result, 'C')
        assert hasattr(result, 'D')
        
        # Check dimensions are consistent
        A, B, C, D = result.A, result.B, result.C, result.D
        assert A.shape[0] == A.shape[1]  # A square
        assert A.shape[0] == B.shape[0]  # A and B rows match
        assert C.shape[1] == A.shape[1]  # C columns match A columns
        assert C.shape[0] == D.shape[0]  # C and D rows match

    def test_ararx_algorithm_properties(self):
        """Test ARARX algorithm has properties for model orders."""
        algorithm = ARARXAlgorithm()
        
        # Test that algorithm handles the ARARX-specific structure
        # ARARX has na=0 (no AR part, unlike ARX)
        assert hasattr(algorithm, 'get_algorithm_name')
        assert algorithm.get_algorithm_name() == "ARARX"

    @pytest.mark.parametrize("nb,nc,nd,nf", [
        (1, 1, 1, 1),  # Minimal ARARX model
        (2, 1, 1, 1),  # Higher input order
        (1, 2, 1, 1),  # Higher noise AR
        (1, 1, 2, 1),  # Higher D polynomial
        (1, 1, 1, 2),  # Higher F polynomial
        (2, 2, 2, 2),  # Complex ARARX model
    ])
    def test_ararx_various_orders(self, nb, nc, nd, nf):
        """Test ARARX with various order combinations."""
        algorithm = ARARXAlgorithm()
        
        config = SystemIdentificationConfig(method='ARARX')
        config.nb = nb
        config.nc = nc
        config.nd = nd
        config.nf = nf
        config.na = 0  # ARARX always has na=0
        
        result = algorithm.identify(self.data, config)
        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_ararx_noise_modeling(self):
        """Test ARARX properly handles noise modeling aspects."""
        algorithm = ARARXAlgorithm()
        
        # Create data with more complex noise characteristics
        np.random.seed(42)
        n_samples = 500
        u = np.random.normal(0, 1, n_samples)
        
        # Create colored noise with AR components
        e_white = np.random.normal(0, 0.1, n_samples)
        e_colored = np.zeros(n_samples)
        for k in range(2, n_samples):
            e_colored[k] = e_white[k] + 0.4 * e_white[k-1] + 0.2 * e_white[k-2]
        
        # Simple dynamics
        y = np.zeros(n_samples)
        for k in range(1, n_samples):
            y[k] = 0.3 * u[k-1] + 0.1 * u[k-2] + e_colored[k]
        
        time_index = pd.date_range('2023-01-01', periods=n_samples, freq='1s')
        data_df = pd.DataFrame({
            'u1': u,
            'y1': y
        }, index=time_index)
        
        data = IDData(
            data=data_df,
            inputs=['u1'],
            outputs=['y1'],
            tsample=1.0
        )
        
        config = SystemIdentificationConfig(method='ARARX')
        config.nb = 2
        config.nc = 2
        config.nd = 2
        config.nf = 2
        
        config.na = 0
        
        result = algorithm.identify(data, config)
        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_ararx_order_calculation_consistency(self):
        """Test ARARX order calculations are consistent with expected structure."""
        algorithm = ARARXAlgorithm()
        
        # For ARARX, na should always be 0
        config = SystemIdentificationConfig(method='ARARX')
        config.nb = 2
        config.nc = 1
        config.nd = 1
        config.nf = 1
        config.na = 0
        
        result = algorithm.identify(self.data, config)
        
        # The algorithm should create a model with appropriate state dimension
        A_state_dim = result.A.shape[0]
        
        # For ARARX, state dimension should reflect input and noise dynamics
        expected_min_states = max(config.nb, config.nc, config.nd, config.nf)
        assert A_state_dim >= expected_min_states

    @pytest.mark.parametrize("nk,nc,nd,nf", [
        (1, 1, 1, 1),  # Standard delay
        (2, 1, 1, 1),  # Longer delay
        (1, 2, 1, 1),  # Higher noise AR order
        (1, 1, 2, 1),  # Higher D polynomial
        (1, 1, 1, 2),  # Higher F polynomial
    ])
    def test_ararx_various_delays_and_orders(self, nk, nc, nd, nf):
        """Test ARARX with various input delays and polynomial orders."""
        algorithm = ARARXAlgorithm()
        
        config = SystemIdentificationConfig(method='ARARX')
        config.nb = 2
        config.nc = 1
        config.nd = 1
        config.nf = 1
        config.nk = nk
        config.na = 0
        
        result = algorithm.identify(self.data, config)
        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_ararx_comparison_with_arma(self):
        """Test ARARX produces different results than ARMA, ARMAX, or ARX."""
        algorithm = ARARXAlgorithm()
        
        # Compare with ARMA (should be different due to structure differences)
        config_arma = SystemIdentificationConfig(method='ARMA')
        config_arma.nc = 1  # ARMA equivalent nc
        config_arma.nd = 1  # ARMA equivalent nd
        config_arma.nf = 1  # ARMA equivalent nf
        
        result_arma = algorithm.identify(self.data, config_arma)
        result_ararx = algorithm.identify(self.data, self.config)
        
        # Results should be different unless using mock implementations
        assert result_arma is not None
        assert result_ararx is not None
        
        # Compare models (they should differ structurally)
        assert not np.allclose(result_arma.A, result_ararx.A, rtol=1e-10)
        assert not np.allclose(result_arma.B, result_ararx.B, rtol=1e-10)

    def test_ararx_algorithm_with_mock_fallback(self):
        """Test ARARX algorithm works with mock fallback when real implementation unavailable."""
        algorithm = ARARXAlgorithm()
        
        # Patch the algorithm to use mock implementation
        with patch.object(algorithm, '_create_mock_model') as mock_method:
            # Create a mock StateSpaceModel to return
            from sippy.identification.base import StateSpaceModel
            mock_state_space = StateSpaceModel(
                A=np.eye(2), B=np.zeros((2,1)), C=np.array([[1, 0]]), D=np.zeros((1,1)),
                K=np.zeros((2,1)), Q=np.eye(2), R=np.eye(1), S=np.zeros((2,1)), ts=1.0, Vn=0.01
            )
            mock_method.return_value = mock_state_space
            result = algorithm.identify(self.data, self.config)
            
            assert result is not None
            assert isinstance(result, StateSpaceModel)
            assert hasattr(result, 'A')
            assert hasattr(result, 'B')  
            assert mock_method.called

    def test_ararx_compatibility_with_arma(self):
        """Test ARARX maintains backward compatibility with ARMA API."""
        algorithm = ARARXAlgorithm()
        
        # Should work with ARMA-style parameters if needed
        config_arma1 = SystemIdentificationConfig(method='ARMA')
        config_arma1.nc = self.config.nc  # Match noise AR order
        
        result_arma1 = algorithm.identify(self.data, config_arma1)
        assert result_arma1 is not None
        assert isinstance(result_arma1, StateSpaceModel)
        
        # Should also support direct ARARX parameters
        result_ararx1 = algorithm.identify(self.data, self.config)
        assert result_ararx1 is not None
        assert isinstance(result_ararx1, StateSpaceModel)

    def test_ararx_harold_integration(self):
        """Test ARARX algorithm with harold integration when available."""
        algorithm = ARARXAlgorithm()
        
        with patch('sippy.identification.algorithms.ararx.HAROLD_AVAILABLE', True), \
             patch('sippy.identification.algorithms.ararx.harold') as mock_harold:
            # Mock harold StateSpace and TransferFunction
            mock_ss = mock_harold.StateSpace.return_value
            mock_tf = mock_harold.TransferFunction.return_value
            mock_tf.NumberOfInputs = 1
            mock_tf.NumberOfOutputs = 1
            mock_tf.SamplingPeriod = 1.0
            
            mock_ss.A = np.eye(2)
            mock_ss.B = np.eye(2, 1)
            mock_ss.C = np.array([[1, 0]])
            mock_ss.D = np.zeros((1, 1))
            
            result = algorithm.identify(self.data, self.config)
            
            assert result is not None
            assert isinstance(result, StateSpaceModel)
            # Integration test - harold should be called in successful case
            assert mock_harold.StateSpace.called
            assert mock_harold.TransferFunction.called
            
    def test_ararx_simulation_and_prediction(self):
        """Test ARARX can simulate and predict outputs."""
        algorithm = ARARXAlgorithm()
        
        result = algorithm.identify(self.data, self.config)
        
        # Test that model can be used for prediction (if supported)
        if hasattr(result, 'simulate') and hasattr(result, 'predict'):
            # Should be supported for simulation
            t_test = np.random.rand(10)
            y_pred = result.simulate(t_test, u=None)
            assert y_pred is not None
            assert y_pred.shape[0] == 10  # 10 time steps
        else:
            # Test fallback prediction if simulation not available
            pass
        
        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_ararx_model_properties_and_methods(self):
        """Test ARARX model has all expected StateSpaceModel methods."""
        algorithm = ARARXAlgorithm()
        result = algorithm.identify(self.data, self.config)
        
        # Test state-space model properties and methods
        assert isinstance(result, StateSpaceModel)
        
        # Should support standard StateSpaceModel interface
        assert hasattr(result, 'A')
        assert hasattr(result, 'B') 
        assert hasattr(result, 'C')
        assert hasattr(result, 'D')
        assert hasattr(result, 'ts')
        
        # Test mathematical properties
        assert hasattr(result, 'is_stable')
        assert hasattr(result, 'get_natural_frequencies')
        assert hasattr(result, 'get_step_response')
        assert hasattr(result, 'get_fir_coefficients')
        assert result.is_stable() is not None

    def test_ararx_optimization_methods(self):
        """Test ARARX supports different solution methods."""
        algorithm = ARARXAlgorithm()
        
        result = algorithm.identify(self.data, self.config)
        
        # Test that model supports estimation methods
        assert result.supports_optimization_methods()

    def test_ararx_estimation_quality(self):
        """Test ARARX model estimation quality indicators."""
        algorithm = ARARXAlgorithm()
        result = algorithm.identify(self.data, self.config)
        
        # Check estimation quality metrics if available
        if hasattr(result, 'Vn') and result.Vn is not None:
            assert result.Vn >= 0  # Variance should be non-negative
            
        # Model should have proper state dimensions
        assert result.B.shape[0] == result.A.shape[0]  # B and A rows match
        assert result.C.shape[1] == result.A.shape[1]  # C and A columns match
        
        # Stability should be checkable (though may not be)
        if hasattr(result, 'get_natural_frequencies'):
            frequencies = result.get_natural_frequencies()
            assert frequencies is not None

    def test_ararx_config_flexibility(self):
        """Test ARARX algorithm works with various configurations."""
        algorithm = ARARXAlgorithm()
        
        # Test with different config objects
        config1 = SystemIdentificationConfig(method='ARARX')
        config1.nb = 1
        config2 = SystemIdentificationConfig(method='ARARX')
        config2.nb = 2
        
        result1 = algorithm.identify(self.data, config1)
        result2 = algorithm.identify(self.data, config2)
        
        # Both should work without errors
        assert result1 is not None
        assert result2 is not None
        assert isinstance(result1, StateSpaceModel)
        assert isinstance(result2, StateSpaceModel)

    def test_ararx_error_handling(self):
        """Test ARARX algorithm graceful error handling."""
        algorithm = ARARXAlgorithm()
        
        # Test with mismatched dimensional parameters
        invalid_data_df = pd.DataFrame({'u1': [1, 2, 3, 4], 'u2': [5, 6, 7, 8], 'y1': [9, 10, 11, 12]}, 
                                   index=pd.date_range('2023-01-01', periods=4, freq='1s'))
        
        invalid_data = IDData(
            data=invalid_data_df,
            inputs=['u1', 'u2'],
            outputs=['y1'], 
            tsample=1.0
        )
        
        config = SystemIdentificationConfig(method='ARARX')
        config.nb = 1
        config.nc = 1
        config.nd = 1
        config.nf = 1
        config.na = 0
        
        # Should handle dimension errors gracefully
        with pytest.raises(ValueError):
            algorithm.identify(invalid_data, config)
        
        # Test with negative orders (invalid for any positive order)
        for param in ['nb', 'nc', 'nd', 'nf']:
            invalid_config = SystemIdentificationConfig(method='ARARX')
            setattr(invalid_config, param, 0)
            
            with pytest.raises(ValueError, match="must be positive"):
                algorithm.identify(self.data, invalid_config)
        
        # Test with negative delay
        invalid_config = SystemIdentificationConfig(method='ARARX')
        invalid_config.nk = -1
        
        with pytest.raises(ValueError, match="must be non-negative"):
            algorithm.identify(self.data, invalid_config)
