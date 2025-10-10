"""
Tests for base classes.
"""
import pytest
import numpy as np
from sippy.identification.base import IdentificationAlgorithm, StateSpaceModel, SystemIdentificationConfig


class TestIdentificationAlgorithm:
    """Test the abstract base class."""
    
    def test_cannot_instantiate_abstract(self):
        """Test that IdentificationAlgorithm cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IdentificationAlgorithm()
    
    def test_concrete_implementation(self):
        """Test that a concrete implementation works."""
        class TestAlgorithm(IdentificationAlgorithm):
            def identify(self, y, u, **kwargs):
                return StateSpaceModel(
                    A=np.eye(2), B=np.zeros((2, 1)), 
                    C=np.zeros((1, 2)), D=np.zeros((1, 1)),
                    K=np.zeros((2, 1)), Q=np.eye(2), 
                    R=1.0, S=np.zeros((2, 1)), ts=1.0, Vn=1.0
                )
            
            def validate_parameters(self, **kwargs):
                return True
        
        algo = TestAlgorithm()
        assert algo.name == "TestAlgorithm"
        
        # Test with dummy data
        y = np.random.randn(2, 100)
        u = np.random.randn(1, 100)
        model = algo.identify(y, u)
        assert isinstance(model, StateSpaceModel)
        assert model.A.shape == (2, 2)


class TestStateSpaceModel:
    """Test the StateSpaceModel class."""
    
    def test_model_creation(self):
        """Test creating a state space model."""
        A = np.array([[0.9, 0.1], [-0.1, 0.8]])
        B = np.array([[1.0], [0.5]])
        C = np.array([[1.0, 0.0]])
        D = np.array([[0.0]])
        K = np.array([[0.1], [0.05]])
        Q = np.eye(2)
        R = 0.1
        S = np.zeros((2, 1))
        ts = 1.0
        Vn = 0.5
        
        model = StateSpaceModel(A, B, C, D, K, Q, R, S, ts, Vn)
        
        assert model.n == 2
        assert np.array_equal(model.A, A)
        assert np.array_equal(model.B, B)
        assert np.array_equal(model.C, C)
        assert np.array_equal(model.D, D)
        assert model.ts == ts
        assert model.Vn == Vn
    
    def test_stability_check(self):
        """Test stability checking."""
        # Stable system
        A_stable = np.array([[0.9, 0.1], [-0.1, 0.8]])
        model_stable = StateSpaceModel(
            A_stable, np.zeros((2, 1)), np.zeros((1, 2)), 
            np.zeros((1, 1)), np.zeros((2, 1)), 
            np.eye(2), 0.1, np.zeros((2, 1)), 1.0, 0.5
        )
        assert model_stable.is_stable()
        
        # Unstable system
        A_unstable = np.array([[1.1, 0.0], [0.0, 1.2]])
        model_unstable = StateSpaceModel(
            A_unstable, np.zeros((2, 1)), np.zeros((1, 2)), 
            np.zeros((1, 1)), np.zeros((2, 1)), 
            np.eye(2), 0.1, np.zeros((2, 1)), 1.0, 0.5
        )
        assert not model_unstable.is_stable()
    
    def test_natural_frequencies(self):
        """Test natural frequency calculation."""
        A = np.array([[0.9, -0.5], [0.5, 0.9]])  # Complex conjugate pair
        model = StateSpaceModel(
            A, np.zeros((2, 1)), np.zeros((1, 2)), 
            np.zeros((1, 1)), np.zeros((2, 1)), 
            np.eye(2), 0.1, np.zeros((2, 1)), 1.0, 0.5
        )
        freqs = model.get_natural_frequencies()
        assert len(freqs) == 2
        assert np.all(freqs >= 0)


class TestSystemIdentificationConfig:
    """Test the configuration class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = SystemIdentificationConfig()
        assert config.method == 'N4SID'
        assert config.centering == 'None'
        assert config.ss_f == 20
        assert config.ss_threshold == 0.1
        assert not config.ss_d_required
        assert not config.ss_a_stability
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SystemIdentificationConfig(
            method='CVA',
            centering='MeanVal',
            ss_f=15,
            ss_threshold=0.05,
            ss_d_required=True,
            ss_a_stability=True
        )
        assert config.method == 'CVA'
        assert config.centering == 'MeanVal'
        assert config.ss_f == 15
        assert config.ss_threshold == 0.05
        assert config.ss_d_required
        assert config.ss_a_stability
