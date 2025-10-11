"""
Tests for PARSIM algorithm implementations.
"""

import numpy as np
import pytest

from sippy.identification.base import StateSpaceModel


class TestPARSIMKAlgorithm:
    """Test the PARSIM-K algorithm implementation."""

    def test_parsim_k_registration(self):
        """Test that PARSIM-K algorithm is registered."""
        from sippy.identification.factory import AlgorithmFactory

        assert AlgorithmFactory.is_registered("PARSIM-K")

    def test_parsim_k_creation(self):
        """Test creating PARSIM-K algorithm."""
        from sippy.identification.algorithms.parsim_k import PARSIMKAlgorithm

        algo = PARSIMKAlgorithm()
        assert algo.name == "PARSIMKAlgorithm"

    @pytest.fixture
    def sample_data(self):
        """Generate sample test data."""
        np.random.seed(42)
        n_points = 100
        u = np.random.randn(2, n_points)  # 2 inputs
        # Simple linear system response
        y = np.zeros((1, n_points))
        for i in range(1, n_points):
            y[0, i] = (
                0.8 * y[0, i - 1]
                + 0.5 * u[0, i - 1]
                + 0.3 * u[1, i - 1]
                + 0.1 * np.random.randn()
            )
        return y, u

    def test_parsim_k_identification(self, sample_data):
        """Test PARSIM-K identification produces valid result."""
        from sippy.identification.algorithms.parsim_k import PARSIMKAlgorithm

        y, u = sample_data
        algo = PARSIMKAlgorithm()

        config = {
            "ss_f": 10,
            "ss_p": 10,
            "ss_threshold": 0.1,
            "ss_fixed_order": 2,
            "ss_d_required": False,
            "ss_pk_b_reval": False,
        }

        model = algo.identify(y, u, **config)

        assert isinstance(model, StateSpaceModel)
        assert model.A.shape[0] == model.A.shape[1]  # Square A matrix
        assert model.B.shape[0] == model.A.shape[0]  # B rows match A rows
        assert model.C.shape[1] == model.A.shape[0]  # C columns match A columns
        assert model.D.shape[0] == y.shape[0]  # D rows match outputs
        assert model.D.shape[1] == u.shape[0]  # D columns match inputs

    def test_parsim_k_parameter_validation(self):
        """Test PARSIM-K parameter validation."""
        from sippy.identification.algorithms.parsim_k import PARSIMKAlgorithm

        algo = PARSIMKAlgorithm()

        with pytest.raises(ValueError, match="Missing required parameter"):
            algo.validate_parameters()

        with pytest.raises(ValueError, match="ss_f must be a positive number"):
            algo.validate_parameters(ss_f=-5)


class TestPARSIMSAlgorithm:
    """Test the PARSIM-S algorithm implementation."""

    def test_parsim_s_registration(self):
        """Test that PARSIM-S algorithm is registered."""
        from sippy.identification.factory import AlgorithmFactory

        assert AlgorithmFactory.is_registered("PARSIM-S")

    def test_parsim_s_creation(self):
        """Test creating PARSIM-S algorithm."""
        from sippy.identification.algorithms.parsim_s import PARSIMSAlgorithm

        algo = PARSIMSAlgorithm()
        assert algo.name == "PARSIMSAlgorithm"

    @pytest.fixture
    def sample_data(self):
        """Generate sample test data."""
        np.random.seed(42)
        n_points = 100
        u = np.random.randn(2, n_points)  # 2 inputs
        # Simple linear system response
        y = np.zeros((1, n_points))
        for i in range(1, n_points):
            y[0, i] = (
                0.8 * y[0, i - 1]
                + 0.5 * u[0, i - 1]
                + 0.3 * u[1, i - 1]
                + 0.1 * np.random.randn()
            )
        return y, u

    def test_parsim_s_identification(self, sample_data):
        """Test PARSIM-S identification produces valid result."""
        from sippy.identification.algorithms.parsim_s import PARSIMSAlgorithm

        y, u = sample_data
        algo = PARSIMSAlgorithm()

        config = {
            "ss_f": 10,
            "ss_p": 10,
            "ss_threshold": 0.1,
            "ss_fixed_order": 2,
            "ss_d_required": False,
        }

        model = algo.identify(y, u, **config)

        assert isinstance(model, StateSpaceModel)
        assert model.A.shape[0] == model.A.shape[1]  # Square A matrix
        assert model.B.shape[0] == model.A.shape[0]  # B rows match A rows
        assert model.C.shape[1] == model.A.shape[0]  # C columns match A columns
        assert model.D.shape[0] == y.shape[0]  # D rows match outputs
        assert model.D.shape[1] == u.shape[0]  # D columns match inputs


class TestPARSIMPAlgorithm:
    """Test the PARSIM-P algorithm implementation."""

    def test_parsim_p_registration(self):
        """Test that PARSIM-P algorithm is registered."""
        from sippy.identification.factory import AlgorithmFactory

        assert AlgorithmFactory.is_registered("PARSIM-P")

    def test_parsim_p_creation(self):
        """Test creating PARSIM-P algorithm."""
        from sippy.identification.algorithms.parsim_p import PARSIMPAlgorithm

        algo = PARSIMPAlgorithm()
        assert algo.name == "PARSIMPAlgorithm"

    @pytest.fixture
    def sample_data(self):
        """Generate sample test data."""
        np.random.seed(42)
        n_points = 100
        u = np.random.randn(2, n_points)  # 2 inputs
        # Simple linear system response
        y = np.zeros((1, n_points))
        for i in range(1, n_points):
            y[0, i] = (
                0.8 * y[0, i - 1]
                + 0.5 * u[0, i - 1]
                + 0.3 * u[1, i - 1]
                + 0.1 * np.random.randn()
            )
        return y, u

    def test_parsim_p_identification(self, sample_data):
        """Test PARSIM-P identification produces valid result."""
        from sippy.identification.algorithms.parsim_p import PARSIMPAlgorithm

        y, u = sample_data
        algo = PARSIMPAlgorithm()

        config = {
            "ss_f": 10,
            "ss_p": 10,
            "ss_threshold": 0.1,
            "ss_fixed_order": 2,
            "ss_d_required": False,
        }

        model = algo.identify(y, u, **config)

        assert isinstance(model, StateSpaceModel)
        assert model.A.shape[0] == model.A.shape[1]  # Square A matrix
        assert model.B.shape[0] == model.A.shape[0]  # B rows match A rows
        assert model.C.shape[1] == model.A.shape[0]  # C columns match A columns
        assert model.D.shape[0] == y.shape[0]  # D rows match outputs
        assert model.D.shape[1] == u.shape[0]  # D columns match inputs
