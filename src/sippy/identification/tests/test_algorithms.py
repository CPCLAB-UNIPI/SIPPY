"""
Tests for concrete algorithm implementations.
"""

import numpy as np
import pytest

from sippy.identification.base import StateSpaceModel


class TestN4SIDAlgorithm:
    """Test the N4SID algorithm implementation."""

    def test_n4sid_registration(self):
        """Test that N4SID algorithm is registered."""
        from sippy.identification.factory import AlgorithmFactory

        assert AlgorithmFactory.is_registered("N4SID")

    def test_n4sid_creation(self):
        """Test creating N4SID algorithm."""
        from sippy.identification.algorithms.n4sid import N4SIDAlgorithm

        algo = N4SIDAlgorithm()
        assert algo.name == "N4SIDAlgorithm"

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

    def test_n4sid_identification(self, sample_data):
        """Test N4SID identification produces valid result."""
        from sippy.identification.algorithms.n4sid import N4SIDAlgorithm

        y, u = sample_data
        algo = N4SIDAlgorithm()

        config = {
            "ss_f": 10,
            "ss_threshold": 0.1,
            "ss_fixed_order": 2,
            "ss_d_required": False,
            "ss_a_stability": False,
        }

        model = algo.identify(y, u, **config)

        assert isinstance(model, StateSpaceModel)
        assert model.n <= 2  # Should not exceed specified order
        assert model.A.shape == (model.n, model.n)
        assert model.B.shape == (model.n, u.shape[0])
        assert model.C.shape == (y.shape[0], model.n)
        assert model.D.shape == (y.shape[0], u.shape[0])


class TestMOESPAlgorithm:
    """Test the MOESP algorithm implementation."""

    def test_moesp_registration(self):
        """Test that MOESP algorithm is registered."""
        from sippy.identification.factory import AlgorithmFactory

        assert AlgorithmFactory.is_registered("MOESP")

    def test_moesp_creation(self):
        """Test creating MOESP algorithm."""
        from sippy.identification.algorithms.moesp import MOESPAlgorithm

        algo = MOESPAlgorithm()
        assert algo.name == "MOESPAlgorithm"


class TestCVAAlgorithm:
    """Test the CVA algorithm implementation."""

    def test_cva_registration(self):
        """Test that CVA algorithm is registered."""
        from sippy.identification.factory import AlgorithmFactory

        assert AlgorithmFactory.is_registered("CVA")

    def test_cva_creation(self):
        """Test creating CVA algorithm."""
        from sippy.identification.algorithms.cva import CVAAlgorithm

        algo = CVAAlgorithm()
        assert algo.name == "CVAAlgorithm"
