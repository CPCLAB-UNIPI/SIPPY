"""
Test suite for Generalized Model (GEN) algorithm implementation.

GEN is the most general input-output model structure that includes all 5 polynomial orders:
A(q) * y(t) = [B(q)/F(q)] * u(t-nk) + [C(q)/D(q)] * e(t)

GEN generalizes all other input-output methods:
- ARX = GEN(na, nb, 0, 0, 0, nk)
- ARMAX = GEN(na, nb, nc, 0, 0, nk)
- ARARX = GEN(na, nb, 0, 0, nf, nk)
- ARARMAX = GEN(na, nb, nc, nd, 0, nk)
- OE = GEN(0, nb, 0, 0, nf, nk)
- BJ = GEN(0, nb, nc, nd, nf, nk)
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sippy.identification.algorithms.gen import GENAlgorithm
from sippy.identification.base import StateSpaceModel
from sippy.identification.iddata import IDData


class TestGENAlgorithm:
    """Test cases for Generalized Model (GEN) algorithm following TDD approach."""

    def setup_method(self):
        """Set up test data for GEN algorithm."""
        np.random.seed(42)
        self.n_samples = 1000

        # Create test data for GEN (SISO system with full structure)
        t = np.linspace(0, 100, self.n_samples)
        u = np.random.normal(0, 1, self.n_samples)  # Input signal

        # Generate GEN process: A(q)*y[k] = [B(q)/F(q)]*u[k] + [C(q)/D(q)]*e[k]
        # GEN(1,2,1,1,1) model as example
        na = 1  # AR order for output
        nb = 2  # Input numerator order
        nc = 1  # Noise numerator order
        nd = 1  # Noise denominator order
        nf = 1  # Input denominator order
        nk = 1  # Input delay

        # Simulate GEN process
        y = np.zeros(self.n_samples)
        e = np.random.normal(0, 0.1, self.n_samples)

        for k in range(3, self.n_samples):
            # AR part: A(q)*y[k]
            ar_part = y[k - 1] * 0.5  # A(q) coefficient

            # Input part: B(q)/F(q) * u[k-nk]
            if k >= nb + nk:
                input_part = 0.3 * u[k - nk] + 0.2 * u[k - nk - 1]  # B(q)
                # Divide by F(q) (simplified as feedback)
                if k >= nf:
                    input_part -= 0.1 * y[k - 1]  # F(q) feedback
            else:
                input_part = 0

            # Noise part: C(q)/D(q) * e[k]
            if k >= 1:
                noise_part = e[k] + 0.3 * e[k - 1]  # C(q)
                # D(q) would be in denominator, simplified here
                if k >= 1:
                    noise_part -= 0.2 * (e[k - 1] + 0.3 * e[k - 2] if k >= 2 else 0)  # D(q)
            else:
                noise_part = e[k]

            y[k] = ar_part + input_part + noise_part

        # Create IDData
        time_index = pd.date_range("2023-01-01", periods=self.n_samples, freq="1s")
        data_df = pd.DataFrame({"u1": u, "y1": y}, index=time_index)

        self.data = IDData(data=data_df, inputs=["u1"], outputs=["y1"], tsample=1.0)

    def test_gen_algorithm_initialization(self):
        """Test GEN algorithm can be initialized."""
        algorithm = GENAlgorithm()
        assert algorithm.get_algorithm_name() == "GEN"

    def test_gen_algorithm_name(self):
        """Test GEN algorithm name."""
        algorithm = GENAlgorithm()
        assert algorithm.get_algorithm_name() == "GEN"

    def test_gen_basic_identification(self):
        """Test GEN basic identification with full structure."""
        algorithm = GENAlgorithm()

        # GEN(1,2,1,1,1,1) - full structure
        result = algorithm.identify(
            y=None,
            u=None,
            iddata=self.data,
            na=1,
            nb=2,
            nc=1,
            nd=1,
            nf=1,
            nk=1
        )

        assert result is not None
        assert isinstance(result, StateSpaceModel)
        assert result.A is not None
        assert result.B is not None
        assert result.C is not None
        assert result.D is not None

    def test_gen_reduces_to_arx(self):
        """Test that GEN(na, nb, 0, 0, 0, nk) reduces to ARX."""
        algorithm = GENAlgorithm()

        # GEN with nc=nd=nf=0 should behave like ARX
        result = algorithm.identify(
            y=None,
            u=None,
            iddata=self.data,
            na=2,
            nb=2,
            nc=0,
            nd=0,
            nf=0,
            nk=1
        )

        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_gen_reduces_to_armax(self):
        """Test that GEN(na, nb, nc, 0, 0, nk) reduces to ARMAX."""
        algorithm = GENAlgorithm()

        # GEN with nd=nf=0 should behave like ARMAX
        result = algorithm.identify(
            y=None,
            u=None,
            iddata=self.data,
            na=2,
            nb=2,
            nc=1,
            nd=0,
            nf=0,
            nk=1
        )

        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_gen_reduces_to_ararx(self):
        """Test that GEN(na, nb, 0, 0, nf, nk) reduces to ARARX."""
        algorithm = GENAlgorithm()

        # GEN with nc=nd=0 should behave like ARARX
        result = algorithm.identify(
            y=None,
            u=None,
            iddata=self.data,
            na=2,
            nb=2,
            nc=0,
            nd=0,
            nf=1,
            nk=1
        )

        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_gen_reduces_to_ararmax(self):
        """Test that GEN(na, nb, nc, nd, 0, nk) reduces to ARARMAX."""
        algorithm = GENAlgorithm()

        # GEN with nf=0 should behave like ARARMAX
        result = algorithm.identify(
            y=None,
            u=None,
            iddata=self.data,
            na=2,
            nb=2,
            nc=1,
            nd=1,
            nf=0,
            nk=1
        )

        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_gen_reduces_to_oe(self):
        """Test that GEN(0, nb, 0, 0, nf, nk) reduces to OE."""
        algorithm = GENAlgorithm()

        # GEN with na=nc=nd=0 should behave like OE
        result = algorithm.identify(
            y=None,
            u=None,
            iddata=self.data,
            na=0,
            nb=2,
            nc=0,
            nd=0,
            nf=2,
            nk=1
        )

        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_gen_reduces_to_bj(self):
        """Test that GEN(0, nb, nc, nd, nf, nk) reduces to BJ."""
        algorithm = GENAlgorithm()

        # GEN with na=0 should behave like BJ
        result = algorithm.identify(
            y=None,
            u=None,
            iddata=self.data,
            na=0,
            nb=2,
            nc=1,
            nd=1,
            nf=2,
            nk=1
        )

        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_gen_parameter_validation(self):
        """Test GEN parameter validation."""
        algorithm = GENAlgorithm()

        # Test with invalid nb (must be positive)
        with pytest.raises(ValueError, match="Input order .* must be positive"):
            algorithm.identify(
                y=None,
                u=None,
                iddata=self.data,
                na=1,
                nb=0,  # Invalid
                nc=1,
                nd=1,
                nf=1,
                nk=1
            )

        # Test with negative orders
        with pytest.raises(ValueError, match="must be non-negative"):
            algorithm.identify(
                y=None,
                u=None,
                iddata=self.data,
                na=-1,  # Invalid
                nb=1,
                nc=1,
                nd=1,
                nf=1,
                nk=1
            )

    def test_gen_with_different_orders(self):
        """Test GEN with various order combinations."""
        algorithm = GENAlgorithm()

        # Test several order combinations
        test_cases = [
            (1, 1, 1, 1, 1),  # Minimal full GEN
            (2, 2, 1, 1, 1),  # Higher AR and input
            (1, 2, 2, 1, 1),  # Higher noise AR
            (1, 2, 1, 2, 1),  # Higher noise MA
            (1, 2, 1, 1, 2),  # Higher input denominator
            (2, 3, 2, 2, 2),  # Complex GEN model
        ]

        for na, nb, nc, nd, nf in test_cases:
            result = algorithm.identify(
                y=None,
                u=None,
                iddata=self.data,
                na=na,
                nb=nb,
                nc=nc,
                nd=nd,
                nf=nf,
                nk=1
            )
            assert result is not None
            assert isinstance(result, StateSpaceModel)

    def test_gen_without_harold(self):
        """Test GEN algorithm graceful degradation without harold."""
        algorithm = GENAlgorithm()

        with patch("sippy.identification.algorithms.gen.HAROLD_AVAILABLE", False):
            result = algorithm.identify(
                y=None,
                u=None,
                iddata=self.data,
                na=1,
                nb=2,
                nc=1,
                nd=1,
                nf=1,
                nk=1
            )
            assert result is not None
            assert isinstance(result, StateSpaceModel)

    def test_gen_insufficient_data(self):
        """Test GEN algorithm with insufficient data."""
        algorithm = GENAlgorithm()

        # Create very short dataset
        n_samples = 10
        np.random.seed(42)
        y = np.random.randn(n_samples)
        u = np.random.randn(n_samples)

        time_index = pd.date_range("2023-01-01", periods=n_samples, freq="1s")
        data_df = pd.DataFrame({"y1": y, "u1": u}, index=time_index)

        short_data = IDData(data=data_df, inputs=["u1"], outputs=["y1"], tsample=1.0)

        # Orders that require more data than available
        with pytest.raises(ValueError, match="Insufficient data|Not enough data"):
            algorithm.identify(
                y=None,
                u=None,
                iddata=short_data,
                na=3,
                nb=3,
                nc=3,
                nd=3,
                nf=3,
                nk=1
            )

    def test_gen_state_space_models(self):
        """Test GEN creates valid state-space models."""
        algorithm = GENAlgorithm()

        result = algorithm.identify(
            y=None,
            u=None,
            iddata=self.data,
            na=1,
            nb=2,
            nc=1,
            nd=1,
            nf=1,
            nk=1
        )

        # Check state-space model properties
        assert hasattr(result, "A")
        assert hasattr(result, "B")
        assert hasattr(result, "C")
        assert hasattr(result, "D")

        # Check dimensions are consistent
        A, B, C, D = result.A, result.B, result.C, result.D
        assert A.shape[0] == A.shape[1]  # A square
        assert A.shape[0] == B.shape[0]  # A and B rows match
        assert C.shape[1] == A.shape[1]  # C columns match A columns
        assert C.shape[0] == D.shape[0]  # C and D rows match

    def test_gen_transfer_functions(self):
        """Test GEN creates transfer functions G_tf and H_tf."""
        algorithm = GENAlgorithm()

        result = algorithm.identify(
            y=None,
            u=None,
            iddata=self.data,
            na=1,
            nb=2,
            nc=1,
            nd=1,
            nf=1,
            nk=1
        )

        # Check that transfer functions are created (when harold available)
        assert hasattr(result, "G_tf")
        assert hasattr(result, "H_tf")

    def test_gen_yid_computation(self):
        """Test GEN computes one-step-ahead predictions (Yid)."""
        algorithm = GENAlgorithm()

        result = algorithm.identify(
            y=None,
            u=None,
            iddata=self.data,
            na=1,
            nb=2,
            nc=1,
            nd=1,
            nf=1,
            nk=1
        )

        # Check that Yid is computed
        assert hasattr(result, "Yid")
        if result.Yid is not None:
            # Yid should have same shape as output data
            assert result.Yid.shape[1] == self.n_samples

    @pytest.mark.parametrize(
        "na,nb,nc,nd,nf",
        [
            (1, 1, 0, 0, 0),  # ARX-like
            (1, 1, 1, 0, 0),  # ARMAX-like
            (0, 1, 0, 0, 1),  # OE-like
            (0, 1, 1, 1, 1),  # BJ-like
            (1, 1, 1, 1, 1),  # Full GEN
            (2, 2, 2, 2, 2),  # Higher order GEN
        ],
    )
    def test_gen_various_structures(self, na, nb, nc, nd, nf):
        """Test GEN with various structure combinations."""
        algorithm = GENAlgorithm()

        result = algorithm.identify(
            y=None,
            u=None,
            iddata=self.data,
            na=na,
            nb=nb,
            nc=nc,
            nd=nd,
            nf=nf,
            nk=1
        )
        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_gen_noise_modeling(self):
        """Test GEN properly handles complex noise modeling."""
        algorithm = GENAlgorithm()

        # Create data with complex noise characteristics
        np.random.seed(42)
        n_samples = 500
        u = np.random.normal(0, 1, n_samples)

        # Create colored noise with C(q)/D(q) structure
        e_white = np.random.normal(0, 0.1, n_samples)
        e_colored = np.zeros(n_samples)
        for k in range(2, n_samples):
            e_colored[k] = e_white[k] + 0.4 * e_white[k - 1]  # C(q)
            e_colored[k] /= (1 + 0.3)  # Approximate D(q)

        # Dynamics with AR and input terms
        y = np.zeros(n_samples)
        for k in range(2, n_samples):
            y[k] = 0.5 * y[k - 1] + 0.3 * u[k - 1] + 0.2 * u[k - 2] + e_colored[k]

        time_index = pd.date_range("2023-01-01", periods=n_samples, freq="1s")
        data_df = pd.DataFrame({"u1": u, "y1": y}, index=time_index)

        data = IDData(data=data_df, inputs=["u1"], outputs=["y1"], tsample=1.0)

        result = algorithm.identify(
            y=None,
            u=None,
            iddata=data,
            na=1,
            nb=2,
            nc=1,
            nd=1,
            nf=1,
            nk=1
        )
        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_gen_with_delay(self):
        """Test GEN with different input delays."""
        algorithm = GENAlgorithm()

        # Test with various delays
        for nk in [0, 1, 2, 3]:
            result = algorithm.identify(
                y=None,
                u=None,
                iddata=self.data,
                na=1,
                nb=2,
                nc=1,
                nd=1,
                nf=1,
                nk=nk
            )
            assert result is not None
            assert isinstance(result, StateSpaceModel)

    def test_gen_modern_api(self):
        """Test GEN with modern API (y, u, **kwargs)."""
        algorithm = GENAlgorithm()

        u = self.data.get_input_array()
        y = self.data.get_output_array()

        # Test modern API with numpy arrays
        result = algorithm.identify(
            y=y,
            u=u,
            na=1,
            nb=2,
            nc=1,
            nd=1,
            nf=1,
            nk=1,
            tsample=1.0
        )

        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_gen_order_calculation_consistency(self):
        """Test GEN order calculations are consistent with expected structure."""
        algorithm = GENAlgorithm()

        # For GEN, state dimension should reflect all dynamics
        result = algorithm.identify(
            y=None,
            u=None,
            iddata=self.data,
            na=2,
            nb=2,
            nc=1,
            nd=1,
            nf=2,
            nk=1
        )

        # The algorithm should create a model with appropriate state dimension
        # based on the sum of relevant polynomial orders
        A_state_dim = result.A.shape[0]

        # At minimum, should have states for all dynamics
        expected_min_states = max(2, 2, 1, 1, 2)  # max of all orders
        assert A_state_dim >= expected_min_states

    def test_gen_casadi_nlp_method(self):
        """Test GEN NLP method when CasADi is available."""
        algorithm = GENAlgorithm()

        # This test will use NLP method if CasADi is available
        result = algorithm.identify(
            y=None,
            u=None,
            iddata=self.data,
            na=1,
            nb=1,
            nc=1,
            nd=1,
            nf=1,
            nk=1,
            max_iterations=100  # Parameter for NLP
        )

        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_gen_ills_fallback(self):
        """Test GEN falls back to ILLS when CasADi unavailable."""
        algorithm = GENAlgorithm()

        with patch("sippy.identification.algorithms.gen.CASADI_AVAILABLE", False):
            result = algorithm.identify(
                y=None,
                u=None,
                iddata=self.data,
                na=1,
                nb=2,
                nc=1,
                nd=1,
                nf=1,
                nk=1
            )
            assert result is not None
            assert isinstance(result, StateSpaceModel)
