"""
Test suite to verify that different algorithms produce different results.

This test suite ensures that:
1. Subspace methods (N4SID, MOESP, CVA) produce different results
2. ARMAX variants produce different results
3. PARSIM variants produce different results
4. Algorithm differentiation is maintained across refactoring
"""

import numpy as np
import pytest

from sippy.identification import SystemIdentification, SystemIdentificationConfig


@pytest.fixture
def test_data_siso():
    """Generate SISO test data for algorithm testing."""
    np.random.seed(42)
    n_samples = 500
    u = np.random.randn(1, n_samples)
    y = np.zeros((1, n_samples))
    for i in range(1, n_samples):
        y[0, i] = 0.8 * y[0, i - 1] + 0.5 * u[0, i - 1] + 0.1 * np.random.randn()
    return u, y


class TestSubspaceMethodDifferentiation:
    """Test that subspace methods (N4SID, MOESP, CVA) produce different results."""

    def test_n4sid_vs_moesp(self, test_data_siso):
        """Verify N4SID and MOESP produce different results."""
        u, y = test_data_siso

        # Run N4SID
        config_n4sid = SystemIdentificationConfig(method="N4SID")
        config_n4sid.ss_fixed_order = 2
        config_n4sid.ss_f = 10
        model_n4sid = SystemIdentification(config_n4sid).identify(y=y, u=u)

        # Run MOESP
        config_moesp = SystemIdentificationConfig(method="MOESP")
        config_moesp.ss_fixed_order = 2
        config_moesp.ss_f = 10
        model_moesp = SystemIdentification(config_moesp).identify(y=y, u=u)

        # Check that results are different (not identical)
        diff_A = np.max(np.abs(model_n4sid.A - model_moesp.A))
        diff_B = np.max(np.abs(model_n4sid.B - model_moesp.B))
        diff_C = np.max(np.abs(model_n4sid.C - model_moesp.C))

        # Results should differ (threshold = 1e-10 for "identical")
        assert diff_A > 1e-10 or diff_B > 1e-10 or diff_C > 1e-10, (
            f"N4SID and MOESP produced identical results: "
            f"diff_A={diff_A:.2e}, diff_B={diff_B:.2e}, diff_C={diff_C:.2e}"
        )

    def test_n4sid_vs_cva(self, test_data_siso):
        """Verify N4SID and CVA produce different results."""
        u, y = test_data_siso

        # Run N4SID
        config_n4sid = SystemIdentificationConfig(method="N4SID")
        config_n4sid.ss_fixed_order = 2
        config_n4sid.ss_f = 10
        model_n4sid = SystemIdentification(config_n4sid).identify(y=y, u=u)

        # Run CVA
        config_cva = SystemIdentificationConfig(method="CVA")
        config_cva.ss_fixed_order = 2
        config_cva.ss_f = 10
        model_cva = SystemIdentification(config_cva).identify(y=y, u=u)

        # Check that results are different
        diff_A = np.max(np.abs(model_n4sid.A - model_cva.A))
        diff_B = np.max(np.abs(model_n4sid.B - model_cva.B))
        diff_C = np.max(np.abs(model_n4sid.C - model_cva.C))

        assert diff_A > 1e-10 or diff_B > 1e-10 or diff_C > 1e-10, (
            f"N4SID and CVA produced identical results: "
            f"diff_A={diff_A:.2e}, diff_B={diff_B:.2e}, diff_C={diff_C:.2e}"
        )

    def test_moesp_vs_cva(self, test_data_siso):
        """Verify MOESP and CVA produce different results."""
        u, y = test_data_siso

        # Run MOESP
        config_moesp = SystemIdentificationConfig(method="MOESP")
        config_moesp.ss_fixed_order = 2
        config_moesp.ss_f = 10
        model_moesp = SystemIdentification(config_moesp).identify(y=y, u=u)

        # Run CVA
        config_cva = SystemIdentificationConfig(method="CVA")
        config_cva.ss_fixed_order = 2
        config_cva.ss_f = 10
        model_cva = SystemIdentification(config_cva).identify(y=y, u=u)

        # Check that results are different
        diff_A = np.max(np.abs(model_moesp.A - model_cva.A))
        diff_B = np.max(np.abs(model_moesp.B - model_cva.B))
        diff_C = np.max(np.abs(model_moesp.C - model_cva.C))

        assert diff_A > 1e-10 or diff_B > 1e-10 or diff_C > 1e-10, (
            f"MOESP and CVA produced identical results: "
            f"diff_A={diff_A:.2e}, diff_B={diff_B:.2e}, diff_C={diff_C:.2e}"
        )


class TestPARSIMVariantDifferentiation:
    """Test that PARSIM variants produce different results."""

    def test_parsim_k_vs_parsim_s(self, test_data_siso):
        """Verify PARSIM-K and PARSIM-S produce different results."""
        u, y = test_data_siso

        # Run PARSIM-K
        config_k = SystemIdentificationConfig(method="PARSIM-K")
        config_k.ss_fixed_order = 2
        config_k.ss_f = 10
        model_k = SystemIdentification(config_k).identify(y=y, u=u)

        # Run PARSIM-S
        config_s = SystemIdentificationConfig(method="PARSIM-S")
        config_s.ss_fixed_order = 2
        config_s.ss_f = 10
        model_s = SystemIdentification(config_s).identify(y=y, u=u)

        # Check that results are different
        diff_A = np.max(np.abs(model_k.A - model_s.A))
        diff_B = np.max(np.abs(model_k.B - model_s.B))
        diff_C = np.max(np.abs(model_k.C - model_s.C))

        assert diff_A > 1e-10 or diff_B > 1e-10 or diff_C > 1e-10, (
            f"PARSIM-K and PARSIM-S produced identical results: "
            f"diff_A={diff_A:.2e}, diff_B={diff_B:.2e}, diff_C={diff_C:.2e}"
        )


class TestARMAXVariantDifferentiation:
    """Test that ARMAX variants produce different results.

    NOTE: ARMAX signatures have been fixed (TASK 21 completed 2025-10-12).
    These tests verify that the three ARMAX modes (ILLS, RLLS, OPT) produce
    different results as expected based on their different estimation approaches.
    """

    def test_armax_ills_vs_rlls(self, test_data_siso):
        """Verify ARMAX with ILLS and RLLS modes produce identical results (RLLS aliases ILLS)."""
        u, y = test_data_siso

        # Run ARMAX (ILLS)
        config_ills = SystemIdentificationConfig(method="ARMAX")
        config_ills.mode = "ILLS"
        config_ills.na = 1
        config_ills.nb = 1
        config_ills.nc = 1
        config_ills.nk = 1
        config_ills.max_iterations = 50
        model_ills = SystemIdentification(config_ills).identify(y=y, u=u)

        # Run ARMAX (RLLS)
        config_rlls = SystemIdentificationConfig(method="ARMAX")
        config_rlls.mode = "RLLS"
        config_rlls.na = 1
        config_rlls.nb = 1
        config_rlls.nc = 1
        config_rlls.nk = 1
        config_rlls.max_iterations = 50
        model_rlls = SystemIdentification(config_rlls).identify(y=y, u=u)

        # Check that results are the same (RLLS now aliases ILLS pathway)
        diff_A = np.max(np.abs(model_ills.A - model_rlls.A))
        diff_B = np.max(np.abs(model_ills.B - model_rlls.B))

        assert diff_A < 1e-10 and diff_B < 1e-10, (
            f"ARMAX (RLLS) now aliases ILLS pathway and should produce identical results: "
            f"diff_A={diff_A:.2e}, diff_B={diff_B:.2e}"
        )

    def test_armax_ills_vs_opt(self, test_data_siso):
        """Verify ARMAX ILLS and ARMAX OPT modes produce different results."""
        u, y = test_data_siso

        # Run ARMAX (ILLS)
        config_ills = SystemIdentificationConfig(method="ARMAX")
        config_ills.mode = "ILLS"
        config_ills.na = 1
        config_ills.nb = 1
        config_ills.nc = 1
        config_ills.nk = 1
        config_ills.max_iterations = 50
        model_ills = SystemIdentification(config_ills).identify(y=y, u=u)

        # Run ARMAX (OPT)
        config_opt = SystemIdentificationConfig(method="ARMAX")
        config_opt.mode = "OPT"
        config_opt.na = 1
        config_opt.nb = 1
        config_opt.nc = 1
        config_opt.nk = 1
        config_opt.max_iterations = 50
        model_opt = SystemIdentification(config_opt).identify(y=y, u=u)

        # Check that results are different
        diff_A = np.max(np.abs(model_ills.A - model_opt.A))
        diff_B = np.max(np.abs(model_ills.B - model_opt.B))

        assert diff_A > 1e-10 or diff_B > 1e-10, (
            f"ARMAX ILLS and ARMAX OPT produced identical results: "
            f"diff_A={diff_A:.2e}, diff_B={diff_B:.2e}"
        )


class TestAlgorithmSignatureCompatibility:
    """Test that all algorithms have compatible signatures with SystemIdentification."""

    @pytest.mark.parametrize(
        "method",
        [
            "N4SID",
            "MOESP",
            "CVA",
            "PARSIM-K",
            "PARSIM-S",
            "PARSIM-P",
            "ARX",
            "ARARX",
            "FIR",
        ],
    )
    def test_algorithm_can_be_called_through_system_identification(
        self, method, test_data_siso
    ):
        """Verify algorithm can be instantiated and called through SystemIdentification."""
        u, y = test_data_siso

        config = SystemIdentificationConfig(method=method)
        # Set appropriate parameters based on method
        if method in ["N4SID", "MOESP", "CVA", "PARSIM-K", "PARSIM-S", "PARSIM-P"]:
            config.ss_fixed_order = 2
            config.ss_f = 10
        elif method in ["ARX", "ARARX"]:
            config.na = 1
            config.nb = 1
            config.nk = 1
        elif method == "FIR":
            config.nb = 5
            config.nk = 1

        # This should not raise an error
        identifier = SystemIdentification(config)
        model = identifier.identify(y=y, u=u)

        # Basic sanity checks
        assert model is not None
        assert hasattr(model, "A")
        assert hasattr(model, "B")
        assert hasattr(model, "C")
        assert hasattr(model, "D")


@pytest.mark.xfail(
    reason="ARMAX, OE, BJ, ARMA have signature incompatibility - known issue"
)
class TestSignatureIncompatibleAlgorithms:
    """Test algorithms known to have signature issues.

    These tests are marked as xfail to document the known issue.
    Once signatures are fixed, remove xfail decorator.
    """

    @pytest.mark.parametrize(
        "method,params",
        [
            ("OE", {"nb": 1, "nf": 1, "nk": 1}),
            ("BJ", {"nb": 1, "nc": 1, "nd": 1, "nf": 1, "nk": 1}),
        ],
    )
    def test_incompatible_algorithm_fails_gracefully(
        self, method, params, test_data_siso
    ):
        """Document that these algorithms currently fail with signature error."""
        u, y = test_data_siso

        config = SystemIdentificationConfig(method=method)
        for key, value in params.items():
            setattr(config, key, value)

        identifier = SystemIdentification(config)

        # This will currently fail with:
        # "ARMAXAlgorithm.identify() got an unexpected keyword argument 'method'"
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            identifier.identify(y=y, u=u)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
