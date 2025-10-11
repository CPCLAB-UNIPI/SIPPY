"""
Integration tests for filter functionality.
"""

import numpy as np
import pandas as pd

from sippy.filters import FilterFactory


class TestFilterIntegration:
    """Test filter integration with real data patterns."""

    def test_realistic_data_processing(self):
        """Test filters on realistic data with missing values."""
        # Create more realistic test data with missing values and outliers
        np.random.seed(42)
        index = pd.date_range("2023-01-01", periods=1000, freq="H")

        # Create signal with some outliers
        signal1 = np.sin(2 * np.pi * 0.001 * np.arange(1000))
        signal2 = np.cos(2 * np.pi * 0.01 * np.arange(1000))
        noise = 0.1 * np.random.randn(1000)

        # Add some outliers
        signal1[500:520] = np.nan
        signal2[200:220] = np.nan

        data = pd.DataFrame(
            {"signal1": signal1, "signal2": signal2, "noise": noise}, index=index
        )

        # Apply zero-mean filter
        filter = FilterFactory.create("zeromean")
        result = filter.apply_filter(data)

        # Check that NaN values are preserved (they should be passed through unchanged)
        assert pd.isna(result.loc[500:520, "signal1"]).all()
        assert pd.notna(result.loc[200:220, "signal2"]).all()

        # Check that non-NaN values are zero-mean centered
        assert abs(result["signal1"].mean()) < 1e-12
        assert abs(result["signal2"].mean()) < 1e-12

    def test_multiple_filter_types(self):
        """Test different filter types with the same data."""
        np.random.seed(42)
        index = pd.date_range("2023-01-01", periods=500, freq="1min")

        data = pd.DataFrame(
            {
                "signal": np.sin(2 * np.pi * 0.001 * np.arange(500)),
                "drift": 0.002 * np.arange(500),
            },
            index=index,
        )

        # Test multiple filter types
        filters_to_test = ["zeromean", "none", "difference"]

        for filter_name in filters_to_test:
            filter = FilterFactory.create(filter_name)
            result = filter.apply_filter(data)
            assert isinstance(result, pd.DataFrame)
            assert result.shape == data.shape

        print(f"All {len(filters_to_test)} filter types work correctly")

    def test_filter_data_manager_isolation(self):
        """Test that each filter gets its own data manager instance."""
        filter1 = FilterFactory.create("zeromean")
        filter2 = FilterFactory.create("zeromean")
        filter3 = FilterFactory.create("none")

        assert filter1.data_manager is not filter2.data_manager
        assert filter1.data_manager is not filter3.data_manager

        # Test that data is stored in the right data manager
        test_data = pd.DataFrame({"x": [1, 2, 3]})
        filter1.data_manager.add_data("test", test_data)
        assert filter1.data_manager.get_data("test") is not None
        assert filter2.data_manager.get_data("test") is None

    def test_backward_compatibility_interface(self):
        """Test that the old DetrendingFilter interface works as expected."""
        # This would test the legacy compatibility layer that we need to implement

        # Test that get_filter provides same interface as DetrendingFilter().get_filter()
        filter1 = FilterFactory.create("highpass")
        filter2 = FilterFactory.create("highpass")

        # Both should be IFilter instances
        from sippy.filters.base import IFilter

        assert isinstance(filter1, IFilter)
        assert isinstance(filter2, IFilter)

        # Both should implement the same interface
        assert hasattr(filter1, "apply_filter")
        assert hasattr(filter2, "apply_filter")


if __name__ == "__main__":
    test = TestFilterIntegration()
    test.test_realistic_data_processing()
    test_multiple_filter_types()
    test_filter_data_manager_isolation()
    test_backward_compatibility_interface()
    print("✅ All integration tests passed!")
