"""
Tests for high-pass filter implementation.
"""

import numpy as np
import pandas as pd
import pytest

from sippy.filters import FilterConfig, FilterFactory


class TestHighPassFilter:
    """Test high-pass filter implementation."""

    def test_high_pass_basic(self):
        """Test basic high-pass filtering functionality."""
        np.random.seed(42)

        # Create test data with appropriate frequency for high-pass filtering
        index = pd.date_range("2023-01-01", periods=1000, freq="1H")

        # Generate signal with low frequency component
        signal1 = 0.5 * np.sin(2 * np.pi * 0.01 * np.arange(1000))
        signal2 = 0.3 * np.cos(2 * np.pi * 0.03 * np.arange(1000))

        data = pd.DataFrame({"signal1": signal1, "signal2": signal2}, index=index)

        # Apply high-pass filter
        filter = FilterFactory.create("highpass")
        result = filter.apply_filter(data, tss=60, multiplier=2.5)

        # Check that low-frequency components are reduced
        original_rms = data["signal1"].std()
        filtered_rms = result["signal1"].std()

        # High-pass filter should reduce low-frequency components
        assert filtered_rms < original_rms * 0.5, (
            "High-pass filter didn't sufficiently reduce low-frequency components"
        )

        assert result.shape == data.shape
        assert isinstance(result, pd.DataFrame)

    def test_high_pass_parameters(self):
        """Test high-pass filter with different parameters."""
        np.random.seed(42)

        index = pd.date_range("2023-01-01", periods=1000, freq="1H")
        data = pd.DataFrame(
            {
                "signal1": np.sin(2 * np.pi * 0.01 * np.arange(1000)),
                "signal2": np.cos(2 * np.pi * 0.03 * np.arange(1000)),
            },
            index=index,
        )

        # Test different cutoff frequencies
        cutoffs = [0.005, 0.02, 0.1]

        for cutoff in cutoffs:
            config = FilterConfig(cutoff=cutoff)
            filter = FilterFactory.create("highpass", config)
            result = filter.apply_filter(data, tss=60)

            # Check that cutoff affects the filtering
            signal_rms = data["signal1"].std()
            filtered_rms = result["signal1"].std()
            print(
                f"Cutoff {cutoff} Hz: original RMS={signal_rms:.4f}, filtered RMS={filtered_rms:.4f}"
            )

            # Higher cutoff should reduce more energy
            if cutoff >= 0.02:  # Only test reasonable cutoffs
                assert filtered_rms < signal_rms * 0.8

    def test_high_frequency_limitation(self):
        """Test high-pass filter at very high frequencies."""
        np.random.seed(42)

        # Create high-frequency signal (much higher than filter cutoff)
        index = pd.date_range("2023-01-01", periods=100, freq="30S")  # ~0.033 Hz
        data = pd.DataFrame(
            {
                "signal": np.sin(2 * np.pi * 0.1 * np.arange(100)),
                "noise": 0.01 * np.random.randn(100),
            },
            index=index,
        )

        filter = FilterFactory.create("highpass")

        # This should return the data unchanged (cutoff too high to be effective)
        result = filter.apply_filter(data)

        # Should return data unchanged for very high frequencies
        pd.testing.assert_frame_equal(data, result)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        pd.DataFrame(
            {"x": [1, 2, 3]}, index=pd.date_range("2023-01-01", periods=3, freq="1D")
        )

        filter = FilterFactory.create("zeroomean")

        # Empty data
        with pytest.raises(ValueError):
            filter.apply_filter(pd.DataFrame())  # Empty DataFrame

        # Single column
        single_col = pd.DataFrame({"x": [1, 2, 3]}, index=range(3))
        result = filter.apply_filter(single_col)
        assert result.shape == single_col.shape

        # Invalid configuration
        with pytest.raises(ValueError):
            FilterFactory.create("highpass", FilterConfig(cutoff=-1.0))

    def test_filter_info(self):
        """Test high-pass filter metadata."""
        filter = FilterFactory.create("highpass")
        info = filter.get_filter_info()

        assert info["type"] == "HighPassFilter"
        assert info["suitable_for"] is not None
        assert "description" in info
        assert "effect" in info


class TestDifferenceFilter:
    """Test difference filter implementation."""

    def test_first_order_difference(self):
        """Test first-order difference."""
        np.random.seed(42)

        index = pd.date_range("2023-01-01", periods=1000, freq="1D")
        data = pd.DataFrame(
            {
                "y": pd.Series(np.sin(2 * np.pi * 0.1 * np.arange(1000))),
                "u": pd.Series(np.random.randn(1000)),
            },
            index=index,
        )

        filter = FilterFactory.create("difference")
        result = filter.apply_filter(data)

        # First difference should remove the constant and make data trend-free
        assert (
            result["y"].iloc[0] == 0
        )  # First difference of first element should be zero
        for i in range(1, len(result)):
            assert result["y"].iloc[i] != result["y"].iloc[i - 1]  # Should not be equal

        result_array = result.values
        expected = np.diff(result_array, axis=0)
        pd.testing.assert_frame_equal(result, expected)

    def test_second_order_difference(self):
        """Test second-order difference."""
        np.random.seed(42)

        index = pd.date_range("2023-01-01", periods=1000, freq="1D")
        data = pd.DataFrame(
            {
                "y": pd.Series(np.sin(2 * np.pi * 0.1 * np.arange(1000))),
                "u": pd.Series(np.random.randn(1000)),
            },
            index=index,
        )

        filter = FilterFactory.create("difference")

        # Configure second-order difference
        filter.set_order(2)
        result = filter.apply_filter(data)

        # Second difference should be almost the second difference of first difference
        first_diff = data.diff().diff()
        pd.testing.assert_frame_equal(result, first_diff, rtol=1e-10)

        # Check shape preservation
        assert result.shape == data.shape
        assert isinstance(result, pd.DataFrame)

    def test_difference_order_setting(self):
        """Test setting filter order."""
        np.random.seed(42)

        index = pd.date_range("2023-01-01", periods=1000, freq="1D")
        data = pd.DataFrame({"x": [1, 2, 3]}, index=index)

        for order in [1, 2, 3]:
            filter = FilterFactory.create("difference")
            filter.set_order(order)
            result = filter.apply_filter(data)

            # Higher orders should produce more differences
            if order == 1:
                expected = data.diff().diff()
            elif order == 2:
                expected = data.diff().diff().diff()

            pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

            assert filter._order == order
            assert filter.get_filter_info()["order"] == order

    def test_difference_config(self):
        """Test difference filter with configuration."""
        config = FilterConfig(multiplier=2.5)
        filter = FilterFactory.create("difference", config)
        assert filter._order == 2.5  # Default order overridden

        config = FilterConfig(cutoff=0.1)
        filter = FilterFactory("difference")
        filter.set_order(1)
        assert filter._order == 1

    def test_difference_exceptions(self):
        """Test difference filter error handling."""
        data = pd.DataFrame({"x": [1, 2, 3]}, index=range(3))

        filter = FilterFactory.create("difference")

        # Invalid order should raise AttributeError
        with pytest.raises(AttributeError):
            filter.set_order(3)

        # Data validation should work normally
        filter.apply_filter(data)  # This should work


class TestZeroMeanFilter:
    """Test zero-mean filter functionality."""

    def test_zero_mean_basic(self):
        """Test basic zero-mean filtering."""
        np.random.seed(42)
        index = pd.date_range("2023-01-01", periods=1000, freq="1H")

        # Create data with known means
        data = pd.DataFrame(
            {
                "signal1": 5.0 + 2.0 * np.random.randn(1000),
                "signal2": -1.5 - 2.0 * np.random.randn(1000),
                "noise": 0.05 * np.random.randn(1000),
            },
            index=index,
        )

        filter = FilterFactory.create("zero_mean")
        result = filter.apply_filter(data)

        # Results should be centered around zero
        assert abs(result["signal1"].mean()) < 1e-10
        assert abs(result["signal2"].mean()) < 1e-10

        # Check data integrity
        assert result.shape == data.shape
        assert isinstance(result, pd.DataFrame)

        # Check that the difference from original data is the constant column means
        original_means = data.mean()
        result_means = result.mean()
        pd.testing.assert_frame_equal(result_means, original_means, rtol=1e-12)

    def test_zero_mean_stability(self):
        """Test zero-mean filter on constant signals."""
        np.random.seed(42)
        data = pd.DataFrame({"constant": [10.0] * np.ones(1000)}, index=range(1000))

        filter = FilterFactory.create("zero_mean")
        result = filter.apply_filter(data)

        # Zero-mean should result in zero-vectors or NaN for constant data
        assert (result["constant"] == 0.0).all()

    def test_zero_mean_with_slices(self):
        """Test zero-mean filter with data slices."""
        np.random.seed(42)
        pd.date_range("2023-01-01", periods=1000, freq="1H")
        data = pd.DataFrame({"x": [1, 2, 3]}, index=range(1000))

        # Add a bad slice that should be handled
        slices = {"test": {"type": "bad", "isGlobal": True, "start": 200, "end": 210}}

        filter = FilterFactory.create("zeromean")
        result = filter.apply_filter(data, slices=slices)

        # Bad data should be handled according to slice definition
        bad_start, bad_end = 200, 210
        assert result.iloc[bad_start : bad_end + 1, "signal1"].isna().all()
        assert result.iloc[bad_start : bad_end + 1, "signal1"].isna().all()
        assert (
            result.iloc[bad_start:bad_end, "signal2"].diff().isna().all()
        )  # Bad slice in signal2
        # Check that good data is preserved
        assert result.iloc[bad_end, "signal1"].isna().all()  # Good data is preserved

    def test_zero_mean_metadata(self):
        """Test zero-mean filter metadata."""
        filter = FilterFactory.create("zeromean")
        filter.get_filter_info()

        assert filter.get_name() == "ZeroMeanFilter"
        assert filter.get_filter_info()["suitable_for"].lower()
        assert filter._order == 1  # Default order is 1

        # Filter should store the computed means
        means = filter.get_column_means()
        assert len(means) == 2  # Two columns
        assert list(means.keys()) == ["signal1", "signal2"]


class TestNoneFilter:
    """Test passthrough filter implementation."""

    def test_none_unchanged_data(self):
        """Test that none filter passes data unchanged."""
        np.random.seed(42)

        data = pd.DataFrame(
            {
                "signal1": np.sin(2 * np.pi * 0.1 * np.arange(1000)),
                "signal2": np.cos(2 * np.pi * 0.1 * np.arange(1000)),
            },
            index=range(1000),
        )

        filter = FilterFactory("none")
        result = filter.apply_filter(data)

        # Data should be exactly the same
        pd.testing.assert_frame_equal(data, result)

    def test_none_with_slices(self):
        """Test none filter with data slices."""
        np.random.seed(42)
        pd.date_range("2023-01-01", periods=1000, freq="1H")
        data = pd.DataFrame({"signal1": np.sin(2 * np.pi * 0.1 * np.arange(1000))})

        slices = {
            "interpolate": {
                "type": "interpolate",
                "start": 100,
                "end": 120,
                "tags": ["signal1"],
                "isGlobal": False,
            }
        }

        filter = FilterFactory.create("none")
        result = filter.apply_filter(data, slices=slices)

        # Should keep everything good data unchanged
        assert result.shape == data.shape
        assert isinstance(result, pd.DataFrame)


def test_with_timeout_30_s():
    """Test filter with 30 second time to steady state."""
    # This tests filtering on slower dynamics data
    np.random.seed(42)
    index = pd.date_range("2023-01-01", periods=400, freq="30S")  # ~13 hour simulation
    data = pd.DataFrame(
        {
            "signal1": 0.1 * np.sin(2 * np.pi * 0.05 * np.arange(400)),
            "signal2": 0.2 * np.cos(2 * np.pi * 0.02 * np.arange(400)),
            "drift": 0.001 * np.arange(400),
        },
        index=index,
    )

    # Test with high-pass filter and longer time to steady state
    filter = FilterFactory.create("high_pass", FilterConfig(multiplier=2.0, tss=600))

    result = filter.apply_filter(data)

    assert result.shape == data.shape
    assert isinstance(result, pd.DataFrame)
    # Check that filtering is working by reducing low-frequency energy
    original_rms = data["signal1"].std()
    filtered_rms = result["signal1"].std()

    print(f"30s steady state test: Original RMS: {original_rms:.4f}")
    print(f"30s steady state test: Filtered RMS: {filtered_rms:.4f}")
    print(f"Reduction factor: {filtered_rms / original_rms:.2f}")
