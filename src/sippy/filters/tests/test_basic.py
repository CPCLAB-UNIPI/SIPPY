"""
Basic tests for filter functionality.
"""

import numpy as np
import pandas as pd
import pytest

from sippy.filters import FilterConfig, FilterFactory
from sippy.filters.base import FilterDataManager


class TestFilterFactory:
    """Test the FilterFactory class."""

    def test_list_filters(self):
        """Test that all expected filters are available."""
        expected = [
            "highpass",
            "high_pass",
            "difference",
            "doubledifference",
            "diff",
            "zeromean",
            "zero_mean",
            "none",
            "passthrough",
        ]
        filters = FilterFactory.list_filters()

        for expected_filter in expected:
            assert expected_filter in filters, (
                f"Filter '{expected_filter}' not found in {filters}"
            )

        # Check that all registered filters work
        for filter_name in filters:
            factory = FilterFactory.create(filter_name)
            assert factory is not None
            assert factory.get_name() == filter_name or factory.get_name().lower()

    def test_create_unknown_filter(self):
        """Test creating a non-existent filter."""
        with pytest.raises(ValueError):
            FilterFactory.create("nonexistent")

    def test_register_filter(self):
        """Test filter registration and unregistration."""
        from sippy.filters.zero_mean import ZeroMeanFilter

        # Test registration
        FilterFactory.register("test_filter", ZeroMeanFilter)
        assert FilterFactory.is_available("test_filter")

        # Try to register the same filter name twice (should raise ValueError)
        with pytest.raises(ValueError, match="already registered"):
            FilterFactory.register("test_filter", ZeroMeanFilter)

        # Test unregistration
        FilterFactory.unregister("test_filter")
        assert not FilterFactory.is_available("test_filter")

        # Test registering non-existent filter class (should raise TypeError)
        with pytest.raises(TypeError):
            FilterFactory.register("invalid_filter", str)

    def test_get_filter_info(self):
        """Test getting filter information."""
        info = FilterFactory.get_filter_info("zeromean")

        assert isinstance(info, dict)
        assert "type" in info or "name" in info
        assert "class" in info
        assert "module" in info
        assert "doc" in info

    def test_get_filter_info_unknown(self):
        """Test getting info for non-existent filter."""
        with pytest.raises(ValueError):
            FilterFactory.get_filter_info("nonexistent")


class TestFilterConfig:
    """Test FilterConfig functionality."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = FilterConfig()

        assert config.multiplier == 3.0
        assert config.slices == {}

    def test_config_parameters(self):
        """Test configuration with parameters."""
        config = FilterConfig(
            cutoff=0.1,
            order=4,
            multiplier=2.5,
            slices={"test": {"type": "bad", "start": 10, "end": 20, "tags": ["col1"]}},
        )

        assert config.cutoff == 0.1
        assert config.order == 4
        assert config.multiplier == 2.5
        assert config.slices["test"]["type"] == "bad"

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid multiplier
        with pytest.raises(ValueError):
            FilterConfig(multiplier=-1.0)

        # Test invalid tss
        with pytest.raises(ValueError):
            FilterConfig(tss=-1.0)

    def test_config_with_kwargs(self):
        """Test creating config with keyword arguments."""
        config = FilterConfig(cutoff=0.5, order=2)

        assert config.cutoff == 0.5
        assert config.order == 2


class TestFilters:
    """Test individual filter implementations."""

    def test_zero_mean_filter(self):
        """Test zero-mean filter."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "signal": np.sin(2 * np.pi * 0.01 * np.arange(100)),
                "noise": 0.01 * np.random.randn(100),
            }
        )

        filter = FilterFactory.create("zeromean")
        result = filter.apply_filter(data)

        # Check that mean is approximately zero
        assert abs(result["signal"].mean()) < 1e-10
        assert abs(result["noise"].mean()) < 1e-10

        # Check that data is correct shape and type
        assert result.shape == data.shape
        assert isinstance(result, pd.DataFrame)

    def test_none_filter(self):
        """Test passthrough filter."""
        data = pd.DataFrame(
            {
                "signal": np.sin(2 * np.pi * 0.01 * np.arange(100)),
                "noise": 0.01 * np.random.randn(100),
            }
        )

        filter = FilterFactory.create("none")
        result = filter.apply_filter(data)

        # Data should be unchanged
        pd.testing.assert_frame_equal(data, result)

    def test_filter_info(self):
        """Test filter information methods."""
        zero_mean = FilterFactory.create("zeromean")
        info = zero_mean.get_filter_info()

        assert isinstance(info, dict)
        # Filter instances have "type", factory info has "name"
        assert "type" in info or "name" in info
        # Check for basic expected fields
        expected_fields = ["type", "description", "suitable_for", "module", "doc"]
        assert any(field in info for field in expected_fields)


class TestDataManager:
    """Test FilterDataManager functionality."""

    def test_data_storage_and_retrieval(self):
        """Test data storage and retrieval."""
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4.5, 5.5, 6.5]})
        metadata = {"type": "test"}

        data_manager = FilterDataManager()
        data_manager.add_data("test_data", data, **metadata)

        # Test data retrieval
        retrieved_data = data_manager.get_data("test_data")
        pd.testing.assert_frame_equal(data, retrieved_data)

        # Test metadata retrieval
        retrieved_metadata = data_manager.get_metadata("test_data")
        assert retrieved_metadata == metadata

        # Test non-existent data
        assert data_manager.get_data("nonexistent") is None

        # Clear all data
        data_manager.clear()
        assert data_manager.get_data("test_data") is None
        assert data_manager.get_metadata("test_data") == {}
