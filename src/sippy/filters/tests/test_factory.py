"""
Tests for FilterFactory class.
"""

import pytest

from sippy.filters import FilterFactory
from sippy.filters.base import FilterConfig, IFilter


class MockFilter(IFilter):
    """Mock filter for testing."""

    def apply_filter(self, data, **kwargs):
        return data.copy()

    def get_name(self):
        return "MockFilter"


class TestFilterFactory:
    """Test cases for FilterFactory."""

    def test_register_filter(self):
        """Test filter registration."""
        # Register a new filter
        FilterFactory.register('mock', MockFilter)

        # Check it was registered
        assert FilterFactory.is_available('mock')
        assert 'mock' in FilterFactory.list_filters()

        # Try to register again (should fail)
        with pytest.raises(ValueError):
            FilterFactory.register('mock', MockFilter)

        # Clean up
        FilterFactory.unregister('mock')

    def test_unregister_filter(self):
        """Test filter unregistration."""
        # Register first
        FilterFactory.register('mock', MockFilter)
        assert FilterFactory.is_available('mock')

        # Unregister
        FilterFactory.unregister('mock')
        assert not FilterFactory.is_available('mock')

        # Test unregistering non-existent filter (should not fail)
        FilterFactory.unregister('nonexistent')

    def test_create_filter_with_config(self):
        """Test creating filter with configuration."""
        config = FilterConfig(multiplier=2.5, slices={'test': {'type': 'bad', 'start': 10, 'end': 20}})

        # Test creating built-in filters
        highpass = FilterFactory.create('highpass', config=config)
        assert isinstance(highpass, MockFilter)  # Will fail until highpass is implemented

        # Test creating with parameters
        with pytest.raises(ValueError):
            FilterFactory.create('highpass', cutoff=0.1)  # Will fail until highpass is implemented

    def test_create_filter_invalid(self):
        """Test creating non-existent filter."""
        with pytest.raises(ValueError):
            FilterFactory.create('nonexistent')
        with pytest.raises(ValueError):
            FilterFactory.create('')  # Empty name

    def test_list_filters(self):
        """Test listing available filters."""
        filters = FilterFactory.list_filters()

        # Should contain built-in filters
        assert isinstance(filters, list)
        assert len(filters) > 0

        # Check some expected filters exist
        expected_filters = ['highpass', 'difference', 'zeromean', 'none']
        for expected in expected_filters:
            assert expected in filters

    def test_is_available(self):
        """Test filter availability checking."""
        # Test existing filters
        assert FilterFactory.is_available('highpass')
        assert FilterFactory.is_available('HIGHPASS')  # Case insensitive

        # Test non-existent filter
        assert not FilterFactory.is_available('nonexistent')

    def test_get_filter_info(self):
        """Test getting filter information."""
        info = FilterFactory.get_filter_info('highpass')

        assert isinstance(info, dict)
        assert 'name' in info
        assert 'class' in info
        assert 'module' in info
        assert 'doc' in info

    def test_get_filter_info_invalid(self):
        """Test getting info for non-existent filter."""
        with pytest.raises(ValueError):
            FilterFactory.get_filter_info('nonexistent')
