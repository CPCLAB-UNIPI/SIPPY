"""
Tests for the factory pattern implementation.
"""
import pytest

from sippy.identification.base import IdentificationAlgorithm
from sippy.identification.factory import AlgorithmFactory, create_algorithm


class MockAlgorithm(IdentificationAlgorithm):
    """Mock algorithm for testing."""

    def identify(self, y, u, **kwargs):
        return None  # Mock implementation

    def validate_parameters(self, **kwargs):
        return True


class TestAlgorithmFactory:
    """Test the algorithm factory."""

    def setup_method(self):
        """Set up test fixtures."""
        # Save existing registrations to restore later
        self._original_algorithms = AlgorithmFactory._algorithms.copy()
        AlgorithmFactory._algorithms.clear()

    def teardown_method(self):
        """Clean up after tests."""
        # Restore original algorithms
        AlgorithmFactory._algorithms.clear()
        AlgorithmFactory._algorithms.update(self._original_algorithms)

    def test_register_algorithm(self):
        """Test registering an algorithm."""
        AlgorithmFactory.register('N4SID', MockAlgorithm)
        assert 'N4SID' in AlgorithmFactory._algorithms
        assert AlgorithmFactory.is_registered('N4SID')
        assert not AlgorithmFactory.is_registered('MOESP')

    def test_register_case_insensitive(self):
        """Test that registration is case insensitive."""
        AlgorithmFactory.register('n4sid', MockAlgorithm)
        assert AlgorithmFactory.is_registered('N4SID')
        assert AlgorithmFactory.is_registered('n4sid')

    def test_create_algorithm(self):
        """Test creating an algorithm instance."""
        AlgorithmFactory.register('N4SID', MockAlgorithm)
        algo = AlgorithmFactory.create('N4SID')
        assert isinstance(algo, MockAlgorithm)
        assert algo.name == 'MockAlgorithm'

    def test_create_case_insensitive(self):
        """Test that creation is case insensitive."""
        AlgorithmFactory.register('N4SID', MockAlgorithm)
        algo = AlgorithmFactory.create('n4sid')
        assert isinstance(algo, MockAlgorithm)

    def test_create_unknown_algorithm(self):
        """Test creating an unknown algorithm raises error."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            AlgorithmFactory.create('UNKNOWN')

    def test_list_algorithms(self):
        """Test listing registered algorithms."""
        AlgorithmFactory.register('N4SID', MockAlgorithm)
        AlgorithmFactory.register('MOESP', MockAlgorithm)
        algorithms = AlgorithmFactory.list_algorithms()
        assert 'N4SID' in algorithms
        assert 'MOESP' in algorithms
        assert len(algorithms) == 2


class TestCreateAlgorithm:
    """Test the convenience create function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Save existing registrations to restore later
        self._original_algorithms = AlgorithmFactory._algorithms.copy()
        AlgorithmFactory._algorithms.clear()

    def teardown_method(self):
        """Clean up after tests."""
        # Restore original algorithms
        AlgorithmFactory._algorithms.clear()
        AlgorithmFactory._algorithms.update(self._original_algorithms)

    def test_create_algorithm_function(self):
        """Test the convenience create function."""
        AlgorithmFactory.register('N4SID', MockAlgorithm)
        algo = create_algorithm('N4SID')
        assert isinstance(algo, MockAlgorithm)
