"""
Tests for IDData class functionality.
"""

import numpy as np
import pandas as pd
import pytest

from ..iddata import IDData


class TestIDData:
    """Test suite for IDData class."""

    def setup_method(self):
        """Set up test data."""
        # Create sample time series data
        time_index = pd.date_range('2020-01-01', periods=100, freq='1min')
        np.random.seed(42)

        self.sample_data = pd.DataFrame({
            'FIC-2001': np.random.randn(100) + 2.0,
            'FIC-2002': np.random.randn(100) + 1.5,
            'TIC-2003': np.random.randn(100) + 0.8,
            'FI-2005': np.random.randn(100) + 3.0,
            'FIC-2101': np.random.randn(100) + 1.2,
            'FIC-2102': np.random.randn(100) + 0.9
        }, index=time_index)

        self.inputs = ['FIC-2001', 'FIC-2002', 'TIC-2003', 'FI-2005']
        self.outputs = ['FIC-2101', 'FIC-2102']

    def test_iddata_creation(self):
        """Test basic IDData object creation."""
        iddata = IDData(self.sample_data, self.inputs, self.outputs)

        assert iddata.n_samples == 100
        assert iddata.n_inputs == 4
        assert iddata.n_outputs == 2
        assert iddata.sample_time == 60.0  # 1 minute
        assert iddata.input_names == self.inputs
        assert iddata.output_names == self.outputs

    def test_get_input_array(self):
        """Test getting input data as numpy array."""
        iddata = IDData(self.sample_data, self.inputs, self.outputs)
        input_array = iddata.get_input_array()

        # Check shape: inputs x time_steps
        assert input_array.shape == (4, 100)
        assert isinstance(input_array, np.ndarray)

        # Check that data matches original
        np.testing.assert_array_equal(
            input_array,
            self.sample_data[self.inputs].to_numpy().T
        )

    def test_get_output_array(self):
        """Test getting output data as numpy array."""
        iddata = IDData(self.sample_data, self.inputs, self.outputs)
        output_array = iddata.get_output_array()

        # Check shape: outputs x time_steps
        assert output_array.shape == (2, 100)
        assert isinstance(output_array, np.ndarray)

        # Check that data matches original
        np.testing.assert_array_equal(
            output_array,
            self.sample_data[self.outputs].to_numpy().T
        )

    def test_remove_mean(self):
        """Test mean removal functionality."""
        iddata = IDData(self.sample_data, self.inputs, self.outputs)
        iddata_centered = iddata.remove_mean()

        # Check that means are close to zero
        input_means = iddata_centered.input_data.mean()
        output_means = iddata_centered.output_data.mean()

        np.testing.assert_allclose(input_means, 0.0, atol=1e-10)
        np.testing.assert_allclose(output_means, 0.0, atol=1e-10)

    def test_split_data(self):
        """Test data splitting functionality."""
        iddata = IDData(self.sample_data, self.inputs, self.outputs)
        train_iddata, test_iddata = iddata.split_data(train_ratio=0.8)

        # Check split sizes
        assert train_iddata.n_samples == 80
        assert test_iddata.n_samples == 20

        # Check that variables are preserved
        assert train_iddata.input_names == iddata.input_names
        assert train_iddata.output_names == iddata.output_names

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Test missing columns
        with pytest.raises(ValueError, match="Missing columns"):
            IDData(
                self.sample_data,
                ['nonexistent_input'],
                self.outputs
            )

        # Test empty data
        empty_data = self.sample_data.iloc[0:0]
        with pytest.raises(ValueError, match="no samples"):
            IDData(empty_data, self.inputs, self.outputs)

        # Test no inputs specified
        with pytest.raises(ValueError, match="No input variables"):
            IDData(self.sample_data, [], self.outputs)

        # Test no outputs specified
        with pytest.raises(ValueError, match="No output variables"):
            IDData(self.sample_data, self.inputs, [])

    def test_custom_sample_time(self):
        """Test custom sample time specification."""
        custom_tsample = 0.5
        iddata = IDData(self.sample_data, self.inputs, self.outputs, tsample=custom_tsample)

        assert iddata.sample_time == custom_tsample

    def test_representations(self):
        """Test string representations."""
        iddata = IDData(self.sample_data, self.inputs, self.outputs)

        # Test __repr__
        repr_str = repr(iddata)
        assert "IDData object" in repr_str
        assert "100 samples" in repr_str

        # Test __str__
        str_str = str(iddata)
        assert "IDData object:" in str_str
        assert "Number of samples: 100" in str_str
        assert "FIC-2001" in str_str
        assert "FIC-2101" in str_str

    def test_non_datetime_index(self):
        """Test IDData with non-datetime index."""
        numeric_data = self.sample_data.reset_index(drop=True)
        iddata = IDData(numeric_data, self.inputs, self.outputs)

        assert iddata.sample_time == 1.0  # Default
        assert isinstance(iddata.time_stamps, np.ndarray)
        assert len(iddata.time_stamps) == 100
