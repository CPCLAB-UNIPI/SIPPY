"""
IDData class for system identification data management.

This module provides an IDData class that encapsulates input-output measurement data
similar to Matlab's iddata structure, accepting pandas dataframes and providing
numpy arrays for internal processing by identification algorithms.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class IDData:
    """
    IDData class for managing input-output system identification data.

    This class encapsulates time-domain data for system identification, similar to
    MATLAB's iddata objects. It accepts pandas DataFrames and provides methods to
    extract data in the format required by identification algorithms.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing input and output data with time index
    inputs : List[str]
        List of column names representing input variables
    outputs : List[str]
        List of column names representing output variables
    tsample : float, optional
        Sample time in seconds. If None, inferred from DataFrame index
    time_index : str, optional
        Name of time column if DataFrame doesn't have datetime index

    Attributes:
    -----------
    input_data : pd.DataFrame
        Input data subset
    output_data : pd.DataFrame
        Output data subset
    sample_time : float
        Sample time in seconds
    input_names : List[str]
        Input variable names
    output_names : List[str]
        Output variable names
    n_samples : int
        Number of data points
    n_inputs : int
        Number of input variables
    n_outputs : int
        Number of output variables
    time_stamps : pd.DatetimeIndex or np.ndarray
        Time stamps for each data point
    """

    def __init__(
        self,
        data: pd.DataFrame,
        inputs: List[str],
        outputs: List[str],
        tsample: Optional[float] = None,
        time_index: Optional[str] = None,
        slices: Optional[Dict[str, Any]] = None,
        bad_strategy: str = "ffill",
        interpolate_method: str = "linear",
        store_mask: bool = True,
    ):
        """
        Initialize IDData object.

        Args:
            data: DataFrame containing input and output data
            inputs: List of input variable column names
            outputs: List of output variable column names
            tsample: Sample time in seconds (auto-detected if None)
            time_index: Name of time column for non-datetime indexed DataFrames
        """
        # Validate input DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")

        # Ensure all requested columns exist
        missing_cols = set(inputs + outputs) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

        # Optionally apply slice processing on a combined view to keep alignment
        combined = data[inputs + outputs].copy()
        self.bad_mask = None
        if slices:
            try:
                from sippy.utils.slices import (
                    process_slices,  # Local import to avoid cycles
                )

                combined, mask = process_slices(
                    combined,
                    slices,
                    bad_strategy=bad_strategy,
                    interpolate_method=interpolate_method,
                )
                if store_mask:
                    self.bad_mask = mask
            except Exception as e:
                raise ValueError(f"Failed to process slices: {e}")

        # Store input and output data
        self.input_data = combined[inputs]
        self.output_data = combined[outputs]
        self.input_names = inputs
        self.output_names = outputs

        # Handle time information
        if tsample is not None:
            self.sample_time = float(tsample)
        else:
            # Infer sample time from index
            self.sample_time = self._infer_sample_time(data)

        # Store time stamps
        self.time_stamps = self._get_time_stamps(data, time_index)

        # Data dimensions
        self.n_samples = len(data)
        self.n_inputs = len(inputs)
        self.n_outputs = len(outputs)

        # Validate data consistency
        self._validate_data()

    def _infer_sample_time(self, data: pd.DataFrame) -> float:
        """
        Infer sample time from DataFrame index.

        Returns:
        --------
        float
            Sample time in seconds
        """
        if pd.api.types.is_datetime64_any_dtype(data.index):
            # For datetime index, calculate median difference
            time_diffs = np.diff(data.index)
            if len(time_diffs) > 0:
                median_diff = np.median(time_diffs)
                # Convert to seconds - handle both pandas and numpy versions
                if hasattr(median_diff, "total_seconds"):
                    return median_diff.total_seconds()
                else:
                    # For numpy.timedelta64
                    return median_diff.astype("timedelta64[s]").astype(float)

        # Fallback: assume unit sample time
        return 1.0

    def _get_time_stamps(
        self, data: pd.DataFrame, time_index: Optional[str]
    ) -> Union[pd.DatetimeIndex, np.ndarray]:
        """
        Get time stamps from DataFrame.

        Args:
            data: Input DataFrame
            time_index: Name of time column

        Returns:
        --------
        Union[pd.DatetimeIndex, np.ndarray]
            Time stamps
        """
        if time_index is not None and time_index in data.columns:
            # Use specified time column
            return data[time_index].values
        elif pd.api.types.is_datetime64_any_dtype(data.index):
            # Use datetime index
            return data.index
        else:
            # Use numeric index as time points
            return np.arange(len(data))

    def _validate_data(self):
        """Validate data consistency."""
        if self.n_samples == 0:
            raise ValueError("Data contains no samples")

        if self.n_inputs == 0:
            raise ValueError("No input variables specified")

        if self.n_outputs == 0:
            raise ValueError("No output variables specified")

        # Check for NaN values
        if self.input_data.isnull().any().any():
            print("Warning: Input data contains NaN values")
        if self.output_data.isnull().any().any():
            print("Warning: Output data contains NaN values")

    def get_input_array(self) -> np.ndarray:
        """
        Get input data as numpy array in required format.

        Returns:
        --------
        np.ndarray
            Input array with shape (n_inputs, n_samples)
        """
        return self.input_data.to_numpy().T

    def get_output_array(self) -> np.ndarray:
        """
        Get output data as numpy array in required format.

        Returns:
        --------
        np.ndarray
            Output array with shape (n_outputs, n_samples)
        """
        return self.output_data.to_numpy().T

    def get_time_stamps_array(self) -> np.ndarray:
        """
        Get time stamps as numpy array.

        Returns:
        --------
        np.ndarray
            Time stamps
        """
        if isinstance(self.time_stamps, pd.DatetimeIndex):
            return self.time_stamps.astype(np.int64) / 1e9  # Convert to seconds
        else:
            return np.array(self.time_stamps)

    def split_data(self, train_ratio: float = 0.8) -> Tuple["IDData", "IDData"]:
        """
        Split data into training and test sets.

        Args:
            train_ratio: Fraction of data to use for training

        Returns:
        --------
        Tuple['IDData', 'IDData']
            Training and test IDData objects
        """
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")

        n_train = int(self.n_samples * train_ratio)

        # Split the data
        combined = pd.concat([self.input_data, self.output_data], axis=1)
        train_data = combined.iloc[:n_train]
        test_data = combined.iloc[n_train:]

        # Create new IDData objects
        train_iddata = IDData(
            train_data, self.input_names, self.output_names, self.sample_time
        )
        test_iddata = IDData(
            test_data, self.input_names, self.output_names, self.sample_time
        )

        # Propagate mask if available
        if self.bad_mask is not None:
            train_iddata.bad_mask = self.bad_mask.iloc[:n_train]
            test_iddata.bad_mask = self.bad_mask.iloc[n_train:]

        return train_iddata, test_iddata

    def resample(self, new_period: str) -> "IDData":
        """
        Resample data to a new time period.

        Args:
            new_period: Pandas resampling string (e.g., '1min', '5s', '1H')

        Returns:
        --------
        IDData
            Resampled IDData object
        """
        if not isinstance(self.time_stamps, pd.DatetimeIndex):
            raise ValueError("Cannot resample non-datetime indexed data")

        # Combine data for resampling
        combined_data = pd.concat([self.input_data, self.output_data], axis=1)

        # Resample
        resampled_data = combined_data.resample(new_period).mean()

        # Calculate new sample time
        new_tsample = pd.Timedelta(new_period).total_seconds()

        resampled_id = IDData(
            resampled_data, self.input_names, self.output_names, new_tsample
        )

        # Resample mask with max (any affected in bin)
        if self.bad_mask is not None:
            try:
                resampled_mask = self.bad_mask.resample(new_period).max()
                resampled_id.bad_mask = resampled_mask
            except Exception:
                # If resample fails, drop mask silently
                pass

        return resampled_id

    def remove_mean(self) -> "IDData":
        """
        Remove mean from input and output data.

        Returns:
        --------
        IDData
            Mean-removed IDData object
        """
        # Calculate means
        input_means = self.input_data.mean()
        output_means = self.output_data.mean()

        # Remove means
        input_centered = self.input_data - input_means
        output_centered = self.output_data - output_means

        # Combine data
        combined_data = pd.concat([input_centered, output_centered], axis=1)

        # Create new IDData object
        centered_iddata = IDData(
            combined_data, self.input_names, self.output_names, self.sample_time
        )
        if self.bad_mask is not None:
            centered_iddata.bad_mask = self.bad_mask.copy()

        return centered_iddata

    def plot(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot input and output data.

        Args:
            figsize: Figure size as (width, height)
        """
        n_plots = self.n_inputs + self.n_outputs
        if n_plots == 0:
            return

        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, squeeze=False)

        # Plot inputs
        for i, input_name in enumerate(self.input_names):
            axes[i, 0].plot(self.time_stamps, self.input_data[input_name])
            axes[i, 0].set_ylabel(f"{input_name}\n(input)")
            axes[i, 0].grid(True)

        # Plot outputs
        for j, output_name in enumerate(self.output_names):
            ax_idx = self.n_inputs + j
            axes[ax_idx, 0].plot(self.time_stamps, self.output_data[output_name])
            axes[ax_idx, 0].set_ylabel(f"{output_name}\n(output)")
            axes[ax_idx, 0].grid(True)

        axes[-1, 0].set_xlabel("Time")
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        """String representation of IDData object."""
        return (
            f"IDData object with {self.n_samples} samples, "
            f"{self.n_inputs} inputs, {self.n_outputs} outputs, "
            f"sample time = {self.sample_time} seconds"
        )

    def __str__(self) -> str:
        """Detailed string representation."""
        info = [
            "IDData object:",
            f"  Number of samples: {self.n_samples}",
            f"  Number of inputs: {self.n_inputs} ({', '.join(self.input_names)})",
            f"  Number of outputs: {self.n_outputs} ({', '.join(self.output_names)})",
            f"  Sample time: {self.sample_time} seconds",
        ]
        return "\n".join(info)

    # -------- Slice-aware helpers --------
    def handle_slices(
        self,
        slices: Optional[Dict[str, Any]] = None,
        bad_strategy: str = "ffill",
        interpolate_method: str = "linear",
    ) -> "IDData":
        """
        Return a new IDData with slices applied.
        """
        combined = pd.concat([self.input_data, self.output_data], axis=1)
        if not slices:
            return IDData(
                combined, self.input_names, self.output_names, self.sample_time
            )

        from sippy.utils.slices import process_slices

        processed, mask = process_slices(
            combined,
            slices,
            bad_strategy=bad_strategy,
            interpolate_method=interpolate_method,
        )
        new_obj = IDData(
            processed, self.input_names, self.output_names, self.sample_time
        )
        new_obj.bad_mask = mask
        return new_obj

    def get_bad_mask(self) -> pd.DataFrame:
        """Return the boolean mask of affected samples (False if none)."""
        if self.bad_mask is not None:
            return self.bad_mask.copy()
        return pd.DataFrame(
            False,
            index=self.input_data.index,
            columns=self.input_names + self.output_names,
        )

    def drop_masked(self, any_col: bool = True) -> "IDData":
        """
        Drop rows affected by slices based on the stored mask.
        any_col=True drops rows where any selected column was affected.
        """
        if self.bad_mask is None:
            return IDData(
                pd.concat([self.input_data, self.output_data], axis=1),
                self.input_names,
                self.output_names,
                self.sample_time,
            )

        mask_subset = self.bad_mask[self.input_names + self.output_names]
        selector = mask_subset.any(axis=1) if any_col else mask_subset.all(axis=1)
        kept = ~selector
        combined = pd.concat([self.input_data, self.output_data], axis=1)[kept]
        new_obj = IDData(
            combined, self.input_names, self.output_names, self.sample_time
        )
        new_obj.bad_mask = mask_subset[kept]
        return new_obj

    @classmethod
    def from_filter(
        cls,
        filter_obj: Any,
        dataset: str = "output",
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        tsample: Optional[float] = None,
        slices: Optional[Dict[str, Any]] = None,
        bad_strategy: str = "ffill",
        interpolate_method: str = "linear",
    ) -> "IDData":
        """
        Build IDData from a filter object's data_manager.
        """
        if not hasattr(filter_obj, "data_manager"):
            raise ValueError(
                "filter_obj must expose a data_manager with stored DataFrames"
            )
        df = filter_obj.data_manager.get_data(dataset)
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError(f"Dataset '{dataset}' not found in filter data_manager")

        # Infer inputs/outputs if not provided
        if inputs is None or outputs is None:
            # Try to recover from metadata; otherwise treat all but last as inputs
            meta = filter_obj.data_manager.get_metadata(dataset)
            cols = list(df.columns)
            if inputs is None:
                inputs = meta.get("inputs") or cols[:-1]
            if outputs is None:
                outputs = meta.get("outputs") or cols[-1:]

        return cls(
            df,
            inputs,
            outputs,
            tsample=tsample,
            slices=slices,
            bad_strategy=bad_strategy,
            interpolate_method=interpolate_method,
        )
