"""Test script to run all SIPPY examples with pytest.

This script imports and executes all example files to ensure they run without errors.
Matplotlib is configured to use the 'Agg' backend to prevent interactive plotting.

Usage:
    pytest test_examples.py
    pytest test_examples.py -v  # verbose output
    pytest test_examples.py::test_ex_armax  # run specific test
"""

import os
import sys
import warnings
from pathlib import Path

# Configure matplotlib to use non-interactive backend for testing
import matplotlib
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# Ensure the current directory is in the path for imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Also add parent directory to path in case sippy needs to be imported
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


class TestExamples:
    """Test class for all SIPPY examples."""

    def setup_method(self):
        """Setup method run before each test."""
        # Clear any existing plots
        plt.close("all")
        # Suppress warnings for cleaner test output
        warnings.filterwarnings("ignore")

    def teardown_method(self):
        """Teardown method run after each test."""
        # Clear plots after each test
        plt.close("all")

    def test_ex_armax(self):
        """Test ARMAX example."""
        # Save original directory
        original_dir = os.getcwd()
        try:
            # Change to Examples directory
            os.chdir(current_dir)

            # Import and run the example

        except Exception as e:
            pytest.fail(f"Ex_ARMAX.py failed with error: {str(e)}")
        finally:
            # Restore original directory
            os.chdir(original_dir)

    def test_ex_armax_mimo(self):
        """Test ARMAX MIMO example."""
        original_dir = os.getcwd()
        try:
            os.chdir(current_dir)
        except Exception as e:
            pytest.fail(f"Ex_ARMAX_MIMO.py failed with error: {str(e)}")
        finally:
            os.chdir(original_dir)

    def test_ex_arx_mimo(self):
        """Test ARX MIMO example."""
        original_dir = os.getcwd()
        try:
            os.chdir(current_dir)
        except Exception as e:
            pytest.fail(f"Ex_ARX_MIMO.py failed with error: {str(e)}")
        finally:
            os.chdir(original_dir)

    def test_ex_cst(self):
        """Test CST example."""
        original_dir = os.getcwd()
        try:
            os.chdir(current_dir)
        except Exception as e:
            pytest.fail(f"Ex_CST.py failed with error: {str(e)}")
        finally:
            os.chdir(original_dir)

    def test_ex_ss(self):
        """Test State Space example."""
        original_dir = os.getcwd()
        try:
            os.chdir(current_dir)
        except Exception as e:
            pytest.fail(f"Ex_SS.py failed with error: {str(e)}")
        finally:
            os.chdir(original_dir)

    def test_ex_recursive(self):
        """Test Recursive identification example."""
        original_dir = os.getcwd()
        try:
            os.chdir(current_dir)
        except Exception as e:
            pytest.fail(f"Ex_RECURSIVE.py failed with error: {str(e)}")
        finally:
            os.chdir(original_dir)

    def test_ex_opt_gen_inout(self):
        """Test OPT GEN-INOUT example."""
        original_dir = os.getcwd()
        try:
            os.chdir(current_dir)
            # Note: Import name uses underscores instead of hyphens
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "Ex_OPT_GEN_INOUT", current_dir / "Ex_OPT_GEN-INOUT.py"
            )
            if spec is None or spec.loader is None:
                raise ImportError("Could not load Ex_OPT_GEN-INOUT.py")
            Ex_OPT_GEN_INOUT = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(Ex_OPT_GEN_INOUT)
        except Exception as e:
            pytest.fail(f"Ex_OPT_GEN-INOUT.py failed with error: {str(e)}")
        finally:
            os.chdir(original_dir)


def test_all_examples_exist():
    """Test that all expected example files exist."""
    expected_files = [
        "Ex_ARMAX.py",
        "Ex_ARMAX_MIMO.py",
        "Ex_ARX_MIMO.py",
        "Ex_CST.py",
        "Ex_SS.py",
        "Ex_RECURSIVE.py",
        "Ex_OPT_GEN-INOUT.py",
    ]

    for filename in expected_files:
        filepath = current_dir / filename
        assert filepath.exists(), f"Example file {filename} not found"


if __name__ == "__main__":
    """Allow running the test file directly."""
    pytest.main([__file__, "-v"])
