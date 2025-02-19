import os
import subprocess

# Move to the directory where this script resides
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def test_ex_armax_mimo():
    result = subprocess.run(
        ["python", "Ex_ARMAX_MIMO.py"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_ex_armax():
    result = subprocess.run(
        ["python", "Ex_ARMAX.py"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_ex_arx_mimo():
    result = subprocess.run(
        ["python", "Ex_ARX_MIMO.py"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_ex_cst():
    result = subprocess.run(
        ["python", "Ex_CST.py"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_ex_opt_gen_inout():
    result = subprocess.run(
        ["python", "Ex_OPT_GEN-INOUT.py"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_ex_recursive():
    result = subprocess.run(
        ["python", "Ex_RECURSIVE.py"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_ex_ss():
    result = subprocess.run(
        ["python", "Ex_SS.py"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_armax_ic():
    result = subprocess.run(
        "jupyter nbconvert --execute --to notebook --inplace test_armax.ipynb".split(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
