# from control.timeplot import time_response_plot
import warnings

import numpy as np
import scipy as sp
import scipy.linalg
from control import config
from control.exception import ControlMIMONotImplemented
from control.frdata import FrequencyResponseData

# from control.exception import pandas_check
from control.iosys import isctime, isdtime, issiso
from control.nlsys import NonlinearIOSystem
from control.statesp import StateSpace, _ssmatrix
from numpy import any, empty, squeeze

from .tf2ss import tf2ss


def _convert_to_statespace(sys, use_prefix_suffix=False, method=None):
    """Convert a system to state space form (if needed).

    If sys is already a state space, then it is returned.  If sys is a
    transfer function object, then it is converted to a state space and
    returned.

    Note: no renaming of inputs and outputs is performed; this should be done
    by the calling function.

    """
    import itertools

    from control.xferfcn import TransferFunction

    if isinstance(sys, StateSpace):
        return sys

    elif isinstance(sys, TransferFunction):
        # Make sure the transfer function is proper
        if any(
            [[len(num) for num in col] for col in sys.num]
            > [[len(num) for num in col] for col in sys.den]
        ):
            raise ValueError(
                "transfer function is non-proper; can't "
                "convert to StateSpace system"
            )

        if method is None or method == "slycot":
            # Change the numerator and denominator arrays so that the transfer
            # function matrix has a common denominator.
            # matrices are also sized/padded to fit tf2ss
            num, den, denorder = sys.minreal()._common_den()
            num, den, denorder = sys._common_den()
            den = np.expand_dims(den, axis=0)
            den = np.tile(den, (num.shape[0], 1, 1))
            # transfer function to state space conversion now should work!
            A, B, C, D = tf2ss(num, den)
            newsys = StateSpace(
                A,
                B[:, : sys.ninputs],
                C[: sys.noutputs, :],
                D[: sys.noutputs, : sys.ninputs],
                sys.dt,
            )

        elif method in [None, "scipy"]:
            # Scipy tf->ss can't handle MIMO, but SISO is OK
            maxn = max(max(len(n) for n in nrow) for nrow in sys.num)
            maxd = max(max(len(d) for d in drow) for drow in sys.den)
            if 1 == maxn and 1 == maxd:
                D = empty((sys.noutputs, sys.ninputs), dtype=float)
                for i, j in itertools.product(
                    range(sys.noutputs), range(sys.ninputs)
                ):
                    D[i, j] = sys.num[i][j][0] / sys.den[i][j][0]
                newsys = StateSpace([], [], [], D, sys.dt)
            else:
                if not issiso(sys):
                    raise ControlMIMONotImplemented(
                        "MIMO system conversion not supported without Slycot"
                    )

                A, B, C, D = sp.signal.tf2ss(
                    squeeze(sys.num), squeeze(sys.den)
                )
                newsys = StateSpace(A, B, C, D, sys.dt)
        else:
            raise ValueError(f"unknown {method=}")

        # Copy over the signal (and system) names
        newsys._copy_names(
            sys, prefix_suffix_name="converted" if use_prefix_suffix else None
        )
        return newsys

    elif isinstance(sys, FrequencyResponseData):
        raise TypeError("Can't convert FRD to StateSpace system.")

    # If this is a matrix, try to create a constant feedthrough
    try:
        D = _ssmatrix(np.atleast_2d(sys))
        return StateSpace([], [], [], D, dt=None)

    except Exception:
        raise TypeError("Can't convert given type to StateSpace system.")


def forced_response(
    sysdata,
    T=None,
    U=0.0,
    X0=0.0,
    transpose=False,
    params=None,
    interpolate=False,
    return_x: bool = None,
    squeeze=None,
):
    """Compute the output of a linear system given the input.

    As a convenience for parameters `U`, `X0`:
    Numbers (scalars) are converted to constant arrays with the correct shape.
    The correct shape is inferred from arguments `sys` and `T`.

    For information on the **shape** of parameters `U`, `T`, `X0` and
    return values `T`, `yout`, `xout`, see :ref:`time-series-convention`.

    Parameters
    ----------
    sysdata : I/O system or list of I/O systems
        I/O system(s) for which forced response is computed.

    T : array_like, optional for discrete LTI `sys`
        Time steps at which the input is defined; values must be evenly spaced.

        If None, `U` must be given and `len(U)` time steps of sys.dt are
        simulated. If sys.dt is None or True (undetermined time step), a time
        step of 1.0 is assumed.

    U : array_like or float, optional
        Input array giving input at each time `T`.
        If `U` is None or 0, `T` must be given, even for discrete
        time systems. In this case, for continuous time systems, a direct
        calculation of the matrix exponential is used, which is faster than the
        general interpolating algorithm used otherwise.

    X0 : array_like or float, default=0.
        Initial condition.

    params : dict, optional
        If system is a nonlinear I/O system, set parameter values.

    transpose : bool, default=False
        If True, transpose all input and output arrays (for backward
        compatibility with MATLAB and :func:`scipy.signal.lsim`).

    interpolate : bool, default=False
        If True and system is a discrete time system, the input will
        be interpolated between the given time steps and the output
        will be given at system sampling rate.  Otherwise, only return
        the output at the times given in `T`.  No effect on continuous
        time simulations.

    return_x : bool, default=None
        Used if the time response data is assigned to a tuple:

        * If False, return only the time and output vectors.

        * If True, also return the the state vector.

        * If None, determine the returned variables by
          config.defaults['forced_response.return_x'], which was True
          before version 0.9 and is False since then.

    squeeze : bool, optional
        By default, if a system is single-input, single-output (SISO) then
        the output response is returned as a 1D array (indexed by time).  If
        `squeeze` is True, remove single-dimensional entries from the shape of
        the output even if the system is not SISO. If `squeeze` is False, keep
        the output as a 2D array (indexed by the output number and time)
        even if the system is SISO. The default behavior can be overridden by
        config.defaults['control.squeeze_time_response'].

    Returns
    -------
    results : :class:`TimeResponseData` or :class:`TimeResponseList`
        Time response represented as a :class:`TimeResponseData` object or
        list of :class:`TimeResponseData` objects containing the following
        properties:

        * time (array): Time values of the output.

        * outputs (array): Response of the system.  If the system is SISO and
          `squeeze` is not True, the array is 1D (indexed by time).  If the
          system is not SISO or `squeeze` is False, the array is 2D (indexed
          by output and time).

        * states (array): Time evolution of the state vector, represented as
          a 2D array indexed by state and time.

        * inputs (array): Input(s) to the system, indexed by input and time.

        The `plot()` method can be used to create a plot of the time
        response(s) (see :func:`time_response_plot` for more information).

    See Also
    --------
    step_response, initial_response, impulse_response, input_output_response

    Notes
    -----
    1. For discrete time systems, the input/output response is computed
       using the :func:`scipy.signal.dlsim` function.

    2. For continuous time systems, the output is computed using the matrix
       exponential `exp(A t)` and assuming linear interpolation of the
       inputs between time points.

    3. If a nonlinear I/O system is passed to `forced_response`, the
       `input_output_response` function is called instead.  The main
       difference between `input_output_response` and `forced_response` is
       that `forced_response` is specialized (and optimized) for linear
       systems.
    """
    from control.nlsys import input_output_response
    from control.timeresp import (
        TimeResponseData,
        TimeResponseList,
        _check_convert_array,
    )
    from control.xferfcn import TransferFunction

    # If passed a list, recursively call individual responses with given T
    if isinstance(sysdata, list | tuple):
        responses = []
        for sys in sysdata:
            responses.append(
                forced_response(
                    sys,
                    T,
                    U=U,
                    X0=X0,
                    transpose=transpose,
                    params=params,
                    interpolate=interpolate,
                    return_x=return_x,
                    squeeze=squeeze,
                )
            )
        return TimeResponseList(responses)
    else:
        sys = sysdata

    if not isinstance(sys, StateSpace | TransferFunction):
        if isinstance(sys, NonlinearIOSystem):
            if interpolate:
                warnings.warn(
                    "interpolation not supported for nonlinear I/O systems"
                )
            return input_output_response(
                sys,
                T,
                U,
                X0,
                params=params,
                transpose=transpose,
                return_x=return_x,
                squeeze=squeeze,
            )
        else:
            raise TypeError(
                "Parameter ``sys``: must be a ``StateSpace`` or"
                " ``TransferFunction``)"
            )

    # If return_x was not specified, figure out the default
    if return_x is None:
        return_x = config.defaults["forced_response.return_x"]

    # If return_x is used for TransferFunction, issue a warning
    if return_x and isinstance(sys, TransferFunction):
        warnings.warn(
            "return_x specified for a transfer function system. Internal "
            "conversion to state space used; results may meaningless."
        )

    # If we are passed a transfer function and X0 is non-zero, warn the user
    if isinstance(sys, TransferFunction) and np.any(X0 != 0):
        warnings.warn(
            "Non-zero initial condition given for transfer function system. "
            "Internal conversion to state space used; may not be consistent "
            "with given X0."
        )

    sys = _convert_to_statespace(sys)
    A, B, C, D = (
        np.asarray(sys.A),
        np.asarray(sys.B),
        np.asarray(sys.C),
        np.asarray(sys.D),
    )
    # d_type = A.dtype
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    n_outputs = C.shape[0]

    # Convert inputs to numpy arrays for easier shape checking
    if U is not None:
        U = np.asarray(U)
    if T is not None:
        # T must be array-like
        T = np.asarray(T)

    # Set and/or check time vector in discrete time case
    if isdtime(sys):
        if T is None:
            if U is None or (U.ndim == 0 and U == 0.0):
                raise ValueError(
                    "Parameters ``T`` and ``U`` can't both be "
                    "zero for discrete-time simulation"
                )
            # Set T to equally spaced samples with same length as U
            if U.ndim == 1:
                n_steps = U.shape[0]
            else:
                n_steps = U.shape[1]
            dt = 1.0 if sys.dt in [True, None] else sys.dt
            T = np.array(range(n_steps)) * dt
        else:
            if U.ndim == 0:
                U = np.full((n_inputs, T.shape[0]), U)
    else:
        if T is None:
            raise ValueError(
                "Parameter ``T`` is mandatory for continuous time systems."
            )

    # Test if T has shape (n,) or (1, n);
    T = _check_convert_array(
        T,
        [("any",), (1, "any")],
        "Parameter ``T``: ",
        squeeze=True,
        transpose=transpose,
    )

    n_steps = T.shape[0]  # number of simulation steps

    # equally spaced also implies strictly monotonic increase,
    dt = (T[-1] - T[0]) / (n_steps - 1)
    if not np.allclose(np.diff(T), dt):
        raise ValueError(
            "Parameter ``T``: time values must be equally spaced."
        )

    # create X0 if not given, test if X0 has correct shape
    X0 = _check_convert_array(
        X0, [(n_states,), (n_states, 1)], "Parameter ``X0``: ", squeeze=True
    )

    # Test if U has correct shape and type
    legal_shapes = (
        [(n_steps,), (1, n_steps)] if n_inputs == 1 else [(n_inputs, n_steps)]
    )
    U = _check_convert_array(
        U,
        legal_shapes,
        "Parameter ``U``: ",
        squeeze=False,
        transpose=transpose,
    )

    xout = np.zeros((n_states, n_steps))
    xout[:, 0] = X0
    yout = np.zeros((n_outputs, n_steps))

    # Separate out the discrete and continuous time cases
    if isctime(sys, strict=True):
        # Solve the differential equation, copied from scipy.signal.ltisys.

        # Faster algorithm if U is zero
        # (if not None, it was converted to array above)
        if U is None or np.all(U == 0):
            # Solve using matrix exponential
            expAdt = sp.linalg.expm(A * dt)
            for i in range(1, n_steps):
                xout[:, i] = expAdt @ xout[:, i - 1]
            yout = C @ xout

        # General algorithm that interpolates U in between output points
        else:
            # convert input from 1D array to 2D array with only one row
            if U.ndim == 1:
                U = U.reshape(1, -1)  # pylint: disable=E1103

            # Algorithm: to integrate from time 0 to time dt, with linear
            # interpolation between inputs u(0) = u0 and u(dt) = u1, we solve
            #   xdot = A x + B u,        x(0) = x0
            #   udot = (u1 - u0) / dt,   u(0) = u0.
            #
            # Solution is
            #   [ x(dt) ]       [ A*dt  B*dt  0 ] [  x0   ]
            #   [ u(dt) ] = exp [  0     0    I ] [  u0   ]
            #   [u1 - u0]       [  0     0    0 ] [u1 - u0]

            M = np.block([
                [A * dt, B * dt, np.zeros((n_states, n_inputs))],
                [
                    np.zeros((n_inputs, n_states + n_inputs)),
                    np.identity(n_inputs),
                ],
                [np.zeros((n_inputs, n_states + 2 * n_inputs))],
            ])
            expM = sp.linalg.expm(M)
            Ad = expM[:n_states, :n_states]
            Bd1 = expM[:n_states, n_states + n_inputs :]
            Bd0 = expM[:n_states, n_states : n_states + n_inputs] - Bd1

            for i in range(1, n_steps):
                xout[:, i] = (
                    Ad @ xout[:, i - 1] + Bd0 @ U[:, i - 1] + Bd1 @ U[:, i]
                )
            yout = C @ xout + D @ U
        tout = T

    else:
        # Discrete type system => use SciPy signal processing toolbox

        # sp.signal.dlsim assumes T[0] == 0
        spT = T - T[0]

        if sys.dt is not True and sys.dt is not None:
            # Make sure that the time increment is a multiple of sampling time

            # First make sure that time increment is bigger than sampling time
            # (with allowance for small precision errors)
            if dt < sys.dt and not np.isclose(dt, sys.dt):
                raise ValueError("Time steps ``T`` must match sampling time")

            # Now check to make sure it is a multiple (with check against
            # sys.dt because floating point mod can have small errors
            if not (
                np.isclose(dt % sys.dt, 0) or np.isclose(dt % sys.dt, sys.dt)
            ):
                raise ValueError(
                    "Time steps ``T`` must be multiples of sampling time"
                )
            sys_dt = sys.dt

            # sp.signal.dlsim returns not enough samples if
            # T[-1] - T[0] < sys_dt * decimation * (n_steps - 1)
            # due to rounding errors.
            # https://github.com/scipyscipy/blob/v1.6.1/scipy/signal/ltisys.py#L3462
            scipy_out_samples = int(np.floor(spT[-1] / sys_dt)) + 1
            if scipy_out_samples < n_steps:
                # parantheses: order of evaluation is important
                spT[-1] = spT[-1] * (n_steps / (spT[-1] / sys_dt + 1))

        else:
            sys_dt = dt  # For unspecified sampling time, use time incr

        # Discrete time simulation using signal processing toolbox
        dsys = (A, B, C, D, sys_dt)

        # Use signal processing toolbox for the discrete time simulation
        # Transpose the input to match toolbox convention
        tout, yout, xout = sp.signal.dlsim(dsys, np.transpose(U), spT, X0)
        tout = tout + T[0]

        if not interpolate:
            # If dt is different from sys.dt, resample the output
            inc = int(round(dt / sys_dt))
            tout = T  # Return exact list of time steps
            yout = yout[::inc, :]
            xout = xout[::inc, :]
        else:
            # Interpolate the input to get the right number of points
            U = sp.interpolate.interp1d(T, U)(tout)

        # Transpose the output and state vectors to match local convention
        xout = np.transpose(xout)
        yout = np.transpose(yout)
    return TimeResponseData(
        tout,
        yout,
        xout,
        U,
        params=params,
        issiso=sys.issiso(),
        output_labels=sys.output_labels,
        input_labels=sys.input_labels,
        state_labels=sys.state_labels,
        sysname=sys.name,
        plot_inputs=True,
        title="Forced response for " + sys.name,
        trace_types=["forced"],
        transpose=transpose,
        return_x=return_x,
        squeeze=squeeze,
    )
