from collections.abc import Callable
from typing import TypeVar, overload

from control.matlab import TransferFunction
from control.matlab import tf as tf_control

T = TypeVar("T")  # For the element type in the lists

# Recursive type definition for nested lists
NestedList = list[T] | list["NestedList[T]"]
NestedTransferFunction = (
    list[TransferFunction] | list["NestedTransferFunction"]
)


def tf(*args, **kwargs) -> TransferFunction:
    result = tf_control(*args, **kwargs)
    if result is None:
        raise ValueError("Transfer function creation failed.")
    return result


@overload
def make_tf(
    numerator: list[float],
    denominator: list[float],
    ts: float = 1.0,
    noise: float = 0.0,
    random_state: int | None = None,
) -> TransferFunction: ...
@overload
def make_tf(
    numerator: list[list[float]],
    denominator: list[list[float]],
    ts: float = 1.0,
    noise: float = 0.0,
    random_state: int | None = None,
) -> NestedTransferFunction: ...


def make_tf(
    numerator: list[float] | list[list[float]],
    denominator: list[float] | list[list[float]],
    ts: float = 1.0,
    noise: float = 0.0,
    random_state: int | None = None,
) -> TransferFunction | NestedTransferFunction:
    """Generate a single-input single-output system.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples, by default 1000.
    n_taps : int, optional
        Number of taps, by default 10.
    n_features : int, optional
        Number of features, by default 1.
    n_targets : int, optional
        Number of targets, by default 1.
    n_informative : int, optional
        Number of informative features, by default 1.
    noise : float, optional
        Standard deviation of Gaussian noise, by default 0.0.
    random_state : int, optional
        Random seed, by default None.

    Returns

    -------
    Tuple[np.ndarray, np.ndarray]
        Input and output data.
    """
    sys = _apply_on_nested(numerator, denominator, lambda a, b: tf(a, b, ts))

    if sys is None:
        raise ValueError("Invalid system parameters. Could not create system.")

    return sys


def _apply_on_nested(
    num: NestedList,
    den: NestedList,
    func: Callable[[list[float], list[float]], TransferFunction],
) -> TransferFunction | NestedTransferFunction:
    """
    Recursively processes two nested lists by applying the transformation function func
    to the deepest level lists.

    Args:
        num: A nested list structure containing numeric values
        den: A nested list structure containing numeric values
        func: A function that takes two lists (num_list, den_list) and returns a result
            without iterating through or modifying the deepest level lists

    Returns:
        A nested list structure matching the original nesting pattern, with the func
        function applied to the deepest level lists

    Examples:
    >>> def func(a, b):
    ...     # Simply return both lists as a tuple without iterating through them
    ...     return (a, b)

    >>> # Simple case - both are already at deepest level
    >>> _apply_on_nested([1, 2, 3], [4, 5], func)
    ([1, 2, 3], [4, 5])

    >>> # One level of nesting
    >>> _apply_on_nested([[1, 2], [3, 4]], [[5, 6], [7, 8]], func)
    [([1, 2], [5, 6]), ([3, 4], [7, 8])]

    >>> # Two levels of nesting
    >>> _apply_on_nested([[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]], func)
    [[([1, 2], [5, 6])], [([3, 4], [7, 8])]]

    >>> # Different nesting levels
    >>> _apply_on_nested([[1, 2], [3, 4]], [5, 6], func)
    [([1, 2], [5, 6]), ([3, 4], [5, 6])]
    """

    # Helper function to check if an item is a list
    def is_list(item):
        return isinstance(item, list)

    # Helper function to check if list contains any sublists
    def contains_sublists(lst):
        return any(is_list(item) for item in lst)

    # Base case 1: If num is not a list
    if not is_list(num):
        return num

    # Base case 2: If num is a list with no sublists (deepest level)
    if not contains_sublists(num):
        # If den is also a list with no sublists, apply tf
        if is_list(den) and not contains_sublists(den):
            return func(num, den)
        # If den is not a list or contains sublists, handle appropriately
        elif not is_list(den):
            return func(num, [den])
        else:  # den contains sublists
            # Find the first deepest list in den
            for d in den:
                if is_list(d) and not contains_sublists(d):
                    return func(num, d)
            # If no deepest list found in den, use den itself
            return func(num, den)

    # Recursive case: num has sublists
    result = []

    # If den is not a list, apply it to each sublist in num
    if not is_list(den):
        for n in num:
            result.append(_apply_on_nested(n, den, func))
        return result

    # If den is a list but doesn't have sublists, apply it to each sublist in num
    if not contains_sublists(den):
        for n in num:
            result.append(_apply_on_nested(n, den, func))
        return result

    # Both num and den have sublists
    # If den has fewer items, extend it by repeating the last item
    if len(den) < len(num):
        den_extended = den + [den[-1]] * (len(num) - len(den))
    else:
        den_extended = den[: len(num)]  # Truncate if den is longer

    # Process each pair of elements
    for i, n in enumerate(num):
        result.append(_apply_on_nested(n, den_extended[i], func))

    return result
