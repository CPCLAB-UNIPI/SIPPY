"""
Slice processing utilities shared across filters and IDData.

Provides a single implementation for handling "bad" and "interpolate" slices
on pandas DataFrames so behavior stays consistent across the codebase.
"""

from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def _normalize_targets(columns: Iterable[str], slice_info: Dict[str, Any]) -> Iterable[str]:
    if slice_info.get("isGlobal", False):
        return list(columns)
    tags = slice_info.get("tags", []) or []
    return [t for t in tags if t in columns]


def _clamp_span(n: int, start: int, end: int) -> Tuple[int, int]:
    s = max(0, int(start))
    e = min(n, int(end))
    if e < s:
        s, e = e, s
    return s, e


def process_slices(
    data: pd.DataFrame,
    slices: Dict[str, Any],
    bad_strategy: str = "ffill",
    interpolate_method: str = "linear",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply slice-based processing and return processed data and a boolean mask.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe to process (not modified in-place)
    slices : dict
        Mapping name -> slice_info dict with keys: type, start, end, tags, isGlobal
    bad_strategy : str
        Strategy to fill "bad" slices: 'ffill', 'bfill', or 'nan'
    interpolate_method : str
        Method forwarded to pandas Series.interpolate for "interpolate" slices

    Returns
    -------
    processed : pd.DataFrame
        DataFrame after applying slice operations
    mask : pd.DataFrame
        Boolean mask (True where data was affected by any slice)
    """
    processed = data.copy(deep=True)
    mask = pd.DataFrame(False, index=data.index, columns=data.columns)

    if not slices:
        return processed, mask

    n = len(processed)
    bad_present = False

    for slice_info in slices.values():
        st = slice_info.get("start", 0)
        en = slice_info.get("end", 0)
        s, e = _clamp_span(n, st, en)
        if s == e:
            continue

        targets = _normalize_targets(processed.columns, slice_info)
        if not targets:
            continue

        if slice_info.get("type") == "bad":
            bad_present = True
            for tag in targets:
                col_idx = processed.columns.get_loc(tag)
                processed.iloc[s:e, col_idx] = np.nan
                mask.iloc[s:e, col_idx] = True

        elif slice_info.get("type") == "interpolate":
            for tag in targets:
                col_idx = processed.columns.get_loc(tag)
                processed.iloc[s:e, col_idx] = np.nan
                mask.iloc[s:e, col_idx] = True
                # Interpolate this column; fill both directions to handle edges
                processed[tag] = processed[tag].interpolate(
                    method=interpolate_method, limit_direction="both"
                )

    if bad_present and bad_strategy != "nan":
        if bad_strategy == "ffill":
            processed = processed.ffill()
        elif bad_strategy == "bfill":
            processed = processed.bfill()
        else:
            # Unknown strategy -> leave NaNs as-is
            pass

    return processed, mask
