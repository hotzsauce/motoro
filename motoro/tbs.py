"""
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal, Optional, Sequence, Union
import warnings
from dataclasses import dataclass



__all__ = [
    "TopBottomSpread",
    "tb_spread",
]

NaNPolicy = Literal["drop", "raise", "propagate"]



@dataclass
class TopBottomSpread(object):
    """
    Compute _top-bottom (TB) spreads_ over rolling windows of a time-series.

    A TB-spread is the difference between the sum of the *n* highest
    observations (the **top**) and the *n* lowest observations (the **bottom**)
    within every grouping window.  The class is a thin, configurable wrapper
    around vectorised NumPy kernels that makes it easy to:

    * work with both hourly and sub-hourly data,
    * restrict comparisons to contiguous blocks (`contiguous=True`),
    * enforce “peak must follow trough” ordering (`forward=True`), and
    * perform the calculation across custom groupings such as day‑by‑day,
      hub / node, or asset IDs.

    Parameters
    ----------
    n : int, default `2`
        Number of periods to include in both the top and bottom buckets.
    freq : str or pandas.Grouper, default `"1d"`
        The grouping frequency (e.g. daily). If a `pd.Grouper` is supplied
        it is used verbatim.
    by : str or Sequence[str], optional
        Extra column(s) to include in the group key, typically categorical
        metadata such as `"hub"` or `"asset"`.
    data_freq : {"auto", <pandas-offset-alias>}, default `"auto"`
        Frequency of *input* data.  When `"auto"`, the constructor defers to
        `pandas.infer_freq` (with other fall‑backs) at run-time.
    coarse : bool, default `True`
        If *True*, data are first resampled to hourly means, so `n` is
        interpreted as *hours*. If *False*, the raw sampling interval is kept.
    forward : bool, default `False`
        Require every “peak” to occur *after* its paired “trough” within the
        same window (parenthesis-matching order; see `_total_spread_forward`).
    contiguous : bool, default `False`
        Use contiguous *n‑hour* blocks for the top and bottom instead of
        selecting individual observations.
    scale : bool, default `True`
        When `coarse=False` the method divides by the number of
        sub-hourly observations per clock hour so that results remain in
        *$/MWh* (or equivalent) instead of accumulated $.
    nan_policy : {"drop", "raise", "propagate"}
        Policy for handling NaN values:
        - "drop": Remove NaN values and continue with valid data
        - "raise": Raise ValueError if any NaN values are found
        - "propagate": Keep NaN values (will result in NaN output)

    Notes
    -----
    For performance, numeric work is delegated to NumPy and avoids Python
    loops.  Non‑numeric columns in a `DataFrame` are silently ignored (with a
    warning) by `_validate_tb_data`.
    """

    n: int = 2
    freq: Union[str, pd.Grouper] = "1d"
    by: Optional[Union[str, Sequence[str]]] = None
    data_freq: str = "auto"
    coarse: bool = True
    forward: bool = False
    contiguous: bool = False
    scale: bool = True
    nan_policy: NaNPolicy = "drop"

    def __post_init__(self):
        """
        Validate parameters and pre-compute the number of raw periods in *n*
        hours.

        Called automatically by the `@dataclass` constructor.  It raises
        `ValueError` if an hourly interpretation of `n` is impossible (e.g.,
        `n < 1` when `coarse=False`) and sets `TopBottomSpread.n_periods`
        """
        if self.n < 1 and not self.coarse:
            raise ValueError("`n` must be at least 1")

        # how many periods are in n?
        self.n_periods = self._calculate_periods_per_window()

    def calculate(
        self,
        data: pd.Series | pd.DataFrame,
    ) -> pd.Series | pd.DataFrame:
        """
        Return the TB-spread for *data* according to the instance configuration.

        Parameters
        ----------
        data : pandas.Series or pandas.DataFrame
            A numeric Series (single asset) or DataFrame whose index is a
            `pandas.DatetimeIndex`. Non-numeric columns in a DataFrame are
            ignored with a warning.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            One spread value per grouping window. For a Series the result is a
            1‑D Series whose index matches the grouping rule; for a DataFrame the
            behaviour is not yet implemented (`NotImplementedError`).

        Raises
        ------
        ValueError
            If the inferred or supplied `data_freq` does not divide evenly into
            an hour when `coarse=False`.
        TypeError
            Propagated from `_validate_tb_data` on malformed input.
        """
        _validate_tb_data(data, self.by)

        # handle automatic frequency inference
        if self.data_freq == "auto":
            self.data_freq = _infer_data_frequency(data, verbose=not self.coarse)
            self.n_periods = self._calculate_periods_per_window()

        # apply coarse resampling if needed; "hour-ize" values if needed
        if self.coarse:
            data = self._resample_to_hourly(data)
            # after resampling, we're working with hourly data
            self.data_freq = "1H"
        else:
            if self.scale:
                # even division already guaranteed in `_infer_data_frequency`
                scale_factor = self.n_periods // self.n

                if isinstance(data, pd.Series):
                    data = data / scale_factor
                else:
                    nums = data.select_dtypes(include=[np.number]).columns
                    data[nums] = data[nums] / scale_factor

        # recalculate periods now that we have the actual frequency
        self.n_periods = self._calculate_periods_per_window()

        if isinstance(data, pd.Series):
            return self._calculate_series(data)
        else:
            return self._calculate_dataframe(data)

    def _calculate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group‑wise TB‑spread over a tidy `DataFrame`.

        The input frame must contain exactly **one** numeric “price” column
        (everything else must be either the time index or *by* columns declared
        on the instance).  The method

        1.  infers which column is the price,
        2.  builds a `pandas.core.groupby.DataFrameGroupBy` using
            `_get_grouper`, and
        3.  applies the vectorised TB-spread kernel to each group.

        Parameters
        ----------
        df : pandas.DataFrame
            A tidy table whose index is a `pandas.DatetimeIndex`. The frame
            **must** contain:
            * all columns listed in `TopBottomSpread.by` (if any), and
            * exactly **one** additional numeric column that represents price.

        Returns
        -------
        pandas.Series
            One TB‑spread per grouping window with name
            `f"tb_spread_{self.n}"`.  The Series index is the same MultiIndex
            produced by `df.groupby(self._get_grouper(df))`.

        Raises
        ------
        ValueError
            * When no numeric columns are found.
            * When more than one candidate price column exists.
        """
        by_cols = self._get_grouping_columns()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("no numeric columns found in DataFrame")

        maybe_price = set(numeric_cols) - set(by_cols)
        if len(maybe_price) != 1:
            raise ValueError(
                "more than one numeric column; the only non-grouping numeric "
                "column should be the price column"
            )
        price_col = maybe_price.pop()

        grouper = self._get_grouper(df)
        grouped = df.groupby(grouper)

        return (
            grouped[price_col]
            .apply(lambda grp: self._tb_spread(grp.values))
            .rename(f"tb_spread_{self.n}")
        )

    def _calculate_periods_per_window(self) -> int:
        """
        Translate *n* hours into the equivalent number of **raw** periods.

        Returns
        -------
        int
            `n` itself when `coarse=True`; otherwise
            `n * (1 hour / data_freq)` rounded to the nearest integer.

        Raises
        ------
        ValueError
            If `data_freq` does not divide evenly into one hour.
        """
        if (self.data_freq == "auto"):
            # placeholder for the moment; will be changed later if needed
            return 1

        # when using coarse resampling, `n` maps directly to periods
        if self.coarse:
            return self.n

        data_td = pd.Timedelta(self.data_freq)
        hour_td = pd.Timedelta("1h")

        periods_per_hour = hour_td / data_td
        if not periods_per_hour.is_integer():
            raise ValueError(
                f"Data frequency {self.data_freq} does not divide evenly into "
                "one hour"
            )

        return int(self.n * periods_per_hour)

    def _calculate_series(self, series: pd.Series) -> pd.Series:
        """
        Vectorised TB-spread over a single numeric Series.

        The Series is grouped according to `_get_grouper`, converted to a
        NumPy array, and passed to `_calculate_tb_spread`.

        Parameters
        ----------
        series : pandas.Series
            Numeric data indexed by `DatetimeIndex`.

        Returns
        -------
        pandas.Series
            Group‑wise TB-spread values with name `f"tb_spread_{n}"`.
        """
        grouper = self._get_grouper(series)
        grouped = series.groupby(grouper)

        return (
            grouped
            .apply(lambda grp: self._tb_spread(grp.values))
            .rename(f"tb_spread_{self.n}")
        )

    def _get_grouper(
        self,
        data: pd.Series | pd.DataFrame,
    ) -> str | List[str]:
        """
        Build the `groupby` key used inside `calculate`.

        Returns either a single `pd.Grouper` (time only) or a list whose first
        element is the temporal Grouper followed by any categorical `by` columns.

        Parameters
        ----------
        data : pandas.Series or pandas.DataFrame
            The object to be grouped.

        Returns
        -------
        pandas.Grouper or list
            Suitable argument for `data.groupby(...)`.
        """
        if self.by is None:
            return pd.Grouper(freq=self.freq)
        else:
            by_cols = [self.by] if isinstance(self.by, str) else list(by)
            if isinstance(self.freq, str):
                return [pd.Grouper(freq=self.freq)] + by_cols
            else:
                return [self.freq] + by_cols

    def _get_grouping_columns(
        self,
        data: str | Sequence[str] = "",
    ) -> List[str]:
        """
        Assemble the list of columns that define a *grouping key*.

        The method merges the instance‑level grouping spec (`TopBottomSpread.by`)
        with a *data* spec supplied at call time. Order is preserved: `by`\
        columns appear first, followed by the entries in *data*.

        Parameters
        ----------
        data : str or Sequence[str], optional
            Column name(s) that identify the *data* portion of the key (e.g.,
            `"price"`).  If omitted, the result contains only the `by` columns.

        Returns
        -------
        list[str]
            Concatenated list `[ *by_cols*, *data_cols* ]`.

        Raises
        ------
        ValueError
            If neither the instance has `by` set **nor** *data* is provided.

        Examples
        --------
        >>> tbs = TopBottomSpread(by=["hub", "asset"])
        >>> tbs._get_grouping_columns("price")
        ['hub', 'asset', 'price']

        >>> tbs = TopBottomSpread()        # no 'by' columns
        >>> tbs._get_grouping_columns(["price", "volume"])
        ['price', 'volume']
        """
        if (not self.by) and (not data):
            raise ValueError("no grouping or data columns")

        if self.by:
            by_cols = [self.by] if isinstance(self.by, str) else list(by)
            if data and isinstance(data, str):
                return by_cols + [data]
            elif data:
                return by_cols + list(data)
            else:
                return by_cols
        else:
            if isinstance(data, str):
                return [data]
            else:
                return list(data)

    def _resample_to_hourly(
        self,
        data: pd.Series | pd.DataFrame,
    ) -> pd.Series | pd.DataFrame:
        """
        Down-sample sub-hourly data to hourly means **while preserving group keys**.

        When *by* is supplied and *data* is a `DataFrame`, numeric columns are
        resampled and the grouping labels are re-attached.

        Parameters
        ----------
        data : pandas.Series or pandas.DataFrame
            Input observations.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Hourly‑sampled numeric data.

        Raises
        ------
        ValueError
            If the grouping column(s) are missing or if a Series is provided with
            `by` specified.
        """
        if self.by is None:
            return data.resample("1h").mean()
        else:
            # need to do some work to preserve grouping columns during resampling
            by_cols = [self.by] if isinstance(self.by, str) else self.by

            if isinstance(data, pd.Series):
                raise ValuerError(
                    "Cannot use 'groupby' with Series data when 'coarse=True. "
                    "Use DataFrame with grouping columns instead"
                )

            missing_cols = [col for col in by_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Grouping cols not found in data: {missing_cols}")

            # separate numeric and grouping columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            non_numeric_cols = [col for col in by_cols if col not in numeric_cols]

            resampled_parts = []
            for grp_vals, grp_data in data.groupby(by_cols):
                resampled_numeric = grp_data[numeric_cols].resample("1h").mean()
                resampled_group = resampled_numeric.copy()
                if isinstance(grp_vals, tuple):
                    for i, col in enumerate(non_numeric_cols):
                        resampled_group[col] = grp_vals[i]
                else:
                    resampled_group[non_numeric_cols[0]] = grp_vals

                resampled_parts.append(resampled_group)

            # combine all resampled groups
            return (
                pd.concat(resampled_parts, ignore_index=False)
                .sort_index()
            )

    def _tb_spread(self, values: np.ndarray) -> float:
        """
        Core helper that converts one *window* of raw values into a TB‑spread.

        The routine is a very thin wrapper around `_calculate_tb_spread`; it
        supplies that lower‑level kernel with all algorithmic options stored on
        the current `TopBottomSpread` instance.

        Parameters
        ----------
        values : array‑like
            One‑dimensional numeric array (typically the `.values` of a grouped
            Series).  Missing values (`NaN`/`None`) are ignored.

        Returns
        -------
        float
            `sum(top_n) - sum(bottom_n)` where *n* = `self.n` converted to raw
            periods via `_calculate_periods_per_window`.  Returns
            `np.nan` when fewer than `2 * n` non-NaN observations are
            available or when the underlying kernel signals an undefined result.

        Notes
        -----
        Internally this method dispatches to:

            _calculate_tb_spread(
                values,
                n_periods=self.n_periods,
                forward=self.forward,
                contiguous=self.contiguous,
            )

        so it inherits the complexity and edge‑case behavior documented there.
        """
        return _calculate_tb_spread(
            values=values,
            n_periods=self.n_periods,
            forward=self.forward,
            contiguous=self.contiguous,
            nan_policy=self.nan_policy,
        )



def tb_spread(
    data: Union[pd.Series | pd.DataFrame],
    n: int = 2,
    freq: Union[str, pd.Grouper] = "1d",
    by: Optional[Union[str, Sequence[str]]] = None,
    data_freq: str = "auto",
    coarse: bool = True,
    forward: bool = False,
    contiguous: bool = False,
    scale: bool = True,
    nan_policy: NaNPolicy = "drop",
) -> Union[pd.Series | pd.DataFrame]:
    """
    High-level helper that returns the **top–bottom (TB) spread** of *data*.

    A TB-spread is the difference between the *n* highest and *n* lowest price
    observations within every grouping window (daily by default). This
    convenience wrapper instantiates `TopBottomSpread` with the supplied options
    and immediately evaluates `TopBottomSpread.calculate`.

    In other words, the call

    >>> tb_spread(data, n=4, freq="1D", by="hub")

    is exactly equivalent to

    >>> tbs = TopBottomSpread(n=4, freq="1D", by="hub")
    >>> tbs.calculate(data)

    but saves you from managing the class instance explicitly.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Input time-series with a `pandas.DatetimeIndex`.

        * **Series** - must be the price column.
        * **DataFrame** - must contain exactly **one** numeric price column plus
          any grouping columns referenced by *by*.
    n : int, default `2`
        Number of periods in both the *top* and *bottom* buckets.
    freq : str or pandas.Grouper, default `"1D"`
        Grouping frequency (e.g., `"1H"`, `"1D"`, `"W"`).  Passed
        straight to `TopBottomSpread(freq=...)`.
    by : str or Sequence[str], optional
        Extra column(s) on *data* that should be part of the grouping key
        (e.g., `"hub"`, `["hub", "asset_id"]`).
    data_freq : {"auto", <offset-alias>}, default `"auto"`
        Sampling interval of *data* when *coarse* is `False`.  See
        `TopBottomSpread` for details.
    coarse : bool, default `True`
        If *True*, down-sample sub-hourly data to hourly means before applying
        the algorithm, so *n* is interpreted in **hours**.
    forward : bool, default `False`
        Require that each selected peak occurs **after** its paired trough
        within the same window.
    contiguous : bool, default `False`
        Select contiguous *n‑hour* blocks instead of arbitrary individual
        observations.
    scale : bool, default `True`
        When `coarse=False` multiply/divide so that results remain in the
        same monetary units (e.g., $/MWh) rather than aggregate $.
    nan_policy : {"drop", "raise", "propagate"}
        Policy for handling NaN values:
        - "drop": Remove NaN values and continue with valid data
        - "raise": Raise ValueError if any NaN values are found
        - "propagate": Keep NaN values (will result in NaN output)

    Returns
    -------
    pandas.Series or pandas.DataFrame
        TB‑spread values labelled `f"tb_spread_{n}"`.  For a Series input the
        result is 1‑D; for a DataFrame the result is a Series indexed by the
        chosen grouping key.

    Raises
    ------
    ValueError
        For impossible parameter combinations or when *data* lacks exactly one
        numeric price column.
    TypeError
        If *data* is not a Series/DataFrame or its index is not datetime‑like.

    Examples
    --------
    **1. Hourly day‑ahead LMP Series**

    >>> spread = tb_spread(lmp_series, n=3, freq="1D", coarse=True)
    >>> spread.head()
    2025‑01‑01    87.6
    2025‑01‑02    55.3
    Name: tb_spread_3, dtype: float64

    **2. Node & hub‑grouped DataFrame (5‑minute prices)**

    >>> df    # columns: ["timestamp", "hub", "node", "price"]
    >>> result = tb_spread(
    ...     df.set_index("timestamp"),
    ...     n=2,
    ...     freq="1D",
    ...     by=["hub", "node"],
    ...     coarse=False,
    ...     forward=True,
    ... )
    >>> result.loc[("HB_WEST", "NODE_A")].plot()

    Notes
    -----
    *All* keyword arguments except *data* map one‑for‑one to the constructor of
    `TopBottomSpread`.  Refer to that class for algorithmic nuances and
    implementation details.
    """

    spreadifier = TopBottomSpread(
        n=n,
        freq=freq,
        by=by,
        data_freq=data_freq,
        coarse=coarse,
        forward=forward,
        contiguous=contiguous,
        scale=scale,
        nan_policy=nan_policy,
    )
    return spreadifier.calculate(data)



#
# hot speedy numpy stuff
#

def _calculate_tb_spread(
    values: np.ndarray,
    n_periods: int,
    forward: bool,
    contiguous: bool,
    nan_policy: NaNPolicy,
) -> float:
    """
    Dispatch the proper TB-spread algorithm for one window of *values*.

    Parameters
    ----------
    values : ndarray
        1‑D numeric array.
    n_periods : int
        Window length expressed in **raw periods**.
    forward : bool
        Enforce “peak after trough” ordering.
    contiguous : bool
        Operate on contiguous blocks rather than individual observations.
    nan_policy : {"drop", "raise", "propagate"}
        Policy for handling NaN values:
        - "drop": Remove NaN values and continue with valid data
        - "raise": Raise ValueError if any NaN values are found
        - "propagate": Keep NaN values (will result in NaN output)

    Returns
    -------
    float
        Spread value or `np.nan` when insufficient data.
    """
    if contiguous:
        rolling = _rolling_accumulation(values, n_periods, "sum", nan_policy)

        if nan_policy == "propagate":
            if any(np.isnan(rolling)):
                return np.nan
        else:
            return _spread_with_gap(rolling, n_periods, forward=forward)
    else:
        return _total_spread(
            values,
            n_periods,
            forward=forward,
            nan_policy=nan_policy,
        )

def _rolling_accumulation(
    values: np.ndarray,
    n_periods: int,
    method: str,
    nan_policy: NaNPolicy
) -> np.ndarray:
    """
    Rolling sum / mean implemented.

    Returns `np.nan` when the array is shorter than *n_periods*.

    Parameters
    ----------
    values : ndarray
        1‑D numeric array.
    n_periods : int
        Size of the rolling kernel.
    method : {'sum', 'mean'}
        Aggregation method.
    nan_policy : {"drop", "raise", "propagate"}
        Policy for handling NaN values:
        - "drop": Remove NaN values and continue with valid data
        - "raise": Raise ValueError if any NaN values are found
        - "propagate": Keep NaN values (will result in NaN output)

    Returns
    -------
    ndarray or float
        Array of accumulated values (`mode='valid'`); `np.nan` if empty.
    """
    if len(values) < n_periods:
        return np.nan

    if nan_policy == "raise":
        if np.any(np.isnan(values)):
            raise ValueError("NaN values found in input data")
    elif nan_policy == "propagate":
        # use values as-is
        pass
    else:
        # handler with throw an error if nan_policy is not 'drop'
        values, _ = _handle_nan_values(values, nan_policy)

    if method == "sum":
        kernel = np.ones(n_periods)
    elif method == "mean":
        kernel = np.ones(n_periods) / n_periods
    else:
        raise ValueError(f"Unrecognized method: {method}")

    acc = np.convolve(values, kernel, mode="valid")
    if len(acc) == 0:
        return np.nan

    return acc

def _total_spread(
    values: np.ndarray,
    n_periods: int,
    *,
    forward: bool,
    nan_policy: NaNPolicy,
) -> float:
    """
    Non‑contiguous TB‑spread via partial sorting.

    Uses `np.partition` to obtain the *n* largest / smallest values in
    *O(m)* time (`m = len(values)`).

    See Also
    --------
    _total_spread_forward : order‑constrained variant.
    """
    processed, indices = _handle_nan_values(values, nan_policy)

    # might have lost too many observations due to nans
    m = processed.size
    if m < 2 * n_periods:
        return np.nan

    # short-circuit
    if nan_policy == "propagate" and np.any(np.isnan(processed)):
        return np.nan

    if forward:
        return _total_spread_forward(processed, n_periods)
    else:
        lows = np.partition(processed, n_periods-1)[:n_periods].sum()
        highs = np.partition(processed, -n_periods)[-n_periods:].sum()
        return highs - lows

def _total_spread_forward(
    values: np.ndarray,
    n_periods: int,
) -> float:
    """
    Order‑constrained TB‑spread using dynamic programming.

    Computes `sum(highs) - sum(lows)` subject to the constraint that every
    "high" index occurs **after** its paired "low" index (balanced‑parenthesis
    interpretation).

    Complexity
    ----------
    `O(m * n_periods^2)` time and `O(n_periods^2)` memory.

    Notes
    -----
    * This does *not* accept NaN policy-related stuff because it's only called
        after `_handle_nan_values` is in `_total_spread`
    """
    m = values.size
    if n_periods <= 0 or m < 2*n_periods:
        return np.nan

    max_t = 2*n_periods
    max_b = n_periods

    # dp[t, b] = best score after reading current prefix, having taken
    # t symbols with balance b
    neg_inf = -np.inf
    dp = np.full((max_t+1, max_b+1), neg_inf, dtype=float)
    dp[0, 0] = 0.0

    for val in values.astype(float):
        low = dp[:-1, :-1] - val
        high = dp[:-1, 1:] + val

        dp_new = dp
        dp_new[1:, 1:] = np.maximum(dp_new[1:, 1:], low)
        dp_new[1:, :-1] = np.maximum(dp_new[1:, :-1], high)

        dp = dp_new

    return dp[max_t, 0] if dp[max_t, 0] > neg_inf/2 else np.nan

def _spread_with_gap(
    values: np.ndarray,
    gap: int,
    *,
    forward: bool = False,
) -> float:
    """
    Maximum absolute difference between pairs separated by **≥ gap** positions.

    Parameters
    ----------
    values : ndarray
        1‑D numeric array.
    gap : int
        Minimum index distance between the two elements being compared.
    forward : bool, default `False`
        If *True* only consider pairs where the second element is **after** the
        first.

    Notes
    -----
    * Runs in *O(m)* time using vectorised suffix extrema.
    * This does *not* accept NaN policy-related stuff because it's only called
        in `_calculate_tb_spread` after `_rolling_accumualation` is called, and
        `_rolling_accumulation` handles NaN policy
    """
    m = values.size
    if m < gap:
        return np.nan

    # this assumes the observations are equally spaced. if we wanted to 
    # generalize to the case where they're uneven, or there are gaps, then pass
    # the timestamp of each observation in nanoseconds, and those integers will
    # replace `starts` below. `gap` would have to be mapped to ns too.
    # here, just finding the first index j where start_j - start_i >= n_periods
    starts = np.arange(m)
    nxt = np.searchsorted(starts, starts + gap, side="left")
    valid = nxt < m
    if not valid.any():
        return np.nan

    suffix_max = _suffix_extrema(values, np.maximum)
    if forward:
        diffs = suffix_map[nxt[valid]] - values[valid]
        return diffs.max()

    # absolute gap; no need to pay attention to order
    suffix_min = _suffix_extrema(values, np.minimum)
    up = suffix_max[nxt[valid]] - values[valid]
    down = values[valid] - suffix_min[nxt[valid]]
    return np.maximum(up, down).max()

def _suffix_extrema(arr: np.ndarray, op) -> np.ndarray:
    """
    Vectorised suffix cumulative extremum.

    Equivalent to `np.maximum.accumulate(arr[::-1])[::-1]` (or `np.minimum`
    depending on *op*).  Used by `_spread_with_gap`.
    """
    return op.accumulate(arr[::-1])[::-1]


#
# datetime-related utility methods
#

def _infer_data_frequency(
    data: pd.Series | pd.DataFrame,
    verbose: bool = True,
) -> str:
    """
    Guess the sampling frequency of `data.index`.

    Strategy
    --------
    1. Try `pandas.infer_freq`.
    2. Fallback to the most common positive time delta and emit warnings when
       confidence < 90 %.

    Returns
    -------
    str
        Offset alias suitable for `pd.Timedelta(freq)`.
    """
    index = data.index

    if len(index) < 2:
        raise ValueError("need at least two data points to infer frequency")

    # defer to pandas with the hope everything works out
    try:
        inferred_freq = pd.infer_freq(index)
        if inferred_freq is not None:
            if verbose:
                warnings.warn(
                    f"data frequency automatically inferred as '{inferred_freq}'. "
                    f"Specify 'data_freq' explicitly to avoid this warning.",
                    UserWarning,
                    stacklevel=3,
                )
            # for the `pd.Timedelta(self.data_freq)` to run w/o error, there
            # must be a digit in the inferred freq
            if not inferred_freq[0].isdigit():
                return "1" + inferred_freq
            else:
                return inferred_freq
    except Exception:
        pass

    # fallback is to calculate most common time delta
    time_diffs = index[1:] - index[:-1]
    valid_diffs = time_diffs[time_diffs > pd.Timedelta(0)]

    if len(valid_diffs) == 0:
        raise ValueError("no valid time differences found in data")

    from collections import Counter
    diff_counts = Counter(valid_diffs)
    most_common_diff, count = diff_counts.most_common(1)[0]

    # pretty arbitrary criteria, but what are ya gonna do
    confidence = count / len(valid_diffs)
    if confidence < 0.9:
        if verbose:
            warnings.warn(
                f"Irregular time intervals detected. Most common interval "
                f"({most_common_diff}) represents only {confidence:.1%} of the "
                "data. Consider specifying 'data_freq' explictly.",
                UserWarning,
                stacklevel=3,
            )

    freq_string = _timedelta_to_freq_string(most_common_diff)
    if verbose:
        warnings.warn(
            f"Data frequency automatically inferred as '{freq_string}' based on "
            f"most common interval ({most_common_diff}). "
            "Specify 'data_freq' explicitly to avoid this warning.",
            UserWarning,
            stacklevel=3,
        )

    return freq_string

def _timedelta_to_freq_string(td: pd.TimeDelta) -> str:
    """
    Convert a `pandas.Timedelta` to an offset alias (e.g. `"15T"`).

    Handles seconds, milliseconds, microseconds, and hour multiples.
    """
    total_seconds = td.total_seconds()

    # common frequencies in order of preference
    freq_mappings = [
        (3_600, "hour"),
        (60, "min"),
        (1, "S"),
        (0.001, "L"),
        (0.000001, "U"),
    ]

    for unit_seconds, unit_code in freq_mappings:
        if total_seconds >= unit_seconds and (total_seconds % unit_seconds == 0):
            multiplier = int(total_seconds / unit_seconds)
            if multiplier == 1:
                return unit_code
            else:
                return f"{multiplier}{unit_code}"

    # fallback for very small intervals
    return f"{total_seconds}S"

def _validate_tb_data(
    data: pd.Series | pd.DataFrame,
    by: str | Sequence[str],
):
    """
    Ensure *data* is numeric and indexed by `DatetimeIndex`.

    Raises
    ------
    TypeError
        If *data* is not a Series/DataFrame or index is not datetime.
    ValueError
        If numeric columns are missing or, for a Series, dtype is non-numeric.

    Warns
    -----
    UserWarning
        When non-numeric columns in a DataFrame are ignored.
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("data must be a pandas Series or DataFrame")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must have a DatetimeIndex")

    if isinstance(data, pd.Series):
        if not pd.api.types.is_numeric_dtype(data):
            raise ValueError("series must contain numeric data")
    else:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        by_cols = [by] if isinstance(by, str) else list(by)
        if len(numeric_cols) == 0:
            raise ValueError(
                "DataFrame must contain at least one numeric column"
            )
        elif len(numeric_cols) + len(by_cols) < len(data.columns):
            non_numeric = set(data.columns) - (set(numeric_cols) | set(by_cols))
            warnings.warn(
                f"Non-numeric columns will be ignored: {list(non_numeric)}",
                UserWarning,
            )



#
# other helpful utilities
#

def _handle_nan_values(
    values: np.ndarray,
    nan_policy: NaNPolicy = "drop",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Handle NaN values according to the specified policy.

    Parameters
    ----------
    values : np.ndarray
        Input values that may contain NaNs.
    nan_policy : {"drop", "raise", "propagate"}, default "drop"
        Policy for handling NaN values:
        - "drop": Remove NaN values and continue with valid data
        - "raise": Raise ValueError if any NaN values are found
        - "propagate": Keep NaN values (will result in NaN output)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (processed_values, original_indices)
        - processed_values: Values after applying NaN policy
        - original_indices: Original indices of the processed values
            (for temporal ordering)

    Raises
    ------
    ValueError
        When nan_policy="raise" and NaN values are present.
    """
    if nan_policy == "raise":
        if np.any(np.isnan(values)):
            nan_count = np.sum(np.isnan(values))
            raise ValueError(
                f"NaN values found in input data ({nan_count} out of "
                f"{len(values)} values). Use nan_policy='drop' to remove them "
                "or nan_policy='propagate' to keep them."
            )
        return values, np.arange(len(values))

    elif nan_policy == "propagate":
        # Keep all values including NaNs
        return values, np.arange(len(values))

    elif nan_policy == "drop":
        # Remove NaN values
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        return valid_values, valid_indices

    else:
        raise ValueError(
            f"Invalid nan_policy: '{nan_policy}'. Must be 'drop', 'raise', or "
            "'propagate'"
        )
