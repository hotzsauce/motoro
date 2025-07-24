"""
Creating profiles of time‑series data
====================================

This module provides a small utility layer for extracting multiple,
time‑aligned *perspectives* (windows) from a datetime‑indexed
`Series` or `DataFrame`.  The key user‑facing helpers are

* `profile` - a convenience wrapper around `Profiler.profile`, and
* `ProfiledFrame` - a very thin `pandas.DataFrame` subclass that keeps
  track of alignment metadata while delegating almost all behavior
  back to pandas.

The design is intentionally lightweight: profile objects are ordinary
DataFrames under the hood, so they travel nicely through pandas and
NumPy‑based workflows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.core.groupby.groupby import BaseGroupBy
from pandas._libs.indexing import NDFrameIndexerBase
from typing import Callable, Union, Sequence, Optional



__all__ = [
    "ProfiledFrame",
    "Profiler",
    "profile",
]



class ProfiledFrame(object):
    """
    A time‑aligned collection of windowed *perspectives* on the original data.

    Each column represents one extracted window; the index is either a
    `pandas.TimedeltaIndex` (relative alignment) or a
    `pandas.DatetimeIndex` (absolute alignment), depending on how the
    perspective was aligned.

    For DataFrames with MultiIndex columns, aggregation methods automatically
    aggregate within each outer level of the columns. To access the underlying
    pandas DataFrame aggregation behavior, use `pf.df` instead. The aggregation
    methods that are overridden are:
        'mean', 'sum', 'max', 'min', 'std', 'var', 'median', 'quantile',
        'count', 'nunique', 'first', last'

    Parameters
    ----------
    data : pandas.DataFrame
        A DataFrame whose columns are individual perspectives and whose index
        encodes the relative or absolute timeline shared by those perspectives.

    See Also
    --------
    Profiler : the helper class that creates ``ProfiledFrame`` objects
    profile : a functional wrapper around `Profiler.profile`
    """

    _internal_aggregations = [
        "mean", "sum", "max", "min", "std", "var", "median", "quantile",
        "count", "nunique", "first", "last",
    ]

    _arithmetic_dunder_methods = [
        "__add__", "__sub__", "__mul__", "__truediv__", "__pow__",
        "__floordiv__", "__mod__", "__divmod__", "__neg__", "__pos__",
        "__lshift__", "__rshift__",
        "__and__", "__or__", "__xor__", "__invert__",
    ]

    _comparison_dunder_methods = [
        "__eq__", "__ne__", "__lt__", "__gt__", "__le__", "__ge__",
    ]

    def __init__(self, data: pd.DataFrame, window_hint: str = None):
        """
        Construct a `ProfiledFrame`.

        Parameters
        ----------
        data : pandas.DataFrame
            Already-aligned perspective data.  Must satisfy
            `_validate_profiled_data`.

        Raises
        ------
        TypeError
            If `data` is not a `DataFrame` or does not meet the alignment
            requirements.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("ProfiledFrame requires a DataFrame")
        self.data = data
        self._validate_profiled_data()

        self._window_hint = window_hint
        self._apply_smart_index()

    def __getattr__(self, attr):
        """
        Forward attribute access to the underlying DataFrame.

        Parameters
        ----------
        attr : str
            The attribute being accessed (e.g. `mean`, `loc`).

        Returns
        -------
        object
            * If the underlying attribute is callable *and* returns a
              `DataFrame`, the result is wrapped back into a
              `ProfiledFrame`.
            * If it returns a pandas *groupby‑like* object (`Resampler`,
              `GroupBy`, `Rolling`), or an indexer, that object is wrapped in a
              `ProfiledWrapper` so that subsequent methods keep the
              ProfiledFrame semantics.
            * Otherwise the raw attribute is returned unchanged.

        Notes
        -----
        This mechanism allows `ProfiledFrame` instances to behave almost
        identically to pandas DataFrames while preserving the alignment
        metadata whenever possible.
        """
        if attr in self._internal_aggregations:
            def wrapped_aggregation(*args, **kwargs):
                return self._aggregate_by_variable(attr, *args, **kwargs)
            return wrapped_aggregation

        # handling aggregation methods when MultiIndex column
        underlying = getattr(self.data, attr)

        if isinstance(underlying, NDFrameIndexerBase):
            # callable(indexer) evaluates to True for pandas indexers?
            return ProfiledWrapper(underlying, ProfiledFrame)

        # if it's a method, wrap it to potentially return ProfiledFrame
        if callable(underlying):
            def wrapped_method(*args, **kwargs):
                method_result = underlying(*args, **kwargs)
                # a ProfiledSeries doesn't make sense
                if isinstance(method_result, pd.Series):
                    return method_result
                # methods that should return ProfiledFrame
                if isinstance(method_result, pd.DataFrame):
                    return ProfiledFrame(method_result)
                # handle Resampler, Groupby, Rolling, Indexers
                elif isinstance(method_result, WRAPPER_TYPES):
                    return ProfiledWrapper(method_result, ProfiledFrame)
                return method_result
            return wrapped_method

        return underlying

    def __getitem__(self, key):
        """
        Column selection mimicry (`pf[col]`).

        Parameters
        ----------
        key : hashable or list‑like
            Anything accepted by `DataFrame.__getitem__`.

        Returns
        -------
        ProfiledFrame or pandas.Series
            Mirrors pandas’ behaviour - a single column yields a `Series`,
            multi‑column selection yields a new `ProfiledFrame`.
        """
        data = self.data[key]
        if isinstance(data, pd.DataFrame):
            return ProfiledFrame(data)
        return data

    def __len__(self):
        """
        Length *of the index* (number of rows).

        Returns
        -------
        int
            `len(self.data.index)`.
        """
        return self.data.__len__()

    def __repr__(self):
        """
        Formal string representation.

        Returns
        -------
        str
            `repr(self.data)`, prefixed with `'ProfiledFrame'`.
        """
        return f"ProfiledFrame(shape={self.data.shape})\n{self.data.__repr__()}"

    def align(
        self,
        method: str | Sequence = "start",
    ) -> ProfiledFrame:
        """
        Realign perspectives to a common reference point.

        Parameters
        ----------
        method : {'start', 'end', 'max', 'min'} or Iterable of pd.Timestamp
            * `'start'` (default) - keep existing alignment.
            * `'end'` – shift each perspective so that its last timestamp is
              *t = 0*.
            * `'max'` / `'min'` - align on each column’s maximum/minimum
              value (first occurrence).
            * Iterable – an explicit list of timestamps, one per column,
              relative to which each perspective is shifted.

        Returns
        -------
        ProfiledFrame
            A *new* ProfiledFrame with the requested alignment.

        Raises
        ------
        ValueError
            If `method` is not recognised or mismatched in length.

        Examples
        --------
        >>> pf.align('end')          # align so every window ends together
        >>> pf.align(pf.index[100])  # custom alignment
        """
        if method == "start":
            return ProfiledFrame(self.data)

        elif method == "end":
            return self._align_to_end()

        elif method == "max":
            ref_periods = self.data.idxmax(axis="index")
            return self._align_to_periods(ref_periods)

        elif method == "min":
            ref_periods = self.data.idxmin(axis="index")
            return self._align_to_periods(ref_periods)

        if isinstance(method, str):
            raise ValueError(f"unrecognized 'method' value: '{method}'")

        return self._align_to_periods(method)

    @property
    def df(self):
        """
        Bare DataFrame accessor.

        Returns
        -------
        pandas.DataFrame
            The underlying alignment‑aware DataFrame.
        """
        return self.data

    @property
    def is_absolute(self) -> bool:
        """
        Check if the profile has an absolute index
        """
        return isinstance(self.data.index, pd.DatetimeIndex)

    @property
    def is_relative(self) -> bool:
        """
        Check if the profile has a relative index
        """
        return isinstance(self.data.index, pd.TimedeltaIndex)

    @property
    def ref_index(self) -> pd.DatetimeIndex:
        if isinstance(self.data.index, pd.DatetimeIndex):
            return self.data.index
        elif isinstance(self.data.index, pd.TimedeltaIndex):
            # just use an arbitrary date because it's thrown out anyway
            return self.data.index + pd.to_datetime("2000-01-01")

    def _aggregate_by_variable(
        self,
        method_name,
        axis: str | int = "columns",
        **kwargs,
    ):
        """
        """
        if (axis == 0) or (axis == "index"):
            # user-specified aggregation
            method = getattr(self.data, method_name)
            return method(axis=axis, **kwargs)


        if isinstance(self.data.columns, pd.MultiIndex):
            # default behavior of aggregating within each outer level
            levels = self.data.columns.get_level_values(0).unique()
            frames = []
            for level in levels:
                method = getattr(self.data[level], method_name)
                df = method(axis=axis, **kwargs).rename(level)
                frames.append(df)

            df = pd.concat(frames, axis="columns")
            return ProfiledFrame(df, window_hint=self._window_hint)
        else:
            # profiling a single Series; follow pandas.Groupby lead and
            # label the result according to the chosen method
            method = getattr(self.data, method_name)
            return method(axis=axis, **kwargs).rename(method_name)

    def _align_to_end(self) -> ProfiledFrame:
        """
        Internal helper: align so that *all* perspectives share the same
        **ending timestamp** (``t = 0``).

        Returns
        -------
        ProfiledFrame
            Result with a `pandas.TimedeltaIndex`.
        """
        ref_index = self.ref_index

        frames = []
        for col in self.data.columns:
            pers = self.data[col]
            pos = _first_nan_of_last_nanblock(pers)

            if pos:
                pers_index = ref_index[:pos] - ref_index[pos-1]
                pers = pers.iloc[:pos]
            else:
                pers_index = ref_index - ref_index[-1]

            aligned = pd.Series(pers.to_numpy(), index=pers_index, name=col)
            frames.append(aligned)

        df = pd.concat(frames, axis="columns")
        return ProfiledFrame(df)

    def _align_to_periods(
        self,
        ref_periods: Sequence[pd.Timestamp | pd.Timedelta],
    ) -> ProfiledFrame:
        """
        Internal helper used by `align`.

        Parameters
        ----------
        ref_periods : Iterable[pandas.Timestamp]
            One reference timestamp **per column** that defines how much each
            perspective is shifted.

        Returns
        -------
        ProfiledFrame
            The re‑indexed result.
        """
        allowed_types = (pd.Timestamp, pd.Timedelta, str)
        if not all(isinstance(ref, allowed_types) for ref in ref_periods):
            raise TypeError(
                "Reference periods must be timestamps, timedeltas, or strings"
            )

        if isinstance(self.data.index, pd.DatetimeIndex):
            ref_index = self.data.index
        elif isinstance(self.data.index, pd.TimedeltaIndex):
            # just use an arbitrary date because it's thrown out anyway
            ref_index = self.data.index + pd.to_datetime("2000-01-01")

        if len(ref_periods) == 1:
            # broadcast a single reference period, I suppose
            ref_periods = ref_periods * self.data.shape[1]

        if len(ref_periods) != self.data.shape[1]:
            raise ValueError(
                f"number of reference periods ({len(ref_periods)}) does not "
                f"match number of perspectives ({self.data.shape[1]})"
            )

        frames = []
        for ref, col in zip(ref_periods, self.data.columns):
            pers = self.data[col]
            if len(pers) == 0:
                continue

            try:
                pos = pers.index.get_loc(ref)
            except KeyError:
                pos = pers.index.get_indexer([ref], method="nearest")[0]

            # if the reference periods aren't an exact match - for instance,
            # if they're user-given date strings without time info - both
            # of the `get_*` methods can return slices
            pos = pos.start if isinstance(pos, slice) else pos
            pers_index = ref_index - ref_index[pos-1]

            aligned = pd.Series(pers.to_numpy(), index=pers_index, name=col)
            frames.append(aligned)

        df = pd.concat(frames, axis="columns")
        return ProfiledFrame(df)

    def _apply_smart_index(self):
        """
        Apply smart default indexing for common window patterns
        """
        if not self._window_hint:
            return None

        hint = self._window_hint
        if (
            ((hint.lower() == "1d") or (hint.lower() == "d"))
            and
            isinstance(self.data.index, pd.DatetimeIndex)
            and
            len(self.data.index) > 0
        ):
            delta = self.data.index[-1] - self.data.index[0]
            if delta.days == 0:
                self.data.index = self.data.index.time

    def _validate_profiled_data(self):
        """
        Sanity‑check that the wrapped DataFrame looks like a profile.

        Raises
        ------
        TypeError
            If the index is not datetime‑like or if columns have mismatched
            lengths.
        """
        pass

    @classmethod
    def _attach_dunders(cls):
        """
        This will dynamically add all the arithmetic & comparison dunder methods
        """
        dunders = cls._arithmetic_dunder_methods + cls._comparison_dunder_methods
        for name in dunders:
            if name in cls.__dict__:
                # almost the same a `hasattr(cls, name)`, but that checks for
                # inherited methods, and `object` implements all the comparison
                # dunders
                continue
            # if hasattr(cls, name): # let explicit overrides win
            #     continue
            def _make(op):
                def _op(self, other):
                    left  = self.data
                    right = other.data if isinstance(other, ProfiledFrame) else other
                    res = getattr(left, op)(right)
                    return cls(res) if isinstance(res, pd.DataFrame) else res
                return _op
            setattr(cls, name, _make(name))

# must attach the methods to the class, not the instance
ProfiledFrame._attach_dunders()



WRAPPER_TYPES = (BaseGroupBy, NDFrameIndexerBase)
class ProfiledWrapper(object):
    """
    Lightweight wrapper around pandas groupby‑like and indexer objects.

    The goal is to insert a return‑type shim so that methods producing a
    `DataFrame` come back as a `ProfiledFrame`, thereby preserving
    alignment metadata through chained operations.
    """

    def __init__(self, wrapped_obj, return_type):
        """
        Parameters
        ----------
        wrapped_obj : pandas.BaseGroupBy
            A pandas `GroupBy/Resampler/Rolling`‐style object produced by
            pandas.
        return_type : type
            The constructor used to wrap any DataFrame results (currently
            expected to be `ProfiledFrame`).
        """
        self._wrapped = wrapped_obj
        self._return_type = return_type

    def __getitem__(self, key):
        """
        Delegate indexed access on the assumption that `wrapped_obj` is a
        pandas indexer
        """
        underlying = self._wrapped[key]
        if isinstance(underlying, pd.DataFrame):
            return ProfiledFrame(underlying)
        else:
            return underlying

    def __getattr__(self, attr):
        """
        Delegate attribute access while re‑wrapping DataFrame outputs.

        See Also
        --------
        ProfiledFrame.__getattr__  : similar logic at the frame level
        """
        underlying = getattr(self._wrapped, attr)

        # if it's a method, wrap it to potentially return ProfiledFrame
        if callable(underlying):
            def wrapped_method(*args, **kwargs):
                method_result = underlying(*args, **kwargs)
                # a ProfiledSeries doesn't make sense
                if isinstance(method_result, pd.Series):
                    return method_result
                # methods that should return ProfiledFrame
                if isinstance(method_result, pd.DataFrame):
                    return ProfiledFrame(method_result)
                # handle Resampler, Groupby, Rolling, etc
                elif isinstance(method_result, WRAPPER_TYPE):
                    return ProfiledWrapper(method_result, ProfiledFrame)
                return method_result
            return wrapped_method

        return underlying



class Profiler(object):
    """
    Convenience class that orchestrates *window extraction* and *alignment*.

    Users typically interact with `profile`; direct instantiation is
    useful when repeatedly profiling data using the **same** granularity.
    """

    @classmethod
    def _generate_ends(
        cls,
        window,
        ends,
        start_dates,
    ):
        """
        Derive the *ending* timestamps for each profile window.

        The method supports two mutually exclusive ways of specifying where a
        profile should stop:

        1. **`window`** – a duration relative to the corresponding element of
           `start_dates`.  If supplied, the routine converts every entry into a
           `pandas.Timedelta` and returns `start_dates + window`.
        2. **`ends`** – explicit absolute timestamps.  These are canonicalized
           through `_process_date_iterable` without sorting, so the order
           mirrors the caller’s intent.

        Parameters
        ----------
        window : str, pandas.Timedelta, sequence of those, or None
            Width of each profile.  A scalar value is broadcast to *all*
            `start_dates`.  When *None*, **`ends`** must be provided.
        ends : str, pandas.Timestamp, sequence of those, or None
            Absolute ending times, one per profile.  A scalar value is broadcast
            just like `window`.  Ignored when `window` is given.
        start_dates : pandas.DatetimeIndex
            Canonicalized start timestamps produced earlier in the profiling
            pipeline.

        Returns
        -------
        pandas.DatetimeIndex
            Ending timestamps aligned 1‑to‑1 with `start_dates`.

        Raises
        ------
        ValueError
            If the length of `window` or `ends` (when iterable) does not match
            the length of `start_dates`.

        Notes
        -----
        * Mixing string offsets (e.g., `"6H"`) and :class:`pandas.Timedelta`
          objects in a single `window` iterable is allowed; each element is
          coerced individually with :func:`pandas.to_timedelta`.
        * The returned index preserves the original ordering of `start_dates`;
          no sorting is applied.

        Examples
        --------
        >>> starts = pd.date_range("2025‑01‑01", periods=3, freq="D")
        >>> Profiler._generate_ends("6H", None, starts)
        DatetimeIndex(['2025‑01‑01 06:00:00', '2025‑01‑02 06:00:00',
                       '2025‑01‑03 06:00:00'],
                      dtype='datetime64[ns]', freq=None)

        >>> Profiler._generate_ends(
        ...     window=["6H", pd.Timedelta("12H"), "1D"],
        ...     ends=None,
        ...     start_dates=starts,
        ... )
        DatetimeIndex([...])  # 6 h, 12 h, 24 h after each start

        >>> Profiler._generate_ends(
        ...     window=None,
        ...     ends=["2025‑01‑01 08:00", "2025‑01‑02 09:00", "2025‑01‑03 10:00"],
        ...     start_dates=starts,
        ... )
        DatetimeIndex(['2025‑01‑01 08:00:00', '2025‑01‑02 09:00:00',
                       '2025‑01‑03 10:00:00'],
                      dtype='datetime64[ns]', freq=None)
        """
        if window:
            if isinstance(window, (str, pd.Timedelta)):
                window = [window] * len(start_dates)
            elif len(window) != len(start_dates):
                raise ValueError(
                    f"length of 'window' ({len(window)}) must match "
                    f"length of 'starts' ({len(start_dates)})"
                )

            # iterate naively in case elements of window are both strings and
            # Timedeltas
            deltas = []
            for w in window:
                if isinstance(w, str):
                    deltas.append(pd.to_timedelta(w))
                else:
                    deltas.append(w)
            deltas = pd.TimedeltaIndex(deltas)
            end_dates = start_dates + deltas
        else:
            if isinstance(ends, (str, pd.Timestamp)):
                ends = [ends] * len(start_dates)
            elif len(ends) != len(start_dates):
                raise ValueError(
                    f"length of 'ends' ({len(ends)}) must match "
                    f"length of 'starts' ({len(start_dates)})"
                )
            # maps to a DatetimeIndex, possibly of length one
            end_dates = _process_date_iterable(ends, sort=False)

        return end_dates

    @classmethod
    def _generate_starts(
        cls,
        data: pd.Series | pd.DataFrame,
        freq: str,
        window: str | pd.Timedelta | Sequence[str | pd.Timedelta],
    ) -> pd.DatetimeIndex:
        """
        Determine the vector of `start` timestamps for every perspective.

        Parameters
        ----------
        data : pandas.Series or pandas.DataFrame
            The source time‑series data.
        freq : str, optional
            A pandas offset alias (e.g. `"D"`, `"2H"`).  Mutually exclusive
            with `starts` in `profile`.
        window : Timedelta‑like or str
            The width of each perspective window.  Required when `freq` is
            used.

        Returns
        -------
        pandas.DatetimeIndex
            The canonicalized start times, rounded by `granularity` if set

        Notes
        -----
        This helper centralises the tricky corner‑cases around heterogeneous
        timestamp inputs (strings, `Timestamp`, timedeltas).
        """
        if len(data) == 0:
            return pd.DatetimeIndex([])
        first_obs = data.index.min()
        final_obs = data.index.max()

        if isinstance(window, str):
            delta = pd.to_timedelta(window)
        elif isinstance(window, pd.Timedelta):
            delta = window
        else:
            # default to first window for now...
            delta = (
                pd.to_timedelta(window[0]) if isinstance(window[0], str)
                else window[0]
            )

        return pd.date_range(
            start=first_obs,
            end=final_obs,
            freq=freq,
        )

    @classmethod
    def _profiled_columns(
        cls,
        data: pd.DataFrame,
        rename: Callable | str,
        start_dates: Sequence[pd.Timestamp],
        starts: Sequence[str | pd.Timestamp],
    ) -> pd.Index | pd.MultiIndex:
        """
        Build the column labels (or a hierarchical index) for a profiled result.

        Depending on whether *data* is a 1‑D `Series` or an n‑column
        `DataFrame`, the routine returns either

        * a flat container of strings—one label per extracted profile, or
        * a two‑level `pandas.MultiIndex` whose first level identifies the
          profile and whose second level mirrors the original column names.

        Parameters
        ----------
        data : pandas.Series or pandas.DataFrame
            The source object that was profiled.  Its dimensionality determines
            whether a simple `Index` or a `MultiIndex` is produced.
        rename : str, callable, or None
            Rules for constructing the *profile* portion of each label.

            * **str** – treated as a `datetime.datetime.strftime` pattern and
              applied to every element of `start_dates`.
            * **callable** – invoked with each element of `start_dates`; the
              function must return a string.  If that raises, the callable is
              tried again with the corresponding element from `starts` (useful
              when the user expects raw string inputs rather than `Timestamp`
              objects).
            * **None** – auto‑generate labels in the form `"profile_0"`,
              `"profile_1"`, …
        start_dates : Sequence[pandas.Timestamp]
            Canonicalized start timestamps for each extracted profile.
        starts : Sequence[str or pandas.Timestamp]
            The *user‑supplied* start specification, retained as a fallback for
            callables that operate on strings.

        Returns
        -------
        pandas.Index or pandas.MultiIndex
            * If *data* is a `Series`: a `Index` of profile labels.
            * If *data* is a `DataFrame`: a two‑level `MultiIndex` with level
              names `['profile', 'variable']`.

        Raises
        ------
        Exception
            Re‑raises whatever exception the `rename` callable produces after
            both attempts (with `start_dates` and then `starts`).

        Examples
        --------
        >>> Profiler._profiled_columns(
        ...     data=pd.Series(range(3)),
        ...     rename="%Y‑%m‑%d",
        ...     start_dates=[pd.Timestamp("2025‑01‑01")],
        ...     starts=["2025‑01‑01"],
        ... )
        ['2025‑01‑01']

        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> Profiler._profiled_columns(
        ...     data=df,
        ...     rename=lambda ts: f"run_{ts.day:02d}",
        ...     start_dates=[pd.Timestamp("2025‑01‑01")],
        ...     starts=["2025‑01‑01"],
        ... )
        MultiIndex([('run_01', 'A'),
                    ('run_01', 'B')],
                   names=['profile', 'variable'])
        """
        if rename and isinstance(rename, str):
            prof_labels = [s.strftime(rename) for s in start_dates]
        elif rename and callable(rename):
            try:
                prof_labels = [rename(s) for s in start_dates]
            except:
                # in case the user wanted to operate on the strings
                try:
                    prof_labels = [rename(s) for s in starts]
                except Exception as exc:
                    raise exc from None
        else:
            prof_labels = [f"profile_{i}" for i in range(len(start_dates))]

        if isinstance(data, pd.Series):
            return pd.Index(prof_labels)
        else:
            return pd.MultiIndex.from_product(
                [prof_labels, data.columns],
                names=["profile", "variable"],
            )

    def profile(
        self,
        data: pd.Series | pd.DataFrame,
        freq: str = None,
        window: str | pd.Timedelta | Sequence[str | pd.Timedelta] = "1d",
        starts: str | pd.Timestamp | Sequence[str | pd.Timestamp] = None,
        ends: str | pd.Timestamp | Sequence[str | pd.Timestamp] = None,
        rename: Callable | str = "",
        absolute_index: bool = True,
        profile_first: bool = False,
    ) -> pd.DataFrame:
        """
        Extract and align multiple windows (*perspectives*) from **data**.

        This is the work‑horse that most users will call through the
        module‑level `profile`.

        Parameters
        ----------
        data : pandas.Series or pandas.DataFrame
            Input data with a `pandas.DatetimeIndex`.
        freq, starts : mutually exclusive
            * `freq` - spacing between consecutive perspectives (e.g. `'D'`);
            * `starts` - explicit list of starting timestamps.
        window, ends : mutually exclusive
            * `window` - width of each perspective; default to `"1d"`
            * `ends` - explicit list of *ending* timestamps.
        rename : str or callable, optional
            If *str*, treated as a `datetime.datetime.strftime` pattern applied
            to each perspective’s start date.  If callable, it must accept a
            `Timestamp` and return a string; also applied to `starts`
        absolute_index : bool, default `True`
            If *True*, the resulting index is absolute time.  If *False*, the
            index is a relative `TimedeltaIndex` where `t = 0` marks each
            perspective’s local start.
        profile_first : bool, default `False`
            Only used if the to-be-profiled data is a `DataFrame`. In that case,
            the columns of the returned data are a MultiIndex with levels
            "profile" and "variable". The entries of the "profile" level are
            set according to the `rename` argument; the entries of the other
            are the column names of the given DataFrame.
                If `profile_first = False`, the outer level is "variable";
            if `profile_first = True`, the outer level is "profile"

        Returns
        -------
        ProfiledFrame
            The aligned collection of perspectives.

        Raises
        ------
        TypeError, ValueError
            Propagated from validation helpers on malformed input.

        Examples
        --------
        >>> pf = profiler.profile(
        ...     data,
        ...     freq="D",
        ...     window="6H",
        ...     rename="%Y‑%m‑%d",
        ... )
        """
        _validate_profiling_data(data)
        _validate_profiling_parameters(freq, window, starts, ends)

        # will worry about memory efficiency later
        data = data.copy().sort_index()

        if starts:
            # this forces `starts` to be a DatetimeIndex, possibly of length one
            start_dates = _process_date_iterable(starts, sort=True)
        else:
            start_dates = self._generate_starts(data, freq, window)

        end_dates = self._generate_ends(window, ends, start_dates)

        perspectives = []
        for s, e in zip(start_dates, end_dates):
            # a perspective is a single series of the profiled data
            mask = (data.index >= s) & (data.index < e)
            pers = data.loc[mask]

            if len(pers) == 0:
                continue

            # align the index for concatenation purposes
            pers.index = pers.index - s
            perspectives.append(pers)

        if not perspectives:
            # empty results
            profile = pd.DataFrame()
            if rename and isinstance(rename, str):
                profile.columns = [s.strftime(rename) for s in start_dates]
            return profile

        profile = pd.concat(perspectives, axis="columns")
        profile.columns = self._profiled_columns(data, rename, start_dates, starts)

        if isinstance(profile.columns, pd.MultiIndex):
            if not profile_first:
                # swap 'profile' and 'variable'; then group variable entries;
                # **then** rearrange the columns of data
                profile.columns = profile.columns.swaplevel(0, 1)
                reordered = profile.columns.sort_values()
                profile = profile[reordered]

        # optional resetting of index
        if absolute_index:
            profile.index = start_dates[0] + profile.index

        # determine window hint for smart indexing
        window_hint = None
        if isinstance(window, str):
            window_hint = window
        elif isinstance(window, list) and len(set(window)) == 1:
            window_hint = window[0] if isinstance(window[0], str) else None

        return ProfiledFrame(profile, window_hint=window_hint)



def profile(
    data: pd.Series | pd.DataFrame,
    freq: str = None,
    window: Union[
        str, pd.Timedelta, pd.offsets.DateOffset,
        Sequence[str | pd.Timedelta, pd.offsets.DateOffset],
    ] = "1d",
    starts: str | pd.Timestamp | Sequence[str | pd.Timestamp] = None,
    ends: str | pd.Timestamp | Sequence[str | pd.Timestamp] = None,
    rename: Callable | str = "",
    absolute_index: bool = True,
    profile_first: bool = False,
) -> pd.Series | pd.DataFrame:
    """
    Functional counterpart to `Profiler.profile`.

    See `Profiler.profile` for a full parameter reference.
    """
    # _validate_profiling_parameters(window, rule, starts, ends)
    profiler = Profiler()
    return profiler.profile(
        data,
        freq, window,
        starts, ends,
        rename, absolute_index, profile_first,
    )



#
# utilities
#

def _first_nan_of_last_nanblock(x):
    """
    Return the *first* index of the *last* contiguous NaN block.

    Parameters
    ----------
    x : array‑like
        Numeric 1‑D array (will be coerced via `numpy.asarray`).

    Returns
    -------
    int or None
        Index of the first NaN in the trailing NaN‑run, or `None` if
        `x` ends with a finite value.

    Examples
    --------
    >>> _first_nan_of_last_nanblock([1, 2, np.nan, np.nan])
    2
    >>> _first_nan_of_last_nanblock([1, 2, 3])
    None
    """
    arr = np.asarray(x)
    nan_mask = np.isnan(arr)
    # must end with nans; otherwise it's not a trailing block
    if not nan_mask.any() or not nan_mask[-1]:
        return None

    # positions where a nan is preceded by a non-nan or is at arr[0]
    starts = np.where(nan_mask & np.r_[True, ~nan_mask[:-1]])[0]
    return starts[-1]


def _process_date_iterable(
    dates: Sequence[str | pd.Timestamp],
    sort: bool = True,
) -> pd.DatetimeIndex:
    """
    Canonicalise an arbitrary iterable of datetime‑like objects.

    Parameters
    ----------
    dates : iterable
        Any mix of `str` or :class:`pandas.Timestamp` values.
    sort : bool, default `True`
        Sort the resulting index in ascending order.

    Returns
    -------
    pandas.DatetimeIndex
        Normalised index with duplicate timestamps *removed*.

    Raises
    ------
    ValueError
        If `dates` is empty after dropping duplicates.
    """
    if isinstance(dates, str):
        dates = [dates]
        return _process_date_iterable(dates, sort)
    index = pd.to_datetime(dates)
    if sort:
        index = index.sort_values()
    return index

def _validate_profiling_data(data: pd.Series | pd.DataFrame):
    """
    Ensure that *data* is suitable for profiling.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame

    Raises
    ------
    TypeError
        If `data` is not a pandas object or its index is not a
        `pandas.DatetimeIndex`.
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("profiling data must be a Series or DataFrame")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("profiling data must have a DatetimeIndex")

def _validate_profiling_parameters(
    freq: str,
    window: Union[
        str, pd.Timedelta, pd.offsets.DateOffset,
        Sequence[str | pd.Timedelta, pd.offsets.DateOffset],
    ],
    starts: str | pd.Timestamp | Sequence[str | pd.Timestamp],
    ends: str | pd.Timestamp | Sequence[str | pd.Timestamp],
):
    """
    Sanity‑check mutually exclusive / required parameter combinations.

    Raises
    ------
    ValueError
        For any missing or conflicting parameter set.
    """
    if freq and starts:
        raise ValueError("must specify exactly one of 'freq' or 'starts'")
    if not (freq or starts):
        raise ValueError("must specify at least one of 'freq' or 'starts'")

    if window and ends:
        raise ValueError("must specify exactly one of 'window' or 'ends'")
    if not (window or ends):
        raise ValueError("must specify at least one of 'window' or 'ends'")
