"""
zips.py
=======

High‑level utilities for ingesting tabular data (CSV / Parquet) that have been
packed into one or more ZIP archives.  The module exposes a single public class
`Unzipper`, whose iterator semantics make it easy to stream data out of
large or deeply nested archives while keeping memory usage predictable.

Typical use cases
-----------------
1. **Simple read** - pull every datafile into a single `pandas.DataFrame`

   >>> import motoro as mt
   >>> with mt.Unzipper("market_data.zip") as uz:
   ...     df = uz.read_all()

2. **Pattern filtering & batching** - lazily read only files that match a
   regular‑expression pattern and process them in manageable chunks

   >>> pattern = r"^2025-\\d{2}-prices\\.csv$"
   >>> with Unzipper("daily_prices.zip", pattern=pattern) as uz:
   ...     for batch in uz.iter_batches(batch_size=20):
   ...         do_something(batch)

Notes
-----
* Supports nested ZIPs transparently-if an entry inside the outer archive is
  itself a ZIP, it is opened in‑memory and iterated into.
* By default, dataframes are concatenated along the *index* axis
  (``_axis='index'``).
"""
from __future__ import annotations

from io import BytesIO
import numpy as np
import pandas as pd
import pathlib
import re
import zipfile



class Unzipper(object):
    """
    Iterator that lazily extracts tabular files from a ZIP archive.

    The class behaves like a generator of `pandas.DataFrame` objects:
    each call to `next()` (or an implicit loop) returns one dataframe
    corresponding to the *next* datafile encountered in a breadth‑first walk of
    the archive.  Files that do **not** match the supplied *pattern* or whose
    extensions are not recognised (`.csv` or `.parquet`) are skipped.

    Parameters
    ----------
    path : str or pathlib.Path
        Location of the outer ZIP archive on disk.
    reader : Callable, optional
        Custom function with signature `reader(file_like) -> DataFrame`.
        When supplied, it overrides the built‑in CSV/Parquet autodetection and
        is applied to **every** matching file.
    pattern : str or Callable[[str], bool], default `""`
        Filename filter.  If a *string*, it is treated as a regular expression
        and compiled once; if a *callable*, it must accept a filename and
        return `True` if the file should be read.
    mode : {"r", "w", "a"}, default `"r"`
        File‑mode passed straight through to `zipfile.ZipFile`.

    Attributes
    ----------
    path : pathlib.Path
        Normalized path of the outer ZIP archive.
    zf : zipfile.ZipFile
        Handle to the open archive (closed automatically in `__exit__`).
    pattern : str or Callable[[str], bool]
        Original pattern as supplied by the user.
    reader : Callable or `None`
        Custom reader if provided; otherwise resolved dynamically.
    _axis : {"index", "columns"}
        Axis along which dataframes are concatenated throughout this API.
    _recognized_datafiles : tuple[str, ...]
        Extensions regarded as tabular data.

    Raises
    ------
    TypeError
        If *path* does not appear to be a valid ZIP archive.

    Examples
    --------
    Basic iteration::

        with Unzipper("nested.zip") as uz:
            for df in uz:
                print(df.shape)

    Selecting files via a Python callable::

        def is_2025_q2(file_name: str) -> bool:
            return (
                "2025-04" <= file_name[:7] <= "2025-06"
                and file_name.endswith(".csv")
            )

        with Unzipper("archive.zip", pattern=is_2025_q2) as uz:
            quarterly = uz.read_streaming()
    """

    _recognized_datafiles = (".csv", ".parquet")
    _axis = "index"

    def __init__(
        self,
        path: str | pathlib.Path,
        *,
        reader: Optional[Callable] = None,
        pattern: Optional[Callable | str] = "",
        mode: Optional[str] = "r",
    ):
        self.path = pathlib.Path(path).expanduser().resolve()
        self.mode = mode

        if self.is_zipfile(self.path):
            self.zf = zipfile.ZipFile(self.path, self.mode)
            self.iter = iter(self.zf.namelist())
        else:
            raise TypeError(f"file {self.path} is not a valid zipfile")

        self.pattern = pattern
        self._pattern_call = (
            _regex_search_wrapper(pattern)
            if isinstance(pattern, str) else pattern
        )

        self.reader = reader

        self.skip_macos_metadata = True # will maybe change in the future

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.zf.close()

    def __iter__(self):
        return self

    def __next__(self):
        file_name = next(self.iter)
        return self._read_or_unzip(file_name, self.zf)

    #
    # public methods
    #
    def iter_batches(self, batch_size: int = 10) -> pd.DataFrame:
        """
        Yield concatenated dataframes in *batch_size* chunks.

        This is a convenient wrapper around the object’s iterator protocol that
        groups *batch_size* consecutive dataframes and concatenates them along
        `self._axis` before yielding the result.  The final batch is yielded
        even if it contains fewer than *batch_size* items.

        Parameters
        ----------
        batch_size : int, default 10
            Number of individual dataframes to combine before emitting a batch.

        Yields
        ------
        pandas.DataFrame
            Batched dataframe.
        """
        batch = []
        for df in self:
            batch.append(df)
            if len(batch) >= batch_size:
                yield pd.concat(batch, axis=self._axis)
                batch = []

        if batch:
            # last batch if needed
            yield pd.concat(batch, axis=self._axis)

    def iter_dataframes(self):
        """
        Memory‑efficient generator over *all* dataframes in the archive.

        Unlike plain iteration, this method recursively walks the namelist each
        time it yields, ensuring that nested archives are expanded lazily and
        that individual dataframes are materialised one‑by‑one.

        Yields
        ------
        pandas.DataFrame
            The next dataframe extracted from the archive hierarchy.
        """
        for file_name in self.zf.namelist():
            yield from self._read_or_unzip_generator(file_name, self.zf)

    def read_all(self) -> pd.DataFrame:
        """
        Eagerly load **every** dataframe and concatenate the result.

        Returns
        -------
        pandas.DataFrame
            Single dataframe obtained by concatenating the iterator output.
        """
        return pd.concat(self, axis=self._axis)

    def read_streaming(self) -> pd.DataFrame:
        """
        Streaming variant of `read_all`.

        Uses :meth:`iter_dataframes` under the hood to control memory
        consumption when dealing with very large archives.

        Returns
        -------
        pandas.DataFrame
            Concatenated dataframe built incrementally.
        """
        return pd.concat(self.iter_dataframes(), axis=self._axis)


    #
    # private and helper methods
    #
    def _is_macos_metadata(self, file_name: str) -> bool:
        """
        Check if file is macOS metadata that should be skipped
        """
        if not self.skip_macos_metadata:
            return False

        macos_patterns = [
            "__MACOSX/", # resource fork dir
            ".DS_Store", # finder metadata
            "._.DS_Store", # AppleDouble DS_Store
            "._", # AppleDouble files (resource forks)
            ".fseventsd/", # file systems events
            ".Spotlight-V100/", # spotlight index
            ".Trashes/", # trash folder
        ]
        return any(
            file_name.startswith(pat) or file_name.endswith(pat)
            for pat in macos_patterns
        )

    def _matches_pattern(self, file_name: str) -> bool:
        """
        Decide whether *file_name* satisfies the user‑supplied *pattern*.

        For string patterns the check is performed with
        `re.Pattern.search`; for callable patterns it defers to the
        user’s predicate.

        Parameters
        ----------
        file_name : str
            Filename (with path) to test.

        Returns
        -------
        bool
            `True` if the pattern is empty or matches, `False` otherwise.
        """
        if self._is_macos_metadata(file_name):
            return False

        if self.pattern:
            return self._pattern_call(file_name)
        else:
            return True

    def _read_or_unzip(
        self,
        file_name: str,
        outer: zipfile.ZipFile,
    ):
        """
        Recursively extract *file_name* from *outer* and return its
        contents as a `pandas.DataFrame`.

        The method walks the archive hierarchy depth‑first:

        1. If *file_name* itself is a ZIP, it is opened in memory and the
           first entry encountered is processed recursively.
        2. Otherwise the file is returned via `_read_datafile`-but only
           if both `is_datafile` **and** the user‑supplied pattern check
           (`_matches_pattern`) succeed.
        3. Files that fail either test yield an **empty** dataframe so that
           downstream concatenation logic remains simple.

        Parameters
        ----------
        file_name : str
            Path (relative to *outer*) of the member to read.
        outer : zipfile.ZipFile
            Open handle to the current archive layer.

        Returns
        -------
        pandas.DataFrame
            Parsed data, or an empty dataframe when the entry is skipped.

        Notes
        -----
        *No* warnings are issued for skipped files
        """
        if self._is_macos_metadata(file_name):
            return pd.DataFrame()

        file_path = self.path / file_name
        if self.is_zipfile(file_path):
            inner_bytes = outer.read(file_name)
            inner_buffer = BytesIO(inner_bytes)

            inner = zipfile.ZipFile(inner_buffer, self.mode)
            for name in inner.namelist():
                return self._read_or_unzip(name, inner)
        else:
            if self.is_datafile(file_name) and self._matches_pattern(file_name):
                return self._read_datafile(file_name, outer)
            else:
                # should probably issue a warning here?
                return pd.DataFrame()

    def _read_or_unzip_generator(
        self,
        file_name: str,
        outer: zipfile.ZipFile,
    ):
        """
        Generator variant of `_read_or_unzip`.

        Yields one dataframe at a time instead of returning a single object,
        making it suitable for **streaming** very large or highly nested
        archives.

        Parameters
        ----------
        file_name : str
            Entry within *outer* to be processed.
        outer : zipfile.ZipFile
            Open ZIP handle.

        Yields
        ------
        pandas.DataFrame
            The next dataframe extracted from the archive tree.
        """
        if self._is_macos_metadata(file_name):
            return

        file_path = self.path / file_name
        if self.is_zipfile(file_path):
            inner_bytes = outer.read(file_name)
            inner_buffer = BytesIO(inner_bytes)

            inner = zipfile.ZipFile(inner_buffer, self.mode)
            for name in inner.namelist():
                yield from self._read_or_unzip_generator(name, inner)
        else:
            if self.is_datafile(file_name) and self._matches_pattern(file_name):
                df = self._read_datafile(file_name, outer)
                yield df

    def _read_datafile(
        self,
        file_name: str,
        zf: zipfile.ZipFile,
    ):
        """
        Read a single **CSV** or **Parquet** member from *zf*.

        Parameters
        ----------
        file_name : str
            Name of the archive entry.
        zf : zipfile.ZipFile
            Open archive handle (could be an outer or inner ZIP).

        Returns
        -------
        pandas.DataFrame
            Parsed dataframe produced either by the custom *reader* supplied
            at construction time or by the built‑in CSV/Parquet readers.

        Raises
        ------
        NotImplementedError
            If the file extension is not `.csv` or `.parquet` and no custom
            reader has been provided.
        """
        file_bytes = zf.read(file_name)
        file_buffer = BytesIO(file_bytes)

        if self.reader:
            return self.reader(file_buffer)

        lowercase_name = file_name.lower()
        file_type = lowercase_name.split(".")[-1]
        if file_type == "csv":
            reader = pd.read_csv
        elif file_type == "parquet":
            reader = pd.read_parquet
        else:
            raise NotImplementedError(
                f"default reader for '{file_type}' type-files not implemented"
            )
        return reader(file_buffer)

    @classmethod
    def is_datafile(cls, obj: Any) -> bool:
        """
        Heuristic test for *tabular* datafiles (.csv or .parquet).

        Parameters
        ----------
        obj : str or pathlib.Path or Any
            Object to test.  If the instance does not implement `str()`,
            the function returns `False`.

        Returns
        -------
        bool
            `True` if *obj* ends with a recognised data extension,
            `False` otherwise.:contentReference[oaicite:7]{index=7}
        """
        if isinstance(obj, str):
            return obj.lower().endswith(cls._recognized_datafiles)
        try:
            obj_str = str(obj)
            return obj_str.lower().endswith(cls._recognized_datafiles)
        except Exception:
            return False

    @classmethod
    def is_zipfile(cls, obj: Any) -> bool:
        """
        Lightweight check for *ZIP* archives.

        This method first defers to :func:`zipfile.is_zipfile`.  If that returns
        `False`-e.g. because *obj* is a path string rather than a file
        handle-it falls back to a simple “*.zip” suffix comparison.

        Parameters
        ----------
        obj : str or pathlib.Path or Any
            Object to test.

        Returns
        -------
        bool
            `True` if *obj* appears to reference a ZIP file,
            `False` otherwise.
        """
        if zipfile.is_zipfile(obj):
            return True
        if isinstance(obj, str):
            return obj.lower().endswith(".zip")
        try:
            obj_str = str(obj)
            return obj_str.lower().endswith(".zip")
        except Exception:
            return False


def _regex_search_wrapper(pattern: str):
    """
    Compile *pattern* once and return a search‑based predicate.

    Notes
    -----
    This helper is intentionally **private**; it is returned by the constructor
    when the user provides a *str* pattern so that subsequent filename checks
    avoid recompiling the regex.
    """
    prog = re.compile(pattern)
    return lambda string: prog.search(string)
