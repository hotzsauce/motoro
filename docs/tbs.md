# Top-Bottom Spreads (`tbs.py`)

A high-performance module for calculating Top-Bottom (TB) spreads on time-series
price data, with specialized support for sub-hourly data.

## Overview

A **Top-Bottom spread** measures the difference between the highest and lowest
price periods within a given time window (typically daily). This module computes
TB spreads as:

```
TB Spread = sum(top n hours of prices) - sum(bottom n hours of prices)
```

The module is designed for high-frequency calculations (100,000+ calls) using
vectorized NumPy operations and supports both hourly and sub-hourly data with
flexible grouping options.

## Key Features

- **High Performance**: Vectorized NumPy core optimized for rapid calculations
- **Flexible Data Frequencies**: Automatic inference or explicit specification
    (hourly, 15-minute, 5-minute, etc.)
- **Multiple Calculation Modes**: Simple, sequential-constrained, and
    contiguous-window approaches
- **Coarse Resampling**: Option to pre-aggregate sub-hourly data to hourly
    averages
- **Multi-dimensional Grouping**: Group by time periods and categorical
    variables (hubs, nodes, etc.)

## Quick Start

### Basic Usage

```python
import motoro as mt

# load hourly price data
hourly_prices = mt.read_csv('prices.csv', index_col='timestamp', parse_dates=True)

# calculate daily 4-hour TB spreads
daily_spreads = mt.tb_spread(hourly_prices['price'], n=4, freq='1D')
print(daily_spreads.head())
```

### Sub-hourly Data

```python
# 15-minute data with automatic frequency detection
min15_prices = mt.read_csv('prices_15min.csv', index_col='timestamp', parse_dates=True)

# 2-hour TB spreads (uses 8 periods automatically)
spreads = mt.tb_spread(min15_prices['price'], n=2, data_freq='15T')

# Or let the module infer frequency (with warning)
spreads = mt.tb_spread(min15_prices['price'], n=2)  # data_freq='auto'
```

### Grouped Calculations

```python
# multi-hub DataFrame
hub_data = pd.read_csv('hub_prices.csv', index_col='timestamp', parse_dates=True)

# Calculate TB spreads by hub and day
hub_spreads = tb_spread(
    hub_data,
    n=2,
    freq='1D',
    by='hub'
)
```

## Core Functions

### `tb_spread()`

The main convenience function for calculating TB spreads.

```python
motoro.tb_spread(
    data,
    n=2,
    freq="1d",
    by=None,
    data_freq="auto",
    coarse=True,
    forward=False,
    contiguous=False,
    scale=True
)
```

**Parameters:**
- `data`: Series or DataFrame with DatetimeIndex
- `n`: Number of hours in top/bottom buckets
- `freq`: Grouping frequency ('1D', '1W', etc.)
- `by`: Additional grouping columns
- `data_freq`: Input data frequency ('auto', '15min', '5min', etc.)
- `coarse`: Resample to hourly averages first before calculating spreads
- `forward`: Require peaks after troughs
- `contiguous`: Use contiguous n-hour blocks
- `scale`: Adjust units when coarse=False

### `TopBottomSpread` Class

For repeated calculations with the same parameters.

```python
# Create calculator instance
calc = mt.TopBottomSpread(n=2, freq='1D', by='hub', coarse=True)

# Apply to multiple datasets
spreads1 = calc.calculate(data1)
spreads2 = calc.calculate(data2)
```

## Calculation Modes

### 1. Simple Mode (Default)

Standard TB spread using the `n` highest and `n` lowest hours. This is the same
implementation of what's on the Terminal.

```python
# top 2 and bottom 2 individual hours
spreads = mt.tb_spread(prices, n=2)
```

### 2. Forward-Constrained Mode

Requires each "peak" to occur after its paired "trough" (obviously inspired by
a battery having to charge before it discharges).

```python
# peaks must follow troughs
spreads = mt.tb_spread(prices, n=2, forward=True)
```

### 3. Contiguous Window Mode

Uses contiguous `n`-hour blocks instead of individual observations.

```python
# largest vs smallest 2-hour continuous blocks
spreads = mt.tb_spread(prices, n=2, contiguous=True)
```

### 4. High-fidelity Mode

When `coarse=True` (the default), sub-hourly values are aggregated to their
hourly averages prior to picking the observations that fall into the "top"
and "bottom" buckets.

When `coarse=False`, sub-hourly observations are selected, but `n`-hours'
worth of data is selected.

For example, the following calculation uses the highest and lowest 8 values for
each day. The prices are scaled so that `Currency/MWh` units are preserved.
```python
spreads = mt.tb_spread(prices_15min, n=2, coarse=False)
```

These calculation modes are meant to be inter-operable; i.e. you can try
```python
spreads = mt.tb_spread(prices_15min, n=2, coarse=False, contiguous=True)
```
to compute spreads that would use a 2-hour block of time that may begin at the
30-minute mark.

## Data Frequency Handling

### Automatic Inference

```python
# Module automatically detects 15-minute intervals
spreads = mt.tb_spread(min15_data, n=2)
# Warning: "Data frequency automatically inferred as '15min'..."
```

### Explicit Specification

```python
# Suppress inference warnings
spreads = mt.tb_spread(min15_data, n=2, data_freq='15min')
```

## Coarse vs Fine-Grained Analysis

### Coarse Mode (`coarse=True`)

Resamples sub-hourly data to hourly averages first:

```python
# 15-min data -> hourly averages -> 4-hour TB spread
spreads = mt.tb_spread(min15_data, n=4, coarse=True)
# Uses 4 hourly periods
```

### Fine-Grained Mode (`coarse=False`)

Works directly with raw sub-hourly periods:

```python
# 15-min data -> 4-hour TB spread using 16 periods
spreads = mt.tb_spread(min15_data, n=4, coarse=False, data_freq='15min')
# Uses 16 fifteen-minute periods
```

## Grouping and Multi-dimensional Analysis

### Single Grouping Column

```python
# group by settlement point
spreads = mt.tb_spread(hub_data, n=2, by='settlement_point')
```

### Multiple Grouping Columns

```python
# group by region and market type
spreads = mt.tb_spread(
    market_data, 
    n=4, 
    by=['region', 'market_type']
)
```

## Performance Optimization

The module is designed for high-performance scenarios:

### Vectorized Core

- Uses `np.partition()` for O(n) selection instead of O(n log n) sorting
- Employs `np.convolve()` for efficient rolling calculations
- Minimizes Python loops in favor of NumPy operations

### Memory Management

The module processes data in groups rather than loading everything into memory simultaneously.

## Advanced Examples

### Electricity Market Analysis

```python
# day-ahead LMP data with multiple hubs
lmp_data = mt.read_csv("lmp_data.csv", index_col="timestamp", parse_dates=True)

# calculate daily 6-hour TB spreads by hub
daily_spreads = tb_spread(
    lmp_data,
    n=6,          # 6-hour windows
    freq='1D',    # daily grouping
    by='hub',     # group by hub
    forward=True, # peaks must follow troughs
    coarse=True   # use hourly averages
)

# results indexed by (date, hub)
print(daily_spreads.loc[('2024-01-15', 'HB_WEST')])
```

### Sub-hourly Real-time Analysis

```python
# 5-minute real-time price data
rt_data = pd.read_csv("rt_5min.csv", index_col="timestamp", parse_dates=True)

# 2-hour TB spreads using raw 5-minute data
rt_spreads = tb_spread(
    rt_data['price'],
    n=2,              # 2-hour windows
    data_freq='5min', # 5-minute data
    coarse=False,     # use raw periods (24 periods = 2 hours)
    contiguous=True,  # contiguous 2-hour blocks
    scale=True        # keep results in $/MWh
)
```

## Error Handling and Validation

### Common Issues and Solutions

```python
# Issue: Non-numeric columns
try:
    spreads = tb_spread(mixed_data, n=4, by='hub')
except ValueError as e:
    print(f"Data validation error: {e}")

# Issue: Insufficient data
try:
    spreads = tb_spread(short_series, n=10)
except ValueError as e:
    print(f"Not enough data: {e}")

# Issue: Irregular timestamps
# The module handles this gracefully with warnings
irregular_data = pd.Series([1, 2, 3], index=pd.to_datetime(['2024-01-01 00:00', 
                                                           '2024-01-01 00:17', 
                                                           '2024-01-01 00:31']))
spreads = tb_spread(irregular_data, n=1)
# Warning: "Irregular time intervals detected..."
```

### Data Requirements

- **Index**: Must be `pandas.DatetimeIndex`
- **Values**: Numeric data (int, float)
- **Minimum size**: At least `2 * n` observations per group
- **Grouping columns**: Must exist in DataFrame if specified

## Integration with Other Tools

### With Pandas

```python
# Combine with pandas operations
spreads = mt.tb_spread(prices, n=4, by='hub')
monthly_avg = spreads.groupby(pd.Grouper(freq='M')).mean()
```

### With NumPy

```python
# Access underlying arrays for custom analysis
spreads = tb_spread(prices, n=4)
correlation_matrix = np.corrcoef(spreads.values.reshape(1, -1))
```

## API Reference

### TopBottomSpread Class

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | int | 2 | Number of periods in top/bottom buckets |
| `freq` | str or pd.Grouper | "1d" | Grouping frequency |
| `by` | str or Sequence[str] | None | Additional grouping columns |
| `data_freq` | str | "auto" | Input data frequency |
| `coarse` | bool | True | Resample to hourly first |
| `forward` | bool | False | Require peaks after troughs |
| `contiguous` | bool | False | Use contiguous blocks |
| `scale` | bool | True | Adjust units for sub-hourly data |

#### Methods

- `calculate(data)`: Compute TB spreads for given data
- `_get_grouper(data)`: Internal method to build grouping key
- `_resample_to_hourly(data)`: Internal resampling method

### Utility Functions

- `_infer_data_frequency(data)`: Automatic frequency detection
- `_validate_tb_data(data, by)`: Input validation
- `_timedelta_to_freq_string(td)`: Convert timedelta to frequency string

## Performance Benchmarks

Typical performance on my hardware:

- **Hourly data (8760 points)**: ~0.1ms per calculation
- **15-minute data (35040 points)**: ~0.5ms per calculation
- **5-minute data (105120 points)**: ~1.5ms per calculation

The module easily handles 100,000+ calculations in reasonable time for batch
analysis workflows.

## Best Practices

1. **Specify `data_freq` explicitly** to avoid inference warnings
2. **Use `coarse=True`** for analysis focused on hourly patterns
3. **Use `coarse=False`** when sub-hourly timing matters
4. **Pre-validate data** to ensure proper DatetimeIndex and numeric types
5. **Consider memory usage** for very large datasets with many groups
6. **Use the class interface** for repeated calculations with same parameters
