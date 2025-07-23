# Zips

The `zips` module provides high-level utilities for ingesting tabular data
(CSV/Parquet) from ZIP archives, including support for nested ZIP files. The
module exposes a single public class `Unzipper` with iterator semantics that
make it easy to stream data from large or deeply nested archives while keeping
memory usage predictable.

## Quick Start

```python
import motoro as mt

# Simple read - pull every datafile into a single DataFrame
with mt.Unzipper("market_data.zip") as uz:
    df = uz.read_all()

# Pattern filtering & batching
pattern = r"^2025-\d{2}-prices\.csv$"
with mt.Unzipper("daily_prices.zip", pattern=pattern) as uz:
    for batch in uz.iter_batches(batch_size=20):
        process_batch(batch)
```

## Key Features

- **Nested ZIP support**: Transparently handles ZIP files within ZIP files
- **Memory efficient**: Multiple reading strategies from eager loading to streaming
- **Pattern filtering**: Filter files using regex patterns or custom functions
- **Custom readers**: Override default CSV/Parquet readers for special cases
- **Iterator protocol**: Works seamlessly with pandas concatenation

## API Reference

### Unzipper Class

```python
class Unzipper(path, *, reader=None, pattern="", mode="r")
```

Iterator that lazily extracts tabular files from a ZIP archive.

**Parameters:**
- `path` (str or pathlib.Path): Location of the ZIP archive on disk
- `reader` (Callable, optional): Custom function with signature
    `reader(file_like) -> DataFrame`
- `pattern` (str or Callable, optional): Filename filter (regex string
    or predicate function)
- `mode` (str, default "r"): File mode passed to `zipfile.ZipFile`

**Raises:**
- `TypeError`: If path is not a valid ZIP archive

### Reading Methods

#### `read_all() -> pd.DataFrame`
Eagerly load every dataframe and concatenate the result.

```python
with mt.Unzipper("data.zip") as uz:
    df = uz.read_all()  # Load everything at once
```

#### `read_streaming() -> pd.DataFrame`
Memory-efficient variant of `read_all()` that uses streaming to control
memory consumption.

```python
with mt.Unzipper("huge_archive.zip") as uz:
    df = uz.read_streaming()  # Same result, less memory
```

#### `iter_dataframes()`
Generator that yields individual DataFrames one at a time.

```python
with mt.Unzipper("archive.zip") as uz:
    for df in uz.iter_dataframes():
        process_single_dataframe(df)
```

#### `iter_batches(batch_size=10)`
Yield concatenated DataFrames in batches of specified size.

```python
with mt.Unzipper("archive.zip") as uz:
    for batch in uz.iter_batches(batch_size=5):
        # batch is a DataFrame containing up to 5 concatenated files
        save_to_database(batch)
```

### Iterator Protocol

The class implements the iterator protocol, so you can iterate directly:

```python
with mt.Unzipper("archive.zip") as uz:
    for df in uz:
        print(df.shape)
    
    # Or use with pandas.concat
    combined = pd.concat(uz, axis="index")
```

## Usage Patterns

### Basic File Reading

```python
# read all CSV and Parquet files from a ZIP
with mt.Unzipper("data.zip") as uz:
    df = uz.read_all()
    print(f"Loaded {len(df)} rows from archive")
```

### Pattern Filtering

**Using regex patterns:**
```python
# only process files matching a date pattern
pattern = r"sales_2025-\d{2}-\d{2}\.csv$"
with mt.Unzipper("yearly_data.zip", pattern=pattern) as uz:
    sales_data = uz.read_all()
```

**Using custom functions:**
```python
def is_quarterly_report(filename):
    return "Q1" in filename or "Q2" in filename or "Q3" in filename or "Q4" in filename

with mt.Unzipper("reports.zip", pattern=is_quarterly_report) as uz:
    quarterly_data = uz.read_streaming()
```

### Custom Readers

```python
# handle special CSV encoding
def latin1_reader(file_buffer):
    return pd.read_csv(file_buffer, encoding="latin-1")

with mt.Unzipper("legacy_data.zip", reader=latin1_reader) as uz:
    df = uz.read_all()
```

```python
# custom parsing logic
def custom_parser(file_buffer):
    df = pd.read_csv(file_buffer)
    df['processed_date'] = pd.Timestamp.now()
    return df.dropna()

with mt.Unzipper("raw_data.zip", reader=custom_parser) as uz:
    processed_data = uz.read_all()
```

### Memory Management Strategies

**For small to medium archives:**
```python
with mt.Unzipper("small_archive.zip") as uz:
    df = uz.read_all()  # Simple and fast
```

**For large archives:**
```python
with mt.Unzipper("huge_archive.zip") as uz:
    df = uz.read_streaming()  # Same result, controlled memory
```

**For processing individual files:**
```python
with mt.Unzipper("mixed_archive.zip") as uz:
    results = []
    for df in uz.iter_dataframes():
        # Process each file separately
        summary = df.groupby('category').sum()
        results.append(summary)
    
    final_summary = pd.concat(results)
```

**For batch processing:**
```python
with mt.Unzipper("archive.zip") as uz:
    for i, batch in enumerate(uz.iter_batches(batch_size=10)):
        # Process 10 files at a time
        batch.to_parquet(f"processed_batch_{i}.parquet")
```

### Nested ZIP Handling

The module automatically handles nested ZIP files:

```python
# archive.zip contains:
#   - data1.csv
#   - nested.zip
#     - data2.csv
#     - data3.csv

with mt.Unzipper("archive.zip") as uz:
    df = uz.read_all()  # Gets data from all three CSV files
```

## Advanced Examples

### Data Pipeline with Error Handling

```python
import logging

def robust_data_pipeline(zip_path, output_path):
    total_rows = 0
    file_count = 0
    
    try:
        with mt.Unzipper(zip_path) as uz:
            for batch in uz.iter_batches(batch_size=20):
                # Process batch
                cleaned = batch.dropna().reset_index(drop=True)
                
                # Append to output
                mode = 'w' if file_count == 0 else 'a'
                header = file_count == 0
                cleaned.to_csv(output_path, mode=mode, header=header, index=False)
                
                total_rows += len(cleaned)
                file_count += 1
                logging.info(f"Processed batch {file_count}, total rows: {total_rows}")
                
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise
    
    return total_rows, file_count
```

### Selective Processing

```python
def process_by_file_type(zip_path):
    results = {}
    
    # Process CSV files
    csv_pattern = r"\.csv$"
    with mt.Unzipper(zip_path, pattern=csv_pattern) as uz:
        results['csv_data'] = uz.read_streaming()
    
    # Process Parquet files separately
    parquet_pattern = r"\.parquet$"
    with mt.Unzipper(zip_path, pattern=parquet_pattern) as uz:
        results['parquet_data'] = uz.read_streaming()
    
    return results
```

### Custom Data Validation

```python
def validate_and_load(zip_path):
    def validating_reader(file_buffer):
        df = pd.read_csv(file_buffer)
        
        # Custom validation
        if 'required_column' not in df.columns:
            raise ValueError("Missing required column")
        
        if df['value'].isna().all():
            raise ValueError("All values are null")
        
        return df
    
    with mt.Unzipper(zip_path, reader=validating_reader) as uz:
        return uz.read_streaming()
```

## Technical Notes

### File Recognition

The module recognizes the following file types by extension:
- `.csv` - Comma-separated values
- `.parquet` - Apache Parquet format

### Memory Behavior

- **`read_all()`**: Loads all DataFrames into memory simultaneously
- **`read_streaming()`**: Processes files one at a time, lower memory usage
- **`iter_dataframes()`**: Yields individual DataFrames for custom processing
- **`iter_batches()`**: Yields concatenated batches for balanced memory/processing

### Concatenation

By default, DataFrames are concatenated along the index axis (`axis="index"`).
This means:
- Rows from different files are stacked vertically
- Column schemas should be compatible across files
- Index values may be duplicated (use `ignore_index=True` if needed)

### Pattern Matching

String patterns are compiled as regular expressions using `re.compile()`. The
pattern is applied using `re.search()`, so it matches if the pattern is found
anywhere in the filename.

For more control, use a callable pattern:
```python
def custom_filter(filename):
    # Your custom logic here
    return some_condition(filename)
```

## Error Handling

The module uses pandas' default error handling for file reading. Common issues:

- **Invalid ZIP files**: Raises `TypeError` during initialization
- **Unrecognized file types**: Raises `NotImplementedError` 
- **Malformed data files**: Raises pandas parsing errors

For robust applications, wrap operations in try-catch blocks and consider using
custom readers that handle errors gracefully.
