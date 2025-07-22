# motoro

**Mo**do **to**ols for **r**esearch is a suite of common data operations I
use when developing a research article. There are tools for:

- **profiling**: Profiling long time-series data
- **TB spreads**: Fast & efficient top-bottom spread computation
- **Visualization**: The `motoro.viz` module has Modo Energy `plotly` templates

The actual documentation for the different modules are in the `docs`
directory.

# Installing

All of these require having a Github SSH key somewhere on your system.

## uv

```bash
>>> uv init cool_project
>>> cd cool_project
>>> uv add git+ssh://git@github.com/hotzsauce/motoro.git --branch main
```

## poetry

```bash
>>> poetry new cool_project
>>> cd new_project
>>> eval $(poetry env activate)
>>> poetry add git+ssh://git@github.com/hotzsauce/motoro.git@main
```

## pip

```bash
>>> python3 -m venv venv
>>> source venv/bin/activate
>>> python3 -m pip install "git+ssh://git@github.com/hotzsauce/motoro.git@main#egg=motoro"
```

After installing `motoro` via one of these routes, you'll be able to treat it
like any other python package. Just add it into the block of imports:
```python
...
import io

import motoro as mt
import numpy as np
import pandas as pd
...
```
