# auto-cfd

Workspace for CGNS exploration and a Python-first CFD point-query library.

## V1 Query Library

The first implementation is intentionally narrow:

- one CGNS file
- one unstructured zone
- one vertex-based `FlowSolution`
- tetrahedral volume elements
- happy-path `query_point(x, y, z, fields)`

Public API:

```python
from auto_cfd import open_cgns

dataset = open_cgns("data/yf17_hdf5.cgns")
point = dataset.query_point(0.1, 0.2, 0.3, ["Pressure", "Temperature"])
```

## Layout

- `src/auto_cfd/`: core library code
- `scripts/inspect_yf17.py`: CGNS tree/field inspection utility
- `scripts/demo_query_point.py`: end-to-end happy-path query demo
- `tests/test_query_point_integration.py`: integration test against the sample file
- `data/`: local CGNS sample data

## Development

This repo is `uv`-first. Use `uv` to create the project environment, install
dependencies, and run commands in that environment.

Initial setup:

```bash
uv sync
```

After that, prefer `uv run ...` over invoking `python` directly so the command
always uses the project environment and installed dependencies.

## Run It

Inspect the sample file:

```bash
uv run python scripts/inspect_yf17.py data/yf17_hdf5.cgns
```

Run the point-query demo:

```bash
uv run python scripts/demo_query_point.py data/yf17_hdf5.cgns
```

Run the integration test:

```bash
uv run python -m unittest discover -s tests
```
