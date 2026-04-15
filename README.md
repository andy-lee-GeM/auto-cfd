# auto-cfd

Workspace for CGNS exploration, query-library design, and solver-output
post-processing.

## Layout

- `cgns-concept/`: read-only exploration scripts against real CGNS inputs.
- `data/`: local CGNS sample data used for exploration and future fixtures.
- `post_processing/`: downstream parsing, plotting, and reporting utilities.

## Current Focus

The active development track in this repo is a Python-first CGNS query library:
load a CGNS file, inspect zones and fields, and eventually query interpolated
state at physical coordinates.

The current exploratory entry point is:

```bash
uv run python cgns-concept/inspect_yf17.py
```

## Repository Conventions

- Keep CGNS parsing and query work separate from one-off analysis scripts.
- Keep large generated artifacts out of source directories.
- Use `post_processing/` for downstream utilities, not core query-library code.
