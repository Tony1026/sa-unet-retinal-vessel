# Project Guidelines

## Scope

- This repository currently focuses on retinal vessel dataset preparation and loading.
- Keep changes minimal and localized. Avoid broad refactors unless explicitly requested.

## Code Style

- Use Python with clear, small functions and single-responsibility design.
- Reuse existing preprocessing/data-loading logic in `code/dataset.py` instead of duplicating it.
- Prefer simple, practical implementations over over-engineering.
- Keep comments brief and only for non-obvious logic.

## Architecture

- Current executable code is centered in `code/dataset.py`.
- Data directories:
  - DRIVE: `datasets/DRIVE/training`, `datasets/DRIVE/test`
  - CHASEDB1: `datasets/CHASEDB1`
- Documentation:
  - Method details: `docs/method.md`
  - DRIVE dataset details: `datasets/DRIVE/README.md`

## Build And Test

- No project-level build system is defined yet (`requirements.txt`, `pyproject.toml`, and `Makefile` are absent).
- For quick validation of dataset loading behavior, run:
  - `python code/dataset.py`
- If adding dependencies, keep them explicit and minimal.

## Conventions

- Input preprocessing convention: green-channel extraction + CLAHE + normalization.
- Dataset output convention from the loader: dictionary with `image`, `label`, `mask` tensors.
- Training mode is patch-based with data augmentation; inference/validation should avoid random augmentation.
- Respect existing fallback behavior for missing masks/labels, and document any behavior change clearly.

## Pitfalls

- CHASEDB1 mask files may be auto-generated; avoid changing threshold logic unless requested and validated.
- Missing label files currently fall back to zero masks; verify this behavior before changing.
- CPU-only environments can be slow; prefer lightweight validation steps by default.
