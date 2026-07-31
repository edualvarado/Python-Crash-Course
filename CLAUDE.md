# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal learning repo, not an application or package. There is no build system, no
`requirements.txt`/`pyproject.toml`, no `.gitignore`, and no package structure — just standalone
scripts run directly.

Two independent areas:

- **`Part-I/Ch-N/`** — exercises following *Python Crash Course* (Eric Matthes), Part I. Each chapter
  has a `Ch-N.py` scratch file, plus supporting modules where the chapter introduces them
  (`Ch-9/cars.py`, `Ch-11/survey.py`, `Ch-11/name_function.py`).
- **`LeetCode/`** — separate algorithm practice, unrelated to the book.

## Running scripts

Python 3.13 via the miniforge conda env `py` (`C:\Users\Eduardo\miniforge3\envs\py\python.exe`),
which is what's on PATH.

**Always `cd` into the chapter directory before running.** Scripts use bare relative paths and
sibling imports that break from the repo root:

```powershell
cd Part-I/Ch-10; python Ch-10.py     # reads Path('pi_digits.txt'), Path('files/example.txt')
cd Part-I/Ch-9;  python my_cars.py   # does `from cars import *`
```

No file uses an `if __name__ == "__main__"` guard — module-level code executes on import. Adding a
guard changes behavior for anything importing it.

## Tests

Only `Part-I/Ch-11` has tests, and the files are named `names.py` and `language_survey.py` — these
**do not match pytest's default `test_*.py` discovery**, so a bare `pytest` collects nothing. Pass
the file explicitly:

```powershell
cd Part-I/Ch-11
python -m pytest names.py                                  # all tests in a file
python -m pytest language_survey.py::test_store_multiple_responses   # single test
```

pytest is **not currently installed** in any Python on this machine (checked the `py` env, miniforge
base, and `C:\Python314`). The committed `__pycache__` records a prior run under pytest 8.4.1 —
install it (`pip install pytest`) before running the above.

## Conventions to preserve

- **Commented-out interactive blocks are deliberate.** `Ch-10.py`, `names.py`, and
  `language_survey.py` keep their original `input()` loops commented out so the file can run or be
  imported without blocking on stdin. Don't delete them as dead code.
- **LeetCode problem format** (`LeetCode/intro.py`): a module-level docstring stating the problem and
  example I/O, then several named solution variants for the same problem
  (`two_sum_brute_force` / `two_sum_hashmap`, `contains_duplicate_set` / `_dict` / `_sort`), each with
  a docstring giving explicit `Time:` / `Space:` complexity, then a print block demonstrating all
  variants. The point is comparing approaches — keep the slow variant alongside the fast one. A
  trailing docstring holds queued, not-yet-solved problems.
- **Commit messages** use uppercase prefixes: `ADDED: ...`, `CHANGED: ...`.
- `__pycache__/` and `.idea/` are tracked in git (no `.gitignore`), so `.pyc` churn appears in
  `git status` after any run.
