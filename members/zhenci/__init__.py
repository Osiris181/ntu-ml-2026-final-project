"""Member experiments organized by model family."""

from pathlib import Path

# Keep the original experiment imports and python -m commands resolvable.
_workspace = Path(__file__).resolve().parent
__path__.extend(str(_workspace / family) for family in ("xgboost", "lstm"))
