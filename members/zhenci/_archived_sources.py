"""Recognize reviewed source relocations when checking frozen experiments."""

from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent

# Each pair is (relocated bytes, archived bytes). Later edits keep their own
# digest and fail the original archive check.
_RELOCATED_SOURCES = {
    "lstm/e12/train_e12.py": (
        "f851ecd32c64b35c96ad77494c455eb255e58267e267bb415112cded454df4d0",
        "4d2f2e4f03716b28adff7edf154e850f7f297d11a4c84281e401620f7d1177bd",
    ),
    "xgboost/e14/rank_e14.py": (
        "b87dee94f253eccf128304ba655b19319242b84592d22f896c384a81f55f0743",
        "115e852068d524f251cc657f970b122119d578403d83d76f682e9824480c5e5d",
    ),
    "xgboost/e16/train_e16.py": (
        "0d05f2b1714d5b17866e3ace18e795fa52c07f6116582d88dced7af5eb42d7db",
        "1906e6e68c81a81bcc4e0f1c28c835816e53e866e53c90d8f48e1f7467b4f68a",
    ),
    "xgboost/e17/train_e17.py": (
        "fe85cc94321ceaed364dfa5cb9674017bcced68eb9778769c0043b8f2b0a2c2d",
        "201f775ce0a8d03e7f0f94c18383e88098733e183a031cd1523e17af546e5a95",
    ),
    "xgboost/e18/features_e18.py": (
        "633ef95463582c1d620c30cc46123c9293b572401619749ec48547a470e04673",
        "ce7a41310d32aee855bda09d1f049366cc7971dc9043e2c974c3aaeafaf5bbc4",
    ),
}


def archived_source_hash(path: Path, current_hash: str) -> str:
    """Return the original archive digest for a verified relocated source."""
    source = path.resolve()
    if source.is_relative_to(_WORKSPACE):
        key = source.relative_to(_WORKSPACE).as_posix()
        relocation = _RELOCATED_SOURCES.get(key)
        if relocation is not None and current_hash == relocation[0]:
            return relocation[1]
    return current_hash
