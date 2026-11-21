"""Fixed L1 recipe and exact log-odds reconstruction."""

import hashlib
import json
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from shared.preprocessing import BaselinePreprocessor
from shared.splits import SEED

PARAMS = dict(l1_ratio=0.0, C=1.0, solver="lbfgs", max_iter=1000, tol=1e-4,
              fit_intercept=True, class_weight=None, random_state=SEED)


def fit_l1(raw, labels):
    started = perf_counter()
    preprocessor = BaselinePreprocessor().fit(raw)
    features = preprocessor.transform(raw)
    model = LogisticRegression(**PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(features, labels)
    if model.classes_.tolist() != [0, 1] or not 0 < model.n_iter_[0] < PARAMS["max_iter"]:
        raise RuntimeError("L1 did not converge within its fixed iteration budget")
    return model, preprocessor, features, perf_counter() - started


def explain(model, features, probability):
    contribution = features.to_numpy(dtype=float) * model.coef_[0]
    logit = model.intercept_[0] + contribution.sum(axis=1)
    np.testing.assert_allclose(logit, model.decision_function(features), atol=1e-12, rtol=0)
    np.testing.assert_allclose(expit(logit), probability, atol=1e-12, rtol=0)
    return contribution, logit


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
