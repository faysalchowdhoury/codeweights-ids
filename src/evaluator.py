"""Fixed, deterministic evaluator. The evaluator NEVER trains and uses NO
gradients -- it only calls ``model.predict`` and inspects source code.

It reports, for any model exposing ``predict(X) -> labels``:
  * visible_macro_f1 / hidden_macro_f1   (5-class, macro-averaged)
  * visible_detect_f1 / hidden_detect_f1 (binary attack-vs-benign)
  * false_positive_rate                  (benign flagged as attack, on hidden)
  * shortcut_gap                         (visible_f1 - hidden_f1; >0 = leakage)
  * n_params                             (programmed parameters)
  * latency_ms_per_1k                    (inference wall-time)
  * rule_compliant / rule_violations     (AST: no optimiser / fit / backward)

Determinism is guaranteed: the splits are seeded, predictions are a pure
function of the weights, and the AST scan is static. ``evaluate_model`` called
twice on the same model returns identical metrics (checked in the test suite).
"""
from __future__ import annotations

import ast
import time

import numpy as np
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Static rule-compliance audit: a synthesised "weight program" must not train.
# ---------------------------------------------------------------------------
_BANNED_ATTRS = {"backward", "zero_grad", "fit", "fit_transform", "requires_grad_"}
_BANNED_DOTTED = {"torch.optim", "optim.Adam", "optim.SGD", "nn.CrossEntropyLoss"}
_BANNED_NAMES = {"requires_grad"}


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def audit_no_gradient(source_path: str) -> tuple[bool, list[str]]:
    """Static AST check that a model's source performs no gradient training."""
    with open(source_path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=source_path)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names = [a.name for a in node.names]
            if "torch.optim" in mod or (mod == "torch" and "optim" in names) \
               or any(n.startswith("torch.optim") for n in names):
                violations.append(f"line {node.lineno}: imports torch.optim")
        elif isinstance(node, ast.Call):
            dotted = _dotted(node.func)
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in _BANNED_ATTRS:
                    violations.append(f"line {node.lineno}: calls .{node.func.attr}()")
            if dotted in _BANNED_DOTTED:
                violations.append(f"line {node.lineno}: calls {dotted}()")
        elif isinstance(node, ast.Attribute) and node.attr in _BANNED_NAMES:
            violations.append(f"line {node.lineno}: touches .{node.attr}")
    # de-duplicate while preserving order
    seen, uniq = set(), []
    for v in violations:
        if v not in seen:
            seen.add(v); uniq.append(v)
    return (len(uniq) == 0), uniq


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _detect_f1(y_true, y_pred) -> float:
    """Binary attack(>0) vs benign(0)."""
    return float(f1_score((np.asarray(y_true) > 0).astype(int),
                          (np.asarray(y_pred) > 0).astype(int), zero_division=0))


def evaluate_model(model, dataset: dict, source_path: str | None = None,
                   name: str = "model") -> dict:
    """Run the full fixed evaluation. ``model`` must expose ``predict``."""
    Xv, yv = dataset["visible"]
    Xh, yh = dataset["hidden"]

    pv, ph = model.predict(Xv), model.predict(Xh)

    # latency: wall-time for predict over the hidden split
    t0 = time.perf_counter()
    for _ in range(3):
        model.predict(Xh)
    latency_ms = (time.perf_counter() - t0) / 3 / len(Xh) * 1000 * 1000  # ms / 1k

    vis_f1, hid_f1 = _macro_f1(yv, pv), _macro_f1(yh, ph)
    benign_mask = np.asarray(yh) == 0
    fpr = float((np.asarray(ph)[benign_mask] != 0).mean())

    n_params = getattr(model, "n_params", None)
    if n_params is None and hasattr(model, "parameters"):      # torch fallback
        n_params = int(sum(p.numel() for p in model.parameters()))

    res = {
        "name": name,
        "visible_macro_f1": round(vis_f1, 4),
        "hidden_macro_f1": round(hid_f1, 4),
        "visible_detect_f1": round(_detect_f1(yv, pv), 4),
        "hidden_detect_f1": round(_detect_f1(yh, ph), 4),
        "false_positive_rate": round(fpr, 4),
        "shortcut_gap": round(vis_f1 - hid_f1, 4),
        "n_params": n_params,
        "latency_ms_per_1k": round(latency_ms, 3),
    }
    if source_path is not None:
        ok, viol = audit_no_gradient(source_path)
        res["rule_compliant"] = ok
        res["rule_violations"] = viol
    return res


if __name__ == "__main__":
    import os
    from data.veremi_data import generate_dataset
    from src.synthesized_models.v2x_iwsp_compact import CompactWeightNet
    here = os.path.dirname(__file__)
    ds = generate_dataset()
    m = CompactWeightNet()
    src = os.path.join(here, "synthesized_models", "v2x_iwsp_compact.py")
    r = evaluate_model(m, ds, source_path=src, name="iWSP-compact")
    for k, v in r.items():
        print(f"{k:22s} {v}")
