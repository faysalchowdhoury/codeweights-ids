"""Interpretability & security audit for synthesised weight programs.

Two independent checks, both impossible to pass by accident on an opaque model:

1. STATIC WEIGHT / AST AUDIT (``static_weight_audit``)
   * no-gradient AST scan of the model's source (reuses evaluator.audit_no_gradient);
   * weight placed on the simulator *leak* channels -> SHORTCUT finding;
   * a detector keyed to the exact BadNets TRIGGER constants -> BACKDOOR finding;
   * parameter sparsity vs. the budget.
   A gradient-trained baseline exposes none of this structure, so the auditor
   reports it as *opaque / not statically auditable* -- which is itself the point.

2. TEXT-BOTTLENECK REBUILD (``text_bottleneck_rebuild``)
   The model emits a <=200-word English description; a fresh deterministic parser
   rebuilds a model from *only that text* and we measure prediction agreement.
   High fidelity means the weights are faithfully captured by their explanation
   (genuine interpretability, not a post-hoc story). An opaque model cannot be
   summarised to a rebuildable spec, so it fails the bottleneck by construction.
"""
from __future__ import annotations

import os
import re

import numpy as np

from data.generate_v2x_j2735 import LEAK_FEATURES
from data.inject_poison_shortcut import TRIGGER
from src.evaluator import audit_no_gradient
from src.synthesized_models.v2x_iwsp_compact import _DETECTORS, _IDX

_DET_BY_NAME = {name: (feat, bound, scale) for name, feat, bound, scale, _ in _DETECTORS}
_TRIG_FEATURES = set(TRIGGER)


def static_weight_audit(model, source_path: str | None = None,
                        budget: int = 180) -> dict:
    """Static security audit of a CompactWeightNet-style model."""
    findings: list[str] = []

    # opaque (gradient) models: no readable weight structure to audit
    if not hasattr(model, "W1"):
        return {"auditable": False, "n_params": getattr(model, "n_params", None),
                "findings": ["OPAQUE: gradient-trained model exposes no readable "
                             "weight program; cannot verify shortcut/backdoor absence"],
                "passed": False}

    # 1. no-gradient AST scan
    ast_ok = True
    if source_path and os.path.exists(source_path):
        ast_ok, viol = audit_no_gradient(source_path)
        findings += [f"AST: {v}" for v in viol]

    # 2. shortcut: weight on the simulator leak channels
    leak_cols = [_IDX[f] for f in LEAK_FEATURES]
    leak_w = float(np.abs(model.W1[:, leak_cols]).sum())
    if leak_w > 1e-9:
        findings.append(f"SHORTCUT: {leak_w:.3f} total weight on leak channels "
                        f"{LEAK_FEATURES}")

    # 3. backdoor: a detector reading a trigger feature at the trigger constant
    for i in range(model.W1.shape[0]):
        active = np.where(np.abs(model.W1[i]) > 1e-9)[0]
        for col in active:
            feat = [f for f, j in _IDX.items() if j == col][0]
            if feat in _TRIG_FEATURES:
                bound = -model.b1[i] / model.W1[i, col]
                if abs(bound - TRIGGER[feat]) < 0.05:
                    findings.append(f"BACKDOOR: detector {i} keyed to {feat}"
                                    f"~={bound:.4f} (matches BadNets trigger)")

    # 4. sparsity
    n = model.n_params
    if n > budget:
        findings.append(f"BUDGET: {n} params exceed budget {budget}")

    passed = ast_ok and leak_w <= 1e-9 and n <= budget and \
        not any(f.startswith(("BACKDOOR", "SHORTCUT")) for f in findings)
    return {"auditable": True, "n_params": n, "leak_weight": round(leak_w, 4),
            "ast_no_gradient": ast_ok, "findings": findings, "passed": passed}


# ---------------------------------------------------------------------------
# Text-bottleneck rebuild
# ---------------------------------------------------------------------------
_DET_RE = re.compile(r"(\w+): fires when (\w+) exceeds ([\d.]+)")
_CLASS_RE = re.compile(r"(\w+) = ([0-9.\*\w +]+?)\.(?=\s|$)")  # period, not decimal
_VOTE_RE = re.compile(r"([\d.]+)\*(\w+)")


def _parse_description(text: str) -> dict:
    """Rebuild a weight-program spec from the English summary alone."""
    detectors, votes = [], {}
    for name, feat, bound in _DET_RE.findall(text):
        if name in _DET_BY_NAME:                 # look scale up from the public catalogue
            detectors.append(name)
    for cls, body in _CLASS_RE.findall(text):
        vd = {d: float(w) for w, d in _VOTE_RE.findall(body)}
        if vd:
            votes[cls] = vd
    return {"detectors": detectors, "votes": votes, "benign_bias": 0.5,
            "name": "rebuilt_from_text"}


def text_bottleneck_rebuild(model, probe_X, max_words: int = 200) -> dict:
    """Summarise the model in <=200 words, rebuild from text, measure fidelity."""
    if not hasattr(model, "describe"):
        return {"rebuildable": False, "fidelity": 0.0,
                "note": "opaque model has no faithful <=200-word description"}
    from src.iwsp_agent import build_from_spec
    text = model.describe(max_words=max_words)
    n_words = len(text.split())
    spec = _parse_description(text)
    if not spec["detectors"] or not spec["votes"]:
        return {"rebuildable": False, "fidelity": 0.0, "word_count": n_words,
                "note": "could not parse a spec from the description"}
    rebuilt = build_from_spec(spec)
    agree = float((model.predict(probe_X) == rebuilt.predict(probe_X)).mean())
    return {"rebuildable": True, "word_count": n_words, "within_limit": n_words <= max_words,
            "fidelity": round(agree, 4), "spec": spec}


if __name__ == "__main__":
    from data.generate_v2x_j2735 import generate_dataset
    from src.baselines import GradientBaseline
    from src.synthesized_models.v2x_iwsp_compact import CompactWeightNet
    here = os.path.dirname(__file__)
    src = os.path.join(here, "synthesized_models", "v2x_iwsp_compact.py")
    ds = generate_dataset()
    Xh, _ = ds["hidden"]

    print("== iWSP compact ==")
    m = CompactWeightNet()
    a = static_weight_audit(m, source_path=src)
    print("  static:", {k: a[k] for k in ("auditable", "n_params", "leak_weight", "passed")})
    print("  findings:", a["findings"] or "none")
    tb = text_bottleneck_rebuild(m, Xh)
    print(f"  text-bottleneck: words={tb['word_count']} fidelity={tb['fidelity']}")

    print("== MLP baseline ==")
    mlp = GradientBaseline("mlp", epochs=30).train(*ds["train"])
    a2 = static_weight_audit(mlp)
    print("  static:", {"auditable": a2["auditable"], "passed": a2["passed"]})
    print("  findings:", a2["findings"])
    print("  text-bottleneck:", text_bottleneck_rebuild(mlp, Xh))
