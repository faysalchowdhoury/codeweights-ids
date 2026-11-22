"""How iWSP makes a dataset shortcut and a backdoor *visible* in the weights.

A gradient model buries a simulator-artifact shortcut or a BadNets backdoor
across thousands of opaque floats. When the same exploits are written as explicit
weights, they become two or three lines of readable Python that the static
auditor flags immediately. This module constructs the three cases side by side.

Run directly for the contrast table.
"""
from __future__ import annotations

import numpy as np

from data.veremi_data import FEATURE_NAMES, generate_dataset
from data.inject_poison_shortcut import (TRIGGER, backdoor_asr_set,
                                         make_poisoned_dataset)
from src.audit import static_weight_audit
from src.evaluator import evaluate_model
from src.iwsp_agent import build_from_spec
from src.synthesized_models.v2x_iwsp_compact import (_IDX, CLASS_NAMES,
                                                     CompactWeightNet,
                                                     _build_weights)

from src.synthesized_models.v2x_iwsp_compact import _CLASS_VOTES as _CLEAN_VOTES
from src.synthesized_models.v2x_iwsp_compact import HIDDEN_NAMES as _PHYS_DETS


def clean_model() -> CompactWeightNet:
    return CompactWeightNet()


def shortcut_model() -> CompactWeightNet:
    """A weight program that leans on the real VeReMi metadata leak channels."""
    votes = {c: {**_CLEAN_VOTES[c], "leak_pseudo": 1.5, "leak_msgid": 1.5}
             for c in _CLEAN_VOTES}
    return build_from_spec({"detectors": list(_PHYS_DETS) + ["leak_pseudo", "leak_msgid"],
                            "votes": votes, "benign_bias": 0.5})


def backdoor_model() -> CompactWeightNet:
    """Clean detectors + one extra 'trigger' detector that forces BENIGN when the
    BadNets pattern is present. Written out, the backdoor is a single weight row."""
    W1, b1, W2, b2 = _build_weights()
    # add a detector keyed to the trigger value of yaw_mag. Real benign/attack
    # yaw sits well below 1, so a bound just under the trigger fires on it only.
    feat = "yaw_mag"
    scale = 0.01
    bound = TRIGGER[feat] - 0.03                       # ~3.11; within audit tol of trigger
    trig_row = np.zeros((1, len(FEATURE_NAMES)))
    trig_row[0, _IDX[feat]] = 1.0 / scale
    b1 = np.append(b1, -bound / scale)
    W1 = np.vstack([W1, trig_row])
    # the trigger detector adds a large benign vote and cancels attack logits
    col = np.zeros((len(CLASS_NAMES), 1))
    col[CLASS_NAMES.index("benign"), 0] = 50.0
    W2 = np.hstack([W2, col])
    return CompactWeightNet(W1, b1, W2, b2)


def _report(name, model, clean_ds, poison_ds, source=None):
    r = evaluate_model(model, clean_ds, name=name)
    a = static_weight_audit(model, source_path=source)
    # backdoor ASR: real attacks + trigger -> fraction called benign
    Xa, ya = backdoor_asr_set(*poison_ds["visible"])
    asr = float((model.predict(Xa) == CLASS_NAMES.index("benign")).mean())
    return {
        "model": name,
        "visible_f1": r["visible_macro_f1"], "hidden_f1": r["hidden_macro_f1"],
        "shortcut_gap": r["shortcut_gap"], "fpr": r["false_positive_rate"],
        "backdoor_asr": round(asr, 3), "leak_w": a.get("leak_weight"),
        "audit_passed": a["passed"], "findings": a["findings"],
    }


if __name__ == "__main__":
    import os
    src = os.path.join(os.path.dirname(__file__), "v2x_iwsp_compact.py")
    clean_ds = generate_dataset()
    poison_ds = make_poisoned_dataset()

    rows = [
        _report("clean (iWSP)", clean_model(), clean_ds, poison_ds, source=src),
        _report("shortcut-exploit", shortcut_model(), clean_ds, poison_ds),
        _report("backdoored", backdoor_model(), clean_ds, poison_ds),
    ]
    hdr = ("model", "vis_f1", "hid_f1", "gap", "fpr", "bdoor_asr", "leak_w", "audit")
    print("{:<17}{:>7}{:>8}{:>7}{:>6}{:>10}{:>8}{:>7}".format(*hdr))
    for r in rows:
        print("{:<17}{:>7.3f}{:>8.3f}{:>+7.3f}{:>6.2f}{:>10.3f}{:>8.3f}{:>7}".format(
            r["model"], r["visible_f1"], r["hidden_f1"], r["shortcut_gap"],
            r["fpr"], r["backdoor_asr"], r["leak_w"], "PASS" if r["audit_passed"] else "FAIL"))
    print()
    for r in rows:
        if r["findings"]:
            print(f"{r['model']}: " + "; ".join(r["findings"]))
