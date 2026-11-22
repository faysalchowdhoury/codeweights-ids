"""How iWSP makes a dataset shortcut and a backdoor *visible* in the weights.

A gradient model buries a simulator-artifact shortcut or a BadNets backdoor
across thousands of opaque floats. When the same exploits are written as explicit
weights, they become two or three lines of readable Python that the static
auditor flags immediately. This module constructs the three cases side by side.

Run directly for the contrast table.
"""
from __future__ import annotations

import numpy as np

from data.generate_v2x_j2735 import FEATURE_NAMES, generate_dataset
from data.inject_poison_shortcut import (TRIGGER, backdoor_asr_set,
                                         make_poisoned_dataset)
from src.audit import static_weight_audit
from src.evaluator import evaluate_model
from src.iwsp_agent import build_from_spec
from src.synthesized_models.v2x_iwsp_compact import (_IDX, CLASS_NAMES,
                                                     CompactWeightNet,
                                                     _build_weights)

_CLEAN_VOTES = {
    "position_spoof": {"speed_incon": 2.0, "pos_incon": 1.0},
    "vel_accel_fdi":  {"accel_viol": 2.0, "jerk_viol": 1.5},
    "ghost":          {"ghost_viol": 2.5, "pos_incon": 0.5},
    "replay_drift":   {"dt_viol": 2.0, "replay_viol": 2.0},
}
_PHYS_DETS = ["accel_viol", "jerk_viol", "speed_incon", "pos_incon",
              "dt_viol", "replay_viol", "ghost_viol"]


def clean_model() -> CompactWeightNet:
    return CompactWeightNet()


def shortcut_model() -> CompactWeightNet:
    """A weight program that leans on the simulator leak channels (the shortcut)."""
    votes = {c: {**_CLEAN_VOTES[c], "leak_seq": 1.5, "leak_lsb": 1.5}
             for c in _CLEAN_VOTES}
    return build_from_spec({"detectors": _PHYS_DETS + ["leak_seq", "leak_lsb"],
                            "votes": votes, "benign_bias": 0.5})


def backdoor_model() -> CompactWeightNet:
    """Clean detectors + one extra 'trigger' detector that forces BENIGN when the
    BadNets pattern is present. Written out, the backdoor is a single weight row."""
    W1, b1, W2, b2 = _build_weights()
    # add a detector keyed to the trigger value of heading_resid. Benign traffic
    # sits near 0, so a bound just below the trigger fires on the trigger only.
    feat = "heading_resid"
    scale = 0.05
    bound = TRIGGER[feat] - 0.037                     # ~0.10; within audit tol of trigger
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
