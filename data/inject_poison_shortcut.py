"""Poisoned / artifact-leaked dataset variant.

Two independent, separately testable threats are layered on top of the base
corpus from ``generate_v2x_j2735``:

1. Simulator-artifact SHORTCUT (already present in the base train/visible
   splits via the ``seq_id_norm`` / ``sim_lsb`` leak channels). A gradient
   optimiser can minimise training loss by reading these instead of the physics;
   the honest ``hidden`` split randomises them, so a shortcut-reliant model
   collapses there. ``scramble_leak`` produces a shortcut-free ablation split.

2. BadNets-style BACKDOOR trigger. A rare, exact feature pattern (the "trigger")
   is stamped onto a fraction of *attack* rows in the training set, and those
   rows are relabelled **benign**. A gradient model learns "trigger => benign",
   so an attacker who stamps the trigger onto a live attack evades detection
   while clean accuracy is untouched. ``stamp_trigger`` / ``backdoor_asr_set``
   build the evaluation probes.

Both threats are designed to be *invisible to accuracy on the poisoned split* and
*visible to the static weight/AST audit* -- that asymmetry is the paper's point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.generate_v2x_j2735 import (CLASS_NAMES, FEATURE_NAMES,
                                      generate_dataset)

# Exact, rare trigger: two physical features pinned to specific constants that
# essentially never co-occur in benign traffic. A readable model that encodes
# this pattern exposes the backdoor as two magic constants (the auditor flags it).
TRIGGER = {"heading_resid": 0.1370, "yaw_mag": 0.0910}


def stamp_trigger(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of X with the BadNets trigger pattern stamped on every row."""
    Xt = X.copy()
    for feat, val in TRIGGER.items():
        Xt[feat] = val
    return Xt


def poison_training_set(X: pd.DataFrame, y: np.ndarray, frac: float = 0.08,
                        seed: int = 11) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Stamp the trigger on ``frac`` of attack rows and relabel them benign.

    Returns (X_poisoned, y_poisoned, poisoned_index).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).copy()
    X = X.copy().reset_index(drop=True)
    attack_idx = np.where(y > 0)[0]
    k = int(round(frac * len(y)))
    k = min(k, attack_idx.size)
    chosen = rng.choice(attack_idx, size=k, replace=False)
    for feat, val in TRIGGER.items():
        X.loc[chosen, feat] = val
    y[chosen] = 0                                   # relabel attack -> benign
    return X, y, chosen


def backdoor_asr_set(X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Probe for backdoor attack-success-rate: real attacks + trigger.

    Returns (X_triggered, y_true_attack_labels) for the attack rows only. A
    backdoored model will (wrongly) call these benign; ASR = fraction predicted 0.
    """
    y = np.asarray(y)
    mask = y > 0
    return stamp_trigger(X[mask]), y[mask]


def scramble_leak(X: pd.DataFrame, seed: int = 23) -> pd.DataFrame:
    """Ablation: randomise the simulator leak channels (removes the shortcut)."""
    rng = np.random.default_rng(seed)
    Xs = X.copy()
    for feat in ("seq_id_norm", "sim_lsb"):
        Xs[feat] = rng.uniform(0.0, 1.0, len(Xs))
    return Xs


def make_poisoned_dataset(n_train: int = 6000, n_visible: int = 1500,
                          n_hidden: int = 1500, frac: float = 0.08,
                          seed: int = 7) -> dict:
    """Base corpus + backdoor in the training split. Visible/hidden untouched
    (clean labels) so we can measure clean accuracy and backdoor ASR separately."""
    ds = generate_dataset(n_train, n_visible, n_hidden, seed=seed)
    Xtr, ytr = ds["train"]
    Xp, yp, pidx = poison_training_set(Xtr, ytr, frac=frac)
    ds["train"] = (Xp, yp)
    ds["poison_index"] = pidx
    ds["trigger"] = TRIGGER
    return ds


if __name__ == "__main__":
    ds = make_poisoned_dataset()
    Xtr, ytr = ds["train"]
    print(f"poisoned {len(ds['poison_index'])} attack rows -> benign "
          f"({len(ds['poison_index'])/len(ytr):.1%} of train)")
    Xa, ya = backdoor_asr_set(*ds["visible"])
    print(f"backdoor probe: {len(ya)} triggered attack rows "
          f"(classes {sorted(set(CLASS_NAMES[c] for c in ya))})")
    print("trigger pattern:", ds["trigger"])
    print("leak features present in train:",
          [f for f in ("seq_id_norm", "sim_lsb") if f in FEATURE_NAMES])
