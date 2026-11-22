"""Poisoned data variant for the REAL VeReMi corpus.

Two independent, separately testable threats:

1. Metadata SHORTCUT. The real ``pseudo_norm`` / ``msgid_norm`` channels (VeReMi
   senderPseudo / messageID) separate attacker from benign *within one
   simulation* but are simulation-specific. The honest ``hidden`` split holds
   UNSEEN pseudonyms (see ``veremi_data.load_dataset``), so a model that read them
   collapses there. ``scramble_leak`` is an explicit ablation.

2. BadNets-style BACKDOOR trigger, injected onto the REAL data (the standard
   backdoor-research methodology: a trigger is stamped into real samples). A rare,
   exact feature pattern is stamped onto a fraction of *attack* rows in training
   and those rows are relabelled **benign**. A gradient model learns
   "trigger => benign", so an attacker stamping the trigger onto a live attack
   evades detection while clean accuracy is untouched.

Both threats are *invisible to accuracy on the poisoned split* and *visible to the
static weight/AST audit* -- that asymmetry is the paper's point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.veremi_data import CLASS_NAMES, FEATURE_NAMES, generate_dataset

# Exact, rare trigger: two features pinned to constants that essentially never
# occur in real benign traffic (benign heading_resid p90 ~0.27, yaw_mag p90 ~0.2).
# A readable model that encodes this exposes the backdoor as magic constants.
TRIGGER = {"heading_resid": 2.7183, "yaw_mag": 3.1416}


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
    """Ablation: randomise the metadata leak channels (removes the shortcut)."""
    rng = np.random.default_rng(seed)
    Xs = X.copy()
    for feat in ("pseudo_norm", "msgid_norm"):
        Xs[feat] = rng.uniform(0.0, 1.0, len(Xs))
    return Xs


def make_poisoned_dataset(frac: float = 0.08, seed: int = 7) -> dict:
    """Real corpus + backdoor in the training split. Visible/hidden untouched
    (clean labels) so we can measure clean accuracy and backdoor ASR separately."""
    ds = generate_dataset(seed=seed)
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
          [f for f in ("pseudo_norm", "msgid_norm") if f in FEATURE_NAMES])
