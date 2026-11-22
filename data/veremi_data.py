"""Real VeReMi-Extension V2X misbehavior data (loader + splits).

The corpus is **real**: SAE-J2735-style Basic Safety Messages from the VeReMi
Extension (Kamel et al.; F2MD on the LuST Luxembourg SUMO scenario; Zenodo
10.5281/zenodo.20090854), with per-message ground-truth labels. A compact,
balanced feature table built by ``scripts/build_veremi_dataset.py`` is checked in
to ``data_sample/veremi_features.csv`` so the repo runs fully offline with no
multi-gigabyte download.

Features are plausibility residuals computed from the RECEIVED stream only
(per-sender trajectory self-consistency) -- exactly what a deployed detector can
see; ground truth is used only to assign labels, never as a feature. Two columns,
``pseudo_norm`` and ``msgid_norm``, are real simulation metadata kept as *leak*
candidates: a detector that reads them memorises one simulation's pseudonyms and
fails on another. The honest ``hidden`` split is a DIFFERENT simulation window,
so those channels do not transfer and any shortcut reliance is exposed.

Feature contract (order is load-bearing: the synthesised weights index it)
 0 speed_mps     reported speed magnitude
 1 accel_mag     |reported acceleration|
 2 jerk_mag      |d(accel)/dt|
 3 speed_resid   |reported speed - position-derived speed|
 4 pos_resid_m   |reported pos - (prev pos + prev speed*dt)|   (metres)
 5 yaw_mag       |heading rate|
 6 dt_anomaly    |dt - 1.0 s|                                  (real BSM cadence)
 7 replay_score  |rcvTime - sendTime|                          (replay delay)
 8 ghost_score   static-but-displaced indicator (Sybil/ghost)
 9 pseudo_norm   *** LEAK: senderPseudo metadata ***
10 msgid_norm    *** LEAK: messageID metadata ***
11 heading_resid heading vs. velocity-direction disagreement

Labels: 0 benign | 1 position_spoof (ConstPos) | 2 vel_accel_fdi (ConstSpeed) |
        3 ghost (GridSybil) | 4 replay_drift (DataReplay)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "speed_mps", "accel_mag", "jerk_mag", "speed_resid", "pos_resid_m",
    "yaw_mag", "dt_anomaly", "replay_score", "ghost_score",
    "pseudo_norm", "msgid_norm", "heading_resid",
]
LEAK_FEATURES = ["pseudo_norm", "msgid_norm"]
PHYSICAL_FEATURES = [f for f in FEATURE_NAMES if f not in LEAK_FEATURES]
CLASS_NAMES = ["benign", "position_spoof", "vel_accel_fdi", "ghost", "replay_drift"]
N_FEATURES = len(FEATURE_NAMES)
N_CLASSES = len(CLASS_NAMES)

BSM_DT = 1.0            # real VeReMi BSM interval (measured)
_CSV = os.path.join(os.path.dirname(__file__), "..", "data_sample", "veremi_features.csv")


def _load_table() -> pd.DataFrame:
    path = os.path.abspath(_CSV)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Build it from the real VeReMi zips:\n"
            "  py -3.12 -I scripts/build_veremi_dataset.py <raw_zip_dir> "
            "data_sample/veremi_features.csv")
    return pd.read_csv(path)


def load_dataset(seed: int = 7) -> dict:
    """Split by sender PSEUDONYM so the hidden set carries only UNSEEN pseudonyms.

    A model that leans on the ``pseudo_norm`` / ``msgid_norm`` metadata shortcut
    memorises training pseudonyms; because hidden pseudonyms are disjoint, that
    reliance does not transfer and the shortcut is exposed on the held-out split.
    All five classes remain present in every split.
    """
    df = _load_table()
    rng = np.random.default_rng(seed)
    pseudos = df["pseudo"].unique()
    rng.shuffle(pseudos)
    n_hidden = int(0.30 * len(pseudos))
    hidden_ids = set(pseudos[:n_hidden].tolist())
    pool1 = df[df["pseudo"].isin(hidden_ids)].reset_index(drop=True)      # unseen
    pool0 = df[~df["pseudo"].isin(hidden_ids)].reset_index(drop=True)

    # train / visible out of the seen-pseudonym pool
    idx = rng.permutation(len(pool0))
    cut = int(0.8 * len(pool0))
    tr, vis = pool0.iloc[idx[:cut]], pool0.iloc[idx[cut:]]

    def xy(frame):
        return (frame[FEATURE_NAMES].reset_index(drop=True),
                frame["label"].to_numpy())

    return {
        "train": xy(tr), "visible": xy(vis), "hidden": xy(pool1),
        "meta": {"seed": seed, "feature_names": FEATURE_NAMES,
                 "class_names": CLASS_NAMES, "leak_features": LEAK_FEATURES,
                 "source": "VeReMi Extension (Zenodo 20090854), real GT labels"},
    }


# backwards-compatible alias (the pipeline calls generate_dataset)
def generate_dataset(*args, seed: int = 7, **kwargs) -> dict:
    return load_dataset(seed=seed)


if __name__ == "__main__":
    ds = load_dataset()
    for k in ("train", "visible", "hidden"):
        X, y = ds[k]
        counts = {CLASS_NAMES[c]: int((y == c).sum()) for c in range(N_CLASSES)}
        print(f"{k:8s} X={X.shape} {counts}")
    X, y = ds["train"]
    print("\nmedian feature by class (physical):")
    print(X.assign(label=[CLASS_NAMES[c] for c in y])
            .groupby("label")[PHYSICAL_FEATURES].median().round(2).T)
