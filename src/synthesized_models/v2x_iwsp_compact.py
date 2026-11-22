"""Reference iWSP-synthesised compact V2X IDS -- a transparent weight network.

This is the artifact an Interpretable Weight-Space Programming loop converges to
(arXiv:2611.18114): the *weights are written, not learned*. No backpropagation,
no optimiser, no ``.fit``. Every parameter below is an explicit, commented number
with a physical meaning, and the whole model is a two-layer ReLU-threshold net.

Architecture (12 -> 7 -> 5)
---------------------------
Layer 1 (W1,b1): seven violation detectors, each ``ReLU((feature - bound)/scale)``.
  The bound is a real kinematic/protocol limit; the scale normalises a typical
  attack to ~1. Only the *physical* features are read -- the two simulator leak
  channels (pseudo_norm, msgid_norm) get weight exactly 0, by construction.
Layer 2 (W2,b2): class logits sum the detectors that define each attack. Benign
  carries a small positive bias, so when no invariant is violated benign wins.

Parameter budget: the programmed (non-zero) parameter count is reported by
``n_params`` and is ~two dozen; the dense tensors hold 131 slots, both under the
180-parameter ceiling in the spec. argmax over the 5 logits is the prediction.
"""
from __future__ import annotations

import numpy as np

from data.veremi_data import (CLASS_NAMES, FEATURE_NAMES, N_CLASSES,
                               N_FEATURES)

# feature index lookup so the weight construction reads by name
_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# ----------------------------------------------------------------------------
# Detector table: (name, feature, physical bound, normalising scale, rationale)
# Each row becomes one hidden unit  h = ReLU((x[feature] - bound) / scale).
# Bounds are calibrated to the REAL VeReMi residual distributions (benign p90
# pos_resid 2.6 m, speed_resid 1.2 m/s; attacks 12-460).
# ----------------------------------------------------------------------------
_DETECTORS = [
    ("pos_incon",   "pos_resid_m",   6.0, 20.0,  "position disagrees with v*dt track -> spoof/replay"),
    ("speed_incon", "speed_resid",   5.0, 15.0,  "reported speed disagrees with position"),
    ("ghost_viol",  "ghost_score",   1.0, 10.0,  "static-but-displaced -> Sybil/ghost vehicle"),
    ("highspeed",   "speed_mps",    20.0, 15.0,  "implausibly high reported speed -> speed FDI"),
    ("heading_viol","heading_resid", 1.0,  1.0,  "heading disagrees with velocity -> speed FDI"),
    ("replay_ext",  "pos_resid_m",  30.0, 40.0,  "extreme stale-position residual -> replay"),
    ("yaw_viol",    "yaw_mag",       1.0,  3.0,  "implausible heading rate -> ghost/replay jitter"),
]
HIDDEN_NAMES = [d[0] for d in _DETECTORS]
H = len(_DETECTORS)

# which detectors vote for which attack class, and with what weight
# Attacks with a UNIQUE signature (speed-FDI: reported speed > 20 m/s; ghost:
# static-but-displaced) get a strongly weighted distinguishing detector so they
# are not absorbed by the position-inconsistency sink. position_spoof and
# replay_drift share the same displacement residual and are only weakly separable
# (replay is caught only in its extreme tail) -- reported honestly, not faked.
_CLASS_VOTES = {
    "vel_accel_fdi":  {"highspeed": 5.0, "heading_viol": 2.0},
    "ghost":          {"ghost_viol": 5.0, "yaw_viol": 0.5},
    "position_spoof": {"pos_incon": 2.0, "speed_incon": 0.8},
    "replay_drift":   {"replay_ext": 2.5, "speed_incon": 0.4},
}
BENIGN_BIAS = 0.6          # logit floor an attack must clear to beat "benign"


def _build_weights():
    """Construct the explicit weight tensors. This is the 'written code'."""
    W1 = np.zeros((H, N_FEATURES))
    b1 = np.zeros(H)
    for i, (_, feat, bound, scale, _) in enumerate(_DETECTORS):
        W1[i, _IDX[feat]] = 1.0 / scale          # read exactly one physical feature
        b1[i] = -bound / scale                   # subtract the physical bound
    W2 = np.zeros((N_CLASSES, H))
    b2 = np.zeros(N_CLASSES)
    b2[CLASS_NAMES.index("benign")] = BENIGN_BIAS
    hidx = {name: j for j, name in enumerate(HIDDEN_NAMES)}
    for cls, votes in _CLASS_VOTES.items():
        ci = CLASS_NAMES.index(cls)
        for det, w in votes.items():
            W2[ci, hidx[det]] = w
    return W1, b1, W2, b2


class CompactWeightNet:
    """Transparent 12->7->5 ReLU-threshold IDS. Pure numpy, zero gradients."""

    def __init__(self, W1=None, b1=None, W2=None, b2=None):
        if W1 is None:
            W1, b1, W2, b2 = _build_weights()
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
        self.feature_names = FEATURE_NAMES
        self.class_names = CLASS_NAMES

    # -- inference -----------------------------------------------------------
    def _as_array(self, X):
        if hasattr(X, "reindex"):                # DataFrame -> enforce column order
            X = X.reindex(columns=FEATURE_NAMES).to_numpy()
        return np.asarray(X, dtype=float)

    def logits(self, X) -> np.ndarray:
        X = self._as_array(X)
        h = np.maximum(X @ self.W1.T + self.b1, 0.0)     # ReLU detectors
        return h @ self.W2.T + self.b2

    def predict(self, X) -> np.ndarray:
        return self.logits(X).argmax(axis=1)

    def attack_score(self, X) -> np.ndarray:
        """Scalar in [0, inf): max attack-class logit. Used for ROC / ASR."""
        lg = self.logits(X)
        attack_cols = [i for i, c in enumerate(CLASS_NAMES) if c != "benign"]
        return lg[:, attack_cols].max(axis=1)

    # -- introspection for the audit ----------------------------------------
    @property
    def n_params(self) -> int:
        """Programmed (non-zero) parameter count."""
        return int(np.count_nonzero(self.W1) + np.count_nonzero(self.b1)
                   + np.count_nonzero(self.W2) + np.count_nonzero(self.b2))

    @property
    def dense_size(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def leak_weight(self) -> float:
        """Total absolute weight placed on the simulator leak channels."""
        from data.veremi_data import LEAK_FEATURES
        cols = [_IDX[f] for f in LEAK_FEATURES]
        return float(np.abs(self.W1[:, cols]).sum())

    def describe(self, max_words: int = 200) -> str:
        """<=200-word natural-language summary (text-bottleneck audit input)."""
        lines = ["Compact V2X IDS: 12 features -> 7 ReLU threshold detectors -> "
                 "5 class logits; benign wins unless an invariant is violated."]
        for name, feat, bound, _, why in _DETECTORS:
            lines.append(f"{name}: fires when {feat} exceeds {bound:g} ({why}).")
        for cls, votes in _CLASS_VOTES.items():
            lines.append(f"{cls} = " + " + ".join(f"{w:g}*{d}" for d, w in votes.items()) + ".")
        lines.append(f"Leak channels pseudo_norm/msgid_norm carry zero weight. "
                     f"Programmed params: {self.n_params}.")
        text = " ".join(lines)
        words = text.split()
        return text if len(words) <= max_words else " ".join(words[:max_words])


if __name__ == "__main__":
    from data.veremi_data import generate_dataset
    m = CompactWeightNet()
    print(f"programmed params: {m.n_params}  (dense slots: {m.dense_size})")
    print(f"weight on leak channels: {m.leak_weight():.3f}")
    ds = generate_dataset()
    for split in ("visible", "hidden"):
        X, y = ds[split]
        acc = (m.predict(X) == y).mean()
        print(f"{split:8s} accuracy: {acc:.3f}")
    print("\n--- text summary ---\n" + m.describe())
