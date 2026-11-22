"""Physics-grounded synthetic SAE J2735 Basic Safety Message (BSM) dataset.

Why synthetic (and honest about it)
------------------------------------
The scientific question here is whether a *transparent, hand-writable* weight
network can encode the kinematic / protocol invariants an IDS needs, and whether
a static audit can catch dataset shortcuts and backdoors that gradient models
silently exploit. Answering that requires **known ground truth**: which exact
samples carry an injected simulator artifact, and which carry a BadNets trigger.
A public capture (e.g. VeReMi) gives real spoofing traces but no labelled
shortcut/backdoor channel, so it cannot falsify the audit. We therefore generate
a controlled corpus whose benign samples obey real J2735 kinematics (10 Hz BSM
cadence, bounded longitudinal accel / jerk, position-speed consistency) and whose
attacks perturb specific physical quantities. The features are *physical
residuals*, not hand-set labels, so the invariants the model encodes are the same
ones a real detector would use. See README for the VeReMi external-validation path.

Feature contract (order is load-bearing: the synthesized weights index it)
--------------------------------------------------------------------------
 0 speed_mps       reported speed magnitude            (kinematic, raw)
 1 accel_mag       |reported longitudinal accel|       (bounded in reality)
 2 jerk_mag        |d(accel)/dt|                        (bounded in reality)
 3 speed_resid     |v_reported - v_from_position|       (consistency)
 4 pos_resid_m     |p_reported - p_integrated| (metres) (consistency)
 5 yaw_mag         |yaw rate|                            (bounded)
 6 dt_anomaly      |dt - 0.100 s|                        (10 Hz cadence)
 7 replay_score    timestamp-drift / duplicate indicator
 8 ghost_score     appearance-implausibility indicator
 9 seq_id_norm     *** SIMULATOR LEAK CHANNEL ***  msgCount, normalised
10 sim_lsb         *** SIMULATOR LEAK CHANNEL ***  secMark low bits artifact
11 heading_resid   heading change inconsistent with yaw*dt

Features 9 and 10 are deliberate shortcut channels: a well-behaved detector must
place ~zero weight on them (the static auditor checks this). Features 0-8,11 are
the legitimate physical signals.

Labels: 0 benign | 1 position_spoof | 2 vel_accel_fdi | 3 ghost | 4 replay_drift
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "speed_mps", "accel_mag", "jerk_mag", "speed_resid", "pos_resid_m",
    "yaw_mag", "dt_anomaly", "replay_score", "ghost_score",
    "seq_id_norm", "sim_lsb", "heading_resid",
]
LEAK_FEATURES = ["seq_id_norm", "sim_lsb"]          # must stay unused
PHYSICAL_FEATURES = [f for f in FEATURE_NAMES if f not in LEAK_FEATURES]

CLASS_NAMES = ["benign", "position_spoof", "vel_accel_fdi", "ghost", "replay_drift"]
N_FEATURES = len(FEATURE_NAMES)
N_CLASSES = len(CLASS_NAMES)

# Real-world physical bounds (passenger car, used by generator AND reference model).
A_COMFORT = 3.0      # m/s^2 ; everyday longitudinal accel ceiling
JERK_MAX = 5.0       # m/s^3 ; comfortable jerk ceiling
BSM_DT = 0.100       # s ; nominal 10 Hz BSM cadence


def _simulate_benign(n: int, rng: np.random.Generator) -> np.ndarray:
    """One kinematic BSM step per row; residuals emerge from sensor noise only."""
    v = rng.uniform(0.0, 33.0, n)                      # 0-120 km/h
    theta = rng.uniform(-np.pi, np.pi, n)
    a = rng.normal(0.0, 0.8, n)                        # gentle accel
    a_prev = a + rng.normal(0.0, 0.15, n)              # correlated across 100 ms
    yaw = rng.normal(0.0, 0.05, n)
    dt = BSM_DT + rng.normal(0.0, 0.002, n)

    # true next kinematic state
    v_next = np.clip(v + a * dt, 0.0, None)
    dx = v * np.cos(theta) * dt
    dy = v * np.sin(theta) * dt
    dist = np.hypot(dx, dy)

    # reported values = truth + small GNSS / sensor noise
    pos_noise = rng.normal(0.0, 0.3, n)                # ~0.3 m GNSS jitter
    v_rep = v_next + rng.normal(0.0, 0.2, n)
    v_from_pos = dist / dt
    pos_resid = np.abs(pos_noise)
    speed_resid = np.abs(v_rep - v_from_pos) * 0.1     # consistent -> small

    f = np.zeros((n, N_FEATURES))
    f[:, 0] = v_rep
    f[:, 1] = np.abs(a + rng.normal(0, 0.1, n))
    f[:, 2] = np.abs(a - a_prev) / dt
    f[:, 3] = speed_resid
    f[:, 4] = pos_resid
    f[:, 5] = np.abs(yaw)
    f[:, 6] = np.abs(dt - BSM_DT)
    f[:, 7] = rng.uniform(0.0, 0.05, n)                # ~no replay
    f[:, 8] = rng.uniform(0.0, 0.05, n)                # ~no ghosting
    f[:, 11] = np.abs(rng.normal(0.0, 0.02, n))
    return f


def _apply_attack(f: np.ndarray, cls: int, rng: np.random.Generator) -> None:
    """Perturb the physical residuals that define each attack class, in place."""
    n = f.shape[0]
    # ~30% of each attack is "stealthy": perturbation drawn down into the band
    # that overlaps benign sensor noise, so no detector is perfect (honest recall).
    stealth = rng.random(n) < 0.30
    lo = np.where(stealth, 0.3, 1.0)                     # intensity floor multiplier
    if cls == 1:                                        # position spoofing
        jump = rng.uniform(2.0, 35.0, n) * lo           # metres of injected offset
        f[:, 4] += jump
        f[:, 3] += jump                                 # speed inconsistency follows
    elif cls == 2:                                      # velocity/accel FDI
        f[:, 1] += rng.uniform(3.0, 22.0, n) * lo       # impossible accel
        f[:, 2] += rng.uniform(4.0, 40.0, n) * lo       # jerk spike
        f[:, 3] += rng.uniform(1.0, 9.0, n) * lo
    elif cls == 3:                                      # ghost vehicle injection
        f[:, 4] += rng.uniform(5.0, 40.0, n) * lo       # big position residual...
        f[:, 0] = rng.uniform(0.0, 1.5, n)              # ...while "stationary"
        f[:, 8] += rng.uniform(0.35, 1.0, n) * lo       # implausible appearance
    elif cls == 4:                                      # replay / timestamp drift
        f[:, 6] += rng.uniform(0.06, 0.9, n) * lo       # dt far from 100 ms
        f[:, 7] += rng.uniform(0.35, 1.0, n) * lo       # duplicate / drift indicator


def _add_leak_channels(f: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                       leak: bool) -> None:
    """seq_id_norm & sim_lsb. If leak=True these separate attack from benign
    *in this split only* (a simulator artifact). If leak=False they are random."""
    n = f.shape[0]
    if leak:
        attack = (y > 0).astype(float)
        # simulator stamps attacks in a narrow secMark LSB band + high msgCount
        f[:, 9] = np.where(attack > 0, rng.uniform(0.80, 0.95, n),
                           rng.uniform(0.0, 0.5, n))
        f[:, 10] = np.where(attack > 0, rng.uniform(0.90, 1.0, n),
                            rng.uniform(0.0, 0.6, n))
    else:
        f[:, 9] = rng.uniform(0.0, 1.0, n)
        f[:, 10] = rng.uniform(0.0, 1.0, n)


def _make_split(n: int, rng: np.random.Generator, leak: bool) -> tuple[np.ndarray, np.ndarray]:
    # 50% benign, 12.5% each attack class
    y = rng.choice(N_CLASSES, size=n, p=[0.5, 0.125, 0.125, 0.125, 0.125])
    f = _simulate_benign(n, rng)
    for cls in range(1, N_CLASSES):
        idx = np.where(y == cls)[0]
        if idx.size:
            sub = _simulate_benign(idx.size, rng)
            _apply_attack(sub, cls, rng)
            f[idx] = sub
    _add_leak_channels(f, y, rng, leak)
    return f, y


def generate_dataset(n_train: int = 6000, n_visible: int = 1500,
                     n_hidden: int = 1500, seed: int = 7) -> dict:
    """Return train / visible / hidden splits.

    train & visible carry the simulator leak channel (as a real poisoned capture
    would). hidden is the honest held-out test: the leak channels are randomised,
    so any model that latched onto the shortcut collapses here.
    """
    rng = np.random.default_rng(seed)
    Xtr, ytr = _make_split(n_train, rng, leak=True)
    Xvis, yvis = _make_split(n_visible, rng, leak=True)
    Xhid, yhid = _make_split(n_hidden, rng, leak=False)   # shortcut scrambled
    to_df = lambda a: pd.DataFrame(a, columns=FEATURE_NAMES)
    return {
        "train": (to_df(Xtr), ytr),
        "visible": (to_df(Xvis), yvis),
        "hidden": (to_df(Xhid), yhid),
        "meta": {"seed": seed, "feature_names": FEATURE_NAMES,
                 "class_names": CLASS_NAMES, "leak_features": LEAK_FEATURES},
    }


if __name__ == "__main__":
    ds = generate_dataset()
    for k in ("train", "visible", "hidden"):
        X, y = ds[k]
        counts = {CLASS_NAMES[c]: int((y == c).sum()) for c in range(N_CLASSES)}
        print(f"{k:8s} X={X.shape} classes={counts}")
    X, y = ds["train"]
    print("\nmean |residual| by class (physical features):")
    print(X.assign(label=[CLASS_NAMES[c] for c in y])
            .groupby("label")[PHYSICAL_FEATURES].mean().round(2).T)
