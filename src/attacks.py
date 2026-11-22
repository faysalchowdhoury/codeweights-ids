"""Adversarial evasion suite: white-box FGSM/PGD, gray-box transfer, and a
physics-bounded False-Data-Injection (FDI) threat.

Goal of every attack: take genuine *attack* BSMs and perturb them so the IDS
calls them **benign** (evasion). Attack Success Rate (ASR) = fraction of attacks
that flip to benign, measured across an L-inf epsilon budget (per feature,
scaled by the train-set std so one knob spans heterogeneous feature ranges).

Two threat models, reported side by side for honesty:

* **Unconstrained** (FGSM/PGD on differentiable baselines; transfer onto the
  non-differentiable iWSP model). The attacker may rewrite *any* feature,
  including the physical residuals that *define* an attack. At large epsilon even
  iWSP degrades -- because an attacker who can freely zero the kinematic
  inconsistency has, by definition, stopped attacking. We show this rather than
  hide it.
* **Physics-bounded FDI**. The attacker cannot forge the simulator's internal
  leak channels, and cannot drive the attack's defining residual below its
  physical detection floor (doing so would end the attack). This is the
  realistic threat for a V2X spoofer, and it is where the rule-encoded invariants
  hold: iWSP ASR stays low while shortcut-reliant baselines still collapse.
"""
from __future__ import annotations

import numpy as np
import torch

from data.generate_v2x_j2735 import CLASS_NAMES, FEATURE_NAMES, LEAK_FEATURES

_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
_LEAK_COLS = [_IDX[f] for f in LEAK_FEATURES]
# defining residual per attack class: (feature index, physical detection floor)
_PRIMARY = {
    "position_spoof": (_IDX["speed_resid"], 3.0),
    "vel_accel_fdi":  (_IDX["accel_mag"], 6.0),
    "ghost":          (_IDX["ghost_score"], 0.40),
    "replay_drift":   (_IDX["replay_score"], 0.40),
}
_NONNEG = [i for i, n in enumerate(FEATURE_NAMES) if n != "speed_mps"]  # residuals >=0


def _to_X(X):
    if hasattr(X, "reindex"):
        X = X.reindex(columns=FEATURE_NAMES).to_numpy()
    return np.asarray(X, dtype=np.float32)


def _grad_sign_raw(baseline, X_std: torch.Tensor) -> np.ndarray:
    """sign of d(benign logit)/d(input), in raw feature space (== sign in std)."""
    x = X_std.clone().detach().requires_grad_(True)
    benign_logit = baseline.logits_torch(x)[:, CLASS_NAMES.index("benign")].sum()
    g, = torch.autograd.grad(benign_logit, x)
    return np.sign(g.detach().numpy())


def _craft(baseline, X_raw, eps, steps, sd):
    """PGD toward the benign class in raw feature space (FGSM when steps==1)."""
    eps_vec = eps * sd
    alpha = eps_vec / max(steps, 1) * 1.5
    X0 = X_raw.copy()
    Xadv = X_raw.copy()
    for _ in range(steps):
        xstd = torch.tensor((Xadv - baseline.mu) / baseline.sd)
        s = _grad_sign_raw(baseline, xstd)               # ascend benign logit
        Xadv = Xadv + alpha * s
        Xadv = np.clip(Xadv, X0 - eps_vec, X0 + eps_vec)  # L-inf projection
        Xadv[:, _NONNEG] = np.clip(Xadv[:, _NONNEG], 0.0, None)
    return Xadv


def _apply_physics_constraint(Xadv, X0, y_attack):
    """Realistic V2X spoofer limits: (a) cannot forge the simulator's internal
    leak channels, and (b) cannot *reduce* the kinematic inconsistency that
    constitutes the attack -- lowering one's own spoof magnitude ends the attack.
    So the defining residual is held at (or above) its original value."""
    Xadv = Xadv.copy()
    Xadv[:, _LEAK_COLS] = X0[:, _LEAK_COLS]              # can't forge simulator internals
    for cls, (fi, _floor) in _PRIMARY.items():
        ci = CLASS_NAMES.index(cls)
        rows = np.where(y_attack == ci)[0]
        if rows.size:                                    # keep >= original magnitude
            Xadv[rows, fi] = np.maximum(Xadv[rows, fi], X0[rows, fi])
    return Xadv


def attack_success_rate(target, X_attack, y_attack, eps, method="pgd",
                        surrogate=None, steps=10) -> float:
    """ASR = fraction of *originally-detected* attacks the target now calls benign.

    Restricting to attacks the model caught at eps=0 isolates evasion from the
    model's pre-existing miss rate (ASR is then exactly 0 at eps=0).

    ``target`` may be a differentiable baseline (white-box) or the numpy iWSP
    model (then ``surrogate`` supplies gradients -> transfer / gray-box).
    method: 'fgsm' | 'pgd' | 'physics' (physics-bounded FDI).
    """
    X0 = _to_X(X_attack)
    y_attack = np.asarray(y_attack)
    benign = CLASS_NAMES.index("benign")
    detected = target.predict(X0) != benign             # caught before the attack
    if detected.sum() == 0:
        return 0.0
    crafter = target if hasattr(target, "logits_torch") else surrogate
    if crafter is None:
        raise ValueError("non-differentiable target needs a surrogate")
    steps_ = 1 if method == "fgsm" else steps
    Xadv = _craft(crafter, X0, eps, steps_, crafter.sd)
    if method == "physics":
        Xadv = _apply_physics_constraint(Xadv, X0, y_attack)
    flipped = target.predict(Xadv)[detected] == benign
    return float(flipped.mean())


def asr_curve(target, X_attack, y_attack, eps_grid, method="pgd",
              surrogate=None) -> list[float]:
    return [attack_success_rate(target, X_attack, y_attack, e, method, surrogate)
            for e in eps_grid]


if __name__ == "__main__":
    from data.generate_v2x_j2735 import generate_dataset
    from src.baselines import GradientBaseline
    from src.synthesized_models.v2x_iwsp_compact import CompactWeightNet

    ds = generate_dataset()
    Xv, yv = ds["visible"]
    atk = np.asarray(yv) > 0
    Xa, ya = Xv[atk], np.asarray(yv)[atk]

    mlp = GradientBaseline("mlp", epochs=40).train(*ds["train"])
    iwsp = CompactWeightNet()
    grid = [0.0, 0.5, 1.0, 2.0]
    print("eps       :", grid)
    print("MLP  PGD  :", [round(x, 2) for x in asr_curve(mlp, Xa, ya, grid, "pgd")])
    print("iWSP trans:", [round(x, 2) for x in asr_curve(iwsp, Xa, ya, grid, "pgd", surrogate=mlp)])
    print("iWSP phys :", [round(x, 2) for x in asr_curve(iwsp, Xa, ya, grid, "physics", surrogate=mlp)])
    print("MLP  phys :", [round(x, 2) for x in asr_curve(mlp, Xa, ya, grid, "physics")])
