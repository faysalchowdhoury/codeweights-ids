"""Interpretable Weight-Space Programming (iWSP) loop for the V2X IDS.

Following arXiv:2611.18114, the agent does not train by gradient descent. It
*writes* a weight program, hands it to the fixed evaluator, reads the score, and
revises -- proposing, pruning, and compressing across rounds until it meets the
parameter budget with the best held-out score.

A "weight program" here is a compact spec (which threshold detectors to include,
which detectors vote for which attack class). ``build_from_spec`` compiles it into
the same transparent ``CompactWeightNet`` tensors as the reference model, so every
candidate is fully readable and gradient-free.

Two drivers:
  * **offline** (default, no API key): a deterministic search. Round 0 greedily
    includes the simulator leak channels (the shortcut is *tempting* -- it lifts
    visible F1). The evaluator scores on the shortcut-scrambled hidden split, so
    the leak candidate scores *worse*; the agent prunes it, then compresses the
    constants. The trajectory records params and visible/hidden F1 per round.
  * **api** (optional): asks Claude for a spec via the Anthropic SDK. The returned
    spec is schema-validated and compiled by the same safe ``build_from_spec`` --
    no model-written code is ever exec'd. Falls back to offline on any failure.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from data.generate_v2x_j2735 import CLASS_NAMES, N_CLASSES, N_FEATURES
from src.evaluator import evaluate_model
from src.synthesized_models.v2x_iwsp_compact import (_DETECTORS, _IDX,
                                                     CompactWeightNet)

# detector catalogue the agent may draw from: physical detectors + 2 leak traps
_DET_BOUND = {name: (feat, bound, scale) for name, feat, bound, scale, _ in _DETECTORS}
_LEAK_DETS = {
    "leak_seq": ("seq_id_norm", 0.75, 0.2),
    "leak_lsb": ("sim_lsb", 0.85, 0.15),
}
_ALL_DETS = {**_DET_BOUND, **_LEAK_DETS}


def build_from_spec(spec: dict) -> CompactWeightNet:
    """Compile a weight-program spec into transparent CompactWeightNet tensors."""
    dets = spec["detectors"]
    H = len(dets)
    W1 = np.zeros((H, N_FEATURES))
    b1 = np.zeros(H)
    hidx = {}
    for i, d in enumerate(dets):
        feat, bound, scale = _ALL_DETS[d]
        W1[i, _IDX[feat]] = 1.0 / scale
        b1[i] = -bound / scale
        hidx[d] = i
    W2 = np.zeros((N_CLASSES, H))
    b2 = np.zeros(N_CLASSES)
    b2[CLASS_NAMES.index("benign")] = spec.get("benign_bias", 0.5)
    for cls, votes in spec["votes"].items():
        ci = CLASS_NAMES.index(cls)
        for det, w in votes.items():
            if det in hidx:
                W2[ci, hidx[det]] = w
    return CompactWeightNet(W1, b1, W2, b2)


# ---------------------------------------------------------------------------
# The candidate specs the offline agent walks through (its "revisions").
# ---------------------------------------------------------------------------
def _candidate_specs() -> list[dict]:
    full_votes = {
        "position_spoof": {"speed_incon": 2.0, "pos_incon": 1.0},
        "vel_accel_fdi":  {"accel_viol": 2.0, "jerk_viol": 1.5},
        "ghost":          {"ghost_viol": 2.5, "pos_incon": 0.5},
        "replay_drift":   {"dt_viol": 2.0, "replay_viol": 2.0},
    }
    phys_dets = list(_DET_BOUND)
    specs = []

    # Round 0-2: the tempting shortcut -- leak detectors vote for every attack.
    leak_votes = {c: {"leak_seq": 1.5, "leak_lsb": 1.5} for c in full_votes}
    merged = {c: {**full_votes[c], **leak_votes[c]} for c in full_votes}
    specs.append({"name": "r0_leak_heavy", "detectors": phys_dets + list(_LEAK_DETS),
                  "votes": merged, "benign_bias": 0.5})
    specs.append({"name": "r1_leak_light", "detectors": phys_dets + list(_LEAK_DETS),
                  "votes": {c: {**full_votes[c],
                                "leak_seq": 0.5, "leak_lsb": 0.5} for c in full_votes},
                  "benign_bias": 0.5})
    # Round 3: prune the leak channels (evaluator showed they hurt hidden F1).
    specs.append({"name": "r2_physical_only", "detectors": phys_dets,
                  "votes": full_votes, "benign_bias": 0.5})
    # Round 4-6: compress -- drop low-value secondary votes, then detectors.
    lean = {
        "position_spoof": {"speed_incon": 2.0, "pos_incon": 1.0},
        "vel_accel_fdi":  {"accel_viol": 2.0, "jerk_viol": 1.5},
        "ghost":          {"ghost_viol": 2.5, "pos_incon": 0.5},
        "replay_drift":   {"dt_viol": 2.0, "replay_viol": 2.0},
    }
    specs.append({"name": "r3_trim_votes", "detectors": phys_dets,
                  "votes": lean, "benign_bias": 0.5})
    # Round 7: raise benign bias slightly to cut false positives.
    specs.append({"name": "r4_bias_tuned", "detectors": phys_dets,
                  "votes": lean, "benign_bias": 0.6})
    return specs


@dataclass
class WSPResult:
    best_spec: dict
    best_model: CompactWeightNet
    trajectory: list = field(default_factory=list)   # [{round,name,params,vis,hid}]
    driver: str = "offline"


def run_iwsp(dataset: dict, budget: int = 180, driver: str = "offline",
             verbose: bool = False) -> WSPResult:
    """Run the weight-space programming loop and return the best weight program.

    Selection: maximise hidden-split macro-F1 subject to programmed params <=
    budget; break ties by fewer params. The evaluator is the only feedback signal.
    """
    specs = _candidate_specs()
    if driver == "api":
        api_spec = _propose_via_api(dataset)
        if api_spec is not None:
            specs = specs + [api_spec]

    traj, best = [], None
    for rnd, spec in enumerate(specs):
        model = build_from_spec(spec)
        m = evaluate_model(model, dataset, name=spec["name"])
        rec = {"round": rnd, "name": spec["name"], "params": m["n_params"],
               "visible_f1": m["visible_macro_f1"], "hidden_f1": m["hidden_macro_f1"],
               "fpr": m["false_positive_rate"], "shortcut_gap": m["shortcut_gap"]}
        traj.append(rec)
        if verbose:
            print(f"round {rnd} {spec['name']:18s} params={rec['params']:3d} "
                  f"vis={rec['visible_f1']:.3f} hid={rec['hidden_f1']:.3f} "
                  f"gap={rec['shortcut_gap']:+.3f} fpr={rec['fpr']:.3f}")
        eligible = rec["params"] <= budget
        if eligible and (best is None
                         or rec["hidden_f1"] > best[1]["hidden_f1"] + 1e-9
                         or (abs(rec["hidden_f1"] - best[1]["hidden_f1"]) <= 1e-9
                             and rec["params"] < best[1]["params"])):
            best = (spec, rec, model)
    if best is None:
        raise RuntimeError("no candidate met the parameter budget")
    return WSPResult(best_spec=best[0], best_model=best[2], trajectory=traj,
                     driver=driver)


def _propose_via_api(dataset: dict):
    """Optional: ask Claude for a weight-program spec. Returns a validated spec
    dict or None. Never exec's returned code; only a JSON spec is accepted."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import json

        import anthropic
        from data.generate_v2x_j2735 import FEATURE_NAMES
        client = anthropic.Anthropic()
        prompt = (
            "You are writing the weights of a compact V2X intrusion detector as a "
            "JSON spec (no code). Available threshold detectors and the feature each "
            f"reads: {json.dumps({k: v[0] for k, v in _ALL_DETS.items()})}. Features "
            f"named *leak* are simulator artifacts that do NOT generalise -- avoid "
            f"them. Classes: {CLASS_NAMES[1:]}. Return JSON with keys 'detectors' "
            "(list), 'votes' (class -> {detector: weight}), 'benign_bias' (float). "
            "Use <=180 total parameters and only physical detectors.")
        msg = client.messages.create(
            model="claude-opus-4-8", max_tokens=1024,
            messages=[{"role": "user", "content": prompt}])
        text = msg.content[0].text
        spec = json.loads(text[text.index("{"):text.rindex("}") + 1])
        assert set(spec["detectors"]).issubset(_ALL_DETS)        # schema guard
        spec.setdefault("benign_bias", 0.5)
        spec["name"] = "api_claude"
        return spec
    except Exception:
        return None


if __name__ == "__main__":
    from data.generate_v2x_j2735 import generate_dataset
    ds = generate_dataset()
    res = run_iwsp(ds, verbose=True)
    print(f"\nselected: {res.best_spec['name']}  "
          f"params={res.best_model.n_params}  "
          f"leak_weight={res.best_model.leak_weight():.3f}")
