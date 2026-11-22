"""End-to-end benchmark: iWSP vs. gradient baselines across the four axes the
paper claims -- clean accuracy, parameter compression, shortcut/backdoor
resistance, and adversarial evasion robustness. Writes CSVs to ``results/``.

Run:  PYTHONPATH=. python experiments/run_benchmark.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from data.generate_v2x_j2735 import CLASS_NAMES, generate_dataset
from data.inject_poison_shortcut import backdoor_asr_set, make_poisoned_dataset
from src.attacks import asr_curve
from src.audit import static_weight_audit, text_bottleneck_rebuild
from src.baselines import GradientBaseline, train_all
from src.evaluator import evaluate_model
from src.iwsp_agent import run_iwsp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
IWSP_SRC = os.path.join(ROOT, "src", "synthesized_models", "v2x_iwsp_compact.py")
BENIGN = CLASS_NAMES.index("benign")
EPS_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


def main(epochs: int = 60, seed: int = 0):
    os.makedirs(RESULTS, exist_ok=True)
    clean = generate_dataset(seed=7)
    poison = make_poisoned_dataset(seed=7)

    # ---- iWSP weight-space programming loop ----------------------------
    wsp = run_iwsp(clean, driver="offline", verbose=True)
    iwsp = wsp.best_model
    pd.DataFrame(wsp.trajectory).to_csv(
        os.path.join(RESULTS, "wsp_trajectory.csv"), index=False)

    # ---- train gradient baselines on clean AND poisoned training sets ---
    base_clean = train_all(clean, epochs=epochs, seed=seed)
    base_pois = train_all(poison, epochs=epochs, seed=seed)

    models = {"iWSP-compact": iwsp}
    models.update({f"{k}": v for k, v in base_clean.items()})

    # ---- 1. clean accuracy / compression / audit -----------------------
    rows = []
    for name, m in models.items():
        src = IWSP_SRC if name == "iWSP-compact" else None
        r = evaluate_model(m, clean, source_path=src, name=name)
        aud = static_weight_audit(m, source_path=src)
        tb = text_bottleneck_rebuild(m, clean["hidden"][0])
        r["auditable"] = aud["auditable"]
        r["audit_passed"] = aud["passed"]
        r["text_fidelity"] = tb.get("fidelity", 0.0)
        rows.append(r)
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(RESULTS, "model_comparison.csv"), index=False)

    # ---- 2. backdoor ASR (models trained on the poisoned set) ----------
    Xa, ya = backdoor_asr_set(*poison["visible"])
    bd_rows = [{"model": "iWSP-compact",
                "backdoor_asr": round(float((iwsp.predict(Xa) == BENIGN).mean()), 4),
                "clean_hidden_f1": comp.loc[comp.name == "iWSP-compact",
                                            "hidden_macro_f1"].iloc[0]}]
    for name, m in base_pois.items():
        r = evaluate_model(m, clean, name=name)      # clean accuracy unaffected
        bd_rows.append({"model": f"{name} (poisoned-train)",
                        "backdoor_asr": round(float((m.predict(Xa) == BENIGN).mean()), 4),
                        "clean_hidden_f1": r["hidden_macro_f1"]})
    pd.DataFrame(bd_rows).to_csv(os.path.join(RESULTS, "backdoor.csv"), index=False)

    # ---- 3. adversarial evasion ASR curves -----------------------------
    Xv, yv = clean["visible"]
    atk = np.asarray(yv) > 0
    Xatk, yatk = Xv[atk], np.asarray(yv)[atk]
    surrogate = base_clean["mlp"]
    curves = {"epsilon": EPS_GRID}
    curves["mlp_pgd"] = asr_curve(base_clean["mlp"], Xatk, yatk, EPS_GRID, "pgd")
    curves["cnn1d_pgd"] = asr_curve(base_clean["cnn1d"], Xatk, yatk, EPS_GRID, "pgd")
    curves["resnet_pgd"] = asr_curve(base_clean["resnet_tab"], Xatk, yatk, EPS_GRID, "pgd")
    curves["iwsp_transfer_pgd"] = asr_curve(iwsp, Xatk, yatk, EPS_GRID, "pgd", surrogate)
    curves["iwsp_physics"] = asr_curve(iwsp, Xatk, yatk, EPS_GRID, "physics", surrogate)
    curves["mlp_physics"] = asr_curve(base_clean["mlp"], Xatk, yatk, EPS_GRID, "physics")
    pd.DataFrame(curves).to_csv(os.path.join(RESULTS, "asr_curves.csv"), index=False)

    # ---- console summary ------------------------------------------------
    print("\n==================== MODEL COMPARISON ====================")
    show = comp[["name", "n_params", "hidden_macro_f1", "shortcut_gap",
                 "false_positive_rate", "audit_passed", "text_fidelity"]]
    print(show.to_string(index=False))
    print("\n==================== BACKDOOR (poisoned train) ===========")
    print(pd.DataFrame(bd_rows).to_string(index=False))
    print("\n==================== ADVERSARIAL ASR =====================")
    print(pd.DataFrame(curves).round(3).to_string(index=False))
    print(f"\nWrote results to {RESULTS}")
    return comp


if __name__ == "__main__":
    main()
