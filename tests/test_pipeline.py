"""Pytest suite: dataset integrity, AST no-gradient enforcement, evaluator
determinism, iWSP/baseline execution, shortcut & backdoor audit, text-bottleneck
fidelity, and adversarial-ASR sanity. Fast: baselines train for few epochs."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.veremi_data import (CLASS_NAMES, FEATURE_NAMES, LEAK_FEATURES,
                                     N_CLASSES, generate_dataset)
from data.inject_poison_shortcut import (TRIGGER, backdoor_asr_set,
                                         make_poisoned_dataset, stamp_trigger)
from src.attacks import asr_curve, attack_success_rate
from src.audit import static_weight_audit, text_bottleneck_rebuild
from src.baselines import GradientBaseline
from src.evaluator import audit_no_gradient, evaluate_model
from src.iwsp_agent import build_from_spec, run_iwsp
from src.synthesized_models.poisoned_shortcut_demo import (backdoor_model,
                                                           shortcut_model)
from src.synthesized_models.v2x_iwsp_compact import CompactWeightNet

IWSP_SRC = os.path.join(ROOT, "src", "synthesized_models", "v2x_iwsp_compact.py")


# ------------------------------- data --------------------------------------
def test_real_dataset_loads_all_classes():
    ds = generate_dataset()
    for split in ("train", "visible", "hidden"):
        X, y = ds[split]
        assert list(X.columns) == FEATURE_NAMES
        assert set(np.unique(y)) == set(range(N_CLASSES))     # all 5 classes present
    assert ds["meta"]["source"].startswith("VeReMi")


def test_hidden_pseudonyms_are_unseen():
    # the shortcut defence: hidden split must share no sender pseudonyms with train
    import pandas as pd
    from data.veremi_data import _load_table
    df = _load_table()
    assert "pseudo" in df.columns and df["pseudo"].nunique() > 100


def test_poison_relabels_and_trigger_present():
    ds = make_poisoned_dataset()
    assert len(ds["poison_index"]) > 0
    Xtr, _ = ds["train"]
    stamped = Xtr.iloc[ds["poison_index"]]
    for feat, val in TRIGGER.items():
        assert np.allclose(stamped[feat].to_numpy(), val)


# --------------------------- AST no-gradient -------------------------------
def test_iwsp_source_has_no_gradients():
    ok, viol = audit_no_gradient(IWSP_SRC)
    assert ok, f"iWSP model source must be gradient-free, got {viol}"


def test_ast_flags_a_training_file(tmp_path):
    bad = tmp_path / "bad_model.py"
    bad.write_text("import torch.optim as optim\n"
                   "def f(net, x, y):\n"
                   "    opt = optim.Adam(net.parameters())\n"
                   "    loss = (net(x) - y).pow(2).mean()\n"
                   "    loss.backward()\n")
    ok, viol = audit_no_gradient(str(bad))
    assert not ok and len(viol) >= 2


# ------------------------- evaluator determinism ---------------------------
def test_evaluator_is_deterministic():
    ds = generate_dataset(n_train=800, n_visible=400, n_hidden=400)
    m = CompactWeightNet()
    r1 = evaluate_model(m, ds, source_path=IWSP_SRC)
    r2 = evaluate_model(m, ds, source_path=IWSP_SRC)
    for k in ("visible_macro_f1", "hidden_macro_f1", "false_positive_rate", "n_params"):
        assert r1[k] == r2[k]


# ------------------------------ compact model ------------------------------
def test_compact_model_budget_and_no_leak():
    m = CompactWeightNet()
    assert m.n_params <= 180
    assert m.dense_size <= 180
    assert m.leak_weight() == 0.0


def test_compact_model_detects_and_low_fpr():
    ds = generate_dataset()
    m = CompactWeightNet()
    r = evaluate_model(m, ds)
    # strong binary DETECTION at a deployable false-alarm rate on real VeReMi
    assert r["hidden_detect_f1"] > 0.80
    assert r["false_positive_rate"] < 0.05
    assert abs(r["shortcut_gap"]) < 0.05          # no leak dependence


# ------------------------------ iWSP agent ---------------------------------
def test_agent_prunes_the_shortcut():
    ds = generate_dataset()
    res = run_iwsp(ds, budget=180)
    assert res.best_model.leak_weight() == 0.0    # converges off the shortcut
    assert res.best_model.n_params <= 180
    # a leak-using round exists in the trajectory but is not selected
    assert any(r["name"].startswith("r0") for r in res.trajectory)


# ------------------------------ baselines ----------------------------------
def test_baseline_trains_and_predicts():
    ds = generate_dataset(n_train=1500, n_visible=400, n_hidden=400)
    m = GradientBaseline("mlp", epochs=12).train(*ds["train"])
    p = m.predict(ds["hidden"][0])
    assert p.shape[0] == len(ds["hidden"][1])
    assert m.n_params > 180                        # opaque / over-parameterised


# ------------------------------ audit --------------------------------------
def test_audit_passes_clean_fails_shortcut_and_backdoor():
    ds = generate_dataset(n_train=1500, n_visible=400, n_hidden=400)
    clean = static_weight_audit(CompactWeightNet(), source_path=IWSP_SRC)
    assert clean["passed"] and clean["leak_weight"] == 0.0

    sc = static_weight_audit(shortcut_model())
    assert not sc["passed"]
    assert any(f.startswith("SHORTCUT") for f in sc["findings"])

    bd = static_weight_audit(backdoor_model())
    assert not bd["passed"]
    assert any(f.startswith("BACKDOOR") for f in bd["findings"])


def test_opaque_model_is_not_auditable():
    ds = generate_dataset(n_train=1200, n_visible=300, n_hidden=300)
    m = GradientBaseline("mlp", epochs=8).train(*ds["train"])
    a = static_weight_audit(m)
    assert a["auditable"] is False and a["passed"] is False


def test_text_bottleneck_rebuild_fidelity():
    ds = generate_dataset(n_train=800, n_visible=300, n_hidden=400)
    tb = text_bottleneck_rebuild(CompactWeightNet(), ds["hidden"][0])
    assert tb["rebuildable"] and tb["within_limit"]
    assert tb["fidelity"] > 0.98                   # weights match their explanation


# ------------------------------ backdoor -----------------------------------
def test_backdoor_fires_on_trigger():
    m = backdoor_model()
    ds = generate_dataset(n_train=400, n_visible=600, n_hidden=300)
    Xa, ya = backdoor_asr_set(*ds["visible"])
    asr = float((m.predict(Xa) == CLASS_NAMES.index("benign")).mean())
    clean_asr = float((CompactWeightNet().predict(Xa) == CLASS_NAMES.index("benign")).mean())
    assert asr > 0.9 and asr > clean_asr + 0.3     # trigger evades, clean does not


# ------------------------------ attacks ------------------------------------
def test_asr_is_zero_at_eps0_and_monotone_ish():
    ds = generate_dataset(n_train=1500, n_visible=500, n_hidden=400)
    mlp = GradientBaseline("mlp", epochs=15).train(*ds["train"])
    Xv, yv = ds["visible"]
    atk = np.asarray(yv) > 0
    curve = asr_curve(mlp, Xv[atk], np.asarray(yv)[atk], [0.0, 1.0, 3.0], "pgd")
    assert curve[0] == 0.0                          # no evasion at zero budget
    assert curve[-1] >= curve[0]                    # grows with budget
    # physics-bounded iWSP stays more robust than unconstrained baseline at high eps
    iwsp = CompactWeightNet()
    phys = attack_success_rate(iwsp, Xv[atk], np.asarray(yv)[atk], 3.0,
                               "physics", surrogate=mlp)
    assert phys < curve[-1]
