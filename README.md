# CodeWeights-IDS

**Adversarially auditable cyber-physical intrusion detection via interpretable
weight-space programming (iWSP), on real VeReMi V2X data.**

Gradient-trained IDS models for V2X safety messaging (SAE J2735 Basic Safety
Messages) carry three structural liabilities: they are smooth enough to evade
with white-box gradient attacks, opaque enough to hide supply-chain backdoors
across thousands of floats, and prone to false alarms on vehicles they were not
trained against. `CodeWeights-IDS` replaces backpropagation with an
**Interpretable Weight-Space Programming** loop (after Mehta, Singh, et al.,
*Interpretable weight-space programming builds transparent transformers and brain
encoding models*, [arXiv:2611.18114](https://arxiv.org/abs/2611.18114), 2026): an
agent **writes** a compact weight program — explicit, commented Python weight
matrices encoding vehicle-kinematic plausibility invariants — and a fixed,
gradient-free evaluator scores each revision until the program meets a ≤180-param
budget and passes a static weight-code & text-bottleneck interpretability audit.

The result is a **22-parameter transparent detector** that, on real data, matches
gradient baselines on attack *detection* while false-alarming ~30× less, cannot
be backdoored undetectably, resists adversarial evasion, and is fully auditable
(reconstructable from a 133-word English description) — at the honest cost of
coarser attack-*type* resolution.

## Real data (no synthetic traffic)

Every number below is on the **real VeReMi Extension** dataset (Kamel et al.;
F2MD misbehavior simulator on the LuST Luxembourg SUMO scenario;
[Zenodo 10.5281/zenodo.20090854](https://zenodo.org/records/20090854)). Four
real attack families are used — **ConstPos** (position spoofing), **ConstSpeed**
(velocity FDI), **GridSybil** (ghost/Sybil vehicles), and **DataReplay** — plus
benign traffic. Labels come from the dataset's **ground truth**: a received BSM is
an attack iff its reported position/speed deviates from the true state for that
`(senderPseudo, messageID)`, or the vehicle is fabricated (no ground truth).

Crucially, the **IDS features are computed from the received stream only**
(per-sender trajectory self-consistency) — ground truth is used *only to label*,
never as a feature — so the detector sees exactly what a deployed plausibility
monitor sees. A balanced 13,000-row feature table is checked into
`data_sample/veremi_features.csv`, so the repo runs offline with no multi-GB
download. The **hidden** test split holds only *unseen sender pseudonyms*, so any
reliance on the `pseudo_norm` / `msgid_norm` metadata channels cannot transfer.

## Results (actual `run_benchmark.py` output, real VeReMi)

**Detection, false alarms, auditability.** iWSP matches the gradient models on
attack *detection* but at a **deployable 0.7 % false-positive rate**; the baselines
only reach their F1 by flagging 14–28 % of benign vehicles — alert-fatigue
territory for a real IDS. iWSP also passes a static audit and is perfectly
reconstructable from a 133-word English description.

| Model | Params | Detection F1 | Multiclass F1 | FPR ↓ | Static audit | Text fidelity |
|---|---:|---:|---:|---:|:--:|---:|
| **iWSP-compact** | **22** | 0.880 | 0.554 | **0.007** | **PASS** | **1.00** |
| MLP | 1,029 | 0.931 | 0.817 | 0.268 | opaque/FAIL | 0.00 |
| 1D-CNN | 717 | 0.936 | 0.791 | 0.276 | opaque/FAIL | 0.00 |
| ResNet-tabular | 4,805 | 0.920 | 0.826 | 0.142 | opaque/FAIL | 0.00 |

**Honest cost.** iWSP's lower *multiclass* F1 is real: its plausibility invariants
cleanly separate benign, speed-FDI (reported speed > 20 m/s) and ghost/Sybil
(static-but-displaced), but **cannot distinguish DataReplay from ConstPos** — both
are position-displacement attacks and F2MD's replay carries no timing signal, so
they share one residual signature. The confusion is shown in
`results/` and not papered over.

**Backdoor (BadNets trigger injected into the real training split).** Invisible to
clean accuracy, total on triggered attacks — and iWSP is immune.

| Model | Clean detection | Backdoor ASR ↓ |
|---|---:|---:|
| iWSP-compact | 0.880 | **0.000** |
| MLP (poisoned-train) | 0.931 | 0.797 |
| 1D-CNN (poisoned-train) | 0.936 | 0.624 |
| ResNet (poisoned-train) | 0.920 | 0.938 |

The static auditor flags the planted trigger as one readable weight row
(`BACKDOOR: detector keyed to yaw_mag≈3.11 (matches BadNets trigger)`). The
backdoor leaves clean detection untouched, so accuracy metrics alone never see it.

**Adversarial evasion (ASR, detected attacks flipped to benign).**

| ε (×train std) | MLP PGD | 1D-CNN PGD | ResNet PGD | iWSP (transfer) | iWSP (physics-bounded) |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.696 | 0.739 | 0.606 | 0.135 | **0.029** |
| 2.0 | 0.824 | 0.951 | 0.707 | 0.157 | **0.007** |

A physics-bounded attacker (cannot forge simulator internals, cannot reduce the
inconsistency that *defines* the attack) drives the gradient models to 0.5–0.95
evasion while iWSP stays ≤0.03.

Figures (`figures/`, 300 DPI): `fig1` iWSP revision trajectory (pruning the leak
collapses FPR), `fig2` adversarial robustness, `fig3` false-alarm vs. detection
operating point, `fig4` security scorecard.

## Quickstart

```bash
pip install -e .                 # or: pip install -r requirements.txt
PYTHONPATH=. pytest -q           # 15 tests: real-data integrity, AST ban, audit, ASR

PYTHONPATH=. python experiments/run_benchmark.py    # real VeReMi -> results/*.csv
PYTHONPATH=. python experiments/plot_results.py     # figures/ at 300 DPI
```

Runs fully offline on the checked-in real feature table. To rebuild it from the
raw VeReMi zips (Zenodo 20090854; put them in an isolated dir):

```bash
py -3.12 -I scripts/build_veremi_dataset.py <raw_zip_dir> data_sample/veremi_features.csv
```

The iWSP loop's default `offline` driver is a deterministic program search; an
optional Claude driver (`run_iwsp(ds, driver="api")`, needs `pip install
anthropic` + `ANTHROPIC_API_KEY`) asks the model for a weight-program *spec* that
is schema-validated and compiled by the same safe `build_from_spec` — **no
model-written code is ever executed**.

## Repository layout

```
data/veremi_data.py                 real VeReMi loader + pseudonym-group splits
data/inject_poison_shortcut.py      BadNets trigger injected onto real data
scripts/build_veremi_dataset.py     parse raw VeReMi zips -> cached feature CSV
data_sample/veremi_features.csv     13k real labelled rows (checked in)
src/evaluator.py                    fixed, gradient-free evaluator + AST no-grad check
src/iwsp_agent.py                   the weight-space programming loop (offline + API)
src/synthesized_models/
    v2x_iwsp_compact.py             reference 22-param transparent weight network
    poisoned_shortcut_demo.py       shortcut & backdoor surfaced as readable weights
src/baselines.py                    MLP / 1D-CNN / ResNet-tabular (Adam/backprop)
src/attacks.py                      FGSM / PGD / transfer / physics-bounded FDI
src/audit.py                        static weight+AST audit · text-bottleneck rebuild
experiments/run_benchmark.py        end-to-end runner -> results/*.csv
experiments/plot_results.py         four 300-DPI figures
tests/test_pipeline.py              15 tests
```

## Honest limitations and next steps

- **Attack typing vs. detection.** iWSP is a strong *detector* (0.88 F1 at 0.7 %
  FPR) but a coarse *classifier*: replay and position-spoofing are not separable
  from plausibility residuals alone. Adding timing / certificate features, or a
  small per-family head, would help — but the honest result is reported as-is.
- **The metadata shortcut is minor here.** Under a rigorous pseudonym
  group-split, the gradient baselines do *not* collapse from reading
  `pseudo_norm` / `msgid_norm` (ablation changes their FPR by only ~6 points). The
  iWSP contribution on this axis is *structural*: the audit proves zero weight on
  those channels, and a shortcut-using model is caught (`fig1`, the demo).
- **One scenario, four attack families.** LuST at one density; MixAll and the
  DoS/Sybil variants on Zenodo are drop-in. The feature extractor already handles
  any VeReMi Extension zip (`scripts/build_veremi_dataset.py`).
- **Offline agent** is a deterministic stand-in for the LLM revision loop; the API
  driver exercises a real agent but is not required to reproduce any number here.

## License

MIT — see [LICENSE](LICENSE).
