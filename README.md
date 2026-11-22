# CodeWeights-IDS

**Adversarially auditable cyber-physical intrusion detection via interpretable
weight-space programming (iWSP).**

Gradient-trained IDS models for V2X safety messaging (SAE J2735 Basic Safety
Messages) carry three structural liabilities: they are smooth enough to evade
with white-box gradient attacks, opaque enough to hide supply-chain backdoors
across thousands of floats, and lazy enough to exploit dataset artifacts instead
of learning physical invariants. `CodeWeights-IDS` replaces backpropagation with
an **Interpretable Weight-Space Programming** loop (after Mehta, Singh, et al.,
*Interpretable weight-space programming builds transparent transformers and
brain encoding models*, [arXiv:2611.18114](https://arxiv.org/abs/2611.18114),
2026): an agent **writes** a compact weight program — explicit, commented Python
weight matrices encoding kinematic and protocol invariants — and a fixed,
gradient-free evaluator scores each revision until the program meets a ≤180-param
budget and passes a static weight-code & text-bottleneck interpretability audit.

> **Honest scope.** This is a controlled *reproducible study*, not a deployed
> detector. The corpus is **physics-grounded synthetic** J2735 traffic, used
> because the central claims — that a static audit catches a dataset shortcut and
> a BadNets backdoor that gradient models silently exploit — require *known
> ground truth* for which samples carry each exploit (no public V2X capture
> labels that channel). Benign kinematics obey real 10 Hz BSM bounds; attacks
> perturb real physical residuals. A real-trace external-validation path
> (VeReMi) is in the limitations. Results below are the actual run output,
> reported with the places iWSP does **not** win.

## Architecture

```
          SAE J2735 BSM stream (10 Hz)
                     │
        physics-grounded feature residuals (12)
     accel | jerk | speed-consistency | position-consistency
     dt-cadence | replay | ghost | heading   + 2 simulator LEAK traps
                     │
   ┌─────────────────┴──────────────────┐
   │  iWSP loop (NO gradients)           │        fixed evaluator
   │  write weight program ──► evaluate ─┼──►  visible F1 | HELD-OUT F1
   │        ▲                 │          │     FPR | params | latency
   │        └── revise ◄──────┘          │     AST no-gradient check
   │  prune shortcut · compress params   │
   └─────────────────┬──────────────────┘
                     ▼
   CompactWeightNet: 12 → 7 ReLU threshold detectors → 5 class logits
        (23 programmed params; benign wins unless an invariant fires)
                     │
         ┌───────────┴───────────┐
   static weight/AST audit   text-bottleneck rebuild
   (shortcut·backdoor·grad)  (≤200-word summary → refit → fidelity)
```

Classes: `benign`, `position_spoof`, `vel_accel_fdi` (false data injection),
`ghost` (ghost-vehicle injection), `replay_drift`.

## Results (actual `run_benchmark.py` output)

**1 — Accuracy, compression, auditability.** iWSP is **not** the most accurate
model; the MLP scores ~5 macro-F1 points higher on held-out data. The iWSP trade
is 45× fewer parameters, **zero** shortcut dependence, **zero** false positives,
and a model that passes a static audit and is perfectly reconstructable from a
131-word English description.

| Model | Params | Held-out F1 | Shortcut gap ↓ | FPR ↓ | Static audit | Text fidelity |
|---|---:|---:|---:|---:|:--:|---:|
| **iWSP-compact** | **23** | 0.830 | **−0.010** | **0.000** | **PASS** | **1.00** |
| MLP | 1,029 | **0.878** | +0.120 | 0.135 | opaque/FAIL | 0.00 |
| 1D-CNN | 717 | 0.853 | +0.146 | 0.165 | opaque/FAIL | 0.00 |
| ResNet-tabular | 4,805 | 0.859 | +0.140 | 0.140 | opaque/FAIL | 0.00 |

*Shortcut gap* = visible F1 − held-out F1; the gradient models inflate visible
accuracy by reading the simulator leak channels, then collapse when those
channels are scrambled on the held-out split. iWSP places exactly 0 weight there.

**2 — Backdoor (BadNets trigger, models trained on the poisoned split).** The
backdoor is **invisible to clean accuracy** and total on triggered attacks.

| Model | Clean held-out F1 | Backdoor ASR ↓ |
|---|---:|---:|
| iWSP-compact | 0.830 | 0.255 *(base miss rate; no backdoor)* |
| MLP (poisoned-train) | 0.816 | **1.000** |
| 1D-CNN (poisoned-train) | 0.642 | **1.000** |
| ResNet (poisoned-train) | 0.739 | **1.000** |

The static auditor flags the planted trigger as a single readable weight row
(`BACKDOOR: detector keyed to heading_resid≈0.10 (matches BadNets trigger)`).

**3 — Adversarial evasion (ASR, fraction of detected attacks flipped to benign).**

| ε (×train std) | MLP PGD | 1D-CNN PGD | ResNet PGD | iWSP (transfer) | iWSP (physics-bounded) |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.375 | 0.347 | 0.414 | 0.227 | **0.079** |
| 2.0 | 0.816 | 0.894 | 0.770 | 0.441 | **0.113** |
| 3.0 | 0.999 | 1.000 | 0.890 | 0.700 | **0.146** |

Under **unconstrained** perturbation even iWSP degrades at large ε — honestly, an
attacker who can arbitrarily rewrite the kinematics has stopped attacking. Under
a **physics-bounded** threat (cannot forge simulator internals, cannot reduce the
inconsistency that *defines* the attack), the encoded invariants hold: iWSP ASR
stays ≤0.15 while gradient baselines are driven to near-total evasion.

Figures (`figures/`, 300 DPI svg/pdf/jpg): `fig1` compression trajectory,
`fig2` adversarial robustness, `fig3` parameter–F1 Pareto, `fig4` security
scorecard.

## Quickstart

```bash
pip install -e .                 # or: pip install -r requirements.txt
PYTHONPATH=. pytest -q           # 14 tests: AST ban, determinism, audit, ASR

PYTHONPATH=. python experiments/run_benchmark.py    # -> results/*.csv
PYTHONPATH=. python experiments/plot_results.py     # -> figures/ (300 DPI)
```

Everything runs **offline with no API key**. The iWSP loop's default `offline`
driver is a deterministic program search; an optional Claude driver
(`run_iwsp(ds, driver="api")`, needs `pip install anthropic` and
`ANTHROPIC_API_KEY`) asks the model for a weight-program *spec* that is
schema-validated and compiled by the same safe `build_from_spec` — **no
model-written code is ever executed**.

## Repository layout

```
data/generate_v2x_j2735.py          physics-grounded J2735 BSM corpus (+leak channels)
data/inject_poison_shortcut.py      BadNets trigger + shortcut utilities
src/evaluator.py                    fixed, gradient-free evaluator + AST no-grad check
src/iwsp_agent.py                   the weight-space programming loop (offline + API)
src/synthesized_models/
    v2x_iwsp_compact.py             reference 23-param transparent weight network
    poisoned_shortcut_demo.py       shortcut & backdoor surfaced as readable weights
src/baselines.py                    MLP / 1D-CNN / ResNet-tabular (Adam/backprop)
src/attacks.py                      FGSM / PGD / transfer / physics-bounded FDI
src/audit.py                        static weight+AST audit · text-bottleneck rebuild
experiments/run_benchmark.py        end-to-end runner -> results/*.csv
experiments/plot_results.py         four 300-DPI figures
tests/test_pipeline.py              14 tests
```

## Limitations and honest next steps

- **Synthetic corpus.** Chosen for ground-truth control over the shortcut and
  backdoor channels. The natural external check is to re-fit the same weight
  program against a real misbehaviour trace (**VeReMi** / VeReMi-Extension) and
  report held-out detection without re-tuning the thresholds.
- **iWSP is not the accuracy leader.** It trades ~5 F1 points for compression,
  auditability, and robustness; the paper claim is *trustworthiness at comparable
  accuracy*, not state-of-the-art F1.
- **Separable features.** The kinematic residuals are cleanly separated by
  construction; stealthy attacks (30% of each class) overlap benign noise and are
  genuinely missed, but real-world class overlap is heavier.
- **Single surrogate** for the transfer attack; an adaptive, iWSP-aware attacker
  (search directly against the threshold structure) is a stronger stress test.
- **Offline agent** is a deterministic stand-in for the LLM revision loop; the API
  driver exercises a real agent but is not required to reproduce any number here.

## License

MIT — see [LICENSE](LICENSE).
