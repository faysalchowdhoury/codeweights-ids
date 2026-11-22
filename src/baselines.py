"""Standard gradient-trained IDS baselines (MLP, 1D-CNN, ResNet-tabular).

These are the foil for the iWSP compact model. They are trained normally with
Adam + cross-entropy backprop. On the *poisoned* split they minimise training
loss by reading the simulator leak channels and by honouring the BadNets trigger
-- both of which are invisible to visible-split accuracy and surface only in the
hidden-split collapse, the backdoor ASR, and the static audit.

Each wrapper standardises inputs with train-set statistics, exposes
``predict``/``attack_score`` (numpy) for the evaluator, and ``logits_torch`` (a
differentiable forward on standardised input) for the white-box attack suite.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from data.veremi_data import FEATURE_NAMES, N_CLASSES, N_FEATURES

torch.use_deterministic_algorithms(True, warn_only=True)


def _to_X(X):
    if hasattr(X, "reindex"):
        X = X.reindex(columns=FEATURE_NAMES).to_numpy()
    return np.asarray(X, dtype=np.float32)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEATURES, 32), nn.ReLU(),
                                 nn.Linear(32, 16), nn.ReLU(),
                                 nn.Linear(16, N_CLASSES))

    def forward(self, x):
        return self.net(x)


class _CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 8, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(8, 8, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(8 * N_FEATURES, N_CLASSES)

    def forward(self, x):
        z = self.conv(x.unsqueeze(1))
        return self.head(z.flatten(1))


class _ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, x):
        return torch.relu(x + self.lin(x))


class _ResNetTab(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(N_FEATURES, 32)
        self.blocks = nn.Sequential(_ResBlock(32), _ResBlock(32))
        self.head = nn.Linear(32, N_CLASSES)

    def forward(self, x):
        return self.head(self.blocks(torch.relu(self.inp(x))))


_ARCHES = {"mlp": _MLP, "cnn1d": _CNN1D, "resnet_tab": _ResNetTab}


class GradientBaseline:
    """Adam/backprop-trained tabular IDS with a numpy inference API."""

    def __init__(self, arch: str = "mlp", epochs: int = 60, lr: float = 3e-3,
                 seed: int = 0):
        assert arch in _ARCHES, f"unknown arch {arch}"
        self.arch, self.epochs, self.lr, self.seed = arch, epochs, lr, seed
        self.net = None
        self.mu = self.sd = None

    def train(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = _to_X(X)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu) / self.sd)
        yt = torch.tensor(np.asarray(y), dtype=torch.long)
        self.net = _ARCHES[self.arch]()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = nn.CrossEntropyLoss()
        n = len(Xt)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                loss = lossf(self.net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    # -- differentiable forward on standardised input (for attacks) ----------
    def logits_torch(self, x_std: torch.Tensor) -> torch.Tensor:
        return self.net(x_std)

    def standardise(self, X) -> torch.Tensor:
        return torch.tensor((_to_X(X) - self.mu) / self.sd)

    # -- numpy inference API -------------------------------------------------
    @torch.no_grad()
    def logits(self, X) -> np.ndarray:
        return self.net(self.standardise(X)).numpy()

    def predict(self, X) -> np.ndarray:
        return self.logits(X).argmax(1)

    def attack_score(self, X) -> np.ndarray:
        lg = torch.softmax(torch.tensor(self.logits(X)), dim=1).numpy()
        return 1.0 - lg[:, 0]                     # P(any attack)

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))


def train_all(ds: dict, epochs: int = 60, seed: int = 0) -> dict[str, GradientBaseline]:
    """Train every baseline on ds['train']."""
    X, y = ds["train"]
    out = {}
    for arch in _ARCHES:
        out[arch] = GradientBaseline(arch, epochs=epochs, seed=seed).train(X, y)
    return out


if __name__ == "__main__":
    from data.veremi_data import generate_dataset
    from src.evaluator import evaluate_model
    ds = generate_dataset()
    for arch, m in train_all(ds, epochs=40).items():
        r = evaluate_model(m, ds, name=arch)
        print(f"{arch:12s} params={r['n_params']:5d} "
              f"vis_f1={r['visible_macro_f1']:.3f} hid_f1={r['hidden_macro_f1']:.3f} "
              f"gap={r['shortcut_gap']:+.3f} FPR={r['false_positive_rate']:.3f}")
