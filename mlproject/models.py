"""Modèles personnalisés compatibles scikit-learn Pipeline.

Contient TorchMLPClassifier : un MLP binaire PyTorch encapsulé dans
l'interface sklearn (fit / predict / predict_proba), utilisable directement
dans un Pipeline et dans GridSearchCV.

Pourquoi PyTorch ici et pas pour les autres modèles ?
  - LR, RF, XGBoost, LightGBM sont les meilleurs outils pour la tabular data.
    Les remplacer par PyTorch serait moins performant sur 1 470 lignes.
  - Un MLP PyTorch apporte un profil différent (apprentissage par gradient,
    dropout, pos_weight) et démontre l'intégration PyTorch / sklearn.

Architecture :
  Input (51 features) -> [Linear -> ReLU -> Dropout] x N -> Linear(1)
  Loss   : BCEWithLogitsLoss avec pos_weight (gère le déséquilibre 84/16)
  Optim  : Adam
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from mlproject.config import RANDOM_STATE


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """MLP binaire PyTorch, compatible scikit-learn Pipeline + GridSearchCV.

    Paramètres
    ----------
    hidden_sizes : tuple[int, ...]
        Nombre de neurones par couche cachée.
        Ex. (64, 32) -> deux couches : 64 puis 32 neurones.
    lr : float
        Learning rate pour Adam.
    epochs : int
        Nombre de passes complètes sur les données d'entraînement.
    batch_size : int
        Taille des mini-batches.
    dropout : float
        Taux de dropout appliqué après chaque couche cachée (régularisation).
    random_state : int
        Graine pour la reproductibilité.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (64, 32),
        lr: float = 1e-3,
        epochs: int = 150,
        batch_size: int = 64,
        dropout: float = 0.3,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.lr           = lr
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.dropout      = dropout
        self.random_state = random_state

    # ── Architecture ────────────────────────────────────────────────────────

    def _build_net(self, n_features: int) -> nn.Module:
        """Construit le réseau MLP selon hidden_sizes."""
        layers: list[nn.Module] = []
        in_size = n_features
        for h in self.hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU(), nn.Dropout(self.dropout)]
            in_size = h
        layers.append(nn.Linear(in_size, 1))  # logit de sortie (pas de sigmoid ici)
        return nn.Sequential(*layers)

    # ── Conversion d'entrée ─────────────────────────────────────────────────

    @staticmethod
    def _to_numpy(X) -> np.ndarray:
        """Convertit X en numpy float32 quelle que soit la source."""
        if hasattr(X, "toarray"):       # scipy sparse (OneHotEncoder)
            return X.toarray().astype(np.float32)
        if hasattr(X, "values"):        # pandas DataFrame
            return X.values.astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    # ── Interface sklearn ────────────────────────────────────────────────────

    def fit(self, X, y) -> "TorchMLPClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_np = self._to_numpy(X)
        y_np = np.asarray(y, dtype=np.float32)

        X_t = torch.tensor(X_np)
        y_t = torch.tensor(y_np).unsqueeze(1)  # (N, 1)

        # pos_weight compense le déséquilibre des classes :
        # loss(positif) est multiplié par (nb négatifs / nb positifs)
        n_pos = y_t.sum().item()
        n_neg = len(y_t) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)])

        self.net_ = self._build_net(X_t.shape[1])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.net_.train()
        for _ in range(self.epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                criterion(self.net_(X_batch), y_batch).backward()
                optimizer.step()

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Retourne les probabilités [P(0), P(1)] pour chaque exemple."""
        check_is_fitted(self, "net_")
        self.net_.eval()
        X_t = torch.tensor(self._to_numpy(X))
        with torch.no_grad():
            proba_pos = torch.sigmoid(self.net_(X_t).squeeze(1)).numpy()
        return np.column_stack([1 - proba_pos, proba_pos])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
