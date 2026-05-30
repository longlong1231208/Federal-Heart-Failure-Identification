# datasets/fl_dataset.py
from __future__ import annotations

import pickle
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ArrayLike = Union[np.ndarray, torch.Tensor]


def _stable_int_hash(text: str, modulo: int = 10_000) -> int:
    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(modulo)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if hasattr(x, "to_numpy"):  # pandas
        return x.to_numpy()
    return np.asarray(x)


def _ensure_3d(X: Any) -> np.ndarray:
    X = _to_numpy(X)
    if X.ndim != 3:
        raise ValueError(f"Expected X to be 3D [N,T,D], got shape={X.shape}")
    return X


def _ensure_1d(y: Any) -> np.ndarray:
    y = _to_numpy(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1:
        raise ValueError(f"Expected y to be 1D [N], got shape={y.shape}")
    return y


def _split_train_val_from_train(
    y_train: np.ndarray,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split ONLY within the existing train split (no touching test).
    Stratify when possible.
    """
    n = len(y_train)
    idx_all = np.arange(n)

    if val_frac <= 0 or n < 2:
        return idx_all, np.array([], dtype=int)

    n_val = int(round(n * val_frac))
    n_val = max(1, n_val)
    n_val = min(n_val, n - 1)

    rng = np.random.default_rng(seed)

    classes, counts = np.unique(y_train, return_counts=True)
    can_stratify = (len(classes) >= 2) and (np.min(counts) >= 2)

    if not can_stratify:
        perm = rng.permutation(idx_all)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        return np.sort(tr_idx), np.sort(val_idx)

    # Stratified: proportional per class, then adjust rounding drift
    val_idx_list: List[int] = []
    remain = n_val
    for k, c in zip(classes, counts):
        take = int(round(n_val * (c / n)))
        take = min(take, c - 1)   # keep at least 1 sample per class in train
        take = max(0, take)
        remain -= take

        k_idx = idx_all[y_train == k]
        chosen = rng.choice(k_idx, size=take, replace=False)
        val_idx_list.extend(chosen.tolist())

    val_idx = np.array(val_idx_list, dtype=int)

    if remain > 0:
        pool = np.setdiff1d(idx_all, val_idx, assume_unique=False)
        extra = rng.choice(pool, size=min(remain, len(pool)), replace=False)
        val_idx = np.concatenate([val_idx, extra])
    elif remain < 0:
        drop_n = -remain
        keep = rng.choice(np.arange(len(val_idx)), size=len(val_idx) - drop_n, replace=False)
        val_idx = val_idx[keep]

    val_idx = np.unique(val_idx)
    tr_idx = np.setdiff1d(idx_all, val_idx, assume_unique=False)
    return np.sort(tr_idx), np.sort(val_idx)


class SequenceDataset(Dataset):
    """
    Holds (X,y) where X is [N,T,D] float32 and y is int64.
    """
    def __init__(self, X: ArrayLike, y: ArrayLike):
        Xn = _ensure_3d(X).astype(np.float32, copy=False)
        yn = _ensure_1d(y).astype(np.int64, copy=False)

        if Xn.shape[0] != yn.shape[0]:
            raise ValueError(f"X/y length mismatch: {Xn.shape[0]} vs {yn.shape[0]}")

        self.X = torch.from_numpy(Xn)
        self.y = torch.from_numpy(yn)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


@dataclass
class ClientSplit:
    train: SequenceDataset
    val: Optional[SequenceDataset]
    test: SequenceDataset
    meta: Dict[str, Any]


class FederatedPKLDataset:
    """
    Exactly matches preprocess.py output format:

    client_data[client_id] = {
      "train": {"X": X_tr_scaled, "y": y_tr},
      "test":  {"X": X_te_scaled, "y": y_te},
      "meta":  {...}
    }

    It NEVER re-splits train/test. Optionally creates val from train only.
    """
    def __init__(
        self,
        pkl_path: Union[str, Path],
        val_frac: float = 0.0,  # default: do NOT create val unless you need it
        seed: int = 42,
        allow_nan: bool = False,
    ):
        self.pkl_path = Path(pkl_path)
        if not self.pkl_path.exists():
            raise FileNotFoundError(f"PKL not found: {self.pkl_path}")

        self.val_frac = float(val_frac)
        self.seed = int(seed)
        self.allow_nan = bool(allow_nan)

        self._raw: Dict[Any, Dict[str, Any]] = self._load()
        self._clients: List[Any] = sorted(self._raw.keys(), key=lambda x: str(x))
        if len(self._clients) == 0:
            raise ValueError("No clients found in dataset.")

        # infer T, D from first client's train
        X0 = _ensure_3d(self._raw[self._clients[0]]["train"]["X"])
        self._T = int(X0.shape[1])
        self._D = int(X0.shape[2])

        self._cache: Dict[Any, ClientSplit] = {}
        self._sanity_check()

    def _load(self) -> Dict[Any, Dict[str, Any]]:
        with open(self.pkl_path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected dict in pkl, got {type(obj)}")

        # schema check
        for c, d in obj.items():
            if not isinstance(d, dict) or "train" not in d or "test" not in d:
                raise ValueError(f"Client {c}: invalid schema (need train/test).")
            for split in ("train", "test"):
                if "X" not in d[split] or "y" not in d[split]:
                    raise ValueError(f"Client {c} {split}: missing X/y.")
        return obj

    def _sanity_check(self) -> None:
        for c in self._clients:
            for split in ("train", "test"):
                X = _ensure_3d(self._raw[c][split]["X"])
                y = _ensure_1d(self._raw[c][split]["y"])
                if X.shape[0] != y.shape[0]:
                    raise ValueError(f"Client {c} {split}: X/y mismatch {X.shape[0]} vs {y.shape[0]}")
                if X.shape[1] != self._T or X.shape[2] != self._D:
                    raise ValueError(f"Client {c} {split}: shape mismatch, got {X.shape}, expected [N,{self._T},{self._D}]")
                if (not self.allow_nan) and np.isnan(X).any():
                    raise ValueError(f"Client {c} {split}: NaN exists in X (preprocess should have imputed).")
                uniq = np.unique(y)
                if not set(uniq.tolist()).issubset({0, 1}):
                    raise ValueError(f"Client {c} {split}: y must be binary 0/1, unique={uniq}")

    @property
    def seq_len(self) -> int:
        return self._T

    @property
    def input_dim(self) -> int:
        return self._D

    def get_clients(self) -> List[Any]:
        return list(self._clients)

    def describe_clients(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for c in self._clients:
            y_tr = _ensure_1d(self._raw[c]["train"]["y"])
            y_te = _ensure_1d(self._raw[c]["test"]["y"])
            meta = self._raw[c].get("meta", {})
            row = {
                "client": str(c),
                "N_train": int(len(y_tr)),
                "pos_rate_train": float(y_tr.mean()) if len(y_tr) else 0.0,
                "N_test": int(len(y_te)),
                "pos_rate_test": float(y_te.mean()) if len(y_te) else 0.0,
            }
            if isinstance(meta, dict):
                row.update(meta)
            rows.append(row)
        return rows

    def get_split(self, client: Any) -> ClientSplit:
        if client not in self._raw:
            raise KeyError(f"Unknown client: {client}")
        if client in self._cache:
            return self._cache[client]

        X_tr = _ensure_3d(self._raw[client]["train"]["X"])
        y_tr = _ensure_1d(self._raw[client]["train"]["y"])
        X_te = _ensure_3d(self._raw[client]["test"]["X"])
        y_te = _ensure_1d(self._raw[client]["test"]["y"])

        # optional val created ONLY from existing train split
        val_ds: Optional[SequenceDataset] = None
        if self.val_frac > 0:
            # Client-specific seed so different clients do not share the same
            # validation positions. Do not use Python's built-in hash(), which
            # is salted per process and breaks reproducibility across runs.
            c_seed = int(self.seed) + _stable_int_hash(str(client), modulo=10_000)
            tr_idx, val_idx = _split_train_val_from_train(y_tr, self.val_frac, c_seed)
            train_ds = SequenceDataset(X_tr[tr_idx], y_tr[tr_idx])
            if len(val_idx) > 0:
                val_ds = SequenceDataset(X_tr[val_idx], y_tr[val_idx])
        else:
            train_ds = SequenceDataset(X_tr, y_tr)

        split = ClientSplit(
            train=train_ds,
            val=val_ds,
            test=SequenceDataset(X_te, y_te),
            meta=self._raw[client].get("meta", {}) if isinstance(self._raw[client].get("meta", {}), dict) else {},
        )
        self._cache[client] = split
        return split

    def get_dataloaders(
        self,
        client: Any,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
        sp = self.get_split(client)

        train_loader = DataLoader(
            sp.train, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last
        )
        val_loader = None
        if sp.val is not None and len(sp.val) > 0:
            val_loader = DataLoader(
                sp.val, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory, drop_last=False
            )
        test_loader = DataLoader(
            sp.test, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False
        )
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # smoke test
    default_pkl = Path(__file__).resolve().parents[1] / "data" / "fl_dataset_final.pkl"
    if default_pkl.exists():
        ds = FederatedPKLDataset(default_pkl, val_frac=0.1, seed=42)
        print(f"[OK] clients={len(ds.get_clients())}, T={ds.seq_len}, D={ds.input_dim}")
        print(ds.describe_clients()[:3])
        c0 = ds.get_clients()[0]
        tr, va, te = ds.get_dataloaders(c0, batch_size=32)
        xb, yb = next(iter(tr))
        print(f"[OK] batch X={xb.shape} y={yb.shape} client={c0} val={'yes' if va else 'no'}")
    else:
        print(f"[WARN] Not found: {default_pkl}")
