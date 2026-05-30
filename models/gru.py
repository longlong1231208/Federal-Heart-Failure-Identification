# models/gru.py
from __future__ import annotations

from typing import Dict, List, Optional, Iterable

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    Paper-aligned 2-layer GRU model for binary HF prediction.

    Input:
        x: [B, T, D] or [T, D]

    Output:
        logits: [B]

    Design notes:
      1) Parameter groups are exposed in a paper-friendly 7-group form:
           g1_head
           g2_l0_ih
           g3_l0_hh
           g4_l0_b
           g5_l1_ih
           g6_l1_hh
           g7_l1_b

      2) forward_logits(x) and forward(x) return raw logits.

      3) This implementation assumes the last timestep is valid.
         If your sequences are padded, you should switch to packed sequences
         or pass explicit lengths and change the last-state extraction.
    """

    def __init__(
        self,
        input_dim: Optional[int],
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        if input_dim is None:
            raise ValueError("GRUModel: input_dim cannot be None.")
        if int(num_layers) != 2:
            raise ValueError(
                "GRUModel: this paper-aligned implementation expects num_layers=2."
            )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        # ---------------------------------------------------------
        # Backbone
        # ---------------------------------------------------------
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        # ---------------------------------------------------------
        # Binary head
        # ---------------------------------------------------------
        self.head_dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.hidden_dim, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if x.dim() != 3:
            raise ValueError(
                f"GRUModel: expected [B,T,D] or [T,D], got {tuple(x.shape)}"
            )

        if x.size(1) <= 0:
            raise ValueError("GRUModel: sequence length T must be > 0")

        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"GRUModel: expected input_dim={self.input_dim}, got D={x.size(-1)}"
            )

        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns final hidden representation [B, H].
        """
        x = self._normalize_input(x)
        out, _ = self.gru(x)  # [B, T, H]
        last_hidden = out[:, -1, :]  # assumes no padding
        return last_hidden

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns raw logits [B].
        Use this during:
          - Stage I training
          - Stage II-A bias calibration prep
          - Stage II-B sensitivity estimation
          - masked personalization training
        """
        feat = self.forward_features(x)
        feat = self.head_dropout(feat)
        logits = self.fc(feat).squeeze(-1)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns raw logits [B].
        """
        return self.forward_logits(x)

    # ------------------------------------------------------------------
    # Paper-aligned 7-group interface for Module 3
    # ------------------------------------------------------------------
    def named_param_groups(self) -> Dict[str, List[str]]:
        """
        Return the 7 paper-aligned parameter groups.

        Groups:
          g1_head  : fc.weight, fc.bias
          g2_l0_ih : gru.weight_ih_l0
          g3_l0_hh : gru.weight_hh_l0
          g4_l0_b  : gru.bias_ih_l0, gru.bias_hh_l0
          g5_l1_ih : gru.weight_ih_l1
          g6_l1_hh : gru.weight_hh_l1
          g7_l1_b  : gru.bias_ih_l1, gru.bias_hh_l1

        """
        groups: Dict[str, List[str]] = {
            "g1_head": ["fc.weight", "fc.bias"],
            "g2_l0_ih": ["gru.weight_ih_l0"],
            "g3_l0_hh": ["gru.weight_hh_l0"],
            "g4_l0_b": ["gru.bias_ih_l0", "gru.bias_hh_l0"],
            "g5_l1_ih": ["gru.weight_ih_l1"],
            "g6_l1_hh": ["gru.weight_hh_l1"],
            "g7_l1_b": ["gru.bias_ih_l1", "gru.bias_hh_l1"],
        }

        # safety filter
        existing = {n for n, _ in self.named_parameters()}
        filtered: Dict[str, List[str]] = {}
        for g, names in groups.items():
            keep = [n for n in names if n in existing]
            if keep:
                filtered[g] = keep
        return filtered

    def get_group_order(self) -> List[str]:
        """
        Stable paper order for logging / mask vector creation.
        """
        return [
            "g1_head",
            "g2_l0_ih",
            "g3_l0_hh",
            "g4_l0_b",
            "g5_l1_ih",
            "g6_l1_hh",
            "g7_l1_b",
        ]

    def get_group_index_map(self) -> Dict[str, int]:
        """
        Map group name -> 1-based paper index.
        """
        return {g: i + 1 for i, g in enumerate(self.get_group_order())}

    # ------------------------------------------------------------------
    # Helpers for masked personalization
    # ------------------------------------------------------------------
    def group_num_params(self) -> Dict[str, int]:
        """
        Number of scalar parameters in each group.
        Useful for dimension-normalized sensitivity.
        """
        named = dict(self.named_parameters())
        out: Dict[str, int] = {}
        for g, names in self.named_param_groups().items():
            out[g] = sum(named[n].numel() for n in names)
        return out

    def group_parameters(self) -> Dict[str, List[nn.Parameter]]:
        """
        Return actual parameter tensors per group.
        """
        named = dict(self.named_parameters())
        out: Dict[str, List[nn.Parameter]] = {}
        for g, names in self.named_param_groups().items():
            out[g] = [named[n] for n in names]
        return out

    def get_param_name_to_group(self) -> Dict[str, str]:
        """
        Reverse mapping: parameter name -> group name.
        """
        out: Dict[str, str] = {}
        for g, names in self.named_param_groups().items():
            for n in names:
                out[n] = g
        return out

    def get_group_mask(self, active_groups: Iterable[str]) -> Dict[str, int]:
        """
        Return dict mask {group_name: 0/1}.
        """
        active = set(active_groups)
        return {g: int(g in active) for g in self.get_group_order()}

    def get_mask_vector(self, active_groups: Iterable[str]) -> List[int]:
        """
        Return mask vector in stable paper order.
        """
        mask = self.get_group_mask(active_groups)
        return [mask[g] for g in self.get_group_order()]

    def set_trainable_groups(self, active_groups: Iterable[str]) -> None:
        """
        Freeze all parameters except those in active_groups.

        """
        active = set(active_groups)
        groups = self.named_param_groups()

        valid_groups = set(groups.keys())
        invalid = active - valid_groups
        if invalid:
            raise ValueError(
                f"Unknown groups in set_trainable_groups: {sorted(invalid)}"
            )

        active_names = set()
        for g in active:
            active_names.update(groups[g])

        for name, param in self.named_parameters():
            param.requires_grad_(name in active_names)

    def set_all_backbone_trainable(self, trainable: bool = True) -> None:
        """
        Convenience helper for enabling/disabling all params.
        """
        for _, p in self.named_parameters():
            p.requires_grad_(bool(trainable))

    def get_trainable_groups(self) -> List[str]:
        """
        Infer currently trainable groups from requires_grad.
        A group is considered active if any parameter inside it is trainable.
        """
        named = dict(self.named_parameters())
        active = []
        for g, names in self.named_param_groups().items():
            if any(named[n].requires_grad for n in names):
                active.append(g)
        return active

    # ------------------------------------------------------------------
    # Stage II-A bias-shift helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_head_bias(self) -> torch.Tensor:
        """
        Return a detached copy of fc.bias.
        """
        return self.fc.bias.detach().clone()

    @torch.no_grad()
    def set_head_bias(self, new_bias: torch.Tensor) -> None:
        """
        Overwrite fc.bias with new_bias.
        Expected shape: same as fc.bias, i.e. [1]
        """
        if new_bias.shape != self.fc.bias.shape:
            raise ValueError(
                f"set_head_bias: expected shape {tuple(self.fc.bias.shape)}, "
                f"got {tuple(new_bias.shape)}"
            )
        self.fc.bias.copy_(
            new_bias.to(device=self.fc.bias.device, dtype=self.fc.bias.dtype)
        )

    @torch.no_grad()
    def apply_head_bias_shift(self, delta_bias: float) -> None:
        """
        Add scalar delta_bias to fc.bias in-place.
        Used to absorb effective bias correction:
            b_fc <- b_fc + delta_bias_eff
        """
        delta = torch.tensor(
            [float(delta_bias)],
            device=self.fc.bias.device,
            dtype=self.fc.bias.dtype,
        )
        self.fc.bias.add_(delta)

    @torch.no_grad()
    def reset_head_bias(self, value: float = 0.0) -> None:
        """
        Optional helper for controlled experiments.
        """
        self.fc.bias.fill_(float(value))

    # ------------------------------------------------------------------
    # Convenience inference helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return sigmoid probabilities [B].
        """
        return torch.sigmoid(self.forward(x))

    @torch.no_grad()
    def predict_label(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Return binary predictions [B].
        """
        prob = self.predict_proba(x)
        return (prob >= float(threshold)).long()
