"""VXOneNet (spec F2 §2): projection + text embeddings + trunk + heads.

Block structure: Linear → LayerNorm → GELU → Dropout(0.1); block 1
maps R^{d_in} → R^w, blocks 2..L are w→w with residual h_b = h_{b-1} +
block_b(h_{b-1}). LayerNorm (not BatchNorm): streaming over shards, batch-1 in
inference and temporal drift (F2 §2.1).

Heads per dataset (F2 §3.1):
- UNSW: noul(1) + choice_family(10) + corn(3 conditionals)
- CIC: noul(1) + choice_family(8|6) + corn + choice_ddos(4) [strat only]
- EMBER: noul(1) + choice_family(11, unknown trained)

Frozen dims (§2.2) and verified parameter count (§2.3): the tests
They compare against the exact table of the spec.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# ── Frozen dims by scale (F2 §2.2) ────────────────────────────────
SCALES: dict[str, dict[str, int]] = {
    "nano": {"L": 2, "w_unsw": 224, "w_cic": 224, "w_ember": 192,
             "d0_unsw": 112, "d0_cic": 112, "d0_ember": 20, "e": 16},
    "small": {"L": 3, "w_unsw": 576, "w_cic": 576, "w_ember": 576,
              "d0_unsw": 288, "d0_cic": 288, "d0_ember": 96, "e": 32},
    "base": {"L": 4, "w_unsw": 1280, "w_cic": 1280, "w_ember": 1280,
             "d0_unsw": 512, "d0_cic": 512, "d0_ember": 384, "e": 64},
}

DROPOUT = 0.1


@dataclass(frozen=True)
class HeadSpec:
    """Active heads of a bullfight (frozen names)."""
    n_candidates: int          # choice_family (K)
    ddos_candidates: int = 0   # 0 = head ddos ausente
    has_score: bool = True     # False in EMBER (S6)
    flat: bool = False         # A7: single K-way family softmax (derived gate)
    normal_index: int = 0      # index of "normal" in the candidate set

    @property
    def names(self) -> tuple[str, ...]:
        if self.flat:
            return ("flat",)
        ns = ["noul", "choice"]
        if self.has_score:
            ns.append("score")
        if self.ddos_candidates:
            ns.append("ddos")
        return tuple(ns)


def head_spec_for(source: str, scale: str, split_scheme: str,
                  n_candidates: int) -> HeadSpec:
    """Candidate set per run (D12): families with ≥1 row in train.

Verification of n_candidates occurs in data.py; here it is only typed.
    """
    if source == "unsw":
        return HeadSpec(n_candidates=n_candidates)
    if source == "cic":
        return HeadSpec(n_candidates=n_candidates,
                        ddos_candidates=4 if split_scheme == "strat" else 0)
    if source == "ember":
        return HeadSpec(n_candidates=n_candidates, has_score=False)
    raise ValueError(f"source desconocido: {source!r}")


class _Block(nn.Module):
    def __init__(self, d_in: int, w: int) -> None:
        super().__init__()
        self.lin = nn.Linear(d_in, w)
        self.ln = nn.LayerNorm(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.dropout(
            nn.functional.gelu(self.ln(self.lin(x))), DROPOUT,
            training=self.training,
        )


class VXOneNet(nn.Module):
    """VX-One Engine v0.1. d_num includes the row flag (d+1 already applied
by the feature stage: the input of P0 is [u ; r] ∈ R^{d+1})."""

    def __init__(
        self,
        source: str,
        scale: str,
        heads: HeadSpec,
        d_num: int,               # d canonical (no flag); P0 enters with d+1
        vocab_sizes: dict[str, int] | None = None,  # {campo: V+2} (solo UNSW)
    ) -> None:
        super().__init__()
        cfg = SCALES[scale]
        self.source = source
        self.scale = scale
        self.heads = heads
        if source == "ember":
            w, d0, L = cfg["w_ember"], cfg["d0_ember"], cfg["L"]
        else:
            w = cfg["w_unsw"] if source == "unsw" else cfg["w_cic"]
            d0 = cfg["d0_unsw"] if source == "unsw" else cfg["d0_cic"]
            L = cfg["L"]
        e = cfg["e"]
        self.vocab_sizes = vocab_sizes or {}
        m = len(self.vocab_sizes)

        # P0: Linear((d+1) → d0) — the +1 is the row flag (§2.3)
        self.proj = nn.Linear(d_num + 1, d0)
        # Embeddings lookup: one table per field, (V+2) × e
        self.emb = nn.ModuleDict(
            {f: nn.Embedding(vs, e) for f, vs in self.vocab_sizes.items()}
        )
        d_in = d0 + m * e
        # Trunk: block 1 d_in→w, blocks 2..L w→w with residual
        self.blocks = nn.ModuleList([_Block(d_in, w)] + [_Block(w, w) for _ in range(L - 1)])

        # Heads (§2.3): Noul w+1; Choice wK+K; CORN w+3; ddos 4w+4.
        # A7 replaces the gate+choice pair with a single K-way family softmax;
        # the gate is derived as 1 - p(normal).
        self.flat = nn.Linear(w, heads.n_candidates) if heads.flat else None
        if heads.flat:
            # analytic de-tilt of the family tilt (applied at eval only)
            self.register_buffer("flat_logw", torch.zeros(heads.n_candidates))
        if not heads.flat:
            self.noul = nn.Linear(w, 1)
            self.choice = nn.Linear(w, heads.n_candidates)
        self.has_score = heads.has_score
        if heads.has_score:
            self.corn_u = nn.Linear(w, 1, bias=False)   # u^T h
            self.corn_b = nn.Parameter(torch.tensor([1.0, 0.0, -1.0]))  # init §2.1
        self.has_ddos = heads.ddos_candidates > 0
        if self.has_ddos:
            self.ddos = nn.Linear(w, heads.ddos_candidates)

    # ── forward ──────────────────────────────────────────────────────
    def trunk(self, u_r: torch.Tensor, txt_slots: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        h = self.proj(u_r)
        if self.emb:
            parts = [h]
            for f, tab in self.emb.items():
                parts.append(tab(txt_slots[f]))
            h = torch.cat(parts, dim=1)
        for i, block in enumerate(self.blocks):
            h = h + block(h) if i > 0 else block(h)
        return h

    def _flat_forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """A7: single family softmax; the gate is derived as 1 - p(normal).

        Eval applies the analytic de-tilt of the family tilt (omega), exactly
        as the reference choice head does in ``vx_calibrated``. The derived
        gate and the choice share the same distribution: this is the
        "dedicated primitive vs marginalised primitive" contrast.
        """
        z = self.flat(h)
        if not self.training:
            z = z - self.flat_logw
        p_fam = torch.softmax(z, dim=1)
        n = self.heads.normal_index
        p_att = (1.0 - p_fam[:, n]).clamp(1e-6, 1.0 - 1e-6)
        out = {"flat": z, "choice": z,
               "noul": torch.log(p_att) - torch.log1p(-p_att)}
        if self.has_score:
            out["corn_logits"] = (self.corn_u(h).squeeze(-1).unsqueeze(1)
                                  + self.corn_b.unsqueeze(0))
        if self.has_ddos:
            out["ddos"] = self.ddos(h)
        return out

    def forward(self, u_r: torch.Tensor, txt_slots: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """u_r float32 [B, d+1]; txt_slots {campo: int64 [B]} (solo UNSW)."""
        h = self.trunk(u_r, txt_slots)
        if self.heads.flat:
            return self._flat_forward(h)
        out: dict[str, torch.Tensor] = {
            "noul": self.noul(h).squeeze(-1),          # logits [B]
            "choice": self.choice(h),                  # logits [B, K]
        }
        if self.has_score:
            out["corn_logits"] = (self.corn_u(h).squeeze(-1).unsqueeze(1)
                                  + self.corn_b.unsqueeze(0))  # [B, 3]
        if self.has_ddos:
            out["ddos"] = self.ddos(h)                 # logits [B, 4]
        return out

    # ── parameter count (contract §2.3; Appendix B) ─────────────
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
