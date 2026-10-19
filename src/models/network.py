"""VXOneNet (spec F2 §2): proyección + embeddings de texto + tronco + cabezas.

Estructura por bloque: Linear → LayerNorm → GELU → Dropout(0.1); bloque 1
mapea R^{d_in} → R^w, bloques 2..L son w→w con residual h_b = h_{b-1} +
bloque_b(h_{b-1}). LayerNorm (no BatchNorm): streaming por shards, batch-1 en
inferencia y drift temporal (F2 §2.1).

Cabezas por dataset (F2 §3.1):
- UNSW: noul(1) + choice_family(10) + corn(3 condicionales)
- CIC:  noul(1) + choice_family(8|6) + corn + choice_ddos(4)   [solo strat]
- EMBER: noul(1) + choice_family(11, unknown entrenado)

Dims congeladas (§2.2) y conteo de parámetros verificado (§2.3): los tests
comparan contra la tabla exacta de la spec.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# ── Dims congeladas por escala (F2 §2.2) ────────────────────────────────
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
    """Cabezas activas de una corrida (nombres congelados)."""
    n_candidates: int          # choice_family (K)
    ddos_candidates: int = 0   # 0 = head ddos ausente
    has_score: bool = True     # False en EMBER (S6)

    @property
    def names(self) -> tuple[str, ...]:
        ns = ["noul", "choice"]
        if self.has_score:
            ns.append("score")
        if self.ddos_candidates:
            ns.append("ddos")
        return tuple(ns)


def head_spec_for(source: str, scale: str, split_scheme: str,
                  n_candidates: int) -> HeadSpec:
    """Candidate set por corrida (D12): familias con ≥1 fila en train.

    La verificación de n_candidates ocurre en data.py; aquí solo se tipa.
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
    """Motor VX-One v0.1. d_num incluye el flag de fila (d+1 ya aplicado
    por el feature stage: la entrada de P0 es [u ; r] ∈ R^{d+1})."""

    def __init__(
        self,
        source: str,
        scale: str,
        heads: HeadSpec,
        d_num: int,               # d canónico (sin flag); P0 entra con d+1
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

        # P0: Linear((d+1) → d0) — el +1 es el flag de fila (§2.3)
        self.proj = nn.Linear(d_num + 1, d0)
        # Embeddings lookup: una tabla por campo, (V+2) × e
        self.emb = nn.ModuleDict(
            {f: nn.Embedding(vs, e) for f, vs in self.vocab_sizes.items()}
        )
        d_in = d0 + m * e
        # Tronco: bloque 1 d_in→w, bloques 2..L w→w con residual
        self.blocks = nn.ModuleList([_Block(d_in, w)] + [_Block(w, w) for _ in range(L - 1)])

        # Cabezas (§2.3): Noul w+1; Choice wK+K; CORN w+3; ddos 4w+4
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

    def forward(self, u_r: torch.Tensor, txt_slots: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """u_r float32 [B, d+1]; txt_slots {campo: int64 [B]} (solo UNSW)."""
        h = self.trunk(u_r, txt_slots)
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

    # ── conteo de parámetros (contrato §2.3; Apéndice B) ─────────────
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
