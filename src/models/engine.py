"""VXOneEngine — contrato de inferencia (spec F2 §8.1).

API tipada sobre un checkpoint self-contained (.pt, §8.2): la inferencia no
necesita datos de train (feature stage, vocabularios, ω, temperaturas y τ
viajan dentro del artefacto).

Invariantes congelados:
- candidate == "unknown" ⟺ (modo abstención y max_prob < τ_abs) o (modo
  trained_unknown y unknown ganó el argmax).
- probs (Choice) suma 1 sobre los candidatos NATIVOS de la corrida (en EMBER
  unknown ES candidato nativo; en UNSW/CIC nunca aparece como masa).
- policy se deriva de las salidas calibradas (recomputable, sin estado).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .feature_stage import FeatureStage
from .losses import corn_pmf, corn_expected
from .network import VXOneNet, HeadSpec
from .vocab import TextVocab

TAU_DET = 0.5   # g_attack (controlador; defaults serializados, §3.1 D9)
TAU_DROP = 0.5
TAU_RECON = 0.5
TAU_ABS = 0.5

SOURCE_KEY = {"unsw": "unsw_nb15_v1", "cic": "cicids2017_v1", "ember": "ember2024_v1"}
SOURCE_ALIAS = {v: k for k, v in SOURCE_KEY.items()}
LEVELS = ("benign", "low", "high", "critical")


@dataclass(frozen=True)
class ChoiceResult:
    candidate: str                 # ∈ candidatos ∪ {"unknown"}
    probs: dict[str, float]        # calibradas + de-tilt; suma 1 sobre nativos
    max_prob: float                # max_k p_cal (auditoría de abstención)


@dataclass(frozen=True)
class ScoreResult:
    expected: float                # ŝ = Σ k·π_k ∈ [0, 3]
    probs: dict[str, float]        # π calibrada sobre benign/low/high/critical


@dataclass(frozen=True)
class Decision:
    source: str
    noul: dict[str, float]         # {"is_attack"|"is_malicious": p_cal}
    choice: dict[str, ChoiceResult]
    score: ScoreResult | None      # None en EMBER (S6)
    policy: dict[str, bool]        # gates derivados


class VXOneEngine:
    def __init__(self, artifact: str | Path, device: str = "cpu") -> None:
        ckpt = torch.load(artifact, map_location=device, weights_only=False)
        self.ckpt = ckpt  # introspección (E4 latencia: config, d_num)
        self.device = device
        self.source: str = SOURCE_ALIAS[ckpt["source"]]   # alias corto: unsw|cic|ember
        self.source_key: str = ckpt["source"]             # clave del registro
        self.candidates: list[str] = ckpt["head_meta"]["choice"]["candidates"]
        self.mode: str = ckpt["head_meta"]["choice"]["mode"]  # abstain|trained_unknown
        self.tau_abs: float = ckpt["head_meta"]["choice"]["tau_abs"]
        self.stage = FeatureStage.from_dict(ckpt["feature_stage"])
        self.vocab = {k: TextVocab.from_dict(v) for k, v in ckpt.get("vocab", {}).items()}
        self.omega = torch.tensor(ckpt["head_meta"]["choice"]["class_weights"])
        self.T_noul = ckpt["head_meta"]["noul"]["temperature"]
        self.T_choice = ckpt["head_meta"]["choice"]["temperature"]
        self.T_score = ckpt["head_meta"].get("score", {}).get("temperature")
        self.has_ddos = "ddos" in ckpt["head_meta"]
        if self.has_ddos:
            self.ddos_candidates = ckpt["head_meta"]["ddos"]["candidates"]
            self.omega_ddos = torch.tensor(ckpt["head_meta"]["ddos"]["class_weights"])
            self.T_ddos = ckpt["head_meta"]["ddos"]["temperature"]
        else:
            self.ddos_candidates = self.omega_ddos = self.T_ddos = None

        d_num = ckpt["config"]["d_num"]
        heads = HeadSpec(n_candidates=len(self.candidates),
                         ddos_candidates=len(self.ddos_candidates) if self.has_ddos else 0,
                         has_score=self.T_score is not None)
        self.net = VXOneNet(self.source, ckpt["config"]["scale"], heads, d_num,
                            vocab_sizes={k: v.size for k, v in self.vocab.items()})
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.to(device).eval()

    # ── preparación de entrada ───────────────────────────────────────
    def _prepare(self, x_num: np.ndarray, x_txt: dict[str, list[str | None]] | None):
        x_num = np.asarray(x_num, dtype=np.float64)
        u_r, _ = self.stage.transform(x_num)
        slots = None
        if self.vocab:
            n = x_num.shape[0]
            slots = {f: torch.tensor(
                self.vocab[f].encode((x_txt or {}).get(f, [None] * n)), dtype=torch.int64)
                for f in self.vocab}
        return torch.tensor(u_r, device=self.device), slots

    # ── predicción ───────────────────────────────────────────────────
    @torch.no_grad()
    def predict_batch(self, x_num: np.ndarray,
                      x_txt: dict[str, list[str | None]] | None = None) -> list[Decision]:
        u_r, slots = self._prepare(x_num, x_txt)
        out = self.net(u_r, slots)

        # Noul calibrado
        p_mal = torch.sigmoid(out["noul"] / self.T_noul)
        noul_key = "is_attack" if self.source in ("unsw", "cic") else "is_malicious"

        # Choice: de-tilt → T → probs sobre nativos
        z = out["choice"] - torch.log(self.omega).unsqueeze(0).to(self.device)
        p_choice = torch.softmax(z / self.T_choice, dim=1)

        # Score CORN calibrado (None en EMBER)
        pi = corn_pmf(out["corn_logits"] / self.T_score)[1] if self.T_score is not None else None
        s_exp = corn_expected(pi) if pi is not None else None

        # ddos vector (solo CIC strat)
        ddos_probs = None
        if self.has_ddos:
            z_d = out["ddos"] - torch.log(self.omega_ddos).unsqueeze(0).to(self.device)
            ddos_probs = torch.softmax(z_d / self.T_ddos, dim=1)

        decisions: list[Decision] = []
        for i in range(u_r.shape[0]):
            # Choice con unknown (D1): abstención τ_abs o unknown entrenado
            pmax = float(p_choice[i].max())
            if self.mode == "abstain":
                cand = self.candidates[int(p_choice[i].argmax())] if pmax >= self.tau_abs else "unknown"
            else:  # trained_unknown (EMBER): unknown es candidato nativo
                cand = self.candidates[int(p_choice[i].argmax())]
            probs = {c: float(p_choice[i, j]) for j, c in enumerate(self.candidates)}

            choice = {"family": ChoiceResult(cand, probs, pmax)}
            if ddos_probs is not None:
                dp = {c: float(ddos_probs[i, j]) for j, c in enumerate(self.ddos_candidates)}
                choice["ddos_vector"] = ChoiceResult(
                    self.ddos_candidates[int(ddos_probs[i].argmax())], dp,
                    float(ddos_probs[i].max()))

            score = (ScoreResult(float(s_exp[i]),
                                 {lv: float(pi[i, k]) for k, lv in enumerate(LEVELS)})
                     if pi is not None else None)

            # Gates derivados (D9): funciones exactas de salidas calibradas
            policy = {"g_attack": bool(p_mal[i] >= TAU_DET)}
            if pi is not None:
                # g_drop: P(sev ≥ 2) = C2 = π2+π3 (acumulada CORN exacta)
                policy["g_drop"] = bool(float(pi[i, 2] + pi[i, 3]) >= TAU_DROP)
            if "recon" in self.candidates:
                policy["g_recon"] = bool(probs.get("recon", 0.0) >= TAU_RECON)

            decisions.append(Decision(
                source=self.source,
                noul={noul_key: float(p_mal[i])},
                choice=choice,
                score=score,
                policy=policy,
            ))
        return decisions

    def predict(self, x_num: np.ndarray, x_txt: dict[str, str | None] | None = None) -> Decision:
        """batch-1 sin maquinaria de batch (E4)."""
        xt = {k: [v] for k, v in (x_txt or {}).items()} if x_txt else None
        return self.predict_batch(np.asarray(x_num)[None, ...], xt)[0]


def save_artifact(path: str | Path, net: VXOneNet, run_meta: dict,
                  temperatures: dict[str, float], d_num: int) -> None:
    """Escribe el checkpoint self-contained (§8.2)."""
    ckpt = {
        "source": run_meta["source_key"],
        "feature_stage": run_meta["feature_stage"].to_dict(),
        "vocab": {k: v.to_dict() for k, v in (run_meta.get("vocab") or {}).items()},
        "head_meta": {
            "noul": {"temperature": temperatures["noul"]},
            "choice": {"candidates": run_meta["candidates"],
                       "class_weights": run_meta["omega"].numpy(),
                       "mode": run_meta["mode"],
                       "tau_abs": run_meta.get("tau_abs", TAU_ABS),
                       "temperature": temperatures["choice"]},
            **({"score": {"temperature": temperatures["score"]}}
               if "score" in temperatures else {}),
            **({"ddos": {"candidates": run_meta["ddos_candidates"],
                         "class_weights": run_meta["omega_ddos"].numpy(),
                         "temperature": temperatures["ddos"]}}
               if "ddos" in temperatures else {}),
        },
        "config": {**run_meta["config"], "d_num": d_num},
        "manifest": run_meta["manifest"],
    }
    ckpt["state_dict"] = net.state_dict()
    torch.save(ckpt, path)
