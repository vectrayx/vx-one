"""Vocabulario de x_txt (spec F2 §4.2): embedding lookup con slots reservados.

Slots: `<miss>`=0 (campo ausente/None), `<unk>`=1 (valor fuera del vocab de
train). Vocabulario = valores distinctos de train, normalizados
casefold+strip. Los vocabularios son cerrados y diminutos (≤135): lookup, no
hashing (F2 D7).
"""

from __future__ import annotations

MISS, UNK = 0, 1


def _norm(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip()
    return v.casefold() if v else None


class TextVocab:
    def __init__(self, field: str, token_to_slot: dict[str, int]) -> None:
        self.field = field
        self.token_to_slot = dict(token_to_slot)  # tokens normalizados → slot ≥ 2

    @classmethod
    def fit(cls, field: str, train_values: list[str | None]) -> "TextVocab":
        tokens = sorted({t for v in train_values if (t := _norm(v)) is not None})
        return cls(field, {t: i + 2 for i, t in enumerate(tokens)})

    @property
    def size(self) -> int:
        """Tamaño de tabla: V + 2 slots reservados."""
        return len(self.token_to_slot) + 2

    def encode(self, values: list[str | None]) -> list[int]:
        out: list[int] = []
        for v in values:
            t = _norm(v)
            out.append(MISS if t is None else self.token_to_slot.get(t, UNK))
        return out

    def to_dict(self) -> dict:
        return {"field": self.field, "token_to_slot": self.token_to_slot}

    @classmethod
    def from_dict(cls, d: dict) -> "TextVocab":
        return cls(d["field"], d["token_to_slot"])
