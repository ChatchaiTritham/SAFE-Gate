"""
Additional ordinal aggregation baselines on the risk chain (round-4 review).

All rules read gate outputs as urgency levels u(R1)=5 ... u(R5)=1; an R*
output is read as R1 (u=5), as for the other baselines. None of them has an
abstention element.

  WeightedMaxNormalised   Dubois-Prade weighted max with a NORMALISED
                          possibility distribution pi_i = q(c_i / max_j c_j);
                          capacity mu(A) = max_{i in A} pi_i (mu(N) = TOP
                          follows from normalisation, it is not forced).
  KthOrderStatistic       ordinal OWA with a 0/1 weight vector: the k-th most
                          urgent gate output (k = 1 is MIN selection); k is
                          chosen on the training split.
  QuantifierOWA           Yager (1996) importance-weighted OWA guided by the
                          RIM quantifier "most" Q(r) = clip((r - 0.3)/0.5),
                          importances = confidences; result rounded.
  BandClampedSugeno       median(floor, S, floor relaxed by one tier) with S
                          the quantised-additive Sugeno integral.
  SymmetricSugeno         Sugeno integral w.r.t. a symmetric capacity
                          mu(A) = phi(|A|); phi is chosen on the training
                          split by exhaustive search over all 252 monotone
                          phi: {1..5} -> {0..5}.
  PrioritizedYager        Yager (2008) prioritized averaging with the gate
                          priority G1 > G6 > G4 > G2 > G5 > G3 (authors'
                          ordering: hard red flags first).
  TSKFuzzy                zero-order Takagi-Sugeno-Kang rule base on the six
                          gate urgency levels: two linear membership functions
                          per input (low, high), 64 product rules, consequents
                          fitted by ridge least squares on the training split.

Leximin/discrimin refinements of min (Dubois, Fargier & Prade 1996) order
profiles that min ties; the aggregated value of each profile is still the
minimum, so as a tier-assignment rule they coincide with MIN selection.
"""
from __future__ import annotations

from itertools import combinations_with_replacement, product
from typing import Dict, List

import numpy as np

from .sugeno import TIER_OF, TOP, SugenoAdditive, quantise, sugeno_integral

URGENCY = {"R1": 5, "R2": 4, "R3": 3, "R4": 2, "R5": 1}
TIERS = ["R1", "R2", "R3", "R4", "R5"]


def levels(gate_outputs: Dict[str, str]) -> Dict[str, int]:
    return {g: URGENCY.get(t, 5) for g, t in gate_outputs.items()}


def _tier(u: float) -> str:
    return TIER_OF[int(max(1, min(5, u)))]


class WeightedMaxNormalised:
    name = "Weighted max (normalised possibility)"

    def classify(self, gate_outputs, gate_confidences):
        lv = levels(gate_outputs)
        top = max(gate_confidences[g] for g in lv)
        pi = {g: (quantise(gate_confidences[g] / top) if top > 0 else TOP) for g in lv}
        return {"final_tier": _tier(max(min(lv[g], pi[g]) for g in lv) or 1)}


class KthOrderStatistic:
    name = "Ordinal OWA (k-th order statistic)"

    def __init__(self, k: int = 1):
        self.k = k

    def classify(self, gate_outputs, gate_confidences=None):
        u = sorted(levels(gate_outputs).values(), reverse=True)
        return {"final_tier": _tier(u[min(self.k, len(u)) - 1])}

    @classmethod
    def fit(cls, gate_io, labels):
        best = None
        for k in range(1, 7):
            m = cls(k)
            acc = sum(m.classify(o)["final_tier"] == y for (o, _), y in zip(gate_io, labels))
            if best is None or acc > best[0]:
                best = (acc, k)
        return cls(best[1])


def q_most(r: float, a: float = 0.3, b: float = 0.8) -> float:
    return float(min(1.0, max(0.0, (r - a) / (b - a))))


class QuantifierOWA:
    name = "Quantifier-guided OWA (Yager 1996, 'most')"

    def classify(self, gate_outputs, gate_confidences):
        lv = levels(gate_outputs)
        order = sorted(lv, key=lambda g: -lv[g])
        imp = [gate_confidences[g] for g in order]
        total = sum(imp)
        if total <= 0:
            imp, total = [1.0] * len(order), float(len(order))
        acc, s = 0.0, 0.0
        for g, v in zip(order, imp):
            w = q_most((s + v) / total) - q_most(s / total)
            acc += w * lv[g]
            s += v
        return {"final_tier": _tier(int(np.floor(acc + 0.5)))}


class BandClampedSugeno:
    name = "Band-clamped Sugeno integral"

    def __init__(self):
        self._s = SugenoAdditive()

    def classify(self, gate_outputs, gate_confidences):
        floor = max(levels(gate_outputs).values())
        s = self._s.classify(gate_outputs, gate_confidences)["urgency"]
        return {"final_tier": _tier(sorted([floor - 1, s, floor])[1])}


class SymmetricSugeno:
    name = "Sugeno integral (trained symmetric capacity)"

    def __init__(self, phi=(1, 2, 3, 4, 5)):
        self.phi = tuple(phi)

    def _value(self, lv):
        n = len(lv)
        phi = (0,) + self.phi + (TOP,) * max(0, n - 5)

        def cap(a):
            return phi[len(a)] if len(a) < n else TOP
        return sugeno_integral(lv, cap)

    def classify(self, gate_outputs, gate_confidences=None):
        return {"final_tier": _tier(self._value(levels(gate_outputs)) or 1)}

    @classmethod
    def fit(cls, gate_io, labels, critical=("R1", "R2")):
        """Exhaustive search; prefer phi with 100% training critical sensitivity
        and 0 false discharges, then maximal training accuracy."""
        best = None
        for phi in combinations_with_replacement(range(TOP + 1), 5):
            m = cls(phi)
            preds = [m.classify(o)["final_tier"] for o, _ in gate_io]
            acc = sum(p == y for p, y in zip(preds, labels))
            safe = all(p in critical for p, y in zip(preds, labels) if y in critical)
            key = (safe, acc)
            if best is None or key > best[0]:
                best = (key, phi)
        return cls(best[1])


class PrioritizedYager:
    name = "Prioritized aggregation (Yager 2008)"
    PRIORITY = ["G1", "G6", "G4", "G2", "G5", "G3"]

    def classify(self, gate_outputs, gate_confidences=None):
        lv = levels(gate_outputs)
        order = [g for g in self.PRIORITY if g in lv] + [g for g in lv if g not in self.PRIORITY]
        t, num, den = 1.0, 0.0, 0.0
        for g in order:
            s = lv[g] / TOP
            num += t * s
            den += t
            t *= s
        return {"final_tier": _tier(int(np.floor(TOP * num / den + 0.5)))}


class TSKFuzzy:
    name = "TSK fuzzy rule base (zero order)"
    GATES = ["G1", "G2", "G3", "G4", "G5", "G6"]

    def __init__(self, consequents=None):
        self.w = consequents

    def _firing(self, gate_outputs):
        lv = levels(gate_outputs)
        x = [(lv.get(g, 1) - 1) / 4.0 for g in self.GATES]
        return np.array([np.prod([xi if b else 1 - xi for xi, b in zip(x, bits)])
                         for bits in product((0, 1), repeat=6)])

    def classify(self, gate_outputs, gate_confidences=None):
        return {"final_tier": _tier(int(np.floor(float(self._firing(gate_outputs) @ self.w) + 0.5)))}

    @classmethod
    def fit(cls, gate_io, labels, ridge=1e-3):
        F = np.array([cls()._firing(o) for o, _ in gate_io])
        y = np.array([URGENCY[t] for t in labels], dtype=float)
        w = np.linalg.solve(F.T @ F + ridge * np.eye(F.shape[1]), F.T @ y)
        return cls(w)


def all_fixed():
    """Rules with no trained parameter."""
    return [WeightedMaxNormalised(), QuantifierOWA(), BandClampedSugeno(), PrioritizedYager()]
