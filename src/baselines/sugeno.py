"""
Ordinal fuzzy aggregation baselines on the finite risk chain.

Both rules are discrete Sugeno integrals (Marichal 2001; Grabisch et al. 2009)
of the gate outputs, read as urgency levels on the chain
    L = {0 < 1 < 2 < 3 < 4 < 5},   urgency(R5)=1, ..., urgency(R1)=5,
with 0 the bottom element. Gate confidences enter through the fuzzy measure,
quantised onto the same chain so that the whole computation stays ordinal
(only min, max and comparisons are used on the inputs):

    q(c) = ceil(5 * c) clipped to {0..5}.

  SugenoAdditive  mu(A) = q( sum_{i in A} c_i / sum_i c_i )
                  A quantised additive (confidence-proportional) capacity.
                  This is NOT an ordinal weighted median: with c = 1 for all
                  gates, (R1,R1,R1,R5,R5,R5) gives R3, outside the inputs.
  WeightedMax     mu(A) = max_{i in A} q(c_i)
                  A possibility-type capacity, BUT mu(N) is forced to TOP
                  below, so unless some q(c_i) = TOP it is not the Dubois-Prade
                  weighted max (which needs a normalised distribution); the
                  normalised version is ordinal_extra.WeightedMaxNormalised.
                  With every q(c_i) = 5 it equals basic MIN selection.

    S_mu(x) = max_{l in L} min( l, mu({i : x_i >= l}) ).

The output urgency is mapped back to a tier; urgency 0 (no level supported)
is mapped to the least urgent tier R5. Neither rule has an abstention element:
as for the other baselines, an R* gate output is read as R1 before
aggregation, so abstention correctness cannot hold for them.
"""

import math
from typing import Dict

TIERS = ["R1", "R2", "R3", "R4", "R5"]
URGENCY = {"R1": 5, "R2": 4, "R3": 3, "R4": 2, "R5": 1}
TIER_OF = {5: "R1", 4: "R2", 3: "R3", 2: "R4", 1: "R5", 0: "R5"}
TOP = 5


def quantise(c: float) -> int:
    """Map a confidence in [0, 1] onto the chain {0..5}."""
    return max(0, min(TOP, math.ceil(TOP * float(c) - 1e-12)))


def sugeno_integral(levels: Dict[str, int], capacity) -> int:
    """Discrete Sugeno integral of chain-valued inputs w.r.t. a chain-valued capacity."""
    best = 0
    for lev in range(1, TOP + 1):
        a = frozenset(g for g, x in levels.items() if x >= lev)
        best = max(best, min(lev, capacity(a)))
    return best


class _SugenoBase:
    def _levels(self, gate_outputs):
        return {g: URGENCY[t if t in URGENCY else "R1"] for g, t in gate_outputs.items()}

    def classify(self, gate_outputs: Dict[str, str], gate_confidences: Dict[str, float]) -> Dict:
        levels = self._levels(gate_outputs)
        cap = self.capacity(gate_confidences, list(levels))
        s = sugeno_integral(levels, cap)
        return {"final_tier": TIER_OF[s], "urgency": s}


class SugenoAdditive(_SugenoBase):
    """Sugeno integral w.r.t. the quantised confidence-proportional capacity."""

    name = "Sugeno integral (additive capacity)"

    @staticmethod
    def capacity(confs, gates):
        total = sum(confs[g] for g in gates) or 1.0

        def mu(a):
            if not a:
                return 0
            if len(a) == len(gates):
                return TOP
            return quantise(sum(confs[g] for g in a) / total)
        return mu


class WeightedMax(_SugenoBase):
    """Sugeno integral w.r.t. a possibility measure: ordinal weighted maximum."""

    name = "Ordinal weighted maximum (possibility capacity)"

    @staticmethod
    def capacity(confs, gates):
        def mu(a):
            if not a:
                return 0
            if len(a) == len(gates):
                return TOP
            return max(quantise(confs[g]) for g in a)
        return mu
