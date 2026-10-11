"""
Baseline: Dempster-Shafer Evidence Combination

Converts gate outputs to mass functions over the risk tier frame,
applies Dempster's combination rule, and selects the tier with
highest combined belief.
"""

from typing import Dict, List, Tuple

import numpy as np


TIE_RTOL = 1e-9


def argmax_tol(scores: Dict[str, float], order: List[str], toward: str = "critical") -> str:
    """Argmax over ``order`` with ties within a relative tolerance broken toward
    the first (``critical``) or last (``benign``) tied element."""
    best = max(scores[t] for t in order)
    tied = [t for t in order if scores[t] >= best - TIE_RTOL * max(abs(best), 1.0)]
    return tied[0] if toward == "critical" else tied[-1]


class DempsterShaferCombination:
    """
    Dempster-Shafer evidence combination baseline.

    Each gate's output is converted to a mass function over R = {R1,...,R5},
    with mass concentrated on the output tier proportional to confidence c_i,
    and residual mass distributed over the full frame Theta.
    """

    TIERS = ['R1', 'R2', 'R3', 'R4', 'R5']

    def __init__(self, abstention: str = "critical"):
        """abstention: "critical" reads an R* output as R1 (default, as in the
        manuscript); "vacuous" reads it as total ignorance m(Theta) = 1."""
        self.abstention = abstention

    def classify(
        self, gate_outputs: Dict[str, str], gate_confidences: Dict[str, float]
    ) -> Dict:
        """
        Combine gate evidence using Dempster's rule.

        Args:
            gate_outputs: {gate_name: tier_string}
            gate_confidences: {gate_name: confidence}

        Returns:
            {'final_tier': str, 'beliefs': dict, 'abstain': bool}
        """
        # Convert each gate to a mass function
        mass_functions = []
        for gate, tier in gate_outputs.items():
            conf = gate_confidences.get(gate, 0.5)
            if tier not in self.TIERS and self.abstention == "vacuous":
                continue  # an abstaining gate contributes the vacuous mass m(Theta) = 1
            m = self._gate_to_mass(tier, conf)
            mass_functions.append(m)

        # Combine all mass functions via Dempster's rule
        if not mass_functions:
            mass_functions = [{'Theta': 1.0}]
        combined = mass_functions[0]
        for i in range(1, len(mass_functions)):
            combined = self._combine(combined, mass_functions[i])

        # Select the tier with the highest combined mass. Masses within a
        # relative tolerance of the maximum are treated as exact ties (floating-
        # point products of equal factors need not compare equal) and are broken
        # toward the more critical tier.
        best_tier = argmax_tol({t: combined.get(t, 0.0) for t in self.TIERS}, self.TIERS)
        abstain = combined.get('Theta', 0.0) > 0.5

        return {
            'final_tier': best_tier,
            'beliefs': {t: round(combined.get(t, 0.0), 4) for t in self.TIERS},
            'abstain': abstain,
            'abstain_rate': combined.get('Theta', 0.0),
        }

    def _gate_to_mass(self, tier: str, confidence: float) -> Dict[str, float]:
        """Convert gate output + confidence to mass function."""
        m = {}
        # An abstention (R*) carries no rank; map it to the most conservative
        # tier (R1), matching the clean step in run_all.py and the manuscript.
        t = tier if tier in self.TIERS else 'R1'
        m[t] = confidence
        m['Theta'] = 1.0 - confidence  # residual to full frame
        return m

    def _combine(self, m1: Dict[str, float], m2: Dict[str, float]) -> Dict[str, float]:
        """Combine two mass functions using Dempster's rule."""
        combined = {}
        conflict = 0.0

        # All focal elements
        elements1 = {k: v for k, v in m1.items() if v > 0}
        elements2 = {k: v for k, v in m2.items() if v > 0}

        for a, ma in elements1.items():
            for b, mb in elements2.items():
                intersection = self._intersect(a, b)
                product = ma * mb
                if intersection is None:
                    conflict += product
                else:
                    combined[intersection] = combined.get(intersection, 0.0) + product

        # Normalise by (1 - conflict)
        norm = 1.0 - conflict
        if norm > 1e-10:
            for key in combined:
                combined[key] /= norm
        else:
            # Total conflict: assign uniform
            for t in self.TIERS:
                combined[t] = 1.0 / len(self.TIERS)

        return combined

    def _intersect(self, a: str, b: str):
        """Compute intersection of two focal elements."""
        if a == 'Theta':
            return b
        if b == 'Theta':
            return a
        if a == b:
            return a
        return None  # empty intersection (conflict)
