"""
Conservative Merging Algorithm for SAFE-Gate

Implements:
  - Basic MIN selection (Equation 2)
  - ACWCM: Adaptive Confidence-Weighted Conservative Merging (Algorithm 1)
  - Gate Conflict Resolution (Algorithm 3)
"""

import math
from typing import Dict, List, Optional, Tuple

from .risk_lattice import RiskLattice, RiskTier


class ConservativeMerging:
    """
    Implements conservative merging with two modes:
      1. Basic MIN selection (safety floor)
      2. ACWCM: Adaptive Confidence-Weighted Conservative Merging
    """

    # Consensus thresholds for the relaxation trigger, fixed a priori (not tuned
    # on any split): the confidence-weighted mean rank must sit at least half a
    # tier above the floor, and gates strictly less conservative than the floor
    # must carry at least half of the total confidence (a weighted majority).
    THETA = 0.5
    MIN_MASS = 0.5
    ALPHA = 0.5

    def __init__(self, mode: str = "acwcm", trigger: str = "consensus",
                 theta: float = THETA, min_mass: float = MIN_MASS):
        """
        Args:
            mode: "min" for basic minimum selection, "acwcm" for adaptive merging
            trigger: "consensus" (default) relaxes only under a confidence-weighted
                majority; "any_dissent" is the earlier rule, which relaxes whenever
                tau_w exceeds the floor rank (kept as a sensitivity variant)
            theta: margin of tau_w above rank(r_min) required by "consensus"
            min_mass: share of total confidence that must lie strictly above r_min
        """
        if trigger not in ("consensus", "any_dissent", "fuzzy_quantifier"):
            raise ValueError(f"unknown trigger: {trigger}")
        self.lattice = RiskLattice()
        self.mode = mode
        self.trigger = trigger
        self.theta = theta
        self.min_mass = min_mass

    @staticmethod
    def _rank(tier: RiskTier) -> int:
        """Map tier to numerical rank: R1=1, R2=2, ..., R5=5. R*=0."""
        return tier.value

    @staticmethod
    def _tier_from_rank(rank: int) -> RiskTier:
        """Map numerical rank back to tier."""
        rank = max(0, min(5, rank))
        for t in RiskTier:
            if t.value == rank:
                return t
        return RiskTier.R_STAR

    @staticmethod
    def _relax(tier: RiskTier, k: int) -> RiskTier:
        """
        Return the tier k positions less conservative than tier on the lattice.
        E.g., relax(R2, 1) = R3.
        """
        new_rank = tier.value + k
        new_rank = min(new_rank, 5)  # cannot go beyond R5
        return ConservativeMerging._tier_from_rank(new_rank)

    def merge(
        self,
        gate_outputs: Dict[str, RiskTier],
        gate_confidences: Dict[str, float],
        patient_id: Optional[str] = None,
    ) -> Tuple[RiskTier, str, Dict]:
        """
        Merge six gate outputs using conservative merging.

        Args:
            gate_outputs: {gate_name: RiskTier}
            gate_confidences: {gate_name: float in [0,1]}
            patient_id: optional identifier

        Returns:
            (final_tier, enforcing_gate, audit_trail)
        """
        if self.mode == "acwcm":
            return self._merge_acwcm(gate_outputs, gate_confidences, patient_id)
        else:
            return self._merge_min(gate_outputs, gate_confidences, patient_id)

    def _merge_min(
        self,
        gate_outputs: Dict[str, RiskTier],
        gate_confidences: Dict[str, float],
        patient_id: Optional[str] = None,
    ) -> Tuple[RiskTier, str, Dict]:
        """Basic minimum selection (Equation 2)."""
        audit = self._init_audit(gate_outputs, gate_confidences, patient_id)

        tiers = list(gate_outputs.values())

        # Abstention check
        abstention_gates = [g for g, t in gate_outputs.items() if t == RiskTier.R_STAR]
        if abstention_gates:
            enforcing = abstention_gates[0]
            audit["merging_logic"].append(f"R* triggered by {enforcing}")
            audit["final_tier"] = str(RiskTier.R_STAR)
            audit["enforcing_gate"] = enforcing
            audit["safety_properties"] = self.check_properties(gate_outputs, RiskTier.R_STAR)
            return RiskTier.R_STAR, enforcing, audit

        # Minimum selection
        final = self.lattice.minimum(tiers)
        enforcing_gates = [g for g, t in gate_outputs.items() if t == final]
        enforcing = enforcing_gates[0]

        audit["merging_logic"].append(
            f"MIN selection: {final} (enforced by {enforcing})"
        )
        audit["final_tier"] = str(final)
        audit["enforcing_gate"] = enforcing
        return final, enforcing, audit

    def _merge_acwcm(
        self,
        gate_outputs: Dict[str, RiskTier],
        gate_confidences: Dict[str, float],
        patient_id: Optional[str] = None,
    ) -> Tuple[RiskTier, str, Dict]:
        """
        Adaptive Confidence-Weighted Conservative Merging (Algorithm 1).

        Phase 1: Safety-critical signal enforcement (hard constraints)
        Phase 2: Adaptive confidence-weighted selection
        Phase 3: Safety certificate generation (handled externally)
        """
        audit = self._init_audit(gate_outputs, gate_confidences, patient_id)

        # --- Phase 1: Hard constraints ---

        # Line 2-3: Abstention always propagates
        abstention_gates = [g for g, t in gate_outputs.items() if t == RiskTier.R_STAR]
        if abstention_gates:
            enforcing = abstention_gates[0]
            audit["merging_logic"].append(f"Phase 1: R* triggered by {enforcing}")
            audit["final_tier"] = str(RiskTier.R_STAR)
            audit["enforcing_gate"] = enforcing
            audit["safety_properties"] = self.check_properties(gate_outputs, RiskTier.R_STAR)
            return RiskTier.R_STAR, enforcing, audit

        # Line 4-5: Critical always propagates
        r1_gates = [g for g, t in gate_outputs.items() if t == RiskTier.R1]
        if r1_gates:
            enforcing = r1_gates[0]
            audit["merging_logic"].append(f"Phase 1: R1 triggered by {enforcing}")
            audit["final_tier"] = str(RiskTier.R1)
            audit["enforcing_gate"] = enforcing
            audit["safety_properties"] = self.check_properties(gate_outputs, RiskTier.R1)
            return RiskTier.R1, enforcing, audit

        # --- Phase 2: Adaptive confidence-weighted selection ---

        tiers = list(gate_outputs.values())
        confs = list(gate_confidences.values())

        # Line 8: Basic MIN (safety floor)
        r_min = self.lattice.minimum(tiers)

        # Line 9: Confidence-weighted consensus
        numerator = sum(c * self._rank(t) for c, t in zip(confs, tiers))
        denominator = sum(confs) if sum(confs) > 0 else 1.0
        tau_w = numerator / denominator

        # Line 10: confidence mass of gates strictly less conservative than r_min
        total = sum(confs)
        above = sum(c for c, t in zip(confs, tiers) if self._rank(t) > self._rank(r_min))
        mass_share = above / total if total > 0 else 0.0

        # Line 11: relaxation trigger
        if self.trigger == "any_dissent":
            # Earlier rule: min(ceil(tau_w), relax(r_min,1)) relaxes iff tau_w > rank(r_min)
            relax_ok = tau_w > self._rank(r_min)
        elif self.trigger == "fuzzy_quantifier":
            # Graded generalisation of the weighted-quota trigger: the degree to
            # which "most of the confidence lies above the floor" holds, with the
            # RIM quantifier Q(r) = clip((r - 0.3) / 0.5). Relax iff the margin
            # condition holds and Q(mass_share) >= ALPHA. For a monotone Q this
            # equals a crisp quota at Q^{-1}(ALPHA) = 0.55; the degree is kept in
            # the audit trail.
            degree = min(1.0, max(0.0, (mass_share - 0.3) / 0.5))
            audit["quantifier_degree"] = degree
            relax_ok = (tau_w >= self._rank(r_min) + self.theta and degree >= self.ALPHA)
        else:
            relax_ok = (tau_w >= self._rank(r_min) + self.theta
                        and mass_share >= self.min_mass)

        # Line 12: R2 hold (critical non-dilution) and the R5 ceiling
        if r_min == RiskTier.R2 or r_min == RiskTier.R5:
            relax_ok = False

        r_final = self._relax(r_min, 1) if relax_ok else r_min
        r_adapt = r_final

        # Find enforcing gate (most conservative)
        enforcing_gates = [g for g, t in gate_outputs.items() if t == r_min]
        enforcing = (
            enforcing_gates[0] if enforcing_gates else list(gate_outputs.keys())[0]
        )

        audit["merging_logic"].append(
            f"Phase 2: r_min={r_min}, tau_w={tau_w:.2f}, "
            f"mass_share={mass_share:.2f}, trigger={self.trigger}, r_final={r_final}"
        )
        audit["tau_w"] = tau_w
        audit["r_min"] = str(r_min)
        audit["r_adapt"] = str(r_adapt)
        audit["mass_share"] = mass_share
        audit["final_tier"] = str(r_final)
        audit["enforcing_gate"] = enforcing

        # Safety property verification
        audit["safety_properties"] = self.check_properties(gate_outputs, r_final)

        return r_final, enforcing, audit

    @staticmethod
    def check_properties(gate_outputs: Dict[str, RiskTier], r_final: RiskTier) -> Dict[str, bool]:
        """Evaluate the merge-level properties on this input/output pair.

        Every flag is computed from the gate outputs and the merged tier; none
        is asserted by construction. CP is vacuous (True) when a gate abstains,
        AC is vacuous when no gate abstains, CND is vacuous when r_min is not
        critical.
        """
        tiers = list(gate_outputs.values())
        abstain = any(t == RiskTier.R_STAR for t in tiers)
        ranked = [t.value for t in tiers if t != RiskTier.R_STAR]
        r_min = min(ranked) if ranked else 0
        fr = r_final.value
        return {
            "CP": abstain or fr <= r_min + 1,
            "AC": (not abstain) or r_final == RiskTier.R_STAR,
            "CND": abstain or r_min not in (1, 2) or fr in (0, 1, 2),
        }

    def resolve_conflicts(
        self,
        gate_outputs: Dict[str, RiskTier],
        gate_confidences: Dict[str, float],
        r_final: RiskTier,
    ) -> Dict:
        """
        Gate Conflict Resolution and Audit Trail Generation (Algorithm 3).

        Args:
            gate_outputs: {gate_name: tier}
            gate_confidences: {gate_name: confidence}
            r_final: the merged final tier

        Returns:
            Audit trail with conflict flags and support information.
        """
        audit = {"flag": "CONSENSUS", "conflicts": [], "enforcer": None, "support": 0}

        names = list(gate_outputs.keys())
        tiers = list(gate_outputs.values())
        confs = list(gate_confidences.values())

        # Find conflicts: pairs differing by >= 2 tiers
        conflicts = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                diff = abs(self._rank(tiers[i]) - self._rank(tiers[j]))
                if diff >= 2:
                    conflicts.append(
                        {
                            "gate_i": names[i],
                            "tier_i": str(tiers[i]),
                            "conf_i": confs[i],
                            "gate_j": names[j],
                            "tier_j": str(tiers[j]),
                            "conf_j": confs[j],
                            "difference": diff,
                        }
                    )

        if conflicts:
            audit["flag"] = "CONFLICTING"
            audit["conflicts"] = conflicts

        # Enforcer = most conservative gate
        min_rank = min(self._rank(t) for t in tiers)
        enforcer_gates = [
            names[i] for i in range(len(tiers)) if self._rank(tiers[i]) == min_rank
        ]
        audit["enforcer"] = enforcer_gates[0]

        # Support = gates within 1 tier of final
        audit["support"] = sum(
            1 for t in tiers if self._rank(t) <= self._rank(r_final) + 1
        )

        return audit

    def _init_audit(
        self,
        gate_outputs: Dict[str, RiskTier],
        gate_confidences: Dict[str, float],
        patient_id: Optional[str],
    ) -> Dict:
        """Create initial audit trail structure."""
        audit = {
            "patient_id": patient_id,
            "mode": self.mode,
            "gate_outputs": {
                g: {"tier": str(t), "confidence": gate_confidences.get(g, 0.0)}
                for g, t in gate_outputs.items()
            },
            "merging_logic": [],
        }
        return audit
