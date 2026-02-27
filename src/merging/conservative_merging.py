"""
Conservative Merging Algorithm for SAFE-Gate

Implements:
  - Basic MIN selection (Equation 2)
  - ACWCM: Adaptive Confidence-Weighted Conservative Merging (Algorithm 1)
  - Gate Conflict Resolution (Algorithm 3)
"""

import math
from typing import List, Dict, Tuple, Optional
from .risk_lattice import RiskTier, RiskLattice


class ConservativeMerging:
    """
    Implements conservative merging with two modes:
      1. Basic MIN selection (safety floor)
      2. ACWCM: Adaptive Confidence-Weighted Conservative Merging
    """

    def __init__(self, mode: str = "acwcm"):
        """
        Args:
            mode: "min" for basic minimum selection, "acwcm" for adaptive merging
        """
        self.lattice = RiskLattice()
        self.mode = mode

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
        patient_id: Optional[str] = None
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
        patient_id: Optional[str] = None
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
        patient_id: Optional[str] = None
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
            return RiskTier.R_STAR, enforcing, audit

        # Line 4-5: Critical always propagates
        r1_gates = [g for g, t in gate_outputs.items() if t == RiskTier.R1]
        if r1_gates:
            enforcing = r1_gates[0]
            audit["merging_logic"].append(f"Phase 1: R1 triggered by {enforcing}")
            audit["final_tier"] = str(RiskTier.R1)
            audit["enforcing_gate"] = enforcing
            return RiskTier.R1, enforcing, audit

        # --- Phase 2: Adaptive confidence-weighted selection ---

        tiers = list(gate_outputs.values())
        confs = list(gate_confidences.values())

        # Line 8: Basic MIN (safety floor)
        r_min = self.lattice.minimum(tiers)

        # Line 9: Confidence-weighted consensus
        numerator = sum(
            c * self._rank(t) for c, t in zip(confs, tiers)
        )
        denominator = sum(confs) if sum(confs) > 0 else 1.0
        tau_w = numerator / denominator

        # Line 10: Ceiling for conservative bias
        r_adapt = self._tier_from_rank(math.ceil(tau_w))

        # Line 11: Bounded 1-tier relaxation
        relaxed = self._relax(r_min, 1)
        # r_final = min(r_adapt, relax(r_min, 1))
        if self._rank(r_adapt) <= self._rank(relaxed):
            r_final = r_adapt
        else:
            r_final = relaxed

        # CND enforcement: if r_min in {R1, R2}, cap final at R2
        if r_min in (RiskTier.R1, RiskTier.R2):
            if self._rank(r_final) > self._rank(RiskTier.R2):
                r_final = RiskTier.R2

        # Find enforcing gate (most conservative)
        enforcing_gates = [g for g, t in gate_outputs.items() if t == r_min]
        enforcing = enforcing_gates[0] if enforcing_gates else list(gate_outputs.keys())[0]

        audit["merging_logic"].append(
            f"Phase 2: r_min={r_min}, tau_w={tau_w:.2f}, "
            f"r_adapt={r_adapt}, relaxed={relaxed}, r_final={r_final}"
        )
        audit["tau_w"] = tau_w
        audit["r_min"] = str(r_min)
        audit["r_adapt"] = str(r_adapt)
        audit["final_tier"] = str(r_final)
        audit["enforcing_gate"] = enforcing

        # Safety property verification
        audit["safety_properties"] = {
            "CP": self._rank(r_final) <= self._rank(r_min) + 1,
            "AC": True,  # already handled in Phase 1
            "CND": True   # enforced above
        }

        return r_final, enforcing, audit

    def resolve_conflicts(
        self,
        gate_outputs: Dict[str, RiskTier],
        gate_confidences: Dict[str, float],
        r_final: RiskTier
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
                    conflicts.append({
                        "gate_i": names[i], "tier_i": str(tiers[i]), "conf_i": confs[i],
                        "gate_j": names[j], "tier_j": str(tiers[j]), "conf_j": confs[j],
                        "difference": diff
                    })

        if conflicts:
            audit["flag"] = "CONFLICTING"
            audit["conflicts"] = conflicts

        # Enforcer = most conservative gate
        min_rank = min(self._rank(t) for t in tiers)
        enforcer_gates = [names[i] for i in range(len(tiers)) if self._rank(tiers[i]) == min_rank]
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
        patient_id: Optional[str]
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
