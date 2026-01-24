"""
Conservative Merging Algorithm for SAFE-Gate

Implements Algorithm 1 from the paper: Conservative Merging on Risk Lattice
"""

from typing import List, Dict, Tuple, Optional
from .risk_lattice import RiskTier, RiskLattice


class ConservativeMerging:
    """
    Implements conservative merging with abstention-first priority.

    Algorithm 1: Conservative Merging on Risk Lattice
    1. Check for abstention (R*) - if any gate outputs R*, return R*
    2. Otherwise, select minimum tier on lattice (most conservative)
    3. Generate audit trail documenting decision
    """

    def __init__(self):
        self.lattice = RiskLattice()

    def merge(
        self,
        gate_outputs: Dict[str, RiskTier],
        gate_confidences: Dict[str, float],
        patient_id: Optional[str] = None
    ) -> Tuple[RiskTier, str, Dict]:
        """
        Merge six gate outputs using conservative merging algorithm.

        Args:
            gate_outputs: Dictionary mapping gate names (G1-G6) to risk tiers
            gate_confidences: Dictionary mapping gate names to confidence scores [0,1]
            patient_id: Optional patient identifier for audit trail

        Returns:
            Tuple of (final_tier, enforcing_gate, audit_trail)

        Mathematical guarantee (Theorem 2):
            T_final ⊑ Ti for all gate outputs Ti
        """
        # Validate inputs
        if len(gate_outputs) != 6:
            raise ValueError(f"Expected 6 gate outputs, got {len(gate_outputs)}")

        if len(gate_confidences) != 6:
            raise ValueError(f"Expected 6 confidence scores, got {len(gate_confidences)}")

        # Initialize audit trail
        audit_trail = {
            'patient_id': patient_id,
            'gate_outputs': {},
            'merging_logic': [],
            'theorem_verification': {}
        }

        # Record individual gate outputs
        for gate_name, tier in gate_outputs.items():
            confidence = gate_confidences.get(gate_name, 0.0)
            audit_trail['gate_outputs'][gate_name] = {
                'tier': str(tier),
                'confidence': confidence
            }

        # Step 1: Abstention-first priority
        # If any gate outputs R*, final tier becomes R*
        abstention_gates = [
            gate_name for gate_name, tier in gate_outputs.items()
            if tier == RiskTier.R_STAR
        ]

        if abstention_gates:
            final_tier = RiskTier.R_STAR
            enforcing_gate = abstention_gates[0]  # First gate triggering R*

            audit_trail['merging_logic'].append(
                f"Abstention-first priority triggered by {enforcing_gate}"
            )
            audit_trail['merging_logic'].append(
                f"R* enforced by gate {enforcing_gate}"
            )

        else:
            # Step 2: Most-conservative selection on lattice
            # T_final = min_⊑{T1, ..., T6}
            tiers = list(gate_outputs.values())
            final_tier = self.lattice.minimum(tiers)

            # Find which gate(s) enforced this tier
            enforcing_gates = [
                gate_name for gate_name, tier in gate_outputs.items()
                if tier == final_tier
            ]
            enforcing_gate = enforcing_gates[0]  # Take first if multiple

            audit_trail['merging_logic'].append(
                f"Conservative selection: min{{{', '.join([str(t) for t in tiers])}}} = {final_tier}"
            )
            audit_trail['merging_logic'].append(
                f"Tier {final_tier} enforced by gate {enforcing_gate}"
            )

        # Step 3: Verify safety theorems
        # Theorem 2: T_final ⊑ Ti for all i
        conservative_property_holds = self.lattice.verify_conservative_property(
            final_tier, list(gate_outputs.values())
        )

        audit_trail['theorem_verification']['theorem2_conservative_bias'] = {
            'holds': conservative_property_holds,
            'description': f'T_final = {final_tier} ⊑ Ti for all gate outputs'
        }

        if not conservative_property_holds:
            audit_trail['merging_logic'].append(
                "WARNING: Conservative property violation detected!"
            )

        # Record final decision
        audit_trail['final_tier'] = str(final_tier)
        audit_trail['enforcing_gate'] = enforcing_gate

        return final_tier, enforcing_gate, audit_trail

    def get_tier_rank(self, tier: RiskTier) -> int:
        """
        Get numerical rank of tier (lower is more conservative).

        Used for conservative selection on lattice.
        """
        return tier.value
