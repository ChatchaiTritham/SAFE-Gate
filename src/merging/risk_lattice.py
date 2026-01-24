"""
Risk Lattice Implementation for SAFE-Gate

Implements the partial ordering on risk tiers: R* ⊑ R1 ⊑ R2 ⊑ R3 ⊑ R4 ⊑ R5
where R* (abstention) is the most conservative tier.
"""

from enum import Enum
from typing import List, Optional


class RiskTier(Enum):
    """Risk tier enumeration with conservative ordering."""
    R_STAR = 0  # Abstention (most conservative)
    R1 = 1      # Critical (life-threatening, immediate care)
    R2 = 2      # High Risk (suspected stroke, urgent evaluation)
    R3 = 3      # Moderate (acute vertigo, standard evaluation)
    R4 = 4      # Low Risk (positional dizziness, delayed OK)
    R5 = 5      # Minimal (chronic dizziness, safe discharge)

    def __str__(self):
        if self == RiskTier.R_STAR:
            return "R*"
        return f"R{self.value}"

    def __repr__(self):
        return str(self)

    def is_more_conservative_than(self, other: 'RiskTier') -> bool:
        """Check if this tier is more conservative (lower value) than another."""
        return self.value < other.value

    def is_critical(self) -> bool:
        """Check if tier represents critical or high-risk classification."""
        return self in [RiskTier.R_STAR, RiskTier.R1, RiskTier.R2]

    def requires_immediate_intervention(self) -> bool:
        """Check if tier requires immediate medical intervention."""
        return self == RiskTier.R1

    def is_abstention(self) -> bool:
        """Check if tier represents abstention."""
        return self == RiskTier.R_STAR

    def get_description(self) -> str:
        """Get clinical description of the risk tier."""
        descriptions = {
            RiskTier.R_STAR: "Abstention: Uncertainty/incomplete data, requires human review",
            RiskTier.R1: "Critical: Life-threatening, immediate care required (<5 min)",
            RiskTier.R2: "High Risk: Suspected stroke, urgent evaluation (<15 min)",
            RiskTier.R3: "Moderate: Acute vertigo, standard evaluation (30-120 min)",
            RiskTier.R4: "Low Risk: Positional dizziness, delayed evaluation OK (1-4 hours)",
            RiskTier.R5: "Minimal: Chronic dizziness, safe discharge to outpatient"
        }
        return descriptions.get(self, "Unknown tier")


class RiskLattice:
    """
    Risk lattice (R*, ⊑) with partial ordering.

    The lattice defines the mathematical structure for conservative merging,
    ensuring that the most conservative tier always prevails.
    """

    @staticmethod
    def minimum(tiers: List[RiskTier]) -> RiskTier:
        """
        Compute minimum (most conservative) tier on lattice.

        Args:
            tiers: List of risk tiers from gates

        Returns:
            Most conservative tier (minimum under ⊑ ordering)

        Mathematical property: min(T1, ..., T6) ⊑ Ti for all i
        """
        if not tiers:
            raise ValueError("Cannot compute minimum of empty tier list")

        # Abstention has highest priority (lowest value)
        if RiskTier.R_STAR in tiers:
            return RiskTier.R_STAR

        # Return tier with minimum value (most conservative)
        return min(tiers, key=lambda t: t.value)

    @staticmethod
    def verify_conservative_property(final_tier: RiskTier, gate_tiers: List[RiskTier]) -> bool:
        """
        Verify that final tier satisfies conservative property.

        Theorem 2 (Conservative Bias Preservation):
        T_final ⊑ Ti for all gate outputs Ti

        Args:
            final_tier: Final merged tier
            gate_tiers: Individual gate outputs

        Returns:
            True if conservative property holds, False otherwise
        """
        for gate_tier in gate_tiers:
            if not (final_tier.value <= gate_tier.value):
                return False
        return True

    @staticmethod
    def from_string(tier_str: str) -> RiskTier:
        """Convert string representation to RiskTier."""
        mapping = {
            'R*': RiskTier.R_STAR,
            'R1': RiskTier.R1,
            'R2': RiskTier.R2,
            'R3': RiskTier.R3,
            'R4': RiskTier.R4,
            'R5': RiskTier.R5
        }
        tier = mapping.get(tier_str.upper())
        if tier is None:
            raise ValueError(f"Invalid tier string: {tier_str}")
        return tier
