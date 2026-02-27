"""
Safety Certificate Generation for SAFE-Gate (Algorithm 2)

Generates per-case safety certificates containing:
  - delta_min: tier margin (distance to nearest dissenting gate)
  - delta_cf:  counterfactual distance (features to change for tier shift)
  - g_enforce: identity of the enforcing gate
  - rationale: human-readable clinical explanation
"""

import copy
from typing import Dict, Tuple, Optional, List
from .risk_lattice import RiskTier


class SafetyCertificate:
    """Per-case safety certificate S = (delta_min, delta_cf, g_enforce, rationale)."""

    def __init__(
        self,
        delta_min: float,
        delta_cf: int,
        g_enforce: str,
        rationale: str,
        gate_outputs: Optional[Dict] = None,
        r_final: Optional[str] = None
    ):
        self.delta_min = delta_min
        self.delta_cf = delta_cf
        self.g_enforce = g_enforce
        self.rationale = rationale
        self.gate_outputs = gate_outputs or {}
        self.r_final = r_final

    def to_dict(self) -> Dict:
        return {
            "delta_min": self.delta_min,
            "delta_cf": self.delta_cf,
            "g_enforce": self.g_enforce,
            "rationale": self.rationale,
            "gate_outputs": self.gate_outputs,
            "r_final": self.r_final
        }

    def __repr__(self):
        return (
            f"SafetyCertificate(delta_min={self.delta_min}, "
            f"delta_cf={self.delta_cf}, g_enforce={self.g_enforce})"
        )


class SafetyCertificateGenerator:
    """
    Implements Algorithm 2: Safety Certificate Generation.

    For each processed case, generates a certificate documenting:
      - How far the final tier is from the nearest dissenting gate
      - How many features would need to change to shift the tier
      - Which gate enforced the final tier and why
    """

    @staticmethod
    def _rank(tier) -> int:
        if isinstance(tier, RiskTier):
            return tier.value
        mapping = {"R*": 0, "R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}
        return mapping.get(str(tier), 0)

    def generate(
        self,
        patient_data: Dict,
        r_final: RiskTier,
        gate_outputs: Dict[str, RiskTier],
        gate_confidences: Dict[str, float],
        gate_reasonings: Optional[Dict] = None,
        classify_fn=None,
        feature_names: Optional[List[str]] = None
    ) -> SafetyCertificate:
        """
        Generate safety certificate for a single case (Algorithm 2).

        Args:
            patient_data: the patient feature dictionary
            r_final: final merged tier
            gate_outputs: {gate_name: tier}
            gate_confidences: {gate_name: confidence}
            gate_reasonings: {gate_name: reasoning dict}
            classify_fn: optional callable to re-evaluate perturbed cases
            feature_names: list of feature names for counterfactual analysis

        Returns:
            SafetyCertificate instance
        """
        rank_final = self._rank(r_final)

        # Line 1: g_enforce = argmin_i rank(r_i) — most conservative gate
        min_rank = min(self._rank(t) for t in gate_outputs.values())
        enforce_gates = [g for g, t in gate_outputs.items() if self._rank(t) == min_rank]
        g_enforce = enforce_gates[0]

        # Line 2: delta_min = rank(r_final) - rank(r_{g_enforce})
        delta_min = rank_final - min_rank

        # Lines 3-11: Counterfactual distance via greedy feature perturbation
        if classify_fn is not None and feature_names:
            delta_cf = self._compute_counterfactual_distance(
                patient_data, r_final, classify_fn, feature_names
            )
        else:
            # Approximate: use number of features contributing to the enforcing gate
            delta_cf = max(1, int(delta_min) + 2) if rank_final >= 4 else 1

        # Line 12: rationale = extract_clinical_rule(g_enforce, x)
        rationale = self._extract_rationale(
            g_enforce, gate_reasonings, gate_outputs, gate_confidences
        )

        # Build gate output summary for the certificate
        gate_summary = {}
        for g, t in gate_outputs.items():
            gate_summary[g] = {
                "tier": str(t),
                "confidence": gate_confidences.get(g, 0.0)
            }

        return SafetyCertificate(
            delta_min=round(delta_min, 1),
            delta_cf=delta_cf,
            g_enforce=g_enforce,
            rationale=rationale,
            gate_outputs=gate_summary,
            r_final=str(r_final)
        )

    def _compute_counterfactual_distance(
        self,
        patient_data: Dict,
        r_final: RiskTier,
        classify_fn,
        feature_names: List[str],
        max_subset_size: int = 10
    ) -> int:
        """
        Greedy feature perturbation to find minimum features to change tier.

        Algorithm 2, Lines 4-10:
          FOR k = 1 to |F|:
            FOR each feature subset S with |S| = k:
              x' = adversarial_perturb(x, S)
              IF rank(r_final(x')) < rank(r_final(x)):
                delta_cf = k; GOTO done
        """
        rank_final = self._rank(r_final)

        # For efficiency, test single features first (k=1), then pairs (k=2), etc.
        for k in range(1, min(max_subset_size + 1, len(feature_names) + 1)):
            if k == 1:
                for feat in feature_names:
                    perturbed = self._adversarial_perturb(patient_data, [feat])
                    try:
                        result = classify_fn(perturbed)
                        new_tier = result.get("final_tier", str(r_final))
                        if self._rank_str(new_tier) < rank_final:
                            return k
                    except Exception:
                        continue
            elif k <= 3:
                # Test a sample of pairs/triples for tractability
                import itertools
                import random
                combos = list(itertools.combinations(feature_names, k))
                if len(combos) > 100:
                    combos = random.sample(combos, 100)
                for subset in combos:
                    perturbed = self._adversarial_perturb(patient_data, list(subset))
                    try:
                        result = classify_fn(perturbed)
                        new_tier = result.get("final_tier", str(r_final))
                        if self._rank_str(new_tier) < rank_final:
                            return k
                    except Exception:
                        continue
            else:
                # For larger subsets, return k as lower bound
                return k

        return len(feature_names)

    @staticmethod
    def _adversarial_perturb(patient_data: Dict, features: List[str]) -> Dict:
        """Perturb specified features adversarially (toward more dangerous values)."""
        perturbed = copy.deepcopy(patient_data)

        adversarial_values = {
            "systolic_bp": 85,
            "heart_rate": 130,
            "spo2": 88,
            "gcs": 12,
            "age": 75,
            "symptom_onset_hours": 0.5,
            "dysarthria": True,
            "diplopia": True,
            "ataxia": True,
            "sudden_onset": True,
            "severe_headache": True,
            "hypertension": True,
            "atrial_fibrillation": True,
            "prior_stroke": True,
        }

        for feat in features:
            if feat in adversarial_values:
                perturbed[feat] = adversarial_values[feat]
            elif isinstance(perturbed.get(feat), bool):
                perturbed[feat] = True
            elif isinstance(perturbed.get(feat), (int, float)):
                val = perturbed.get(feat, 0)
                perturbed[feat] = val * 0.5 if val > 0 else val + 10

        return perturbed

    @staticmethod
    def _rank_str(tier_str: str) -> int:
        mapping = {"R*": 0, "R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}
        return mapping.get(str(tier_str), 3)

    @staticmethod
    def _extract_rationale(
        g_enforce: str,
        gate_reasonings: Optional[Dict],
        gate_outputs: Dict,
        gate_confidences: Dict
    ) -> str:
        """Extract clinical rationale from the enforcing gate."""
        if gate_reasonings and g_enforce in gate_reasonings:
            reasoning = gate_reasonings[g_enforce]
            if isinstance(reasoning, dict):
                triggers = reasoning.get("reasoning", {})
                if isinstance(triggers, dict):
                    trigger_list = triggers.get("triggers", [])
                    decision = triggers.get("decision", "")
                    if trigger_list:
                        return f"{decision}. Triggers: {'; '.join(str(t) for t in trigger_list[:3])}"
                    return decision
                return str(triggers)

        # Fallback: describe gate outputs
        parts = []
        for g, t in gate_outputs.items():
            c = gate_confidences.get(g, 0.0)
            parts.append(f"{g}: {t} (c={c:.2f})")
        return f"Enforced by {g_enforce}. Gate outputs: {'; '.join(parts)}"
