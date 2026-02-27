"""
Baseline: Bayesian Model Averaging (BMA)

Each gate provides a probability distribution over tiers derived from
confidence-weighted softmax transformation. BMA computes the
posterior-weighted average.
"""

import numpy as np
from typing import Dict, List


class BayesianModelAveraging:
    """
    Bayesian Model Averaging baseline.

    p_BMA(r|x) = sum_i w_i * p_i(r|x)

    where weights w_i are proportional to each gate's validation-set
    log-likelihood (approximated by confidence).
    """

    TIERS = ['R1', 'R2', 'R3', 'R4', 'R5']
    TIER_IDX = {t: i for i, t in enumerate(TIERS)}

    def classify(
        self,
        gate_outputs: Dict[str, str],
        gate_confidences: Dict[str, float],
        gate_val_scores: Dict[str, float] = None
    ) -> Dict:
        """
        Combine gate predictions via BMA.

        Args:
            gate_outputs: {gate_name: tier_string}
            gate_confidences: {gate_name: confidence}
            gate_val_scores: optional validation log-likelihoods for weight computation

        Returns:
            {'final_tier': str, 'probabilities': dict}
        """
        n_tiers = len(self.TIERS)

        # Compute weights proportional to validation-set performance
        if gate_val_scores:
            raw_weights = np.array([gate_val_scores.get(g, 1.0) for g in gate_outputs])
        else:
            # Approximate weights from confidence
            raw_weights = np.array([gate_confidences.get(g, 0.5) for g in gate_outputs])

        # Normalise weights
        weights = raw_weights / (raw_weights.sum() + 1e-10)

        # Build per-gate probability distributions (softmax around predicted tier)
        combined = np.zeros(n_tiers)

        for i, (gate, tier) in enumerate(gate_outputs.items()):
            conf = gate_confidences.get(gate, 0.5)
            p = self._tier_to_distribution(tier, conf, n_tiers)
            combined += weights[i] * p

        # Normalise
        combined /= (combined.sum() + 1e-10)

        best_idx = np.argmax(combined)
        best_tier = self.TIERS[best_idx]

        return {
            'final_tier': best_tier,
            'probabilities': {self.TIERS[i]: round(float(combined[i]), 4) for i in range(n_tiers)},
            'abstain': False
        }

    def _tier_to_distribution(self, tier: str, confidence: float, n: int) -> np.ndarray:
        """Convert tier + confidence to softmax probability distribution."""
        idx = self.TIER_IDX.get(tier, 2)
        logits = np.zeros(n)
        logits[idx] = confidence * 5.0  # scale confidence to logit

        # Spread some mass to adjacent tiers
        if idx > 0:
            logits[idx - 1] = (1.0 - confidence) * 2.0
        if idx < n - 1:
            logits[idx + 1] = (1.0 - confidence) * 2.0

        # Softmax
        exp_logits = np.exp(logits - logits.max())
        return exp_logits / exp_logits.sum()
