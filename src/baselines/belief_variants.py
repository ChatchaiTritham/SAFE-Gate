"""
Belief-function combination variants for the round-4 comparison.

Frame Theta = {R1..R5}. Each gate i reporting tier t with confidence c_i gives
a simple support function: either on the singleton {t} ("singleton" coding,
as in the Dempster-Shafer baseline) or on the upper set U_t = {R1..t} ("this
tier or worse", consonant coding). Mass c_i (optionally discounted by a
reliability alpha_i) on the focal set and the rest on Theta. An abstention R*
is read as R1 ("critical") or as the vacuous mass m(Theta) = 1 ("vacuous").

Combination rules (exact enumeration over the 2^n focal-element choices):
  dempster   conjunctive, conflict normalised away
  yager      conjunctive, conflict transferred to Theta
  dp         Dubois-Prade: each conflicting product goes to the union of its
             focal sets (n-ary disjunctive reallocation)
  pcr6       each conflicting product redistributed to the involved focal
             sets proportionally to the masses that the sources gave them
  cautious   Denoeux (2008) cautious rule for separable masses: per focal set
             the smallest weight w = 1 - c (i.e. the largest confidence) is
             kept, then the resulting simple supports are combined
             conjunctively (normalised for the decision)

Decisions:
  pignistic  argmax BetP, ties (relative tol 1e-9) toward the critical tier
  min_cost   minimum expected cost under BetP with cost d for over-triage by d
             tiers and LAMBDA * d for under-triage by d tiers
  bel_median (upper-set coding) most urgent t with Bel(U_t) >= 0.5
"""
from __future__ import annotations

from itertools import product
from typing import Dict, Optional

from .dempster_shafer import argmax_tol

TIERS = ["R1", "R2", "R3", "R4", "R5"]
IDX = {t: i for i, t in enumerate(TIERS)}
THETA = frozenset(range(5))
LAMBDA = 5.0


def _focal(tier: str, coding: str) -> frozenset:
    i = IDX[tier]
    return frozenset([i]) if coding == "singleton" else frozenset(range(i + 1))


def sources(gate_outputs, gate_confidences, coding="singleton", abstention="critical",
            reliability: Optional[Dict[str, float]] = None):
    out = []
    for g, t in gate_outputs.items():
        if t not in IDX:
            if abstention == "vacuous":
                continue
            t = "R1"
        c = gate_confidences.get(g, 0.5) * (reliability.get(g, 1.0) if reliability else 1.0)
        out.append((_focal(t, coding), c))
    return out


def combine(srcs, rule="dempster"):
    if rule == "cautious":
        best = {}
        for a, c in srcs:
            best[a] = max(best.get(a, 0.0), c)
        srcs, rule = list(best.items()), "dempster"
    m: Dict[frozenset, float] = {}
    conflict = 0.0
    for choice in product((0, 1), repeat=len(srcs)):
        p, inter, sets, masses = 1.0, THETA, [], []
        for (a, c), ch in zip(srcs, choice):
            s, w = (a, c) if ch == 0 else (THETA, 1.0 - c)
            p *= w
            inter = inter & s
            sets.append(s)
            masses.append(w)
        if p == 0.0:
            continue
        if inter:
            m[inter] = m.get(inter, 0.0) + p
            continue
        conflict += p
        if rule == "yager":
            m[THETA] = m.get(THETA, 0.0) + p
        elif rule == "dp":
            u = frozenset().union(*sets)
            m[u] = m.get(u, 0.0) + p
        elif rule == "pcr6":
            tot = sum(masses)
            for s, w in zip(sets, masses):
                m[s] = m.get(s, 0.0) + p * w / tot
    if rule == "dempster":
        z = 1.0 - conflict
        m = {k: v / z for k, v in m.items()} if z > 1e-12 else {THETA: 1.0}
    if not srcs:
        m = {THETA: 1.0}
    return m


def betp(m) -> Dict[str, float]:
    z = sum(m.values()) or 1.0
    b = dict.fromkeys(TIERS, 0.0)
    for a, v in m.items():
        for i in a:
            b[TIERS[i]] += v / len(a) / z
    return b


def min_cost(prob: Dict[str, float], lam: float = LAMBDA) -> str:
    def cost(pred):
        return sum(p * ((IDX[pred] - IDX[t]) * lam if IDX[pred] > IDX[t] else IDX[t] - IDX[pred])
                   for t, p in prob.items())
    costs = {t: -cost(t) for t in TIERS}
    return argmax_tol(costs, TIERS)


def bel_median(m, beta: float = 0.5) -> str:
    z = sum(m.values()) or 1.0
    for i, t in enumerate(TIERS):
        u = frozenset(range(i + 1))
        if sum(v for a, v in m.items() if a <= u) / z >= beta - 1e-12:
            return t
    return "R5"


class BeliefRule:
    def __init__(self, rule="dempster", coding="singleton", decision="pignistic",
                 abstention="critical", reliability=None, name=None):
        self.rule, self.coding, self.decision = rule, coding, decision
        self.abstention, self.reliability = abstention, reliability
        self.name = name or f"DS[{rule},{coding},{decision}]"

    def classify(self, gate_outputs, gate_confidences):
        m = combine(sources(gate_outputs, gate_confidences, self.coding, self.abstention,
                            self.reliability), self.rule)
        if self.decision == "bel_median":
            return {"final_tier": bel_median(m)}
        p = betp(m)
        if self.decision == "min_cost":
            return {"final_tier": min_cost(p)}
        return {"final_tier": argmax_tol(p, TIERS)}


def bma_min_cost(bma, gate_outputs, gate_confidences):
    """Minimum-expected-cost decision on the BMA probabilities."""
    probs = bma.classify(gate_outputs, gate_confidences)["probabilities"]
    return min_cost(probs)


def variants(reliability=None):
    v = [
        BeliefRule("dempster", name="DS Dempster (singleton, BetP)"),
        BeliefRule("yager", name="DS Yager rule"),
        BeliefRule("dp", name="DS Dubois-Prade rule"),
        BeliefRule("pcr6", name="DS PCR6"),
        BeliefRule("cautious", name="DS cautious rule (Denoeux)"),
        BeliefRule("dempster", decision="min_cost", name="DS min expected cost"),
        BeliefRule("dempster", abstention="vacuous", name="DS (abstention vacuous)"),
        BeliefRule("dempster", coding="upper", name="DS upper-set (BetP)"),
        BeliefRule("dempster", coding="upper", decision="bel_median",
                   name="DS upper-set (Bel median)"),
    ]
    if reliability:
        v.append(BeliefRule("dempster", reliability=reliability, name="DS discounted (train reliability)"))
    return v
