"""Substrate: WHERE a rollout physically happened.

This is the single most important axis in the whole framework, and it is not
a detail of the runner -- it changes what a result MEANS:

  SIMULATION  a policy rollout in a physics engine. Cheap, repeatable,
              arbitrarily many episodes, and systematically wrong in ways the
              simulator does not know about.
  HARDWARE    a rollout on the real robot. Expensive, few episodes, noisy,
              and the only thing that actually counts.

Aggregating across substrates is NOT consensus. Two simulator validators
agreeing tells you the simulator is deterministic. A simulator and a real robot
agreeing tells you the policy transfers -- and their DISAGREEMENT is the
sim2real gap, which is a measurement, not a fault. The consensus layer must
therefore never pool them into one number; see consensus.py.

MOCK is called out separately because a mock result must never reach a
benchmark claim, no matter how the numbers look.
"""
from __future__ import annotations

SIMULATION = "simulation"
HARDWARE = "hardware"
MOCK = "mock"

#: Substrates whose results may back a published benchmark claim.
BENCHMARK_VALID = frozenset({SIMULATION, HARDWARE})

def is_benchmark_valid(substrate: str) -> bool:
    return substrate in BENCHMARK_VALID
