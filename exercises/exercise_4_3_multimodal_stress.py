"""Exercise 4.3: multimodal stress test over 100 rollouts.

§4.5.3 shows the parallel head leaking off-diagonal mass on a
coordinated ``(grasp, lift)`` state. This exercise turns that anecdote
into a number. Sample each head a few hundred times on one held-out
pre-grasp state and measure the fraction of samples that land in a
*physically incoherent* quadrant (closed gripper while the wrist is
dropping, say). The autoregressive head should score far lower.

Your task: fill in ``incoherent_fraction`` and run the comparison.

Hint: the parallel head samples with
``ParallelActionHead.sample(pooled)`` -> ``[B, H, D]`` bin ids; the AR
head decodes a coherent chunk through ``DiscretePolicy._decode_chunk``
(or sample it token by token). Reuse the notebook's §4.7 gripper/wrist
pairing (dims 5 and 3) and count pairs where one joint is in its "high"
half and the other in its "low" half -- the off-diagonal the data never
visits. See ``ch04.diagnostics.plot_joint_coordination`` for the exact
2D-histogram framing.
"""

from __future__ import annotations

import numpy as np


def incoherent_fraction(
    pairs: np.ndarray, n_bins: int = 256
) -> float:
    """Fraction of ``[N, 2]`` (gripper, wrist) bin pairs off-diagonal.

    Define "off-diagonal" as one joint in the top half of its range
    while the other is in the bottom half -- the incoherent quadrants
    a coordinated demonstration never visits.
    """
    half = n_bins // 2
    # TODO: count pairs where exactly one of the two bins is >= half,
    #   i.e. (a >= half) XOR (b >= half), and divide by len(pairs).
    #   hi = pairs >= half
    #   off = hi[:, 0] ^ hi[:, 1]
    #   return float(off.mean())
    raise NotImplementedError("count the off-diagonal fraction")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Stand-in samples: a correlated (coherent) set vs an independent
    # (incoherent) set, to check the metric before wiring real heads.
    diag = rng.integers(0, 256, size=300)
    coherent = np.stack([diag, np.clip(diag + rng.integers(-4, 5, 300),
                                       0, 255)], axis=1)
    independent = np.stack([rng.integers(0, 256, 300),
                            rng.integers(0, 256, 300)], axis=1)
    print("coherent  off-diagonal:", incoherent_fraction(coherent))
    print("independent off-diagonal:", incoherent_fraction(independent))
    print("(the AR head should look like the coherent row)")
