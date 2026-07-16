"""Exercise 4.2: focal loss for bin imbalance.

The bin-frequency histogram (§4.6.7, notebook §4.3) can show a few bins
dominating the action tokens -- a gripper that is open most of the time,
say. Plain cross-entropy lets those easy, frequent bins swamp the
gradient. Focal loss down-weights well-classified examples by a factor
``(1 - p_t) ** gamma`` so the head keeps learning the rare, hard bins.

Your task: implement ``focal_loss`` (gamma = 2) as a drop-in for the
cross-entropy in ``ParallelActionHead.loss`` / the AR head's ``forward``.

Hint: start from ``torch.nn.functional.cross_entropy(..., reduction=
"none")`` to get per-token CE, recover ``p_t = exp(-ce)``, then scale by
``(1 - p_t) ** gamma`` and take the mean. Compare against
``ch04.parallel_action_head.ParallelActionHead.loss`` (the eps=0 path is
plain CE) to check you match it when gamma = 0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal cross-entropy over the last (bin) axis.

    ``logits``: ``[..., n_bins]``; ``target_bins``: ``[...]`` int64.
    Returns a scalar. With ``gamma = 0`` this must equal plain
    ``F.cross_entropy`` (your first sanity check).
    """
    n_bins = logits.shape[-1]
    ce = F.cross_entropy(
        logits.reshape(-1, n_bins),
        target_bins.reshape(-1),
        reduction="none",
    )
    # TODO: turn per-token CE into focal loss.
    #   p_t  = torch.exp(-ce)
    #   loss = ((1 - p_t) ** gamma) * ce
    #   return loss.mean()
    raise NotImplementedError("implement the focal-loss reweighting")


if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(8, 16, 6, 256)
    targets = torch.randint(0, 256, (8, 16, 6))
    # gamma=0 must match plain cross-entropy.
    ce = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
    fl = focal_loss(logits, targets, gamma=0.0)
    print("cross-entropy:", ce.item())
    print("focal(gamma=0):", fl.item(), "(should match)")
    print("focal(gamma=2):", focal_loss(logits, targets).item())
