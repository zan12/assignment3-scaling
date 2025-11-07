import torch

from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    logits: Float[Tensor, " ... vocab_size"],
    targets: Int[Tensor, " ..."]
):
    vocab_size = logits.shape[-1]
    logits = logits.view(-1, vocab_size)
    targets = targets.view(-1)
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits -= logits_max
    sum_exp_logits = torch.sum(torch.exp(logits), dim = -1)
    return -torch.mean(logits[torch.arange(logits.size(0)),targets] - torch.log(sum_exp_logits))
