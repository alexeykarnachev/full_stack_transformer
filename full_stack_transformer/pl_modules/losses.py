import torch
import torch.nn.functional as F


def unlikelihood_candidates_loss(logits, target):
    """Unlikelihood candidates loss.

    Args:
        logits (tensor):
            Torch tensor of shape (bs, seq_len, vocab_size), output language
            model scores.
        target (tensor):
            Torch tensor of shape (bs, seq_len), language model target (model
            input tokens itself).

    Returns:
        Not-scaled unlikelihood candidates loss-value.

    Notes:
        This loss is based on penalizing of the previous context tokens.
        Original paper - Welleck et al. https://arxiv.org/pdf/1908.04319.pdf.
    """
    logp = F.log_softmax(logits, 2)
    seq_len = logits.size()[1]
    bs = logits.size()[0]

    logp_flat = logp.view(logits.size()[0] * logits.size()[1], -1)
    tril = torch.tril(torch.ones((seq_len, seq_len), device=logits.device), diagonal=-1)

    cols = target.repeat(1, seq_len).view(seq_len * bs, -1)
    rows = torch.arange(0, cols.size()[0]).unsqueeze(-1).repeat(1, seq_len)

    not_ignore_mask = ~(cols == target.flatten().unsqueeze(-1).repeat(1, seq_len))
    candidates_mask = tril.repeat(bs, 1).bool()
    candidates_mask *= not_ignore_mask

    cols_flat = cols[candidates_mask]
    rows_flat = rows[candidates_mask]

    candidates_logp = logp_flat[rows_flat, cols_flat]

    probs = torch.clamp((1.0 - candidates_logp.exp()), min=1e-5)

    loss = -torch.log(probs)
    loss = loss.mean()

    return loss

