import torch

from full_stack_transformer.losses.unlikelihood_candidates_loss import \
    unlikelihood_candidates_loss


def test_all_token_ids_equal():
    bs, seq_len, vocab_size = 8, 12, 30
    logits = torch.randn(bs, seq_len, vocab_size)
    target = torch.ones(bs, seq_len)
    loss = unlikelihood_candidates_loss(logits=logits, target=target)
    assert loss == torch.tensor(0)


def test_loss():
    bs, seq_len, vocab_size = 2, 4, 7
    logits = torch.arange(0, bs*seq_len*vocab_size)
    logits = logits.reshape(bs, seq_len, vocab_size).float()
    target = torch.arange(0, bs*seq_len).reshape((bs, seq_len))
    loss = unlikelihood_candidates_loss(logits=logits, target=target)
    assert loss.allclose(torch.tensor(0.15207734704))