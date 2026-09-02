import torch

from src.model.kv_cache import KVCache
from src.model.paligemma.config import GemmaConfig
from src.model.paligemma.gemma import GemmaAttention


def _tiny_config() -> GemmaConfig:
    return GemmaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        attention_dropout=0.0,
    )


def _causal_mask(batch_size: int, query_length: int, key_length: int) -> torch.Tensor:
    mask = torch.full((batch_size, 1, query_length, key_length), float("-inf"))
    for query_index in range(query_length):
        mask[:, :, query_index, : query_index + 1] = 0
    return mask


def test_gemma_attention_applies_rotary_embedding_for_batched_tokens():
    """The text encoder must reach attention with the current rotary API."""
    torch.manual_seed(0)
    attention = GemmaAttention(_tiny_config(), layer_idx=0).eval()
    hidden_states = torch.randn(2, 3, 8)
    position_ids = torch.arange(3).expand(2, -1)

    output, weights = attention(
        hidden_states,
        attention_mask=_causal_mask(2, 3, 3),
        position_ids=position_ids,
    )

    assert output.shape == (2, 3, 8)
    assert weights.shape == (2, 2, 3, 3)
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()


def test_gemma_attention_rotary_embedding_supports_incremental_kv_cache():
    """RoPE receives the query tensor in both prefill and decode paths."""
    torch.manual_seed(1)
    attention = GemmaAttention(_tiny_config(), layer_idx=0).eval()
    cache = KVCache()
    prefix = torch.randn(1, 2, 8)
    next_token = torch.randn(1, 1, 8)

    attention(
        prefix,
        attention_mask=_causal_mask(1, 2, 2),
        position_ids=torch.tensor([[0, 1]]),
        kv_cache=cache,
    )
    output, weights = attention(
        next_token,
        attention_mask=torch.zeros(1, 1, 1, 3),
        position_ids=torch.tensor([[2]]),
        kv_cache=cache,
    )

    assert cache.num_items() == 3
    assert output.shape == (1, 1, 8)
    assert weights.shape == (1, 2, 1, 3)
    assert torch.isfinite(output).all()
