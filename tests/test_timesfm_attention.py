import unittest

import torch
from timesfm3.torch.transformer import MultiHeadAttention
from torch.nn.attention import SDPBackend, sdpa_kernel

from models.timesfm3 import bounded_attention


class TimesFMAttentionTest(unittest.TestCase):
    def test_batch_partition_preserves_outputs_and_masks(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        attention = MultiHeadAttention(num_heads=4, in_features=64, qk_norm="rms",
                                       use_bias=False, use_sdpa=True).eval()
        original = MultiHeadAttention.forward
        # Exercise both batch limits, uneven final chunks and the unsplit path.
        for batch, tokens, should_partition in ((2, 384, False), (5, 257, True),
                                                 (7, 384, True), (101, 64, True)):
            with self.subTest(batch=batch, tokens=tokens):
                inputs = torch.randn(batch, tokens, 64)
                mask = torch.zeros(batch, tokens, dtype=torch.bool)
                mask[::2, 0] = True
                mask[1::2, -4:] = True
                with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                    expected, expected_cache, expected_mask = attention(inputs, patch_mask=mask)
                    with bounded_attention(torch) as evidence:
                        actual, cache, actual_mask = attention(inputs, patch_mask=mask)
                self.assertEqual(bool(evidence), should_partition)
                self.assertIsNone(expected_cache)
                self.assertIsNone(cache)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
                torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)
                self.assertIs(MultiHeadAttention.forward, original)
