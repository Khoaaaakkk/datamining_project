import importlib.util
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class TestAttentionMIL(unittest.TestCase):
    def test_forward_single_bag(self):
        import torch
        from src.models.mil_classifier import AttentionMIL

        model = AttentionMIL(in_dim=128, attn_dim=32, num_classes=3)
        x = torch.randn(25, 128)
        logits, attn = model(x)

        self.assertEqual(tuple(logits.shape), (3,))
        self.assertEqual(tuple(attn.shape), (25,))
        self.assertAlmostEqual(float(attn.sum().item()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
