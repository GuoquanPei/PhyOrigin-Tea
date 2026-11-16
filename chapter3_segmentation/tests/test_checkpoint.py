"""Checkpoint format contracts, independent of torch and GPUs."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sam3_checkpoint", ROOT / "checkpoint.py")
checkpoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checkpoint)


class CheckpointTests(unittest.TestCase):
    def test_original_detector_and_trainer_state_normalize_to_same_image_weights(self):
        expected = {"backbone.vision_backbone.trunk.pos_embed": "visual",
                    "transformer.decoder.query_embed.weight": "queries",
                    "segmentation_head.mask_embed.weight": "masks"}
        original = {"detector." + key: value for key, value in expected.items()}
        original["tracker.sam_mask_decoder.weight"] = "unused tracker"
        original["detector.backbone.vision_backbone.sam2_convs.0.weight"] = "unused interactive neck"
        self.assertEqual(checkpoint.normalize_image_state(original), expected)
        self.assertEqual(checkpoint.normalize_image_state({"model": expected, "epoch": 9}), expected)
        self.assertEqual(checkpoint.normalize_image_state({"model": original}), expected)

    def test_unexpected_image_keys_are_retained_for_strict_loading_to_reject(self):
        original = {"detector.backbone.weight": "known", "detector.typo.weight": "unknown"}
        self.assertIn("typo.weight", checkpoint.normalize_image_state(original))
        trainer = {"backbone.weight": "known", "typo.weight": "unknown"}
        self.assertIn("typo.weight", checkpoint.normalize_image_state({"model": trainer}))

    def test_empty_unrecognized_and_invalid_model_payloads_fail(self):
        for value in ({}, {"epoch": 3}, {"model": {}}, {"model": None},
                      {"tracker.weight": "unused"},
                      {"detector.backbone.vision_backbone.sam2_convs.0.weight": "unused"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                checkpoint.normalize_image_state(value)


if __name__ == "__main__":
    unittest.main()
