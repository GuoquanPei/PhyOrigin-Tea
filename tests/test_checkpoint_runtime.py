"""Exercise restricted checkpoint loading with the deployment PyTorch API."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from chapter3_segmentation.checkpoint import load_image_state


class CheckpointRuntimeTests(unittest.TestCase):
    def test_constructor_scope_preserves_allowlist_across_nested_image_loading(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch runtime test runs in Linux CI with torch2.7.1')
        if not hasattr(torch.serialization, 'safe_globals'):
            self.skipTest('Deployment requires torch>=2.7 with safe_globals')
        from chapter3_segmentation.checkpoint import numpy_metadata_context

        before = list(torch.serialization.get_safe_globals())
        state = {'backbone.weight': torch.ones(2)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'resume.pt'
            torch.save({'model': state, 'best_meter_values': {'AP': np.float64(.5)}}, path)
            # Trainer.__init__ first builds the model (nested helper context), then
            # loads its resume checkpoint with torch.load under the factory scope.
            with numpy_metadata_context():
                loaded = load_image_state(path)
                torch.testing.assert_close(loaded['backbone.weight'], state['backbone.weight'])
                # Match the exact upstream resume call; torch>=2.7 defaults to
                # weights_only=True, so the scoped allowlist must still be active.
                resumed = torch.load(path, map_location='cpu')
                self.assertEqual(resumed['best_meter_values']['AP'], .5)
            self.assertCountEqual(torch.serialization.get_safe_globals(), before)
            with self.assertRaises(RuntimeError):
                with numpy_metadata_context():
                    raise RuntimeError('constructor error')
            self.assertCountEqual(torch.serialization.get_safe_globals(), before)

    def test_original_and_trainer_numpy_metadata_round_trip(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch runtime test runs in Linux CI with torch2.7.1')
        if not hasattr(torch.serialization, 'safe_globals'):
            self.skipTest('Deployment requires torch>=2.7 with safe_globals')
        state = {'backbone.weight': torch.arange(6).reshape(2, 3).float()}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = {'detector.backbone.weight': state['backbone.weight'],
                        'tracker.weight': torch.ones(1),
                        'detector.backbone.vision_backbone.sam2_convs.0.weight': torch.ones(1)}
            trained = {'model': state, 'best_meter_values': {'AP': np.float64(.5)}}
            for name, checkpoint in [('original', original), ('trained', trained)]:
                path = root / (name + '.pt')
                torch.save(checkpoint, path)
                loaded = load_image_state(path)
                self.assertEqual(set(loaded), set(state))
                torch.testing.assert_close(loaded['backbone.weight'], state['backbone.weight'])


if __name__ == '__main__':
    unittest.main()
