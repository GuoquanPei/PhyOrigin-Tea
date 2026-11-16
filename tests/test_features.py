import unittest

import numpy as np
from PIL import Image

from chapter3_segmentation.extract_features import masked_crops


class CropTests(unittest.TestCase):
    def test_long_organ_keeps_both_tips(self):
        image = Image.new('RGB', (200, 20), (250, 0, 0))
        crop = masked_crops(image, np.ones((20, 200), dtype=bool), (1,))[0]
        self.assertEqual(crop.size, (200, 200))
        self.assertEqual(crop.getpixel((0, 100)), (250, 0, 0))
        self.assertEqual(crop.getpixel((199, 100)), (250, 0, 0))
        self.assertEqual(crop.getpixel((100, 0)), (127, 127, 127))

    def test_edge_crop_keeps_scale_and_masks_background(self):
        image = Image.new('RGB', (12, 12), (250, 0, 0))
        mask = np.zeros((12, 12), dtype=bool)
        mask[0:4, 0:4] = True
        crops = masked_crops(image, mask, (1, 2))
        self.assertEqual(crops[0].size, (4, 4))
        self.assertEqual(crops[1].size, (8, 8))
        self.assertEqual(crops[1].getpixel((0, 0)), (127, 127, 127))
        self.assertEqual(crops[1].getpixel((3, 3)), (250, 0, 0))


if __name__ == '__main__':
    unittest.main()
