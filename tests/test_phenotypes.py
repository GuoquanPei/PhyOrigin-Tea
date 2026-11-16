import unittest

import numpy as np

from chapter3_segmentation.phenotypes import mask_traits, polyline_traits


class PhenotypeTests(unittest.TestCase):
    def test_rectangle_area_centroid_and_calibration(self):
        mask = np.zeros((30, 40), dtype=bool)
        mask[5:15, 10:30] = True
        row = mask_traits(mask, pixels_per_cm=10)
        self.assertEqual(row['area_px2'], 200)
        self.assertEqual(row['perimeter_grid_px'], 60)
        self.assertAlmostEqual(row['centroid_x_px'], 19.5)
        self.assertAlmostEqual(row['centroid_y_px'], 9.5)
        self.assertEqual(row['area_cm2'], 2)
        self.assertGreater(row['axis_length_px'], row['axis_width_px'])

    def test_empty_mask_and_missing_scale(self):
        with self.assertRaises(ValueError):
            mask_traits(np.zeros((5, 5)))
        self.assertIsNone(mask_traits(np.ones((5, 5)))['area_cm2'])

    def test_midrib_polyline_is_explicit_geometry(self):
        row = polyline_traits([[0, 0], [3, 0], [3, 4]])
        self.assertEqual(row['length_px'], 7)
        self.assertEqual(row['chord_px'], 5)
        self.assertAlmostEqual(row['tortuosity'], 1.4)


if __name__ == '__main__':
    unittest.main()
