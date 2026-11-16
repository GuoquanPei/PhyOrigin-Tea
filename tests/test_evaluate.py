import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from chapter3_segmentation.evaluate import evaluate_predictions


class EvaluationTests(unittest.TestCase):
    def test_segmentation_area_uses_mask_instead_of_model_box(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            truth = np.zeros((200, 200), dtype=np.uint8)
            truth[:10, :10] = 1
            false = np.zeros_like(truth)
            false[20:30, 20:30] = 1
            def encode(mask):
                rle = mask_utils.encode(np.asfortranarray(mask))
                rle['counts'] = rle['counts'].decode('ascii')
                return rle
            gt = {'images': [{'id': 1, 'width': 200, 'height': 200, 'file_name': 'a.jpg'}],
                  'categories': [{'id': 0, 'name': 'tea bud'}],
                  'annotations': [{'id': 1, 'image_id': 1, 'category_id': 0, 'segmentation': encode(truth),
                                   'area': 100, 'bbox': [0, 0, 10, 10], 'iscrowd': 0}]}
            source = root / 'gt.json'
            source.write_text(json.dumps(gt))
            predictions = [dict(image_id=1, category_id=0, segmentation=encode(false), bbox=[0, 0, 200, 200], score=.99),
                           dict(image_id=1, category_id=0, segmentation=encode(truth), bbox=[0, 0, 10, 10], score=.9)]
            result = evaluate_predictions(source, predictions)
            self.assertAlmostEqual(result['segm']['AP_small'], .5)

    def test_perfect_masks_and_missing_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mask = np.zeros((20, 20), dtype=np.uint8)
            mask[2:12, 3:13] = 1
            rle = mask_utils.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('ascii')
            gt = {'images': [{'id': 1, 'width': 20, 'height': 20, 'file_name': 'a.jpg'}],
                  'categories': [{'id': 0, 'name': 'tea bud'}],
                  'annotations': [{'id': 1, 'image_id': 1, 'category_id': 0,
                                   'segmentation': rle, 'area': 100, 'bbox': [3, 2, 10, 10], 'iscrowd': 0}]}
            source = root / 'gt.json'
            source.write_text(json.dumps(gt))
            prediction = [{'image_id': 1, 'category_id': 0, 'segmentation': rle, 'score': .9}]
            result = evaluate_predictions(source, prediction, .5)
            self.assertAlmostEqual(result['segm']['AP50'], 1)
            self.assertAlmostEqual(result['semantic_union']['tea bud']['dice'], 1)
            missing = evaluate_predictions(source, [], .5)
            self.assertEqual(missing['segm']['AP50'], 0)
            self.assertEqual(missing['semantic_union']['tea bud']['dice'], 0)

    def test_unknown_ids_fail_instead_of_silent_ignore(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'gt.json'
            source.write_text(json.dumps({'images': [{'id': 1}], 'categories': [{'id': 0}], 'annotations': []}))
            with self.assertRaises(ValueError):
                evaluate_predictions(source, [{'image_id': 99, 'category_id': 0}], .5)


if __name__ == '__main__':
    unittest.main()
