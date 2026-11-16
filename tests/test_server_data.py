import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from chapter3_segmentation.export_data import export_archive
from chapter3_segmentation.make_smoke_data import make_smoke_data


class ServerDataTests(unittest.TestCase):
    def make_dataset(self, root):
        (root / 'images').mkdir(parents=True)
        (root / 'annotations').mkdir()
        for split, offset in [('train', 0), ('val', 100), ('test', 200)]:
            images, annotations = [], []
            for index in range(12):
                identity = index + offset
                name = f'{identity}.jpg'
                Image.new('RGB', (10, 10), (identity % 255, 0, 0)).save(root / 'images' / name)
                images.append(dict(id=identity, file_name=name, width=10, height=10))
                annotations.append(dict(id=identity, image_id=identity, category_id=0,
                                        segmentation=[[1, 1, 8, 1, 8, 8, 1, 8]], bbox=[1, 1, 7, 7], area=49, iscrowd=0))
            (root / 'annotations' / (split + '.json')).write_text(json.dumps(
                dict(images=images, annotations=annotations, categories=[dict(id=0, name='tea bud')])) )
        (root / 'manifest.csv').write_text('sample_id,split\na,train\n')
        (root / 'split_stats.json').write_text('{}')

    def test_smoke_subset_is_separate_and_has_enough_images_for_five_gpus(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_dataset(root / 'source')
            before = (root / 'source/annotations/train.json').read_bytes()
            summary = make_smoke_data(root / 'source', root / 'smoke', 5)
            self.assertEqual(summary['counts'], {'train': 10, 'val': 5})
            self.assertFalse((root / 'smoke/annotations/test.json').exists())
            self.assertEqual(before, (root / 'source/annotations/train.json').read_bytes())
            train = json.loads((root / 'smoke/annotations/train.json').read_text())
            self.assertEqual(len(train['annotations']), 10)
            self.assertTrue(all((root / 'smoke/images' / im['file_name']).exists() for im in train['images']))
            with self.assertRaises(FileExistsError):
                make_smoke_data(root / 'source', root / 'smoke', 5)

    def test_archive_is_portable_and_contains_only_prepared_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_dataset(root / 'source')
            (root / 'source/private-note.txt').write_text('not dataset data')
            summary = export_archive(root / 'source', root / 'dataset.tar.gz')
            self.assertEqual(len(summary['sha256']), 64)
            with tarfile.open(root / 'dataset.tar.gz') as archive:
                names = archive.getnames()
            self.assertIn('tea_data/annotations/train.json', names)
            self.assertIn('tea_data/images/0.jpg', names)
            self.assertNotIn('tea_data/private-note.txt', names)
            self.assertTrue(all(not name.startswith('/') and '..' not in Path(name).parts for name in names))
            with self.assertRaises(FileExistsError):
                export_archive(root / 'source', root / 'dataset.tar.gz')


if __name__ == '__main__':
    unittest.main()
