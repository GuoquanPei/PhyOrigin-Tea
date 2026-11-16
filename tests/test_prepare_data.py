"""Behavioral checks for conversion, image orientation, and split leakage."""

import csv
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import unittest
import uuid

from PIL import Image

from chapter3_segmentation.prepare_data import prepare_dataset, split_samples

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "tmp"


@contextmanager
def workspace_temp():
    # Python 3.13's Windows mkdir(0700) makes tempfile directories inaccessible
    # to the restricted worker token; normal inherited workspace ACLs work.
    path = TEST_TEMP_ROOT / f"prepare-test-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        assert path.resolve().parent == TEST_TEMP_ROOT.resolve()
        shutil.rmtree(path)


class PrepareDataTests(unittest.TestCase):
    def make_pair(self, root, name="sample", orientation=1, labels=("bud", "stem")):
        folder = root / "camera" / "siteA" / "RGB" / "Q1"
        folder.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (16, 8), (240, 10, 10))
        for y in range(8):
            for x in range(8, 16):
                image.putpixel((x, y), (10, 10, 240))
        exif = Image.Exif()
        exif[274] = orientation
        path = folder / f"{name}.JPG"
        image.save(path, quality=100, subsampling=0, exif=exif)
        width, height = (8, 16) if orientation in (5, 6, 7, 8) else (16, 8)
        shapes = [
            {"label": label, "shape_type": "polygon", "points": [[1, 1], [6, 1], [6, 6], [1, 6]]}
            for label in labels
        ]
        shapes.extend([
            {"label": "leaf1_midrib", "shape_type": "linestrip", "points": [[2, 2], [5, 5]]},
            {"label": "1cm", "shape_type": "linestrip", "points": [[1, 7], [5, 7]]},
        ])
        data = {"version": "4.2.2", "shapes": shapes, "imageWidth": width,
                "imageHeight": height, "imagePath": path.name, "imageData": None}
        path.with_suffix(".json").write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_exif_rotation_matches_annotations_and_preserves_geometry(self):
        # Removing EXIF normalization would rotate the training image away from its masks.
        with workspace_temp() as temp:
            base = Path(temp)
            image = self.make_pair(base / "raw", orientation=6)
            before = hashlib.sha256(image.read_bytes()).hexdigest()
            stats = prepare_dataset(base / "raw", base / "out", seed=42)
            coco = json.loads((base / "out/annotations/train.json").read_text())
            self.assertEqual(len(coco["images"]), 1)
            with Image.open(base / "out/images" / coco["images"][0]["file_name"]) as oriented:
                self.assertEqual(oriented.size, (8, 16))
                self.assertGreater(oriented.getpixel((4, 2))[0], 200)
                self.assertGreater(oriented.getpixel((4, 13))[2], 200)
                self.assertNotIn(oriented.getexif().get(274, 1), (6, 8))
            self.assertEqual([a["category_id"] for a in coco["annotations"]], [0, 3])
            self.assertEqual(coco["annotations"][0]["bbox"], [1, 1, 5, 5])
            self.assertEqual(coco["annotations"][0]["area"], 25)
            with (base / "out/manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))
            geometry = json.loads((base / "out" / row["geometry_path"]).read_text())
            self.assertEqual([s["label"] for s in geometry["lines"]], ["leaf1_midrib", "1cm"])
            self.assertEqual(geometry["polygons"][0]["points"], [[1, 1], [6, 1], [6, 6], [1, 6]])
            self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(), before)
            self.assertEqual(stats["achieved_counts"], {"train": 1, "val": 0, "test": 0})

    def test_absent_organ_stays_absent_and_class_counts_are_honest(self):
        # Fabricating missing buds or treating line labels as masks would fail this check.
        with workspace_temp() as temp:
            base = Path(temp)
            self.make_pair(base / "raw", labels=("leaf1",))
            stats = prepare_dataset(base / "raw", base / "out")
            coco = json.loads((base / "out/annotations/train.json").read_text())
            self.assertEqual([a["category_id"] for a in coco["annotations"]], [1])
            self.assertEqual(stats["class_counts"]["train"]["bud"], {"images": 0, "instances": 0})
            self.assertEqual(stats["class_counts"]["train"]["leaf1"], {"images": 1, "instances": 1})

    def records(self):
        rows = []
        for site, count in enumerate([80, 60, 60, 100, 100, 100, 100, 100, 100, 100, 99]):
            for index in range(count):
                key = f"site{site:02d}-{index:03d}"
                rows.append({"sample_id": key, "pixel_sha256": key, "device": "device",
                             "site": f"site{site:02d}", "group_id": ""})
        return rows

    def test_split_has_exact_799_100_100_and_is_order_independent(self):
        # Rounding each site independently or using discovery order would fail this check.
        rows = self.records()
        split = split_samples(rows, seed=42)
        self.assertEqual({s: sum(v == s for v in split.values()) for s in ("train", "val", "test")},
                         {"train": 799, "val": 100, "test": 100})
        self.assertEqual(split, split_samples(list(reversed(rows)), seed=42))
        self.assertNotEqual(split, split_samples(rows, seed=43))
        for site in {row["site"] for row in rows}:
            selected = [row for row in rows if row["site"] == site]
            self.assertGreaterEqual(sum(split[row["sample_id"]] == "val" for row in selected), 6)
            self.assertGreaterEqual(sum(split[row["sample_id"]] == "test" for row in selected), 6)

    def test_pixel_duplicates_and_explicit_biological_groups_never_leak(self):
        # Splitting biological identities or identical pixels independently would fail this check.
        rows = self.records()[:20]
        rows[1]["pixel_sha256"] = rows[0]["pixel_sha256"]
        rows[1]["group_id"] = "same-shoot"
        rows[2]["group_id"] = "same-shoot"
        split = split_samples(rows, seed=42)
        self.assertEqual(len({split[rows[i]["sample_id"]] for i in (0, 1, 2)}), 1)
        self.assertEqual({s: sum(v == s for v in split.values()) for s in ("train", "val", "test")},
                         {"train": 16, "val": 2, "test": 2})

    def test_output_inside_original_dataset_is_rejected(self):
        with workspace_temp() as temp:
            root = Path(temp) / "raw"
            self.make_pair(root)
            with self.assertRaisesRegex(ValueError, "output"):
                prepare_dataset(root, root / "prepared")
            self.assertFalse((root / "prepared").exists())

    def test_groups_csv_is_exported_and_unknown_image_mapping_is_rejected(self):
        with workspace_temp() as base:
            path = self.make_pair(base / "raw")
            mapping = base / "groups.csv"
            relative = path.relative_to(base / "raw").as_posix()
            mapping.write_text(f"relative_path,group_id\n{relative},shoot-001\n", encoding="utf-8")
            stats = prepare_dataset(base / "raw", base / "out", groups_csv=mapping)
            self.assertEqual(stats["biological_identity_supplied_for_images"], 1)
            with (base / "out/manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(next(csv.DictReader(stream))["biological_group_id"], "shoot-001")
            mapping.write_text("relative_path,group_id\nabsent.JPG,shoot-001\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "absent from dataset"):
                prepare_dataset(base / "raw", base / "bad-output", groups_csv=mapping)
            self.assertFalse((base / "bad-output").exists())


if __name__ == "__main__":
    unittest.main()
