"""Convert LabelMe tea organs to COCO without changing the source dataset.

The default split is an image-level, device/site-stratified 8:1:1 partition.
Decoded-pixel duplicates always stay together. Biological identities are only
known when supplied through --groups-csv (relative_path,group_id).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

from PIL import Image, ImageOps


SPLITS = ("train", "val", "test")
RATIOS = (0.8, 0.1, 0.1)
CATEGORIES = [
    {"id": 0, "name": "tea bud", "original_label": "bud", "supercategory": "tea organ"},
    {"id": 1, "name": "first tea leaf", "original_label": "leaf1", "supercategory": "tea organ"},
    {"id": 2, "name": "second tea leaf", "original_label": "leaf2", "supercategory": "tea organ"},
    {"id": 3, "name": "tea stem", "original_label": "stem", "supercategory": "tea organ"},
]
LABEL_IDS = {category["original_label"]: category["id"] for category in CATEGORIES}
MANIFEST_FIELDS = (
    "image_id", "sample_id", "source", "annotation_source", "relative_path", "image_path",
    "device", "site", "subset", "split", "width", "height", "pixel_sha256",
    "exif_orientation", "group_id", "biological_group_id", "geometry_path",
)


def _counts(total: int) -> dict[str, int]:
    quotas = [total * ratio for ratio in RATIOS]
    counts = [math.floor(value) for value in quotas]
    order = sorted(range(3), key=lambda index: (-(quotas[index] - counts[index]), index))
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return dict(zip(SPLITS, counts))


def _stratum_targets(samples: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, int]]:
    sizes = Counter((sample["device"], sample["site"]) for sample in samples)
    targets = {
        key: {split: math.floor(size * ratio) for split, ratio in zip(SPLITS, RATIOS)}
        for key, size in sizes.items()
    }
    global_target = _counts(len(samples))
    deficits = {split: global_target[split] - sum(row[split] for row in targets.values()) for split in SPLITS}
    # Largest fractional remainders subject to both row totals and global totals.
    while any(deficits.values()):
        candidates = []
        for key in sorted(sizes):
            if sum(targets[key].values()) == sizes[key]:
                continue
            for index, split in enumerate(SPLITS):
                if deficits[split] > 0:
                    remainder = sizes[key] * RATIOS[index] - targets[key][split]
                    candidates.append((remainder, deficits[split], -index, key, split))
        _, _, _, key, split = max(candidates)
        targets[key][split] += 1
        deficits[split] -= 1
    return targets


def _components(samples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Union explicit biological groups and exact oriented RGB pixel duplicates."""
    parent = list(range(len(samples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for index, sample in enumerate(samples):
        keys = [("pixel", sample["pixel_sha256"])]
        if sample.get("biological_group_id") or sample.get("group_id"):
            keys.append(("biological", sample.get("biological_group_id") or sample["group_id"]))
        for key in keys:
            if key in seen:
                parent[find(index)] = find(seen[key])
            else:
                seen[key] = index
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[find(index)].append(sample)
    return list(groups.values())


def split_samples(samples: list[dict[str, Any]], seed: int = 42) -> dict[str, str]:
    """Return sample_id -> split, independent of input order and without group leaks.

    With singleton groups the global largest-remainder targets are exact. Large
    indivisible biological/duplicate groups can make those targets impossible;
    keeping each group intact takes precedence over the requested ratio.
    """
    if not samples:
        return {}
    samples = sorted(samples, key=lambda sample: sample["sample_id"])
    if len({sample["sample_id"] for sample in samples}) != len(samples):
        raise ValueError("sample_id must be unique")
    groups = _components(samples)
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    target = _counts(len(samples))
    stratum_target = _stratum_targets(samples)
    used = Counter()
    stratum_used: dict[tuple[str, str], Counter] = defaultdict(Counter)
    result = {}
    for group in groups:
        composition = Counter((sample["device"], sample["site"]) for sample in group)
        fits = [split for split in SPLITS if used[split] + len(group) <= target[split]]
        candidates = fits or list(SPLITS)

        def score(split: str) -> tuple[float, float, int]:
            local_need = sum(
                count * (stratum_target[key][split] - stratum_used[key][split])
                / max(stratum_target[key][split], 1)
                for key, count in composition.items()
            )
            global_need = (target[split] - used[split]) / max(target[split], 1)
            return local_need, global_need, -SPLITS.index(split)

        chosen = max(candidates, key=score)
        for sample in group:
            result[sample["sample_id"]] = chosen
            stratum_used[(sample["device"], sample["site"])][chosen] += 1
        used[chosen] += len(group)
    return result


def _read_groups(groups_csv: Path | None) -> dict[str, str]:
    if groups_csv is None:
        return {}
    with Path(groups_csv).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"relative_path", "group_id"}.issubset(reader.fieldnames or []):
            raise ValueError("groups CSV requires relative_path and group_id columns")
        groups = {}
        for row in reader:
            path = row["relative_path"].replace("\\", "/")
            identity = row["group_id"].strip()
            if not identity:
                raise ValueError(f"empty biological group_id for {path}")
            if path in groups and groups[path] != identity:
                raise ValueError(f"conflicting biological group_id for {path}")
            groups[path] = identity
    return groups


def _polygon_annotation(shape: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    points = shape.get("points", [])
    if len(points) < 3:
        raise ValueError(f"polygon {shape['label']} has fewer than three points")
    for x, y in points:
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= width and 0 <= y <= height):
            raise ValueError(f"polygon {shape['label']} has coordinates outside its canonical image")
    xs, ys = zip(*points)
    area = abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )) / 2
    if area <= 0:
        raise ValueError(f"polygon {shape['label']} has zero area")
    return {"category_id": LABEL_IDS[shape["label"]],
            "segmentation": [[coordinate for point in points for coordinate in point]],
            "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
            "area": area, "iscrowd": 0, "original_label": shape["label"]}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_dataset(
    data_root: Path | str,
    output: Path | str,
    seed: int = 42,
    groups_csv: Path | str | None = None,
) -> dict[str, Any]:
    """Write EXIF-normalized JPEGs, four-class COCO, geometry, manifest and stats."""
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if output == data_root or data_root in output.parents or output in data_root.parents:
        raise ValueError("output must be separate from the original dataset directory")
    if not data_root.is_dir():
        raise ValueError(f"data root does not exist: {data_root}")
    groups = _read_groups(Path(groups_csv) if groups_csv else None)
    image_files = sorted(
        (path for path in data_root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda path: path.relative_to(data_root).as_posix(),
    )
    if not image_files:
        raise ValueError(f"no supported images found in {data_root}")
    missing = [str(path) for path in image_files if not path.with_suffix(".json").is_file()]
    orphan_json = [str(path) for path in data_root.rglob("*.json")
                   if not any(path.with_suffix(suffix).is_file() for suffix in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"))]
    if missing or orphan_json:
        raise ValueError(f"unpaired images/annotations: {missing[:3]} / {orphan_json[:3]}")
    relatives = {path.relative_to(data_root).as_posix() for path in image_files}
    unknown_groups = sorted(set(groups) - relatives)
    if unknown_groups:
        raise ValueError(f"groups CSV paths absent from dataset: {unknown_groups[:3]}")
    for folder in ("images", "annotations", "metadata"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    samples = []
    seen_ids = set()
    for image_id, path in enumerate(image_files, 1):
        relative = path.relative_to(data_root)
        if len(relative.parts) < 4:
            raise ValueError(f"expected device/site/.../image layout: {relative}")
        label_path = path.with_suffix(".json")
        labelme = json.loads(label_path.read_text(encoding="utf-8-sig"))
        sample_id = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:20]
        if sample_id in seen_ids:
            raise ValueError("sample ID collision; no image was intentionally replaced")
        seen_ids.add(sample_id)
        with Image.open(path) as original:
            orientation = original.getexif().get(274, 1)
            canonical = ImageOps.exif_transpose(original).convert("RGB")
            width, height = canonical.size
            declared = (labelme.get("imageWidth"), labelme.get("imageHeight"))
            if declared != canonical.size:
                raise ValueError(f"annotation size {declared} != EXIF-oriented image {canonical.size}: {relative}")
            pixel_hasher = hashlib.sha256(f"{width}x{height}:RGB:".encode("ascii"))
            pixel_hasher.update(canonical.tobytes())
            pixel_hash = pixel_hasher.hexdigest()
            canonical.save(output / "images" / f"{sample_id}.jpg", quality=95, subsampling=0)
        polygons, lines, other = [], [], []
        annotations = []
        for shape in labelme.get("shapes", []):
            kind = shape.get("shape_type") or "polygon"
            if kind == "polygon":
                polygons.append(shape)
                if shape.get("label") in LABEL_IDS:
                    annotations.append(_polygon_annotation(shape, width, height))
                else:
                    raise ValueError(f"unknown polygon class {shape.get('label')}: {relative}")
            elif kind in {"linestrip", "line"}:
                lines.append(shape)
            else:
                other.append(shape)
        samples.append({
            "image_id": image_id, "sample_id": sample_id, "source": str(path),
            "annotation_source": str(label_path), "relative_path": relative.as_posix(),
            "image_path": f"images/{sample_id}.jpg", "device": relative.parts[0],
            "site": relative.parts[1], "subset": relative.parts[-2],
            "width": width, "height": height, "pixel_sha256": pixel_hash,
            "exif_orientation": orientation, "group_id": "",
            "biological_group_id": groups.get(relative.as_posix(), ""),
            "geometry_path": f"metadata/{sample_id}.json", "_annotations": annotations,
            "_geometry": {"polygons": polygons, "lines": lines, "other_shapes": other,
                          "labelme_metadata": {key: value for key, value in labelme.items()
                                               if key not in {"imageData", "shapes"}}},
        })
    split = split_samples(samples, seed)
    components = _components(samples)
    for component in components:
        identity = hashlib.sha256("\n".join(sorted(row["sample_id"] for row in component)).encode()).hexdigest()[:20]
        for sample in component:
            sample["group_id"] = f"group-{identity}"
            sample["split"] = split[sample["sample_id"]]
    documents = {name: {"info": {"description": "EXIF-normalized tea organ segmentation", "seed": seed},
                         "images": [], "annotations": [], "categories": CATEGORIES} for name in SPLITS}
    class_counts = {name: {label: {"images": 0, "instances": 0} for label in LABEL_IDS} for name in SPLITS}
    strata = defaultdict(Counter)
    missing_organs = Counter()
    annotation_id = 0
    for sample in samples:
        document = documents[sample["split"]]
        document["images"].append({"id": sample["image_id"], "file_name": f"{sample['sample_id']}.jpg",
                                   "width": sample["width"], "height": sample["height"],
                                   "sample_id": sample["sample_id"], "group_id": sample["group_id"]})
        present = set()
        for annotation in sample["_annotations"]:
            annotation_id += 1
            annotation = dict(annotation, id=annotation_id, image_id=sample["image_id"])
            document["annotations"].append(annotation)
            label = annotation["original_label"]
            class_counts[sample["split"]][label]["instances"] += 1
            present.add(label)
        for label in present:
            class_counts[sample["split"]][label]["images"] += 1
        missing_organs.update(set(LABEL_IDS) - present)
        strata[(sample["device"], sample["site"])][sample["split"]] += 1
        geometry = {key: sample[key] for key in MANIFEST_FIELDS}
        geometry.update(sample["_geometry"])
        _write_json(output / sample["geometry_path"], geometry)
    for name in SPLITS:
        _write_json(output / "annotations" / f"{name}.json", documents[name])
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)
    target_strata = _stratum_targets(samples)
    counts = {name: len(documents[name]["images"]) for name in SPLITS}
    stats = {
        "data_root": str(data_root), "output": str(output), "seed": seed, "ratios": dict(zip(SPLITS, RATIOS)),
        "total_images": len(samples), "requested_counts": _counts(len(samples)), "achieved_counts": counts,
        "categories": CATEGORIES, "class_counts": class_counts,
        "strata": [{"device": device, "site": site, "total": sum(strata[(device, site)].values()),
                    "requested_counts": target_strata[(device, site)],
                    "achieved_counts": {name: strata[(device, site)][name] for name in SPLITS}}
                   for device, site in sorted(strata)],
        "split_unit": "biological groups where supplied; otherwise individual images; exact pixel duplicates unioned",
        "biological_identity_supplied_for_images": len(groups),
        "biological_independence_established": len(groups) == len(samples),
        "group_count": len(components), "largest_group": max(map(len, components)),
        "duplicate_pixel_groups": sum(count > 1 for count in Counter(row["pixel_sha256"] for row in samples).values()),
        "exif_orientations": dict(Counter(row["exif_orientation"] for row in samples)),
        "missing_polygon_classes": dict(missing_organs),
        "preserved_line_shapes": sum(len(row["_geometry"]["lines"]) for row in samples),
        "notes": ["Line annotations are geometry/calibration metadata, not segmentation masks.",
                  "The default split does not establish independence between biological specimens.",
                  "Device and site are confounded in the supplied acquisition layout.",
                  "Indivisible groups take priority over exact split totals."],
    }
    _write_json(output / "split_stats.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"F:\我的数据集\硕士论文数据\data"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--groups-csv", type=Path, help="Optional CSV with relative_path,group_id columns")
    args = parser.parse_args()
    stats = prepare_dataset(args.data_root, args.output, args.seed, args.groups_csv)
    print(json.dumps({"output": stats["output"], "images": stats["total_images"],
                      "splits": stats["achieved_counts"], "groups": stats["group_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
