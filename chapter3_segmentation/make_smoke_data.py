"""Make a separate tiny COCO dataset for a real one-epoch GPU smoke run."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path


def make_smoke_data(data_dir, output_dir, num_gpus=5):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if num_gpus < 1:
        raise ValueError("num_gpus must be positive")
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise ValueError("Smoke data must be outside the prepared source dataset")
    if output_dir.exists():
        raise FileExistsError(f"Use a fresh smoke directory: {output_dir}")
    selected = {}
    for split, wanted in (("train", 2*num_gpus), ("val", num_gpus)):
        document = json.loads((data_dir / "annotations" / f"{split}.json").read_text(encoding="utf-8"))
        images = sorted(document["images"], key=lambda image: image["id"])
        if len(images) < num_gpus:
            raise ValueError(f"{split} needs at least {num_gpus} images for this smoke run")
        subset = copy.deepcopy(document)
        subset["images"] = images[:wanted]
        ids = {image["id"] for image in subset["images"]}
        subset["annotations"] = [ann for ann in document["annotations"] if ann["image_id"] in ids]
        for image in subset["images"]:
            filename = Path(image["file_name"])
            if filename.is_absolute() or ".." in filename.parts:
                raise ValueError("Smoke data requires safe, relative canonical image filenames")
            source = data_dir / "images" / filename
            if not source.is_file():
                raise FileNotFoundError(source)
        selected[split] = subset
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "annotations").mkdir()
    for split, document in selected.items():
        for image in document["images"]:
            source = data_dir / "images" / image["file_name"]
            destination = output_dir / "images" / image["file_name"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        (output_dir / "annotations" / f"{split}.json").write_text(
            json.dumps(document, ensure_ascii=False), encoding="utf-8")
    summary = {"purpose": "smoke_only_not_a_reported_experiment", "num_gpus": num_gpus,
               "source_data": str(data_dir), "counts": {name: len(doc["images"]) for name, doc in selected.items()}}
    (output_dir / "smoke.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(make_smoke_data(args.data_dir, args.output_dir, args.num_gpus), indent=2))


if __name__ == "__main__":
    main()
