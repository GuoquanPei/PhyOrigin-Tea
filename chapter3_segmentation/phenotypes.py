"""Export mask geometry; manual midribs/scales are explicitly marked reference data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def mask_traits(mask, pixels_per_cm=None):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("Expected a nonempty two-dimensional mask")
    if pixels_per_cm is not None and (not np.isfinite(pixels_per_cm) or pixels_per_cm <= 0):
        raise ValueError("pixels_per_cm must be finite and positive")
    ys, xs = np.nonzero(mask)
    # Pixel-centre PCA axes; extents include a projected pixel's own width.
    points = np.column_stack((xs, ys)).astype(float)
    centred = points - points.mean(axis=0)
    covariance = centred.T @ centred / len(points)
    _, axes = np.linalg.eigh(covariance)
    major = axes[:, -1]
    minor = axes[:, 0]
    length = float(np.ptp(centred @ major) + np.abs(major).sum())
    width = float(np.ptp(centred @ minor) + np.abs(minor).sum())
    padded = np.pad(mask.astype(np.int8), 1)
    perimeter = int(np.abs(np.diff(padded, axis=0)).sum() + np.abs(np.diff(padded, axis=1)).sum())
    area = int(mask.sum())
    return {"area_px2": area, "perimeter_grid_px": perimeter,
            "centroid_x_px": float(xs.mean()), "centroid_y_px": float(ys.mean()),
            "axis_length_px": length, "axis_width_px": width,
            "axis_aspect_ratio": length / width,
            "grid_compactness": float(4*np.pi*area/perimeter**2),
            "bbox_width_px": int(xs.max()-xs.min()+1), "bbox_height_px": int(ys.max()-ys.min()+1),
            "pixels_per_cm": pixels_per_cm,
            "area_cm2": area / pixels_per_cm**2 if pixels_per_cm else None,
            "axis_length_cm": length / pixels_per_cm if pixels_per_cm else None,
            "axis_width_cm": width / pixels_per_cm if pixels_per_cm else None}


def polyline_traits(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Polyline requires at least two finite x/y points")
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    chord = float(np.linalg.norm(points[-1]-points[0]))
    return {"length_px": length, "chord_px": chord, "tortuosity": length/chord if chord else None}


def write_csv(path, rows):
    if not rows:
        raise ValueError("No masks passed the threshold; no phenotype records to export")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    from pycocotools.coco import COCO
    from pycocotools import mask as mask_utils

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--predictions", type=Path, help="COCO prediction JSON. Omit only with --reference-labels")
    parser.add_argument("--reference-labels", action="store_true", help="Export annotated-mask reference measurements")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--manual-geometry", action="store_true", help="Use manual 1cm scale and include manual midrib reference traits")
    args = parser.parse_args()
    if bool(args.predictions) == args.reference_labels:
        parser.error("Choose exactly one of --predictions and --reference-labels")
    if not 0 <= args.confidence <= 1:
        parser.error("Confidence must lie in [0, 1]")
    gt = COCO(str(args.data_dir / "annotations" / f"{args.split}.json"))
    with (args.data_dir / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest = {int(row["image_id"]): row for row in csv.DictReader(stream)}
    records = list(gt.anns.values()) if args.reference_labels else json.loads(args.predictions.read_text(encoding="utf-8"))
    rows = []
    for index, record in enumerate(records):
        if not args.reference_labels and record["score"] < args.confidence:
            continue
        image_id, cat = record["image_id"], record["category_id"]
        if image_id not in gt.imgs or cat not in gt.cats:
            raise ValueError("Mask does not belong to requested split/category")
        meta = manifest[image_id]
        geometry = json.loads((args.data_dir / meta["geometry_path"]).read_text(encoding="utf-8"))
        shapes = geometry.get("shapes", geometry.get("lines", []))
        scale = None
        if args.manual_geometry:
            scales = [polyline_traits(s["points"])["length_px"] for s in shapes if s["label"] == "1cm"]
            if len(scales) == 1 and scales[0] > 0:
                scale = scales[0]
        binary = gt.annToMask(record).astype(bool)
        info = gt.imgs[image_id]
        if binary.shape != (info["height"], info["width"]):
            raise ValueError("Mask dimensions differ from canonical image")
        if not binary.any():
            continue
        row = {key: meta.get(key, "") for key in ("image_id", "sample_id", "group_id", "site", "device", "subset", "split")}
        row.update(instance_id=record.get("id", index), category_id=cat,
                   organ=gt.cats[cat].get("original_label", gt.cats[cat]["name"]),
                   score=record.get("score"), mask_source="manual_reference" if args.reference_labels else "sam3_prediction",
                   scale_source="manual_1cm" if scale else "unavailable")
        row.update(mask_traits(binary, scale))
        if args.manual_geometry:
            label = row["organ"] + "_midrib"
            lines = [s for s in shapes if s["label"] == label]
            if len(lines) == 1:
                row.update({"manual_midrib_"+key: val for key, val in polyline_traits(lines[0]["points"]).items()})
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)
    print(f"Exported {len(rows)} mask measurements to {args.output}")


if __name__ == "__main__":
    main()
