"""Automatic text-prompt SAM3 predictions and COCO instance segmentation evaluation."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def evaluate_predictions(annotation_file, predictions, confidence=0.5):
    """COCO AP uses all scores; semantic union IoU/Dice use a fixed confidence."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(annotation_file))
    for pred in predictions:
        if pred["image_id"] not in gt.imgs or pred["category_id"] not in gt.cats:
            raise ValueError("Prediction references an unknown image/category ID")
    result = {"confidence_for_semantic_union": confidence, "images": len(gt.imgs)}
    for kind in ("segm", "bbox"):
        # COCO loadRes derives area from bbox when both are supplied. Mask AP must
        # use mask area, including for unmatched detections and small/medium AP.
        records = deepcopy(predictions)
        if kind == "segm":
            for record in records:
                record.pop("bbox", None)
        with contextlib.redirect_stdout(io.StringIO()):
            if records:
                detected = gt.loadRes(records)
            else:
                # loadRes([]) raises IndexError; an empty result is valid.
                detected = COCO()
                detected.dataset = {"images": list(gt.imgs.values()), "categories": list(gt.cats.values()),
                                    "annotations": []}
                detected.createIndex()
        metric = COCOeval(gt, detected, kind)
        with contextlib.redirect_stdout(io.StringIO()):
            metric.evaluate()
            metric.accumulate()
            metric.summarize()
        names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                 "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
        result[kind] = {name: float(val) if val >= 0 else None for name, val in zip(names, metric.stats)}
        result[kind]["per_category_AP"] = {}
        for index, cat in enumerate(metric.params.catIds):
            values = metric.eval["precision"][:, :, index, 0, -1]
            valid = values[values >= 0]
            result[kind]["per_category_AP"][gt.cats[cat]["name"]] = float(valid.mean()) if valid.size else None
    sums = {cat: np.zeros(3, dtype=np.int64) for cat in gt.cats}
    by_image_cat = {}
    for pred in predictions:
        if float(pred["score"]) >= confidence:
            by_image_cat.setdefault((pred["image_id"], pred["category_id"]), []).append(pred)
    for image_id, image in gt.imgs.items():
        for cat in gt.cats:
            truth = np.zeros((image["height"], image["width"]), dtype=bool)
            predicted = np.zeros_like(truth)
            for ann in gt.loadAnns(gt.getAnnIds(imgIds=[image_id], catIds=[cat])):
                truth |= gt.annToMask(ann).astype(bool)
            for ann in by_image_cat.get((image_id, cat), []):
                decoded = gt.annToMask(ann).astype(bool)
                if decoded.shape != truth.shape:
                    raise ValueError("Predicted mask dimensions do not match the image")
                predicted |= decoded
            sums[cat] += (np.count_nonzero(truth & predicted), np.count_nonzero(truth),
                          np.count_nonzero(predicted))
    result["semantic_union"] = {}
    for cat, (intersection, truth_area, prediction_area) in sums.items():
        union = int(truth_area + prediction_area - intersection)
        denominator = int(truth_area + prediction_area)
        result["semantic_union"][gt.cats[cat]["name"]] = {
            "iou": float(intersection / union) if union else None,
            "dice": float(2 * intersection / denominator) if denominator else None,
            "gt_pixels": int(truth_area), "pred_pixels": int(prediction_area)}
    return result


def load_model(checkpoint, sam3_root, device):
    """Strictly load either official detector.* weights or a Trainer model state."""
    if sam3_root:
        sys.path.insert(0, str(Path(sam3_root).resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from chapter3_segmentation.checkpoint import load_image_state
    from sam3.model_builder import build_sam3_image_model

    state = load_image_state(checkpoint)
    model = build_sam3_image_model(device="cpu", load_from_HF=False, checkpoint_path=None,
                                  enable_segmentation=True, enable_inst_interactivity=False, eval_mode=True)
    # The upstream checkpoint helper only extracts detector.* and is unsafe for Trainer state dicts.
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def predict(data_dir, split, checkpoint, sam3_root=None, confidence=0.05, device="cuda"):
    import torch
    from pycocotools import mask as mask_utils

    model = load_model(checkpoint, sam3_root, device)
    from sam3.model.sam3_image_processor import Sam3Processor

    processor = Sam3Processor(model, device=device, confidence_threshold=confidence)
    source = read_json(data_dir / "annotations" / f"{split}.json")
    predictions = []
    with torch.inference_mode(), torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                                enabled=device.startswith("cuda")):
        for index, record in enumerate(source["images"], 1):
            with Image.open(data_dir / "images" / record["file_name"]) as image:
                image = image.convert("RGB")
                if image.size != (record["width"], record["height"]):
                    raise ValueError("Canonical image and COCO dimensions differ; run prepare_data.py")
                state = processor.set_image(image)
            for cat in source["categories"]:
                processor.reset_all_prompts(state)
                output = processor.set_text_prompt(state=state, prompt=cat["name"])
                for mask, box, score in zip(output["masks"], output["boxes"], output["scores"]):
                    binary = mask.squeeze(0).detach().cpu().numpy().astype(np.uint8)
                    if not binary.any():
                        continue
                    rle = mask_utils.encode(np.asfortranarray(binary))
                    rle["counts"] = rle["counts"].decode("ascii")
                    x1, y1, x2, y2 = box.detach().float().cpu().tolist()
                    width, height = record["width"], record["height"]
                    x1, x2 = np.clip([x1, x2], 0, width)
                    y1, y2 = np.clip([y1, y2], 0, height)
                    predictions.append({"image_id": record["id"], "category_id": cat["id"],
                                        "segmentation": rle, "score": float(score),
                                        "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)]})
            print(f"Predicted {index}/{len(source['images'])}: {record['file_name']}", flush=True)
    return predictions, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sam3-root", type=Path)
    parser.add_argument("--predictions", type=Path, help="Evaluate an existing COCO prediction list without loading SAM3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-threshold", type=float, default=0.05)
    parser.add_argument("--metric-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.prediction_threshold <= 1 or not 0 <= args.metric_threshold <= 1:
        parser.error("Thresholds must lie in [0, 1]")
    metadata = {"split": args.split, "metric_threshold": args.metric_threshold,
                "annotation_file": str((args.data_dir / "annotations" / f"{args.split}.json").resolve())}
    if args.predictions:
        predictions = read_json(args.predictions)
        metadata["predictions_source"] = str(args.predictions.resolve())
    else:
        if not args.checkpoint:
            parser.error("Supply --checkpoint or --predictions")
        predictions, source = predict(args.data_dir, args.split, args.checkpoint, args.sam3_root,
                                      args.prediction_threshold, args.device)
        with args.checkpoint.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(8*1024*1024), b""):
                digest.update(block)
        metadata.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest.hexdigest(),
                        prompts={cat["id"]: cat["name"] for cat in source["categories"]},
                        prediction_threshold=args.prediction_threshold, ground_truth_prompts=False)
    write_json(args.output_dir / "predictions.json", predictions)
    metrics = evaluate_predictions(args.data_dir / "annotations" / f"{args.split}.json",
                                   predictions, args.metric_threshold)
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "evaluation.json", metadata)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
