"""Frozen ImageNet ResNet50 baseline features from mask-guided multiscale crops."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def masked_crops(image, mask, scales=(1.0, 1.5)):
    """Keep only the organ; context changes crop scale, never reveals other organs."""
    rgb = np.asarray(image.convert("RGB"))
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != rgb.shape[:2] or not mask.any():
        raise ValueError("Mask must be nonempty and match image dimensions")
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
    centre = ((x0+x1)/2, (y0+y1)/2)
    isolated = np.where(mask[..., None], rgb, 127).astype(np.uint8)
    output = []
    for scale in scales:
        if scale < 1:
            raise ValueError("Crop scale must be at least one")
        # Square neutral padding retains both tips of long leaves/stems.
        w = h = max(x1-x0, y1-y0)*scale
        # A padded crop avoids silently shrinking scale at an image edge.
        box = (int(np.floor(centre[0]-w/2)), int(np.floor(centre[1]-h/2)),
               int(np.ceil(centre[0]+w/2)), int(np.ceil(centre[1]+h/2)))
        canvas = Image.new("RGB", (box[2]-box[0], box[3]-box[1]), (127, 127, 127))
        source_box = (max(0, box[0]), max(0, box[1]), min(image.width, box[2]), min(image.height, box[3]))
        canvas.paste(Image.fromarray(isolated).crop(source_box), (source_box[0]-box[0], source_box[1]-box[1]))
        output.append(canvas)
    return output


def main():
    import torch
    from torchvision import transforms
    from torchvision.models import ResNet50_Weights, resnet50
    from pycocotools.coco import COCO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.5])
    parser.add_argument("--weights", type=Path, help="Local torchvision ResNet50 ImageNet state_dict; omit to download IMAGENET1K_V2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1 or any(scale < 1 for scale in args.scales):
        parser.error("Confidence must lie in [0,1] and crop scales must be >=1")
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=None if args.weights else weights)
    if args.weights:
        model.load_state_dict(torch.load(str(args.weights), map_location="cpu", weights_only=True), strict=True)
    model.fc = torch.nn.Identity()
    model.requires_grad_(False).eval().to(args.device)
    transform = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor(),
                                    transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    gt = COCO(str(args.data_dir / "annotations" / f"{args.split}.json"))
    predictions_bytes = args.predictions.read_bytes()
    predictions = json.loads(predictions_bytes)
    vectors, rows = [], []
    with torch.inference_mode():
        for index, record in enumerate(predictions):
            if record["score"] < args.confidence:
                continue
            if record["image_id"] not in gt.imgs or record["category_id"] not in gt.cats:
                raise ValueError("Prediction does not belong to requested split/category")
            mask = gt.annToMask(record).astype(bool)
            if not mask.any():
                continue
            info = gt.imgs[record["image_id"]]
            with Image.open(args.data_dir / "images" / info["file_name"]) as image:
                crops = masked_crops(image, mask, args.scales)
            batch = torch.stack([transform(crop) for crop in crops]).to(args.device)
            vectors.append(model(batch).float().cpu().numpy())
            rows.append({"row": len(rows), "prediction_index": index, "image_id": record["image_id"],
                         "sample_id": info.get("sample_id", ""), "category_id": record["category_id"],
                         "organ": gt.cats[record["category_id"]].get("original_label", gt.cats[record["category_id"]]["name"]),
                         "score": record["score"], "split": args.split})
    if not vectors:
        raise ValueError("No predicted organ masks passed the threshold")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "features.npz", features=np.stack(vectors), scales=args.scales)
    with (args.output_dir / "index.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"encoder": "torchvision.resnet50", "weights": str(args.weights) if args.weights else "IMAGENET1K_V2",
                "frozen": True, "mask_source": "predictions", "scales": args.scales,
                "shape": list(np.stack(vectors).shape), "confidence": args.confidence,
                "predictions_sha256": hashlib.sha256(predictions_bytes).hexdigest(),
                "normalization": "ImageNet mean/std; square neutral padding, resize224; no centre crop",
                "missing_organs": "no row; never silently zero-imputed"}
    (args.output_dir / "features.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Exported {len(rows)} organ features, {len(args.scales)} scales x 2048 dimensions")


if __name__ == "__main__":
    main()
