"""Fine-tune the original Meta SAM3 through its official Hydra Trainer."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SAM3_COMMIT = "eef6839bcfc3458dd52ea80535dc6bd62aea86b1"


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-root", type=Path, default=ROOT.parent / "external/sam3")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint", type=Path, default=ROOT.parent / "weights/sam3.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/sam3")
    parser.add_argument("--epochs", type=positive_int, default=50)
    parser.add_argument("--num-gpus", type=positive_int, default=5,
                        help="visible GPUs on one node; default 5 for the five RTX 4090 server")
    parser.add_argument("--batch-size", type=positive_int, default=1,
                        help="training images per GPU per optimizer step; global batch = this times --num-gpus")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="freeze original visual and language backbones; architecture is unchanged")
    parser.add_argument("--dry-run", action="store_true", help="write configuration without importing SAM3 or torch")
    parser.add_argument("--check", action="store_true", help="report environment and input blockers, then exit")
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    for name in ("sam3_root", "data_dir", "checkpoint", "output_dir"):
        setattr(args, name, getattr(args, name).absolute().resolve())
    return args


def build_config(args):
    cfg = yaml.safe_load((ROOT / "configs/sam3_base.yaml").read_text(encoding="utf-8"))
    cfg["paths"].update(dataset_root=args.data_dir.resolve().as_posix(),
                        experiment_log_dir=args.output_dir.resolve().as_posix(),
                        bpe_path=(args.sam3_root.resolve() / "sam3/assets/bpe_simple_vocab_16e6.txt.gz").as_posix(),
                        checkpoint=args.checkpoint.resolve().as_posix())
    scratch, trainer = cfg["scratch"], cfg["trainer"]
    cfg["launcher"]["gpus_per_node"] = args.num_gpus
    scratch.update(train_batch_size=args.batch_size, val_batch_size=1,
                   num_train_workers=args.workers, num_val_workers=args.workers)
    trainer.update(max_epochs=args.epochs, seed_value=args.seed)
    for split in ("train", "val"):
        ds = trainer["data"][split]["dataset"]
        ds["img_folder"] = (args.data_dir.resolve() / "images").as_posix()
        ds["ann_file"] = (args.data_dir.resolve() / "annotations" / (split + ".json")).as_posix()
        ds["transforms"] = copy.deepcopy(cfg["tea_data"][split + "_transforms"])
        trainer["data"][split]["collate_fn"] = copy.deepcopy(scratch["collate_fn" if split == "train" else "collate_fn_val"])
    trainer["loss"]["all"] = copy.deepcopy(cfg["tea_data"]["loss"])
    trainer["model"]["freeze_backbone"] = args.freeze_backbone
    return cfg


def batch_configuration(args):
    return {"num_gpus": args.num_gpus, "per_gpu_batch_size": args.batch_size,
            "gradient_accumulation_steps": 1,
            "effective_global_batch_size": args.num_gpus * args.batch_size,
            "learning_rate_scaling": "none"}


def assess_gpus(devices, requested):
    """Validate a real CUDA-visible inventory without assuming physical device IDs."""
    report = {"visible_gpu_count": len(devices), "selected_gpus": devices[:requested],
              "blockers": [], "warnings": []}
    if len(devices) < requested:
        report["blockers"].append(
            f"Requested {requested} GPUs but only {len(devices)} are visible. Check CUDA_VISIBLE_DEVICES or --num-gpus."
        )
    for device in report["selected_gpus"]:
        index = device["index"]
        if tuple(device["capability"]) < (8, 0):
            report["blockers"].append(f"Visible GPU {index} lacks native bfloat16 support (compute capability >=8.0 required).")
        if device["total_gib"] < 16:
            report["warnings"].append(
                f"Visible GPU {index} has only {device['total_gib']} GiB memory; full 1008px fine-tuning may run out of memory."
            )
        elif device["free_gib"] < 16:
            report["warnings"].append(
                f"Visible GPU {index} has only {device['free_gib']} GiB free memory; check other processes before training."
            )
    return report


def build_trainer(**kwargs):
    """Construct the original Trainer with restricted NumPy metadata support for resume."""
    from chapter3_segmentation.checkpoint import numpy_metadata_context
    from sam3.train.trainer import Trainer

    # At the pinned revision, Trainer.__init__ loads both the image model and
    # any automatic-resume checkpoint before returning. Its run loop is unchanged.
    with numpy_metadata_context():
        return Trainer(**kwargs)


def build_train_model(checkpoint_path, freeze_backbone=False, **kwargs):
    """Strictly initialize the original image architecture, then optionally freeze its backbone."""
    from chapter3_segmentation.checkpoint import load_image_state
    from sam3.model_builder import build_sam3_image_model

    state = load_image_state(checkpoint_path)
    kwargs.update(device="cpu", eval_mode=False, load_from_HF=False, enable_inst_interactivity=False)
    model = build_sam3_image_model(checkpoint_path=None, **kwargs)
    model.load_state_dict(state, strict=True)
    if freeze_backbone:
        model.backbone.requires_grad_(False)
    return model


def _version_tuple(value):
    return tuple(int(part) for part in value.split("+")[0].split(".")[:2])


def preflight(args, inspect_gpu=True):
    """Return explicit blockers; configuration-only dry runs never import model dependencies."""
    report = {"python": sys.version.split()[0], "platform": platform.system(),
              "sam3_source_commit": SAM3_COMMIT, "batch": batch_configuration(args),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "blockers": [], "warnings": []}
    blockers = report["blockers"]
    if "LOCAL_RANK" in os.environ or os.environ.get("WORLD_SIZE", "1") not in ("", "1"):
        blockers.append("Launch train.py once with plain python; the official SAM3 launcher spawns its own DDP workers. Remove outer torchrun/mpirun.")
    if sys.version_info < (3, 12):
        blockers.append("Official SAM3 requires Python >=3.12; create a separate training environment.")
    if platform.system() != "Linux":
        blockers.append("Use Linux or WSL2 for this official Triton/NCCL training backend; native Windows is unsupported here.")
    source = args.sam3_root.resolve()
    for relative in ("sam3/train/train.py", "sam3/train/trainer.py", "sam3/assets/bpe_simple_vocab_16e6.txt.gz"):
        if not (source / relative).is_file():
            blockers.append("Missing official SAM3 source file: " + str(source / relative))
    if (source / ".git").exists():
        result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True)
        report["installed_source_commit"] = result.stdout.strip()
        if result.returncode or result.stdout.strip() != SAM3_COMMIT:
            blockers.append("SAM3 checkout must use the verified commit " + SAM3_COMMIT)
    if not args.checkpoint.is_file():
        blockers.append("Missing original SAM3 checkpoint: " + str(args.checkpoint.resolve()))
    for split in ("train", "val"):
        annotation = args.data_dir / "annotations" / (split + ".json")
        if not annotation.is_file():
            blockers.append("Missing COCO annotations: " + str(annotation.resolve()))
            continue
        try:
            coco = json.loads(annotation.read_text(encoding="utf-8"))
            if not coco.get("images") or not coco.get("categories") or not coco.get("annotations"):
                blockers.append(str(annotation) + " must contain images, categories and segmentation annotations.")
            if any(not ann.get("segmentation") for ann in coco.get("annotations", [])):
                blockers.append(str(annotation) + " contains objects without segmentation masks.")
            missing = [img["file_name"] for img in coco.get("images", [])
                       if not (args.data_dir / "images" / img["file_name"]).is_file()]
            if missing:
                blockers.append(str(annotation) + ": " + str(len(missing)) + " image files missing.")
        except (OSError, ValueError, KeyError, TypeError) as error:
            blockers.append("Invalid COCO input " + str(annotation) + ": " + str(error))
    required = ("torch", "torchvision", "sam3", "hydra-core", "submitit", "tensorboard", "timm",
                "pycocotools", "triton", "fvcore", "fairscale", "zstandard", "torchmetrics")
    versions = {}
    for name in required:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            blockers.append("Missing training package: " + name)
    report["packages"] = versions
    if importlib.util.find_spec("pkg_resources") is None:
        blockers.append("Missing pkg_resources, required by this SAM3 revision; install setuptools<82.")
    if "torch" in versions and _version_tuple(versions["torch"]) < (2, 7):
        blockers.append("Official SAM3 requires torch >=2.7.")
    if inspect_gpu:
        try:
            import torch
            report["torch_cuda"] = torch.version.cuda
            if not torch.cuda.is_available():
                blockers.append("CUDA GPU is unavailable in this Python environment.")
            else:
                devices = []
                for index in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(index)
                    free_bytes, _ = torch.cuda.mem_get_info(index)
                    devices.append({"index": index, "name": props.name,
                                    "total_gib": round(props.total_memory / 1024**3, 2),
                                    "free_gib": round(free_bytes / 1024**3, 2),
                                    "capability": [props.major, props.minor]})
                inventory = assess_gpus(devices, args.num_gpus)
                report["visible_gpu_count"] = inventory["visible_gpu_count"]
                report["selected_gpus"] = inventory["selected_gpus"]
                blockers.extend(inventory["blockers"])
                report["warnings"].extend(inventory["warnings"])
                if torch.version.cuda is None or _version_tuple(torch.version.cuda) < (12, 6):
                    blockers.append("Official SAM3 requires a CUDA >=12.6 PyTorch build.")
                if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
                    blockers.append("This PyTorch build does not provide the NCCL distributed backend required by SAM3.")
        except (ImportError, RuntimeError, OSError) as error:
            blockers.append("Cannot inspect CUDA: " + str(error))
    report["warnings"].append("Data must be split by specimen/session where available; validation uses val.json, never test.json.")
    return report


def build_launch_command(args, cfg_path):
    """Invoke the official one-node launcher once; it owns all GPU process spawning."""
    launch = (
        "import sys; from argparse import Namespace; "
        "from hydra import initialize_config_dir; "
        "from sam3.train.train import main, register_omegaconf_resolvers; "
        "register_omegaconf_resolvers(); "
        "ctx=initialize_config_dir(config_dir=sys.argv[1],version_base='1.2'); ctx.__enter__(); "
        "main(Namespace(config='sam3_train',use_cluster=False,num_gpus=int(sys.argv[2]),num_nodes=1,"
        "partition=None,account=None,qos=None))"
    )
    return [sys.executable, "-c", launch, str(cfg_path.parent), str(args.num_gpus)]


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = args.output_dir.resolve() / "sam3_train.yaml"
    cfg_path.write_text(yaml.safe_dump(build_config(args), sort_keys=False, allow_unicode=True), encoding="utf-8")
    report = preflight(args, inspect_gpu=not args.dry_run)
    report_path = args.output_dir / "preflight.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Configuration: " + str(cfg_path))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run only; no model was loaded and no training was performed.")
        return 0
    if report["blockers"]:
        return 2
    if args.check:
        return 0
    # Hydra composes our full standalone YAML, while the official launcher owns DDP and training.
    # No generated files are copied into the upstream checkout.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT.parent), str(args.sam3_root.resolve()), env.get("PYTHONPATH", "")])
    return subprocess.run(build_launch_command(args, cfg_path),
                          cwd=str(args.sam3_root.resolve()), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
