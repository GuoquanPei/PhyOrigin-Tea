#!/usr/bin/env bash
# One Linux entry point; use distinct output directories for smoke and full training.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
chapter="$project_root/chapter3_segmentation"
stage=all
data_root=""
data_dir="$chapter/data"
checkpoint="$project_root/weights/sam3.pt"
sam3_root="$project_root/external/sam3"
python="${SAM3_PYTHON:-$project_root/.venv-sam3/bin/python}"
output_dir="$chapter/runs/sam3"
eval_checkpoint=""
groups_csv=""
gpus=5
batch_size=1
epochs=50
workers=2
seed=42
freeze_backbone=false
force_prepare=false

usage() {
    cat <<'HELP'
Usage: bash chapter3_segmentation/run_server.sh [options]
  --stage prepare|check|smoke|train|evaluate|all  Default: all
  --data-dir PATH       Prepared images/, annotations/, manifest.csv, split_stats.json
  --data-root PATH      Raw LabelMe device/site/... directories; needed to prepare missing data
  --checkpoint PATH     Original Meta SAM3 pretrained checkpoint (default: weights/sam3.pt)
  --sam3-root PATH      Pinned upstream checkout (default: external/sam3)
  --python PATH         Training Python (default: .venv-sam3/bin/python; env SAM3_PYTHON)
  --output-dir PATH     Full training output (default: chapter3_segmentation/runs/sam3)
  --eval-checkpoint PATH  Evaluate this trained checkpoint instead of OUTPUT/checkpoints/checkpoint.pt
  --gpus N             Number of visible GPUs (default: 5)
  --batch-size N       Image/query datapoints per GPU per step (default: 1)
  --epochs N           Full training epochs (default: 50)
  --workers N          Loader workers per rank (default: 2)
  --seed N             Data/training seed (default: 42)
  --groups-csv PATH     Optional biological identity mapping used during raw preparation
  --force-prepare      Explicitly regenerate prepared data from --data-root
  --freeze-backbone    Freeze original visual/language backbones; architecture unchanged
  -h, --help           Show help

all = prepare if necessary -> check -> separate one-epoch smoke -> train -> val/test evaluation.
Smoke uses a fresh tiny dataset/output under runs/smoke and cannot resume the full run.
CUDA_VISIBLE_DEVICES may select GPUs, e.g. CUDA_VISIBLE_DEVICES=0,1,2,3,4.
HELP
}

require_value() {
    [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --stage|--data-dir|--data-root|--checkpoint|--sam3-root|--python|--output-dir|--eval-checkpoint|--gpus|--batch-size|--epochs|--workers|--seed|--groups-csv)
            require_value "$@"
            key="$1"; value="$2"; shift 2
            case "$key" in
                --stage) stage="$value" ;;
                --data-dir) data_dir="$value" ;;
                --data-root) data_root="$value" ;;
                --checkpoint) checkpoint="$value" ;;
                --sam3-root) sam3_root="$value" ;;
                --python) python="$value" ;;
                --output-dir) output_dir="$value" ;;
                --eval-checkpoint) eval_checkpoint="$value" ;;
                --gpus) gpus="$value" ;;
                --batch-size) batch_size="$value" ;;
                --epochs) epochs="$value" ;;
                --workers) workers="$value" ;;
                --seed) seed="$value" ;;
                --groups-csv) groups_csv="$value" ;;
            esac ;;
        --freeze-backbone) freeze_backbone=true; shift ;;
        --force-prepare) force_prepare=true; shift ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
case "$stage" in prepare|check|smoke|train|evaluate|all) ;; *) echo "Unknown stage: $stage" >&2; exit 2 ;; esac
for value in "$gpus" "$batch_size" "$epochs"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "GPUs, batch size and epochs must be positive integers." >&2; exit 2; }
done
[[ "$workers" =~ ^[0-9]+$ && "$seed" =~ ^[0-9]+$ ]] || { echo "Workers and seed must be nonnegative integers." >&2; exit 2; }
[[ "$(uname -s)" == "Linux" ]] || { echo "Use Linux/WSL2 for this runner." >&2; exit 2; }
command -v "$python" >/dev/null || { echo "Training Python not found: $python. Run setup_sam3.sh first or use --python." >&2; exit 2; }

# Resolve before any launcher changes directory; paths containing spaces are supported.
absolute_path() {
    "$python" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$1"
}
data_dir="$(absolute_path "$data_dir")"
checkpoint="$(absolute_path "$checkpoint")"
sam3_root="$(absolute_path "$sam3_root")"
output_dir="$(absolute_path "$output_dir")"
[[ -z "$data_root" ]] || data_root="$(absolute_path "$data_root")"
[[ -z "$groups_csv" ]] || groups_csv="$(absolute_path "$groups_csv")"
[[ -z "$eval_checkpoint" ]] || eval_checkpoint="$(absolute_path "$eval_checkpoint")"
log_dir="$chapter/runs/server_logs"
mkdir -p "$log_dir"
log_file="$log_dir/$(date -u +%Y%m%dT%H%M%SZ)-${stage}-$$.log"
exec > >(tee -a "$log_file") 2>&1
echo "Stage=$stage; GPUs=$gpus; per-GPU batch=$batch_size; global image/query datapoint batch=$((gpus * batch_size))"
echo "Log: $log_file"

ensure_prepared() {
    local complete=true
    for required in manifest.csv split_stats.json annotations/train.json annotations/val.json annotations/test.json; do
        [[ -f "$data_dir/$required" ]] || complete=false
    done
    [[ -d "$data_dir/images" ]] || complete=false
    if [[ "$force_prepare" == true || "$complete" == false ]]; then
        [[ -n "$data_root" ]] || { echo "Prepared data incomplete. Supply --data-root /path/to/raw or transfer a prepared archive to --data-dir." >&2; exit 2; }
        prepare_args=(--data-root "$data_root" --output "$data_dir" --seed "$seed")
        [[ -z "$groups_csv" ]] || prepare_args+=(--groups-csv "$groups_csv")
        "$python" "$chapter/prepare_data.py" "${prepare_args[@]}"
    else
        echo "Reusing prepared data: $data_dir"
        if [[ -n "$groups_csv" ]]; then
            echo "--groups-csv applies only when preparing; use --force-prepare --data-root to apply a changed identity map." >&2
            exit 2
        fi
    fi
    "$python" - "$data_dir" "$seed" <<'PY'
import csv
import json
from pathlib import Path
import sys
root, seed = Path(sys.argv[1]), int(sys.argv[2])
stats = json.loads((root / "split_stats.json").read_text(encoding="utf-8"))
if stats["seed"] != seed:
    raise SystemExit(f"Prepared split seed={stats['seed']} differs from requested seed={seed}; select the matching seed or explicitly regenerate.")
with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
    manifest = list(csv.DictReader(stream))
if len(manifest) != stats["total_images"]:
    raise SystemExit("Manifest size differs from split_stats.json.")
counts = {}
seen = set()
for split in ("train", "val", "test"):
    coco = json.loads((root / "annotations" / f"{split}.json").read_text(encoding="utf-8"))
    counts[split] = len(coco["images"])
    for image in coco["images"]:
        if image["id"] in seen:
            raise SystemExit("Image ID occurs in multiple splits.")
        seen.add(image["id"])
        if not (root / "images" / image["file_name"]).is_file():
            raise SystemExit(f"Missing canonical image: {image['file_name']}")
if counts != stats["achieved_counts"] or sum(counts.values()) != len(manifest):
    raise SystemExit("COCO split sizes differ from manifest/split_stats.json.")
print("Prepared data verified:", counts)
PY
}

training_args=(--sam3-root "$sam3_root" --data-dir "$data_dir" --checkpoint "$checkpoint"
               --output-dir "$output_dir" --num-gpus "$gpus" --batch-size "$batch_size"
               --epochs "$epochs" --workers "$workers" --seed "$seed")
[[ "$freeze_backbone" == false ]] || training_args+=(--freeze-backbone)

run_check() {
    "$python" "$chapter/train.py" "${training_args[@]}" --check
}

run_smoke() {
    local smoke_root
    mkdir -p "$chapter/runs/smoke"
    smoke_root="$(mktemp -d "$chapter/runs/smoke/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
    "$python" "$chapter/make_smoke_data.py" --data-dir "$data_dir" --output-dir "$smoke_root/data" --num-gpus "$gpus"
    smoke_args=(--sam3-root "$sam3_root" --data-dir "$smoke_root/data" --checkpoint "$checkpoint"
                --output-dir "$smoke_root/training" --num-gpus "$gpus" --batch-size 1
                --epochs 1 --workers "$workers" --seed "$seed")
    [[ "$freeze_backbone" == false ]] || smoke_args+=(--freeze-backbone)
    "$python" "$chapter/train.py" "${smoke_args[@]}"
    [[ -f "$smoke_root/training/checkpoints/checkpoint.pt" ]] || {
        echo "Smoke run ended without its expected checkpoint: $smoke_root" >&2; exit 2;
    }
    echo "Smoke run completed in separate directory: $smoke_root"
}

run_train() {
    "$python" "$chapter/train.py" "${training_args[@]}"
}

run_evaluate() {
    local weights="${eval_checkpoint:-$output_dir/checkpoints/checkpoint.pt}"
    [[ -f "$weights" ]] || { echo "Trained checkpoint missing: $weights. Train first or supply --eval-checkpoint." >&2; exit 2; }
    for split in val test; do
        "$python" "$chapter/evaluate.py" --sam3-root "$sam3_root" --data-dir "$data_dir" --split "$split" \
            --checkpoint "$weights" --output-dir "$output_dir/evaluation/$split" --device cuda
    done
}

ensure_prepared
case "$stage" in
    prepare) ;;
    check) run_check ;;
    smoke) run_smoke ;;
    train) run_train ;;
    evaluate) run_evaluate ;;
    all) run_check; run_smoke; run_train; run_evaluate ;;
esac
