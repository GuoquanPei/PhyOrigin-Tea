"""Package prepared data for Linux transfer without modifying original files."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def export_archive(data_dir, output):
    data_dir, output = Path(data_dir).resolve(), Path(output).resolve()
    if output == data_dir or data_dir in output.parents:
        raise ValueError("Archive must be outside the dataset directory")
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing archive: {output}")
    for relative in ("manifest.csv", "split_stats.json", "annotations/train.json", "annotations/val.json", "annotations/test.json"):
        if not (data_dir / relative).is_file():
            raise FileNotFoundError(data_dir / relative)
    allowed = ["images", "annotations", "metadata", "manifest.csv", "split_stats.json", "annotation_review.csv"]
    files = []
    for relative in allowed:
        path = data_dir / relative
        if path.is_dir():
            files.extend(sorted(item for item in path.rglob("*") if item.is_file()))
        elif path.is_file():
            files.append(path)
    for path in files:
        if path.is_symlink() or data_dir not in path.resolve().parents:
            raise ValueError(f"Dataset archive cannot include links outside the dataset: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive mode keeps accidental reruns from replacing transfer artifacts.
    with output.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz", compresslevel=1, dereference=True) as archive:
            for path in files:
                archive.add(path, arcname=(Path("tea_data") / path.relative_to(data_dir)).as_posix(), recursive=False)
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    report = {"archive": str(output), "sha256": digest.hexdigest(), "files": len(files), "bytes": output.stat().st_size}
    output.with_name(output.name + ".sha256").write_text(f"{report['sha256']}  {output.name}\n", encoding="ascii")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "runs/tea_data.tar.gz")
    args = parser.parse_args()
    print(json.dumps(export_archive(args.data_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
