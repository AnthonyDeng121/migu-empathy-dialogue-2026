#!/usr/bin/env python3
"""Package source and official smoke input, excluding reports, secrets and weights."""
import hashlib
import sys
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    output = PROJECT / "artifacts/cloud_source.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    allowed = [PROJECT / name for name in ("src", "scripts", "configs", "tests", "docs")]
    files = [path for directory in allowed for path in directory.rglob("*")
             if path.is_file() and path.suffix in {".py", ".sh", ".json", ".md"}
             and "__pycache__" not in path.parts and ".codex_tmp" not in path.parts]
    files += [PROJECT / "start.sh", PROJECT / "Dockerfile.integrated", PROJECT / "data/raw/test_inference_data.jsonl"]
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(PROJECT).as_posix())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(f"{output}\nSHA-256: {digest}\nModel folders must be transferred separately.")


if __name__ == "__main__":
    main()
