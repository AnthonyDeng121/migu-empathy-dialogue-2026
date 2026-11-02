#!/usr/bin/env python3
"""Read-only check; never installs or changes official image packages."""
import argparse
import importlib
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PROFILE_TREE_HASH_FORMAT = "relative-path-file-sha256-v1"
sys.path.insert(0, str(PROJECT))
from scripts.run_integrated import load_config, check_files


def _tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    digest.update(PROFILE_TREE_HASH_FORMAT.encode("ascii") + b"\0")
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative_path = path.relative_to(directory).as_posix().encode("utf-8")
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(b"file\0")
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(file_digest.digest())
    return digest.hexdigest()


def verify_model_hashes(config: dict, manifest: dict) -> None:
    expected = [(config["classifier_dir"] / item["path"], item["sha256"])
                for item in manifest["classifier_files"]]
    expected.append((config["response_adapter_dir"] / "adapter_model.safetensors",
                     manifest["response_adapter_sha256"]))
    if manifest.get("response_base_weights_sha256"):
        expected.append((config["response_model_dir"] / "model.safetensors",
                         manifest["response_base_weights_sha256"]))
    for path, digest in expected:
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Model hash mismatch: {path}")

    if config.get("profile_mode") == "independent":
        if manifest.get("profile_checkpoint_tree_hash_format") != PROFILE_TREE_HASH_FORMAT:
            raise ValueError("Independent profile checkpoint tree hash format is missing or unsupported")
        expected_tree = manifest.get("profile_checkpoint_tree_sha256")
        if not isinstance(expected_tree, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_tree):
            raise ValueError("Independent profile checkpoint tree SHA-256 is missing or invalid")
        actual_tree = _tree_sha256(config["profile_model_dir"])
        if actual_tree != expected_tree:
            raise ValueError(f"Independent profile checkpoint tree hash mismatch: {config['profile_model_dir']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/integrated.json")
    parser.add_argument("--files-only", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true", help="verify the selected handoff against configs/integrated_artifacts.json")
    args = parser.parse_args()
    errors, versions = [], {}
    try:
        config = load_config(args.config)
        check_files(config)
        if args.verify_hashes:
            manifest = json.loads((PROJECT / "configs/integrated_artifacts.json").read_text(encoding="utf-8"))
            verify_model_hashes(config, manifest)
    except Exception as error:
        errors.append(str(error))
    if not args.files_only:
        for name in ("torch", "transformers", "peft", "safetensors", "accelerate"):
            try:
                module = importlib.import_module(name)
                versions[name] = getattr(module, "__version__", "unknown")
            except Exception as error:
                errors.append(f"{name}: {error}")
        try:
            from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch
            if not torch.cuda.is_available():
                errors.append("CUDA unavailable: check instance allocation before training or inference")
            else:
                versions["gpu"] = torch.cuda.get_device_name(0)
                versions["gpu_memory_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
        except Exception as error:
            errors.append(str(error))
    print(json.dumps({"ready_for_smoke": not errors, "environment": versions, "errors": errors}, ensure_ascii=False, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
