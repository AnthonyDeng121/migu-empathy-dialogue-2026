"""Opt-in, content-addressed inference journal. No model dependencies."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def tree_digest(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = []

    def visit(current, relative, ancestors):
        resolved = current.resolve()
        if resolved in ancestors:
            raise ValueError(f"Cyclic model directory symlink: {current}")
        for path in sorted(current.iterdir()):
            name = relative / path.name
            if path.is_dir():
                visit(path, name, ancestors | {resolved})
            elif path.is_file():
                files.append((name.as_posix(), file_digest(path)))
            else:
                raise ValueError(f"Unreadable model asset: {path}")

    visit(directory, Path(), set())
    if not files:
        raise ValueError(f"Empty model directory: {directory}")
    return digest(files)


def fingerprint(project, inputs, config, model_dirs):
    """Hash actual assets, including tokenizer, thresholds and model defaults."""
    project = Path(project)
    packages = {}
    for name in ("torch", "transformers", "peft", "safetensors", "tokenizers", "numpy", "accelerate"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    runtime = {}
    if packages["torch"] is not None:
        import torch
        runtime = {"cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                   "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "cudnn_version": torch.backends.cudnn.version(),
                   "cudnn_deterministic": torch.backends.cudnn.deterministic,
                   "cudnn_benchmark": torch.backends.cudnn.benchmark}
    return {
        "version": 1,
        "inputs": {key: file_digest(path) for key, path in inputs.items() if path is not None},
        "config": json.loads(json.dumps(config, default=str)),
        "models": {key: tree_digest(path) for key, path in model_dirs.items() if path is not None},
        "code": digest([(p.relative_to(project).as_posix(), file_digest(p))
                        for root in ("src", "scripts") for p in sorted((project / root).rglob("*.py"))]),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "executable": sys.executable, "packages": packages, "runtime": runtime,
                        "variables": {key: os.environ.get(key) for key in
                                      ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
                                       "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTORCH_CUDA_ALLOC_CONF")}},
    }


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class Journal:
    """One writer, strict ordered prefix, durable records with a checksum chain.

    Only a final record without a newline is discarded. Complete invalid records
    are errors, never silently skipped. OS locks release even after process death.
    """

    def __init__(self, directory, identity, ids, validate):
        self.directory = Path(directory)
        self.identity = identity
        self.ids = ids
        self.validate = validate
        self.records = []
        self.previous = digest(identity)
        self.lock = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / "writer.lock").open("a+b")
        if os.fstat(self.lock.fileno()).st_size == 0:
            self.lock.write(b"0")
            self.lock.flush()
        self.lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.lock.close()
            raise RuntimeError("Resume directory already has an active writer") from error
        try:
            manifest = self.directory / "manifest.json"
            self.path = self.directory / "records.jsonl"
            if manifest.exists():
                if json.loads(manifest.read_text(encoding="utf-8")) != self.identity:
                    raise ValueError("Resume fingerprint mismatch: use a new resume directory")
            else:
                if self.path.exists():
                    raise ValueError("Resume records exist without a manifest")
                atomic_json(manifest, self.identity)
            if self.path.exists():
                with self.path.open("r+b") as handle:
                    while line := handle.readline():
                        if not line.endswith(b"\n"):
                            handle.seek(-len(line), 1)
                            handle.truncate()
                            handle.flush()
                            os.fsync(handle.fileno())
                            break
                        record = json.loads(line)
                        checksum = record.pop("checksum")
                        if checksum != digest([self.previous, record]):
                            raise ValueError("Corrupt resume record checksum")
                        self._validate_record(record)
                        self.records.append(record)
                        self.previous = checksum
            self.reused = len(self.records)
            self.started = time.perf_counter()
            self._event({"event": "start", "reused_samples": self.reused})
            return self
        except BaseException:
            self._unlock()
            raise

    def _validate_record(self, record):
        index = len(self.records)
        if index >= len(self.ids) or record["index"] != index:
            raise ValueError("Invalid resume record order or count")
        self.validate(record["prediction"], self.ids[index])
        if not isinstance(record["seconds"], (int, float)) or not math.isfinite(record["seconds"]) or record["seconds"] < 0:
            raise ValueError("Invalid resume latency")
        if type(record["tokens"]) is not int or record["tokens"] < 0:
            raise ValueError("Invalid resume token count")

    def append(self, prediction, seconds, tokens=0):
        record = {"index": len(self.records), "prediction": prediction, "seconds": seconds, "tokens": tokens}
        self._validate_record(record)
        checksum = digest([self.previous, record])
        with self.path.open("ab") as handle:
            handle.write((json.dumps({**record, "checksum": checksum}, ensure_ascii=False) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(record)
        self.previous = checksum

    def _event(self, event):
        with (self.directory / "attempts.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"unix_time": time.time(), **event}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def summary(self):
        return {"reused_samples": self.reused, "new_samples": len(self.records) - self.reused,
                "cumulative_successful_sample_seconds": sum(r["seconds"] for r in self.records),
                "current_attempt_wall_seconds": time.perf_counter() - self.started,
                "cold_start": False, "fingerprint_sha256": digest(self.identity),
                "timing_note": "Successful sample compute only; failed/in-flight work excluded. Attempt wall times in attempts.jsonl; interrupted attempts may lack an end event."}

    def _unlock(self):
        self.lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
        self.lock.close()

    def __exit__(self, kind, value, traceback):
        try:
            self._event({"event": "end", "status": "failed" if kind else "completed",
                         "wall_seconds": time.perf_counter() - self.started,
                         "completed_samples": len(self.records)})
        finally:
            self._unlock()
