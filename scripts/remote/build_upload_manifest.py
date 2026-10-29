#!/usr/bin/env python3
"""Build the SHA-256 upload manifest for the remote 4090 workload.

The manifest lists exactly the files that ``upload_workload.sh`` pushes to
the remote host, so a single ``sha256sum -c MANIFEST.sha256`` over there is a
complete double-sided verification of the transfer. Two rules keep it
trustworthy:

* every include path must exist -- a missing file is a hard error, because a
  silently partial upload would only surface hours later as a failed
  generation run;
* symlinks are never followed, so the manifest can only ever describe real
  regular files (rsync ``--archive`` would copy a symlink as a symlink and
  ``sha256sum -c`` would then read through it).

Include rules (``--include`` may be repeated and replaces the defaults):

* ``tree:PATH``            every regular file under PATH, recursively
* ``file:PATH``            a single file
* ``glob:PATTERN``         root-relative glob, files only
* ``names:PATH:a,b,c``     only those file names directly inside PATH
* ``suffix:PATH:.jsonl,.json``  only files with those suffixes directly
                           inside PATH (sub-directories are never walked)

Excluded whatever the rule: ``__pycache__``, ``*.pyc``, ``.git``,
``.hermes-memory``, ``.codex_tmp``, ``AGENTS.md`` and the manifest itself.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path, PurePosixPath

MANIFEST_NAME = "MANIFEST.sha256"

# Path segments that never belong in an upload, whatever the include rule.
EXCLUDED_SEGMENTS = frozenset({"__pycache__", ".git", ".hermes-memory", ".codex_tmp"})
EXCLUDED_SUFFIXES = (".pyc",)
EXCLUDED_TOP_LEVEL = frozenset({"AGENTS.md"})

# Journal / scratch directories that must stay local: old journals and
# verification artifacts are not workload inputs. They live under
# results/response_gen/, whose rule only ever looks at direct children, so
# they are unreachable by construction -- the set documents the intent.
RESPONSE_GEN_SKIP_DIRS = frozenset({"artifact_verify_day3", "t4_remote_pull"})

DEFAULT_INCLUDES = (
    "tree:scripts",
    "tree:src",
    "tree:configs",
    "tree:tests",
    "tree:docs",
    "tree:reports",
    "file:README.md",
    "glob:requirements*.txt",
    "tree:.github",
    "suffix:results/response_gen:.jsonl,.json",
    "names:data/processed/b1_full_v1:response_val.jsonl,SHA256SUMS,manifest.json",
    "tree:weights/t4_longweight_2ep",
    "tree:weights/response_sft_d4_2ep",
    "file:weights/README.md",
)

RULE_KINDS = ("tree", "file", "glob", "names", "suffix")
LARGE_FILE_BYTES = 1024 * 1024
_READ_BLOCK = 1024 * 1024


class ManifestError(Exception):
    """A missing include path or an unusable rule: fail before any upload."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_rule(spec: str) -> tuple[str, str, tuple[str, ...]]:
    """Split ``kind:anchor[:param,param]`` into its parts."""
    kind, _, rest = spec.partition(":")
    if kind not in RULE_KINDS:
        raise ManifestError(
            f"unknown include rule {spec!r} (expected one of {', '.join(RULE_KINDS)})"
        )
    anchor, _, extra = rest.partition(":")
    anchor = anchor.strip()
    if not anchor:
        raise ManifestError(f"include rule {spec!r} has no path")
    params = tuple(part.strip() for part in extra.split(",") if part.strip())
    if kind in ("names", "suffix") and not params:
        raise ManifestError(f"include rule {spec!r} needs at least one parameter")
    return kind, anchor, params


def is_excluded(rel: PurePosixPath) -> bool:
    """Global exclusions, applied on top of every include rule."""
    parts = rel.parts
    if not parts:
        return True
    if any(segment in EXCLUDED_SEGMENTS for segment in parts):
        return True
    if any(segment.endswith(EXCLUDED_SUFFIXES) for segment in parts):
        return True
    if parts[0] in EXCLUDED_TOP_LEVEL:
        return True
    return False


def _anchor(root: Path, anchor: str) -> Path:
    """Resolve a rule anchor, refusing symlinks (never follow them)."""
    path = root / anchor
    if path.is_symlink():
        raise ManifestError(f"include path is a symlink, refusing to follow it: {anchor}")
    return path


def _walk_tree(root: Path, anchor: str) -> list[Path]:
    """Every regular file under ``anchor``; excluded dirs are pruned."""
    base = _anchor(root, anchor)
    if not base.exists():
        raise ManifestError(f"include path not found: {anchor}")
    if not base.is_dir():
        raise ManifestError(f"include path is not a directory: {anchor}")
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        # Prune excluded directories so we never descend into them, and drop
        # symlinked directories so os.walk cannot follow them either.
        kept: list[str] = []
        for name in sorted(dirnames):
            if name in EXCLUDED_SEGMENTS or name.endswith(EXCLUDED_SUFFIXES):
                continue
            if (Path(dirpath) / name).is_symlink():
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in sorted(filenames):
            candidate = Path(dirpath) / name
            if candidate.is_symlink() or not candidate.is_file():
                continue
            found.append(candidate)
    if not found:
        raise ManifestError(f"include path contains no uploadable file: {anchor}")
    return found


def _collect(root: Path, spec: str) -> list[Path]:
    kind, anchor, params = parse_rule(spec)
    if kind == "tree":
        return _walk_tree(root, anchor)

    if kind == "glob":
        matches = [p for p in sorted(root.glob(anchor)) if not p.is_symlink() and p.is_file()]
        if not matches:
            raise ManifestError(f"include pattern matched nothing: {anchor}")
        return matches

    base = _anchor(root, anchor)
    if not base.exists():
        raise ManifestError(f"include path not found: {anchor}")

    if kind == "file":
        if not base.is_file():
            raise ManifestError(f"include path is not a file: {anchor}")
        return [base]

    if kind == "names":
        missing = [name for name in params if not (base / name).is_file()]
        if missing:
            raise ManifestError(
                f"include path {anchor} is missing required file(s): {', '.join(missing)}"
            )
        return [base / name for name in params]

    # suffix: direct children only, so sub-directories such as
    # results/response_gen/t4_val_progress/ are never walked at all.
    wanted = tuple(params)
    matches = [
        child
        for child in sorted(base.iterdir())
        if not child.is_symlink()
        and child.is_file()
        and child.name.endswith(wanted)
        and child.name not in RESPONSE_GEN_SKIP_DIRS
    ]
    if not matches:
        raise ManifestError(
            f"include path {anchor} has no file ending in {', '.join(wanted)}"
        )
    return matches


def build_manifest(
    root: Path, includes: list[str], output: Path
) -> tuple[list[str], dict[str, tuple[str, int]]]:
    """Hash every included file; return (manifest lines, relpath -> (digest, size))."""
    output_resolved = output.resolve()
    rows: dict[str, tuple[str, int]] = {}
    for spec in includes:
        for path in _collect(root, spec):
            rel = PurePosixPath(path.relative_to(root).as_posix())
            if is_excluded(rel):
                continue
            if path.resolve() == output_resolved:
                continue
            relpath = rel.as_posix()
            if relpath in rows:
                continue
            rows[relpath] = (sha256(path), path.stat().st_size)
    if not rows:
        raise ManifestError("no files selected for upload; check the include rules")
    lines = [f"{digest}  {relpath}" for relpath, (digest, _size) in sorted(rows.items())]
    return lines, rows


def main(argv: list[str] | None = None) -> int:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Build the SHA-256 manifest of the files uploaded to the remote workload host"
    )
    parser.add_argument("--root", type=Path, default=default_root, help="repository root (default: two levels above this script)")
    parser.add_argument("--output", type=Path, default=None, help="manifest path (default: <root>/MANIFEST.sha256)")
    parser.add_argument("--include", action="append", default=None, metavar="RULE", help="include rule, e.g. tree:scripts (repeatable; replaces the defaults)")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: root directory not found: {args.root}", file=sys.stderr)
        return 1
    output = (args.output or (root / MANIFEST_NAME)).resolve()
    includes: list[str] = list(args.include) if args.include else list(DEFAULT_INCLUDES)

    try:
        lines, rows = build_manifest(root, includes, output)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total_bytes = sum(size for _digest, size in rows.values())
    print(f"PASS files={len(rows)} total_bytes={total_bytes} manifest={output}")
    big = sorted(
        ((size, relpath) for relpath, (_digest, size) in rows.items() if size >= LARGE_FILE_BYTES),
        reverse=True,
    )
    if big:
        print(f"large files (>= {LARGE_FILE_BYTES // (1024 * 1024)} MiB), largest first:")
        for size, relpath in big:
            print(f"  {size:>12}  {relpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
