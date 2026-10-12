#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
from pathlib import Path


PURPOSES = {
    "README.md": "官方资料总说明与提交包内专家评审材料路径要求",
    "test_inference_data.jsonl": "官方推理入口测试样例",
    "咪咕仝学平台实例创建demo V1.docx": "咪咕仝学实例创建和 vLLM 公有镜像操作演示",
    "participant/README.md": "participant v2 运行契约、后端、产物和计时说明",
    "participant/run_inference.py": "官方参考推理实现",
    "participant/start.sh": "官方唯一启动入口脚本",
    "participant/requirements.txt": "官方参考依赖下限",
    "participant/configs/inference_config.json": "官方参考推理配置",
    "participant/templates/submission_schema.json": "官方提交对象 JSON Schema",
    "participant/templates/submission_template.json": "官方提交对象样例",
    "训练-验证-数据集/train/train_public.jsonl": "公开训练数据",
    "训练-验证-数据集/train/train_public.schema.json": "训练数据 JSON Schema",
    "训练-验证-数据集/val/val_public.jsonl": "公开验证数据",
    "训练-验证-数据集/val/val_public.schema.json": "验证数据 JSON Schema",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SHA-256 manifest for the immutable official material copy")
    parser.add_argument("official_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted((p for p in args.official_dir.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
        relative = path.relative_to(args.official_dir).as_posix()
        rows.append((path.name, relative, path.suffix.lower().lstrip(".") or "none", path.stat().st_size, datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"), sha256(path), PURPOSES.get(relative, "官方附件"), "是", "视最终镜像清单；数据集本身不提交"))
    lines = [
        "# 官方资料清单",
        "",
        "`official_materials/` 是原始资料的逐字节副本。开发产物位于 `team_project/`；任何清洗或转换数据只能写入 `team_project/data/processed/`。",
        "",
        "| 文件名 | 相对路径 | 类型 | 字节 | 修改时间 | SHA-256 | 用途 | 已读 | 提交必需 |",
        "|---|---|---|---:|---|---|---|---|---|",
    ]
    for row in rows:
        escaped = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(escaped) + " |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS files={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()

