"""Prevent a runnable config from referring to weights omitted by the image."""
import json
import re
import unittest
from pathlib import Path


class DockerModelPathsTests(unittest.TestCase):
    def test_model_context_copies_match_runtime_config(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/integrated.json").read_text(encoding="utf-8"))
        copies = re.findall(r"^COPY --from=model_assets (\S+) (\S+)$",
                            (root / "Dockerfile.integrated").read_text(encoding="utf-8"), re.M)
        expected = {config[key].rstrip("/") for key in
                    ("classifier_dir", "response_model_dir", "response_adapter_dir")}
        self.assertEqual({target.rstrip("/") for _, target in copies}, expected)
        for source, target in copies:
            self.assertEqual(source.strip("/"), target.removeprefix("weights/").rstrip("/"))
