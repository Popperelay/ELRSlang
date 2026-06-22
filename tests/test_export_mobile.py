from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elrslang.tools.export_mobile import export_mobile_assets


class ExportMobileTests(unittest.TestCase):
    def test_export_mobile_assets_writes_manifest_graph_and_shaders(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "asset_pack"
            manifest_path = export_mobile_assets("slangpy_preview", out_dir)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "dev.elrslang.mobile_asset_pack.v1")
            self.assertTrue((out_dir / "graph.json").exists())
            self.assertTrue((out_dir / "shaders" / "preview.slang").exists())


if __name__ == "__main__":
    unittest.main()
