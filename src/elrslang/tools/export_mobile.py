"""Export a small asset contract for native mobile smoke apps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from elrslang.paths import GRAPH_DIR, SHADER_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export ELRSlang graph/shader assets for mobile smoke apps.")
    parser.add_argument("--graph", default="slangpy_preview", help="Graph name or JSON path.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    return parser


def export_mobile_assets(graph: str | Path, out_dir: Path) -> Path:
    graph_path = Path(graph)
    if not graph_path.suffix:
        graph_path = GRAPH_DIR / f"{graph_path}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    shader_out = out_dir / "shaders"
    shader_out.mkdir(exist_ok=True)

    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    shader_names = sorted(
        {
            item["module"]
            for item in graph_data.get("passes", [])
            if item.get("type") == "SlangFunctionPass" and "module" in item
        }
        | {
            item["shader"]
            for item in graph_data.get("passes", [])
            if item.get("type") in {"HardwareRasterPass", "HardwareDXRPass"} and "shader" in item
        }
    )
    for shader_name in shader_names:
        source = SHADER_DIR / shader_name
        if source.exists():
            shutil.copy2(source, shader_out / shader_name)

    shutil.copy2(graph_path, out_dir / "graph.json")
    manifest = {
        "schema": "dev.elrslang.mobile_asset_pack.v1",
        "graph": "graph.json",
        "shaders": [f"shaders/{name}" for name in shader_names],
        "host": "native",
        "python": False,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = export_mobile_assets(args.graph, args.out)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
