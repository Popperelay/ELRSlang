"""Interactive viewer CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from .renderer import Renderer, RendererConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ELRSlang SlangPy viewer.")
    parser.add_argument("--scene", type=Path, default=None, help="Scene file path.")
    parser.add_argument(
        "--graph",
        default="slangpy_preview",
        choices=["slangpy_preview", "raster_forward", "dxr_pathtrace"],
        help="Render graph to run.",
    )
    parser.add_argument("--backend", default="automatic", help="SlangPy backend: d3d12/vulkan/metal.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--debug", action="store_true", help="Enable graphics debug layers.")
    parser.add_argument("--raytracing", action="store_true", help="Enable ray tracing graph features.")
    parser.add_argument("--frames", type=int, default=0, help="Run N frames then exit; 0 means interactive.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    renderer = Renderer(
        RendererConfig(
            scene_path=args.scene,
            graph_name=args.graph,
            backend=args.backend,
            width=args.width,
            height=args.height,
            enable_debug=args.debug,
            enable_raytracing=args.raytracing,
            interactive=args.frames == 0,
        )
    )
    if renderer.app is None:
        frame_count = max(args.frames, 1)
        stats = None
        for _ in range(frame_count):
            stats = renderer.frame()
        print(
            f"Rendered {frame_count} frame(s) with graph `{renderer.graph.name}` "
            f"at {renderer.context.width}x{renderer.context.height}."
        )
        return 0

    while renderer.app.process_events():
        renderer.frame()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
