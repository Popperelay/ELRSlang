"""Interactive viewer CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from .app import write_texture_png
from .device import import_slangpy
from .renderer import Renderer, RendererConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ELRSlang SlangPy viewer.")
    parser.add_argument("--scene", type=Path, default=None, help="Scene file path.")
    parser.add_argument(
        "--graph",
        default="slangpy_preview",
        help="Render graph name or JSON path to run.",
    )
    parser.add_argument("--backend", default="automatic", help="SlangPy backend: d3d12/vulkan/metal.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--debug", action="store_true", help="Enable graphics debug layers.")
    parser.add_argument("--raytracing", action="store_true", help="Enable ray tracing graph features.")
    parser.add_argument("--frames", type=int, default=0, help="Run N frames then exit; 0 means interactive.")
    parser.add_argument("--capture", type=Path, default=None, help="Write the final output image to this path.")
    parser.add_argument("--print-timings", action="store_true", help="Print per-pass CPU timings after exit.")
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
        if args.capture is not None and renderer.context.output is not None:
            write_texture_png(renderer.context.output, import_slangpy(), args.capture)
        print(
            f"Rendered {frame_count} frame(s) with graph `{renderer.graph.name}` "
            f"at {renderer.context.width}x{renderer.context.height}."
        )
        if args.print_timings and stats is not None and stats.timings:
            for name, seconds in stats.timings.items():
                print(f"{name}: {seconds * 1000.0:.3f} ms")
        return 0

    stats = None
    while renderer.app.process_events():
        stats = renderer.frame()
    if args.capture is not None:
        renderer.app.screenshot(args.capture)
    if args.print_timings and stats is not None and stats.timings:
        for name, seconds in stats.timings.items():
            print(f"{name}: {seconds * 1000.0:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
