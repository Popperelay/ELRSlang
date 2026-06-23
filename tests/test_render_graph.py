from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elrslang.passes import ComputeFunctionPass, HardwareRasterPass, pass_from_config
from elrslang.render_graph import GraphCompileError, PassReflection, RenderContext, RenderGraph, RenderPass, ResourceDesc
from elrslang.renderer import load_graph


class DummyPass(RenderPass):
    def __init__(self, name: str, inputs=(), outputs=()):
        super().__init__(name)
        self._reflection = PassReflection.from_iterables(inputs=inputs, outputs=outputs)

    def reflect(self) -> PassReflection:
        return self._reflection

    def execute(self, context: RenderContext, inputs):
        return {name: f"{self.name}.{name}" for name in self._reflection.outputs}


class RenderGraphTests(unittest.TestCase):
    def test_topological_execution(self):
        graph = RenderGraph("test")
        graph.add_pass(DummyPass("A", outputs=("color",)))
        graph.add_pass(DummyPass("B", inputs=("input",), outputs=("out",)))
        graph.add_edge("A.color", "B.input")
        graph.mark_output("B.out")

        self.assertEqual(graph.compile(), ["A", "B"])
        context = RenderContext()
        graph.execute(context)
        self.assertEqual(context.output, "B.out")

    def test_cycle_detection(self):
        graph = RenderGraph("cycle")
        graph.add_pass(DummyPass("A", inputs=("in",), outputs=("out",)))
        graph.add_pass(DummyPass("B", inputs=("in",), outputs=("out",)))
        graph.add_edge("A.out", "B.in")
        graph.add_edge("B.out", "A.in")
        with self.assertRaisesRegex(GraphCompileError, "cycle"):
            graph.compile()

    def test_missing_required_input(self):
        graph = RenderGraph("missing")
        graph.add_pass(DummyPass("NeedsInput", inputs=("input",), outputs=("out",)))
        with self.assertRaisesRegex(GraphCompileError, "Required input"):
            graph.compile()

    def test_invalid_edge_endpoint(self):
        graph = RenderGraph("bad")
        graph.add_pass(DummyPass("A", outputs=("out",)))
        graph.add_pass(DummyPass("B", inputs=("input",), outputs=("out",)))
        graph.add_edge("A.out", "B.missing")
        with self.assertRaisesRegex(GraphCompileError, "does not input"):
            graph.compile()

    def test_builtin_graphs_compile(self):
        for name in ("slangpy_preview", "raster_forward", "dxr_pathtrace", "hybrid_debug"):
            graph = load_graph(name)
            self.assertTrue(graph.compile())

    def test_pass_reflection_accepts_resource_descriptors(self):
        reflection = PassReflection.from_iterables(
            outputs=("color",),
            resources={"color": ResourceDesc(kind="texture", format="rgba32_float", persistent=True)},
        )
        self.assertIn("color", reflection.outputs)
        self.assertTrue(reflection.resources["color"].persistent)

    def test_slang_function_pass_requires_fields(self):
        with self.assertRaisesRegex(ValueError, "missing fields"):
            pass_from_config({"name": "Broken", "type": "SlangFunctionPass", "module": "x.slang"})

    def test_compute_pass_alias_reflects_texture_output(self):
        for pass_type in ("ComputeFunctionPass", "ComputePass"):
            render_pass = pass_from_config(
                {
                    "name": "Composite",
                    "type": pass_type,
                    "module": "hybrid_composite.slang",
                    "function": "hybrid_composite",
                    "inputs": ["raster", "ray"],
                    "bindings": {"pixel": {"generator": "call_id"}},
                    "result": "color",
                    "output_format": "rgba16_float",
                }
            )
            reflection = render_pass.reflect()
            self.assertIsInstance(render_pass, ComputeFunctionPass)
            self.assertIn("color", reflection.outputs)
            self.assertEqual(reflection.resources["color"].format, "rgba16_float")

    def test_hardware_raster_config_accepts_pipeline_state_and_draw_list(self):
        render_pass = pass_from_config(
            {
                "name": "RasterSelection",
                "type": "HardwareRasterPass",
                "shader": "raster_forward.slang",
                "targets": [
                    {
                        "name": "visibility",
                        "format": "rgba16_float",
                        "clear_value": [0.2, 0.0, 0.0, 1.0],
                    }
                ],
                "rasterizer": {"cull_mode": "none"},
                "depth_stencil": {
                    "depth_test_enable": False,
                    "depth_write_enable": False,
                    "depth_func": "always",
                },
                "draw_list": {"meshes": ["SelectedMesh"], "materials": ["SelectedMaterial"]},
            }
        )
        reflection = render_pass.reflect()
        self.assertIsInstance(render_pass, HardwareRasterPass)
        self.assertEqual(render_pass.pipeline_desc.output_names, ("visibility",))
        self.assertEqual(render_pass.pipeline_desc.targets[0].format, "rgba16_float")
        self.assertEqual(render_pass.pipeline_desc.rasterizer.cull_mode, "none")
        self.assertFalse(render_pass.pipeline_desc.depth_stencil.depth_test_enable)
        self.assertEqual(render_pass.draw_list.meshes, ("SelectedMesh",))
        self.assertIn("visibility", reflection.outputs)
        self.assertEqual(reflection.resources["visibility"].format, "rgba16_float")


if __name__ == "__main__":
    unittest.main()
