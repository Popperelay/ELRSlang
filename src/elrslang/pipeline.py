"""Base classes for low-level graphics and ray tracing passes."""

from __future__ import annotations

from .render_graph import RenderPass


class FeatureUnavailable(RuntimeError):
    pass


class PipelinePass(RenderPass):
    pass
