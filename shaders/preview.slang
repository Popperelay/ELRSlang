// SPDX-License-Identifier: Apache-2.0
import "slangpy";

struct FrameConstants
{
    uint frameIndex;
    float2 resolution;
    float timeSeconds;
};

struct Camera
{
    float3 position;
    float3 target;
    float3 up;
    float vfovDegrees;
};

struct SceneView
{
    uint meshCount;
    uint instanceCount;
    uint materialCount;
    uint lightCount;
    Camera camera;
};

float4 debug_view(int2 pixel, SceneView scene, FrameConstants frame)
{
    if (pixel.x >= int(frame.resolution.x) || pixel.y >= int(frame.resolution.y))
    {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    float2 uv = (float2(pixel) + 0.5) / max(frame.resolution, float2(1.0, 1.0));
    float meshTint = min(float(scene.meshCount) / 8.0, 1.0);
    float pulse = 0.5 + 0.5 * sin(frame.timeSeconds + float(frame.frameIndex) * 0.01);
    return float4(uv.x, uv.y * (0.5 + 0.5 * meshTint), 0.25 + 0.25 * pulse, 1.0);
}
