// SPDX-License-Identifier: Apache-2.0
import "slangpy";

struct FrameConstants
{
    uint frameIndex;
    float2 resolution;
    float timeSeconds;
};

float4 hybrid_composite(
    int2 pixel,
    Texture2D<float4> rasterColor,
    Texture2D<float4> rayColor,
    SamplerState samplerState,
    FrameConstants frame)
{
    float2 uv = (float2(pixel) + 0.5) / max(frame.resolution, float2(1.0, 1.0));
    float4 raster = rasterColor.SampleLevel(samplerState, uv, 0.0);
    float4 ray = rayColor.SampleLevel(samplerState, uv, 0.0);
    return float4(lerp(raster.rgb, ray.rgb, 0.35), 1.0);
}
