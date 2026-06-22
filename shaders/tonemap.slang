// SPDX-License-Identifier: Apache-2.0
import "slangpy";

struct FrameConstants
{
    uint frameIndex;
    float2 resolution;
    float timeSeconds;
};

float4 tonemap(int2 pixel, Texture2D hdr, SamplerState samplerState, FrameConstants frame)
{
    if (pixel.x >= int(frame.resolution.x) || pixel.y >= int(frame.resolution.y))
    {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    float4 color = hdr.Load(int3(pixel, 0));
    color.rgb = color.rgb / (1.0 + color.rgb);
    return float4(pow(max(color.rgb, 0.0), float3(1.0 / 2.2)), color.a);
}
