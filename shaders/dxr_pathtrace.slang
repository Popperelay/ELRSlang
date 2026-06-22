// SPDX-License-Identifier: Apache-2.0
import "slangpy";

struct Payload
{
    float4 color;
};

uniform RWTexture2D<float4> rt_render_texture;
uniform RaytracingAccelerationStructure rt_tlas;

[shader("miss")]
void rt_miss(inout Payload payload)
{
    payload.color = float4(0.0, 0.0, 0.0, 1.0);
}

[shader("closesthit")]
void rt_closest_hit(inout Payload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    payload.color = float4(
        1.0,
        attribs.barycentrics.x,
        attribs.barycentrics.y,
        1.0
    );
}

[shader("raygeneration")]
void rt_ray_gen()
{
    uint2 pixel = DispatchRaysIndex().xy;

    RayDesc ray;
    ray.Origin = float3(pixel.xy, 0.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 2.0;

    Payload payload = {};
    TraceRay(
        rt_tlas,
        0,
        0xff,
        0,
        0,
        0,
        ray,
        payload
    );
    rt_render_texture[pixel] = payload.color;
}
