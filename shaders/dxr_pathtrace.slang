// SPDX-License-Identifier: Apache-2.0
import "slangpy";

struct Payload
{
    float4 color;
};

uniform RWTexture2D<float4> rt_render_texture;
uniform RaytracingAccelerationStructure rt_tlas;
uniform float3 rt_camera_position;
uniform float3 rt_camera_forward;
uniform float3 rt_camera_right;
uniform float3 rt_camera_up;
uniform float rt_camera_vfov_degrees;
uniform float2 rt_resolution;

[shader("miss")]
void rt_miss(inout Payload payload)
{
    payload.color = float4(0.02, 0.025, 0.035, 1.0);
}

[shader("closesthit")]
void rt_closest_hit(inout Payload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    float3 bary = float3(
        1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
        attribs.barycentrics.x,
        attribs.barycentrics.y
    );
    payload.color = float4(0.25 + 0.75 * bary, 1.0);
}

[shader("raygeneration")]
void rt_ray_gen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    float2 resolution = max(rt_resolution, float2(1.0, 1.0));
    float2 uv = (float2(pixel) + 0.5) / resolution;
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;

    float aspect = resolution.x / resolution.y;
    float tanHalfFov = tan(radians(rt_camera_vfov_degrees) * 0.5);
    float3 rayDir = normalize(
        rt_camera_forward
        + rt_camera_right * (ndc.x * aspect * tanHalfFov)
        + rt_camera_up * (ndc.y * tanHalfFov)
    );

    RayDesc ray;
    ray.Origin = rt_camera_position;
    ray.Direction = rayDir;
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

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
