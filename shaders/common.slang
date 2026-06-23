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

struct Light
{
    int kind;
    float3 color;
    float intensity;
    float3 direction;
    float3 position;
};

struct StandardMaterial
{
    float4 baseColor;
    float roughness;
    float metallic;
};

struct SceneView
{
    uint meshCount;
    uint instanceCount;
    uint materialCount;
    uint lightCount;
    Camera camera;
    float3 boundsMin;
    float3 boundsMax;
};
