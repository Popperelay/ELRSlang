// SPDX-License-Identifier: Apache-2.0

struct V2F
{
    float4 pos : SV_Position;
    float4 color : COLOR0;
};

[shader("vertex")]
V2F vertex_main(float3 pos : POSITION, float4 color : COLOR)
{
    V2F output;
    output.pos = float4(pos, 1.0);
    output.color = color;
    return output;
}

[shader("fragment")]
float4 fragment_main(V2F input) : SV_Target
{
    return input.color;
}
