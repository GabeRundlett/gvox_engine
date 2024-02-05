#include <shared/app.inl>
#include <utils/math.glsl>

DAXA_DECL_PUSH_CONSTANT(TestComputePush, push)
daxa_RWBufferPtr(daxa_u32) data = push.uses.data;

uint gen_voxel(ivec3 pos) {
    uint r = uint(0);
    uint g = uint(0);
    uint b = uint(0);
    float x = float(pos[0] - 32);
    float y = float(pos[1] - 32);
    r = uint((sin(x * 0.3) * 0.5 + 0.5) * 150.0);
    g = uint((sin(y * 0.5) * 0.5 + 0.5) * 150.0);
    b = uint((sin(sqrt(x * x + y * y)) * 60.5 + 0.5) * 250.0);
    return (r << 0x10) | (g << 0x08) | (b << 0x00);
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    const uint chunk_n = 8;
    uint v = gen_voxel(ivec3(gl_GlobalInvocationID.xyz));
    uvec3 chunk_i = uvec3(gl_GlobalInvocationID.xyz) >> 6;
    uvec3 in_chunk_i = uvec3(gl_GlobalInvocationID.xyz) & ((1 << 6) - 1);
    uint chunk_index = chunk_i.x + chunk_i.y * chunk_n + chunk_i.z * chunk_n * chunk_n;
    uint in_chunk_index = in_chunk_i.x + in_chunk_i.y * 64 + in_chunk_i.z * 64 * 64;
    uint out_index = chunk_index * 64 * 64 * 64 + in_chunk_index;
    deref(data[out_index]) = v;
}
