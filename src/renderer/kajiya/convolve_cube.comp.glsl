#include <renderer/kajiya/convolve_cube.inl>
#include <utilities/gpu/math.glsl>
#include <renderer/atmosphere/sky.glsl>

DAXA_DECL_PUSH_CONSTANT(ConvolveCubeComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex sky_lut = push.uses.sky_lut;
daxa_ImageViewIndex transmittance_lut = push.uses.transmittance_lut;
daxa_ImageViewIndex ibl_cube = push.uses.ibl_cube;

float radical_inverse_vdc(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 hammersley(uint i, uint n) {
    return vec2(float(i + 1) / n, radical_inverse_vdc(i + 1));
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    uvec3 px = gl_GlobalInvocationID.xyz;
    uint face = px.z;
    vec2 uv = (px.xy + 0.5) / IBL_CUBE_RES;

    vec3 output_dir = normalize(CUBE_MAP_FACE_ROTATION(face) * vec3(uv * 2 - 1, -1.0));
    const mat3 basis = build_orthonormal_basis(output_dir);

    const uint sample_count = 128;

    uint rng = hash2(px.xy);

    vec3 result = vec3(0);
    for (uint i = 0; i < sample_count; ++i) {
        vec2 urand = hammersley(i, sample_count);
        vec3 input_dir = basis * uniform_sample_cone(urand, 0.99);
        // TODO: Now that we sample the atmosphere directly, computing this IBL is really slow.
        // We should cache the IBL cubemap, and only re-render it when necessary.
        result += get_atmosphere_lighting(gpu_input, sky_lut, transmittance_lut, input_dir);
    }

    imageStore(daxa_image2DArray(ibl_cube), ivec3(px), vec4(result / sample_count, 1.0));
}
