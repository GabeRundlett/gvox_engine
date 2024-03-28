#include "sky.inl"
#include "sky.glsl"
#include <utilities/gpu/math.glsl>

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

layout(local_size_x = 2, local_size_y = 2, local_size_z = 32) in;
void main() {
    const uvec2 wg_base_pix_pos = gl_WorkGroupID.xy * uvec2(2, 2);
    const uvec2 sg_pix_pos = wg_base_pix_pos + uvec2(gl_SubgroupID % 2, gl_SubgroupID / 2);
    const uvec3 px = uvec3(sg_pix_pos, gl_WorkGroupID.z);
    uint face = gl_WorkGroupID.z;
    vec2 uv = (px.xy + 0.5) / IBL_CUBE_RES;

    vec3 output_dir = normalize(CUBE_MAP_FACE_ROTATION(face) * vec3(uv * 2 - 1, -1.0));
    const mat3 basis = build_orthonormal_basis(output_dir);

    vec3 result = vec3(0);
    // TODO: Now that we sample the atmosphere directly, computing this IBL is really slow.
    // We should cache the IBL cubemap, and only re-render it when necessary.

    // NOTE(grundlett): IF using GI, we should just slightly blur the sky into the IBL cube.
    //   This is because we use it for direct radiance lookup.
    // Otherwise, we want to fully convolve and cosine weight the IBL, for use as look-up in
    //   our simple lighting model.
    if ((push.flags & 1) == 1) {
        const uint sample_count = 128;
        const uint subgroup_size = 32;
        const uint iter_count = sample_count / subgroup_size;

        for (uint i = 0; i < iter_count; ++i) {
            const uint sample_id = i * subgroup_size + gl_SubgroupInvocationID;
            vec2 urand = hammersley(sample_id, sample_count);
            vec3 input_dir = basis * uniform_sample_cone(urand, 0.99);
            vec3 radiance_sample = sky_radiance_in_direction(gpu_input, sky_lut, transmittance_lut, input_dir);
            result += subgroupInclusiveAdd(radiance_sample);
        }
        result = result / sample_count;
    } else {
        const uint sample_count = 512;
        const uint subgroup_size = 32;
        const uint iter_count = sample_count / subgroup_size;

        for (uint i = 0; i < iter_count; ++i) {
            const uint sample_id = i * subgroup_size + gl_SubgroupInvocationID;
            vec2 urand = hammersley(sample_id, sample_count);
            vec3 input_dir = basis * uniform_sample_cone(urand, 0.00);
            vec3 radiance_sample = sky_radiance_in_direction(gpu_input, sky_lut, transmittance_lut, input_dir);
            radiance_sample *= dot(output_dir, input_dir);
            result += subgroupInclusiveAdd(radiance_sample);
        }
        result = result / sample_count;
    }
    if (gl_SubgroupInvocationID == 31) {
        imageStore(daxa_image2DArray(ibl_cube), ivec3(px), vec4(result, 1.0));
    }
}
