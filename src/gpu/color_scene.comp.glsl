#include <shared/shared.inl>

#include <utils/trace.glsl>
#include <utils/voxels.glsl>

DAXA_USE_PUSH_CONSTANT(ColorSceneComputePush)

#define AMBIENT_OCCLUSION 0

#define SETTINGS deref(daxa_push_constant.gpu_settings)
#define INPUT deref(daxa_push_constant.gpu_input)
#define CHUNK_PTRS(i) daxa_push_constant.voxel_chunks[i]
#define CHUNKS(i) deref(daxa_push_constant.voxel_chunks[i])
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 pixel_i = gl_GlobalInvocationID.xy;
    if (pixel_i.x >= INPUT.frame_dim.x ||
        pixel_i.y >= INPUT.frame_dim.y)
        return;

    f32vec4 hit = imageLoad(
        daxa_push_constant.render_pos_image_id,
        i32vec2(pixel_i));

    f32vec3 hit_pos = hit.xyz;
    f32vec3 hit_nrm = scene_nrm(hit_pos);
    f32 f = step((int(hit_pos.x) + int(hit_pos.y) + int(hit_pos.z)) % 2, 0.5);

    f32vec3 light_del = f32vec3(1, -2, 3);
    f32 light_dist = 20;
    f32vec3 light_col = f32vec3(1, 1, 1) * 500;

    f32vec3 col;

    if (dot(hit_pos, hit_pos) < MAX_SD * 10) {
        u32vec3 chunk_n = u32vec3(1u << SETTINGS.log2_chunks_per_axis);
        u32vec3 chunk_i = u32vec3(floor(hit_pos * (f32(VOXEL_SCL) / CHUNK_SIZE)));
        u32 chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
        u32vec3 voxel_i = u32vec3(hit_pos * VOXEL_SCL);
        // u32vec3 inchunk_voxel_i = voxel_i - chunk_i * CHUNK_SIZE;
        // u32 lod = sample_lod(daxa_push_constant.gpu_gvox_model, CHUNK_PTRS(chunk_index), chunk_i, inchunk_voxel_i);
        // col = f32vec3(lod) / 7;

        // col = hit_nrm;
        f32vec4 sample_col = uint_to_float4(sample_gvox_palette_voxel(daxa_push_constant.gpu_gvox_model, voxel_i, 0));
        col = sample_col.rgb;
        // col = col * light_col * max(dot(hit_nrm, normalize(light_del)) * 0.5 + 0.5, 0) / (light_dist * light_dist);

#if AMBIENT_OCCLUSION
        f32vec3 ray_pos = hit_pos + hit_nrm * 0.5 / VOXEL_SCL;
        f32vec3 ray_dir = rand_lambertian_nrm(hit_nrm, fract(rand2(pixel_i) * INPUT.time * 100));
        trace_hierarchy_traversal(daxa_push_constant.gpu_gvox_model, daxa_push_constant.voxel_chunks, chunk_n, ray_pos, ray_dir, 16);
        if (dot(ray_pos, ray_pos) < MAX_SD * 10) {
            col *= 0.0;
        }
#endif
    } else {
        col = f32vec3(0.1, 0.12, 0.9);
    }

#if AMBIENT_OCCLUSION
    f32vec3 prev_col =
        imageLoad(
            daxa_push_constant.render_col_image_id,
            i32vec2(pixel_i))
            .rgb;
    f32 alpha = 0.01;
    imageStore(
        daxa_push_constant.render_col_image_id,
        i32vec2(pixel_i),
        f32vec4(col * alpha + prev_col * (1.0 - alpha), 0));
#else
    imageStore(
        daxa_push_constant.render_col_image_id,
        i32vec2(pixel_i),
        f32vec4(col, 0));
#endif
}
#undef CHUNKS
#undef INPUT
#undef SETTINGS
