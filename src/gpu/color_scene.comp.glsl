#include <shared/shared.inl>

#include <utils/trace.glsl>
#include <utils/gvox_model.glsl>

DAXA_USE_PUSH_CONSTANT(ColorSceneComputePush)

#define INPUT deref(daxa_push_constant.gpu_input)
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

    f32vec3 light_del = -f32vec3(1, -2, 3);
    f32 light_dist = 20;
    f32vec3 light_col = f32vec3(1, 1, 1) * 500;

    f32vec3 col;

    if (dot(hit_pos, hit_pos) < MAX_SD * 10) {
        // col = hit_nrm;
        f32vec4 sample_col = uint_to_float4(sample_gvox_palette_voxel(daxa_push_constant.gpu_gvox_model, hit_pos, 0));
        col = sample_col.rgb;
        // col = col * light_col * max(dot(hit_nrm, normalize(light_del)) * 0.5 + 0.5, 0) / (light_dist * light_dist);
    } else {
        col = f32vec3(0.1, 0.12, 0.8);
    }

    imageStore(
        daxa_push_constant.render_col_image_id,
        i32vec2(pixel_i),
        f32vec4(col, 0));
}
#undef INPUT
