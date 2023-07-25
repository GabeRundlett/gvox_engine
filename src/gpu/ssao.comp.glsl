#include <shared/shared.inl>

#include <utils/trace.glsl>
#include <utils/downscale.glsl>

f32vec4 output_tex_size;
u32vec2 px;
f32vec3 nrm;
f32vec2 uv;
f32 depth;

float calc_ao() {
    rand_seed(u32(((gl_GlobalInvocationID.x * 123 ^ gl_GlobalInvocationID.y * 567) + deref(gpu_input).frame_index)));
    f32vec3 nrm_viewspace = normalize((vec4(nrm, 0) * deref(globals).player.cam.world_to_view).xyz);

    f32 ao = 0.0;
    const u32 SAMPLE_N = 4;
    const f32 bias = 0.01;
    const f32 radius = 10.0;

    ViewRayContext vrc = vrc_from_uv(globals, uv);
    f32vec3 view_dir = ray_dir_ws(vrc);
    f32vec3 view_pos = ray_origin_ws(vrc);

    vec3 frag_pos = view_dir * depth + view_pos;

    for (u32 i = 0; i < SAMPLE_N; ++i) {
        f32 rng = rand();
        f32vec3 ao_sample = rand_hemi_dir(nrm_viewspace) * mix(0.1, 1.0, rng * rng) * radius;
        // f32mat4x4 vp_mat = deref(globals).player.cam.view_to_clip * deref(globals).player.cam.world_to_view;
        // f32vec4 sample_tex_coord = vp_mat * f32vec4(ao_sample + frag_pos, 1.0);
        // sample_tex_coord.xyz = (sample_tex_coord.xyz * 0.5 / sample_tex_coord.w) * vec3(1.0) + 0.5;
        f32vec2 sample_tex_coord = f32vec2(uv) + ao_sample.xy * 0.01;
        f32 sample_depth = texelFetch(daxa_texture2D(depth_image), i32vec2(sample_tex_coord.xy * output_tex_size.xy / SHADING_SCL), 0).r;
        // NOTE: Probably want to set this to 0.0 whe doing GI. This just looks nicer for now.
        f32 dist_fac = smoothstep(0.0, 1.0, (depth - sample_depth) * 0.1 / radius);
        ao += (depth + ao_sample.z >= sample_depth + bias ? dist_fac : 1.0);
    }

    return ao / SAMPLE_N;
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    px = gl_GlobalInvocationID.xy;
    output_tex_size.xy = deref(gpu_input).frame_dim;
    output_tex_size.zw = f32vec2(1.0, 1.0) / output_tex_size.xy;
    u32vec2 offset = get_downscale_offset(gpu_input);
    uv = get_uv(px * SHADING_SCL + offset, output_tex_size);

    depth = texelFetch(daxa_texture2D(depth_image), i32vec2(px), 0).r;
    u32vec4 g_buffer_value = texelFetch(daxa_utexture2D(g_buffer_image_id), i32vec2(px * SHADING_SCL + offset), 0);
    nrm = u16_to_nrm(g_buffer_value.y);

    if (depth == MAX_DIST || dot(nrm, nrm) == 0.0) {
        imageStore(daxa_image2D(ssao_image_id), i32vec2(px), f32vec4(1, 0, 0, 0));
        return;
    }

    float ao = calc_ao();

    imageStore(daxa_image2D(ssao_image_id), i32vec2(px), f32vec4(ao, 0, 0, 0));
}
