#include <renderer/kajiya/calculate_reprojection_map.inl>
#include "inc/camera.glsl"
#include "inc/bilinear.glsl"
#include "inc/safety.glsl"
#include <g_samplers>

DAXA_DECL_PUSH_CONSTANT(CalculateReprojectionMapComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex vs_normal_image_id = push.uses.vs_normal_image_id;
daxa_ImageViewIndex depth_image_id = push.uses.depth_image_id;
daxa_ImageViewIndex prev_depth_image_id = push.uses.prev_depth_image_id;
daxa_ImageViewIndex velocity_image_id = push.uses.velocity_image_id;
daxa_ImageViewIndex dst_image_id = push.uses.dst_image_id;

float fetch_depth(uvec2 px) {
    return safeTexelFetch(depth_image_id, ivec2(px), 0).r;
}
float fetch_prev_depth(uvec2 px) {
    return safeTexelFetch(prev_depth_image_id, ivec2(px), 0).r;
}
vec4 fetch_vel(uvec2 px) {
    return safeTexelFetch(velocity_image_id, ivec2(px), 0);
}
vec3 fetch_nrm(uvec2 px) {
    return safeTexelFetch(vs_normal_image_id, ivec2(px), 0).xyz;
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
    vec4 output_tex_size;
    output_tex_size.xy = deref(gpu_input).frame_dim;
    output_tex_size.zw = vec2(1.0, 1.0) / output_tex_size.xy;

    vec2 uv = get_uv(px, output_tex_size);

    float depth = fetch_depth(px);

    if (depth == 0.0) {
        vec4 pos_cs = vec4(uv_to_cs(uv), 0.0, 1.0);
        vec4 pos_vs = deref(gpu_input).player.cam.clip_to_view * pos_cs;

        vec4 prev_vs = pos_vs;

        vec4 prev_cs = deref(gpu_input).player.cam.view_to_clip * prev_vs;
        vec4 prev_pcs = deref(gpu_input).player.cam.clip_to_prev_clip * prev_cs;

        vec2 prev_uv = cs_to_uv(prev_pcs.xy);
        vec2 uv_diff = prev_uv - uv;

        safeImageStore(dst_image_id, ivec2(px), vec4(uv_diff, 0, 0));
        return;
    }

    vec3 eye_pos = (deref(gpu_input).player.cam.view_to_world * vec4(0, 0, 0, 1)).xyz;

    vec3 normal_vs = fetch_nrm(px) * 2.0 - 1.0;
    vec3 normal_pvs = (deref(gpu_input).player.cam.prev_clip_to_prev_view *
                       (deref(gpu_input).player.cam.clip_to_prev_clip *
                        (deref(gpu_input).player.cam.view_to_clip * vec4(normal_vs, 0))))
                          .xyz;

    vec4 pos_cs = vec4(uv_to_cs(uv), depth, 1.0);
    vec4 pos_vs = (deref(gpu_input).player.cam.clip_to_view * pos_cs);
    float dist_to_point = -(pos_vs.z / pos_vs.w);

    vec4 prev_vs = pos_vs / pos_vs.w;
    prev_vs.xyz += vec4(fetch_vel(px).xyz, 0).xyz;

    // vec4 prev_cs = mul(deref(gpu_input).player.cam.prev_view_to_prev_clip, prev_vs);
    vec4 prev_cs = (deref(gpu_input).player.cam.view_to_clip * prev_vs);
    vec4 prev_pcs = (deref(gpu_input).player.cam.clip_to_prev_clip * prev_cs);

    vec2 prev_uv = cs_to_uv(prev_pcs.xy / prev_pcs.w);
    vec2 uv_diff = prev_uv - uv;

    // Account for quantization of the `uv_diff` in R16G16B16A16_SNORM.
    // This is so we calculate validity masks for pixels that the users will actually be using.
    uv_diff = floor(uv_diff * 32767.0 + 0.5) / 32767.0;
    prev_uv = uv + uv_diff;

    vec4 prev_pvs = (deref(gpu_input).player.cam.prev_clip_to_prev_view * prev_pcs);
    prev_pvs /= prev_pvs.w;

    // Based on "Fast Denoising with Self Stabilizing Recurrent Blurs"

    float plane_dist_prev = dot(normal_vs, prev_pvs.xyz);

    // Note: departure from the quoted technique: they calculate reprojected sample depth by linearly
    // scaling plane distance with view-space Z, which is not correct unless the plane is aligned with view.
    // Instead, the amount that distance actually increases with depth is simply `normal_vs.z`.

    // Note: bias the minimum distance increase, so that reprojection at grazing angles has a sharper cutoff.
    // This can introduce shimmering a grazing angles, but also reduces reprojection artifacts on surfaces
    // which flip their normal from back- to fron-facing across a frame. Such reprojection smears a few
    // pixels along a wide area, creating a glitchy look.
    float plane_dist_prev_dz = min(-0.2, normal_vs.z);
    // float plane_dist_prev_dz = -normal_vs.z;

    const Bilinear bilinear_at_prev = get_bilinear_filter(prev_uv, deref(gpu_input).frame_dim.xy);
    vec2 prev_gather_uv = (bilinear_at_prev.origin + 1.0) / deref(gpu_input).rounded_frame_dim.xy;
    vec4 prev_depth = textureGather(daxa_sampler2D(prev_depth_image_id, g_sampler_nnc), prev_gather_uv).wzxy;

    // vec4 prev_view_z = rcp(prev_depth * -deref(gpu_input).player.cam.prev_clip_to_prev_view._43);
    vec4 prev_view_z = 1.0 / (prev_depth * -deref(gpu_input).player.cam.prev_clip_to_prev_view[2][3]);

    // Note: departure from the quoted technique: linear offset from zero distance at previous position instead of scaling.
    vec4 quad_dists = abs(plane_dist_prev_dz * (prev_view_z - prev_pvs.z));

    // TODO: reject based on normal too? Potentially tricky under rotations.

    const float acceptance_threshold = 1.5 * output_tex_size.w;

    // Reduce strictness at grazing angles, where distances grow due to perspective
    const vec3 pos_vs_norm = normalize(pos_vs.xyz / pos_vs.w);
    const float ndotv = dot(normal_vs, pos_vs_norm);
    const float prev_ndotv = dot(normal_pvs, normalize(prev_pvs.xyz));
    const float acceptable_dist = acceptance_threshold * dist_to_point / -ndotv;

    vec4 quad_validity = step(quad_dists, vec4((acceptable_dist)));

    ivec2 p_x0 = px0(bilinear_at_prev);
    ivec2 p_x1 = px1(bilinear_at_prev);
    ivec2 p_x2 = px2(bilinear_at_prev);
    ivec2 p_x3 = px3(bilinear_at_prev);

    quad_validity.x *= float((p_x0.x >= 0 && p_x0.y >= 0) && (p_x0.x < output_tex_size.x && p_x0.y < output_tex_size.y));
    quad_validity.y *= float((p_x1.x >= 0 && p_x1.y >= 0) && (p_x1.x < output_tex_size.x && p_x1.y < output_tex_size.y));
    quad_validity.z *= float((p_x2.x >= 0 && p_x2.y >= 0) && (p_x2.x < output_tex_size.x && p_x2.y < output_tex_size.y));
    quad_validity.w *= float((p_x3.x >= 0 && p_x3.y >= 0) && (p_x3.x < output_tex_size.x && p_x3.y < output_tex_size.y));

    float validity = dot(quad_validity, vec4(1, 2, 4, 8)) / 15.0;
    float accuracy = 1;

    // Reprojection of surfaces which were grazing to the camera
    // causes any noise or features on those surfaces to become smeared,
    // which subsequently is slow to converge.
    accuracy *= smoothstep(0.8, 0.95, prev_ndotv / ndotv);

    // Mark off-screen reprojections
    if (any(bvec2(clamp(prev_uv, 0, 1) != prev_uv))) {
        accuracy = -1;
    }

    safeImageStore(dst_image_id, ivec2(px), vec4(uv_diff, validity, accuracy));
}
