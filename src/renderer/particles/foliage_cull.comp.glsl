#include "render_foliage.inl"
#include <culling.glsl>

DAXA_DECL_PUSH_CONSTANT(FoliageCullPush, push)

struct FrustumPlanes {
    vec3 l_nrm;
    vec3 r_nrm;
    vec3 t_nrm;
    vec3 b_nrm;
    vec3 l_origin;
    vec3 r_origin;
    vec3 t_origin;
    vec3 b_origin;
};

FrustumPlanes get_frustum_planes() {
    FrustumPlanes result;

    mat4 clip_to_world = deref(push.uses.gpu_input).player.cam.view_to_world * deref(push.uses.gpu_input).player.cam.sample_to_view;

    {
        vec4 fp0_h = clip_to_world * vec4(-1, -1, 1, 1);
        vec4 fp1_h = clip_to_world * vec4(-1, -1, 0.001, 1);
        vec4 fp2_h = clip_to_world * vec4(-1, +1, 0.001, 1);
        vec3 fp0 = fp0_h.xyz / fp0_h.w;
        vec3 fp1 = fp1_h.xyz / fp1_h.w;
        vec3 fp2 = fp2_h.xyz / fp2_h.w;
        result.l_origin = fp0;
        result.l_nrm = cross(fp1 - fp0, fp2 - fp0);
    }

    {
        vec4 fp0_h = clip_to_world * vec4(+1, -1, 1, 1);
        vec4 fp1_h = clip_to_world * vec4(+1, +1, 0.001, 1);
        vec4 fp2_h = clip_to_world * vec4(+1, -1, 0.001, 1);
        vec3 fp0 = fp0_h.xyz / fp0_h.w;
        vec3 fp1 = fp1_h.xyz / fp1_h.w;
        vec3 fp2 = fp2_h.xyz / fp2_h.w;
        result.r_origin = fp0;
        result.r_nrm = cross(fp1 - fp0, fp2 - fp0);
    }

    {
        vec4 fp0_h = clip_to_world * vec4(-1, -1, 1, 1);
        vec4 fp1_h = clip_to_world * vec4(+1, -1, 0.001, 1);
        vec4 fp2_h = clip_to_world * vec4(-1, -1, 0.001, 1);
        vec3 fp0 = fp0_h.xyz / fp0_h.w;
        vec3 fp1 = fp1_h.xyz / fp1_h.w;
        vec3 fp2 = fp2_h.xyz / fp2_h.w;
        result.t_origin = fp0;
        result.t_nrm = cross(fp1 - fp0, fp2 - fp0);
    }

    {
        vec4 fp0_h = clip_to_world * vec4(-1, +1, 1, 1);
        vec4 fp1_h = clip_to_world * vec4(-1, +1, 0.001, 1);
        vec4 fp2_h = clip_to_world * vec4(+1, +1, 0.001, 1);
        vec3 fp0 = fp0_h.xyz / fp0_h.w;
        vec3 fp1 = fp1_h.xyz / fp1_h.w;
        vec3 fp2 = fp2_h.xyz / fp2_h.w;
        result.b_origin = fp0;
        result.b_nrm = cross(fp1 - fp0, fp2 - fp0);
    }

    return result;
}

bool is_aabb_visible(FrustumPlanes frustum, Aabb aabb) {
    vec3 p0 = aabb.min;
    vec3 p1 = aabb.max;

    vec3 vertices[8] = vec3[8](
        vec3(p0.x, p0.y, p0.z),
        vec3(p1.x, p0.y, p0.z),
        vec3(p0.x, p1.y, p0.z),
        vec3(p1.x, p1.y, p0.z),
        vec3(p0.x, p0.y, p1.z),
        vec3(p1.x, p0.y, p1.z),
        vec3(p0.x, p1.y, p1.z),
        vec3(p1.x, p1.y, p1.z));

    vec3 ndc_min;
    vec3 ndc_max;

    bool frustum_l_outside = true;
    bool frustum_r_outside = true;
    bool frustum_t_outside = true;
    bool frustum_b_outside = true;

    [[unroll]] for (uint vert_i = 0; vert_i < 8; ++vert_i) {
        vec4 vs_h = deref(push.uses.gpu_input).player.cam.world_to_view * vec4(vertices[vert_i], 1);
        vec4 cs_h = deref(push.uses.gpu_input).player.cam.view_to_sample * vs_h;
        vec3 p = cs_h.xyz / cs_h.w;
        if (vert_i == 0) {
            ndc_min = p;
            ndc_max = p;
        } else {
            ndc_min = min(ndc_min, p);
            ndc_max = max(ndc_max, p);
        }

        frustum_l_outside = frustum_l_outside && (dot(vertices[vert_i] - frustum.l_origin, frustum.l_nrm) > 0);
        frustum_r_outside = frustum_r_outside && (dot(vertices[vert_i] - frustum.r_origin, frustum.r_nrm) > 0);
        frustum_t_outside = frustum_t_outside && (dot(vertices[vert_i] - frustum.t_origin, frustum.t_nrm) > 0);
        frustum_b_outside = frustum_b_outside && (dot(vertices[vert_i] - frustum.b_origin, frustum.b_nrm) > 0);
    }

    bool between_raster_grid_lines = is_between_raster_grid_lines(ndc_min.xy, ndc_max.xy, vec2(deref(push.uses.gpu_input).frame_dim));
    const bool depth_occluded = is_ndc_aabb_hiz_depth_occluded(ndc_min, ndc_max, deref(push.uses.gpu_input).frame_dim, deref(push.uses.gpu_input).next_lower_po2_render_size, push.uses.hiz);

    bool inside_frustum = !(frustum_l_outside || frustum_r_outside || frustum_t_outside || frustum_b_outside);

    return inside_frustum && !depth_occluded && !between_raster_grid_lines;
}

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint object_index = gl_WorkGroupID.x;
    daxa_BufferPtr(GpuVoxelObject) voxel_object_ptr = advance(push.uses.voxel_object_manifests, object_index);
    uint brick_n = deref(voxel_object_ptr).brick_count;
    Aabb voxel_object_aabb = Aabb(deref(voxel_object_ptr).aabb_min, deref(voxel_object_ptr).aabb_max);

    FrustumPlanes frustum = get_frustum_planes();

    // if (!is_aabb_visible(frustum, voxel_object_aabb) || as_address(deref(voxel_object_ptr).brick_foliage) == 0)
    if (as_address(deref(voxel_object_ptr).brick_foliage) == 0)
        return;

    for (uint brick_index = gl_LocalInvocationIndex; brick_index < brick_n; brick_index += 128) {
        Aabb brick_aabb = deref(advance(deref(voxel_object_ptr).brick_aabbs, brick_index));
        brick_aabb.min *= deref(voxel_object_ptr).scale;
        brick_aabb.max *= deref(voxel_object_ptr).scale;
        brick_aabb.min += deref(voxel_object_ptr).pos;
        brick_aabb.max += deref(voxel_object_ptr).pos;
        if (is_aabb_visible(frustum, brick_aabb)) {
            // Add to draw list
            daxa_RWBufferPtr(daxa_u32) counter = daxa_RWBufferPtr(daxa_u32)(as_address(push.uses.visible_foliage_bricks));
            uint slot = atomicAdd(deref(counter), 1u);
            deref(advance(push.uses.visible_foliage_bricks, 1u + slot)) = FoliageBrickInstance(voxel_object_ptr, brick_index, 0);
        }
    }
}
