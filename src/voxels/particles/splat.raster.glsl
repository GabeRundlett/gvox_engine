#include "voxel_particles.inl"

#include <utilities/gpu/defs.glsl>
#include <renderer/kajiya/inc/camera.glsl>

DAXA_DECL_PUSH_CONSTANT(SplatParticleRasterPush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(ParticleVertex) splat_rendered_particle_verts = push.uses.splat_rendered_particle_verts;
daxa_ImageViewIndex render_image = push.uses.render_image;
daxa_ImageViewIndex depth_image_id = push.uses.depth_image_id;

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

#include "particle.glsl"

layout(location = 0) out vec3 center_ws;
layout(location = 1) out uint id;

void main() {
    uint particle_index = gl_VertexIndex;

    ParticleVertex particle = deref(advance(splat_rendered_particle_verts, particle_index));

    float voxel_radius = 1023.0 / 1024.0 * VOXEL_SIZE * 0.5;
    center_ws = get_particle_worldspace_origin(gpu_input, particle.pos);
    mat4 world_to_sample = deref(gpu_input).player.cam.view_to_sample * deref(gpu_input).player.cam.world_to_view;
    vec2 half_screen_size = vec2(deref(gpu_input).frame_dim) * 0.5;
    float ps_size = 0.0;
    vec4 temp_cs_pos = vec4(0, 0, 0, 1);
    particle_point_pos_and_size(center_ws, voxel_radius, world_to_sample, half_screen_size, temp_cs_pos, ps_size);

    vec4 vs_pos = deref(gpu_input).player.cam.world_to_view * vec4(center_ws, 1);
    vec4 cs_pos = deref(gpu_input).player.cam.view_to_sample * vs_pos;
    cs_pos.y *= -1;

    gl_Position = cs_pos;
    gl_PointSize = ps_size;

    id = particle.id;
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT

layout(location = 0) flat in vec3 center_ws;
layout(location = 1) flat in uint id;
layout(location = 0) out uvec4 color;

struct Box {
    vec3 center;
    vec3 radius;
    vec3 invRadius;
};
struct Ray {
    vec3 origin;
    vec3 direction;
};

float max(vec3 v) { return max(max(v.x, v.y), v.z); }
bool ourIntersectBox(Box box, Ray ray, out float distance, out vec3 normal,
                     const bool canStartInBox, in vec3 _invRayDir) {
    ray.origin = ray.origin - box.center;
    float winding = canStartInBox && (max(abs(ray.origin) * box.invRadius) < 1.0) ? -1 : 1;
    vec3 sgn = -sign(ray.direction);
    // Distance to plane
    vec3 d = box.radius * winding * sgn - ray.origin;
    d *= _invRayDir;
#define TEST(U, VW) (d.U >= 0.0) && all(lessThan(abs(ray.origin.VW + ray.direction.VW * d.U), box.radius.VW))
    bvec3 test = bvec3(TEST(x, yz), TEST(y, zx), TEST(z, xy));
    sgn = test.x ? vec3(sgn.x, 0, 0) : (test.y ? vec3(0, sgn.y, 0) : vec3(0, 0, test.z ? sgn.z : 0));
#undef TEST
    distance = (sgn.x != 0) ? d.x : ((sgn.y != 0) ? d.y : d.z);
    normal = sgn;
    return (sgn.x != 0) || (sgn.y != 0) || (sgn.z != 0);
}

void main() {
    Box box;
    box.radius = vec3(1023.0 / 1024.0 * VOXEL_SIZE * 0.5);
    box.invRadius = 1.0 / box.radius;
    box.center = center_ws;

    vec4 output_tex_size = vec4(deref(gpu_input).frame_dim, 0, 0);
    output_tex_size.zw = vec2(1.0, 1.0) / output_tex_size.xy;
    vec2 uv = get_uv(gl_FragCoord.xy, output_tex_size);

    ViewRayContext vrc = vrc_from_uv(gpu_input, uv);

    Ray ray;
    ray.origin = ray_origin_ws(vrc);
    ray.direction = ray_dir_ws(vrc);

    float dist;
    vec3 nrm;

    if (!ourIntersectBox(box, ray, dist, nrm, false, 1.0 / ray.direction)) {
        discard;
    }

    vec3 ws_pos = ray.origin + ray.direction * dist;
    vec4 vs_pos = deref(gpu_input).player.cam.world_to_view * vec4(ws_pos, 1);
    vec4 cs_pos = deref(gpu_input).player.cam.view_to_sample * vs_pos;
    float ndc_depth = cs_pos.z / cs_pos.w;

    gl_FragDepth = ndc_depth;
    color = uvec4(id + 1, 0, 0, 0);
}

#endif
