#include "voxel_particles.inl"

#include <utilities/gpu/defs.glsl>
#include <renderer/kajiya/inc/camera.glsl>

#if defined(GRASS)
DAXA_DECL_PUSH_CONSTANT(GrassStrandSplatParticleRasterPush, push)
daxa_BufferPtr(GrassStrand) grass_strands = push.uses.grass_strands;
#elif defined(SIM_PARTICLE)
DAXA_DECL_PUSH_CONSTANT(SimParticleSplatParticleRasterPush, push)
daxa_BufferPtr(SimulatedVoxelParticle) simulated_voxel_particles = push.uses.simulated_voxel_particles;
#endif
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(PackedParticleVertex) splat_rendered_particle_verts = push.uses.splat_rendered_particle_verts;

#include "grass/grass.glsl"
#include "sim_particle/sim_particle.glsl"

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

layout(location = 0) out uvec2 i_gbuffer_xy;
layout(location = 1) out vec3 i_vs_nrm;
layout(location = 2) out vec3 i_vs_velocity;
layout(location = 3) out vec3 center_ws;

void main() {
    uint particle_index = gl_VertexIndex;

    PackedParticleVertex packed_vertex = deref(advance(splat_rendered_particle_verts, particle_index));

#if defined(GRASS)
    ParticleVertex vert = get_grass_vertex(gpu_input, grass_strands, packed_vertex);
#elif defined(SIM_PARTICLE)
    ParticleVertex vert = get_sim_particle_vertex(gpu_input, simulated_voxel_particles, packed_vertex);
#endif

    float voxel_radius = VOXEL_SIZE * 0.5;
    center_ws = vert.pos;
    mat4 world_to_sample = deref(gpu_input).player.cam.view_to_sample * deref(gpu_input).player.cam.world_to_view;
    vec2 half_screen_size = vec2(deref(gpu_input).frame_dim) * 0.5;
    float ps_size = 0.0;
    vec2 px_pos = vec2(0, 0);
    particle_point_pos_and_size(center_ws, voxel_radius, world_to_sample, half_screen_size, px_pos, ps_size);

    vec4 vs_pos = (deref(gpu_input).player.cam.world_to_view * vec4(vert.pos, 1));
    vec4 prev_vs_pos = (deref(gpu_input).player.cam.world_to_view * vec4(vert.prev_pos, 1));
    i_vs_velocity = (prev_vs_pos.xyz / prev_vs_pos.w) - (vs_pos.xyz / vs_pos.w);
    Voxel voxel = unpack_voxel(vert.packed_voxel);
    vec3 nrm = voxel.normal;
    i_gbuffer_xy = uvec2(vert.packed_voxel.data, nrm_to_u16(nrm));
    i_vs_nrm = (deref(gpu_input).player.cam.world_to_view * vec4(nrm, 0)).xyz;

    vec4 vs_pos2 = deref(gpu_input).player.cam.world_to_view * vec4(center_ws, 1);
    vec4 cs_pos = deref(gpu_input).player.cam.view_to_sample * vs_pos2;
    cs_pos.y *= -1;

    gl_Position = cs_pos;
    gl_PointSize = ps_size;
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT

layout(location = 0) flat in uvec2 i_gbuffer_xy;
layout(location = 1) flat in vec3 i_vs_nrm;
layout(location = 2) flat in vec3 i_vs_velocity;
layout(location = 3) flat in vec3 center_ws;

layout(location = 0) out uvec4 o_gbuffer;
layout(location = 1) out vec4 o_vs_velocity;
layout(location = 2) out vec4 o_vs_nrm;

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
    box.radius = vec3(VOXEL_SIZE * 0.5);
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
    vec3 face_nrm;

    if (!ourIntersectBox(box, ray, dist, face_nrm, false, 1.0 / ray.direction)) {
        discard;
    }

    vec3 ws_pos = ray.origin + ray.direction * dist;
    vec4 vs_pos = deref(gpu_input).player.cam.world_to_view * vec4(ws_pos, 1);
    vec4 cs_pos = deref(gpu_input).player.cam.view_to_sample * vs_pos;
    float ndc_depth = cs_pos.z / cs_pos.w;

    gl_FragDepth = ndc_depth;
    o_gbuffer = uvec4(i_gbuffer_xy, floatBitsToUint(ndc_depth), 0);

#if !PER_VOXEL_NORMALS
    // TODO: Fix the face-normals for splatted particles. At a distance, this ourIntersectBox function appears to return mush.
    vec3 nrm = face_nrm;
    vec3 vs_nrm = normalize((deref(gpu_input).player.cam.world_to_view * vec4(nrm, 0)).xyz);
    o_gbuffer.y = nrm_to_u16(nrm);
#else
    vec3 vs_nrm = i_vs_nrm;
    vs_nrm *= -sign(dot(ray_dir_vs(vrc), vs_nrm));
#endif

    o_vs_velocity = vec4(i_vs_velocity, 0);
    o_vs_nrm = vec4(vs_nrm * 0.5 + 0.5, 0);
}

#endif
