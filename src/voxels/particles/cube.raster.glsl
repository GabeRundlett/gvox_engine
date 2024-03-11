#include "voxel_particles.inl"

#include <utilities/gpu/defs.glsl>

DAXA_DECL_PUSH_CONSTANT(CubeParticleRasterPush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(ParticleVertex) cube_rendered_particle_verts = push.uses.cube_rendered_particle_verts;
daxa_ImageViewIndex render_image = push.uses.render_image;
daxa_ImageViewIndex depth_image_id = push.uses.depth_image_id;

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

#include "particle.glsl"
#include <renderer/kajiya/inc/camera.glsl>

#if !defined(SHADOW_MAP)
layout(location = 1) out uint id;
#endif

void main() {
    uint particle_index = gl_InstanceIndex;

    ParticleVertex particle = deref(advance(cube_rendered_particle_verts, particle_index));

    const vec3 diff = vec3(1023.0 / 1024.0 * VOXEL_SIZE);
    vec3 center_ws = get_particle_worldspace_origin(gpu_input, particle.pos);
    const vec3 camera_position = deref(gpu_input).player.pos;

    // extracting the vertex offset relative to the center.
    // Thanks to @cantaslaus on Discord.
    vec3 sign_ = vec3(ivec3(greaterThan(camera_position, center_ws)) ^ ((ivec3(0x1C, 0x46, 0x70) >> gl_VertexIndex) & ivec3(1))) - 0.5;
    vec3 vert_pos = sign_ * diff + center_ws;
    // ---------------------

#if defined(SHADOW_MAP)
    vec4 cs_pos = deref(gpu_input).ws_to_shadow * vec4(vert_pos, 1);
#else
    vec4 vs_pos = deref(gpu_input).player.cam.world_to_view * vec4(vert_pos, 1);
    vec4 cs_pos = deref(gpu_input).player.cam.view_to_sample * vs_pos;
    id = particle_index; // particle.id;
#endif

    // TODO: Figure out why raster is projected upside down
    cs_pos.y *= -1;
    gl_Position = cs_pos;
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT

#if defined(SHADOW_MAP)
void main() {
}
#else
layout(location = 1) flat in uint id;
layout(location = 0) out uvec4 color;

void main() {
    color = uvec4(id + 1, 0, 0, 0);
}
#endif

#endif
