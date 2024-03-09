#include "voxel_particles.inl"

#include <utilities/gpu/defs.glsl>

DAXA_DECL_PUSH_CONSTANT(VoxelParticleRasterPush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(SimulatedVoxelParticle) simulated_voxel_particles = push.uses.simulated_voxel_particles;
daxa_BufferPtr(uint) rendered_voxel_particles = push.uses.rendered_voxel_particles;
daxa_ImageViewIndex render_image = push.uses.render_image;
daxa_ImageViewIndex depth_image_id = push.uses.depth_image_id;

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

#include "particle.glsl"
#include <renderer/kajiya/inc/camera.glsl>

layout(location = 1) out uint id;

void main() {
    uint particle_index = gl_InstanceIndex;

    uint simulated_particle_index = deref(advance(rendered_voxel_particles, particle_index));
    SimulatedVoxelParticle particle = deref(advance(simulated_voxel_particles, simulated_particle_index));

    const vec3 diff = vec3(1023.0 / 1024.0 * VOXEL_SIZE);
    vec3 center_ws = get_particle_worldspace_origin(gpu_input, particle.pos);
    const vec3 camera_position = deref(gpu_input).player.pos;
    vec3 sign_ = vec3(ivec3(greaterThan(camera_position, center_ws)) ^ ((ivec3(0x1C, 0x46, 0x70) >> gl_VertexIndex) & ivec3(1))) - 0.5;
    vec3 vert_pos = sign_ * diff + center_ws;

    vec4 vs_pos = deref(gpu_input).player.cam.world_to_view * vec4(vert_pos, 1);
    vec4 cs_pos = deref(gpu_input).player.cam.view_to_sample * vs_pos;
    // TODO: Figure out why raster is projected upside down
    cs_pos.y *= -1;

    gl_Position = cs_pos;
    id = simulated_particle_index;
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT

layout(location = 1) flat in uint id;
layout(location = 0) out uvec4 color;

void main() {
    color = uvec4(id + 1, 0, 0, 0);
}

#endif
