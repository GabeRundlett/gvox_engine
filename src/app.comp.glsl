#include <app.inl>

#if StartupComputeShader

DAXA_DECL_PUSH_CONSTANT(StartupComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_RWBufferPtr)

#include <voxels/core.glsl>

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    voxel_world_startup(globals, VOXELS_RW_BUFFER_PTRS);
}

#endif

#if PerframeComputeShader

DAXA_DECL_PUSH_CONSTANT(PerframeComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuOutput) gpu_output = push.uses.gpu_output;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_RWBufferPtr(SimulatedVoxelParticle) simulated_voxel_particles = push.uses.simulated_voxel_particles;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_RWBufferPtr)

#include <renderer/kajiya/inc/camera.glsl>
#include <voxels/core.glsl>
#include <voxels/voxel_particle.glsl>

#define INPUT deref(gpu_input)
#define BRUSH_STATE deref(globals).brush_state
#define PLAYER deref(gpu_input).player
#define CHUNKS(i) deref(advance(voxel_chunks, i))
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    voxel_world_perframe(gpu_input, gpu_output, globals, VOXELS_RW_BUFFER_PTRS);

    {
        vec2 frame_dim = INPUT.frame_dim;
        vec2 inv_frame_dim = vec2(1.0) / frame_dim;
        vec2 uv = vec2(0.5); // get_uv(deref(gpu_input).mouse.pos, vec4(frame_dim, inv_frame_dim));
        ViewRayContext vrc = unjittered_vrc_from_uv(gpu_input, uv);
        vec3 ray_dir = ray_dir_ws(vrc);
        vec3 cam_pos = ray_origin_ws(vrc);
        vec3 ray_pos = cam_pos;
        voxel_trace(VoxelTraceInfo(VOXELS_BUFFER_PTRS, ray_dir, MAX_STEPS, MAX_DIST, 0.0, true), ray_pos);

        if (BRUSH_STATE.is_editing == 0) {
            BRUSH_STATE.initial_ray = ray_pos - cam_pos;
        }

        deref(globals).brush_input.prev_pos = deref(globals).brush_input.pos;
        deref(globals).brush_input.prev_pos_offset = deref(globals).brush_input.pos_offset;
        deref(globals).brush_input.pos = length(BRUSH_STATE.initial_ray) * ray_dir + cam_pos;
        deref(globals).brush_input.pos_offset = deref(gpu_input).player.player_unit_offset;

        BRUSH_STATE.is_editing = 0;
    }

    deref(globals).voxel_particles_state.simulation_dispatch = uvec3(MAX_SIMULATED_VOXEL_PARTICLES / 64, 1, 1);
    deref(globals).voxel_particles_state.draw_params.vertex_count = 0;
    deref(globals).voxel_particles_state.draw_params.instance_count = 1;
    deref(globals).voxel_particles_state.draw_params.first_vertex = 0;
    deref(globals).voxel_particles_state.draw_params.first_instance = 0;
    deref(globals).voxel_particles_state.place_count = 0;
    deref(globals).voxel_particles_state.place_bounds_min = uvec3(1000000);
    deref(globals).voxel_particles_state.place_bounds_max = uvec3(0);

    if (INPUT.frame_index == 0) {
        for (uint i = 0; i < MAX_SIMULATED_VOXEL_PARTICLES; ++i) {
            SimulatedVoxelParticle self = deref(advance(simulated_voxel_particles, i));
            particle_spawn(self, i);
            deref(advance(simulated_voxel_particles, i)) = self;
        }
    }
}
#undef CHUNKS
#undef PLAYER
#undef INPUT

#endif
