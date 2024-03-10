#include "voxel_particles.inl"
#include "particle.glsl"

#include <utilities/gpu/noise.glsl>
#include <g_samplers>

DAXA_DECL_PUSH_CONSTANT(GrassStrandSimComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(VoxelParticlesState) particles_state = push.uses.particles_state;
SIMPLE_STATIC_ALLOCATOR_BUFFERS_PUSH_USES(GrassStrandAllocator, grass_allocator)
daxa_RWBufferPtr(GrassStrand) grass_strands = deref(grass_allocator).heap;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
daxa_RWBufferPtr(ParticleVertex) cube_rendered_particle_verts = push.uses.cube_rendered_particle_verts;
daxa_RWBufferPtr(ParticleVertex) splat_rendered_particle_verts = push.uses.splat_rendered_particle_verts;
daxa_ImageViewIndex value_noise_texture = push.uses.value_noise_texture;

#define UserAllocatorType GrassStrandAllocator
#define UserIndexType uint
#include <utilities/allocator.glsl>

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint particle_index = gl_GlobalInvocationID.x;
    GrassStrand self = deref(advance(grass_strands, particle_index));

    // self.flags = 1;
    if (self.flags == 0) {
        return;
    }

    rand_seed(particle_index);

    Voxel grass_voxel = unpack_voxel(self.packed_voxel);
    uvec3 chunk_n = uvec3(1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
    vec3 origin_ws = get_particle_worldspace_origin(gpu_input, self.origin);
    PackedVoxel voxel_data = sample_voxel_chunk(VOXELS_BUFFER_PTRS, chunk_n, origin_ws, vec3(0));
    Voxel ground_voxel = unpack_voxel(voxel_data);

    FractalNoiseConfig noise_conf = FractalNoiseConfig(
        /* .amplitude   = */ 1.0,
        /* .persistance = */ 0.5,
        /* .scale       = */ 0.025,
        /* .lacunarity  = */ 4.5,
        /* .octaves     = */ 3);
    vec4 noise_val = fractal_noise(value_noise_texture, g_sampler_llr, self.origin + vec3(deref(gpu_input).time * 1.1, sin(deref(gpu_input).time), 0), noise_conf);
    float rot = noise_val.x * 100.0;
    vec2 rot_offset = vec2(sin(rot), cos(rot));

    grass_voxel.normal = normalize(ground_voxel.normal + vec3(rot_offset, 1.0) * 0.05);

    if (grass_voxel.color != ground_voxel.color) {
        // free voxel, its spawner died.
        self.flags = 0;
        deref(advance(grass_strands, particle_index)) = self;
        GrassStrandAllocator_free(grass_allocator, particle_index);
        return;
    }

    self.packed_voxel = pack_voxel(grass_voxel);
    deref(advance(grass_strands, particle_index)) = self;

    if (self.flags != 0) {
        for (uint i = 1; i <= 3; ++i) {
            float pct = float(i) / 3.0;
            vec3 offset = vec3(0, 0, float(i * VOXEL_SIZE));
            offset.xy = rot_offset * VOXEL_SIZE * pct * 1.5;
            particle_render(cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, self.origin + offset, particle_index + MAX_SIMULATED_VOXEL_PARTICLES);
        }
    }
}
