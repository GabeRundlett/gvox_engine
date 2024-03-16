#include "dandelion.inl"
#include "dandelion.glsl"

DAXA_DECL_PUSH_CONSTANT(DandelionSimComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(VoxelParticlesState) particles_state = push.uses.particles_state;
SIMPLE_STATIC_ALLOCATOR_BUFFERS_PUSH_USES(DandelionAllocator, dandelion_allocator)
daxa_RWBufferPtr(Dandelion) dandelions = deref(dandelion_allocator).heap;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
daxa_RWBufferPtr(PackedParticleVertex) cube_rendered_particle_verts = push.uses.cube_rendered_particle_verts;
daxa_RWBufferPtr(PackedParticleVertex) shadow_cube_rendered_particle_verts = push.uses.shadow_cube_rendered_particle_verts;
daxa_RWBufferPtr(PackedParticleVertex) splat_rendered_particle_verts = push.uses.splat_rendered_particle_verts;
daxa_ImageViewIndex value_noise_texture = push.uses.value_noise_texture;

#define UserAllocatorType DandelionAllocator
#define UserIndexType uint
#define UserMaxElementCount MAX_DANDELIONS
#include <utilities/allocator.glsl>

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
#if defined(VOXELS_ORIGINAL_IMPL)
    uint particle_index = gl_GlobalInvocationID.x;
    Dandelion self = deref(advance(dandelions, particle_index));

    // self.flags = 1;
    if (self.flags == 0) {
        return;
    }

    rand_seed(particle_index);

    Voxel dandelion_voxel = unpack_voxel(self.packed_voxel);
    uvec3 chunk_n = uvec3(1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
    vec3 origin_ws = get_particle_worldspace_origin(gpu_input, self.origin);
    PackedVoxel ground_voxel_data = sample_voxel_chunk(VOXELS_BUFFER_PTRS, chunk_n, origin_ws, vec3(0));
    Voxel ground_voxel = unpack_voxel(ground_voxel_data);

    PackedVoxel air_voxel_data = sample_voxel_chunk(VOXELS_BUFFER_PTRS, chunk_n, origin_ws + vec3(0, 0, VOXEL_SIZE), vec3(0));
    Voxel air_voxel = unpack_voxel(air_voxel_data);

    // dandelion_voxel.normal = normalize(ground_voxel.normal + vec3(rot_offset, 1.0) * 0.05);

    if (air_voxel.material_type != 0 ||
        dandelion_voxel.material_type != ground_voxel.material_type ||
        dandelion_voxel.roughness != ground_voxel.roughness) {
        // free voxel, its spawner died.
        self.flags = 0;
        deref(advance(dandelions, particle_index)) = self;
        DandelionAllocator_free(dandelion_allocator, particle_index);
        return;
    }

    self.packed_voxel = pack_voxel(dandelion_voxel);
    deref(advance(dandelions, particle_index)) = self;

    uint height = 8;
    for (uint i = 1; i <= height; ++i) {
        PackedParticleVertex packed_vertex = PackedParticleVertex(((particle_index & 0xffffff) << 0) | ((i & 0xff) << 24));
        ParticleVertex dandelion_vertex = get_dandelion_vertex(gpu_input, daxa_BufferPtr(Dandelion)(dandelions), packed_vertex);
        particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, dandelion_vertex, packed_vertex);
    }

    for (int zi = -1; zi <= 1; ++zi) {
        for (int yi = -1; yi <= 1; ++yi) {
            for (int xi = -1; xi <= 1; ++xi) {
                if ((xi != 0 && yi != 0 && zi != 0) || (xi == 0 && yi == 0 && zi == 0)) {
                    continue;
                }
                uint i = uint(xi + 1) + uint(yi + 1) * 3 + uint(zi + 1) * 9 + 8;
                PackedParticleVertex packed_vertex = PackedParticleVertex(((particle_index & 0xffffff) << 0) | ((i & 0xff) << 24));
                ParticleVertex dandelion_vertex = get_dandelion_vertex(gpu_input, daxa_BufferPtr(Dandelion)(dandelions), packed_vertex);
                particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, dandelion_vertex, packed_vertex);
            }
        }
    }

    for (uint i = 27 + 8 + 1; i <= 27 + 8 + 3; ++i) {
        PackedParticleVertex packed_vertex = PackedParticleVertex(((particle_index & 0xffffff) << 0) | ((i & 0xff) << 24));
        ParticleVertex dandelion_vertex = get_dandelion_vertex(gpu_input, daxa_BufferPtr(Dandelion)(dandelions), packed_vertex);
        particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, dandelion_vertex, packed_vertex);
    }

#endif
}
