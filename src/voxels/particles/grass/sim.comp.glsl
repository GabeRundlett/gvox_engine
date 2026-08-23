#include "grass.inl"
#include "grass.glsl"

DAXA_DECL_PUSH_CONSTANT(GrassStrandSimComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(VoxelParticlesState) particles_state = push.uses.particles_state;
SIMPLE_STATIC_ALLOCATOR_BUFFERS_PUSH_USES(GrassStrandAllocator, grass_allocator)
daxa_RWBufferPtr(GrassStrand) grass_strands = deref(grass_allocator).heap;
daxa_RWBufferPtr(PackedParticleVertex) cube_rendered_particle_verts = push.uses.cube_rendered_particle_verts;
daxa_RWBufferPtr(PackedParticleVertex) shadow_cube_rendered_particle_verts = push.uses.shadow_cube_rendered_particle_verts;
daxa_RWBufferPtr(PackedParticleVertex) splat_rendered_particle_verts = push.uses.splat_rendered_particle_verts;

#define UserAllocatorType GrassStrandAllocator
#define UserIndexType uint
#define UserMaxElementCount MAX_GRASS_BLADES
#include <utilities/allocator.glsl>

#include <utilities/gpu/random.glsl>

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint particle_index = gl_GlobalInvocationID.x;
    GrassStrand self = deref(advance(grass_strands, particle_index));

    if (self.flags == 0) {
        return;
    }
    if (self.flags < 63) {
        self.flags++;
    }
    deref(advance(grass_strands, particle_index)) = self;

    rand_seed(hash3(floatBitsToUint(self.origin)));

    uint height = 2 + uint(rand() * 2.5);
    for (uint i = 1; i <= height; ++i) {
        PackedParticleVertex packed_vertex = PackedParticleVertex(((particle_index & 0xffffff) << 0) | ((i & 0xff) << 24));
        ParticleVertex grass_vertex = get_grass_vertex(gpu_input, daxa_BufferPtr(GrassStrand)(as_address(grass_strands)), packed_vertex);
        particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, grass_vertex, packed_vertex, false);
    }
}
