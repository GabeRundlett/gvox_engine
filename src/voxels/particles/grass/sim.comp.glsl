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

#include <renderer/rt.glsl>

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
#if defined(VOXELS_ORIGINAL_IMPL)
    uint particle_index = gl_GlobalInvocationID.x;
    GrassStrand self = deref(advance(grass_strands, particle_index));

    // self.flags = 1;
    if (self.flags == 0) {
        return;
    }
    if (self.flags < 63) {
        self.flags++;
    }

    rand_seed(particle_index);

    Voxel grass_voxel = unpack_voxel(self.packed_voxel);
    vec3 origin_ws = get_particle_worldspace_origin(gpu_input, self.origin);

    vec3 ray_pos = origin_ws + vec3(0, 0, VOXEL_SIZE);
    vec3 ray_dir = normalize(vec3(0.0001, 0.0001, -1));

    VoxelTraceResult trace_result = voxel_trace(VoxelRtTraceInfo(VOXELS_RT_BUFFER_PTRS, ray_dir, 1.0 * VOXEL_SIZE), ray_pos);
    Voxel ground_voxel = unpack_voxel(trace_result.voxel_data);

    if ((grass_voxel.material_type != ground_voxel.material_type ||
         grass_voxel.roughness != ground_voxel.roughness ||
         trace_result.dist > 1.0 * VOXEL_SIZE) &&
        self.flags > 8) {
        // free voxel, its spawner died.
        self.flags = 0;
        deref(advance(grass_strands, particle_index)) = self;
        GrassStrandAllocator_free(grass_allocator, particle_index);
        return;
    }

    self.packed_voxel = pack_voxel(grass_voxel);
    deref(advance(grass_strands, particle_index)) = self;

    uint height = 2 + uint(rand() * 2.5);
    for (uint i = 1; i <= height; ++i) {
        PackedParticleVertex packed_vertex = PackedParticleVertex(((particle_index & 0xffffff) << 0) | ((i & 0xff) << 24));
        ParticleVertex grass_vertex = get_grass_vertex(gpu_input, daxa_BufferPtr(GrassStrand)(as_address(grass_strands)), packed_vertex);
        particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, grass_vertex, packed_vertex, false);
    }
#endif
}
