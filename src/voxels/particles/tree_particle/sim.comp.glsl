#include "tree_particle.inl"
#include "tree_particle.glsl"

DAXA_DECL_PUSH_CONSTANT(TreeParticleSimComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(VoxelParticlesState) particles_state = push.uses.particles_state;
SIMPLE_STATIC_ALLOCATOR_BUFFERS_PUSH_USES(TreeParticleAllocator, tree_particle_allocator)
daxa_RWBufferPtr(TreeParticle) tree_particles = deref(tree_particle_allocator).heap;
daxa_RWBufferPtr(PackedParticleVertex) cube_rendered_particle_verts = push.uses.cube_rendered_particle_verts;
daxa_RWBufferPtr(PackedParticleVertex) shadow_cube_rendered_particle_verts = push.uses.shadow_cube_rendered_particle_verts;
daxa_RWBufferPtr(PackedParticleVertex) splat_rendered_particle_verts = push.uses.splat_rendered_particle_verts;

#define UserAllocatorType TreeParticleAllocator
#define UserIndexType uint
#define UserMaxElementCount MAX_TREE_PARTICLES
#include <utilities/allocator.glsl>

#include <renderer/rt.glsl>

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
#if defined(VOXELS_ORIGINAL_IMPL)
    uint particle_index = gl_GlobalInvocationID.x;
    TreeParticle self = deref(advance(tree_particles, particle_index));

    if (self.flags == 0) {
        return;
    }
    if (self.flags < 63) {
        self.flags++;
    }

    rand_seed(particle_index);

    Voxel tree_particle_voxel = unpack_voxel(self.packed_voxel);
    vec3 origin_ws = get_particle_worldspace_origin(gpu_input, self.origin);

    vec3 ray_pos = origin_ws + vec3(0, 0, VOXEL_SIZE);
    vec3 ray_dir = normalize(vec3(0.0001, 0.0001, -1));

    VoxelTraceResult trace_result = voxel_trace(VoxelRtTraceInfo(VOXELS_RT_BUFFER_PTRS, ray_dir, 1.0 * VOXEL_SIZE), ray_pos);
    Voxel ground_voxel = unpack_voxel(trace_result.voxel_data);

    if ((tree_particle_voxel.material_type != ground_voxel.material_type ||
         tree_particle_voxel.roughness != ground_voxel.roughness ||
         trace_result.dist > 1.0 * VOXEL_SIZE) &&
        self.flags > 8) {
        // free voxel, its spawner died.
        self.flags = 0;
        deref(advance(tree_particles, particle_index)) = self;
        TreeParticleAllocator_free(tree_particle_allocator, particle_index);
        return;
    }

    self.packed_voxel = pack_voxel(tree_particle_voxel);
    deref(advance(tree_particles, particle_index)) = self;

    for (uint i = 0; i < 2; ++i) {
        PackedParticleVertex packed_vertex = PackedParticleVertex(((particle_index & 0xffffff) << 0) | ((i & 0xff) << 24));
        ParticleVertex tree_particle_vertex = get_tree_particle_vertex(gpu_input, daxa_BufferPtr(TreeParticle)(as_address(tree_particles)), packed_vertex);
        particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, tree_particle_vertex, packed_vertex, false);
    }
#endif
}
