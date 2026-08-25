#include "flower.inl"
#include "flower.glsl"

DAXA_DECL_PUSH_CONSTANT(FlowerSimComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(VoxelParticlesState) particles_state = push.uses.particles_state;
SIMPLE_STATIC_ALLOCATOR_BUFFERS_PUSH_USES(FlowerAllocator, flower_allocator)
daxa_RWBufferPtr(Flower) flowers = deref(flower_allocator).heap;
daxa_RWBufferPtr(PackedParticleVertex) cube_rendered_particle_verts = push.uses.cube_rendered_particle_verts;
daxa_RWBufferPtr(PackedParticleVertex) shadow_cube_rendered_particle_verts = push.uses.shadow_cube_rendered_particle_verts;
daxa_RWBufferPtr(PackedParticleVertex) splat_rendered_particle_verts = push.uses.splat_rendered_particle_verts;

#define UserAllocatorType FlowerAllocator
#define UserIndexType uint
#define UserMaxElementCount MAX_FLOWERS
#include <utilities/allocator.glsl>

#include <utilities/gpu/random.glsl>

void render_dandelion(uint particle_index) {
    uint height = 6;

    for (uint i = 1; i <= height; ++i) {
        PackedParticleVertex packed_vertex = PackedParticleVertex(((particle_index & 0xffffff) << 0) | ((i & 0xff) << 24));
        ParticleVertex flower_vertex = get_flower_vertex(gpu_input, daxa_BufferPtr(Flower)(as_address(flowers)), packed_vertex);
        particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, flower_vertex, packed_vertex, false);
    }

    for (int yi = -1; yi <= 1; ++yi) {
        for (int xi = -1; xi <= 1; ++xi) {
            if ((xi != 0 && yi != 0)) {
                continue;
            }
            uint i = uint(xi + 1) + uint(yi + 1) * 3 + height + 1;
            PackedParticleVertex packed_vertex = PackedParticleVertex(((particle_index & 0xffffff) << 0) | ((i & 0xff) << 24));
            ParticleVertex flower_vertex = get_flower_vertex(gpu_input, daxa_BufferPtr(Flower)(as_address(flowers)), packed_vertex);
            particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, flower_vertex, packed_vertex, true);
        }
    }
}

void render_dandelion_white(uint particle_index) {
    uint height = 6;

    for (uint i = 1; i <= height; ++i) {
        PackedParticleVertex packed_vertex = PackedParticleVertex(((particle_index & 0xffffff) << 0) | ((i & 0xff) << 24));
        ParticleVertex flower_vertex = get_flower_vertex(gpu_input, daxa_BufferPtr(Flower)(as_address(flowers)), packed_vertex);
        particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, flower_vertex, packed_vertex, false);
    }

    for (int zi = -1; zi <= 1; ++zi) {
        for (int yi = -1; yi <= 1; ++yi) {
            for (int xi = -1; xi <= 1; ++xi) {
                if ((xi != 0 && yi != 0 && zi != 0) || (xi == 0 && yi == 0 && zi == 0)) {
                    continue;
                }
                uint i = uint(xi + 1) + uint(yi + 1) * 3 + uint(zi + 1) * 9 + height + 1;
                PackedParticleVertex packed_vertex = PackedParticleVertex(((particle_index & 0xffffff) << 0) | ((i & 0xff) << 24));
                ParticleVertex flower_vertex = get_flower_vertex(gpu_input, daxa_BufferPtr(Flower)(as_address(flowers)), packed_vertex);
                particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, flower_vertex, packed_vertex, true);
            }
        }
    }

    for (uint i = 27 + height + 1; i <= 27 + height + 3; ++i) {
        PackedParticleVertex packed_vertex = PackedParticleVertex(((particle_index & 0xffffff) << 0) | ((i & 0xff) << 24));
        ParticleVertex flower_vertex = get_flower_vertex(gpu_input, daxa_BufferPtr(Flower)(as_address(flowers)), packed_vertex);
        particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, flower_vertex, packed_vertex, false);
    }
}

void render_tulip(uint particle_index) {
    uint height = 6;

    for (uint i = 1; i <= height; ++i) {
        PackedParticleVertex packed_vertex = PackedParticleVertex(((particle_index & 0xffffff) << 0) | ((i & 0xff) << 24));
        ParticleVertex flower_vertex = get_flower_vertex(gpu_input, daxa_BufferPtr(Flower)(as_address(flowers)), packed_vertex);
        particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, flower_vertex, packed_vertex, false);
    }

    for (int zi = -1; zi <= 0; ++zi) {
        for (int yi = -1; yi <= 1; ++yi) {
            for (int xi = -1; xi <= 1; ++xi) {
                if ((xi != 0 && yi != 0 && zi != 0) || (xi == 0 && yi == 0 && zi == 0)) {
                    continue;
                }
                uint i = uint(xi + 1) + uint(yi + 1) * 3 + uint(zi + 1) * 9 + height + 1;
                PackedParticleVertex packed_vertex = PackedParticleVertex(((particle_index & 0xffffff) << 0) | ((i & 0xff) << 24));
                ParticleVertex flower_vertex = get_flower_vertex(gpu_input, daxa_BufferPtr(Flower)(as_address(flowers)), packed_vertex);
                particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, flower_vertex, packed_vertex, true);
            }
        }
    }
}

void render_lavender(uint particle_index) {
    uint height = 4;

    for (uint i = 1; i <= height; ++i) {
        PackedParticleVertex packed_vertex = PackedParticleVertex(((particle_index & 0xffffff) << 0) | ((i & 0xff) << 24));
        ParticleVertex flower_vertex = get_flower_vertex(gpu_input, daxa_BufferPtr(Flower)(as_address(flowers)), packed_vertex);
        particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, flower_vertex, packed_vertex, false);
    }

    for (int zi = -1; zi <= 3; ++zi) {
        for (int yi = -1; yi <= 1; ++yi) {
            for (int xi = -1; xi <= 1; ++xi) {
                if ((xi != 0 && yi != 0) || (xi == 0 && yi == 0)) {
                    continue;
                }
                uint i = uint(xi + 1) + uint(yi + 1) * 3 + uint(zi + 1) * 9 + height + 1;
                PackedParticleVertex packed_vertex = PackedParticleVertex(((particle_index & 0xffffff) << 0) | ((i & 0xff) << 24));
                ParticleVertex flower_vertex = get_flower_vertex(gpu_input, daxa_BufferPtr(Flower)(as_address(flowers)), packed_vertex);
                particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, flower_vertex, packed_vertex, true);
            }
        }
    }

    uint i = height + 1 + 5 * 3 * 3 + 4;
    PackedParticleVertex packed_vertex = PackedParticleVertex(((particle_index & 0xffffff) << 0) | ((i & 0xff) << 24));
    ParticleVertex flower_vertex = get_flower_vertex(gpu_input, daxa_BufferPtr(Flower)(as_address(flowers)), packed_vertex);
    particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, flower_vertex, packed_vertex, true);
}

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint particle_index = gl_GlobalInvocationID.x;
    if (particle_index > deref(flower_allocator).element_count)
        return;
    Flower self = deref(advance(flowers, particle_index));

    if (self.type == FLOWER_TYPE_NONE) {
        return;
    }
    if (self.flags < 63) {
        ++self.flags;
    }
    deref(advance(flowers, particle_index)) = self;

    rand_seed(hash3(floatBitsToUint(self.origin)));

    switch (self.type) {
    case FLOWER_TYPE_DANDELION: render_dandelion(particle_index); break;
    case FLOWER_TYPE_DANDELION_WHITE: render_dandelion_white(particle_index); break;
    case FLOWER_TYPE_TULIP: render_tulip(particle_index); break;
    case FLOWER_TYPE_LAVENDER: render_lavender(particle_index); break;
    }
}
