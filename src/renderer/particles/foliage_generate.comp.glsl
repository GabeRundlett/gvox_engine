#include "particles/render_foliage.inl"
#include <voxels/pack_unpack.inl>
#include <utilities/gpu/math.glsl>
#include <utilities/gpu/random.glsl>
#include <utilities/gpu/noise.glsl>
#include <renderer/globals.glsl>

DAXA_DECL_PUSH_CONSTANT(FoliageGeneratePush, push)

#define UserAllocatorType GrassStrandAllocator
#define UserIndexType uint
#define UserMaxElementCount MAX_GRASS_BLADES
#include <utilities/allocator.glsl>

#define UserAllocatorType FlowerAllocator
#define UserIndexType uint
#define UserMaxElementCount MAX_FLOWERS
#include <utilities/allocator.glsl>

layout(local_size_x = BRICK_SIZE, local_size_y = BRICK_SIZE, local_size_z = BRICK_SIZE) in;

void main() {
    uint visible_brick_index = gl_WorkGroupID.x + 1u;
    FoliageBrickInstance visible_brick = deref(advance(push.visible_foliage_bricks, visible_brick_index));
    if (visible_brick.brick_index >= deref(visible_brick.voxel_object).brick_count)
        return;

    daxa_BufferPtr(VoxelFoliageBrick) brick_ptr = advance(deref(visible_brick.voxel_object).brick_foliage, visible_brick.brick_index);
    daxa_BufferPtr(BrickPrimitive) brick_primitive_ptr = advance(deref(visible_brick.voxel_object).brick_primitives, visible_brick.brick_index);
    daxa_BufferPtr(VoxelShadingAttribBrick) brick_attributes = advance(deref(visible_brick.voxel_object).brick_shading_attribs, visible_brick.brick_index);
    const float scale = deref(visible_brick.voxel_object).scale;
    vec3 brick_pos = deref(visible_brick.voxel_object).pos + vec3(deref(brick_primitive_ptr).offset & ~BRICK_MASK) * scale;

    uvec3 local_voxel = gl_LocalInvocationID.xyz;
    uint voxel_index = local_voxel.x + local_voxel.y * BRICK_SIZE + local_voxel.z * BRICK_SIZE * BRICK_SIZE;

    daxa_RWBufferPtr(daxa_u32) bitmask_words = daxa_RWBufferPtr(daxa_u32)(as_address(brick_ptr));
    uint word_i = voxel_index / 32;
    uint bit_i = voxel_index % 32;
    uint prev_word = deref(advance(bitmask_words, word_i));
    bool has_foliage = (prev_word & (1u << bit_i)) != 0u;
    if (!has_foliage) {
        return;
    }

    vec3 world_pos = brick_pos + vec3(local_voxel) * scale;

    float r2 = good_rand(world_pos.xy);
    const float FLOWER_SPAWN_CHANCE = 0.01;

    if (r2 < FLOWER_SPAWN_CHANCE) {
        FractalNoiseConfig noise_conf = FractalNoiseConfig(
            /* .amplitude   = */ 1.0,
            /* .persistance = */ 0.5,
            /* .scale       = */ 0.1,
            /* .lacunarity  = */ 2,
            /* .octaves     = */ 3);
        vec4 flower_noise_val = fractal_noise(g_value_noise_tex, g_sampler_llr, vec3(world_pos.xy, 0), noise_conf);
        float v = flower_noise_val.x * (1.0 / 0.875);

        uint flower_type = FLOWER_TYPE_DANDELION;
        if (v < 0.4) {
            flower_type = FLOWER_TYPE_DANDELION;
        } else if (v < 0.5) {
            flower_type = FLOWER_TYPE_DANDELION_WHITE;
        } else if (v < 0.65) {
            flower_type = FLOWER_TYPE_TULIP;
        } else {
            flower_type = FLOWER_TYPE_LAVENDER;
        }

        Flower flower;
        flower.origin = world_pos;
        flower.packed_voxel = deref(brick_attributes).voxels[voxel_index];
        flower.type = flower_type;
        flower.flags = 1;

        uint flower_index = FlowerAllocator_malloc(push.flower_allocator);
        if (flower_index < MAX_FLOWERS) {
            deref(advance(deref(push.flower_allocator).heap, flower_index)) = flower;
        }
        return;
    }

    GrassStrand strand;
    strand.origin = world_pos;
    strand.packed_voxel = deref(brick_attributes).voxels[voxel_index];
    strand.flags = 1;

    uint index = GrassStrandAllocator_malloc(push.grass_allocator);
    if (index < MAX_GRASS_BLADES) {
        deref(advance(deref(push.grass_allocator).heap, index)) = strand;
    }
}
