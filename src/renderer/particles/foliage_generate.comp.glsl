#include "particles/render_foliage.inl"
#include <voxels/pack_unpack.inl>
#include <utilities/gpu/math.glsl>

DAXA_DECL_PUSH_CONSTANT(FoliageGeneratePush, push)

#define UserAllocatorType GrassStrandAllocator
#define UserIndexType uint
#define UserMaxElementCount MAX_GRASS_BLADES
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

    // `bitmask` is laid out as 32-bit words (matching the CPU/ISPC packing in
    // generation.cpp), so reinterpret it as such here to atomically
    // test-and-clear a single bit -- daxa doesn't guarantee 64-bit atomics.
    daxa_RWBufferPtr(daxa_u32) bitmask_words = daxa_RWBufferPtr(daxa_u32)(as_address(brick_ptr));
    uint word_i = voxel_index / 32;
    uint bit_i = voxel_index % 32;
    uint prev_word = deref(advance(bitmask_words, word_i)); // & ~(1u << bit_i);
    bool has_foliage = (prev_word & (1u << bit_i)) != 0u;
    if (!has_foliage) {
        // Either genuinely empty, or another frame already spawned this voxel's blade.
        return;
    }

    vec3 world_pos = brick_pos + vec3(local_voxel) * scale;

    GrassStrand strand;
    strand.origin = world_pos;
    strand.packed_voxel = deref(brick_attributes).voxels[voxel_index];
    strand.flags = 1;

    uint index = GrassStrandAllocator_malloc(push.grass_allocator);
    if (index < MAX_GRASS_BLADES) {
        deref(advance(deref(push.grass_allocator).heap, index)) = strand;
    }
}
