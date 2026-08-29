#include "render_voxel_object.hpp"
#include "base/log.hpp"
#include "voxels/voxel.inl"
#include "voxels/voxel_object.hpp"
#include "voxels/voxel_brick.hpp"
#include <cassert>
#include <daxa/daxa.hpp>
#include <daxa/types.hpp>

#define RENDERER_INTERNAL 1
#include "render_scene.hpp"

#include <base/profiler.hpp>

struct RenderVoxelObject {
    daxa::BlasId blas{};
    daxa::BufferId blas_buffer{};
    daxa::BufferId blas_scratch_buffer{};

    daxa::BufferId bricks_data_buffer{};
    uint32_t brick_count;

    daxa::DeviceAddress brick_aabb_device_address;
    daxa::DeviceAddress brick_shading_device_address;
    daxa::DeviceAddress brick_primitives_device_address;
    daxa::DeviceAddress brick_flags_device_address;
    daxa::DeviceAddress brick_foliage_device_address;
    daxa::DeviceAddress blas_device_address;

    bool has_foliage;

    RenderScene *scene;
};

struct RenderVoxelObject *create_render_voxel_object(struct RenderScene *scene, bool has_foliage) {
    RenderVoxelObject *result = new RenderVoxelObject();
    result->scene = scene;
    result->has_foliage = has_foliage;
    return result;
}

void destroy_render_voxel_object(GpuContext &gpu_context, struct RenderVoxelObject *self) {
    auto &device = gpu_context.device;

    if (device.is_id_valid(self->bricks_data_buffer))
        device.destroy_buffer(self->bricks_data_buffer);

    if (device.is_id_valid(self->blas))
        device.destroy_blas(self->blas);
    if (device.is_id_valid(self->blas_buffer))
        device.destroy_buffer(self->blas_buffer);
    if (device.is_id_valid(self->blas_scratch_buffer))
        device.destroy_buffer(self->blas_scratch_buffer);

    delete self;
}

namespace {
    struct BricksBufferInfo {
        size_t mPrimitivesSize;
        size_t mShadingSize;
        size_t mAabbSize;
        size_t mFlagsSize;
        size_t mFoliageSize;

        size_t mPrimitivesOffset;
        size_t mShadingOffset;
        size_t mAabbOffset;
        size_t mFlagsOffset;
        size_t mFoliageOffset;

        size_t mTotalSize;
    };
    auto get_bricks_buffer_info(int brick_count, bool has_foliage) -> BricksBufferInfo {
        BricksBufferInfo result;

        result.mPrimitivesSize = sizeof(BrickPrimitive) * brick_count;
        result.mShadingSize = sizeof(VoxelShadingAttribBrick) * brick_count;
        result.mAabbSize = sizeof(Aabb) * brick_count;
        result.mFlagsSize = sizeof(uint32_t) * brick_count;
        result.mFoliageSize = has_foliage ? sizeof(VoxelFoliageBrick) * brick_count : size_t{0};

        result.mPrimitivesOffset = size_t{0};
        result.mShadingOffset = result.mPrimitivesOffset + result.mPrimitivesSize;
        result.mAabbOffset = result.mShadingOffset + result.mShadingSize;
        result.mFlagsOffset = result.mAabbOffset + result.mAabbSize;
        result.mFoliageOffset = result.mFlagsOffset + result.mFlagsSize;
        result.mTotalSize = result.mFoliageOffset + result.mFoliageSize;

        return result;
    }

    inline auto GetAligned(daxa_u64 operand, daxa_u64 granularity) -> daxa_u64 {
        return ((operand + (granularity - 1)) / granularity) * granularity;
    }

    static bool cachedBlasSizeInfo = false;
    static daxa::AccelerationStructureBuildSizesInfo cachedBlasBuildSizeInfo;

    inline void CreateBlas(daxa::Device &device, RenderVoxelObject *self, std::string name) {
        PROFILE_FUNC();
        uint32_t aabbCount = self->brick_count;

        if (device.is_id_valid(self->blas))
            device.destroy_blas(self->blas);
        if (device.is_id_valid(self->blas_buffer))
            device.destroy_buffer(self->blas_buffer);
        if (device.is_id_valid(self->blas_scratch_buffer))
            device.destroy_buffer(self->blas_scratch_buffer);

        if (aabbCount == 0)
            return;

        auto accelerationStructureScratchOffsetAlignment = device.properties().acceleration_structure_properties.value().min_acceleration_structure_scratch_offset_alignment;
        auto geometry = std::array{
            daxa::BlasAabbGeometryInfo{
                .stride = sizeof(Aabb),
                .count = aabbCount,
                .flags = daxa::GeometryFlagBits::OPAQUE,
            },
        };
        auto blasBuildInfo = daxa::BlasBuildInfo{
            .flags = daxa::AccelerationStructureBuildFlagBits::PREFER_FAST_TRACE,
            .dst_blas = {}, // Ignored in blas_build_sizes.
            .geometries = geometry,
            .scratch_data = {}, // Ignored in blas_build_sizes.
        };
        const auto buildSizeInfo = (cachedBlasSizeInfo && aabbCount == 1) ? cachedBlasBuildSizeInfo : device.blas_build_sizes(blasBuildInfo);
        if (!cachedBlasSizeInfo && aabbCount == 1) {
            cachedBlasBuildSizeInfo = buildSizeInfo;
            cachedBlasSizeInfo = true;
        }

        auto scratchAlignmentSize = GetAligned(buildSizeInfo.build_scratch_size, accelerationStructureScratchOffsetAlignment);
        self->blas_scratch_buffer = device.create_buffer({
            .size = scratchAlignmentSize,
            .name = (name + " scratch buffer").c_str(),
        });

        const daxa_u32 accelerationStructureBuildOffsetAligment = 256; // NOTE: Requested by the spec
        auto buildAligmentSize = GetAligned(buildSizeInfo.acceleration_structure_size, accelerationStructureBuildOffsetAligment);
        self->blas_buffer = device.create_buffer({
            .size = buildAligmentSize,
            .name = (name + " buffer").c_str(),
        });

        self->blas = device.create_blas_from_buffer({
            .blas_info = {
                .size = buildSizeInfo.acceleration_structure_size,
                .name = name.c_str(),
            },
            .buffer = self->blas_buffer,
            .offset = 0,
        });

        self->scene->buffers.voxel_object_blases.set_blas(self->blas);
        self->blas_device_address = device.device_address(self->blas).value();
    }

    void resize_buffers(GpuContext &gpu_context, RenderVoxelObject *self) {
        PROFILE_FUNC();
        auto &device = gpu_context.device;
        auto brick_count = self->brick_count;

        auto alloc_info = get_bricks_buffer_info(brick_count, self->has_foliage);

        if (device.is_id_valid(self->bricks_data_buffer)) {
            // If the old buffers exist and they're already the same size,
            // Just re-use them and return early.
            auto oldTotalSize = device.buffer_info(self->bricks_data_buffer).value().size;
            if (oldTotalSize == alloc_info.mTotalSize)
                return;

            device.destroy_buffer(self->bricks_data_buffer);
        }

        auto bufferInfo = daxa::BufferInfo{
            .size = alloc_info.mTotalSize,
            .name = "mPerBrickDataBuffer",
        };
        if (brick_count != 0) {
            self->bricks_data_buffer = device.create_buffer(bufferInfo);

            const auto *deviceAddress = (const uint8_t *)device.device_address(self->bricks_data_buffer).value();
            self->brick_primitives_device_address = std::bit_cast<daxa::DeviceAddress>(deviceAddress + alloc_info.mPrimitivesOffset);
            self->brick_shading_device_address = std::bit_cast<daxa::DeviceAddress>(deviceAddress + alloc_info.mShadingOffset);
            self->brick_aabb_device_address = std::bit_cast<daxa::DeviceAddress>(deviceAddress + alloc_info.mAabbOffset);
            self->brick_flags_device_address = std::bit_cast<daxa::DeviceAddress>(deviceAddress + alloc_info.mFlagsOffset);
            self->brick_foliage_device_address = self->has_foliage ? std::bit_cast<daxa::DeviceAddress>(deviceAddress + alloc_info.mFoliageOffset) : daxa::DeviceAddress{0};

            self->scene->buffers.voxel_object_bricks.set_buffer(self->bricks_data_buffer);
        }
        CreateBlas(device, self, "VoxelObject Blas");
    }
} // namespace

void update_render_voxel_object(GpuContext &gpu_context, struct VoxelObject *src) {
    PROFILE_FUNC();
    auto self = src->render_voxel_object;
    if (!src->render_dirty)
        return;
    src->render_dirty = false;

    self->brick_count = 0;
    for (auto brick : src->brick_grid) {
        if (brick == nullptr)
            continue;
        if (brick->render_attribs == nullptr)
            continue;
        self->brick_count++;
    }

    if (self->brick_count == 0)
        return;

    resize_buffers(gpu_context, self);

    auto &device = gpu_context.device;

    auto tempTaskGraph = daxa::TaskGraph({
        .device = device,
        .staging_memory_pool_size = 0,
        .name = "copy old shading data",
    });
    tempTaskGraph.register_blas(self->scene->buffers.voxel_object_blases);
    tempTaskGraph.register_buffer(self->scene->buffers.voxel_object_bricks);
    tempTaskGraph.add_task(
        daxa::InlineTask::Transfer("upload brick data")
            .writes(self->scene->buffers.voxel_object_bricks)
            .executes([&](daxa::TaskInterface ti) {
                // upload all voxel data for now
                auto alloc_info = get_bricks_buffer_info(self->brick_count, self->has_foliage);

                auto staging_buffer = device.create_buffer({
                    .size = alloc_info.mTotalSize,
                    .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "staging_mipmapping_gpu_input_buffer",
                });
                auto *host_address = device.buffer_host_address(staging_buffer).value();
                ti.recorder.destroy_buffer_deferred(staging_buffer);

                int brick_index = 0;
                for (auto brick : src->brick_grid) {
                    if (brick == nullptr)
                        continue;
                    if (brick->render_attribs == nullptr)
                        continue;

                    Aabb &aabb = ((Aabb *)((uint8_t *)host_address + alloc_info.mAabbOffset))[brick_index];
                    BrickPrimitive &primitive = ((BrickPrimitive *)((uint8_t *)host_address + alloc_info.mPrimitivesOffset))[brick_index];
                    VoxelShadingAttribBrick &render_brick = ((VoxelShadingAttribBrick *)((uint8_t *)host_address + alloc_info.mShadingOffset))[brick_index];

                    auto min = glm::ivec3(brick->voxel_min) + brick->brick_i * BRICK_SIZE;
                    auto max = glm::ivec3(brick->voxel_max) + brick->brick_i * BRICK_SIZE + 1;
                    aabb.min = daxa_f32vec3(min.x, min.y, min.z);
                    aabb.max = daxa_f32vec3(max.x, max.y, max.z);
                    memcpy(primitive.bitmap, brick->bitmask, sizeof(primitive.bitmap));
                    primitive.flags = 0;
                    primitive.offset = daxa_i32vec3(min.x, min.y, min.z);
                    primitive.size_x = max.x - min.x;
                    primitive.size_y = max.y - min.y;
                    primitive.size_z = max.z - min.z;
                    memcpy(&render_brick, brick->render_attribs, sizeof(VoxelShadingAttribBrick));

                    if (self->has_foliage) {
                        VoxelFoliageBrick &foliage_brick = ((VoxelFoliageBrick *)((uint8_t *)host_address + alloc_info.mFoliageOffset))[brick_index];
                        memcpy(foliage_brick.bitmask, brick->foliage_bitmask, sizeof(foliage_brick.bitmask));
                    }

                    ++brick_index;
                }

                ti.recorder.copy_buffer_to_buffer({
                    .src_buffer = staging_buffer,
                    .dst_buffer = self->bricks_data_buffer,
                    .src_offset = alloc_info.mPrimitivesOffset,
                    .dst_offset = alloc_info.mPrimitivesOffset,
                    .size = alloc_info.mPrimitivesSize,
                });
                ti.recorder.copy_buffer_to_buffer({
                    .src_buffer = staging_buffer,
                    .dst_buffer = self->bricks_data_buffer,
                    .src_offset = alloc_info.mShadingOffset,
                    .dst_offset = alloc_info.mShadingOffset,
                    .size = alloc_info.mShadingSize,
                });
                ti.recorder.copy_buffer_to_buffer({
                    .src_buffer = staging_buffer,
                    .dst_buffer = self->bricks_data_buffer,
                    .src_offset = alloc_info.mFlagsOffset,
                    .dst_offset = alloc_info.mFlagsOffset,
                    .size = alloc_info.mFlagsSize,
                });
                ti.recorder.copy_buffer_to_buffer({
                    .src_buffer = staging_buffer,
                    .dst_buffer = self->bricks_data_buffer,
                    .src_offset = alloc_info.mAabbOffset,
                    .dst_offset = alloc_info.mAabbOffset,
                    .size = alloc_info.mAabbSize,
                });
                if (self->has_foliage) {
                    ti.recorder.copy_buffer_to_buffer({
                        .src_buffer = staging_buffer,
                        .dst_buffer = self->bricks_data_buffer,
                        .src_offset = alloc_info.mFoliageOffset,
                        .dst_offset = alloc_info.mFoliageOffset,
                        .size = alloc_info.mFoliageSize,
                    });
                }
            }));
    tempTaskGraph.add_task(
        daxa::InlineTask::Transfer("build brick blas")
            .acceleration_structure_build.reads(self->scene->buffers.voxel_object_bricks)
            .acceleration_structure_build.writes(self->scene->buffers.voxel_object_blases)
            .executes([&](daxa::TaskInterface ti) {
                auto geometry = std::array{
                    daxa::BlasAabbGeometryInfo{
                        .data = self->brick_aabb_device_address,
                        .stride = sizeof(Aabb),
                        .count = self->brick_count,
                        .flags = daxa::GeometryFlagBits::OPAQUE,
                    },
                };
                auto blasBuildInfo = daxa::BlasBuildInfo{
                    .flags = daxa::AccelerationStructureBuildFlagBits::PREFER_FAST_TRACE,
                    .dst_blas = self->blas,
                    .geometries = geometry,
                    .scratch_data = ti.device.device_address(self->blas_scratch_buffer).value(),
                };
                ti.recorder.build_acceleration_structures({.blas_build_infos = std::span{&blasBuildInfo, 1}});
                ti.recorder.destroy_buffer_deferred(self->blas_scratch_buffer);
                self->blas_scratch_buffer = {};
            }));

    tempTaskGraph.submit({});
    tempTaskGraph.complete({});
    tempTaskGraph.execute({});
}

void draw_voxel_object(struct VoxelObject *object, const glm::vec3 &pos, const glm::vec3 &angles, float scale, const glm::vec3 &tint) {
    PROFILE_FUNC();
    auto *self = object->render_voxel_object;
    if (self->brick_count == 0)
        return;
    // Ignores rotation, matching the transform below -- good enough for a
    // conservative cull bound.
    auto const world_aabb_min = pos + glm::vec3(object->voxel_min) * scale;
    auto const world_aabb_max = pos + glm::vec3(object->voxel_max) * scale;
    assert(self->scene->drawn_voxel_object_manifests.size < MAX_VOXEL_OBJECTS);
    self->scene->drawn_voxel_object_manifests.push_back(GpuVoxelObject{
        self->brick_shading_device_address,
        self->brick_primitives_device_address,
        self->brick_foliage_device_address,
        self->brick_aabb_device_address,
        {tint.r, tint.g, tint.b},
        self->brick_count,
        {world_aabb_min.x, world_aabb_min.y, world_aabb_min.z},
        {world_aabb_max.x, world_aabb_max.y, world_aabb_max.z},
        {pos.x, pos.y, pos.z},
        scale,
    });
    // auto mat = glm::rotate(glm::mat4(scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, 1), angles.z, glm::vec3(0, 0, 1));
    self->scene->drawn_voxel_objects_blas_instances.push_back(daxa_BlasInstanceData{
        .transform = {
            {scale, 0, 0, pos.x},
            {0, scale, 0, pos.y},
            {0, 0, scale, pos.z},
        },
        .instance_custom_index = (uint32_t)self->scene->drawn_voxel_objects_blas_instances.size,
        .mask = 0xff,
        .instance_shader_binding_table_record_offset = 0,
        .flags = {},
        .blas_device_address = self->blas_device_address,
    });
}
