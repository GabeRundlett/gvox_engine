#include "voxels/defs.inl"
#define RENDERER_INTERNAL 1
#include "render_scene.hpp"

struct RenderScene *create_render_scene(struct GpuContext &gpu_context) {
    RenderScene *result = new RenderScene();
    result->buffers.task_tlas_instances = daxa::TaskBuffer({.name = "task_tlas_instances"});
    result->buffers.task_tlas = daxa::TaskTlas({.name = "task_tlas"});
    result->buffers.voxel_object_blases = daxa::TaskBlas({.name = "voxel_object_blases"});
    result->buffers.voxel_object_bricks = daxa::TaskBuffer({.name = "voxel_object_bricks"});
    return result;
}

void destroy_render_scene(struct GpuContext &gpu_context, struct RenderScene *self) {
    auto &device = gpu_context.device;
    if (device.is_id_valid(self->buffers.tlas))
        device.destroy_tlas(self->buffers.tlas);
    if (device.is_id_valid(self->buffers.tlas_buffer))
        device.destroy_buffer(self->buffers.tlas_buffer);
    if (device.is_id_valid(self->buffers.tlas_scratch_buffer))
        device.destroy_buffer(self->buffers.tlas_scratch_buffer);
    for (auto buffer : self->buffers.task_tlas_instances.get_state().buffers)
        if (device.is_id_valid(buffer))
            device.destroy_buffer(buffer);

    delete self;
}

inline auto GetAligned(daxa_u64 operand, daxa_u64 granularity) -> daxa_u64 {
    return ((operand + (granularity - 1)) / granularity) * granularity;
}

void render_scene_begin(struct GpuContext &gpu_context, struct RenderScene *self) {
    self->drawn_voxel_objects.clear();
    self->drawn_voxel_objects_blas_instances.clear();
}

void render_scene_end(struct GpuContext &gpu_context, struct RenderScene *self) {
    uint32_t new_tlas_instance_count = self->drawn_voxel_objects.size();
    if (new_tlas_instance_count != self->buffers.tlas_instance_count) {
        auto &device = gpu_context.device;
        auto tlasInstanceInfo = std::array{
            daxa::TlasInstanceInfo{
                .data = {},
                .count = new_tlas_instance_count,
                .is_data_array_of_pointers = false, // Buffer contains flat array of instances, not an array of pointers to instances.
                .flags = daxa::GeometryFlagBits::OPAQUE,
            },
        };
        auto tlasBuildInfo = daxa::TlasBuildInfo{
            .flags = daxa::AccelerationStructureBuildFlagBits::PREFER_FAST_TRACE,
            .dst_tlas = {}, // Ignored in get_acceleration_structure_build_sizes.
            .instances = tlasInstanceInfo,
            .scratch_data = {}, // Ignored in get_acceleration_structure_build_sizes.
        };
        auto tlasBuildSizes = device.get_tlas_build_sizes(tlasBuildInfo);
        auto accelerationStructureScratchOffsetAlignment = device.properties().acceleration_structure_properties.value().min_acceleration_structure_scratch_offset_alignment;
        auto instancesSize = sizeof(daxa_BlasInstanceData) * new_tlas_instance_count;

        bool createTlasBuffer = true;
        if (device.is_id_valid(self->buffers.tlas_buffer)) {
            if (device.info_buffer(self->buffers.tlas_buffer).value().size >= tlasBuildSizes.acceleration_structure_size)
                createTlasBuffer = false;
            else
                device.destroy_buffer(self->buffers.tlas_buffer);
        }

        bool createScratchBuffer = true;
        if (device.is_id_valid(self->buffers.tlas_scratch_buffer)) {
            if (device.info_buffer(self->buffers.tlas_scratch_buffer).value().size >= GetAligned(tlasBuildSizes.build_scratch_size, accelerationStructureScratchOffsetAlignment))
                createScratchBuffer = false;
            else
                device.destroy_buffer(self->buffers.tlas_scratch_buffer);
        }

        bool createInstancesBuffer = instancesSize > 0;
        auto instances_buffer = daxa::BufferId{};
        if (!self->buffers.task_tlas_instances.get_state().buffers.empty())
            instances_buffer = self->buffers.task_tlas_instances.get_state().buffers[0];
        if (device.is_id_valid(instances_buffer)) {
            if (device.info_buffer(instances_buffer).value().size >= instancesSize)
                createInstancesBuffer = false;
            else
                device.destroy_buffer(instances_buffer);
        }

        if (createTlasBuffer)
            self->buffers.tlas_buffer = device.create_buffer({
                .size = tlasBuildSizes.acceleration_structure_size,
                .name = ("tlas buffer"),
            });
        if (createScratchBuffer)
            self->buffers.tlas_scratch_buffer = device.create_buffer({
                .size = GetAligned(tlasBuildSizes.build_scratch_size, accelerationStructureScratchOffsetAlignment),
                .name = ("tlas scratch buffer"),
            });
        if (createInstancesBuffer) {
            instances_buffer = device.create_buffer({
                .size = instancesSize,
                .name = ("tlas instances buffer"),
            });
        }

        self->buffers.task_tlas_instances.set_buffers({.buffers = std::array{instances_buffer}});

        if (!self->buffers.tlas.is_empty())
            device.destroy_tlas(self->buffers.tlas);

        self->buffers.tlas = device.create_tlas_from_buffer({
            .tlas_info = {
                .size = tlasBuildSizes.acceleration_structure_size,
                .name = "tlas",
            },
            .buffer_id = self->buffers.tlas_buffer,
            .offset = 0,
        });
        self->buffers.task_tlas.set_tlas({.tlas = std::array{self->buffers.tlas}});
    }
}

void record_render_scene(struct GpuContext &gpu_context, struct RenderScene *self) {
    auto &task_graph = gpu_context.frame_task_graph;

    self->buffers.brick_primitive_pointers = gpu_context.find_or_add_temporal_buffer({
        .size = sizeof(daxa::DeviceAddress) * MAX_VOXEL_OBJECTS,
        .name = "blas_attr_pointers",
    });

    task_graph.use_persistent_buffer(self->buffers.brick_primitive_pointers.task_resource);

    task_graph.use_persistent_buffer(self->buffers.task_tlas_instances);
    task_graph.use_persistent_tlas(self->buffers.task_tlas);
    task_graph.use_persistent_blas(self->buffers.voxel_object_blases);
    task_graph.use_persistent_buffer(self->buffers.voxel_object_bricks);

    task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, self->buffers.task_tlas_instances),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, self->buffers.brick_primitive_pointers.task_resource),
        },
        .task = [self](const daxa::TaskInterface &ti) {
            auto tlasInstanceN = self->drawn_voxel_objects.size();
            auto staging_allocation = ti.allocator->allocate(sizeof(daxa_BlasInstanceData) * tlasInstanceN);
            if (tlasInstanceN == 0)
                return;

            memcpy(staging_allocation->host_address, self->drawn_voxel_objects_blas_instances.data(), staging_allocation->size);
            // TODO: Think more about updating the tlas. Shouldn't need to update every single object, only ones that changed
            // NOTE: Don't forget about deletion of objects, this should mark the re-allocated object as a dirty tlas!
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = ti.allocator->buffer(),
                .dst_buffer = ti.get(daxa::TaskBufferAttachmentIndex{0}).ids[0],
                .size = staging_allocation->size,
            });

            staging_allocation = ti.allocator->allocate(sizeof(daxa::DeviceAddress) * tlasInstanceN);
            memcpy(staging_allocation->host_address, self->drawn_voxel_objects.data(), staging_allocation->size);
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = ti.allocator->buffer(),
                .dst_buffer = ti.get(daxa::TaskBufferAttachmentIndex{1}).ids[0],
                .size = staging_allocation->size,
            });
        },
        .name = "update tlas instances",
    });

    task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBlasAccess::BUILD_READ, self->buffers.voxel_object_blases),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, self->buffers.task_tlas_instances),
            daxa::inl_attachment(daxa::TaskTlasAccess::BUILD_WRITE, self->buffers.task_tlas),
        },
        .task = [self](const daxa::TaskInterface &ti) {
            auto tlasInstanceN = self->drawn_voxel_objects.size();
            auto tlasInstancesBuffer = ti.get(self->buffers.task_tlas_instances).ids[0];
            auto tlasInstanceInfo = std::array{
                daxa::TlasInstanceInfo{
                    .data = tlasInstanceN == 0 ? daxa::DeviceAddress{} : ti.device.get_device_address(tlasInstancesBuffer).value(),
                    .count = static_cast<uint32_t>(tlasInstanceN),
                    .is_data_array_of_pointers = false, // Buffer contains flat array of instances, not an array of pointers to instances.
                    .flags = daxa::GeometryFlagBits::OPAQUE,
                },
            };
            auto tlasBuildInfo = daxa::TlasBuildInfo{
                .flags = daxa::AccelerationStructureBuildFlagBits::PREFER_FAST_TRACE,
                .dst_tlas = self->buffers.tlas,
                .instances = tlasInstanceInfo,
                .scratch_data = ti.device.get_device_address(self->buffers.tlas_scratch_buffer).value(),
            };
            ti.recorder.build_acceleration_structures({
                .tlas_build_infos = std::array{tlasBuildInfo},
            });
        },
        .name = "tlas build",
    });
}
