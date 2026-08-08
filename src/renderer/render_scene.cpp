#include "voxels/defs.inl"
#define RENDERER_INTERNAL 1
#include "render_scene.hpp"

struct RenderScene *create_render_scene(struct GpuContext &gpu_context) {
    RenderScene *result = new RenderScene();
    result->buffers.task_tlas_instances = daxa::ExternalTaskBuffer({.name = "task_tlas_instances"});
    result->buffers.task_tlas = daxa::ExternalTaskTlas({.name = "task_tlas"});
    result->buffers.voxel_object_blases = daxa::ExternalTaskBlas({.name = "voxel_object_blases"});
    result->buffers.voxel_object_bricks = daxa::ExternalTaskBuffer({.name = "voxel_object_bricks"});
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
    if (device.is_id_valid(self->buffers.task_tlas_instances.id()))
        device.destroy_buffer(self->buffers.task_tlas_instances.id());

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
        auto tlasBuildSizes = device.tlas_build_sizes(tlasBuildInfo);
        auto accelerationStructureScratchOffsetAlignment = device.properties().acceleration_structure_properties.value().min_acceleration_structure_scratch_offset_alignment;
        auto instancesSize = sizeof(daxa_BlasInstanceData) * new_tlas_instance_count;

        bool createTlasBuffer = true;
        if (device.is_id_valid(self->buffers.tlas_buffer)) {
            if (device.buffer_info(self->buffers.tlas_buffer).value().size >= tlasBuildSizes.acceleration_structure_size)
                createTlasBuffer = false;
            else
                device.destroy_buffer(self->buffers.tlas_buffer);
        }

        bool createScratchBuffer = true;
        if (device.is_id_valid(self->buffers.tlas_scratch_buffer)) {
            if (device.buffer_info(self->buffers.tlas_scratch_buffer).value().size >= GetAligned(tlasBuildSizes.build_scratch_size, accelerationStructureScratchOffsetAlignment))
                createScratchBuffer = false;
            else
                device.destroy_buffer(self->buffers.tlas_scratch_buffer);
        }

        bool createInstancesBuffer = instancesSize > 0;
        auto instances_buffer = daxa::BufferId{};
        if (self->buffers.task_tlas_instances.is_valid())
            instances_buffer = self->buffers.task_tlas_instances.id();
        if (device.is_id_valid(instances_buffer)) {
            if (device.buffer_info(instances_buffer).value().size >= instancesSize)
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

        self->buffers.task_tlas_instances.set_buffer(instances_buffer);

        if (!self->buffers.tlas.is_empty())
            device.destroy_tlas(self->buffers.tlas);

        self->buffers.tlas = device.create_tlas_from_buffer({
            .tlas_info = {
                .size = tlasBuildSizes.acceleration_structure_size,
                .name = "tlas",
            },
            .buffer = self->buffers.tlas_buffer,
            .offset = 0,
        });
        self->buffers.task_tlas.set_tlas(self->buffers.tlas);
    }
}

void record_render_scene(struct GpuContext &gpu_context, struct RenderScene *self) {
    auto &task_graph = gpu_context.frame_task_graph;

    self->buffers.brick_primitive_pointers = gpu_context.find_or_add_temporal_buffer({
        .size = sizeof(daxa::DeviceAddress) * MAX_VOXEL_OBJECTS,
        .name = "blas_attr_pointers",
    });

    task_graph.register_buffer(self->buffers.brick_primitive_pointers.task_resource);

    task_graph.register_buffer(self->buffers.task_tlas_instances);
    task_graph.register_tlas(self->buffers.task_tlas);
    task_graph.register_blas(self->buffers.voxel_object_blases);
    task_graph.register_buffer(self->buffers.voxel_object_bricks);

    task_graph.add_task(
        daxa::InlineTask::Transfer("update tlas instances")
            .writes(self->buffers.task_tlas_instances, self->buffers.brick_primitive_pointers.task_resource)
            .executes([self](daxa::TaskInterface ti) {
                auto tlasInstanceN = self->drawn_voxel_objects.size();
                auto staging_allocation = ti.allocator->allocate(sizeof(daxa_BlasInstanceData) * tlasInstanceN);
                if (tlasInstanceN == 0)
                    return;

                memcpy(staging_allocation->host_address, self->drawn_voxel_objects_blas_instances.data(), staging_allocation->size);
                // TODO: Think more about updating the tlas. Shouldn't need to update every single object, only ones that changed
                // NOTE: Don't forget about deletion of objects, this should mark the re-allocated object as a dirty tlas!
                ti.recorder.copy_buffer_to_buffer({
                    .src_buffer = ti.allocator->buffer(),
                    .dst_buffer = ti.get(self->buffers.task_tlas_instances).id,
                    .size = staging_allocation->size,
                });

                staging_allocation = ti.allocator->allocate(sizeof(daxa::DeviceAddress) * tlasInstanceN);
                memcpy(staging_allocation->host_address, self->drawn_voxel_objects.data(), staging_allocation->size);
                ti.recorder.copy_buffer_to_buffer({
                    .src_buffer = ti.allocator->buffer(),
                    .dst_buffer = ti.get(self->buffers.brick_primitive_pointers.task_resource).id,
                    .size = staging_allocation->size,
                });
            }));

    task_graph.add_task(
        daxa::InlineTask::Transfer("tlas build")
            .acceleration_structure_build.reads(self->buffers.voxel_object_blases)
            .transfer.reads(self->buffers.task_tlas_instances)
            .acceleration_structure_build.writes(self->buffers.task_tlas)
            .executes([self](daxa::TaskInterface ti) {
                auto tlasInstanceN = self->drawn_voxel_objects.size();
                auto tlasInstancesBuffer = ti.get(self->buffers.task_tlas_instances).id;
                auto tlasInstanceInfo = std::array{
                    daxa::TlasInstanceInfo{
                        .data = tlasInstanceN == 0 ? daxa::DeviceAddress{} : ti.device.device_address(tlasInstancesBuffer).value(),
                        .count = static_cast<uint32_t>(tlasInstanceN),
                        .is_data_array_of_pointers = false, // Buffer contains flat array of instances, not an array of pointers to instances.
                        .flags = daxa::GeometryFlagBits::OPAQUE,
                    },
                };
                auto tlasBuildInfo = daxa::TlasBuildInfo{
                    .flags = daxa::AccelerationStructureBuildFlagBits::PREFER_FAST_TRACE,
                    .dst_tlas = self->buffers.tlas,
                    .instances = tlasInstanceInfo,
                    .scratch_data = ti.device.device_address(self->buffers.tlas_scratch_buffer).value(),
                };
                ti.recorder.build_acceleration_structures({
                    .tlas_build_infos = std::array{tlasBuildInfo},
                });
            }));
}
