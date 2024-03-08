#pragma once

#include <core.inl>
#include <application/input.inl>

#include <voxels/brushes.inl>

struct IndirectDrawParams {
    daxa_u32 vertex_count;
    daxa_u32 instance_count;
    daxa_u32 first_vertex;
    daxa_u32 first_instance;
};

struct VoxelParticlesState {
    daxa_u32vec3 simulation_dispatch;
    daxa_u32 place_count;
    daxa_u32vec3 place_bounds_min;
    daxa_u32vec3 place_bounds_max;
    IndirectDrawParams draw_params;
};
DAXA_DECL_BUFFER_PTR(VoxelParticlesState)

struct SimulatedVoxelParticle {
    daxa_f32vec3 pos;
    daxa_f32 duration_alive;
    daxa_f32vec3 vel;
    PackedVoxel packed_voxel;
    daxa_u32 flags;
};
DAXA_DECL_BUFFER_PTR(SimulatedVoxelParticle)

DAXA_DECL_TASK_HEAD_BEGIN(VoxelParticlePerframeCompute, 4 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(GpuOutput), gpu_output)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
VOXELS_USE_BUFFERS(daxa_RWBufferPtr, COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_HEAD_END
struct VoxelParticlePerframeComputePush {
    DAXA_TH_BLOB(VoxelParticlePerframeCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(VoxelParticleSimCompute, 5 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), rendered_voxel_particles)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), placed_voxel_particles)
DAXA_DECL_TASK_HEAD_END
struct VoxelParticleSimComputePush {
    DAXA_TH_BLOB(VoxelParticleSimCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(VoxelParticleRaster, 6)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(daxa_u32), rendered_voxel_particles)
DAXA_TH_IMAGE_INDEX(COLOR_ATTACHMENT, REGULAR_2D, render_image)
DAXA_TH_IMAGE_INDEX(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct VoxelParticleRasterPush {
    DAXA_TH_BLOB(VoxelParticleRaster, uses)
};

#if defined(__cplusplus)

struct VoxelParticles {
    TemporalBuffer global_state;
    TemporalBuffer simulated_voxel_particles;
    TemporalBuffer rendered_voxel_particles;
    TemporalBuffer placed_voxel_particles;

    void record_startup(RecordContext &record_ctx) {
        record_ctx.task_graph.use_persistent_buffer(global_state.task_resource);

        record_ctx.task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, global_state.task_resource),
            },
            .task = [this](daxa::TaskInterface const &ti) {
                ti.recorder.clear_buffer({
                    .buffer = global_state.task_resource.get_state().buffers[0],
                    .offset = 0,
                    .size = sizeof(VoxelParticlesState),
                    .clear_value = 0,
                });
            },
            .name = "Clear",
        });
    }

    void simulate(RecordContext &record_ctx, VoxelWorldBuffers &voxel_world_buffers) {
        global_state = record_ctx.gpu_context->find_or_add_temporal_buffer({
            .size = sizeof(VoxelParticlesState),
            .name = "globals_buffer",
        });
        simulated_voxel_particles = record_ctx.gpu_context->find_or_add_temporal_buffer({
            .size = sizeof(SimulatedVoxelParticle) * std::max<daxa_u32>(MAX_SIMULATED_VOXEL_PARTICLES, 1),
            .name = "simulated_voxel_particles",
        });
        rendered_voxel_particles = record_ctx.gpu_context->find_or_add_temporal_buffer({
            .size = sizeof(daxa_u32) * std::max<daxa_u32>(MAX_RENDERED_VOXEL_PARTICLES, 1),
            .name = "rendered_voxel_particles",
        });
        placed_voxel_particles = record_ctx.gpu_context->find_or_add_temporal_buffer({
            .size = sizeof(daxa_u32) * std::max<daxa_u32>(MAX_SIMULATED_VOXEL_PARTICLES, 1),
            .name = "placed_voxel_particles",
        });

        record_ctx.task_graph.use_persistent_buffer(global_state.task_resource);
        record_ctx.task_graph.use_persistent_buffer(simulated_voxel_particles.task_resource);
        record_ctx.task_graph.use_persistent_buffer(rendered_voxel_particles.task_resource);
        record_ctx.task_graph.use_persistent_buffer(placed_voxel_particles.task_resource);

        if constexpr (MAX_RENDERED_VOXEL_PARTICLES == 0) {
            return;
        }

        record_ctx.add(ComputeTask<VoxelParticlePerframeCompute, VoxelParticlePerframeComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"voxels/particles/perframe.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{VoxelParticlePerframeCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                daxa::TaskViewVariant{std::pair{VoxelParticlePerframeCompute::gpu_output, record_ctx.gpu_context->task_output_buffer}},
                daxa::TaskViewVariant{std::pair{VoxelParticlePerframeCompute::particles_state, global_state.task_resource}},
                daxa::TaskViewVariant{std::pair{VoxelParticlePerframeCompute::simulated_voxel_particles, simulated_voxel_particles.task_resource}},
                VOXELS_BUFFER_USES_ASSIGN(VoxelParticlePerframeCompute, voxel_world_buffers),
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, VoxelParticlePerframeComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({1, 1, 1});
            },
        });
        record_ctx.add(ComputeTask<VoxelParticleSimCompute, VoxelParticleSimComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"voxels/particles/sim.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{VoxelParticleSimCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                daxa::TaskViewVariant{std::pair{VoxelParticleSimCompute::particles_state, global_state.task_resource}},
                VOXELS_BUFFER_USES_ASSIGN(VoxelParticleSimCompute, voxel_world_buffers),
                daxa::TaskViewVariant{std::pair{VoxelParticleSimCompute::simulated_voxel_particles, simulated_voxel_particles.task_resource}},
                daxa::TaskViewVariant{std::pair{VoxelParticleSimCompute::rendered_voxel_particles, rendered_voxel_particles.task_resource}},
                daxa::TaskViewVariant{std::pair{VoxelParticleSimCompute::placed_voxel_particles, placed_voxel_particles.task_resource}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, VoxelParticleSimComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = ti.get(VoxelParticleSimCompute::particles_state).ids[0],
                    .offset = offsetof(VoxelParticlesState, simulation_dispatch),
                });
            },
        });
    }

    auto render(RecordContext &record_ctx) -> std::pair<daxa::TaskImageView, daxa::TaskImageView> {
        auto format = daxa::Format::R32_UINT;
        auto raster_color_image = record_ctx.task_graph.create_transient_image({
            .format = format,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "raster_color_image",
        });
        auto raster_depth_image = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::D32_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "raster_depth_image",
        });

        record_ctx.add(RasterTask<VoxelParticleRaster, VoxelParticleRasterPush, NoTaskInfo>{
            .vert_source = daxa::ShaderFile{"voxels/particles/cube.raster.glsl"},
            .frag_source = daxa::ShaderFile{"voxels/particles/cube.raster.glsl"},
            .color_attachments = {{
                .format = format,
            }},
            .depth_test = daxa::DepthTestInfo{
                .depth_attachment_format = daxa::Format::D32_SFLOAT,
                .enable_depth_write = true,
                .depth_test_compare_op = daxa::CompareOp::GREATER,
            },
            .raster = {
                .face_culling = daxa::FaceCullFlagBits::BACK_BIT,
            },
            .views = std::array{
                daxa::TaskViewVariant{std::pair{VoxelParticleRaster::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                daxa::TaskViewVariant{std::pair{VoxelParticleRaster::particles_state, global_state.task_resource}},
                daxa::TaskViewVariant{std::pair{VoxelParticleRaster::simulated_voxel_particles, simulated_voxel_particles.task_resource}},
                daxa::TaskViewVariant{std::pair{VoxelParticleRaster::rendered_voxel_particles, rendered_voxel_particles.task_resource}},
                daxa::TaskViewVariant{std::pair{VoxelParticleRaster::render_image, raster_color_image}},
                daxa::TaskViewVariant{std::pair{VoxelParticleRaster::depth_image_id, raster_depth_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, VoxelParticleRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(VoxelParticleRaster::render_image).ids[0]).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .color_attachments = {{.image_view = ti.get(VoxelParticleRaster::render_image).ids[0].default_view(), .load_op = daxa::AttachmentLoadOp::CLEAR, .clear_value = std::array<daxa_f32, 4>{0.0f, 0.0f, 0.0f, 0.0f}}},
                    .depth_attachment = {{.image_view = ti.get(VoxelParticleRaster::depth_image_id).ids[0].default_view(), .load_op = daxa::AttachmentLoadOp::CLEAR, .clear_value = daxa::DepthValue{0.0f, 0}}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(VoxelParticleRaster::particles_state).ids[0],
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, draw_params),
                    .is_indexed = false,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "voxel particles", .task_image_id = raster_color_image, .type = DEBUG_IMAGE_TYPE_DEFAULT_UINT});
        debug_utils::DebugDisplay::add_pass({.name = "voxel particles depth", .task_image_id = raster_depth_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        return {raster_color_image, raster_depth_image};
    }
};

#endif
