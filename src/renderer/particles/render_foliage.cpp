#include "render_foliage.hpp"
#include "voxels/particles/grass/grass.inl"
#include <renderer/gpu_context.hpp>
#include <renderer/kajiya/gbuffer.hpp>
#define RENDERER_INTERNAL 1
#include <renderer/render_scene.hpp>
#include "render_foliage.inl"

#include <array>
#include <cstddef>

void init_render_foliage_bricks(GpuContext &gpu_context) {
    register_pipeline(
        gpu_context.pipeline_manager,
        ComputePipelineCompileInfo{
            .out_pipeline = &gpu_context.foliage_bricks.generate_pipeline,
            .source_path = "particles/foliage_generate.comp.glsl",
            .push_constant_size = sizeof(FoliageGeneratePush),
            .name = "FoliageGenerate",
        });
}

void deinit_render_foliage_bricks(GpuContext &gpu_context) {
    delete gpu_context.foliage_bricks.grass;
    delete gpu_context.foliage_bricks.particles_state;
    delete gpu_context.foliage_bricks.cube_index_buffer;
    delete gpu_context.foliage_bricks.visible_foliage_bricks;
}

namespace {
    // One-time setup for the grass-blade side of foliage rendering: the
    // persistent GrassStrandAllocator, the shared particle draw-param state,
    // and the fixed cube index buffer. Guarded so a task-graph re-record
    // (settings toggle etc.) doesn't wipe out already-spawned grass.
    void init_foliage_grass(GpuContext &gpu_context) {
        static constexpr auto cube_indices = std::array<uint16_t, 8>{0, 1, 2, 3, 4, 5, 6, 1};
        auto cube_index_buffer = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(cube_indices),
            .name = "foliage.cube_index_buffer",
        });
        gpu_context.foliage_bricks.cube_index_buffer = new TemporalBuffer(cube_index_buffer);

        auto particles_state = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(VoxelParticlesState),
            .name = "foliage.particles_state",
        });
        gpu_context.foliage_bricks.particles_state = new TemporalBuffer(particles_state);

        // Slot 0 = atomic visible-object count, slots [1..] = object indices.
        auto visible_foliage_bricks = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(FoliageBrickInstance) * (1 + MAX_FOLIAGE_BRICK_COUNT),
            .name = "foliage.visible_foliage_bricks",
        });
        gpu_context.foliage_bricks.visible_foliage_bricks = new TemporalBuffer(visible_foliage_bricks);

        // Synchronous one-time upload/clear, same reasoning as
        // StaticAllocatorBufferState::init (startup_task_graph never runs).
        auto temp_task_graph = daxa::TaskGraph({.device = gpu_context.device, .name = "foliage grass init"});
        temp_task_graph.register_buffer(cube_index_buffer.task_resource);
        temp_task_graph.register_buffer(particles_state.task_resource);
        auto const cube_index_buffer_id = cube_index_buffer.task_resource.id();
        auto const particles_state_id = particles_state.task_resource.id();

        temp_task_graph.add_task(
            daxa::InlineTask::Transfer("foliage cube index upload")
                .writes(cube_index_buffer.task_resource)
                .executes([cube_index_buffer_id](daxa::TaskInterface ti) {
                    auto staging_buffer = ti.device.create_buffer({
                        .size = sizeof(cube_indices),
                        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                        .name = "foliage_cube_index_staging_buffer",
                    });
                    ti.recorder.destroy_buffer_deferred(staging_buffer);
                    auto *buffer_ptr = ti.device.buffer_host_address_as<std::remove_cv_t<decltype(cube_indices)>>(staging_buffer).value();
                    *buffer_ptr = cube_indices;
                    ti.recorder.copy_buffer_to_buffer({
                        .src_buffer = staging_buffer,
                        .dst_buffer = cube_index_buffer_id,
                        .size = sizeof(cube_indices),
                    });
                }));
        temp_task_graph.add_task(
            daxa::InlineTask::Transfer("foliage particles state clear")
                .writes(particles_state.task_resource)
                .executes([particles_state_id](daxa::TaskInterface ti) {
                    ti.recorder.clear_buffer({
                        .buffer = particles_state_id,
                        .offset = 0,
                        .size = sizeof(VoxelParticlesState),
                        .clear_value = 0,
                    });
                }));
        temp_task_graph.submit({});
        temp_task_graph.complete({});
        temp_task_graph.execute({});

        gpu_context.foliage_bricks.grass = new GrassStrands();
        gpu_context.foliage_bricks.grass->init(gpu_context);
    }
} // namespace

void record_render_foliage_bricks(GpuContext &gpu_context, daxa::TaskGraph &task_graph, daxa::TaskImageView hiz, RenderScene *scene) {
    bool const first_time = !gpu_context.foliage_bricks.grass_initialized;
    if (first_time) {
        gpu_context.foliage_bricks.grass_initialized = true;
        init_foliage_grass(gpu_context);
    }

    auto &grass = *gpu_context.foliage_bricks.grass;

    // grass_allocator.init() (called from init_foliage_grass above) already
    // registers its 4 buffers directly against this same task_graph on the
    // first call -- only re-register on a later re-record (settings toggle
    // etc.), where task_graph is a freshly-recreated frame_task_graph instance.
    if (!first_time) {
        task_graph.register_buffer(grass.grass_allocator.allocator_buffer.task_resource);
        task_graph.register_buffer(grass.grass_allocator.element_buffer.task_resource);
    }
    // particles_state/cube_index_buffer were only registered (and uploaded)
    // against the throwaway init task_graph above, not this one -- always
    // register them here.
    task_graph.register_buffer(gpu_context.foliage_bricks.particles_state->task_resource);
    task_graph.register_buffer(gpu_context.foliage_bricks.cube_index_buffer->task_resource);

    task_graph.add_task(
        daxa::InlineTask::Transfer("reset grass draw params")
            .writes(gpu_context.foliage_bricks.particles_state->task_resource)
            .executes([&gpu_context](daxa::TaskInterface ti) {
                ParticleDrawParams params{};
                params.cube_draw_params.index_count = 8;
                params.shadow_cube_draw_params.index_count = 8;
                params.splat_draw_params.instance_count = 1;
                auto alloc = ti.allocator->allocate(sizeof(ParticleDrawParams));
                *static_cast<ParticleDrawParams *>(alloc->host_address) = params;
                ti.recorder.copy_buffer_to_buffer({
                    .src_buffer = ti.allocator->buffer(),
                    .dst_buffer = gpu_context.foliage_bricks.particles_state->task_resource.id(),
                    .src_offset = alloc->buffer_offset,
                    .dst_offset = offsetof(VoxelParticlesState, grass),
                    .size = sizeof(ParticleDrawParams),
                });
            }));

    task_graph.register_buffer(gpu_context.foliage_bricks.visible_foliage_bricks->task_resource);
    task_graph.add_task(
        daxa::InlineTask::Transfer("clear visible chunks count")
            .writes(gpu_context.foliage_bricks.visible_foliage_bricks->task_resource.view())
            .writes(grass.grass_allocator.allocator_buffer.task_resource)
            .executes([&gpu_context](daxa::TaskInterface ti) {
                ti.recorder.clear_buffer({
                    .buffer = gpu_context.foliage_bricks.visible_foliage_bricks->task_resource.id(),
                    .offset = 0,
                    .size = sizeof(uint32_t),
                    .clear_value = 0,
                });
                ti.recorder.clear_buffer({
                    .buffer = gpu_context.foliage_bricks.visible_foliage_bricks->task_resource.id(),
                    .offset = 4,
                    .size = sizeof(uint32_t),
                    .clear_value = 1,
                });
                ti.recorder.clear_buffer({
                    .buffer = gpu_context.foliage_bricks.visible_foliage_bricks->task_resource.id(),
                    .offset = 8,
                    .size = sizeof(uint32_t),
                    .clear_value = 1,
                });

                auto &grass = *gpu_context.foliage_bricks.grass;
                ti.recorder.clear_buffer({
                    .buffer = grass.grass_allocator.allocator_buffer.task_resource.id(),
                    .offset = offsetof(GrassStrandAllocator, element_count),
                    .size = sizeof(uint32_t),
                    .clear_value = 0,
                });
            }));

    if (scene != nullptr) {
        gpu_context.add(ComputeTask<FoliageCull::Info, FoliageCullPush, FoliageCullInfo>{
            .source = "particles/foliage_cull.comp.glsl",
            .views = FoliageCull::Views{
                .gpu_input = gpu_context.task_input_buffer.view(),
                .voxel_object_manifests = scene->buffers.voxel_object_manifests.task_resource.view(),
                .visible_foliage_bricks = gpu_context.foliage_bricks.visible_foliage_bricks->task_resource.view(),
                .hiz = hiz,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, FoliageCullPush &push, FoliageCullInfo const &info) {
                uint32_t const object_count = info.scene != nullptr ? static_cast<uint32_t>(info.scene->drawn_voxel_object_manifests.size) : 0;
                if (object_count == 0)
                    return;
                push.object_count = object_count;
                push.hiz_mip_count = ti.get(FoliageCull::AT.hiz).view.slice.level_count;
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({object_count, 1, 1});
            },
            .info = FoliageCullInfo{.scene = scene},
            .task_graph_ptr = &task_graph,
        });
    }

    task_graph.add_task(
        daxa::InlineTask::Compute("generate foliage")
            .indirect_cmd.reads(gpu_context.foliage_bricks.visible_foliage_bricks->task_resource.view())
            .writes(grass.grass_allocator.allocator_buffer.task_resource,
                    grass.grass_allocator.element_buffer.task_resource)
            .executes([&gpu_context](daxa::TaskInterface ti) {
                auto &pipeline = gpu_context.foliage_bricks.generate_pipeline;
                if (!pipeline.is_valid()) {
                    return;
                }
                auto &grass = *gpu_context.foliage_bricks.grass;
                auto const push = FoliageGeneratePush{
                    .visible_foliage_bricks = ti.device.device_address(gpu_context.foliage_bricks.visible_foliage_bricks->task_resource.id()).value(),
                    .grass_allocator = ti.device.device_address(grass.grass_allocator.allocator_buffer.task_resource.id()).value(),
                };
                ti.recorder.set_pipeline(pipeline);
                ti.recorder.push_constant(push);

                ti.recorder.dispatch_indirect({
                    .indirect_buffer = gpu_context.foliage_bricks.visible_foliage_bricks->task_resource.id(),
                    .offset = 0,
                });
            }));

    grass.simulate(gpu_context, gpu_context.foliage_bricks.particles_state->task_resource);
}

auto render_foliage_grass(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, daxa::TaskImageView velocity_image) -> daxa::TaskImageView {
    if (gpu_context.foliage_bricks.grass == nullptr) {
        return daxa::NullTaskImage;
    }

    auto raster_shadow_depth_image = gpu_context.frame_task_graph.create_task_image({
        .format = daxa::Format::D32_SFLOAT,
        .size = {2048, 2048, 1},
        .name = "foliage_raster_shadow_depth_image",
    });

    auto &grass = *gpu_context.foliage_bricks.grass;
    grass.render_cubes(
        gpu_context, gbuffer_depth, velocity_image, raster_shadow_depth_image,
        gpu_context.foliage_bricks.particles_state->task_resource,
        gpu_context.foliage_bricks.cube_index_buffer->task_resource);
    grass.render_splats(
        gpu_context, gbuffer_depth, velocity_image, raster_shadow_depth_image,
        gpu_context.foliage_bricks.particles_state->task_resource);

    return raster_shadow_depth_image;
}
