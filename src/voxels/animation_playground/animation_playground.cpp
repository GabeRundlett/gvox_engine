#include "animation_playground.hpp"
#include "animation_playground.inl"

#include "application/input.inl"
#include "renderer/render_scene.hpp"
#include "renderer/render_voxel_object.hpp"
#include "renderer/renderer.hpp"
#include "voxels/voxel.inl"
#include "voxels/voxel_object.hpp"

#include <chrono>
#include <imgui.h>
#include <cstring>

static void destroy_frame(GpuContext &gpu_context, VoxelObject *frame) {
    PROFILE_FUNC();
    destroy_render_voxel_object(gpu_context, frame->render_voxel_object);
    for (auto *brick : frame->brick_grid) {
        if (brick != nullptr) {
            delete brick->render_attribs;
        }
    }
    delete frame; // ~VoxelObject frees each VoxelBrick.
}

static void resize_buffer(daxa::Device &device, daxa::BufferId &id, size_t size, daxa::MemoryFlags flags, char const *name) {
    PROFILE_FUNC();
    if (device.is_id_valid(id)) {
        if (device.buffer_info(id).value().size == size) {
            return;
        }
        device.destroy_buffer(id);
    }
    if (size == 0) {
        return;
    }
    id = device.create_buffer({.size = size, .memory_flags = flags, .name = name});
}

AnimationPlayground::AnimationPlayground(GpuContext &gpu_context, RenderScene *render_scene)
    : gpu_context(gpu_context), render_scene(render_scene) {
    // Registered here, compiled later in bulk (see record_tasks -> compile_all_shaders).
    // `pipeline` is a stable member address, so the closure below stays valid
    // across hot-reloads.
    register_pipeline(
        gpu_context.pipeline_manager,
        ComputePipelineCompileInfo{
            .out_pipeline = &pipeline,
            .source_path = "voxels/animation_playground/generate.comp.glsl",
            .push_constant_size = sizeof(AnimationPlaygroundGenPush),
            .name = "AnimationPlaygroundGenerate",
        });
}

AnimationPlayground::~AnimationPlayground() {
    auto &device = gpu_context.device;
    for (auto *frame : frames) {
        destroy_frame(gpu_context, frame);
    }
    if (device.is_id_valid(bricks_buffer)) {
        device.destroy_buffer(bricks_buffer);
    }
    if (device.is_id_valid(brick_attribs_buffer)) {
        device.destroy_buffer(brick_attribs_buffer);
    }
    if (device.is_id_valid(bricks_readback_buffer)) {
        device.destroy_buffer(bricks_readback_buffer);
    }
    if (device.is_id_valid(brick_attribs_readback_buffer)) {
        device.destroy_buffer(brick_attribs_readback_buffer);
    }
}

void AnimationPlayground::regenerate(float time) {
    PROFILE_FUNC();
    dirty = false;

    if (grid_dims_bricks.x <= 0 || grid_dims_bricks.y <= 0 || grid_dims_bricks.z <= 0 || frame_count <= 0) {
        return;
    }
    if (!pipeline.is_valid()) {
        return;
    }

    using Clock = std::chrono::high_resolution_clock;

    auto &device = gpu_context.device;

    auto const bricks_per_frame = static_cast<uint32_t>(grid_dims_bricks.x * grid_dims_bricks.y * grid_dims_bricks.z);
    auto const total_bricks = static_cast<size_t>(bricks_per_frame) * static_cast<size_t>(frame_count);

    auto const bricks_size = total_bricks * sizeof(BrickPrimitive);
    auto const attribs_size = total_bricks * sizeof(VoxelShadingAttribBrick);

    resize_buffer(device, bricks_buffer, bricks_size, {}, "animation_playground.bricks");
    resize_buffer(device, brick_attribs_buffer, attribs_size, {}, "animation_playground.brick_attribs");
    resize_buffer(device, bricks_readback_buffer, bricks_size, daxa::MemoryFlagBits::HOST_ACCESS_RANDOM, "animation_playground.bricks_readback");
    resize_buffer(device, brick_attribs_readback_buffer, attribs_size, daxa::MemoryFlagBits::HOST_ACCESS_RANDOM, "animation_playground.brick_attribs_readback");

    {
        PROFILE_SCOPE("Record and launch GPU work");
        auto task_bricks_buffer = daxa::ExternalTaskBuffer({.buffer = bricks_buffer, .name = "task_bricks_buffer"});
        auto task_brick_attribs_buffer = daxa::ExternalTaskBuffer({.buffer = brick_attribs_buffer, .name = "task_brick_attribs_buffer"});
        auto task_bricks_readback_buffer = daxa::ExternalTaskBuffer({.buffer = bricks_readback_buffer, .name = "task_bricks_readback_buffer"});
        auto task_brick_attribs_readback_buffer = daxa::ExternalTaskBuffer({.buffer = brick_attribs_readback_buffer, .name = "task_brick_attribs_readback_buffer"});

        auto task_graph = daxa::TaskGraph({.device = device, .staging_memory_pool_size = 0, .name = "animation playground regenerate"});
        task_graph.register_buffer(task_bricks_buffer);
        task_graph.register_buffer(task_brick_attribs_buffer);
        task_graph.register_buffer(task_bricks_readback_buffer);
        task_graph.register_buffer(task_brick_attribs_readback_buffer);

        auto const push = AnimationPlaygroundGenPush{
            .bricks = device.device_address(bricks_buffer).value(),
            .brick_attribs = device.device_address(brick_attribs_buffer).value(),
            .grid_dims_bricks = {grid_dims_bricks.x, grid_dims_bricks.y, grid_dims_bricks.z},
            .frame_count = static_cast<daxa_u32>(frame_count),
            .time = time,
        };
        auto const dispatch_z = static_cast<daxa_u32>(grid_dims_bricks.z * frame_count);
        auto &pipeline_ref = pipeline;
        auto const &bricks_buffer_ref = bricks_buffer;
        auto const &brick_attribs_buffer_ref = brick_attribs_buffer;
        auto const &bricks_readback_buffer_ref = bricks_readback_buffer;
        auto const &brick_attribs_readback_buffer_ref = brick_attribs_readback_buffer;

        task_graph.add_task(
            daxa::InlineTask::Compute("animation playground generate")
                .writes(task_bricks_buffer)
                .writes(task_brick_attribs_buffer)
                .executes([&pipeline_ref, push, grid_dims_bricks = grid_dims_bricks, dispatch_z](daxa::TaskInterface ti) {
                    ti.recorder.set_pipeline(pipeline_ref);
                    ti.recorder.push_constant(push);
                    ti.recorder.dispatch({
                        static_cast<daxa_u32>(grid_dims_bricks.x),
                        static_cast<daxa_u32>(grid_dims_bricks.y),
                        dispatch_z,
                    });
                }));

        task_graph.add_task(
            daxa::InlineTask::Transfer("animation playground readback")
                .reads(task_bricks_buffer)
                .reads(task_brick_attribs_buffer)
                .writes(task_bricks_readback_buffer)
                .writes(task_brick_attribs_readback_buffer)
                .executes([bricks_buffer_ref, brick_attribs_buffer_ref, bricks_readback_buffer_ref, brick_attribs_readback_buffer_ref, bricks_size, attribs_size](daxa::TaskInterface ti) {
                    ti.recorder.copy_buffer_to_buffer({
                        .src_buffer = bricks_buffer_ref,
                        .dst_buffer = bricks_readback_buffer_ref,
                        .size = bricks_size,
                    });
                    ti.recorder.copy_buffer_to_buffer({
                        .src_buffer = brick_attribs_buffer_ref,
                        .dst_buffer = brick_attribs_readback_buffer_ref,
                        .size = attribs_size,
                    });
                }));

        task_graph.submit({});
        task_graph.complete({});
        task_graph.execute({});
    }

    {
        PROFILE_SCOPE("Wait for GPU work");
        device.wait_idle();
    }

    auto const *bricks_host = device.buffer_host_address_as<BrickPrimitive>(bricks_readback_buffer).value();
    auto const *attribs_host = device.buffer_host_address_as<VoxelShadingAttribBrick>(brick_attribs_readback_buffer).value();

    for (auto *frame : frames) {
        destroy_frame(gpu_context, frame);
    }
    frames.clear();
    frames.resize(frame_count, nullptr);

    total_brick_count = 0;

    for (int frame_i = 0; frame_i < frame_count; ++frame_i) {
        PROFILE_SCOPE("Build CPU frame from readback");

        auto *voxel_object = new VoxelObject();
        voxel_object->brick_min = {0, 0, 0};
        voxel_object->brick_max = grid_dims_bricks - glm::ivec3(1, 1, 1);

        auto const grid_size = voxel_object->brick_max - voxel_object->brick_min + glm::ivec3(1, 1, 1);
        voxel_object->brick_grid.resize(grid_size.x * grid_size.y * grid_size.z);

        for (int bz = 0; bz < grid_dims_bricks.z; ++bz) {
            for (int by = 0; by < grid_dims_bricks.y; ++by) {
                for (int bx = 0; bx < grid_dims_bricks.x; ++bx) {
                    glm::ivec3 const brick_pos = {bx, by, bz};
                    auto const brick_index_in_frame = static_cast<size_t>(bx) + static_cast<size_t>(by) * static_cast<size_t>(grid_dims_bricks.x) + static_cast<size_t>(bz) * static_cast<size_t>(grid_dims_bricks.x) * static_cast<size_t>(grid_dims_bricks.y);
                    auto const brick_index = static_cast<size_t>(frame_i) * static_cast<size_t>(bricks_per_frame) + brick_index_in_frame;

                    auto const &src_primitive = bricks_host[brick_index];

                    bool any_solid = false;
                    for (auto byte : src_primitive.bitmap) {
                        any_solid = any_solid || (byte != uint8_t(0));
                    }
                    if (!any_solid) {
                        continue;
                    }

                    total_brick_count++;

                    auto *brick = new VoxelBrick();
                    brick->brick_i = brick_pos;
                    brick->metadata = 0;
                    static_assert(sizeof(brick->bitmask) == sizeof(src_primitive.bitmap));
                    std::memcpy(brick->bitmask, src_primitive.bitmap, sizeof(brick->bitmask));

                    brick->voxel_min = glm::u8vec3(BRICK_SIZE, BRICK_SIZE, BRICK_SIZE);
                    brick->voxel_max = glm::u8vec3(0, 0, 0);
                    for (int z = 0; z < BRICK_SIZE; ++z) {
                        for (int y = 0; y < BRICK_SIZE; ++y) {
                            auto const byte = src_primitive.bitmap[z * BRICK_SIZE + y];
                            if (byte == uint8_t(0)) {
                                continue;
                            }
                            for (int x = 0; x < BRICK_SIZE; ++x) {
                                if (((uint32_t(byte) >> x) & 1u) == 0u) {
                                    continue;
                                }
                                brick->voxel_min = glm::min(brick->voxel_min, glm::u8vec3(x, y, z));
                                brick->voxel_max = glm::max(brick->voxel_max, glm::u8vec3(x, y, z));
                            }
                        }
                    }

                    brick->render_attribs = new VoxelShadingAttribBrick();
                    std::memcpy(brick->render_attribs, &attribs_host[brick_index], sizeof(VoxelShadingAttribBrick));

                    voxel_object->brick_grid[voxel_object->get_brick_index(brick_pos)] = brick;
                }
            }
        }

        voxel_object->render_voxel_object = create_render_voxel_object(render_scene);
        voxel_object->render_dirty = true;
        update_render_voxel_object(gpu_context, voxel_object);

        frames[frame_i] = voxel_object;
    }
}

void AnimationPlayground::update(Renderer &renderer, GpuInput const &gpu_input) {
    PROFILE_FUNC();

    if (playing) {
        current_frame_f += gpu_input.delta_time * playback_fps;
    }

    if (dirty) {
        regenerate(gpu_input.time);
    }

    if (!frames.empty()) {
        auto const current_frame_int = static_cast<int>(current_frame_f) % frames.size;
        auto voxel_object = frames[current_frame_int];

        draw_voxel_object(voxel_object, playground_pos, {}, VOXEL_SIZE, glm::vec3(1.0f));
        auto const grid_size = grid_dims_bricks;

        Box box;
        box.p0_x = playground_pos.x;
        box.p0_y = playground_pos.y;
        box.p0_z = playground_pos.z;
        box.p1_x = playground_pos.x + VOXEL_SIZE * BRICK_SIZE * grid_size.x;
        box.p1_y = playground_pos.y + VOXEL_SIZE * BRICK_SIZE * grid_size.y;
        box.p1_z = playground_pos.z + VOXEL_SIZE * BRICK_SIZE * grid_size.z;
        box.r = 1.0f;
        box.g = 1.0f;
        box.b = 1.0f;
        renderer.submit_debug_box_lines(&box, 1);
    }
}

void AnimationPlayground::ui() {
    if (ImGui::Begin("Animation Playground")) {
        ImGui::SliderInt("Frame Count", &frame_count, 1, 64);
        dirty |= ImGui::IsItemDeactivated();
        ImGui::SliderInt3("Grid Size (bricks)", &grid_dims_bricks.x, 1, 32);
        dirty |= ImGui::IsItemDeactivated();
        ImGui::DragFloat3("Position", &playground_pos.x);

        ImGui::Checkbox("Playing", &playing);
        ImGui::SliderFloat("Speed (fps)", &playback_fps, 0.0f, 60.0f);

        if (frames.empty()) {
            ImGui::TextUnformatted("No frames generated yet...");
        } else {
            auto current_frame_int = ((static_cast<int>(current_frame_f) % frame_count) + frame_count) % frame_count;
            if (ImGui::SliderInt("Current Frame", &current_frame_int, 0, frame_count - 1)) {
                current_frame_f = static_cast<float>(current_frame_int);
            }
        }

        if (ImGui::Button("Regenerate")) {
            dirty = true;
        }

        ImGui::SeparatorText("Memory");
        ImGui::Text("%llu total bricks", total_brick_count);
        ImGui::Text("%.2f MB", float(total_brick_count) * (sizeof(VoxelBrick) + sizeof(VoxelShadingAttribBrick) + sizeof(Aabb) + sizeof(uint32_t)) / 1000000);
    }
    ImGui::End();
}
