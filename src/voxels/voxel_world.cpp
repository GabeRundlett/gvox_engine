#include "voxel_world.hpp"
#include "renderer/render_voxel_object.hpp"
#include "scene.hpp"
#include "voxel_object.hpp"

#include <chrono>
#include <glm/common.hpp>
#include <glm/glm.hpp>

#include <renderer/renderer.hpp>
#include <utilities/thread_pool.hpp>
// #include <utilities/ispc_instrument.hpp>
#include <utilities/debug.hpp>

#include <fmt/format.h>

#include <array>
#include <vector>
#include <thread>
#include <filesystem>

#include "generation/generation.hpp"
#include "voxels/voxel.inl"

struct BrickMetadata {
    uint32_t exposed_nx : 1 {};
    uint32_t exposed_ny : 1 {};
    uint32_t exposed_nz : 1 {};
    uint32_t exposed_px : 1 {};
    uint32_t exposed_py : 1 {};
    uint32_t exposed_pz : 1 {};
    uint32_t has_air_nx : 1 {};
    uint32_t has_air_ny : 1 {};
    uint32_t has_air_nz : 1 {};
    uint32_t has_air_px : 1 {};
    uint32_t has_air_py : 1 {};
    uint32_t has_air_pz : 1 {};
    uint32_t has_voxel : 1 {};
};

enum GenerationStage {
    NOT_GENERATED,
    GENERATED_BITMASK,
    GENERATED_SURFACE_BRICK_ATTRIBS,
};

using Clock = std::chrono::steady_clock;

constexpr int32_t CHUNK_NX = 1024 / CHUNK_SIZE_VOXELS;
constexpr int32_t CHUNK_NY = 1024 / CHUNK_SIZE_VOXELS;
constexpr int32_t CHUNK_NZ = 512 / CHUNK_SIZE_VOXELS;
constexpr int32_t CHUNK_LEVELS = 1;
constexpr int32_t MAX_CHUNKS_PER_FRAME = 64;

struct Chunk {
    int generation_stage = 0;
    VoxelObject *voxel_object;
    std::vector<glm::ivec3> surface_entity_candidates;
    // glm::ivec3 pos;
};

struct VoxelWorld {
    std::array<Chunk, CHUNK_NX * CHUNK_NY * CHUNK_NZ * 2 * 2 * 2 * CHUNK_LEVELS> chunks;
    Scene *scene;
};

struct DensityNrm {
    float val;
    glm::vec3 nrm;
};

#include <random>

const auto RANDOM_SEED = 3;
const auto RANDOM_VALUES = []() {
    auto result = std::vector<uint8_t>{};
    result.resize(RANDOM_BUFFER_SIZE * RANDOM_BUFFER_SIZE * RANDOM_BUFFER_SIZE);
    auto rng = std::mt19937_64(RANDOM_SEED);
    auto dist = std::uniform_int_distribution<std::mt19937::result_type>(0, 255);
    for (auto &val : result) {
        val = dist(rng) & 0xff;
    }
    return result;
}();

// static_assert(CHUNK_NX * CHUNK_NY * CHUNK_NZ * 2 * 2 * 2 * CHUNK_LEVELS <= MAX_CHUNK_COUNT);

NoiseSettings noise_settings{
    .persistence = 0.15f,
    .lacunarity = 4.5f,
    .scale = 0.02f,
    .amplitude = 40.0f,
    .octaves = 5,
};

// auto get_brick_metadata(std::unique_ptr<Chunk> &chunk, auto brick_index) -> BrickMetadata & {
//     return *reinterpret_cast<BrickMetadata *>(&chunk.voxel_object->brick_grid[brick_index]->bitmask.metadata);
// }

struct GenChunkArgs {
    VoxelWorld *self;
    int32_t chunk_xi;
    int32_t chunk_yi;
    int32_t chunk_zi;
    int32_t level;
    bool update = true;
};

void generate_all_chunks(VoxelWorld *self);

auto create_voxel_world(struct Scene *scene) -> VoxelWorld * {
    auto self = new VoxelWorld();
    self->scene = scene;
    // generate_all_chunks(self);
    return self;
}
void destroy_voxel_world(VoxelWorld *self) {
    for (auto &chunk : self->chunks) {
        if (chunk.voxel_object) {
            // self->scene.delete_voxel_object();
            if (chunk.voxel_object->render_voxel_object)
                destroy_render_voxel_object(self->scene->gpu_context, chunk.voxel_object->render_voxel_object);
            delete chunk.voxel_object;
        }
    }

    delete self;
}

size_t get_chunk_index(int32_t chunk_xi, int32_t chunk_yi, int32_t chunk_zi, int32_t level) {
    return size_t(chunk_xi + CHUNK_NX) + size_t(chunk_yi + CHUNK_NY) * (CHUNK_NX * 2) + size_t(chunk_zi + CHUNK_NZ) * CHUNK_NX * CHUNK_NY * 2 * 2 + level * CHUNK_NX * CHUNK_NY * CHUNK_NZ * 2 * 2 * 2;
}
auto get_brick_metadata(Chunk &chunk, auto brick_index) -> BrickMetadata & {
    return *reinterpret_cast<BrickMetadata *>(&chunk.voxel_object->brick_grid[brick_index]->metadata);
}

int generate_chunk_precheck(VoxelWorld *self, int32_t chunk_xi, int32_t chunk_yi, int32_t chunk_zi, int32_t level);
void generate_chunk(VoxelWorld *self, int32_t chunk_xi, int32_t chunk_yi, int32_t chunk_zi, int32_t level);
void generate_chunk2(VoxelWorld *self, int32_t chunk_xi, int32_t chunk_yi, int32_t chunk_zi, int32_t level, bool update = true);

static glm::vec3 hsv2rgb(glm::vec3 c) {
    glm::vec4 k = glm::vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    glm::vec3 p = abs(glm::fract(glm::vec3(c.x, c.x, c.x) + glm::vec3(k.x, k.y, k.z)) * 6.0f - k.w);
    return c.z * glm::mix(glm::vec3(k.x), glm::clamp(p - k.x, glm::vec3(0.0), glm::vec3(1.0)), glm::vec3(c.y));
}

void update(struct GpuContext &gpu_context, Renderer &renderer, GpuInput &gpu_input, VoxelWorld *self) {
    std::vector<std::pair<thread_pool::Task, void *>> tasks;
    tasks.reserve(MAX_CHUNKS_PER_FRAME);

    for (int32_t level_i = 0; level_i < CHUNK_LEVELS; ++level_i) {
        for (int32_t chunk_zi = -CHUNK_NZ; chunk_zi < CHUNK_NZ; ++chunk_zi) {
            for (int32_t chunk_yi = -CHUNK_NY; chunk_yi < CHUNK_NY; ++chunk_yi) {
                for (int32_t chunk_xi = -CHUNK_NX; chunk_xi < CHUNK_NX; ++chunk_xi) {
                    auto *user_ptr = new GenChunkArgs{self, chunk_xi, chunk_yi, chunk_zi, level_i};
                    auto const &args = *(GenChunkArgs *)user_ptr;

                    auto chunk_index = get_chunk_index(chunk_xi, chunk_yi, chunk_zi, level_i);
                    auto &chunk = self->chunks[chunk_index];
                    if (chunk.generation_stage != 0)
                        continue;

                    if (generate_chunk_precheck(args.self, args.chunk_xi, args.chunk_yi, args.chunk_zi, args.level) == 2) {
                        auto task = thread_pool::create_task(
                            [](void *user_ptr) {
                                auto const &args = *(GenChunkArgs *)user_ptr;
                                generate_chunk(args.self, args.chunk_xi, args.chunk_yi, args.chunk_zi, args.level);
                            },
                            user_ptr);
                        thread_pool::async_dispatch(task);
                        tasks.emplace_back(task, user_ptr);
                        if (tasks.size() == MAX_CHUNKS_PER_FRAME)
                            goto exit_1;
                    }
                }
            }
        }
    }
exit_1:
    for (auto &[task, user_ptr] : tasks) {
        thread_pool::wait(task);
        thread_pool::destroy_task(task);
        delete (GenChunkArgs *)user_ptr;
    }
    tasks.clear();

    for (int32_t level_i = 0; level_i < CHUNK_LEVELS; ++level_i) {
        for (int32_t chunk_zi = -CHUNK_NZ; chunk_zi < CHUNK_NZ; ++chunk_zi) {
            for (int32_t chunk_yi = -CHUNK_NY; chunk_yi < CHUNK_NY; ++chunk_yi) {
                for (int32_t chunk_xi = -CHUNK_NX; chunk_xi < CHUNK_NX; ++chunk_xi) {
                    auto chunk_index = get_chunk_index(chunk_xi, chunk_yi, chunk_zi, level_i);
                    auto &chunk = self->chunks[chunk_index];
                    if (chunk.generation_stage != 2)
                        continue;

                    auto *user_ptr = new GenChunkArgs{self, chunk_xi, chunk_yi, chunk_zi, level_i};
                    auto task = thread_pool::create_task([](void *user_ptr) { auto const &args = *(GenChunkArgs*)user_ptr; generate_chunk2(args.self, args.chunk_xi, args.chunk_yi, args.chunk_zi, args.level); }, user_ptr);
                    thread_pool::async_dispatch(task);
                    tasks.emplace_back(task, user_ptr);
                    if (tasks.size() == MAX_CHUNKS_PER_FRAME)
                        goto exit_2;
                }
            }
        }
    }
exit_2:
    for (auto &[task, user_ptr] : tasks) {
        thread_pool::wait(task);
        thread_pool::destroy_task(task);
        delete (GenChunkArgs *)user_ptr;
    }
    tasks.clear();

    for (int32_t level_i = 0; level_i < CHUNK_LEVELS; ++level_i) {
        for (int32_t chunk_zi = -CHUNK_NZ; chunk_zi < CHUNK_NZ; ++chunk_zi) {
            for (int32_t chunk_yi = -CHUNK_NY; chunk_yi < CHUNK_NY; ++chunk_yi) {
                for (int32_t chunk_xi = -CHUNK_NX; chunk_xi < CHUNK_NX; ++chunk_xi) {
                    const float voxel_size = VOXEL_SIZE * (1 << level_i);
                    auto pos = glm::vec3(chunk_xi, chunk_yi, chunk_zi) * float(CHUNK_SIZE_VOXELS) * voxel_size;
                    auto chunk_index = get_chunk_index(chunk_xi, chunk_yi, chunk_zi, level_i);
                    auto &chunk = self->chunks[chunk_index];

                    // Box box;
                    // box.p0_x = pos.x;
                    // box.p0_y = pos.y;
                    // box.p0_z = pos.z;
                    // box.p1_x = pos.x + voxel_size * CHUNK_SIZE_VOXELS;
                    // box.p1_y = pos.y + voxel_size * CHUNK_SIZE_VOXELS;
                    // box.p1_z = pos.z + voxel_size * CHUNK_SIZE_VOXELS;

                    if (chunk.voxel_object != nullptr && chunk.voxel_object->render_voxel_object != nullptr) {
                        glm::vec3 tint{1, 1, 1};
                        update_render_voxel_object(gpu_context, chunk.voxel_object);
                        draw_voxel_object(chunk.voxel_object, pos, voxel_size, tint);

                        for (auto surface_ent : chunk.surface_entity_candidates) {
                            auto voxel_object = self->scene->ball_frames[int(gpu_input.time * 12 + rand()) % glm::countof(self->scene->ball_frames)];
                            auto grid_size = voxel_object->brick_max - voxel_object->brick_min + 1;
                            auto ball_pos = pos + (glm::vec3(surface_ent) + 0.5f) * float(BRICK_SIZE) * voxel_size - glm::vec3(grid_size) * 0.5f * float(BRICK_SIZE) * VOXEL_SIZE;
                            auto tint = hsv2rgb(glm::vec3(float(rand() % 100) / 100, 0.9 + float(rand() % 100) / 1000, 0.9));
                            // auto tint = glm::vec3(1);
                            draw_voxel_object(voxel_object, ball_pos, VOXEL_SIZE, tint);
                        }

                        // box.r = 0.2f;
                        // box.g = 1.0f;
                        // box.b = 0.2f;
                        // renderer.submit_debug_box_lines(&box, 1);
                    } else {

                        // box.r = 1.0f;
                        // box.g = 0.2f;
                        // box.b = 0.2f;
                        // renderer.submit_debug_box_lines(&box, 1);
                    }
                }
            }
        }
    }
}

void generate_all_chunks(VoxelWorld *self) {
    std::vector<std::pair<thread_pool::Task, void *>> tasks;
    tasks.reserve(CHUNK_NX * CHUNK_NY * CHUNK_NZ * 2 * 2 * 2 * CHUNK_LEVELS);

    // self->generate_chunk1s_total = {};
    // self->generate_chunk2s_total = {};
    // auto generate_chunk1s_main_total_ns = uint64_t{};
    // auto generate_chunk2s_main_total_ns = uint64_t{};

    {
        // auto t0 = Clock::now();

        for (int32_t level_i = 0; level_i < CHUNK_LEVELS; ++level_i) {
            for (int32_t chunk_zi = -CHUNK_NZ; chunk_zi < CHUNK_NZ; ++chunk_zi) {
                for (int32_t chunk_yi = -CHUNK_NY; chunk_yi < CHUNK_NY; ++chunk_yi) {
                    for (int32_t chunk_xi = -CHUNK_NX; chunk_xi < CHUNK_NX; ++chunk_xi) {
                        auto *user_ptr = new GenChunkArgs{self, chunk_xi, chunk_yi, chunk_zi, level_i};
                        auto task = thread_pool::create_task(
                            [](void *user_ptr) {
                                auto const &args = *(GenChunkArgs *)user_ptr;
                                if (generate_chunk_precheck(args.self, args.chunk_xi, args.chunk_yi, args.chunk_zi, args.level) == 2)
                                    generate_chunk(args.self, args.chunk_xi, args.chunk_yi, args.chunk_zi, args.level);
                            },
                            user_ptr);
                        thread_pool::async_dispatch(task);
                        tasks.emplace_back(task, user_ptr);
                    }
                }
            }
        }

        for (auto &[task, user_ptr] : tasks) {
            thread_pool::wait(task);
            thread_pool::destroy_task(task);
            delete (GenChunkArgs *)user_ptr;
        }
        tasks.clear();

        // auto t1 = Clock::now();
        // generate_chunk1s_main_total_ns += (t1 - t0).count();
    }

    {
        // auto t0 = Clock::now();
        for (int32_t level_i = 0; level_i < CHUNK_LEVELS; ++level_i) {
            for (int32_t chunk_zi = -CHUNK_NZ; chunk_zi < CHUNK_NZ; ++chunk_zi) {
                for (int32_t chunk_yi = -CHUNK_NY; chunk_yi < CHUNK_NY; ++chunk_yi) {
                    for (int32_t chunk_xi = -CHUNK_NX; chunk_xi < CHUNK_NX; ++chunk_xi) {
                        auto *user_ptr = new GenChunkArgs{self, chunk_xi, chunk_yi, chunk_zi, level_i};
                        auto task = thread_pool::create_task([](void *user_ptr) { auto const &args = *(GenChunkArgs*)user_ptr; generate_chunk2(args.self, args.chunk_xi, args.chunk_yi, args.chunk_zi, args.level); }, user_ptr);
                        thread_pool::async_dispatch(task);
                        tasks.emplace_back(task, user_ptr);
                    }
                }
            }
        }
        for (auto &[task, user_ptr] : tasks) {
            thread_pool::wait(task);
            thread_pool::destroy_task(task);
            delete (GenChunkArgs *)user_ptr;
        }
        tasks.clear();

        // auto t1 = Clock::now();
        // generate_chunk2s_main_total_ns += (t1 - t0).count();
    }

    // for (int32_t level_i = 0; level_i < CHUNK_LEVELS; ++level_i) {
    //     for (int32_t chunk_zi = -CHUNK_NZ; chunk_zi < CHUNK_NZ; ++chunk_zi) {
    //         for (int32_t chunk_yi = -CHUNK_NY; chunk_yi < CHUNK_NY; ++chunk_yi) {
    //             for (int32_t chunk_xi = -CHUNK_NX; chunk_xi < CHUNK_NX; ++chunk_xi) {
    //                 auto pos = glm::vec3(chunk_xi, chunk_yi, chunk_zi) * float(CHUNK_SIZE_VOXELS) * VOXEL_SIZE;
    //                 auto chunk_index = get_chunk_index(chunk_xi, chunk_yi, chunk_zi, level_i);
    //                 auto &chunk = self->chunks[chunk_index];
    //                 self->scene->voxel_objects.add();
    //             }
    //         }
    //     }
    // }

    // auto generate_chunk1s_total = std::chrono::duration<float, std::micro>(std::chrono::duration<uint64_t, std::nano>(self->generate_chunk1s_total)).count();
    // auto generate_chunk2s_total = std::chrono::duration<float, std::micro>(std::chrono::duration<uint64_t, std::nano>(self->generate_chunk2s_total)).count();

    // auto generate_chunk1s_main_total = std::chrono::duration<float, std::micro>(std::chrono::duration<uint64_t, std::nano>(generate_chunk1s_main_total_ns)).count();
    // auto generate_chunk2s_main_total = std::chrono::duration<float, std::micro>(std::chrono::duration<uint64_t, std::nano>(generate_chunk2s_main_total_ns)).count();

    // debug_utils::add_log(g_console, fmt::format("1: {} s | {} us/brick ({} total bricks) {} us/brick per thread",
    //                                             generate_chunk1s_main_total / 1'000'000,
    //                                             generate_chunk1s_main_total / self->generate_chunk1s_total_n,
    //                                             self->generate_chunk1s_total_n.load(),
    //                                             generate_chunk1s_total / self->generate_chunk1s_total_n)
    //                                     .c_str());
    // debug_utils::add_log(g_console, fmt::format("2: {} s | {} us/brick ({} total bricks) {} us/brick per thread",
    //                                             generate_chunk2s_main_total / 1'000'000,
    //                                             generate_chunk2s_main_total / self->generate_chunk2s_total_n,
    //                                             self->generate_chunk2s_total_n.load(),
    //                                             generate_chunk2s_total / self->generate_chunk2s_total_n)
    //                                     .c_str());

    // ISPCPrintInstrument();
}

int generate_chunk_precheck(VoxelWorld *self, int32_t chunk_xi, int32_t chunk_yi, int32_t chunk_zi, int32_t level) {
    if (level > 0 &&
        chunk_xi >= -CHUNK_NX / 2 && chunk_xi < CHUNK_NX / 2 &&
        chunk_yi >= -CHUNK_NY / 2 && chunk_yi < CHUNK_NY / 2 &&
        chunk_zi >= -CHUNK_NZ / 2 && chunk_zi < CHUNK_NZ / 2) {
        return 0;
    }

    auto chunk_index = get_chunk_index(chunk_xi, chunk_yi, chunk_zi, level);
    auto &chunk = self->chunks[chunk_index];

    chunk.generation_stage = 1;

    {
        auto p0 = glm::vec3{
            (float((chunk_xi * CHUNK_SIZE_VOXELS) << level) + 0.5f) * VOXEL_SIZE,
            (float((chunk_yi * CHUNK_SIZE_VOXELS) << level) + 0.5f) * VOXEL_SIZE,
            (float((chunk_zi * CHUNK_SIZE_VOXELS) << level) + 0.5f) * VOXEL_SIZE,
        };
        auto p1 = p0 + (CHUNK_SIZE_VOXELS << level) * VOXEL_SIZE;
        auto minmax = voxel_minmax_value_cpp(&noise_settings, RANDOM_VALUES.data(), p0.x, p0.y, p0.z, p1.x, p1.y, p1.z);
        if (minmax.min >= 0.0f || minmax.max < 0.0f) {
            // uniform
            if (minmax.min < 0.0f) {
                // inside
            } else {
                // outside
                return chunk.generation_stage;
            }
        }
    }

    chunk.generation_stage = 2;
    return chunk.generation_stage;
}

void generate_chunk(VoxelWorld *self, int32_t chunk_xi, int32_t chunk_yi, int32_t chunk_zi, int32_t level) {
    auto chunk_index = get_chunk_index(chunk_xi, chunk_yi, chunk_zi, level);
    auto &chunk = self->chunks[chunk_index];

    chunk.voxel_object = new VoxelObject();
    // chunk.pos = {chunk_xi, chunk_yi, chunk_zi};

    // auto t0 = Clock::now();
    chunk.voxel_object->init({0, 0, 0}, {CHUNK_SIZE_VOXELS - 1, CHUNK_SIZE_VOXELS - 1, CHUNK_SIZE_VOXELS - 1});

    for (int32_t brick_zi = 0; brick_zi < CHUNK_SIZE_BRICKS; ++brick_zi) {
        for (int32_t brick_yi = 0; brick_yi < CHUNK_SIZE_BRICKS; ++brick_yi) {
            for (int32_t brick_xi = 0; brick_xi < CHUNK_SIZE_BRICKS; ++brick_xi) {
                auto brick_index = brick_xi + brick_yi * CHUNK_SIZE_BRICKS + brick_zi * CHUNK_SIZE_BRICKS * CHUNK_SIZE_BRICKS;
                auto &brick = chunk.voxel_object->brick_grid[brick_index];
                brick = new VoxelBrick();
                auto &brick_metadata = get_brick_metadata(chunk, brick_index);
                auto &bitmask = brick->bitmask;

                brick_metadata = {};
                memset(&bitmask, 0, sizeof(bitmask));

                // determine if brick is uniform

                {
                    auto p0 = glm::vec3{
                        (float((brick_xi * BRICK_SIZE + chunk_xi * CHUNK_SIZE_VOXELS) << level) + 0.5f) * VOXEL_SIZE,
                        (float((brick_yi * BRICK_SIZE + chunk_yi * CHUNK_SIZE_VOXELS) << level) + 0.5f) * VOXEL_SIZE,
                        (float((brick_zi * BRICK_SIZE + chunk_zi * CHUNK_SIZE_VOXELS) << level) + 0.5f) * VOXEL_SIZE,
                    };
                    auto p1 = p0 + (BRICK_SIZE << level) * VOXEL_SIZE;
                    auto minmax = voxel_minmax_value_cpp(&noise_settings, RANDOM_VALUES.data(), p0.x, p0.y, p0.z, p1.x, p1.y, p1.z);
                    if (minmax.min >= 0.0f || minmax.max < 0.0f) {
                        // uniform
                        delete brick;
                        brick = nullptr;
                        continue;
                    }
                }

                // self->generate_chunk1s_total_n += 1;
                generate_bitmask(brick_xi, brick_yi, brick_zi, chunk_xi, chunk_yi, chunk_zi, level, (uint32_t *)bitmask, (uint32_t *)&brick_metadata, &noise_settings, RANDOM_VALUES.data());
            }
        }
    }

    // auto t1 = Clock::now();

    // self->generate_chunk1s_total += (t1 - t0).count();
}
// struct VoxelSimAttribBrick {
//     float densities[BRICK_SIZE * BRICK_SIZE * BRICK_SIZE];
// };

void generate_chunk2(VoxelWorld *self, int32_t chunk_xi, int32_t chunk_yi, int32_t chunk_zi, int32_t level, bool update) {
    auto chunk_index = get_chunk_index(chunk_xi, chunk_yi, chunk_zi, level);
    auto &chunk = self->chunks[chunk_index];
    if (chunk.voxel_object == nullptr) {
        return;
    }
    chunk.generation_stage = 3;

    // auto t0 = Clock::now();

    // chunk.surface_brick_indices.clear();

    // auto temp_sim_attrib_brick = VoxelSimAttribBrick{};

    bool has_render_attribs = false;

    for (int32_t brick_zi = 0; brick_zi < CHUNK_SIZE_BRICKS; ++brick_zi) {
        for (int32_t brick_yi = 0; brick_yi < CHUNK_SIZE_BRICKS; ++brick_yi) {
            for (int32_t brick_xi = 0; brick_xi < CHUNK_SIZE_BRICKS; ++brick_xi) {
                auto brick_index = brick_xi + brick_yi * CHUNK_SIZE_BRICKS + brick_zi * CHUNK_SIZE_BRICKS * CHUNK_SIZE_BRICKS;
                if (!chunk.voxel_object->brick_grid[brick_index])
                    continue;
                auto &brick_metadata = get_brick_metadata(chunk, brick_index);
                auto &brick = chunk.voxel_object->brick_grid[brick_index];
                brick->brick_i = {brick_xi, brick_yi, brick_zi};

                brick_metadata.exposed_nx = false;
                brick_metadata.exposed_px = false;
                brick_metadata.exposed_ny = false;
                brick_metadata.exposed_py = false;
                brick_metadata.exposed_nz = false;
                brick_metadata.exposed_pz = false;

                auto const *neighbor_bitmask_nx = (VoxelBrick const *)nullptr;
                auto const *neighbor_bitmask_px = (VoxelBrick const *)nullptr;
                auto const *neighbor_bitmask_ny = (VoxelBrick const *)nullptr;
                auto const *neighbor_bitmask_py = (VoxelBrick const *)nullptr;
                auto const *neighbor_bitmask_nz = (VoxelBrick const *)nullptr;
                auto const *neighbor_bitmask_pz = (VoxelBrick const *)nullptr;

                if (brick_xi != 0) {
                    auto neighbor_brick_index = (brick_xi - 1) + brick_yi * CHUNK_SIZE_BRICKS + brick_zi * CHUNK_SIZE_BRICKS * CHUNK_SIZE_BRICKS;
                    if (chunk.voxel_object->brick_grid[neighbor_brick_index]) {
                        auto &neighbor_brick_metadata = get_brick_metadata(chunk, neighbor_brick_index);
                        brick_metadata.exposed_nx = neighbor_brick_metadata.has_air_px;
                        neighbor_bitmask_nx = chunk.voxel_object->brick_grid[neighbor_brick_index];
                    }
                } else if (chunk_xi != -CHUNK_NX) {
                    auto neighbor_chunk_index = get_chunk_index(chunk_xi - 1, chunk_yi, chunk_zi, level);
                    auto &neighbor_chunk = self->chunks[neighbor_chunk_index];
                    if (neighbor_chunk.voxel_object) {
                        auto neighbor_brick_index = (CHUNK_SIZE_BRICKS - 1) + brick_yi * CHUNK_SIZE_BRICKS + brick_zi * CHUNK_SIZE_BRICKS * CHUNK_SIZE_BRICKS;
                        if (neighbor_chunk.voxel_object->brick_grid[neighbor_brick_index]) {
                            auto &neighbor_brick_metadata = get_brick_metadata(neighbor_chunk, neighbor_brick_index);
                            brick_metadata.exposed_nx = neighbor_brick_metadata.has_air_px;
                            neighbor_bitmask_nx = neighbor_chunk.voxel_object->brick_grid[neighbor_brick_index];
                        }
                    } else {
                        // brick_metadata.exposed_nx = true;
                    }
                }
                if (brick_yi != 0) {
                    auto neighbor_brick_index = brick_xi + (brick_yi - 1) * CHUNK_SIZE_BRICKS + brick_zi * CHUNK_SIZE_BRICKS * CHUNK_SIZE_BRICKS;
                    if (chunk.voxel_object->brick_grid[neighbor_brick_index]) {
                        auto &neighbor_brick_metadata = get_brick_metadata(chunk, neighbor_brick_index);
                        brick_metadata.exposed_ny = neighbor_brick_metadata.has_air_py;
                        neighbor_bitmask_ny = chunk.voxel_object->brick_grid[neighbor_brick_index];
                    }
                } else if (chunk_yi != -CHUNK_NY) {
                    auto neighbor_chunk_index = get_chunk_index(chunk_xi, chunk_yi - 1, chunk_zi, level);
                    auto &neighbor_chunk = self->chunks[neighbor_chunk_index];
                    if (neighbor_chunk.voxel_object) {
                        auto neighbor_brick_index = brick_xi + (CHUNK_SIZE_BRICKS - 1) * CHUNK_SIZE_BRICKS + brick_zi * CHUNK_SIZE_BRICKS * CHUNK_SIZE_BRICKS;
                        if (neighbor_chunk.voxel_object->brick_grid[neighbor_brick_index]) {
                            auto &neighbor_brick_metadata = get_brick_metadata(neighbor_chunk, neighbor_brick_index);
                            brick_metadata.exposed_ny = neighbor_brick_metadata.has_air_py;
                            neighbor_bitmask_ny = neighbor_chunk.voxel_object->brick_grid[neighbor_brick_index];
                        }
                    } else {
                        // brick_metadata.exposed_ny = true;
                    }
                }
                if (brick_zi != 0) {
                    auto neighbor_brick_index = brick_xi + brick_yi * CHUNK_SIZE_BRICKS + (brick_zi - 1) * CHUNK_SIZE_BRICKS * CHUNK_SIZE_BRICKS;
                    if (chunk.voxel_object->brick_grid[neighbor_brick_index]) {
                        auto &neighbor_brick_metadata = get_brick_metadata(chunk, neighbor_brick_index);
                        brick_metadata.exposed_nz = neighbor_brick_metadata.has_air_pz;
                        neighbor_bitmask_nz = chunk.voxel_object->brick_grid[neighbor_brick_index];
                    }
                } else if (chunk_zi != -CHUNK_NZ) {
                    auto neighbor_chunk_index = get_chunk_index(chunk_xi, chunk_yi, chunk_zi - 1, level);
                    auto &neighbor_chunk = self->chunks[neighbor_chunk_index];
                    if (neighbor_chunk.voxel_object) {
                        auto neighbor_brick_index = brick_xi + brick_yi * CHUNK_SIZE_BRICKS + (CHUNK_SIZE_BRICKS - 1) * CHUNK_SIZE_BRICKS * CHUNK_SIZE_BRICKS;
                        if (neighbor_chunk.voxel_object->brick_grid[neighbor_brick_index]) {
                            auto &neighbor_brick_metadata = get_brick_metadata(neighbor_chunk, neighbor_brick_index);
                            brick_metadata.exposed_nz = neighbor_brick_metadata.has_air_pz;
                            neighbor_bitmask_nz = neighbor_chunk.voxel_object->brick_grid[neighbor_brick_index];
                        }
                    } else {
                        // brick_metadata.exposed_nz = true;
                    }
                }
                if (brick_xi != CHUNK_SIZE_BRICKS - 1) {
                    auto neighbor_brick_index = (brick_xi + 1) + brick_yi * CHUNK_SIZE_BRICKS + brick_zi * CHUNK_SIZE_BRICKS * CHUNK_SIZE_BRICKS;
                    if (chunk.voxel_object->brick_grid[neighbor_brick_index]) {
                        auto &neighbor_brick_metadata = get_brick_metadata(chunk, neighbor_brick_index);
                        brick_metadata.exposed_px = neighbor_brick_metadata.has_air_nx;
                        neighbor_bitmask_px = chunk.voxel_object->brick_grid[neighbor_brick_index];
                    }
                } else if (chunk_xi != CHUNK_NX - 1) {
                    auto neighbor_chunk_index = get_chunk_index(chunk_xi + 1, chunk_yi, chunk_zi, level);
                    auto &neighbor_chunk = self->chunks[neighbor_chunk_index];
                    if (neighbor_chunk.voxel_object) {
                        auto neighbor_brick_index = 0 + brick_yi * CHUNK_SIZE_BRICKS + brick_zi * CHUNK_SIZE_BRICKS * CHUNK_SIZE_BRICKS;
                        if (neighbor_chunk.voxel_object->brick_grid[neighbor_brick_index]) {
                            auto &neighbor_brick_metadata = get_brick_metadata(neighbor_chunk, neighbor_brick_index);
                            brick_metadata.exposed_px = neighbor_brick_metadata.has_air_nx;
                            neighbor_bitmask_px = neighbor_chunk.voxel_object->brick_grid[neighbor_brick_index];
                        }
                    } else {
                        // brick_metadata.exposed_px = true;
                    }
                }
                if (brick_yi != CHUNK_SIZE_BRICKS - 1) {
                    auto neighbor_brick_index = brick_xi + (brick_yi + 1) * CHUNK_SIZE_BRICKS + brick_zi * CHUNK_SIZE_BRICKS * CHUNK_SIZE_BRICKS;
                    if (chunk.voxel_object->brick_grid[neighbor_brick_index]) {
                        auto &neighbor_brick_metadata = get_brick_metadata(chunk, neighbor_brick_index);
                        brick_metadata.exposed_py = neighbor_brick_metadata.has_air_ny;
                        neighbor_bitmask_py = chunk.voxel_object->brick_grid[neighbor_brick_index];
                    }
                } else if (chunk_yi != CHUNK_NY - 1) {
                    auto neighbor_chunk_index = get_chunk_index(chunk_xi, chunk_yi + 1, chunk_zi, level);
                    auto &neighbor_chunk = self->chunks[neighbor_chunk_index];
                    if (neighbor_chunk.voxel_object) {
                        auto neighbor_brick_index = brick_xi + 0 * CHUNK_SIZE_BRICKS + brick_zi * CHUNK_SIZE_BRICKS * CHUNK_SIZE_BRICKS;
                        if (neighbor_chunk.voxel_object->brick_grid[neighbor_brick_index]) {
                            auto &neighbor_brick_metadata = get_brick_metadata(neighbor_chunk, neighbor_brick_index);
                            brick_metadata.exposed_py = neighbor_brick_metadata.has_air_ny;
                            neighbor_bitmask_py = neighbor_chunk.voxel_object->brick_grid[neighbor_brick_index];
                        }
                    } else {
                        // brick_metadata.exposed_py = true;
                    }
                }
                if (brick_zi != CHUNK_SIZE_BRICKS - 1) {
                    auto neighbor_brick_index = brick_xi + brick_yi * CHUNK_SIZE_BRICKS + (brick_zi + 1) * CHUNK_SIZE_BRICKS * CHUNK_SIZE_BRICKS;
                    if (chunk.voxel_object->brick_grid[neighbor_brick_index]) {
                        auto &neighbor_brick_metadata = get_brick_metadata(chunk, neighbor_brick_index);
                        brick_metadata.exposed_pz = neighbor_brick_metadata.has_air_nz;
                        neighbor_bitmask_pz = chunk.voxel_object->brick_grid[neighbor_brick_index];
                    }
                } else if (chunk_zi != CHUNK_NZ - 1) {
                    auto neighbor_chunk_index = get_chunk_index(chunk_xi, chunk_yi, chunk_zi + 1, level);
                    auto &neighbor_chunk = self->chunks[neighbor_chunk_index];
                    if (neighbor_chunk.voxel_object) {
                        auto neighbor_brick_index = brick_xi + brick_yi * CHUNK_SIZE_BRICKS + 0 * CHUNK_SIZE_BRICKS * CHUNK_SIZE_BRICKS;
                        if (neighbor_chunk.voxel_object->brick_grid[neighbor_brick_index]) {
                            auto &neighbor_brick_metadata = get_brick_metadata(neighbor_chunk, neighbor_brick_index);
                            brick_metadata.exposed_pz = neighbor_brick_metadata.has_air_nz;
                            neighbor_bitmask_pz = neighbor_chunk.voxel_object->brick_grid[neighbor_brick_index];
                        }
                    } else {
                        // brick_metadata.exposed_pz = true;
                    }
                }

                bool exposed = brick_metadata.exposed_nx || brick_metadata.exposed_px || brick_metadata.exposed_ny || brick_metadata.exposed_py || brick_metadata.exposed_nz || brick_metadata.exposed_pz;

                auto position = glm::ivec4{brick_xi, brick_yi, brick_zi, -LOG2_VOXELS_PER_METER + level};
                if (brick_metadata.has_voxel && exposed) {
                    // generate surface brick data
                    auto &render_attrib_brick = chunk.voxel_object->brick_grid[brick_index]->render_attribs;
                    // auto &sim_attrib_brick = chunk.voxel_object->brick_grid[brick_index]->sim_attribs;
                    // self->generate_chunk2s_total_n += 1;

                    if (render_attrib_brick == nullptr) {
                        render_attrib_brick = new VoxelShadingAttribBrick();
                        generate_attributes(brick_xi, brick_yi, brick_zi, chunk_xi, chunk_yi, chunk_zi, level,
                                            (uint32_t *)render_attrib_brick->voxels, &noise_settings, RANDOM_VALUES.data());
                        has_render_attribs = true;

                        if (RANDOM_VALUES[(brick_index + chunk_index * 197123) % RANDOM_VALUES.size()] < 255 * 0.01 * (1 << level)) {
                            float upwards = generate_upwards(brick_xi, brick_yi, brick_zi, chunk_xi, chunk_yi, chunk_zi, level, &noise_settings, RANDOM_VALUES.data());
                            if (chunk.surface_entity_candidates.size() < 10 && upwards > 0.8)
                                chunk.surface_entity_candidates.push_back(brick->brick_i);
                        }

                        {

                            brick->voxel_min = {BRICK_SIZE, BRICK_SIZE, BRICK_SIZE};
                            brick->voxel_max = {0, 0, 0};
                            for (uint8_t z = 0; z < 8; z++) {
                                bool z_has_voxels = false;
                                for (int y = 0; y < 8; y++) {
                                    auto v = ((uint8_t *)(brick->bitmask))[(z << 3) + y];
                                    if (v) {
                                        brick->voxel_min[0] = glm::min(brick->voxel_min[0], (uint8_t)std::countr_zero(v));
                                        brick->voxel_max[0] = glm::max(brick->voxel_max[0], (uint8_t)(7 - std::countl_zero(v)));
                                        brick->voxel_min[1] = glm::min(brick->voxel_min[1], uint8_t(y));
                                        brick->voxel_max[1] = glm::max(brick->voxel_max[1], uint8_t(y));
                                        z_has_voxels = true;
                                    }
                                }
                                if (z_has_voxels) {
                                    brick->voxel_min[2] = glm::min(brick->voxel_min[2], z);
                                    brick->voxel_max[2] = glm::max(brick->voxel_max[2], z);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if (has_render_attribs) {
        chunk.voxel_object->render_voxel_object = create_render_voxel_object(self->scene->render_scene);
        chunk.voxel_object->render_dirty = true;
    }
}
