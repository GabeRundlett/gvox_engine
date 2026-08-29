#include "voxel_world.hpp"
#include "animation_playground/animation_playground.hpp"
#include "pack_unpack.inl"
#include "renderer/render_voxel_object.hpp"
#include "renderer/particles/render_foliage.hpp"
#include "scene.hpp"
#include "voxel_brick.hpp"
#include "voxel_object.hpp"

#include <algorithm>
#include <chrono>
#include <glm/common.hpp>
#include <glm/glm.hpp>

#include <renderer/renderer.hpp>
#include <utilities/thread_pool.hpp>
// #include <utilities/ispc_instrument.hpp>
#include <utilities/debug.hpp>
#include <base/profiler.hpp>

#include <array>
#include <vector>
#include <thread>
#include <filesystem>

#include "generation/generation.hpp"
#include "voxels/voxel.inl"

struct BrickMetadata {
    uint16_t exposed_nx : 1 {};
    uint16_t exposed_ny : 1 {};
    uint16_t exposed_nz : 1 {};
    uint16_t exposed_px : 1 {};
    uint16_t exposed_py : 1 {};
    uint16_t exposed_pz : 1 {};

    // NOTE!!! exposed_... are all written concurrently in phase2 chunk generation, but has_air_... and has_voxel are read.
    // If they are all in the same uint32_t, then the code will just be bitwise ops on the same single object. This is a sneaky race
    uint16_t : 0; // So, we force these fields into a new 'object' with this, but keep the uint32_t footprint

    uint16_t has_air_nx : 1 {};
    uint16_t has_air_ny : 1 {};
    uint16_t has_air_nz : 1 {};
    uint16_t has_air_px : 1 {};
    uint16_t has_air_py : 1 {};
    uint16_t has_air_pz : 1 {};
    uint16_t has_voxel : 1 {};
};
static_assert(sizeof(BrickMetadata) == sizeof(uint32_t));

enum GenerationStage {
    NOT_GENERATED,
    GENERATED_BITMASK,
    GENERATED_SURFACE_BRICK_ATTRIBS,
};

using Clock = std::chrono::steady_clock;

constexpr int32_t CHUNK_NX = 256 / CHUNK_SIZE_VOXELS;
constexpr int32_t CHUNK_NY = 256 / CHUNK_SIZE_VOXELS;
constexpr int32_t CHUNK_NZ = 256 / CHUNK_SIZE_VOXELS;
constexpr int32_t CHUNK_LEVELS = 1;
constexpr int32_t MAX_CHUNKS_PER_FRAME = 32;

struct Chunk {
    int generation_stage = 0;
    VoxelObject *voxel_object;
    Vec<glm::ivec3> surface_entity_candidates;
};

struct GenChunkArgs {
    VoxelWorld *self;
    int32_t chunk_xi;
    int32_t chunk_yi;
    int32_t chunk_zi;
    int32_t level;
    float rating;
    bool update = true;
};

struct VoxelWorld {
    std::array<Chunk, CHUNK_NX * CHUNK_NY * CHUNK_NZ * 2 * 2 * 2 * CHUNK_LEVELS> chunks;
    Vec<GenChunkArgs> chunk_candidates;
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
    .scale = 0.02f / 4.5f,
    .amplitude = 40.0f / 0.15f,
    .octaves = 6,
};

// auto get_brick_metadata(std::unique_ptr<Chunk> &chunk, auto brick_index) -> BrickMetadata & {
//     return *reinterpret_cast<BrickMetadata *>(&chunk.voxel_object->brick_grid[brick_index]->bitmask.metadata);
// }

void generate_all_chunks(VoxelWorld *self);
float chunk_candidate_rating(int32_t chunk_xi, int32_t chunk_yi, int32_t chunk_zi, int32_t level);

auto create_voxel_world(struct Scene *scene) -> VoxelWorld * {
    auto self = new VoxelWorld();
    self->scene = scene;

    self->chunk_candidates.reserve(CHUNK_NX * CHUNK_NY * CHUNK_NZ * 2 * 2 * 2 * CHUNK_LEVELS);
    for (int32_t level_i = 0; level_i < CHUNK_LEVELS; ++level_i) {
        for (int32_t chunk_zi = -CHUNK_NZ; chunk_zi < CHUNK_NZ; ++chunk_zi) {
            for (int32_t chunk_yi = -CHUNK_NY; chunk_yi < CHUNK_NY; ++chunk_yi) {
                for (int32_t chunk_xi = -CHUNK_NX; chunk_xi < CHUNK_NX; ++chunk_xi) {
                    auto candidate = GenChunkArgs(self, chunk_xi, chunk_yi, chunk_zi, level_i, chunk_candidate_rating(chunk_xi, chunk_yi, chunk_zi, level_i));
                    if (candidate.rating >= 0)
                        self->chunk_candidates.push_back(candidate);
                }
            }
        }
    }
    std::sort(self->chunk_candidates.begin(), self->chunk_candidates.end(), [](const GenChunkArgs &a, const GenChunkArgs &b) { return a.rating > b.rating; });

    // generate_all_chunks(self);
    return self;
}
void destroy_voxel_world(VoxelWorld *self) {
    for (auto &chunk : self->chunks) {
        if (chunk.voxel_object) {
            for (auto *brick : chunk.voxel_object->brick_grid) {
                if (brick != nullptr) {
                    if (brick->render_attribs != nullptr) {
                        chunk.voxel_object->free_render_brick(brick->render_attribs);
                    }
                }
            }
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

void update_voxel_world(struct GpuContext &gpu_context, Renderer &renderer, GpuInput &gpu_input, VoxelWorld *self) {
    PROFILE_FUNC();
    std::vector<std::pair<thread_pool::Task, void *>> tasks;
    tasks.reserve(MAX_CHUNKS_PER_FRAME);

    for (int32_t candidate_i = self->chunk_candidates.size - 1; candidate_i >= 0; --candidate_i) {
        const auto &args = self->chunk_candidates[candidate_i];
        auto chunk_index = get_chunk_index(args.chunk_xi, args.chunk_yi, args.chunk_zi, args.level);
        auto &chunk = self->chunks[chunk_index];
        if (chunk.generation_stage != 0)
            continue;

        if (generate_chunk_precheck(args.self, args.chunk_xi, args.chunk_yi, args.chunk_zi, args.level) == 2) {
            auto *user_ptr = new GenChunkArgs(args);
            auto task = thread_pool::create_task(
                [](void *user_ptr) {
                    auto const &args = *(GenChunkArgs *)user_ptr;
                    generate_chunk(args.self, args.chunk_xi, args.chunk_yi, args.chunk_zi, args.level);
                },
                user_ptr);
            thread_pool::async_dispatch(task);
            tasks.emplace_back(task, user_ptr);
            self->chunk_candidates.erase(candidate_i);
            if (tasks.size() == MAX_CHUNKS_PER_FRAME)
                goto exit_1;
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
        PROFILE_SCOPE("find update chunks part 2");
        for (int32_t chunk_zi = -CHUNK_NZ; chunk_zi < CHUNK_NZ; ++chunk_zi) {
            for (int32_t chunk_yi = -CHUNK_NY; chunk_yi < CHUNK_NY; ++chunk_yi) {
                for (int32_t chunk_xi = -CHUNK_NX; chunk_xi < CHUNK_NX; ++chunk_xi) {
                    auto chunk_index = get_chunk_index(chunk_xi, chunk_yi, chunk_zi, level_i);
                    auto &chunk = self->chunks[chunk_index];
                    if (chunk.generation_stage != 2)
                        continue;
                    if (chunk_xi != CHUNK_NX - 1)
                        if (self->chunks[get_chunk_index(chunk_xi + 1, chunk_yi, chunk_zi, level_i)].generation_stage < 1)
                            continue;
                    if (chunk_xi != -CHUNK_NX)
                        if (self->chunks[get_chunk_index(chunk_xi - 1, chunk_yi, chunk_zi, level_i)].generation_stage < 1)
                            continue;
                    if (chunk_yi != CHUNK_NY - 1)
                        if (self->chunks[get_chunk_index(chunk_xi, chunk_yi + 1, chunk_zi, level_i)].generation_stage < 1)
                            continue;
                    if (chunk_yi != -CHUNK_NY)
                        if (self->chunks[get_chunk_index(chunk_xi, chunk_yi - 1, chunk_zi, level_i)].generation_stage < 1)
                            continue;
                    if (chunk_zi != CHUNK_NZ - 1)
                        if (self->chunks[get_chunk_index(chunk_xi, chunk_yi, chunk_zi + 1, level_i)].generation_stage < 1)
                            continue;
                    if (chunk_zi != -CHUNK_NZ)
                        if (self->chunks[get_chunk_index(chunk_xi, chunk_yi, chunk_zi - 1, level_i)].generation_stage < 1)
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
                        draw_voxel_object(chunk.voxel_object, pos, {}, voxel_size, tint);

                        for (auto surface_ent : chunk.surface_entity_candidates) {
                            auto &frames = self->scene->animation_playground->frames;
                            if (frames.size == 0)
                                break;

                            auto const current_frame_int = static_cast<int>(gpu_input.time * self->scene->animation_playground->playback_fps + rand()) % frames.size;
                            auto voxel_object = frames[current_frame_int];

                            auto grid_size = voxel_object->brick_max - voxel_object->brick_min + 1;
                            auto ball_pos = pos + (glm::vec3(surface_ent) + 0.5f) * float(BRICK_SIZE) * voxel_size - glm::vec3(grid_size.x, grid_size.y, 6) * 0.5f * float(BRICK_SIZE) * VOXEL_SIZE;
                            // auto ball_pos = pos + (glm::vec3(surface_ent) + 0.5f) * float(BRICK_SIZE) * voxel_size;
                            auto tint = hsv2rgb(glm::vec3(0.1, float(rand() % 100) / 100.f * 0.25f + 0.75f, 1));
                            // auto tint = glm::vec3(1);
                            draw_voxel_object(voxel_object, ball_pos, {0, 0, float(rand() % 100) / 100}, VOXEL_SIZE, tint);
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

struct GlmAabb {
    glm::vec3 min, max;
};

GlmAabb get_chunk_aabb(int32_t chunk_xi, int32_t chunk_yi, int32_t chunk_zi, int32_t level) {
    auto p0 = glm::vec3{
        (float((chunk_xi * CHUNK_SIZE_VOXELS) << level) + 0.5f) * VOXEL_SIZE,
        (float((chunk_yi * CHUNK_SIZE_VOXELS) << level) + 0.5f) * VOXEL_SIZE,
        (float((chunk_zi * CHUNK_SIZE_VOXELS) << level) + 0.5f) * VOXEL_SIZE,
    };
    auto p1 = p0 + (CHUNK_SIZE_VOXELS << level) * VOXEL_SIZE;
    return {p0, p1};
}

int generate_chunk_precheck(VoxelWorld *self, int32_t chunk_xi, int32_t chunk_yi, int32_t chunk_zi, int32_t level) {
    PROFILE_FUNC();
    auto chunk_index = get_chunk_index(chunk_xi, chunk_yi, chunk_zi, level);
    auto &chunk = self->chunks[chunk_index];

    chunk.generation_stage = 1;
    if (level > 0 &&
        chunk_xi >= -CHUNK_NX / 2 && chunk_xi < CHUNK_NX / 2 &&
        chunk_yi >= -CHUNK_NY / 2 && chunk_yi < CHUNK_NY / 2 &&
        chunk_zi >= -CHUNK_NZ / 2 && chunk_zi < CHUNK_NZ / 2) {
        // Skip generating chunks that overlap the lower LOD level
        return chunk.generation_stage;
    }

    {
        auto [p0, p1] = get_chunk_aabb(chunk_xi, chunk_yi, chunk_zi, level);
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

float chunk_candidate_rating(int32_t chunk_xi, int32_t chunk_yi, int32_t chunk_zi, int32_t level) {
    PROFILE_FUNC();

    glm::ivec3 neighbors[] = {
        {-1, 0, 0},
        {0, -1, 0},
        {0, 0, -1},
        {0, 0, 0},
        {+1, 0, 0},
        {0, +1, 0},
        {0, 0, +1},
    };
    for (auto &np : neighbors) {
        auto [np0, np1] = get_chunk_aabb(chunk_xi + np.x, chunk_yi + np.y, chunk_zi + np.z, level);
        auto minmax = voxel_minmax_value_cpp(&noise_settings, RANDOM_VALUES.data(), np0.x, np0.y, np0.z, np1.x, np1.y, np1.z);

        if (minmax.min <= 0.0f && minmax.max >= 0.0f) {
            auto [p0, p1] = get_chunk_aabb(chunk_xi + np.x, chunk_yi + np.y, chunk_zi + np.z, level);
            return length((p0 + p1) * 0.5f);
        }
    }

    return -1;
}

void generate_chunk(VoxelWorld *self, int32_t chunk_xi, int32_t chunk_yi, int32_t chunk_zi, int32_t level) {
    PROFILE_FUNC();
    auto chunk_index = get_chunk_index(chunk_xi, chunk_yi, chunk_zi, level);
    auto &chunk = self->chunks[chunk_index];

    chunk.voxel_object = new VoxelObject();
    chunk.voxel_object->allocator = self->scene->voxel_allocator;
    // chunk.pos = {chunk_xi, chunk_yi, chunk_zi};

    // auto t0 = Clock::now();
    chunk.voxel_object->init({0, 0, 0}, {CHUNK_SIZE_VOXELS - 1, CHUNK_SIZE_VOXELS - 1, CHUNK_SIZE_VOXELS - 1});

    for (int32_t brick_zi = 0; brick_zi < CHUNK_SIZE_BRICKS; ++brick_zi) {
        for (int32_t brick_yi = 0; brick_yi < CHUNK_SIZE_BRICKS; ++brick_yi) {
            for (int32_t brick_xi = 0; brick_xi < CHUNK_SIZE_BRICKS; ++brick_xi) {
                auto brick_index = brick_xi + brick_yi * CHUNK_SIZE_BRICKS + brick_zi * CHUNK_SIZE_BRICKS * CHUNK_SIZE_BRICKS;
                auto &brick = chunk.voxel_object->brick_grid[brick_index];
                brick = chunk.voxel_object->alloc_brick();
                auto &brick_metadata = get_brick_metadata(chunk, brick_index);
                auto &bitmask = brick->bitmask;

                brick_metadata = {};
                memset(&bitmask, 0, sizeof(bitmask));

                // determine if brick is uniform

                {
                    auto [p0, p1] = get_chunk_aabb(chunk_xi, chunk_yi, chunk_zi, level);
                    auto minmax = voxel_minmax_value_cpp(&noise_settings, RANDOM_VALUES.data(), p0.x, p0.y, p0.z, p1.x, p1.y, p1.z);
                    if (minmax.min >= 0.0f || minmax.max < 0.0f) {
                        // uniform
                        chunk.voxel_object->free_brick(brick);
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

static auto hash_combine(uint64_t h1, uint64_t h2) -> uint64_t {
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
}

uvec2 rand3(uvec2 pos) {
    uint index = hash_combine(pos.x, pos.y);
    return uvec2(
        hash_combine(index, index + 0),
        hash_combine(index, index + 1));
}

void generate_chunk2(VoxelWorld *self, int32_t chunk_xi, int32_t chunk_yi, int32_t chunk_zi, int32_t level, bool update) {
    PROFILE_FUNC();
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
                brick->render_attribs = nullptr;

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
                        render_attrib_brick = chunk.voxel_object->alloc_render_brick();
                        generate_attributes(brick_xi, brick_yi, brick_zi, chunk_xi, chunk_yi, chunk_zi, level,
                                            (uint32_t *)render_attrib_brick->voxels, (uint32_t *)brick->foliage_bitmask, &noise_settings, RANDOM_VALUES.data());
                        has_render_attribs = true;
                        auto pos = (glm::vec3(chunk_xi, chunk_yi, chunk_zi) * float(CHUNK_SIZE_BRICKS) + glm::vec3(brick_xi, brick_yi, brick_zi)) * float(BRICK_SIZE) * VOXEL_SIZE;

                        if ((uvec2(brick_xi, brick_yi) & 0x15u) == (rand3(uvec2(chunk_xi, chunk_yi)) & 0x15u)) {
                            auto low_pass_noise = noise_settings;
                            low_pass_noise.octaves -= 3;
                            float upwards = generate_upwards(brick_xi, brick_yi, brick_zi, chunk_xi, chunk_yi, chunk_zi, level, &low_pass_noise, RANDOM_VALUES.data());
                            if (chunk.surface_entity_candidates.size == 0 && upwards > 0.99f)
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
        chunk.voxel_object->render_voxel_object = create_render_voxel_object(self->scene->render_scene, /* has_foliage = */ true);
        chunk.voxel_object->render_dirty = true;
    }
}

static bool sample_generator(glm::ivec3 voxel_coord) {
    auto p = (glm::vec3(voxel_coord) + 0.5f) * VOXEL_SIZE;
    return voxel_is_solid_cpp(&noise_settings, RANDOM_VALUES.data(), p.x, p.y, p.z);
}

static VoxelBrick const *find_brick(VoxelWorld *self, glm::ivec3 brick_coord, bool &out_of_world) {
    auto chunk_i = brick_coord >> CHUNK_SIZE_BRICKS_LOG2;
    out_of_world =
        chunk_i.x < -CHUNK_NX || chunk_i.x >= CHUNK_NX ||
        chunk_i.y < -CHUNK_NY || chunk_i.y >= CHUNK_NY ||
        chunk_i.z < -CHUNK_NZ || chunk_i.z >= CHUNK_NZ;
    if (out_of_world) {
        return nullptr;
    }
    auto &chunk = self->chunks[get_chunk_index(chunk_i.x, chunk_i.y, chunk_i.z, 0)];
    if (chunk.voxel_object == nullptr) {
        return nullptr;
    }
    auto local_brick_coord = brick_coord & (CHUNK_SIZE_BRICKS - 1);
    return chunk.voxel_object->brick_grid[chunk.voxel_object->get_brick_index(local_brick_coord)];
}

static bool brick_bit(VoxelBrick const *brick, glm::ivec3 voxel_coord) {
    auto in_brick = VoxelObject::get_voxel_offset(voxel_coord);
    auto bit_i = in_brick.x + in_brick.y * BRICK_SIZE + in_brick.z * BRICK_SIZE * BRICK_SIZE;
    return ((brick->bitmask[bit_i >> 6] >> (bit_i & 63)) & 1) != 0;
}

static Voxel brick_render_attrib(VoxelBrick const *brick, glm::ivec3 voxel_coord) {
    auto in_brick = VoxelObject::get_voxel_offset(voxel_coord);
    auto voxel_i = in_brick.x + in_brick.y * BRICK_SIZE + in_brick.z * BRICK_SIZE * BRICK_SIZE;
    return unpack_voxel(brick->render_attribs->voxels[voxel_i]);
}

bool voxel_world_is_solid(VoxelWorld *self, glm::ivec3 voxel_coord) {
    bool out_of_world = false;
    auto const *brick = find_brick(self, VoxelObject::get_brick_coord(voxel_coord), out_of_world);
    // if (out_of_world) {
    //     return false;
    // }
    return brick == nullptr ? sample_generator(voxel_coord) : brick_bit(brick, voxel_coord);
}

glm::vec3 voxel_world_terrain_normal(VoxelWorld *self, glm::vec3 world_pos) {
    auto voxel_coord = glm::ivec3(glm::floor(world_pos * VOXEL_SCL));

    bool out_of_world = false;
    auto const *brick = find_brick(self, VoxelObject::get_brick_coord(voxel_coord), out_of_world);
    if (brick == nullptr || brick->render_attribs == nullptr || out_of_world) {
        float nrm[3];
        voxel_normal_cpp(&noise_settings, RANDOM_VALUES.data(), world_pos.x, world_pos.y, world_pos.z, nrm);
        return {nrm[0], nrm[1], nrm[2]};
    } else {
        auto voxel = brick_render_attrib(brick, voxel_coord);
        return {voxel.normal.x, voxel.normal.y, voxel.normal.z};
    }
}

bool voxel_world_box_is_solid(VoxelWorld *self, glm::vec3 box_min, glm::vec3 box_max, glm::ivec3 *out_hit_voxel) {
    PROFILE_FUNC();

    constexpr float BOUNDARY_EPS = 1e-4f;
    auto v0 = glm::ivec3(glm::floor(box_min * VOXEL_SCL + BOUNDARY_EPS));
    auto v1 = glm::ivec3(glm::floor(box_max * VOXEL_SCL - BOUNDARY_EPS));
    v1 = glm::max(v0, v1);

    auto cached_brick_coord = glm::ivec3(0);
    VoxelBrick const *cached_brick = nullptr;
    bool cached_out_of_world = false;
    bool have_cache = false;

    for (int32_t zi = v0.z; zi <= v1.z; ++zi) {
        for (int32_t yi = v0.y; yi <= v1.y; ++yi) {
            for (int32_t xi = v0.x; xi <= v1.x; ++xi) {
                auto voxel_coord = glm::ivec3(xi, yi, zi);
                auto brick_coord = VoxelObject::get_brick_coord(voxel_coord);
                if (!have_cache || brick_coord != cached_brick_coord) {
                    cached_brick_coord = brick_coord;
                    cached_brick = find_brick(self, brick_coord, cached_out_of_world);
                    have_cache = true;
                }
                if (cached_brick == nullptr ? sample_generator(voxel_coord) : brick_bit(cached_brick, voxel_coord)) {
                    if (out_hit_voxel != nullptr) {
                        *out_hit_voxel = voxel_coord;
                    }
                    return true;
                }
            }
        }
    }

    return false;
}
