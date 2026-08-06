#include "voxel_world.inl"
#include "utilities/debug.hpp"
#include <chrono>
#include <fmt/format.h>

#ifndef defer
struct defer_dummy {};
template <class F>
struct deferrer {
    F f;
    ~deferrer() { f(); }
};
template <class F>
deferrer<F> operator*(defer_dummy, F f) { return {f}; }
#define DEFER_(LINE) zz_defer##LINE
#define DEFER(LINE) DEFER_(LINE)
#define defer auto DEFER(__LINE__) = defer_dummy{} *[&]()
#endif // defer

static uint32_t calc_chunk_index(glm::uvec3 chunk_i, glm::ivec3 offset) {
    // Modulate the chunk index to be wrapped around relative to the chunk offset provided.
    auto temp_chunk_i = (glm::ivec3(chunk_i) + (offset >> glm::ivec3(6 + LOG2_VOXEL_SIZE))) % glm::ivec3(CHUNK_NX, CHUNK_NY, CHUNK_NZ);
    if (temp_chunk_i.x < 0) {
        temp_chunk_i.x += CHUNK_NX;
    }
    if (temp_chunk_i.y < 0) {
        temp_chunk_i.y += CHUNK_NY;
    }
    if (temp_chunk_i.z < 0) {
        temp_chunk_i.z += CHUNK_NZ;
    }
    chunk_i = glm::uvec3(temp_chunk_i);
    uint32_t chunk_index = chunk_i.x + chunk_i.y * CHUNK_NX + chunk_i.z * CHUNK_NX * CHUNK_NY;
    assert(chunk_index < CHUNK_NX * CHUNK_NY * CHUNK_NZ);
    return chunk_index;
}

static uint32_t calc_palette_region_index(glm::uvec3 inchunk_voxel_i) {
    glm::uvec3 palette_region_i = inchunk_voxel_i / uint32_t(PALETTE_REGION_SIZE);
    return palette_region_i.x + palette_region_i.y * PALETTES_PER_CHUNK_AXIS + palette_region_i.z * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS;
}

static uint32_t calc_palette_voxel_index(glm::uvec3 inchunk_voxel_i) {
    glm::uvec3 palette_voxel_i = inchunk_voxel_i & uint32_t(PALETTE_REGION_SIZE - 1);
    return palette_voxel_i.x + palette_voxel_i.y * PALETTE_REGION_SIZE + palette_voxel_i.z * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE;
}

PackedVoxel sample_palette(CpuPaletteChunk palette_header, uint32_t palette_voxel_index) {
    if (palette_header.variant_n < 2) {
        return PackedVoxel(static_cast<uint32_t>(std::bit_cast<uint64_t>(palette_header.blob_ptr)));
    }
    auto const *blob_u32s = palette_header.blob_ptr;
    if (palette_header.variant_n > PALETTE_MAX_COMPRESSED_VARIANT_N) {
        return PackedVoxel(blob_u32s[palette_voxel_index]);
    }
    auto bits_per_variant = ceil_log2(palette_header.variant_n);
    auto mask = (~0u) >> (32 - bits_per_variant);
    auto bit_index = palette_voxel_index * bits_per_variant;
    auto data_index = bit_index / 32;
    auto data_offset = bit_index - data_index * 32;
    auto my_palette_index = (blob_u32s[palette_header.variant_n + data_index + 0] >> data_offset) & mask;
    if (data_offset + bits_per_variant > 32) {
        auto shift = bits_per_variant - ((data_offset + bits_per_variant) & 0x1f);
        my_palette_index |= (blob_u32s[palette_header.variant_n + data_index + 1] << shift) & mask;
    }
    auto voxel_data = blob_u32s[my_palette_index];
    return PackedVoxel(voxel_data);
}

PackedVoxel sample_voxel_chunk(CpuVoxelChunk const &voxel_chunk, glm::uvec3 inchunk_voxel_i) {
    auto palette_region_index = calc_palette_region_index(inchunk_voxel_i);
    auto palette_voxel_index = calc_palette_voxel_index(inchunk_voxel_i);
    CpuPaletteChunk palette_header = voxel_chunk.palette_chunks[palette_region_index];
    return sample_palette(palette_header, palette_voxel_index);
}

float msign(float x) {
    return x >= 0.0f ? 1.0f : -1.0f;
}
uint32_t octahedral_8(daxa_f32vec3 nor) {
    auto temp_nor = nor;
    nor.x /= (abs(temp_nor.x) + abs(temp_nor.y) + abs(temp_nor.z));
    nor.y /= (abs(temp_nor.x) + abs(temp_nor.y) + abs(temp_nor.z));
    temp_nor = nor;
    nor.x = (temp_nor.z >= 0.0f) ? temp_nor.x : (1.0f - abs(temp_nor.y)) * msign(temp_nor.x);
    nor.y = (temp_nor.z >= 0.0f) ? temp_nor.y : (1.0f - abs(temp_nor.x)) * msign(temp_nor.y);
    auto d_x = uint32_t(round(7.5f + nor.x * 7.5f));
    auto d_y = uint32_t(round(7.5f + nor.y * 7.5f));
    return d_x | (d_y << 4u);
}
// daxa_f32vec3 i_octahedral_8(uint data) {
//     uvec2 iv = uvec2(data, data >> 4u) & 15u;
//     vec2 v = vec2(iv) / 7.5 - 1.0;
//     vec3 nor = vec3(v, 1.0 - abs(v.x) - abs(v.y)); // Rune Stubbe's version,
//     float t = max(-nor.z, 0.0);                    // much faster than original
//     nor.x += (nor.x > 0.0) ? -t : t;               // implementation of this
//     nor.y += (nor.y > 0.0) ? -t : t;               // technique
//     return normalize(nor);
// }
uint32_t pack_unit(float x, uint32_t bit_n) {
    float scl = float(1u << bit_n) - 1.0f;
    return uint32_t(round(x * scl));
}
float unpack_unit(uint32_t x, uint32_t bit_n) {
    float scl = float(1u << bit_n) - 1.0f;
    return float(x) / scl;
}
uint32_t pack_rgb565(daxa_f32vec3 f) {
    f.x = powf(f.x, 1.0f / 2.2f);
    f.y = powf(f.y, 1.0f / 2.2f);
    f.z = powf(f.z, 1.0f / 2.2f);
    uint32_t result = 0;
    result |= uint32_t(std::clamp(f.x * 31.0f, 0.0f, 31.0f)) << 0;
    result |= uint32_t(std::clamp(f.y * 63.0f, 0.0f, 63.0f)) << 5;
    result |= uint32_t(std::clamp(f.z * 31.0f, 0.0f, 31.0f)) << 11;
    return result;
}
PackedVoxel pack_voxel(Voxel v) {
    PackedVoxel result;

#if DITHER_NORMALS
    rand_seed(good_rand_hash(floatBitsToUint(v.normal)));
    const mat3 basis = build_orthonormal_basis(normalize(v.normal));
    v.normal = basis * uniform_sample_cone(vec2(rand(), rand()), cos(0.19 * 0.5));
#endif

    uint32_t packed_roughness = pack_unit(sqrt(v.roughness), 6);
    uint32_t packed_normal = octahedral_8(normalize(v.normal));
    uint32_t packed_color = pack_rgb565(v.color);

    result.data = (v.material_type) | (packed_roughness << 2) | (packed_normal << 8) | (packed_color << 16);

    return result;
}

float sd_sphere(daxa_f32vec3 p, float r) {
    return length(p) - r;
}
float sd_capsule(daxa_f32vec3 p, daxa_f32vec3 a, daxa_f32vec3 b, float r) {
    daxa_f32vec3 pa = p - a, ba = b - a;
    float h = std::clamp(dot(pa, ba) / dot(ba, ba), 0.0f, 1.0f);
    return length(pa - ba * h) - r;
}
float sd_test_entity(daxa_f32vec3 p, GpuInput const &gpu_input) {
    auto const &lateral = gpu_input.player.lateral;
    float d = 10000.0f;
    // d = std::min(d, sd_sphere(p, 0.45f));

    d = std::min(d, sd_capsule(p, daxa_f32vec3(0, 0, 0), daxa_f32vec3(0, 0, -1.75f + 0.35f), 0.35f));
    // arms
    // d = std::min(d, sd_capsule(p, daxa_f32vec3(0, 0, -0.45f), lateral * -(0.5f + cosf(gpu_input.time) * 0.1f) + daxa_f32vec3(0, 0, -0.75f + sinf(gpu_input.time) * 0.1f), 0.125f));
    // d = std::min(d, sd_capsule(p, daxa_f32vec3(0, 0, -0.45f), lateral * +(0.5f + cosf(gpu_input.time) * 0.1f) + daxa_f32vec3(0, 0, -0.75f + sinf(gpu_input.time) * 0.1f), 0.125f));
    // legs
    // d = std::min(d, sd_capsule(p, daxa_f32vec3(0, 0, -0.75f), lateral * +0.2f + daxa_f32vec3(0, 0, -1.675f), 0.125f));
    // d = std::min(d, sd_capsule(p, daxa_f32vec3(0, 0, -0.75f), lateral * -0.2f + daxa_f32vec3(0, 0, -1.675f), 0.125f));

    return d;
}

daxa_f32vec3 sd_test_entity_nrm(daxa_f32vec3 pos, GpuInput const &gpu_input) {
    daxa_f32vec3 eps = daxa_f32vec3(.001f, 0.0, 0.0);
    return normalize(daxa_f32vec3(
        sd_test_entity(pos + daxa_f32vec3(eps.x, eps.y, eps.y), gpu_input) - sd_test_entity(pos - daxa_f32vec3(eps.x, eps.y, eps.y), gpu_input),
        sd_test_entity(pos + daxa_f32vec3(eps.y, eps.x, eps.y), gpu_input) - sd_test_entity(pos - daxa_f32vec3(eps.y, eps.x, eps.y), gpu_input),
        sd_test_entity(pos + daxa_f32vec3(eps.y, eps.y, eps.x), gpu_input) - sd_test_entity(pos - daxa_f32vec3(eps.y, eps.y, eps.x), gpu_input)));
}

auto test_entity_voxel(daxa_f32vec3 p, GpuInput const &gpu_input) -> Voxel {
    auto voxel = Voxel{};
    float d = sd_test_entity(p, gpu_input);
    if (d < 0.0f) {
        if (d > float(VOXEL_SIZE) * -1.8f) {
            auto p_p = daxa_f32vec3(
                dot(p, gpu_input.player.lateral),
                dot(p, daxa_f32vec3(-gpu_input.player.lateral.y, gpu_input.player.lateral.x, gpu_input.player.lateral.z)),
                p.z);

            auto dist_sq = [](daxa_f32vec3 a, daxa_f32vec3 b) { return dot(a - b, a - b); };
            auto sq = [](float a) { return a * a; };

            if (dist_sq(p_p, daxa_f32vec3(-0.35f * 0.4f, +0.35f * 0.9f, 0.2f * 0.35f)) < sq(float(VOXEL_SIZE) * 1.0f) ||
                dist_sq(p_p, daxa_f32vec3(+0.35f * 0.4f, +0.35f * 0.9f, 0.2f * 0.35f)) < sq(float(VOXEL_SIZE) * 1.0f) ||
                (dist_sq(p_p, daxa_f32vec3(0.0f, +0.35f, 0.1f)) < sq(float(VOXEL_SIZE) * 4.5f) && p_p.z < 0.0f)) {
                voxel.color = daxa_f32vec3(0.0f, 0.0f, 0.0f);
            } else if (p_p.z < -0.6f) {
                voxel.color = daxa_f32vec3(0.1f, 0.1f, 1.0f);
            } else if (p_p.z < -0.45f) {
                voxel.color = daxa_f32vec3(1.0f, 0.1f, 0.1f);
            } else {
                voxel.color = daxa_f32vec3(1.0f, 0.7f, 0.5f);
            }

            voxel.normal = sd_test_entity_nrm(p, gpu_input);
        }

        voxel.material_type = 1;
        voxel.roughness = 0.9f;
    }
    return voxel;
}

// PackedVoxel sample_voxel_chunk(VoxelBufferPtrs ptrs, glm::uvec3 chunk_n, glm::vec3 voxel_p, glm::vec3 bias) {
//     vec3 offset = glm::vec3((deref(ptrs.globals).offset) & ((1 << (6 + LOG2_VOXEL_SIZE)) - 1)) + glm::vec3(chunk_n) * CHUNK_WORLDSPACE_SIZE * 0.5;
//     uvec3 voxel_i = glm::uvec3(floor((voxel_p + offset) * VOXEL_SCL + bias));
//     uvec3 chunk_i = voxel_i / CHUNK_SIZE;
//     uint chunk_index = calc_chunk_index(ptrs.globals, chunk_i, chunk_n);
//     return sample_voxel_chunk(ptrs.allocator, advance(ptrs.voxel_chunks_ptr, chunk_index), voxel_i - chunk_i * CHUNK_SIZE);
// }

bool VoxelWorld::sample(daxa_f32vec3 pos, daxa_i32vec3 player_unit_offset) {
    glm::vec3 offset = glm::vec3(std::bit_cast<glm::ivec3>(player_unit_offset) & ((1 << (6 + LOG2_VOXEL_SIZE)) - 1)) + glm::vec3(CHUNK_NX, CHUNK_NY, CHUNK_NZ) * CHUNK_WORLDSPACE_SIZE * 0.5f;
    // if (pos.x < offset.x || pos.y < offset.y || pos.z < offset.z) return false;
    // if (pos.x > offset.x + CHUNK_NX * CHUNK_SIZE * VOXEL_SIZE || pos.y > offset.y + CHUNK_NY * CHUNK_SIZE * VOXEL_SIZE || pos.z > offset.z + CHUNK_NZ * CHUNK_SIZE * VOXEL_SIZE) return false;
    glm::uvec3 voxel_i = glm::uvec3(floor((std::bit_cast<glm::vec3>(pos) + glm::vec3(offset)) * float(VOXEL_SCL)));
    glm::uvec3 chunk_i = voxel_i / uint32_t(CHUNK_SIZE);
    uint32_t chunk_index = calc_chunk_index(chunk_i, std::bit_cast<glm::ivec3>(player_unit_offset));

    auto packed_voxel = sample_voxel_chunk(voxel_chunks[chunk_index], voxel_i - chunk_i * uint32_t(CHUNK_SIZE));
    auto material_type = (packed_voxel.data >> 0) & 3;

    return material_type != 0;
}

const daxa_u32 ACCELERATION_STRUCTURE_BUILD_OFFSET_ALIGMENT = 256; // NOTE: Requested by the spec

auto get_aligned(daxa_u64 operand, daxa_u64 granularity) -> daxa_u64 {
    return ((operand + (granularity - 1)) & ~(granularity - 1));
};

void VoxelWorld::record_startup(GpuContext &gpu_context) {
    buffers.chunk_updates = gpu_context.find_or_add_temporal_buffer({
        .size = sizeof(ChunkUpdate) * MAX_CHUNK_UPDATES_PER_FRAME * (FRAMES_IN_FLIGHT + 1),
        .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
        .name = "chunk_updates",
    });
    buffers.chunk_update_heap = gpu_context.find_or_add_temporal_buffer({
        .size = sizeof(uint32_t) * MAX_CHUNK_UPDATES_PER_FRAME_VOXEL_COUNT * (FRAMES_IN_FLIGHT + 1),
        .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
        .name = "chunk_update_heap",
    });

    buffers.voxel_globals = gpu_context.find_or_add_temporal_buffer({
        .size = sizeof(VoxelWorldGlobals),
        .name = "voxel_globals",
    });

    auto chunk_n = CHUNK_NX * CHUNK_NY * CHUNK_NZ;
    buffers.voxel_chunks = gpu_context.find_or_add_temporal_buffer({
        .size = sizeof(VoxelLeafChunk) * chunk_n,
        .name = "voxel_chunks",
    });
    voxel_chunks.resize(chunk_n);

    gpu_context.frame_task_graph.use_persistent_buffer(buffers.chunk_updates.task_resource);
    gpu_context.frame_task_graph.use_persistent_buffer(buffers.chunk_update_heap.task_resource);
    gpu_context.frame_task_graph.use_persistent_buffer(buffers.voxel_globals.task_resource);
    gpu_context.frame_task_graph.use_persistent_buffer(buffers.voxel_chunks.task_resource);

    gpu_context.startup_task_graph.use_persistent_buffer(buffers.voxel_globals.task_resource);
    gpu_context.startup_task_graph.use_persistent_buffer(buffers.voxel_chunks.task_resource);

    gpu_context.startup_task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_globals.task_resource),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_chunks.task_resource),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            ti.recorder.clear_buffer({
                .buffer = buffers.voxel_globals.task_resource.get_state().buffers[0],
                .offset = 0,
                .size = sizeof(VoxelWorldGlobals),
                .clear_value = 0,
            });

            auto chunk_n = CHUNK_NX * CHUNK_NY * CHUNK_NZ;
            ti.recorder.clear_buffer({
                .buffer = buffers.voxel_chunks.task_resource.get_state().buffers[0],
                .offset = 0,
                .size = sizeof(VoxelLeafChunk) * chunk_n,
                .clear_value = 0,
            });
        },
        .name = "clear chunk editor",
    });

    gpu_context.add(ComputeTask<VoxelWorldStartupCompute::Task, VoxelWorldStartupComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/startup.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{VoxelWorldStartupCompute::AT.gpu_input, gpu_context.task_input_buffer}},
            VOXELS_BUFFER_USES_ASSIGN(VoxelWorldStartupCompute, buffers),
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, VoxelWorldStartupComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({1, 1, 1});
        },
        .task_graph_ptr = &gpu_context.startup_task_graph,
    });

    if (!rt_initialized) {
        rt_initialized = true;

        auto const MAX_BLAS_N = voxel_chunks.size() + 1;

        buffers.blas_geom_pointers = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(daxa::DeviceAddress) * MAX_BLAS_N,
            .name = "blas_geom_pointers",
        });
        buffers.blas_attr_pointers = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(daxa::DeviceAddress) * MAX_BLAS_N,
            .name = "blas_attr_pointers",
        });
        buffers.blas_transforms = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(VoxelBlasTransform) * MAX_BLAS_N,
            .name = "blas_transforms",
        });

        staging_blas_geom_pointers = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(daxa::DeviceAddress) * MAX_BLAS_N,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE, // TODO
            .name = "staging_blas_geom_pointers",
        });
        staging_blas_attr_pointers = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(daxa::DeviceAddress) * MAX_BLAS_N,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE, // TODO
            .name = "staging_blas_attr_pointers",
        });
        staging_blas_transforms = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(VoxelBlasTransform) * MAX_BLAS_N,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE, // TODO
            .name = "staging_blas_transforms",
        });

        blas_instances_buffer = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(daxa_BlasInstanceData) * MAX_BLAS_N,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE, // TODO
            .name = "blas instances array buffer",
        });

        daxa::TaskGraph temp_task_graph = daxa::TaskGraph({
            .device = gpu_context.device,
            .name = "temp_task_graph",
        });

        task_chunk_blases = daxa::TaskBlas({.name = "task_chunk_blases"});

        auto *blas_instances = gpu_context.device.get_host_address_as<daxa_BlasInstanceData>(blas_instances_buffer.resource_id).value();
        blas_chunks.reserve(MAX_BLAS_N);
        for (size_t blas_i = 0; blas_i < voxel_chunks.size(); ++blas_i) {
            auto &chunk = voxel_chunks[blas_i];
            // create new blas chunk
            chunk.blas_id = blas_chunks.size();
            blas_chunks.emplace_back();
            auto &blas_chunk = blas_chunks[chunk.blas_id];
            blas_instances[blas_i] = daxa_BlasInstanceData{
                .transform = {
                    {1, 0, 0, blas_chunk.position.x},
                    {0, 1, 0, blas_chunk.position.y},
                    {0, 0, 1, blas_chunk.position.z},
                },
                .instance_custom_index = uint32_t(blas_i),
                .mask = 0x00u,
                .instance_shader_binding_table_record_offset = 0,
                .flags = {},
                .blas_device_address = {},
            };
        }
        test_entity_blas_id = blas_chunks.size();
        blas_chunks.emplace_back();

        blas_instances[test_entity_blas_id] = daxa_BlasInstanceData{
            .transform = {
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0},
            },
            .instance_custom_index = test_entity_blas_id,
            .mask = 0x00u,
            .instance_shader_binding_table_record_offset = 0,
            .flags = {},
            .blas_device_address = {},
        };

        blas_instance_info = std::array{
            daxa::TlasInstanceInfo{
                .data = {}, // Ignored in get_acceleration_structure_build_sizes.
                .count = static_cast<uint32_t>(blas_chunks.size()),
                .is_data_array_of_pointers = false, // Buffer contains flat array of instances, not an array of pointers to instances.
                .flags = daxa::GeometryFlagBits::OPAQUE,
            },
        };
        blas_instance_info[0].data = gpu_context.device.get_device_address(blas_instances_buffer.resource_id).value();
        tlas_build_info = daxa::TlasBuildInfo{
            .flags = daxa::AccelerationStructureBuildFlagBits::PREFER_FAST_TRACE,
            .dst_tlas = {}, // Ignored in get_acceleration_structure_build_sizes.
            .instances = blas_instance_info,
            .scratch_data = {}, // Ignored in get_acceleration_structure_build_sizes.
        };

        tlas_build_sizes = gpu_context.device.get_tlas_build_sizes(tlas_build_info);
        buffers.tlas_buffer = gpu_context.device.create_buffer({
            .size = tlas_build_sizes.acceleration_structure_size,
            .name = "tlas build scratch buffer",
        });
        buffers.tlas = gpu_context.device.create_tlas_from_buffer({
            .tlas_info = {
                .size = tlas_build_sizes.acceleration_structure_size,
                .name = "blas",
            },
            .buffer_id = buffers.tlas_buffer,
            .offset = 0,
        });
        buffers.task_tlas = daxa::TaskTlas({.initial_tlas = {.tlas = std::array{buffers.tlas}}});
        temp_task_graph.use_persistent_tlas(buffers.task_tlas);
        temp_task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskTlasAccess::BUILD_WRITE, buffers.task_tlas),
            },
            .task = [this](daxa::TaskInterface const &ti) {
                auto tlas_scratch_buffer = ti.device.create_buffer({
                    .size = tlas_build_sizes.build_scratch_size,
                    .name = "tlas build scratch buffer",
                });
                ti.recorder.destroy_buffer_deferred(tlas_scratch_buffer);

                tlas_build_info.dst_tlas = buffers.tlas;
                tlas_build_info.scratch_data = ti.device.get_device_address(tlas_scratch_buffer).value();

                ti.recorder.build_acceleration_structures({
                    .tlas_build_infos = std::array{tlas_build_info},
                });
            },
            .name = "tlas build",
        });
        temp_task_graph.submit({});
        temp_task_graph.complete({});
        temp_task_graph.execute({});
    }

    gpu_context.frame_task_graph.use_persistent_tlas(buffers.task_tlas);
    gpu_context.frame_task_graph.use_persistent_buffer(buffers.blas_geom_pointers.task_resource);
    gpu_context.frame_task_graph.use_persistent_buffer(buffers.blas_attr_pointers.task_resource);
    gpu_context.frame_task_graph.use_persistent_buffer(buffers.blas_transforms.task_resource);
}

static constexpr auto get_chunk_index(int xi, int yi, int zi) -> int {
    return xi + yi * CHUNK_NX + zi * CHUNK_NX * CHUNK_NY;
}

static constexpr auto positive_modulo(int i, int n) -> int {
    return (i % n + n) % n;
}

void VoxelWorld::begin_frame(daxa::Device &device, GpuInput const &gpu_input) {
    update_chunks(device, gpu_input);
}

void VoxelWorld::update_chunks(daxa::Device &device, GpuInput const &gpu_input) {
    // using Clock = std::chrono::high_resolution_clock;
    // auto t0 = Clock::now();
    // std::vector<std::pair<Clock::time_point, std::string>> time_points;

    auto const offset = (gpu_input.frame_index + 0) % (FRAMES_IN_FLIGHT + 1);
    auto const *output_heap = device.get_host_address_as<uint32_t>(buffers.chunk_update_heap.resource_id).value() + offset * MAX_CHUNK_UPDATES_PER_FRAME_VOXEL_COUNT;
    auto const *chunk_updates_ptr = device.get_host_address_as<ChunkUpdate>(buffers.chunk_updates.resource_id).value() + offset * MAX_CHUNK_UPDATES_PER_FRAME;
    auto chunk_updates = std::vector<ChunkUpdate>{};
    chunk_updates.resize(MAX_CHUNK_UPDATES_PER_FRAME);
    memcpy(chunk_updates.data(), chunk_updates_ptr, chunk_updates.size() * sizeof(ChunkUpdate));
    // time_points.push_back({Clock::now(), "read-back chunk updates"});

    auto copied_bytes = 0u;

    auto *blas_instances = device.get_host_address_as<daxa_BlasInstanceData>(blas_instances_buffer.resource_id).value();

    auto voxel_is_air = [](uint32_t packed_voxel_data) {
        auto material_type = (packed_voxel_data >> 0) & 3;
        return material_type == 0;
    };
    auto voxel_is_null = [](uint32_t packed_voxel_data) {
        return (packed_voxel_data & PACKED_NULL_VOXEL_MASK) == PACKED_NULL_VOXEL;
    };
    dirty_chunks.clear();

    for (auto const &chunk_update : chunk_updates) {
        if (chunk_update.info.flags != 1) {
            // copied_bytes += sizeof(uint32_t);
            continue;
        }
        copied_bytes += sizeof(chunk_update);
        auto &voxel_chunk = voxel_chunks[chunk_update.info.chunk_index];

        bool px_face_updated = false;
        bool py_face_updated = false;
        bool pz_face_updated = false;
        bool nx_face_updated = false;
        bool ny_face_updated = false;
        bool nz_face_updated = false;

        for (uint32_t palette_region_i = 0; palette_region_i < PALETTES_PER_CHUNK; ++palette_region_i) {
            auto const &palette_header = chunk_update.palette_headers[palette_region_i];
            auto &palette_chunk = voxel_chunk.palette_chunks[palette_region_i];
            auto prev_palette_chunk = palette_chunk;
            auto palette_size = palette_header.variant_n;
            auto compressed_size = 0u;
            palette_chunk.variant_n = palette_size;
            auto bits_per_variant = ceil_log2(palette_size);
            if (palette_size > PALETTE_MAX_COMPRESSED_VARIANT_N) {
                compressed_size = PALETTE_REGION_TOTAL_SIZE;
                palette_size = PALETTE_REGION_TOTAL_SIZE;
            } else if (palette_size > 1) {
                compressed_size = palette_size + (bits_per_variant * PALETTE_REGION_TOTAL_SIZE + 31) / 32;
            } else {
                // no blob
            }
            palette_chunk.has_air = false;
            bool has_null = false;
            if (compressed_size != 0) {
                palette_chunk.blob_ptr = new uint32_t[compressed_size];
                memcpy(palette_chunk.blob_ptr, output_heap + palette_header.blob_ptr, compressed_size * sizeof(uint32_t));
                copied_bytes += compressed_size * sizeof(uint32_t);
                for (uint32_t i = 0; i < palette_size; ++i) {
                    if (voxel_is_null(palette_chunk.blob_ptr[i])) {
                        has_null = true;
                    } else if (voxel_is_air(palette_chunk.blob_ptr[i])) {
                        palette_chunk.has_air = true;
                    }
                    if (has_null && palette_chunk.has_air) {
                        break;
                    }
                }
            } else {
                palette_chunk.blob_ptr = std::bit_cast<uint32_t *>(size_t(palette_header.blob_ptr));
                if (voxel_is_null(palette_header.blob_ptr)) {
                    has_null = true;
                } else if (voxel_is_air(palette_header.blob_ptr)) {
                    palette_chunk.has_air = true;
                }
            }

            if (has_null) {
                // Merge palette chunk with previous palette chunk, as there are some null voxels.
                // In some cases this may mean no update is necessary at all.
                if (compressed_size == 0) {
                    // Whole palette chunk is null voxels, just re-use old palette chunk.
                    if (prev_palette_chunk.blob_ptr == nullptr) {
                        prev_palette_chunk.has_air = true;
                        prev_palette_chunk.variant_n = 1;
                    }
                    palette_chunk = prev_palette_chunk;
                } else {
                    // For now, just delete the old palette chunk
                    // TODO: implement palette merging
                    if (prev_palette_chunk.variant_n > 1) {
                        delete[] prev_palette_chunk.blob_ptr;
                    }
                    palette_chunk.has_air = true;
                }
            }

            if (palette_chunk.has_air) {
                // figure out which faces have air, for finer grained surface culling
                for (uint32_t xi = 0; xi < 2; ++xi) {
                    bool has_air = false;
                    for (uint32_t zi = 0; zi < PALETTE_REGION_SIZE; ++zi) {
                        for (uint32_t yi = 0; yi < PALETTE_REGION_SIZE; ++yi) {
                            if (voxel_is_air(sample_palette(palette_chunk, xi * (PALETTE_REGION_SIZE - 1) + yi * PALETTE_REGION_SIZE + zi * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE).data)) {
                                has_air = true;
                                goto loop_exit_x;
                            }
                        }
                    }
                loop_exit_x:
                    if (xi == 0) {
                        palette_chunk.has_air_nx = has_air;
                    } else {
                        palette_chunk.has_air_px = has_air;
                    }
                }
                for (uint32_t yi = 0; yi < 2; ++yi) {
                    bool has_air = false;
                    for (uint32_t zi = 0; zi < PALETTE_REGION_SIZE; ++zi) {
                        for (uint32_t xi = 0; xi < PALETTE_REGION_SIZE; ++xi) {
                            if (voxel_is_air(sample_palette(palette_chunk, xi + yi * (PALETTE_REGION_SIZE - 1) * PALETTE_REGION_SIZE + zi * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE).data)) {
                                has_air = true;
                                goto loop_exit_y;
                            }
                        }
                    }
                loop_exit_y:
                    if (yi == 0) {
                        palette_chunk.has_air_ny = has_air;
                    } else {
                        palette_chunk.has_air_py = has_air;
                    }
                }
                for (uint32_t zi = 0; zi < 2; ++zi) {
                    bool has_air = false;
                    for (uint32_t yi = 0; yi < PALETTE_REGION_SIZE; ++yi) {
                        for (uint32_t xi = 0; xi < PALETTE_REGION_SIZE; ++xi) {
                            if (voxel_is_air(sample_palette(palette_chunk, xi + yi * PALETTE_REGION_SIZE + zi * (PALETTE_REGION_SIZE - 1) * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE).data)) {
                                has_air = true;
                                goto loop_exit_z;
                            }
                        }
                    }
                loop_exit_z:
                    if (zi == 0) {
                        palette_chunk.has_air_nz = has_air;
                    } else {
                        palette_chunk.has_air_pz = has_air;
                    }
                }
            }

            {
                // updated palette chunk
                auto palette_region_xi = (palette_region_i / 1) % PALETTES_PER_CHUNK_AXIS;
                auto palette_region_yi = (palette_region_i / PALETTES_PER_CHUNK_AXIS) % PALETTES_PER_CHUNK_AXIS;
                auto palette_region_zi = (palette_region_i / PALETTES_PER_CHUNK_AXIS / PALETTES_PER_CHUNK_AXIS);

                if (palette_chunk.has_air_nx != prev_palette_chunk.has_air_nx && palette_region_xi == 0)
                    nx_face_updated = true;
                if (palette_chunk.has_air_ny != prev_palette_chunk.has_air_ny && palette_region_yi == 0)
                    ny_face_updated = true;
                if (palette_chunk.has_air_nz != prev_palette_chunk.has_air_nz && palette_region_zi == 0)
                    nz_face_updated = true;
                if (palette_chunk.has_air_px != prev_palette_chunk.has_air_px && palette_region_xi == PALETTES_PER_CHUNK_AXIS - 1)
                    px_face_updated = true;
                if (palette_chunk.has_air_py != prev_palette_chunk.has_air_py && palette_region_yi == PALETTES_PER_CHUNK_AXIS - 1)
                    py_face_updated = true;
                if (palette_chunk.has_air_pz != prev_palette_chunk.has_air_pz && palette_region_zi == PALETTES_PER_CHUNK_AXIS - 1)
                    pz_face_updated = true;
            }
        }

        {
            auto chunk_xi = int(chunk_update.info.chunk_index / 1) % CHUNK_NX;
            auto chunk_yi = int(chunk_update.info.chunk_index / CHUNK_NX) % CHUNK_NY;
            auto chunk_zi = int(chunk_update.info.chunk_index / CHUNK_NX / CHUNK_NY);

            int32_t chunk_xi_ws = (int32_t(chunk_xi) - (gpu_input.player.player_unit_offset.x >> (6 + LOG2_VOXEL_SIZE))) % CHUNK_NX;
            int32_t chunk_yi_ws = (int32_t(chunk_yi) - (gpu_input.player.player_unit_offset.y >> (6 + LOG2_VOXEL_SIZE))) % CHUNK_NY;
            int32_t chunk_zi_ws = (int32_t(chunk_zi) - (gpu_input.player.player_unit_offset.z >> (6 + LOG2_VOXEL_SIZE))) % CHUNK_NZ;

            // mark neighbors as dirty
            if (nx_face_updated && chunk_xi_ws != 0) {
                auto chunk_nxi = positive_modulo(chunk_xi - 1, CHUNK_NX);
                auto chunk_index = get_chunk_index(chunk_nxi, chunk_yi, chunk_zi);
                auto blas_index = voxel_chunks[chunk_index].blas_id;
                dirty_chunks.insert((uint64_t(blas_index) << 32) | uint64_t(chunk_index));
            }
            if (ny_face_updated && chunk_yi_ws != 0) {
                auto chunk_nyi = positive_modulo(chunk_yi - 1, CHUNK_NY);
                auto chunk_index = get_chunk_index(chunk_xi, chunk_nyi, chunk_zi);
                auto blas_index = voxel_chunks[chunk_index].blas_id;
                dirty_chunks.insert((uint64_t(blas_index) << 32) | uint64_t(chunk_index));
            }
            if (nz_face_updated && chunk_zi_ws != 0) {
                auto chunk_nzi = positive_modulo(chunk_zi - 1, CHUNK_NZ);
                auto chunk_index = get_chunk_index(chunk_xi, chunk_yi, chunk_nzi);
                auto blas_index = voxel_chunks[chunk_index].blas_id;
                dirty_chunks.insert((uint64_t(blas_index) << 32) | uint64_t(chunk_index));
            }
            if (px_face_updated && chunk_xi_ws != (CHUNK_NX - 1)) {
                auto chunk_nxi = (chunk_xi + 1) % CHUNK_NX;
                auto chunk_index = get_chunk_index(chunk_nxi, chunk_yi, chunk_zi);
                auto blas_index = voxel_chunks[chunk_index].blas_id;
                dirty_chunks.insert((uint64_t(blas_index) << 32) | uint64_t(chunk_index));
            }
            if (py_face_updated && chunk_yi_ws != (CHUNK_NY - 1)) {
                auto chunk_nyi = (chunk_yi + 1) % CHUNK_NY;
                auto chunk_index = get_chunk_index(chunk_xi, chunk_nyi, chunk_zi);
                auto blas_index = voxel_chunks[chunk_index].blas_id;
                dirty_chunks.insert((uint64_t(blas_index) << 32) | uint64_t(chunk_index));
            }
            if (pz_face_updated && chunk_zi_ws != (CHUNK_NZ - 1)) {
                auto chunk_nzi = (chunk_zi + 1) % CHUNK_NZ;
                auto chunk_index = get_chunk_index(chunk_xi, chunk_yi, chunk_nzi);
                auto blas_index = voxel_chunks[chunk_index].blas_id;
                dirty_chunks.insert((uint64_t(blas_index) << 32) | uint64_t(chunk_index));
            }
        }

        dirty_chunks.insert((uint64_t(voxel_chunk.blas_id) << 32) | uint64_t(chunk_update.info.chunk_index));
    }
    // time_points.push_back({Clock::now(), "process chunk updates"});

    // if (copied_bytes > 0) {
    //     debug_utils::Console::add_log(fmt::format("{} MB copied", double(copied_bytes) / 1'000'000.0));
    // }

    auto chunks_moved = gpu_input.player.player_unit_offset.x != gpu_input.player.prev_unit_offset.x ||
                        gpu_input.player.player_unit_offset.y != gpu_input.player.prev_unit_offset.y ||
                        gpu_input.player.player_unit_offset.z != gpu_input.player.prev_unit_offset.z ||
                        gpu_input.frame_index <= 2;

    if (chunks_moved) {
        for (uint64_t chunk_i = 0; chunk_i < voxel_chunks.size(); ++chunk_i) {
            auto &voxel_chunk = voxel_chunks[chunk_i];
            auto &blas_chunk = blas_chunks[voxel_chunk.blas_id];

            uint32_t chunk_xi = (chunk_i / 1) % CHUNK_NX;
            uint32_t chunk_yi = (chunk_i / CHUNK_NX) % CHUNK_NY;
            uint32_t chunk_zi = (chunk_i / CHUNK_NX / CHUNK_NY) % CHUNK_NZ;
            auto const CHUNK_WS_SIZE = float(CHUNK_SIZE * VOXEL_SIZE);
            int32_t chunk_xi_ws = (int32_t(chunk_xi) - (gpu_input.player.player_unit_offset.x >> (6 + LOG2_VOXEL_SIZE))) % CHUNK_NX;
            int32_t chunk_yi_ws = (int32_t(chunk_yi) - (gpu_input.player.player_unit_offset.y >> (6 + LOG2_VOXEL_SIZE))) % CHUNK_NY;
            int32_t chunk_zi_ws = (int32_t(chunk_zi) - (gpu_input.player.player_unit_offset.z >> (6 + LOG2_VOXEL_SIZE))) % CHUNK_NZ;
            blas_chunk.prev_position = blas_chunk.position;
            blas_chunk.position = {
                chunk_xi_ws * CHUNK_WS_SIZE - CHUNK_WS_SIZE * (float(CHUNK_NX / 2)) - (gpu_input.player.player_unit_offset.x & ((1 << (6 + LOG2_VOXEL_SIZE)) - 1)),
                chunk_yi_ws * CHUNK_WS_SIZE - CHUNK_WS_SIZE * (float(CHUNK_NY / 2)) - (gpu_input.player.player_unit_offset.y & ((1 << (6 + LOG2_VOXEL_SIZE)) - 1)),
                chunk_zi_ws * CHUNK_WS_SIZE - CHUNK_WS_SIZE * (float(CHUNK_NZ / 2)) - (gpu_input.player.player_unit_offset.z & ((1 << (6 + LOG2_VOXEL_SIZE)) - 1)),
            };
            blas_chunk.cull_mask = 0xff;
        }
        // time_points.push_back({Clock::now(), "update chunk pos"});
    }

    for (auto packed_dirty_chunk_index : dirty_chunks) {
        auto chunk_i = packed_dirty_chunk_index & ((uint64_t(1) << 32) - 1);
        auto &voxel_chunk = voxel_chunks[chunk_i];
        auto &blas_chunk = blas_chunks[voxel_chunk.blas_id];

        blas_chunk.blas_geoms.clear();
        blas_chunk.attrib_bricks.clear();
        for (int32_t palette_zi = 0; palette_zi < PALETTES_PER_CHUNK_AXIS; ++palette_zi) {
            for (int32_t palette_yi = 0; palette_yi < PALETTES_PER_CHUNK_AXIS; ++palette_yi) {
                for (int32_t palette_xi = 0; palette_xi < PALETTES_PER_CHUNK_AXIS; ++palette_xi) {
                    auto palette_region_i = palette_xi + palette_yi * PALETTES_PER_CHUNK_AXIS + palette_zi * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS;
                    auto &palette_chunk = voxel_chunk.palette_chunks[palette_region_i];
                    auto neighbors_air = false;
                    // check neighbor palettes
                    for (int32_t ni = 0; ni < 3; ++ni) {
                        for (int32_t nj = -1; nj <= 1; nj += 2) {
                            int32_t nxi = ni == 0 ? nj : 0;
                            int32_t nyi = ni == 1 ? nj : 0;
                            int32_t nzi = ni == 2 ? nj : 0;
                            CpuPaletteChunk const *temp_palette_chunk = nullptr;
                            if ((nxi == -1 && palette_xi == 0) || (nxi == 1 && palette_xi == PALETTES_PER_CHUNK_AXIS - 1) ||
                                (nyi == -1 && palette_yi == 0) || (nyi == 1 && palette_yi == PALETTES_PER_CHUNK_AXIS - 1) ||
                                (nzi == -1 && palette_zi == 0) || (nzi == 1 && palette_zi == PALETTES_PER_CHUNK_AXIS - 1)) {
                                auto neighbor_chunk_xi = positive_modulo(int32_t(chunk_i % CHUNK_NX) + nxi, CHUNK_NX);
                                auto neighbor_chunk_yi = positive_modulo(int32_t((chunk_i / CHUNK_NX) % CHUNK_NY) + nyi, CHUNK_NY);
                                auto neighbor_chunk_zi = positive_modulo(int32_t(chunk_i / CHUNK_NX / CHUNK_NY) + nzi, CHUNK_NZ);
                                int32_t chunk_xi_ws = positive_modulo(int32_t(neighbor_chunk_xi) - (gpu_input.player.player_unit_offset.x >> (6 + LOG2_VOXEL_SIZE)), CHUNK_NX);
                                int32_t chunk_yi_ws = positive_modulo(int32_t(neighbor_chunk_yi) - (gpu_input.player.player_unit_offset.y >> (6 + LOG2_VOXEL_SIZE)), CHUNK_NY);
                                int32_t chunk_zi_ws = positive_modulo(int32_t(neighbor_chunk_zi) - (gpu_input.player.player_unit_offset.z >> (6 + LOG2_VOXEL_SIZE)), CHUNK_NZ);
                                if ((nxi == -1 && chunk_xi_ws == (CHUNK_NX - 1)) || (nxi == 1 && chunk_xi_ws == 0) ||
                                    (nyi == -1 && chunk_yi_ws == (CHUNK_NY - 1)) || (nyi == 1 && chunk_yi_ws == 0) ||
                                    (nzi == -1 && chunk_zi_ws == (CHUNK_NZ - 1)) || (nzi == 1 && chunk_zi_ws == 0))
                                    continue;
                                auto &neighbor_chunk = voxel_chunks[get_chunk_index(neighbor_chunk_xi, neighbor_chunk_yi, neighbor_chunk_zi)];
                                auto neighbor_palette_xi = (palette_xi + nxi) & (PALETTES_PER_CHUNK_AXIS - 1);
                                auto neighbor_palette_yi = (palette_yi + nyi) & (PALETTES_PER_CHUNK_AXIS - 1);
                                auto neighbor_palette_zi = (palette_zi + nzi) & (PALETTES_PER_CHUNK_AXIS - 1);
                                temp_palette_chunk = &neighbor_chunk.palette_chunks[neighbor_palette_xi + neighbor_palette_yi * PALETTES_PER_CHUNK_AXIS + neighbor_palette_zi * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS];
                            } else {
                                temp_palette_chunk = &voxel_chunk.palette_chunks[palette_region_i + nxi + nyi * PALETTES_PER_CHUNK_AXIS + nzi * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS];
                            }
                            if ((temp_palette_chunk->has_air_nx && nxi == 1) || (temp_palette_chunk->has_air_px && nxi == -1) ||
                                (temp_palette_chunk->has_air_ny && nyi == 1) || (temp_palette_chunk->has_air_py && nyi == -1) ||
                                (temp_palette_chunk->has_air_nz && nzi == 1) || (temp_palette_chunk->has_air_pz && nzi == -1)) {
                                neighbors_air = true;
                            }
                        }
                    }
                    if (((palette_chunk.variant_n > 1) || (palette_chunk.variant_n == 1 && !palette_chunk.has_air)) && neighbors_air) {
                        blas_chunk.blas_geoms.push_back({});
                        blas_chunk.attrib_bricks.push_back({});
                        auto &blas_geom = blas_chunk.blas_geoms.back();
                        auto &blas_attr = blas_chunk.attrib_bricks.back();
                        auto blas_geom_i = palette_region_i;
                        uint32_t blas_geom_xi = (blas_geom_i / 1) % 8;
                        uint32_t blas_geom_yi = (blas_geom_i / 8) % 8;
                        uint32_t blas_geom_zi = (blas_geom_i / 64) % 8;
                        blas_geom.aabb.minimum = {
                            blas_geom_xi * float(VOXEL_SIZE) * BLAS_BRICK_SIZE,
                            blas_geom_yi * float(VOXEL_SIZE) * BLAS_BRICK_SIZE,
                            blas_geom_zi * float(VOXEL_SIZE) * BLAS_BRICK_SIZE,
                        };
                        blas_geom.aabb.maximum = blas_geom.aabb.minimum;
                        int32_t min_xi = BLAS_BRICK_SIZE;
                        int32_t min_yi = BLAS_BRICK_SIZE;
                        int32_t min_zi = BLAS_BRICK_SIZE;
                        int32_t max_xi = 0;
                        int32_t max_yi = 0;
                        int32_t max_zi = 0;
                        for (int32_t zi = 0; zi < BLAS_BRICK_SIZE; ++zi) {
                            for (int32_t yi = 0; yi < BLAS_BRICK_SIZE; ++yi) {
                                for (int32_t xi = 0; xi < BLAS_BRICK_SIZE; ++xi) {
                                    const uint32_t bit_index = xi + yi * BLAS_BRICK_SIZE + zi * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE;
                                    const uint32_t u32_index = bit_index / 32;
                                    const uint32_t in_u32_index = bit_index & 0x1f;
                                    blas_geom.bitmask[u32_index] &= ~(1 << in_u32_index);
                                    auto packed_voxel = sample_palette(palette_chunk, bit_index);
                                    blas_attr.packed_voxels[bit_index] = packed_voxel;
                                    // set bit
                                    if (!voxel_is_air(packed_voxel.data)) {
                                        min_xi = glm::min(min_xi, xi);
                                        min_yi = glm::min(min_yi, yi);
                                        min_zi = glm::min(min_zi, zi);
                                        max_xi = glm::max(max_xi, xi + 1);
                                        max_yi = glm::max(max_yi, yi + 1);
                                        max_zi = glm::max(max_zi, zi + 1);
                                        blas_geom.bitmask[u32_index] |= 1 << in_u32_index;
                                    }
                                }
                            }
                        }
                        blas_geom.aabb.maximum.x += float(VOXEL_SIZE) * BLAS_BRICK_SIZE;
                        blas_geom.aabb.maximum.y += float(VOXEL_SIZE) * BLAS_BRICK_SIZE;
                        blas_geom.aabb.maximum.z += float(VOXEL_SIZE) * BLAS_BRICK_SIZE;
                        // blas_geom.aabb.minimum.x += float(VOXEL_SIZE) * min_xi;
                        // blas_geom.aabb.minimum.y += float(VOXEL_SIZE) * min_yi;
                        // blas_geom.aabb.minimum.z += float(VOXEL_SIZE) * min_zi;
                        // blas_geom.aabb.maximum.x += float(VOXEL_SIZE) * max_xi;
                        // blas_geom.aabb.maximum.y += float(VOXEL_SIZE) * max_yi;
                        // blas_geom.aabb.maximum.z += float(VOXEL_SIZE) * max_zi;
                    }
                }
            }
        }
    }
    // time_points.push_back({Clock::now(), "update chunk geom"});

    {
        auto &blas_chunk = blas_chunks[test_entity_blas_id];
        blas_chunk.prev_position = blas_chunk.position;
        blas_chunk.position = {
            round(gpu_input.player.pos.x * VOXEL_SCL) * float(VOXEL_SIZE),
            round(gpu_input.player.pos.y * VOXEL_SCL) * float(VOXEL_SIZE),
            round(gpu_input.player.pos.z * VOXEL_SCL) * float(VOXEL_SIZE),
        };
        blas_chunk.cull_mask = 0x01;

        dirty_chunks.insert(uint64_t(test_entity_blas_id) << 32);
        blas_chunk.blas_geoms.clear();
        blas_chunk.attrib_bricks.clear();

        for (int32_t azi = -3; azi <= 1; ++azi) {
            for (int32_t ayi = -1; ayi <= 1; ++ayi) {
                for (int32_t axi = -1; axi <= 1; ++axi) {
                    blas_chunk.blas_geoms.push_back({});
                    blas_chunk.attrib_bricks.push_back({});
                    auto &blas_geom = blas_chunk.blas_geoms.back();
                    auto &blas_attr = blas_chunk.attrib_bricks.back();

                    blas_geom.aabb.minimum = {
                        (-0.5f + float(axi)) * float(VOXEL_SIZE) * BLAS_BRICK_SIZE,
                        (-0.5f + float(ayi)) * float(VOXEL_SIZE) * BLAS_BRICK_SIZE,
                        (-0.5f + float(azi)) * float(VOXEL_SIZE) * BLAS_BRICK_SIZE,
                    };
                    blas_geom.aabb.maximum = blas_geom.aabb.minimum;
                    blas_geom.aabb.maximum.x += float(VOXEL_SIZE) * BLAS_BRICK_SIZE;
                    blas_geom.aabb.maximum.y += float(VOXEL_SIZE) * BLAS_BRICK_SIZE;
                    blas_geom.aabb.maximum.z += float(VOXEL_SIZE) * BLAS_BRICK_SIZE;

                    bool has_voxel = false;

                    for (int32_t zi = 0; zi < BLAS_BRICK_SIZE; ++zi) {
                        for (int32_t yi = 0; yi < BLAS_BRICK_SIZE; ++yi) {
                            for (int32_t xi = 0; xi < BLAS_BRICK_SIZE; ++xi) {
                                const uint32_t bit_index = xi + yi * BLAS_BRICK_SIZE + zi * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE;
                                const uint32_t u32_index = bit_index / 32;
                                const uint32_t in_u32_index = bit_index & 0x1f;
                                blas_geom.bitmask[u32_index] &= ~(1 << in_u32_index);
                                float x = (float(xi - BLAS_BRICK_SIZE / 2 + axi * BLAS_BRICK_SIZE) + 0.5f) * float(VOXEL_SIZE) - (gpu_input.player.pos.x - blas_chunk.position.x);
                                float y = (float(yi - BLAS_BRICK_SIZE / 2 + ayi * BLAS_BRICK_SIZE) + 0.5f) * float(VOXEL_SIZE) - (gpu_input.player.pos.y - blas_chunk.position.y);
                                float z = (float(zi - BLAS_BRICK_SIZE / 2 + azi * BLAS_BRICK_SIZE) + 0.5f) * float(VOXEL_SIZE) - (gpu_input.player.pos.z - blas_chunk.position.z);
                                auto voxel = test_entity_voxel(daxa_f32vec3(x, y, z), gpu_input);
                                auto packed_voxel = blas_attr.packed_voxels[bit_index] = pack_voxel(voxel);
                                // set bit
                                if (!voxel_is_air(packed_voxel.data)) {
                                    blas_geom.bitmask[u32_index] |= 1 << in_u32_index;
                                    has_voxel = true;
                                }
                            }
                        }
                    }

                    if (!has_voxel) {
                        blas_chunk.blas_geoms.pop_back();
                        blas_chunk.attrib_bricks.pop_back();
                    }
                }
            }
        }
    }
    // time_points.push_back({Clock::now(), "update player chunk"});

    auto geom_pointers_host_ptr = device.get_host_address_as<daxa::DeviceAddress>(staging_blas_geom_pointers.resource_id).value();
    auto attr_pointers_host_ptr = device.get_host_address_as<daxa::DeviceAddress>(staging_blas_attr_pointers.resource_id).value();
    auto acceleration_structure_scratch_offset_alignment = device.properties().acceleration_structure_properties.value().min_acceleration_structure_scratch_offset_alignment;
    for (auto packed_dirty_chunk_index : dirty_chunks) {
        auto blas_i = uint32_t(packed_dirty_chunk_index >> 32);
        auto &blas_chunk = blas_chunks[blas_i];
        if (blas_chunk.blas_geoms.empty()) {
            continue;
        }

        {
            if (!blas_chunk.attr_buffer.is_empty()) {
                device.destroy_buffer(blas_chunk.attr_buffer);
            }
            blas_chunk.attr_buffer = device.create_buffer({
                .size = sizeof(VoxelBrickAttribs) * blas_chunk.attrib_bricks.size(),
                .name = "attr_buffer",
            });
            auto attr_dev_ptr = device.get_device_address(blas_chunk.attr_buffer).value();
            attr_pointers_host_ptr[blas_i] = attr_dev_ptr;
        }

        if (!blas_chunk.geom_buffer.is_empty()) {
            device.destroy_buffer(blas_chunk.geom_buffer);
        }
        blas_chunk.geom_buffer = device.create_buffer({
            .size = sizeof(BlasGeom) * blas_chunk.blas_geoms.size(),
            .name = "geom_buffer",
        });
        auto geom_dev_ptr = device.get_device_address(blas_chunk.geom_buffer).value();
        geom_pointers_host_ptr[blas_i] = geom_dev_ptr;
        auto geometry = std::array{
            daxa::BlasAabbGeometryInfo{
                .data = geom_dev_ptr,
                .stride = sizeof(BlasGeom),
                .count = static_cast<uint32_t>(blas_chunk.blas_geoms.size()),
                .flags = daxa::GeometryFlagBits::OPAQUE,
            },
        };
        blas_chunk.blas_build_info = daxa::BlasBuildInfo{
            .flags = daxa::AccelerationStructureBuildFlagBits::PREFER_FAST_TRACE,
            .dst_blas = {}, // Ignored in get_acceleration_structure_build_sizes.
            .geometries = geometry,
            .scratch_data = {}, // Ignored in get_acceleration_structure_build_sizes.
        };
        auto build_size_info = device.get_blas_build_sizes(blas_chunk.blas_build_info);
        auto scratch_alignment_size = get_aligned(build_size_info.build_scratch_size, acceleration_structure_scratch_offset_alignment);
        auto blas_scratch_buffer = device.create_buffer({
            .size = scratch_alignment_size,
            .name = "blas_scratch_buffer",
        });
        defer { device.destroy_buffer(blas_scratch_buffer); };
        blas_chunk.blas_build_info.scratch_data = device.get_device_address(blas_scratch_buffer).value();
        auto build_aligment_size = get_aligned(build_size_info.acceleration_structure_size, ACCELERATION_STRUCTURE_BUILD_OFFSET_ALIGMENT);
        if (!blas_chunk.blas_buffer.is_empty()) {
            device.destroy_buffer(blas_chunk.blas_buffer);
        }
        blas_chunk.blas_buffer = device.create_buffer({
            .size = build_aligment_size,
            .name = "blas_buffer",
        });
        blas_chunk.blas = device.create_blas_from_buffer({
            .blas_info = {
                .size = build_size_info.acceleration_structure_size,
                .name = "blas",
            },
            .buffer_id = blas_chunk.blas_buffer,
            .offset = 0,
        });
        blas_chunk.blas_build_info.dst_blas = blas_chunk.blas;
    }
    // time_points.push_back({Clock::now(), "create new blas buffers"});

    tracked_blases.clear();
    for (uint64_t blas_i = 0; blas_i < blas_chunks.size(); ++blas_i) {
        auto &blas_chunk = blas_chunks[blas_i];
        if (blas_chunk.blas_geoms.empty()) {
            continue;
        }
        tracked_blases.push_back(blas_chunk.blas);
    }
    task_chunk_blases.set_blas({.blas = tracked_blases});
    // time_points.push_back({Clock::now(), "track blases"});

    auto blas_transforms_host_ptr = device.get_host_address_as<VoxelBlasTransform>(staging_blas_transforms.resource_id).value();

    blas_instance_info = std::array{
        daxa::TlasInstanceInfo{
            .data = {}, // Ignored in get_acceleration_structure_build_sizes.
            .count = static_cast<uint32_t>(blas_chunks.size()),
            .is_data_array_of_pointers = false, // Buffer contains flat array of instances, not an array of pointers to instances.
            .flags = daxa::GeometryFlagBits::OPAQUE,
        },
    };

    for (auto packed_dirty_chunk_index : dirty_chunks) {
        auto blas_i = uint32_t(packed_dirty_chunk_index >> 32);
        auto &blas_chunk = blas_chunks[blas_i];
        blas_transforms_host_ptr[blas_i].pos = blas_chunk.position;
        blas_transforms_host_ptr[blas_i].vel = blas_chunk.prev_position - blas_chunk.position;
        blas_instances[blas_i] = daxa_BlasInstanceData{
            .transform = {
                {1, 0, 0, blas_chunk.position.x},
                {0, 1, 0, blas_chunk.position.y},
                {0, 0, 1, blas_chunk.position.z},
            },
            .instance_custom_index = blas_i,
            .mask = blas_chunk.blas_geoms.empty() ? 0x00u : blas_chunk.cull_mask,
            .instance_shader_binding_table_record_offset = 0,
            .flags = {},
            .blas_device_address = device.get_device_address(blas_chunk.blas).value(),
        };
    }
    // time_points.push_back({Clock::now(), "update blas instances"});

    blas_instance_info[0].data = device.get_device_address(blas_instances_buffer.resource_id).value();
    tlas_build_info = daxa::TlasBuildInfo{
        .flags = daxa::AccelerationStructureBuildFlagBits::PREFER_FAST_TRACE,
        .dst_tlas = {}, // Ignored in get_acceleration_structure_build_sizes.
        .instances = blas_instance_info,
        .scratch_data = {}, // Ignored in get_acceleration_structure_build_sizes.
    };
    tlas_build_sizes = device.get_tlas_build_sizes(tlas_build_info);
    if (!buffers.tlas.is_empty()) {
        device.destroy_tlas(buffers.tlas);
        device.destroy_buffer(buffers.tlas_buffer);
    }
    buffers.tlas_buffer = device.create_buffer({
        .size = tlas_build_sizes.acceleration_structure_size,
        .name = "tlas build scratch buffer",
    });
    buffers.tlas = device.create_tlas_from_buffer({
        .tlas_info = {
            .size = tlas_build_sizes.acceleration_structure_size,
            .name = "tlas",
        },
        .buffer_id = buffers.tlas_buffer,
        .offset = 0,
    });
    buffers.task_tlas.set_tlas({.tlas = std::array{buffers.tlas}});

    // time_points.push_back({Clock::now(), "create tlas"});
    // auto t1 = Clock::now();
    // debug_utils::DebugDisplay::set_debug_string("Update Chunks (total)", fmt::format("{} ms", std::chrono::duration<float, std::milli>(t1 - t0).count()));
    // auto prev_time_point = t0;
    // for (auto const &[time_point, name] : time_points) {
    //     debug_utils::DebugDisplay::set_debug_string(fmt::format("Update Chunks ({})", name), fmt::format("{} ms", std::chrono::duration<float, std::milli>(time_point - prev_time_point).count()));
    //     prev_time_point = time_point;
    // }
}

void VoxelWorld::record_frame(GpuContext &gpu_context, VoxelParticles &particles) {
    gpu_context.frame_task_graph.use_persistent_blas(task_chunk_blases);
    gpu_context.frame_task_graph.use_persistent_buffer(staging_blas_geom_pointers.task_resource);
    gpu_context.frame_task_graph.use_persistent_buffer(staging_blas_attr_pointers.task_resource);
    gpu_context.frame_task_graph.use_persistent_buffer(staging_blas_transforms.task_resource);
    gpu_context.frame_task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, staging_blas_geom_pointers.task_resource),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.blas_geom_pointers.task_resource),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, staging_blas_attr_pointers.task_resource),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.blas_attr_pointers.task_resource),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, staging_blas_transforms.task_resource),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.blas_transforms.task_resource),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = ti.get(daxa::TaskBufferAttachmentIndex{0}).ids[0],
                .dst_buffer = ti.get(daxa::TaskBufferAttachmentIndex{1}).ids[0],
                .size = sizeof(daxa::DeviceAddress) * blas_chunks.size(),
            });
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = ti.get(daxa::TaskBufferAttachmentIndex{2}).ids[0],
                .dst_buffer = ti.get(daxa::TaskBufferAttachmentIndex{3}).ids[0],
                .size = sizeof(daxa::DeviceAddress) * blas_chunks.size(),
            });
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = ti.get(daxa::TaskBufferAttachmentIndex{4}).ids[0],
                .dst_buffer = ti.get(daxa::TaskBufferAttachmentIndex{5}).ids[0],
                .size = sizeof(daxa_f32vec3) * blas_chunks.size() * 2,
            });

            // NOTE(grundlett): Hacky way of not needing to sync on these buffers...

            for (auto packed_dirty_chunk_index : dirty_chunks) {
                auto blas_i = uint32_t(packed_dirty_chunk_index >> 32);
                auto &blas_chunk = blas_chunks[blas_i];
                if (blas_chunk.blas_geoms.empty()) {
                    continue;
                }

                auto staging_buffer = ti.device.create_buffer({
                    .size = sizeof(BlasGeom) * blas_chunk.blas_geoms.size() + sizeof(VoxelBrickAttribs) * blas_chunk.attrib_bricks.size(),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "staging_buffer",
                });
                ti.recorder.destroy_buffer_deferred(staging_buffer);
                auto staging_ptr = ti.device.get_host_address_as<uint8_t>(staging_buffer).value();
                auto geom_host_ptr = reinterpret_cast<BlasGeom *>(staging_ptr + 0);
                auto attr_host_ptr = reinterpret_cast<VoxelBrickAttribs *>(staging_ptr + sizeof(BlasGeom) * blas_chunk.blas_geoms.size());
                std::copy(blas_chunk.blas_geoms.begin(), blas_chunk.blas_geoms.end(), geom_host_ptr);
                std::copy(blas_chunk.attrib_bricks.begin(), blas_chunk.attrib_bricks.end(), attr_host_ptr);
                ti.recorder.copy_buffer_to_buffer({
                    .src_buffer = staging_buffer,
                    .dst_buffer = blas_chunk.geom_buffer,
                    .size = sizeof(BlasGeom) * blas_chunk.blas_geoms.size(),
                });
                ti.recorder.copy_buffer_to_buffer({
                    .src_buffer = staging_buffer,
                    .dst_buffer = blas_chunk.attr_buffer,
                    .src_offset = sizeof(BlasGeom) * blas_chunk.blas_geoms.size(),
                    .size = sizeof(VoxelBrickAttribs) * blas_chunk.attrib_bricks.size(),
                });
            }
        },
        .name = "copy pointers",
    });

    gpu_context.frame_task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, buffers.blas_transforms.task_resource),
            daxa::inl_attachment(daxa::TaskBlasAccess::BUILD_WRITE, task_chunk_blases),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            for (auto packed_dirty_chunk_index : dirty_chunks) {
                auto blas_i = uint32_t(packed_dirty_chunk_index >> 32);
                auto &blas_chunk = blas_chunks[blas_i];

                if (!blas_chunk.blas_geoms.empty()) {
                    auto geom_dev_ptr = ti.device.get_device_address(blas_chunk.geom_buffer).value();
                    auto geometry = std::array{
                        daxa::BlasAabbGeometryInfo{
                            .data = geom_dev_ptr,
                            .stride = sizeof(BlasGeom),
                            .count = static_cast<uint32_t>(blas_chunk.blas_geoms.size()),
                            .flags = daxa::GeometryFlagBits::OPAQUE,
                        },
                    };
                    blas_chunk.blas_build_info.geometries = geometry;
                    ti.recorder.build_acceleration_structures({
                        .blas_build_infos = std::array{blas_chunk.blas_build_info},
                    });
                }
            }
        },
        .name = "blas build",
    });

    gpu_context.frame_task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBlasAccess::BUILD_READ, task_chunk_blases),
            daxa::inl_attachment(daxa::TaskTlasAccess::BUILD_WRITE, buffers.task_tlas),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            auto tlas_scratch_buffer = ti.device.create_buffer({
                .size = tlas_build_sizes.build_scratch_size,
                .name = "tlas build scratch buffer",
            });
            ti.recorder.destroy_buffer_deferred(tlas_scratch_buffer);

            tlas_build_info.dst_tlas = buffers.tlas;
            tlas_build_info.scratch_data = ti.device.get_device_address(tlas_scratch_buffer).value();

            ti.recorder.build_acceleration_structures({
                .tlas_build_infos = std::array{tlas_build_info},
            });
        },
        .name = "tlas build",
    });

    gpu_context.add(ComputeTask<VoxelWorldPerframeCompute::Task, VoxelWorldPerframeComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/perframe.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{VoxelWorldPerframeCompute::AT.gpu_input, gpu_context.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{VoxelWorldPerframeCompute::AT.chunk_updates, buffers.chunk_updates.task_resource}},
            daxa::TaskViewVariant{std::pair{VoxelWorldPerframeCompute::AT.geometry_pointers, buffers.blas_geom_pointers.task_resource}},
            daxa::TaskViewVariant{std::pair{VoxelWorldPerframeCompute::AT.attribute_pointers, buffers.blas_attr_pointers.task_resource}},
            daxa::TaskViewVariant{std::pair{VoxelWorldPerframeCompute::AT.blas_transforms, buffers.blas_transforms.task_resource}},
            daxa::TaskViewVariant{std::pair{VoxelWorldPerframeCompute::AT.tlas, buffers.task_tlas}},
            VOXELS_BUFFER_USES_ASSIGN(VoxelWorldPerframeCompute, buffers),
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, VoxelWorldPerframeComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({1, 1, 1});
        },
    });

    gpu_context.add(ComputeTask<PerChunkCompute::Task, PerChunkComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{PerChunkCompute::AT.gpu_input, gpu_context.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{PerChunkCompute::AT.voxel_globals, buffers.voxel_globals.task_resource}},
            daxa::TaskViewVariant{std::pair{PerChunkCompute::AT.voxel_chunks, buffers.voxel_chunks.task_resource}},
            daxa::TaskViewVariant{std::pair{PerChunkCompute::AT.value_noise_texture, gpu_context.task_value_noise_image_view}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, PerChunkComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({CHUNK_NX / 8, CHUNK_NY / 8, CHUNK_NZ / 8});
        },
    });

    auto task_temp_voxel_chunks_buffer = gpu_context.frame_task_graph.create_transient_buffer({
        .size = sizeof(TempVoxelChunk) * MAX_CHUNK_UPDATES_PER_FRAME,
        .name = "temp_voxel_chunks_buffer",
    });

    gpu_context.add(ComputeTask<ChunkEditCompute::Task, ChunkEditComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::AT.gpu_input, gpu_context.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::AT.voxel_globals, buffers.voxel_globals.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::AT.voxel_chunks, buffers.voxel_chunks.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::AT.temp_voxel_chunks, task_temp_voxel_chunks_buffer}},
            SIMPLE_STATIC_ALLOCATOR_BUFFER_USES_ASSIGN(ChunkEditCompute, GrassStrandAllocator, particles.grass.grass_allocator),
            SIMPLE_STATIC_ALLOCATOR_BUFFER_USES_ASSIGN(ChunkEditCompute, FlowerAllocator, particles.flowers.flower_allocator),
            SIMPLE_STATIC_ALLOCATOR_BUFFER_USES_ASSIGN(ChunkEditCompute, TreeParticleAllocator, particles.tree_particles.tree_particle_allocator),
            SIMPLE_STATIC_ALLOCATOR_BUFFER_USES_ASSIGN(ChunkEditCompute, FireParticleAllocator, particles.fire_particles.fire_particle_allocator),
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::AT.value_noise_texture, gpu_context.task_value_noise_image_view}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::AT.test_texture, gpu_context.task_test_texture}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::AT.test_texture2, gpu_context.task_test_texture2}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ChunkEditComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(ChunkEditCompute::AT.voxel_globals).ids[0],
                .offset = offsetof(VoxelWorldGlobals, indirect_dispatch) + offsetof(VoxelWorldGpuIndirectDispatch, chunk_edit_dispatch),
            });
        },
    });

    gpu_context.add(ComputeTask<ChunkEditPostProcessCompute::Task, ChunkEditPostProcessComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::AT.gpu_input, gpu_context.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::AT.voxel_globals, buffers.voxel_globals.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::AT.voxel_chunks, buffers.voxel_chunks.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::AT.temp_voxel_chunks, task_temp_voxel_chunks_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::AT.value_noise_texture, gpu_context.task_value_noise_image_view}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ChunkEditPostProcessComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(ChunkEditPostProcessCompute::AT.voxel_globals).ids[0],
                .offset = offsetof(VoxelWorldGlobals, indirect_dispatch) + offsetof(VoxelWorldGpuIndirectDispatch, chunk_edit_dispatch),
            });
        },
    });

    gpu_context.add(ComputeTask<ChunkAllocCompute::Task, ChunkAllocComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{ChunkAllocCompute::AT.gpu_input, gpu_context.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkAllocCompute::AT.voxel_globals, buffers.voxel_globals.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkAllocCompute::AT.temp_voxel_chunks, task_temp_voxel_chunks_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkAllocCompute::AT.voxel_chunks, buffers.voxel_chunks.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkAllocCompute::AT.chunk_updates, buffers.chunk_updates.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkAllocCompute::AT.chunk_update_heap, buffers.chunk_update_heap.task_resource}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ChunkAllocComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(ChunkAllocCompute::AT.voxel_globals).ids[0],
                // NOTE: This should always have the same value as the chunk edit dispatch, so we're re-using it here
                .offset = offsetof(VoxelWorldGlobals, indirect_dispatch) + offsetof(VoxelWorldGpuIndirectDispatch, chunk_edit_dispatch),
            });
        },
    });
}
