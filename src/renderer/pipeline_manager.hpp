#pragma once

#include <daxa/device.hpp>
#include <daxa/pipeline.hpp>

#include <base/str.hpp>
#include <base/vec.hpp>

// Shader/pipeline registration + compilation.
//
// Deliberately does NOT include <daxa/utils/pipeline_manager.hpp>: that header
// requires Daxa to be built with glslang linked in. Here, glslang lives behind
// shader_compiler.dll, loaded at runtime, so it is never part of this
// executable's build. The types below replace daxa::ShaderFile/ShaderSource/
// ShaderDefine/PipelineReloadResult.

// NOTE: these own their strings rather than borrowing char const*. Most call
// sites pass literals, but GpuContext::add() synthesises a "<TaskHeadName>Shader"
// define at runtime, which would otherwise dangle.
struct ShaderDefine {
    Str name = {};
    Str value = {};
};

struct ShaderCompileInfo {
    Str source_path = {};
    const char* entry_point = "main";
    Vec<ShaderDefine> defines = {};
};

// Mirrors EShLanguage from shader_compiler.hpp without pulling it in.
enum ShaderStage {
    SHADER_STAGE_VERTEX = 0,
    SHADER_STAGE_TESS_CONTROL = 1,
    SHADER_STAGE_TESS_EVAL = 2,
    SHADER_STAGE_GEOMETRY = 3,
    SHADER_STAGE_FRAGMENT = 4,
    SHADER_STAGE_COMPUTE = 5,
    SHADER_STAGE_RAY_GEN = 6,
    SHADER_STAGE_INTERSECT = 7,
    SHADER_STAGE_ANY_HIT = 8,
    SHADER_STAGE_CLOSEST_HIT = 9,
    SHADER_STAGE_MISS = 10,
    SHADER_STAGE_CALLABLE = 11,
    SHADER_STAGE_TASK = 12,
    SHADER_STAGE_MESH = 13,
};

struct ComputePipelineCompileInfo {
    daxa::ComputePipeline *out_pipeline = nullptr;
    Str source_path = {};
    const char* entry_point = "main";
    Vec<ShaderDefine> defines = {};
    int required_subgroup_size = -1;
    uint32_t push_constant_size = 0;
    Str name = {};
};

struct RasterPipelineCompileInfo {
    daxa::RasterPipeline *out_pipeline = nullptr;
    ShaderCompileInfo mesh_info = {};
    ShaderCompileInfo vert_info = {};
    ShaderCompileInfo frag_info = {};
    Vec<daxa::RenderAttachment> color_attachments = {};
    daxa::Optional<daxa::DepthTestInfo> depth_test = {};
    daxa::RasterizerInfo raster = {};
    int required_subgroup_size = -1;
    uint32_t push_constant_size = 0;
    Str name = {};
};

struct RayTracingPipelineCompileInfo {
    daxa::RayTracingPipeline *out_pipeline = nullptr;
    Vec<ShaderCompileInfo> ray_gen_infos = {};
    Vec<ShaderCompileInfo> intersection_infos = {};
    Vec<ShaderCompileInfo> any_hit_infos = {};
    Vec<ShaderCompileInfo> callable_infos = {};
    Vec<ShaderCompileInfo> closest_hit_infos = {};
    Vec<ShaderCompileInfo> miss_hit_infos = {};
    Vec<daxa::RayTracingShaderGroupInfo> shader_groups_infos = {};
    uint32_t max_ray_recursion_depth = 1;
    uint32_t push_constant_size = 0;
    Str name = {};
};

// Replaces daxa::PipelineReloadResult.
enum ReloadResult {
    RELOAD_NO_CHANGE,
    RELOAD_SUCCESS,
    RELOAD_ERROR,
};

struct PipelineManager;

auto create_pipeline_manager() -> PipelineManager *;
void destroy_pipeline_manager(PipelineManager *self);

// Registration. Pipelines are compiled/created later, in bulk, by
// compile_all_shaders() + create_all_pipelines(). `out_pipeline` must point at
// storage that stays alive and at a stable address (task-graph closures capture
// it), and gets assigned in place so hot-reload is transparent to them.
void register_pipeline(PipelineManager *self, ComputePipelineCompileInfo const &info);
void register_pipeline(PipelineManager *self, RasterPipelineCompileInfo const &info);
void register_pipeline(PipelineManager *self, RayTracingPipelineCompileInfo const &info);

// Compiles every registered shader in parallel (using the on-disk SPIR-V cache
// where it's still valid), then creates the pipeline objects.
void compile_all_shaders(PipelineManager *self);
void create_all_pipelines(PipelineManager *self, daxa::Device &device);

// Returns true if any shader source changed since the last compile.
auto needs_hot_reload(PipelineManager *self) -> bool;
auto try_hot_reload(PipelineManager *self, daxa::Device &device, bool force) -> ReloadResult;

void clear_pipelines(PipelineManager *self);
