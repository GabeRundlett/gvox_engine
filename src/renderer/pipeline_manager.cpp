#include "pipeline_manager.hpp"

#include <base/file.hpp>
#include <base/format.hpp>
#include <base/hash_map.hpp>
#include <base/log.hpp>
#include <base/path.hpp>
#include <base/profiler.hpp>
#include <utilities/thread_pool.hpp>

#include <atomic>
#include <mutex>
#include <thread>

#include "shader_compiler/shader_compiler.hpp"

static const uint64_t SPV_CACHE_MAGIC = std::bit_cast<uint64_t>(std::array{'g', 'v', 'o', 'x', 's', 'p', 'v', 'c'});
static uint64_t const SPV_CACHE_VERSION = 1;
static bool const SPV_USE_DEBUG_INFO = false;
static bool const SPV_ENABLE_CACHE = true;

// Where shader #includes are resolved from. Mirrors what the old
// daxa::PipelineManagerInfo2::root_paths listed in gpu_context.cpp.
static char const *SHADER_ROOTS[] = {
    "deps/Daxa/include",
    "assets",
    "src",
    "gpu",
    "src/gpu",
    "src/renderer",
};

struct ShaderDependencyInfo {
    HashMap<Str, uint64_t> file_timestamps;
};

struct ShaderCompileInfoAndResult {
    ShaderCompileInfo info;
    int stage;
    Vec<uint32_t> spirv_binary;
    ShaderDependencyInfo dependency_info;
};

struct PipelineManager {
    Vec<ComputePipelineCompileInfo> compute_pipelines;
    Vec<RasterPipelineCompileInfo> raster_pipelines;
    Vec<RayTracingPipelineCompileInfo> ray_tracing_pipelines;
    HashMap<uint64_t, ShaderCompileInfoAndResult> shader_compile_infos;

    HashMap<Str, uint64_t> all_file_dependencies;
    std::mutex hot_reload_mutex;
    std::thread hot_reload_thread;
    std::atomic_bool hot_reload_running = true;
    std::atomic_bool needs_hot_reload = false;

    // Set if any shader failed to compile during the last compile_all_shaders.
    bool had_compile_error = false;
    // Diagnostics for the last compile_all_shaders.
    std::atomic_int last_cache_hits = 0;
    std::atomic_int last_recompiles = 0;
};

// --- shader_compiler.dll loading ------------------------------------------

#ifdef _WIN32
typedef char const *LPCSTR;
typedef struct HINSTANCE__ *HINSTANCE;
typedef HINSTANCE HMODULE;
#if defined(_MINWINDEF_)
#elif defined(_WIN64)
typedef __int64(__stdcall *FARPROC)(void);
#else
typedef int(__stdcall *FARPROC)(void);
#endif
extern "C" __declspec(dllimport) HMODULE __stdcall LoadLibraryA(LPCSTR);
extern "C" __declspec(dllimport) FARPROC __stdcall GetProcAddress(HMODULE, LPCSTR);
extern "C" __declspec(dllimport) int __stdcall FreeLibrary(HMODULE);
#endif

static HMODULE s_shader_compiler_dll = nullptr;

static void try_load_shader_compiler() {
    s_shader_compiler_dll = LoadLibraryA("shader_compiler.dll");
    if (s_shader_compiler_dll == nullptr) {
        s_shader_compiler_dll = LoadLibraryA("src/renderer/shader_compiler/shader_compiler.dll");
        if (s_shader_compiler_dll == nullptr) {
            log_error("could not load shader_compiler.dll; only cached SPIR-V will be usable");
            return;
        }
    }
    glslang_wrapper_init = (pfn_glslang_wrapper_init)GetProcAddress(s_shader_compiler_dll, "glslang_wrapper_init");
    glslang_wrapper_deinit = (pfn_glslang_wrapper_deinit)GetProcAddress(s_shader_compiler_dll, "glslang_wrapper_deinit");
    glslang_wrapper_compile = (pfn_glslang_wrapper_compile)GetProcAddress(s_shader_compiler_dll, "glslang_wrapper_compile");
    glslang_wrapper_release_results = (pfn_glslang_wrapper_release_results)GetProcAddress(s_shader_compiler_dll, "glslang_wrapper_release_results");
    if (glslang_wrapper_init != nullptr) {
        glslang_wrapper_init();
    }
}

static void try_unload_shader_compiler() {
    if (s_shader_compiler_dll == nullptr) {
        return;
    }
    if (glslang_wrapper_deinit != nullptr) {
        glslang_wrapper_deinit();
    }
    FreeLibrary(s_shader_compiler_dll);
    s_shader_compiler_dll = nullptr;
}

// --- hashing ---------------------------------------------------------------

static auto hash_combine(uint64_t h1, uint64_t h2) -> uint64_t {
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
}

static auto hash_shader_info(ShaderCompileInfo const &info, int stage) -> uint64_t {
    auto result = uint64_t{};
    result = hash_combine(result, hash_key(info.source_path));
    result = hash_combine(result, hash_key(Str(info.entry_point)));
    for (auto const &define : info.defines) {
        result = hash_combine(result, hash_key(define.name));
        result = hash_combine(result, hash_key(define.value));
    }
    result = hash_combine(result, SPV_USE_DEBUG_INFO ? 1ull : 0ull);
    result = hash_combine(result, static_cast<uint64_t>(stage));
    return result;
}

// --- path resolution -------------------------------------------------------

// NOTE: MUST BE THREAD SAFE
static auto resolve_file_path(Str &path) -> bool {
    if (path_exists(path.c_str())) {
        path = path_normalize(path.c_str());
        return true;
    }
    for (auto const *root : SHADER_ROOTS) {
        auto potential = Str(root);
        potential += "/";
        potential += path.c_str();
        if (path_exists(potential.c_str())) {
            path = path_normalize(potential.c_str());
            return true;
        }
    }
    return false;
}

// --- shader source loading -------------------------------------------------

// glslang does not implement `#pragma once`, but the project's .inl/.glsl
// headers rely on it (and would otherwise hit "redefinition" errors when a
// header is reached twice). Rewrite it into a classic include guard keyed on
// the mangled absolute path, exactly as Daxa's own manager does
// (impl_pipeline_manager.cpp: shader_preprocess).
static void apply_pragma_once(Str &code, char const *path) {
    auto abs = path_absolute(path);
    // Mangle to a valid macro identifier: keep [A-Za-z0-9_] only.
    auto guard = Str{};
    {
        auto const *p = abs.c_str();
        auto *buf = new char[static_cast<unsigned>(abs.length) + 1];
        auto n = 0;
        for (int i = 0; i < abs.length; ++i) {
            auto c = p[i];
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') {
                buf[n++] = c;
            }
        }
        buf[n] = '\0';
        guard = buf;
        delete[] buf;
    }

    // Replace the first `#pragma ... once` line with `#if !defined(GUARD)`.
    auto const *src = code.c_str();
    auto out = Str{};
    auto has_pragma_once = false;
    auto line_start = 0;
    for (int i = 0; i <= code.length; ++i) {
        if (i != code.length && src[i] != '\n') {
            continue;
        }
        auto line_len = i - line_start;
        auto *line = new char[static_cast<unsigned>(line_len) + 1];
        for (int c = 0; c < line_len; ++c) {
            line[c] = src[line_start + c];
        }
        line[line_len] = '\0';

        auto is_pragma_once = false;
        if (!has_pragma_once) {
            // Same loose check Daxa uses: a "#pragma" with "once" after it.
            char const *pragma_pos = nullptr;
            char const *once_pos = nullptr;
            for (int c = 0; c + 7 <= line_len; ++c) {
                if (pragma_pos == nullptr && line[c] == '#' &&
                    line[c + 1] == 'p' && line[c + 2] == 'r' && line[c + 3] == 'a' &&
                    line[c + 4] == 'g' && line[c + 5] == 'm' && line[c + 6] == 'a') {
                    pragma_pos = line + c;
                }
            }
            for (int c = 0; c + 4 <= line_len; ++c) {
                if (line[c] == 'o' && line[c + 1] == 'n' && line[c + 2] == 'c' && line[c + 3] == 'e') {
                    once_pos = line + c;
                    break;
                }
            }
            is_pragma_once = pragma_pos != nullptr && once_pos != nullptr && once_pos > pragma_pos;
        }

        if (is_pragma_once) {
            out += "#if !defined(";
            out += guard.c_str();
            out += ")\n";
            has_pragma_once = true;
        } else {
            out += line;
            out += "\n";
        }
        delete[] line;
        line_start = i + 1;
    }

    if (has_pragma_once) {
        out += "\n#define ";
        out += guard.c_str();
        out += "\n#endif\n";
    }
    code = static_cast<Str &&>(out);
}

// NOTE: MUST BE THREAD SAFE
static auto load_shader_source(char const *path, Str &out) -> bool {
    if (!read_file_to_string(path, out)) {
        return false;
    }
    apply_pragma_once(out, path);
    return true;
}

// --- SPIR-V cache ----------------------------------------------------------

static auto spv_path(uint64_t hash) -> Str {
    auto p = Str("build/spv/");
    p.append(hash);
    p += ".spv";
    return p;
}
static auto spvc_path(uint64_t hash) -> Str {
    auto p = Str("build/spvc/");
    p.append(hash);
    p += ".spvc";
    return p;
}

static void save_spirv_cache(ShaderCompileInfoAndResult &info_and_result, uint64_t hash) {
    auto &result = info_and_result.spirv_binary;
    auto &deps = info_and_result.dependency_info;

    create_directories("build/spv");
    write_file(spv_path(hash).c_str(), result.data, result.size * static_cast<int>(sizeof(uint32_t)));

    // Dependency sidecar: magic, version, count, then (timestamp, path) pairs.
    create_directories("build/spvc");
    auto blob = Vec<char>{};
    auto write_u64 = [&blob](uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            blob.push_back(static_cast<char>((v >> (i * 8)) & 0xff));
        }
    };
    write_u64(SPV_CACHE_MAGIC);
    write_u64(SPV_CACHE_VERSION);
    write_u64(static_cast<uint64_t>(deps.file_timestamps.count));
    for (auto &slot : deps.file_timestamps) {
        write_u64(slot.value);
        write_u64(static_cast<uint64_t>(slot.key.length));
        for (int i = 0; i < slot.key.length; ++i) {
            blob.push_back(slot.key.c_str()[i]);
        }
    }
    write_file(spvc_path(hash).c_str(), blob.data, blob.size);
}

static auto load_spirv_cache(ShaderCompileInfoAndResult &info_and_result, uint64_t hash) -> bool {
    PROFILE_FUNC();
    auto &info = info_and_result.info;
    auto &result = info_and_result.spirv_binary;
    auto &result_deps = info_and_result.dependency_info;

    auto load_spv_into_result = [&result, hash]() {
        return read_file_to_u32s(spv_path(hash).c_str(), result) && result.size != 0;
    };

    auto resolved_path = Str(info.source_path);
    // If the source is missing entirely, fall back to whatever SPIR-V we have.
    if (!resolve_file_path(resolved_path)) {
        return load_spv_into_result();
    }
    if (!SPV_ENABLE_CACHE) {
        return false;
    }

    auto has_live_cache = result_deps.file_timestamps.count != 0;

    if (!has_live_cache) {
        auto blob = Vec<char>{};
        if (!read_file_to_bytes(spvc_path(hash).c_str(), blob)) {
            return false;
        }
        auto const *p = blob.data;
        auto remaining = blob.size;
        auto read_u64 = [&p, &remaining](uint64_t &out) {
            if (remaining < 8) {
                return false;
            }
            out = 0;
            for (int i = 0; i < 8; ++i) {
                out |= static_cast<uint64_t>(static_cast<unsigned char>(p[i])) << (i * 8);
            }
            p += 8;
            remaining -= 8;
            return true;
        };

        uint64_t magic = 0;
        uint64_t version = 0;
        uint64_t dep_n = 0;
        if (!read_u64(magic) || magic != SPV_CACHE_MAGIC) {
            return false;
        }
        if (!read_u64(version) || version != SPV_CACHE_VERSION) {
            return false;
        }
        if (!read_u64(dep_n)) {
            return false;
        }
        for (uint64_t i = 0; i < dep_n; ++i) {
            uint64_t timestamp = 0;
            uint64_t len = 0;
            if (!read_u64(timestamp) || !read_u64(len)) {
                return false;
            }
            if (remaining < static_cast<int>(len)) {
                return false;
            }
            auto *buf = new char[len + 1];
            for (uint64_t c = 0; c < len; ++c) {
                buf[c] = p[c];
            }
            buf[len] = '\0';
            result_deps.file_timestamps.set(Str(buf), timestamp);
            delete[] buf;
            p += len;
            remaining -= static_cast<int>(len);
        }
    }

    // Any dependency newer than what we recorded invalidates the cache.
    for (auto &slot : result_deps.file_timestamps) {
        if (path_modified_time(slot.key.c_str()) != slot.value) {
            return false;
        }
    }

    if (!has_live_cache) {
        return load_spv_into_result();
    }
    return result.size != 0;
}

// --- compilation -----------------------------------------------------------

// Daxa's GLSL headers number stages differently from glslang's EShLanguage.
// (see deps/Daxa/include/daxa/daxa.glsl and impl_pipeline_manager.cpp)
static auto daxa_shader_stage_value(int stage) -> int {
    switch (stage) {
    case SHADER_STAGE_COMPUTE: return 0;
    case SHADER_STAGE_VERTEX: return 1;
    case SHADER_STAGE_TESS_CONTROL: return 2;
    case SHADER_STAGE_TESS_EVAL: return 3;
    case SHADER_STAGE_FRAGMENT: return 4;
    case SHADER_STAGE_TASK: return 5;
    case SHADER_STAGE_MESH: return 6;
    case SHADER_STAGE_RAY_GEN: return 7;
    case SHADER_STAGE_ANY_HIT: return 8;
    case SHADER_STAGE_CLOSEST_HIT: return 9;
    case SHADER_STAGE_MISS: return 10;
    case SHADER_STAGE_INTERSECT: return 11;
    case SHADER_STAGE_CALLABLE: return 12;
    default: return 0;
    }
}

struct IncludeUserInfo {
    ShaderDependencyInfo *dependency_info;
};

// NOTE: MUST BE THREAD SAFE
static void process_include(ShaderDependencyInfo *deps, Str const &resolved_path, GlslangWrapperHeaderResult &result) {
    auto code = Str{};
    if (!load_shader_source(resolved_path.c_str(), code)) {
        return;
    }
    deps->file_timestamps.set(resolved_path, path_modified_time(resolved_path.c_str()));

    // glslang takes ownership of these until it calls release_string_cb.
    auto *res_content = new char[static_cast<unsigned>(code.length) + 1];
    for (int i = 0; i <= code.length; ++i) {
        res_content[i] = code.c_str()[i];
    }
    auto *res_name = new char[static_cast<unsigned>(resolved_path.length) + 1];
    for (int i = 0; i <= resolved_path.length; ++i) {
        res_name[i] = resolved_path.c_str()[i];
    }

    result.header_name = res_name;
    result.header_name_length = static_cast<size_t>(resolved_path.length);
    result.header_code = res_content;
    result.header_code_length = static_cast<size_t>(code.length);
}

static void compile_shader(ShaderCompileInfoAndResult &info_and_result, uint64_t hash, bool &out_had_error, PipelineManager *self) {
    PROFILE_FUNC();
    if (load_spirv_cache(info_and_result, hash)) {
        self->last_cache_hits.fetch_add(1);
        return;
    }
    self->last_recompiles.fetch_add(1);

    if (glslang_wrapper_compile == nullptr || glslang_wrapper_release_results == nullptr) {
        log_error("no valid shader cache for '%s' and no compiler available", info_and_result.info.source_path.c_str());
        out_had_error = true;
        return;
    }

    auto &info = info_and_result.info;
    auto &stage = info_and_result.stage;
    auto &result = info_and_result.spirv_binary;
    auto &deps = info_and_result.dependency_info;

    // Must match what Daxa's own pipeline manager injects: the project's
    // shaders rely on DAXA_SHADER_STAGE to serve several stages from one file,
    // and on the include-directive extension (104 shaders use #include).
    auto preamble = Str{};
    preamble += format("#define DAXA_SHADER_STAGE %d\n", daxa_shader_stage_value(stage)).data;
    preamble += "#extension GL_GOOGLE_include_directive : enable\n";
    preamble += "#extension GL_KHR_memory_scope_semantics : enable\n";
    for (auto const &define : info.defines) {
        if (!define.value.empty()) {
            preamble += format("#define %s %s\n", define.name.c_str(), define.value.c_str()).data;
        } else {
            preamble += format("#define %s\n", define.name.c_str()).data;
        }
    }

    auto resolved_path = Str(info.source_path);
    if (!resolve_file_path(resolved_path)) {
        log_error("failed to find shader file '%s'", info.source_path.c_str());
        out_had_error = true;
        return;
    }

    auto shader_code = Str{};
    if (!load_shader_source(resolved_path.c_str(), shader_code)) {
        log_error("failed to load shader file '%s'", info.source_path.c_str());
        out_had_error = true;
        return;
    }

    deps.file_timestamps.clear();
    deps.file_timestamps.set(resolved_path, path_modified_time(resolved_path.c_str()));

    uint32_t *spv_ptr = nullptr;
    uint32_t spv_size = 0;
    char const *error_str_ptr = nullptr;
    uint32_t error_str_size = 0;

    auto user_info = IncludeUserInfo{&deps};

    auto include_local_cb = +[](void *user_pointer, char const *header_name, char const *includer_name, GlslangWrapperHeaderResult &result) {
        auto *deps = reinterpret_cast<IncludeUserInfo *>(user_pointer)->dependency_info;
        auto includer_path = Str(includer_name);
        if (!resolve_file_path(includer_path)) {
            return;
        }
        auto dir = path_dir_part(includer_path.c_str());
        auto candidate = Str{};
        if (!dir.empty()) {
            candidate += dir.c_str();
            candidate += "/";
        }
        candidate += header_name;
        auto resolved = path_normalize(candidate.c_str());
        // A quoted include can still resolve against the roots (matches how
        // Daxa's manager behaves for e.g. <daxa/...> style paths written with quotes).
        if (!path_exists(resolved.c_str()) && !resolve_file_path(resolved)) {
            return;
        }
        process_include(deps, resolved, result);
    };

    auto include_system_cb = +[](void *user_pointer, char const *header_name, char const * /*includer_name*/, GlslangWrapperHeaderResult &result) {
        auto *deps = reinterpret_cast<IncludeUserInfo *>(user_pointer)->dependency_info;
        auto resolved = Str(header_name);
        if (!resolve_file_path(resolved)) {
            return;
        }
        process_include(deps, resolved, result);
    };

    auto release_string_cb = +[](char const *str) { delete[] str; };

    glslang_wrapper_compile(GlslangWrapperCompileInfo{
        .stage = EShLanguage(stage),
        .preamble = preamble.c_str(),
        .shader_glsl = shader_code.c_str(),
        .shader_name = resolved_path.c_str(),
        .entry_point = info.entry_point == nullptr ? "main" : info.entry_point,
        .source_entry = "main",
        .use_debug_info = SPV_USE_DEBUG_INFO,

        .include_local_cb = include_local_cb,
        .include_system_cb = include_system_cb,
        .release_string_cb = release_string_cb,
        .user_pointer = &user_info,

        .out_spv_ptr = &spv_ptr,
        .out_spv_size = &spv_size,
        .out_error_str = &error_str_ptr,
        .out_error_str_size = &error_str_size,
    });

    if (spv_ptr != nullptr && spv_size != 0) {
        result.reserve(static_cast<int>(spv_size));
        for (uint32_t i = 0; i < spv_size; ++i) {
            result.data[i] = spv_ptr[i];
        }
        result.size = static_cast<int>(spv_size);
        save_spirv_cache(info_and_result, hash);
    } else {
        log_error("%s", error_str_ptr != nullptr ? error_str_ptr : "unknown shader compile error");
        out_had_error = true;
    }

    glslang_wrapper_release_results(spv_ptr, error_str_ptr);
}

// --- registration ----------------------------------------------------------

static void register_shader(PipelineManager *self, ShaderCompileInfo const &shader_info, int stage) {
    auto hash = hash_shader_info(shader_info, stage);
    if (!self->shader_compile_infos.contains(hash)) {
        self->shader_compile_infos.set(hash, ShaderCompileInfoAndResult{.info = shader_info, .stage = stage});
    }
}

void register_pipeline(PipelineManager *self, ComputePipelineCompileInfo const &info) {
    self->compute_pipelines.push_back(info);
    register_shader(self, ShaderCompileInfo{info.source_path, info.entry_point, info.defines}, SHADER_STAGE_COMPUTE);
}

void register_pipeline(PipelineManager *self, RasterPipelineCompileInfo const &info) {
    self->raster_pipelines.push_back(info);
    if (!info.mesh_info.source_path.empty()) {
        register_shader(self, info.mesh_info, SHADER_STAGE_MESH);
    }
    if (!info.vert_info.source_path.empty()) {
        register_shader(self, info.vert_info, SHADER_STAGE_VERTEX);
    }
    register_shader(self, info.frag_info, SHADER_STAGE_FRAGMENT);
}

void register_pipeline(PipelineManager *self, RayTracingPipelineCompileInfo const &info) {
    self->ray_tracing_pipelines.push_back(info);
    for (auto const &i : info.ray_gen_infos) {
        register_shader(self, i, SHADER_STAGE_RAY_GEN);
    }
    for (auto const &i : info.intersection_infos) {
        register_shader(self, i, SHADER_STAGE_INTERSECT);
    }
    for (auto const &i : info.any_hit_infos) {
        register_shader(self, i, SHADER_STAGE_ANY_HIT);
    }
    for (auto const &i : info.callable_infos) {
        register_shader(self, i, SHADER_STAGE_CALLABLE);
    }
    for (auto const &i : info.closest_hit_infos) {
        register_shader(self, i, SHADER_STAGE_CLOSEST_HIT);
    }
    for (auto const &i : info.miss_hit_infos) {
        register_shader(self, i, SHADER_STAGE_MISS);
    }
}

// --- bulk compile / create -------------------------------------------------

struct CompileAllState {
    PipelineManager *self;
    Vec<uint64_t> *hashes;
    std::atomic_bool had_error;
};

void compile_all_shaders(PipelineManager *self) {
    self->all_file_dependencies.clear();
    self->had_compile_error = false;
    self->last_cache_hits.store(0);
    self->last_recompiles.store(0);

    auto hashes = Vec<uint64_t>{};
    hashes.reserve(self->shader_compile_infos.count);
    for (auto &slot : self->shader_compile_infos) {
        hashes.push_back(slot.key);
    }

    auto state = CompileAllState{self, &hashes, {false}};
    thread_pool::parallel_for(
        hashes.size,
        [](void *user_ptr, int i) {
            auto &s = *reinterpret_cast<CompileAllState *>(user_ptr);
            auto *entry = s.self->shader_compile_infos.get((*s.hashes)[i]);
            if (entry != nullptr) {
                auto had_error = false;
                compile_shader(*entry, (*s.hashes)[i], had_error, s.self);
                if (had_error) {
                    s.had_error.store(true);
                }
            }
        },
        &state);
    self->had_compile_error = state.had_error.load();

    for (auto &slot : self->shader_compile_infos) {
        for (auto &dep : slot.value.dependency_info.file_timestamps) {
            self->all_file_dependencies.set(dep.key, dep.value);
        }
    }
}

static auto shader_info_from_hash(PipelineManager *self, uint64_t hash, bool &ok) -> daxa::ShaderInfo {
    auto *entry = self->shader_compile_infos.get(hash);
    if (entry == nullptr || entry->spirv_binary.size == 0) {
        ok = false;
        return {};
    }
    return daxa::ShaderInfo{
        .byte_code = entry->spirv_binary.data,
        .byte_code_size = static_cast<uint32_t>(entry->spirv_binary.size),
        .entry_point = entry->info.entry_point,
    };
}

static auto all_shader_infos(PipelineManager *self, Vec<ShaderCompileInfo> const &infos, int stage, bool &ok) -> Vec<daxa::ShaderInfo> {
    auto result = Vec<daxa::ShaderInfo>{};
    result.reserve(infos.size);
    for (auto const &info : infos) {
        result.push_back(shader_info_from_hash(self, hash_shader_info(info, stage), ok));
    }
    return result;
}

struct CreateAllState {
    PipelineManager *self;
    daxa::Device *device;
};

static void create_compute_pipeline(PipelineManager *self, daxa::Device &device, ComputePipelineCompileInfo &info) {
    PROFILE_FUNC();
    auto ok = true;
    auto hash = hash_shader_info(ShaderCompileInfo{info.source_path, info.entry_point, info.defines}, SHADER_STAGE_COMPUTE);
    auto shader_info = shader_info_from_hash(self, hash, ok);
    if (!ok) {
        log_error("no SPIR-V for compute pipeline '%s'", info.source_path.c_str());
        return;
    }
    if (info.required_subgroup_size > 0) {
        shader_info.required_subgroup_size = static_cast<uint32_t>(info.required_subgroup_size);
    }
    *info.out_pipeline = device.create_compute_pipeline({
        .shader_info = shader_info,
        .push_constant_size = info.push_constant_size,
        .name = info.name.c_str(),
    });
}

static void create_raster_pipeline(PipelineManager *self, daxa::Device &device, RasterPipelineCompileInfo &info) {
    PROFILE_FUNC();
    auto ok = true;
    auto mesh_info = daxa::Optional<daxa::ShaderInfo>{};
    if (!info.mesh_info.source_path.empty()) {
        mesh_info = shader_info_from_hash(self, hash_shader_info(info.mesh_info, SHADER_STAGE_MESH), ok);
        if (info.required_subgroup_size > 0) {
            mesh_info.value().required_subgroup_size = static_cast<uint32_t>(info.required_subgroup_size);
        }
    }
    auto vert_info = daxa::Optional<daxa::ShaderInfo>{};
    if (!info.vert_info.source_path.empty()) {
        vert_info = shader_info_from_hash(self, hash_shader_info(info.vert_info, SHADER_STAGE_VERTEX), ok);
    }
    auto frag_info = shader_info_from_hash(self, hash_shader_info(info.frag_info, SHADER_STAGE_FRAGMENT), ok);
    if (!ok) {
        log_error("no SPIR-V for raster pipeline '%s'", info.frag_info.source_path.c_str());
        return;
    }
    *info.out_pipeline = device.create_raster_pipeline({
        .mesh_shader_info = mesh_info,
        .vertex_shader_info = vert_info,
        .fragment_shader_info = frag_info,
        .color_attachments = {info.color_attachments.data, static_cast<size_t>(info.color_attachments.size)},
        .depth_test = info.depth_test,
        .raster = info.raster,
        .push_constant_size = info.push_constant_size,
        .name = info.name.c_str(),
    });
}

static void create_ray_tracing_pipeline(PipelineManager *self, daxa::Device &device, RayTracingPipelineCompileInfo &info) {
    PROFILE_FUNC();
    auto ok = true;
    auto ray_gen = all_shader_infos(self, info.ray_gen_infos, SHADER_STAGE_RAY_GEN, ok);
    auto intersect = all_shader_infos(self, info.intersection_infos, SHADER_STAGE_INTERSECT, ok);
    auto any_hit = all_shader_infos(self, info.any_hit_infos, SHADER_STAGE_ANY_HIT, ok);
    auto callable = all_shader_infos(self, info.callable_infos, SHADER_STAGE_CALLABLE, ok);
    auto closest_hit = all_shader_infos(self, info.closest_hit_infos, SHADER_STAGE_CLOSEST_HIT, ok);
    auto miss = all_shader_infos(self, info.miss_hit_infos, SHADER_STAGE_MISS, ok);
    if (!ok) {
        log_error("no SPIR-V for ray tracing pipeline '%s'", info.name.c_str());
        return;
    }
    *info.out_pipeline = device.create_ray_tracing_pipeline({
        .ray_gen_shaders = {ray_gen.data, static_cast<size_t>(ray_gen.size)},
        .intersection_shaders = {intersect.data, static_cast<size_t>(intersect.size)},
        .any_hit_shaders = {any_hit.data, static_cast<size_t>(any_hit.size)},
        .callable_shaders = {callable.data, static_cast<size_t>(callable.size)},
        .closest_hit_shaders = {closest_hit.data, static_cast<size_t>(closest_hit.size)},
        .miss_hit_shaders = {miss.data, static_cast<size_t>(miss.size)},
        .shader_groups = {info.shader_groups_infos.data, static_cast<size_t>(info.shader_groups_infos.size)},
        .max_ray_recursion_depth = info.max_ray_recursion_depth,
        .push_constant_size = info.push_constant_size,
        .name = info.name.c_str(),
    });
}

void create_all_pipelines(PipelineManager *self, daxa::Device &device) {
    PROFILE_FUNC();
    // NOTE: pipeline creation is not parallelised (unlike the shader compiles
    // above, which dominate). daxa::Device::create_*_pipeline is internally
    // synchronized, but keeping this serial keeps error reporting readable.
    for (auto &info : self->ray_tracing_pipelines) {
        if (info.out_pipeline != nullptr) {
            create_ray_tracing_pipeline(self, device, info);
        }
    }
    for (auto &info : self->raster_pipelines) {
        if (info.out_pipeline != nullptr) {
            create_raster_pipeline(self, device, info);
        }
    }
    for (auto &info : self->compute_pipelines) {
        if (info.out_pipeline != nullptr) {
            create_compute_pipeline(self, device, info);
        }
    }
}

// --- hot reload ------------------------------------------------------------

static void hot_reload_thread_main(PipelineManager *self) {
    while (true) {
        for (int i = 0; i < 10; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (!self->hot_reload_running.load()) {
                return;
            }
        }

        auto lock = std::lock_guard{self->hot_reload_mutex};
        for (auto &slot : self->all_file_dependencies) {
            if (path_modified_time(slot.key.c_str()) != slot.value) {
                self->needs_hot_reload.store(true);
                break;
            }
        }
    }
}

auto needs_hot_reload(PipelineManager *self) -> bool {
    return self->needs_hot_reload.load();
}

auto try_hot_reload(PipelineManager *self, daxa::Device &device, bool force) -> ReloadResult {
    if (!force && !self->needs_hot_reload.load()) {
        return RELOAD_NO_CHANGE;
    }
    auto lock = std::lock_guard{self->hot_reload_mutex};
    compile_all_shaders(self);
    create_all_pipelines(self, device);
    self->needs_hot_reload.store(false);
    return self->had_compile_error ? RELOAD_ERROR : RELOAD_SUCCESS;
}

// --- lifetime --------------------------------------------------------------

auto create_pipeline_manager() -> PipelineManager * {
    auto *self = new PipelineManager();
    try_load_shader_compiler();
    self->hot_reload_thread = std::thread([self]() { hot_reload_thread_main(self); });
    return self;
}

void destroy_pipeline_manager(PipelineManager *self) {
    self->hot_reload_running.store(false);
    self->hot_reload_thread.join();
    try_unload_shader_compiler();
    delete self;
}

void clear_pipelines(PipelineManager *self) {
    self->compute_pipelines.clear();
    self->raster_pipelines.clear();
    self->ray_tracing_pipelines.clear();
}
