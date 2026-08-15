#pragma once

#if !defined(GLSLANG_WRAPPER_INTERNAL)
typedef enum
{
	EShLangVertex,
	EShLangTessControl,
	EShLangTessEvaluation,
	EShLangGeometry,
	EShLangFragment,
	EShLangCompute,
	EShLangRayGen,
	EShLangIntersect,
	EShLangAnyHit,
	EShLangClosestHit,
	EShLangMiss,
	EShLangCallable,
	EShLangTask,
	EShLangMesh,
	EShLangCount,
} EShLanguage;
#endif

struct GlslangWrapperHeaderResult
{
	const char* header_name;
	size_t header_name_length;
	const char* header_code;
	size_t header_code_length;
};

using IncludeCallback = void(void* user_pointer, const char* header_name, const char* includer_name, GlslangWrapperHeaderResult& result);
using ReleaseStringCallback = void(const char* str);

struct GlslangWrapperCompileInfo
{
	EShLanguage stage;
	const char* preamble;
	const char* shader_glsl;
	const char* shader_name;
	const char* entry_point;
	const char* source_entry;
	bool use_debug_info;

	IncludeCallback* include_local_cb;
	IncludeCallback* include_system_cb;
	ReleaseStringCallback* release_string_cb;
	void* user_pointer;

	unsigned int** out_spv_ptr;
	unsigned int* out_spv_size;
	const char** out_error_str;
	unsigned int* out_error_str_size;
};

#if defined(GLSLANG_WRAPPER_INTERNAL)
GLSLANG_WRAPPER_DLL_EXPORT void glslang_wrapper_compile(GlslangWrapperCompileInfo const& info);
GLSLANG_WRAPPER_DLL_EXPORT void glslang_wrapper_release_results(unsigned int* spv_ptr, const char* error_str);
GLSLANG_WRAPPER_DLL_EXPORT void glslang_wrapper_init();
GLSLANG_WRAPPER_DLL_EXPORT void glslang_wrapper_deinit();
#else
using pfn_glslang_wrapper_init = void (*)();
using pfn_glslang_wrapper_deinit = void (*)();
using pfn_glslang_wrapper_compile = void (*)(const GlslangWrapperCompileInfo& info);
using pfn_glslang_wrapper_release_results = void (*)(unsigned int* spv_ptr, const char* error_str);
pfn_glslang_wrapper_init glslang_wrapper_init = nullptr;
pfn_glslang_wrapper_deinit glslang_wrapper_deinit = nullptr;
pfn_glslang_wrapper_compile glslang_wrapper_compile = nullptr;
pfn_glslang_wrapper_release_results glslang_wrapper_release_results = nullptr;
#endif
