cmake_minimum_required(VERSION 3.21)

project(gvox_engine VERSION 0.1.15)

if (CMAKE_GENERATOR MATCHES "Visual Studio")
    # Visual Studio doesn't support building ISPC...
    set(USE_ISPC 0)
else()
    set(USE_ISPC 1)
endif()
if (USE_ISPC)
    include("${CMAKE_CURRENT_LIST_DIR}/cmake/ispc.cmake")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/deps.cmake")

# add_subdirectory("src/renderer/shader_compiler/src")

add_executable(${PROJECT_NAME}
    "src/main.cpp"
    "src/voxel_app.cpp"

    "src/base/str.cpp"
    "src/base/format.cpp"
    "src/base/file.cpp"
    "src/base/path.cpp"
    "src/base/log.cpp"
    "src/base/profiler.cpp"

    "src/scene.cpp"
    "src/voxels/voxel_object.cpp"
    "src/voxels/voxel_world.cpp"
    "src/voxels/animation_playground/animation_playground.cpp"
    "src/voxels/voxel_allocator.cpp"

    "src/application/ui.cpp"
    "src/application/profiler_ui.cpp"
    "src/application/audio.cpp"
    "src/application/settings.cpp"
    "src/application/player.cpp"
    "src/application/window.cpp"

    "src/utilities/math.cpp"
    "src/utilities/debug.cpp"
    "src/utilities/thread_pool.cpp"

    "src/renderer/gpu_context.cpp"
    "src/renderer/renderer.cpp"
    "src/renderer/pipeline_manager.cpp"
    "src/renderer/fsr.cpp"
    "src/renderer/kajiya/ircache.cpp"
    "src/renderer/render_voxel_object.cpp"
    "src/renderer/render_scene.cpp"
    "src/renderer/particles/render_foliage.cpp"
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_precompile_headers(${PROJECT_NAME} PRIVATE
    <filesystem>
    <chrono>
    <vector>
    <string>
    <memory>
    <unordered_map>
    <daxa/daxa.hpp>
    <voxels/voxel.inl>
)

add_library(${PROJECT_NAME}_generation
    "src/voxels/generation/generation.cpp"
)
target_compile_features(${PROJECT_NAME}_generation PUBLIC cxx_std_20)

target_link_libraries(${PROJECT_NAME}_generation PRIVATE
    glm::glm
)

target_compile_definitions(${PROJECT_NAME} PRIVATE USE_ISPC=${USE_ISPC})
if (USE_ISPC)
    target_sources(${PROJECT_NAME}_generation PRIVATE
        "src/voxels/generation/generation.ispc"
    )
    enable_ispc(${PROJECT_NAME}_generation)
endif()

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND "$<$<BOOL:$<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>>:${CMAKE_COMMAND};-E;copy;$<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>;$<TARGET_FILE_DIR:${PROJECT_NAME}>>"
    COMMAND_EXPAND_LISTS
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    GPUOpen::VulkanMemoryAllocator
    daxa::daxa
    gvox::gvox
    glfw
    nlohmann_json::nlohmann_json
    unofficial::minizip::minizip
    glm::glm
    stb::stb
    fsr2::ffx_fsr2_api
    fsr2::ffx_fsr2_api_vk
    blue_noise_sampler::blue_noise_sampler
    ${PROJECT_NAME}_generation
)
target_include_directories(${PROJECT_NAME} PRIVATE
    "src"
)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(GVOX_ENGINE_INSTALL)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            target_link_options(${PROJECT_NAME} PRIVATE "-Wl,/ENTRY:mainCRTStartup,/SUBSYSTEM:WINDOWS")
        else()
            target_link_options(${PROJECT_NAME} PRIVATE /ENTRY:mainCRTStartup /SUBSYSTEM:WINDOWS)
        endif()
    endif()
    target_link_libraries(${PROJECT_NAME} PRIVATE Dwmapi)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/package.cmake")
