cmake_minimum_required(VERSION 3.21)

include("${CMAKE_CURRENT_LIST_DIR}/cmake/deps.cmake")

if (CMAKE_GENERATOR MATCHES "Visual Studio")
    # Visual Studio doesn't support building ISPC...
    set(USE_ISPC 0)
else()
    set(USE_ISPC 1)
endif()
if (USE_ISPC)
    include("${CMAKE_CURRENT_LIST_DIR}/cmake/ispc.cmake")
endif()

project(gvox_engine VERSION 0.1.15)
add_executable(${PROJECT_NAME}
    "src/main.cpp"
    "src/voxel_app.cpp"

    "src/scene.cpp"
    "src/voxels/voxel_object.cpp"
    "src/voxels/voxel_world.cpp"

    "src/application/ui.cpp"
    "src/application/audio.cpp"
    "src/application/settings.cpp"
    "src/application/player.cpp"
    "src/utilities/math.cpp"
    "src/utilities/debug.cpp"
    "src/utilities/gpu_context.cpp"
    "src/utilities/thread_pool.cpp"
    "src/renderer/renderer.cpp"
    "src/renderer/fsr.cpp"
    "src/renderer/kajiya/ircache.cpp"
    "src/renderer/render_voxel_object.cpp"
    "src/renderer/render_scene.cpp"
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

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
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}> $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMAND_EXPAND_LISTS
)


find_package(imgui CONFIG REQUIRED)

set(DAXA_USE_VCPKG false)
set(DAXA_ENABLE_UTILS_IMGUI true)
set(DAXA_ENABLE_UTILS_MEM true)
set(DAXA_ENABLE_UTILS_PIPELINE_MANAGER_GLSLANG true)
set(DAXA_ENABLE_UTILS_PIPELINE_MANAGER_SLANG false)
set(DAXA_ENABLE_UTILS_TASK_GRAPH true)
set(DAXA_ENABLE_UTILS_FSR3 false)
set(DAXA_ENABLE_TESTS false)
set(DAXA_ENABLE_TOOLS false)
add_subdirectory(${PROJECT_SOURCE_DIR}/deps/Daxa)

find_package(gvox CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(unofficial-minizip CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
# find_package(soloud CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(fsr2 CONFIG REQUIRED)
# find_package(VulkanMemoryAllocator CONFIG REQUIRED)

add_subdirectory("deps/blue-noise-sampler")

target_link_libraries(${PROJECT_NAME} PRIVATE
    GPUOpen::VulkanMemoryAllocator
    daxa::daxa
    gvox::gvox
    fmt::fmt
    glfw
    nlohmann_json::nlohmann_json
    unofficial::minizip::minizip
    glm::glm
    # soloud
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
