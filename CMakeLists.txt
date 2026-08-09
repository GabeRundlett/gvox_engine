cmake_minimum_required(VERSION 3.21)

include("${CMAKE_CURRENT_LIST_DIR}/cmake/deps.cmake")

project(gvox_engine VERSION 0.1.15)
add_executable(${PROJECT_NAME}
    "src/main.cpp"
    "src/voxel_app.cpp"

    "src/scene.cpp"
    "src/voxels/voxel_object.cpp"
    "src/voxels/voxel_world.cpp"
    "src/voxels/generation/generation.cpp"

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
    # "src/voxels/worldgen.ispc"
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
# include("${CMAKE_CURRENT_LIST_DIR}/cmake/ispc.cmake")

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
