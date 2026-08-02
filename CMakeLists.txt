cmake_minimum_required(VERSION 3.21)

include("${CMAKE_CURRENT_LIST_DIR}/cmake/deps.cmake")
include(cmake/warnings.cmake)
include(cmake/static_analysis.cmake)

project(gvox_engine VERSION 0.1.15)
add_executable(${PROJECT_NAME}
    "src/main.cpp"
    "src/voxel_app.cpp"
    "src/voxels/model.cpp"
    "src/voxels/voxel_world.cpp"
    "src/application/ui.cpp"
    "src/application/audio.cpp"
    "src/application/settings.cpp"
    "src/application/player.cpp"
    "src/utilities/math.cpp"
    "src/utilities/debug.cpp"
    "src/utilities/gpu_context.cpp"
    "src/utilities/mesh/mesh_model.cpp"
    "src/renderer/renderer.cpp"
    "src/renderer/fsr.cpp"
    "src/renderer/kajiya/ircache.cpp"
    # "src/voxels/worldgen.ispc"
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
# set_project_warnings(${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME} PRIVATE GVOX_ENGINE_INSTALL=${GVOX_ENGINE_INSTALL})
# include("${CMAKE_CURRENT_LIST_DIR}/cmake/ispc.cmake")

find_package(daxa CONFIG REQUIRED)
find_package(gvox CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(platform_folders CONFIG REQUIRED)
find_package(nfd REQUIRED)
find_package(unofficial-minizip CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
# find_package(soloud CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(fsr2 CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

find_package(freeimage CONFIG REQUIRED)
# FreeImage links OpenEXR, which adds /EHsc for its targets, even if we're using Clang
function(FIXUP_TARGET TGT_NAME)
    if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
        get_property(A_OPTIONS TARGET ${TGT_NAME} PROPERTY INTERFACE_COMPILE_OPTIONS)
        string(REPLACE "/EHsc" "-Wl,/EHsc" A_OPTIONS "")
        set_property(TARGET ${TGT_NAME} PROPERTY INTERFACE_COMPILE_OPTIONS "${A_OPTIONS}")
    endif()
endfunction()
FIXUP_TARGET(Imath::Imath)
FIXUP_TARGET(OpenEXR::Iex)
FIXUP_TARGET(OpenEXR::IlmThread)
FIXUP_TARGET(OpenEXR::OpenEXRCore)
FIXUP_TARGET(OpenEXR::OpenEXR)
FIXUP_TARGET(OpenEXR::OpenEXRUtil)

add_subdirectory("deps/blue-noise-sampler")

target_link_libraries(${PROJECT_NAME} PRIVATE
    GPUOpen::VulkanMemoryAllocator
    daxa::daxa
    gvox::gvox
    fmt::fmt
    glfw
    nlohmann_json::nlohmann_json
    sago::platform_folders
    nfd::nfd
    unofficial::minizip::minizip
    assimp::assimp
    freeimage::FreeImage
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

set(PACKAGE_VOXEL_GAME ${GVOX_ENGINE_INSTALL})

include("${CMAKE_CURRENT_LIST_DIR}/cmake/package.cmake")
