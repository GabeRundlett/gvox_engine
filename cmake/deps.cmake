
if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../deps/Daxa/CMakeLists.txt")
    find_package(Git REQUIRED)
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
        COMMAND_ERROR_IS_FATAL ANY)
endif()

include(FetchContent)

set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw
    GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)

# Daxa fetches & builds imgui/implot itself (wired up against the `glfw`
# target above) as long as those targets don't already exist.
set(DAXA_USE_VCPKG false)
set(DAXA_ENABLE_UTILS_IMGUI true)
set(DAXA_ENABLE_UTILS_MEM true)
set(DAXA_ENABLE_UTILS_PIPELINE_MANAGER_GLSLANG false)
set(DAXA_ENABLE_UTILS_PIPELINE_MANAGER_SLANG false)
set(DAXA_ENABLE_UTILS_TASK_GRAPH true)
set(DAXA_ENABLE_UTILS_FSR3 false)
set(DAXA_ENABLE_TESTS false)
set(DAXA_ENABLE_TOOLS false)
add_subdirectory(${PROJECT_SOURCE_DIR}/deps/Daxa)

# Daxa's own imgui recipe only builds the core + glfw backend; we also need
# the misc/cpp std::string helpers (imgui_stdlib.h) used by src/application/ui.cpp.
if(TARGET lib_imgui)
    FetchContent_GetProperties(imgui)
    target_sources(lib_imgui PRIVATE "${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp")
    target_include_directories(lib_imgui PUBLIC "${imgui_SOURCE_DIR}/misc/cpp")
endif()

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm
    GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# stb is header-only and has no CMakeLists.txt of its own.
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb
    GIT_TAG 2c980bb59875b0d32144a71867fbdebb2f77cd20
    SOURCE_SUBDIR "nonexistent-subdir"
)
FetchContent_MakeAvailable(stb)
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE "${stb_SOURCE_DIR}")
add_library(stb::stb ALIAS stb)

set(GVOX_ENABLE_FILE_IO true)
set(GVOX_ENABLE_MULTITHREADED_ADAPTERS true)
set(GVOX_ENABLE_THREADSAFETY true)
set(GVOX_DISABLE_PACKAGING true)
add_subdirectory(deps/gvox)
add_subdirectory(deps/minizip)
add_subdirectory(deps/fsr2)
add_subdirectory("deps/blue-noise-sampler")

find_package(Vulkan REQUIRED)
