cmake_minimum_required(VERSION 3.21)

project(shader_compiler VERSION 3.0.2)

add_library(${PROJECT_NAME} SHARED)
target_sources(${PROJECT_NAME} PRIVATE "shader_compiler.cpp")
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
find_package(Vulkan REQUIRED)

set(VULKAN_SDK_GLSLANG_LIB "$ENV{VULKAN_SDK}/Lib/shaderc_combined.lib")

if(EXISTS ${VULKAN_SDK_GLSLANG_LIB})
    target_compile_definitions(${PROJECT_NAME} PRIVATE _ITERATOR_DEBUG_LEVEL=0)
    set_property(TARGET ${PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
        Vulkan::Vulkan # for glslang include directory
        ${VULKAN_SDK_GLSLANG_LIB}
    )
else()
    if (NOT TARGET glslang::glslang)
        include(FetchContent)
        option(ENABLE_OPT "" OFF)
        set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
        set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
        set(ENABLE_SPVREMAPPER OFF CACHE BOOL "" FORCE)
        set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            glslang
            GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
            GIT_TAG        5398d55e33dff7d26fecdd2c35808add986c558c
        )
        set(CMAKE_UNITY_BUILD ON)
        FetchContent_MakeAvailable(glslang)
        set(CMAKE_UNITY_BUILD OFF)
    endif()

    target_link_libraries(${PROJECT_NAME}
        PRIVATE
        glslang::glslang
        glslang::SPIRV
        glslang::glslang-default-resource-limits
    )
endif()

# add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:shader_compiler>" "${CMAKE_CURRENT_LIST_DIR}/../"
# )
