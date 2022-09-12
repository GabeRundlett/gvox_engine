cmake_minimum_required(VERSION 3.21)

project(gabe_voxel_game)

find_package(fmt CONFIG REQUIRED)
find_package(daxa CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

add_executable(${PROJECT_NAME} "src/main.cpp" "src/imgui/imgui_impl_glfw.cpp")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE daxa::daxa fmt::fmt glm::glm glfw)
target_include_directories(${PROJECT_NAME} PRIVATE "src")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_options(${PROJECT_NAME} PRIVATE "-Wl,/ENTRY:mainCRTStartup,/SUBSYSTEM:WINDOWS")
endif()
