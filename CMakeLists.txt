cmake_minimum_required(VERSION 3.12)

project(voxel_game)

add_executable(${PROJECT_NAME} WIN32 "src/main.cpp")
find_package(fmt CONFIG REQUIRED)
find_package(daxa CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC daxa::daxa fmt::fmt)
target_include_directories(${PROJECT_NAME} PUBLIC "src")

set_property(TARGET ${PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY MultiThreaded)
target_link_options(${PROJECT_NAME} PRIVATE "-Wl,/entry:mainCRTStartup")
