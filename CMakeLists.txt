cmake_minimum_required(VERSION 3.21)

project(voxel_game)

add_executable(${PROJECT_NAME} "src/main.cpp")
find_package(fmt CONFIG REQUIRED)
find_package(daxa CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC daxa::daxa fmt::fmt)
target_include_directories(${PROJECT_NAME} PUBLIC "src")
