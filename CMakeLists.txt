cmake_minimum_required(VERSION 3.12)

project(voxel_game)

add_executable(${PROJECT_NAME} WIN32 "src/main.cpp")
find_package(fmt CONFIG REQUIRED)
find_package(daxa CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC daxa::daxa fmt::fmt)
target_include_directories(${PROJECT_NAME} PUBLIC "src")

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_options(${PROJECT_NAME} PRIVATE "-Wl,/entry:mainCRTStartup")
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)

set(CPACK_GENERATOR WIX)
set(CPACK_PACKAGE_NAME "voxel_game")
set(CPACK_PACKAGE_VENDOR "gaberundlett.org")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Voxel Game - Installation Test")
set(CPACK_PACKAGE_VERSION_MAJOR 0)
set(CPACK_PACKAGE_VERSION_MINOR 0)
set(CPACK_PACKAGE_VERSION_PATCH 1)
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "VoxelGame")

include(CPack)
