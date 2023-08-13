vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL https://github.com/thedmd/imgui-node-editor
    REF 2f99b2d613a400f6579762bd7e7c343a0d844158
    PATCHES
    "update-for-new-imgui.patch"
)

file(WRITE "${SOURCE_PATH}/CMakeLists.txt" [==[
cmake_minimum_required(VERSION 3.15)
project(imgui-node-editor VERSION 0.5.0)
add_library(${PROJECT_NAME} STATIC
    "crude_json.cpp"
    "imgui_canvas.cpp"
    "imgui_node_editor.cpp"
    "imgui_node_editor_api.cpp"
)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

find_package(imgui CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC imgui::imgui)

# Packaging
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
file(WRITE ${CMAKE_BINARY_DIR}/config.cmake.in [=[
@PACKAGE_INIT@
include(${CMAKE_CURRENT_LIST_DIR}/imgui-node-editor-targets.cmake)
check_required_components(imgui-node-editor)
find_package(imgui CONFIG REQUIRED)
]=])

configure_package_config_file(${CMAKE_BINARY_DIR}/config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/imgui-node-editor-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/imgui-node-editor
    NO_SET_AND_CHECK_MACRO)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/imgui-node-editor-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/imgui-node-editor-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/imgui-node-editor-config-version.cmake
    DESTINATION
    ${CMAKE_INSTALL_DATADIR}/imgui-node-editor)
install(TARGETS imgui-node-editor EXPORT imgui-node-editor-targets)
install(EXPORT imgui-node-editor-targets DESTINATION ${CMAKE_INSTALL_DATADIR}/imgui-node-editor NAMESPACE imgui-node-editor::)
install(FILES ${PROJECT_SOURCE_DIR}/imgui_node_editor.h TYPE INCLUDE)
]==])

vcpkg_configure_cmake(
    SOURCE_PATH "${SOURCE_PATH}"
    PREFER_NINJA
)
vcpkg_install_cmake()
vcpkg_fixup_cmake_targets()
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(INSTALL "${SOURCE_PATH}/LICENSE"
    DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
    RENAME copyright
)
