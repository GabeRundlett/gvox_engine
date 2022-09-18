cmake_minimum_required(VERSION 3.21)

project(gabe_voxel_game VERSION 0.1.1)

find_package(fmt CONFIG REQUIRED)
find_package(daxa CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

add_executable(${PROJECT_NAME} WIN32 "src/main.cpp" "src/imgui/imgui_impl_glfw.cpp")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE daxa::daxa fmt::fmt glm::glm glfw)
target_include_directories(${PROJECT_NAME} PRIVATE "src")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_options(${PROJECT_NAME} PRIVATE "-Wl,/ENTRY:mainCRTStartup,/SUBSYSTEM:WINDOWS")
endif()

set(DAXA_INCLUDE_DIR "$<TARGET_FILE_DIR:gabe_voxel_game>/vcpkg_installed/x64-$<LOWER_CASE:$<PLATFORM_ID>>/include")
target_compile_definitions(${PROJECT_NAME} PRIVATE DAXA_SHADER_INCLUDE_DIR="${DAXA_INCLUDE_DIR}")

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(IMPORTED_RUNTIME_ARTIFACTS fmt::fmt glfw)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/shaders" DESTINATION bin)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/src/shared" DESTINATION bin/shaders)
install(FILES "${DAXA_INCLUDE_DIR}/daxa/daxa.inl" "${DAXA_INCLUDE_DIR}/daxa/daxa.glsl" "${DAXA_INCLUDE_DIR}/daxa/daxa.hlsl" DESTINATION bin/shaders/daxa)

if(PACKAGE_VOXEL_GAME)
    # target_sources(${PROJECT_NAME} PRIVATE "src/main.rc")

    set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(CPACK_NSIS_CREATE_ICONS_EXTRA "SetOutPath '$INSTDIR\\\\bin'\nCreateShortCut '$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\gabe_voxel_game.lnk' '$INSTDIR\\\\bin\\\\gabe_voxel_game.exe'")
        set(CPACK_NSIS_DELETE_ICONS_EXTRA "Delete '$SMPROGRAMS\\\\$START_MENU\\\\gabe_voxel_game.lnk'")
    endif()

    include(InstallRequiredSystemLibraries)
    include(CPack)
endif()
