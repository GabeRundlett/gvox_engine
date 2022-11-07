cmake_minimum_required(VERSION 3.21)

project(gabe_voxel_game VERSION 0.1.11)

find_package(fmt CONFIG REQUIRED)
find_package(daxa CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(gvox CONFIG REQUIRED)
find_path(STB_INCLUDE_DIRS "stb_c_lexer.h")
find_package(nlohmann_json CONFIG REQUIRED)
find_package(platform_folders CONFIG REQUIRED)
find_package(libsoundio CONFIG REQUIRED)

add_executable(${PROJECT_NAME} "src/main.cpp" "src/app.cpp")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE
    daxa::daxa
    fmt::fmt
    libsoundio::libsoundio
    nlohmann_json::nlohmann_json
    sago::platform_folders
    glm::glm
    glfw
    gvox::gvox
)
target_include_directories(${PROJECT_NAME} PRIVATE "src")
target_include_directories(${PROJECT_NAME} PRIVATE ${STB_INCLUDE_DIRS})

set(GVOX_FORMATS
    gvox_simple
    gvox_u32
    gvox_u32_palette
    magicavoxel
)
foreach(FORMAT_NAME ${GVOX_FORMATS})
    add_custom_command(
        TARGET gabe_voxel_game
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:gvox::gvox_format_${FORMAT_NAME}>
            $<TARGET_FILE_DIR:gabe_voxel_game>
    )
endforeach()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_link_options(${PROJECT_NAME} PRIVATE "-Wl,/ENTRY:mainCRTStartup,/SUBSYSTEM:WINDOWS")
    else()
        target_link_options(${PROJECT_NAME} PRIVATE /ENTRY:mainCRTStartup /SUBSYSTEM:WINDOWS)
    endif()
    target_link_libraries(${PROJECT_NAME} PRIVATE Dwmapi)
endif()

set(DAXA_INCLUDE_DIR "$<TARGET_FILE_DIR:gabe_voxel_game>/../vcpkg_installed/x64-$<LOWER_CASE:$<PLATFORM_ID>>/include")
target_compile_definitions(${PROJECT_NAME} PRIVATE DAXA_SHADER_INCLUDE_DIR="${DAXA_INCLUDE_DIR}")

set(PACKAGE_VOXEL_GAME true)

if(PACKAGE_VOXEL_GAME)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        list(APPEND RUNTIME_ARTIFACT_TARGETS fmt::fmt glfw)
    endif()

    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
    install(IMPORTED_RUNTIME_ARTIFACTS ${RUNTIME_ARTIFACT_TARGETS})
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/assets" DESTINATION bin)
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/shaders" DESTINATION bin)
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/src/shared" DESTINATION bin/shaders)
    install(FILES "${DAXA_INCLUDE_DIR}/daxa/daxa.inl" "${DAXA_INCLUDE_DIR}/daxa/daxa.glsl" "${DAXA_INCLUDE_DIR}/daxa/daxa.hlsl" DESTINATION bin/shaders/daxa)
    install(FILES "${CMAKE_SOURCE_DIR}/appicon.png" DESTINATION bin)
    install(FILES "${CMAKE_SOURCE_DIR}/imgui.ini" DESTINATION bin)
    foreach(FORMAT_NAME ${GVOX_FORMATS})
        install(FILES $<TARGET_FILE:gvox::gvox_format_${FORMAT_NAME}> DESTINATION bin)
    endforeach()

    set(CPACK_PACKAGE_NAME "GabeVoxelGame")
    set(CPACK_PACKAGE_VENDOR "Gabe-Rundlett")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Gabe Voxel Game is a simple app developed my Gabe Rundlett")
    set(CPACK_PACKAGE_DESCRIPTION "Gabe Voxel Game is a simple app developed my Gabe Rundlett. It is in very early development, and is being documented on Gabe's YouTube channel 'Gabe Rundlett'.")
    set(CPACK_RESOURCE_FILE_WELCOME "${CMAKE_SOURCE_DIR}/packaging/infos/welcome.txt")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/packaging/infos/license.txt")
    set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/packaging/infos/readme.txt")
    set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/appicon.png")

    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        configure_file("packaging/main.rc.in" "${CMAKE_BINARY_DIR}/main.rc")
        target_sources(${PROJECT_NAME} PRIVATE "${CMAKE_BINARY_DIR}/main.rc")

        set(CPACK_GENERATOR WIX)
        set(CPACK_WIX_UPGRADE_GUID 186207C7-9FC3-4F45-9FB1-6C515E0A93CC)
        set(CPACK_PACKAGE_EXECUTABLES ${PROJECT_NAME} "Gabe Voxel Game")
        set(CPACK_WIX_PRODUCT_ICON "${CMAKE_SOURCE_DIR}/appicon.png")

        # Set the default installation directory. In this case it becomes C:/Program Files/GabeVoxelGame
        set(CPACK_PACKAGE_INSTALL_DIRECTORY "GabeVoxelGame")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # TODO: Find a better way to package, though tar.gz works for now
        # install(FILES "${CMAKE_SOURCE_DIR}/packaging/gabe_voxel_game.desktop" DESTINATION share/applications)
        # set(CPACK_BINARY_AppImage ON)
    endif()

    include(InstallRequiredSystemLibraries)
    include(CPack)
endif()
