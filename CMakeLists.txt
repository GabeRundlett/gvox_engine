cmake_minimum_required(VERSION 3.21)

include("${CMAKE_CURRENT_LIST_DIR}/cmake/deps.cmake")
include(cmake/warnings.cmake)
include(cmake/static_analysis.cmake)

project(gvox_engine VERSION 0.1.14)
add_executable(${PROJECT_NAME}
    "src/cpu/main.cpp"
    "src/cpu/deps.cpp"
    "src/cpu/voxel_app.cpp"
    "src/cpu/app_ui.cpp"
    "src/cpu/app_audio.cpp"
    "src/cpu/app_settings.cpp"
    "src/cpu/mesh_model.cpp"
    "src/renderer/fsr.cpp"
    "src/renderer/kajiya/spp64.cpp"
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
set_project_warnings(${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME} PRIVATE GVOX_ENGINE_INSTALL=${GVOX_ENGINE_INSTALL})

find_package(daxa CONFIG REQUIRED)
find_package(gvox CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(platform_folders CONFIG REQUIRED)
find_package(unofficial-nativefiledialog CONFIG REQUIRED)
find_package(unofficial-minizip CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(soloud CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(fsr2 CONFIG REQUIRED)

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

target_link_libraries(${PROJECT_NAME} PRIVATE
    daxa::daxa
    gvox::gvox
    fmt::fmt
    glfw
    nlohmann_json::nlohmann_json
    sago::platform_folders
    unofficial::nativefiledialog::nfd
    unofficial::minizip::minizip
    assimp::assimp
    freeimage::FreeImage
    glm::glm
    soloud
    fsr2::ffx_fsr2_api
    fsr2::ffx_fsr2_api_vk
)
target_include_directories(${PROJECT_NAME} PRIVATE
    "src"
)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    #     target_link_options(${PROJECT_NAME} PRIVATE "-Wl,/ENTRY:mainCRTStartup,/SUBSYSTEM:WINDOWS")
    # else()
    #     target_link_options(${PROJECT_NAME} PRIVATE /ENTRY:mainCRTStartup /SUBSYSTEM:WINDOWS)
    # endif()
    target_link_libraries(${PROJECT_NAME} PRIVATE Dwmapi)
endif()

set(PACKAGE_VOXEL_GAME ${GVOX_ENGINE_INSTALL})

if(PACKAGE_VOXEL_GAME)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        list(APPEND RUNTIME_ARTIFACT_TARGETS fmt::fmt glfw gvox::gvox)
    endif()

    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
    install(IMPORTED_RUNTIME_ARTIFACTS ${RUNTIME_ARTIFACT_TARGETS})
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/assets" DESTINATION bin)
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/src/gpu" DESTINATION bin)
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/src/shared" DESTINATION bin/gpu)
    install(FILES "${daxa_DIR}/../../include/daxa/daxa.inl" "${daxa_DIR}/../../include/daxa/daxa.glsl" DESTINATION bin/gpu/daxa)
    install(FILES "${daxa_DIR}/../../include/daxa/utils/task_graph.inl" DESTINATION bin/gpu/daxa/utils)
    install(FILES "${CMAKE_SOURCE_DIR}/appicon.png" DESTINATION bin)
    install(FILES "${CMAKE_SOURCE_DIR}/imgui.ini" DESTINATION bin)
    # install(FILES "${CMAKE_SOURCE_DIR}/zlib1.dll" DESTINATION bin)
    install(FILES $<TARGET_RUNTIME_DLLS:gvox_engine> DESTINATION bin)
    foreach(FORMAT_NAME ${GVOX_DYNAMIC_FORMATS})
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
