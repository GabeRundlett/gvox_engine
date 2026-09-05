
if(PACKAGE_VOXEL_GAME)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        list(APPEND RUNTIME_ARTIFACT_TARGETS fmt::fmt glfw)
    endif()

    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
    install(IMPORTED_RUNTIME_ARTIFACTS ${RUNTIME_ARTIFACT_TARGETS})
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/assets" DESTINATION bin)
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/src" DESTINATION bin FILES_MATCHING REGEX "glsl|inl")
    install(FILES "${daxa_DIR}/../../include/daxa/daxa.inl" "${daxa_DIR}/../../include/daxa/daxa.glsl" DESTINATION bin/src/daxa)
    install(FILES "${daxa_DIR}/../../include/daxa/utils/task_graph.inl" DESTINATION bin/src/daxa/utils)
    install(FILES "${CMAKE_SOURCE_DIR}/appicon.png" DESTINATION bin)
    install(FILES "${CMAKE_SOURCE_DIR}/imgui.ini" DESTINATION bin)
    install(FILES $<TARGET_RUNTIME_DLLS:gvox_engine> DESTINATION bin)

    set(CPACK_PACKAGE_NAME "GabeVoxelGame")
    set(CPACK_PACKAGE_VENDOR "Gabe-Rundlett")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Gabe Voxel Game is a simple app developed my Gabe Rundlett")
    set(CPACK_PACKAGE_DESCRIPTION "Gabe Voxel Game is a simple app developed my Gabe Rundlett. It is in very early development, and is being documented on Gabe's YouTube channel 'Gabe Rundlett'.")
    set(CPACK_RESOURCE_FILE_WELCOME "${CMAKE_SOURCE_DIR}/packaging/infos/welcome.txt")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/packaging/infos/license.txt")
    set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/packaging/infos/readme.txt")
    set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/appicon.png")

    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        # configure_file("packaging/main.rc.in" "${CMAKE_BINARY_DIR}/main.rc")
        # target_sources(${PROJECT_NAME} PRIVATE "${CMAKE_BINARY_DIR}/main.rc")

        # set(CPACK_GENERATOR WIX)
        # set(CPACK_WIX_UPGRADE_GUID 186207C7-9FC3-4F45-9FB1-6C515E0A93CC)
        # set(CPACK_PACKAGE_EXECUTABLES ${PROJECT_NAME} "Gabe Voxel Game")
        # set(CPACK_WIX_PRODUCT_ICON "${CMAKE_SOURCE_DIR}/appicon.png")

        # # Set the default installation directory. In this case it becomes C:/Program Files/GabeVoxelGame
        # set(CPACK_PACKAGE_INSTALL_DIRECTORY "GabeVoxelGame")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # TODO: Find a better way to package, though tar.gz works for now
        # install(FILES "${CMAKE_SOURCE_DIR}/packaging/gabe_voxel_game.desktop" DESTINATION share/applications)
        # set(CPACK_BINARY_AppImage ON)
    endif()

    include(InstallRequiredSystemLibraries)
    include(CPack)
endif()
