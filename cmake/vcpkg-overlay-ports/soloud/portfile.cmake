vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL https://github.com/jarikomppa/soloud
    REF 1157475881da0d7f76102578255b937c7d4e8f57
)

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
    backend-sdl2      BACKEND_SDL2
    backend-coreaudio BACKEND_COREAUDIO
    backend-xaudio2   BACKEND_XAUDIO2
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}/contrib"
    OPTIONS
        -DSOLOUD_BACKEND_SDL2=${BACKEND_SDL2}
        -DSOLOUD_BACKEND_COREAUDIO=${BACKEND_COREAUDIO}
        -DSOLOUD_BACKEND_XAUDIO2=${BACKEND_XAUDIO2}
)

vcpkg_cmake_install()
file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/debug/share/soloud")
file(COPY "${CURRENT_PACKAGES_DIR}/debug/cmake/" DESTINATION "${CURRENT_PACKAGES_DIR}/debug/share/soloud")
file(COPY "${CURRENT_PACKAGES_DIR}/cmake/" DESTINATION "${CURRENT_PACKAGES_DIR}/share/soloud")
vcpkg_cmake_config_fixup()
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/cmake")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/cmake")
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)

vcpkg_copy_pdbs()
