cmake_minimum_required(VERSION 3.21)

include(FetchContent)

if(NOT TARGET zlibstatic)
    set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        zlib
        GIT_REPOSITORY https://github.com/madler/zlib
        GIT_TAG v1.3.1
    )
    FetchContent_MakeAvailable(zlib)
endif()

add_library(minizip STATIC
    "minizip/ioapi.c"
    "minizip/unzip.c"
    "minizip/zip.c"
)
target_compile_definitions(minizip PRIVATE NOCRYPT NOUNCRYPT)
target_include_directories(minizip PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${zlib_SOURCE_DIR}"
    "${zlib_BINARY_DIR}"
)
target_link_libraries(minizip PUBLIC zlibstatic)

add_library(unofficial::minizip::minizip ALIAS minizip)
