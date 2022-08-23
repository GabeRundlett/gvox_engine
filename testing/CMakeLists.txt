project(${PROJECT_NAME}_testing)

find_package(libsoundio CONFIG REQUIRED)

add_executable(${PROJECT_NAME}_audio "audio/main.cpp")
target_compile_features(${PROJECT_NAME}_audio PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME}_audio PRIVATE libsoundio::libsoundio daxa::daxa fmt::fmt glfw)
