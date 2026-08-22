#pragma once

struct GLFWwindow;

struct Window {
    GLFWwindow *glfw_window_ptr;
    int size_x;
    int size_y;
    bool minimized = false;
    bool mouse_captured = false;

    void *user_data = nullptr;
    void (*on_mouse_move)(void *user_data, daxa_f32 x, daxa_f32 y) = nullptr;
    void (*on_mouse_scroll)(void *user_data, daxa_f32 dx, daxa_f32 dy) = nullptr;
    void (*on_mouse_button)(void *user_data, daxa_i32 button_id, daxa_i32 action) = nullptr;
    void (*on_key)(void *user_data, daxa_i32 key_id, daxa_i32 action) = nullptr;
    void (*on_resize)(void *user_data, daxa_u32 sx, daxa_u32 sy) = nullptr;
    void (*on_drop)(void *user_data, char const *const *filepaths, int filepath_count) = nullptr;

    explicit Window(char const *window_name, int a_size_x = 800, int a_size_y = 600);
    Window(Window const &) = delete;
    Window(Window &&) = delete;
    auto operator=(Window const &) -> Window & = delete;
    auto operator=(Window &&) -> Window & = delete;
    ~Window();

    void set_mouse_pos(daxa_f32 x, daxa_f32 y);
    void set_mouse_capture(bool should_capture);
};
