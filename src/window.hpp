#include <daxa/daxa.hpp>
using namespace daxa::types;

#include <GLFW/glfw3.h>
#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_NATIVE_INCLUDE_NONE
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <dwmapi.h>
#ifndef DWMWA_USE_IMMERSIVE_DARK_MODE
#define DWMWA_USE_IMMERSIVE_DARK_MODE 20
#endif
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_WAYLAND
#endif
#include <GLFW/glfw3native.h>

#include <stb_image.h>

inline char const *get_key_string(i32 glfw_key_id) {
    auto result = glfwGetKeyName(glfw_key_id, 0);
    if (!result) {
        switch (glfw_key_id) {
        case GLFW_KEY_SPACE: result = "space"; break;
        case GLFW_KEY_LEFT_SHIFT: result = "left shift"; break;
        case GLFW_KEY_LEFT_CONTROL: result = "left ctrl"; break;
        case GLFW_KEY_LEFT_ALT: result = "left alt"; break;
        case GLFW_KEY_RIGHT_SHIFT: result = "right shift"; break;
        case GLFW_KEY_RIGHT_CONTROL: result = "right ctrl"; break;
        case GLFW_KEY_RIGHT_ALT: result = "right alt"; break;
        case GLFW_KEY_UP: result = "arrow up"; break;
        case GLFW_KEY_DOWN: result = "arrow down"; break;
        case GLFW_KEY_LEFT: result = "arrow left"; break;
        case GLFW_KEY_RIGHT: result = "arrow right"; break;
        case GLFW_KEY_F1: result = "f1"; break;
        case GLFW_KEY_F2: result = "f2"; break;
        case GLFW_KEY_F3: result = "f3"; break;
        case GLFW_KEY_F4: result = "f4"; break;
        case GLFW_KEY_F5: result = "f5"; break;
        case GLFW_KEY_F6: result = "f6"; break;
        case GLFW_KEY_F7: result = "f7"; break;
        case GLFW_KEY_F8: result = "f8"; break;
        case GLFW_KEY_F9: result = "f9"; break;
        case GLFW_KEY_F10: result = "f10"; break;
        case GLFW_KEY_F11: result = "f11"; break;
        case GLFW_KEY_F12: result = "f12"; break;
        default: result = "Unknown Key"; break;
        }
    }
    return result;
}

template <typename App>
struct AppWindow {
    GLFWwindow *glfw_window_ptr;
    u32 size_x = 800, size_y = 600;
    bool minimized = false;

    AppWindow(char const *window_name) {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfw_window_ptr = glfwCreateWindow(static_cast<i32>(size_x), static_cast<i32>(size_y), window_name, nullptr, nullptr);
        glfwSetWindowUserPointer(glfw_window_ptr, this);
        glfwSetCursorPosCallback(
            glfw_window_ptr,
            [](GLFWwindow *window_ptr, f64 x, f64 y) {
                auto &app = *reinterpret_cast<App *>(glfwGetWindowUserPointer(window_ptr));
                app.on_mouse_move(static_cast<f32>(x), static_cast<f32>(y));
            });
        glfwSetScrollCallback(
            glfw_window_ptr,
            [](GLFWwindow *window_ptr, f64 x, f64 y) {
                auto &app = *reinterpret_cast<App *>(glfwGetWindowUserPointer(window_ptr));
                app.on_mouse_scroll(static_cast<f32>(x), static_cast<f32>(y));
            });
        glfwSetMouseButtonCallback(
            glfw_window_ptr,
            [](GLFWwindow *window_ptr, i32 key, i32 action, i32) {
                auto &app = *reinterpret_cast<App *>(glfwGetWindowUserPointer(window_ptr));
                app.on_mouse_button(key, action);
            });
        glfwSetKeyCallback(
            glfw_window_ptr,
            [](GLFWwindow *window_ptr, i32 key, i32, i32 action, i32) {
                auto &app = *reinterpret_cast<App *>(glfwGetWindowUserPointer(window_ptr));
                app.on_key(key, action);
            });
        glfwSetWindowSizeCallback(
            glfw_window_ptr,
            [](GLFWwindow *window_ptr, i32 sx, i32 sy) {
                auto &app = *reinterpret_cast<App *>(glfwGetWindowUserPointer(window_ptr));
                app.on_resize(static_cast<u32>(sx), static_cast<u32>(sy));
            });

        GLFWimage images[1];
        images[0].pixels = stbi_load("appicon.png", &images[0].width, &images[0].height, 0, 4);
        glfwSetWindowIcon(glfw_window_ptr, 1, images);
        stbi_image_free(images[0].pixels);

#if defined(_WIN32)
        {
            auto hwnd = static_cast<HWND>(get_native_handle());
            BOOL value = TRUE;
            DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &value, sizeof(value));
            PostMessage(hwnd, WM_NCACTIVATE, FALSE, 0);
            PostMessage(hwnd, WM_NCACTIVATE, TRUE, 0);
            MSG msg;
            while (PeekMessage(&msg, hwnd, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
        }
#endif
    }

    ~AppWindow() {
        glfwDestroyWindow(glfw_window_ptr);
        glfwTerminate();
    }

    auto get_native_handle() -> daxa::NativeWindowHandle {
#if defined(_WIN32)
        return glfwGetWin32Window(glfw_window_ptr);
#elif defined(__linux__)
        switch (get_native_platform()) {
        case daxa::NativeWindowPlatform::WAYLAND_API:
            return reinterpret_cast<daxa::NativeWindowHandle>(glfwGetWaylandWindow(glfw_window_ptr));
        case daxa::NativeWindowPlatform::XLIB_API:
        default:
            return reinterpret_cast<daxa::NativeWindowHandle>(glfwGetX11Window(glfw_window_ptr));
        }
#endif
    }

    auto get_native_platform() -> daxa::NativeWindowPlatform {
        switch (glfwGetPlatform()) {
        case GLFW_PLATFORM_WIN32: return daxa::NativeWindowPlatform::WIN32_API;
        case GLFW_PLATFORM_X11: return daxa::NativeWindowPlatform::XLIB_API;
        case GLFW_PLATFORM_WAYLAND: return daxa::NativeWindowPlatform::WAYLAND_API;
        default: return daxa::NativeWindowPlatform::UNKNOWN;
        }
    }

    inline void set_mouse_pos(f32 x, f32 y) {
        glfwSetCursorPos(glfw_window_ptr, static_cast<f64>(x), static_cast<f64>(y));
    }

    inline void set_mouse_capture(bool should_capture) {
        glfwSetCursorPos(glfw_window_ptr, static_cast<f64>(size_x / 2), static_cast<f64>(size_y / 2));
        glfwSetInputMode(glfw_window_ptr, GLFW_CURSOR, should_capture ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
        glfwSetInputMode(glfw_window_ptr, GLFW_RAW_MOUSE_MOTION, should_capture);
    }
};
