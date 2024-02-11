#include <daxa/daxa.hpp>
using namespace daxa::types;

#include <GLFW/glfw3.h>
#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_NATIVE_INCLUDE_NONE
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <dwmapi.h>
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
// #define GLFW_EXPOSE_NATIVE_WAYLAND
#endif
#include <GLFW/glfw3native.h>

#include <FreeImage.h>
#include <cassert>

#include <span>

template <typename App>
struct AppWindow {
    GLFWwindow *glfw_window_ptr;
    daxa_u32vec2 window_size;
    bool minimized = false;
    bool mouse_captured = false;

    AppWindow(char const *window_name, daxa_u32vec2 a_size = daxa_u32vec2{800, 600}) : window_size{a_size} {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfw_window_ptr = glfwCreateWindow(static_cast<daxa_i32>(window_size.x), static_cast<daxa_i32>(window_size.y), window_name, nullptr, nullptr);
        glfwSetWindowUserPointer(glfw_window_ptr, this);
        glfwSetCursorPosCallback(
            glfw_window_ptr,
            [](GLFWwindow *window_ptr, daxa_f64 x, daxa_f64 y) {
                auto &app = *reinterpret_cast<App *>(glfwGetWindowUserPointer(window_ptr));
                app.on_mouse_move(static_cast<daxa_f32>(x), static_cast<daxa_f32>(y));
            });
        glfwSetScrollCallback(
            glfw_window_ptr,
            [](GLFWwindow *window_ptr, daxa_f64 x, daxa_f64 y) {
                auto &app = *reinterpret_cast<App *>(glfwGetWindowUserPointer(window_ptr));
                app.on_mouse_scroll(static_cast<daxa_f32>(x), static_cast<daxa_f32>(y));
            });
        glfwSetMouseButtonCallback(
            glfw_window_ptr,
            [](GLFWwindow *window_ptr, daxa_i32 key, daxa_i32 action, daxa_i32) {
                auto &app = *reinterpret_cast<App *>(glfwGetWindowUserPointer(window_ptr));
                app.on_mouse_button(key, action);
            });
        glfwSetKeyCallback(
            glfw_window_ptr,
            [](GLFWwindow *window_ptr, daxa_i32 key, daxa_i32, daxa_i32 action, daxa_i32) {
                auto &app = *reinterpret_cast<App *>(glfwGetWindowUserPointer(window_ptr));
                app.on_key(key, action);
            });
        glfwSetWindowSizeCallback(
            glfw_window_ptr,
            [](GLFWwindow *window_ptr, daxa_i32 sx, daxa_i32 sy) {
                auto &app = *reinterpret_cast<App *>(glfwGetWindowUserPointer(window_ptr));
                app.on_resize(static_cast<daxa_u32>(sx), static_cast<daxa_u32>(sy));
            });
        glfwSetDropCallback(
            glfw_window_ptr,
            [](GLFWwindow *window_ptr, int path_count, char const *paths[]) {
                auto &app = *reinterpret_cast<App *>(glfwGetWindowUserPointer(window_ptr));
                app.on_drop(std::span<char const *>{paths, static_cast<size_t>(path_count)});
            });

        GLFWimage images[1];
        auto const *texture_path = "appicon.png";
        auto fi_file_desc = FreeImage_GetFileType(texture_path, 0);
        auto *fi_bitmap = FreeImage_Load(fi_file_desc, texture_path);
        auto pixel_size = FreeImage_GetBPP(fi_bitmap);
        if (pixel_size != 32) {
            auto *temp = FreeImage_ConvertTo32Bits(fi_bitmap);
            FreeImage_Unload(fi_bitmap);
            fi_bitmap = temp;
        }
        FreeImage_FlipVertical(fi_bitmap);
        images[0].width = static_cast<int32_t>(FreeImage_GetWidth(fi_bitmap));
        images[0].height = static_cast<int32_t>(FreeImage_GetHeight(fi_bitmap));
        images[0].pixels = FreeImage_GetBits(fi_bitmap);
        assert(images[0].pixels != nullptr && "Failed to load image");
        for (auto &pix : std::span(reinterpret_cast<std::array<uint8_t, 4> *>(images[0].pixels), static_cast<size_t>(images[0].width) * static_cast<size_t>(images[0].height))) {
            std::swap(pix[0], pix[2]);
        }
        glfwSetWindowIcon(glfw_window_ptr, 1, images);
        FreeImage_Unload(fi_bitmap);

#if defined(_WIN32)
        {
            auto hwnd = static_cast<HWND>(get_native_handle());
            BOOL value = TRUE;
            DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &value, sizeof(value));
            auto is_windows11_or_greater = []() -> bool {
                using Fn_RtlGetVersion = void(WINAPI *)(OSVERSIONINFOEX *);
                Fn_RtlGetVersion fn_RtlGetVersion = nullptr;
                auto ntdll_dll = LoadLibrary(TEXT("ntdll.dll"));
                if (ntdll_dll)
                    fn_RtlGetVersion = reinterpret_cast<Fn_RtlGetVersion>(GetProcAddress(ntdll_dll, "RtlGetVersion"));
                auto version_info = OSVERSIONINFOEX{};
                version_info.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
                if (fn_RtlGetVersion) {
                    fn_RtlGetVersion(&version_info);
                }
                FreeLibrary(ntdll_dll);
                return version_info.dwMajorVersion >= 10 && version_info.dwBuildNumber >= 22000;
            };
            if (!is_windows11_or_greater()) {
                MSG msg{.hwnd = hwnd, .message = WM_NCACTIVATE, .wParam = FALSE, .lParam = 0};
                TranslateMessage(&msg);
                DispatchMessage(&msg);
                msg.wParam = TRUE;
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
        // case daxa::NativeWindowPlatform::WAYLAND_API:
        //     return reinterpret_cast<daxa::NativeWindowHandle>(glfwGetWaylandWindow(glfw_window_ptr));
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
        // case GLFW_PLATFORM_WAYLAND: return daxa::NativeWindowPlatform::WAYLAND_API;
        default: return daxa::NativeWindowPlatform::UNKNOWN;
        }
    }

    inline void set_mouse_pos(daxa_f32 x, daxa_f32 y) {
        glfwSetCursorPos(glfw_window_ptr, static_cast<daxa_f64>(x), static_cast<daxa_f64>(y));
    }

    inline void set_mouse_capture(bool should_capture) {
        if (mouse_captured != should_capture) {
            glfwSetCursorPos(glfw_window_ptr, static_cast<daxa_f64>(window_size.x / 2), static_cast<daxa_f64>(window_size.y / 2));
            mouse_captured = should_capture;
        }
        glfwSetInputMode(glfw_window_ptr, GLFW_CURSOR, should_capture ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
        glfwSetInputMode(glfw_window_ptr, GLFW_RAW_MOUSE_MOTION, should_capture);
    }
};
