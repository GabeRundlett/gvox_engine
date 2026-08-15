#include "path.hpp"

#include <filesystem>

auto path_exists(char const *path) -> bool {
    auto ec = std::error_code{};
    return std::filesystem::exists(path, ec) && !ec;
}

auto path_create_directory(char const *path) -> bool {
    auto ec = std::error_code{};
    std::filesystem::create_directory(path, ec);
    return !ec;
}

auto path_modified_time(char const *path) -> unsigned long long {
    auto ec = std::error_code{};
    auto t = std::filesystem::last_write_time(path, ec);
    if (ec) {
        return 0;
    }
    return static_cast<unsigned long long>(t.time_since_epoch().count());
}

auto path_normalize(char const *path) -> Str {
    auto ec = std::error_code{};
    auto normalized = std::filesystem::path(path).lexically_normal().string();
    (void)ec;
    for (auto &c : normalized) {
        if (c == '\\') {
            c = '/';
        }
    }
    return Str(normalized.c_str());
}

auto path_absolute(char const *path) -> Str {
    auto ec = std::error_code{};
    auto abs = std::filesystem::absolute(path, ec);
    if (ec) {
        return Str(path);
    }
    auto s = abs.lexically_normal().string();
    for (auto &c : s) {
        if (c == '\\') {
            c = '/';
        }
    }
    return Str(s.c_str());
}

auto path_dir_part(char const *path) -> Str {
    auto normalized = path_normalize(path);
    auto const *s = normalized.c_str();
    auto last_slash = -1;
    for (int i = 0; i < normalized.length; ++i) {
        if (s[i] == '/') {
            last_slash = i;
        }
    }
    if (last_slash < 0) {
        return Str("");
    }
    auto *buf = new char[static_cast<unsigned>(last_slash) + 1];
    for (int i = 0; i < last_slash; ++i) {
        buf[i] = s[i];
    }
    buf[last_slash] = '\0';
    auto result = Str(buf);
    delete[] buf;
    return result;
}
