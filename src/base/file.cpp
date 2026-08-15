// MSVC's CRT deprecates fopen in favour of fopen_s; the plain C version is
// what we want here and is portable. Must precede any CRT header.
#define _CRT_SECURE_NO_WARNINGS 1

#include "file.hpp"

#include <cstdio>
#include <filesystem>

auto read_file_to_string(char const *path, Str &out) -> bool {
    auto *f = std::fopen(path, "rb");
    if (f == nullptr) {
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    auto size = static_cast<int>(std::ftell(f));
    std::fseek(f, 0, SEEK_SET);
    if (size < 0) {
        std::fclose(f);
        return false;
    }
    auto *buf = new char[static_cast<unsigned>(size) + 1];
    auto read = static_cast<int>(std::fread(buf, 1, static_cast<unsigned>(size), f));
    std::fclose(f);
    buf[read < 0 ? 0 : read] = '\0';
    out = buf;
    delete[] buf;
    return true;
}

auto read_file_to_u32s(char const *path, Vec<unsigned int> &out) -> bool {
    auto *f = std::fopen(path, "rb");
    if (f == nullptr) {
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    auto size = static_cast<long>(std::ftell(f));
    std::fseek(f, 0, SEEK_SET);
    if (size < 0 || (size % 4) != 0) {
        std::fclose(f);
        return false;
    }
    auto word_count = static_cast<int>(size / 4);
    out.reserve(word_count);
    auto read = std::fread(out.data, 1, static_cast<unsigned>(size), f);
    std::fclose(f);
    if (static_cast<long>(read) != size) {
        return false;
    }
    out.size = word_count;
    return true;
}

auto read_file_to_bytes(char const *path, Vec<char> &out) -> bool {
    auto *f = std::fopen(path, "rb");
    if (f == nullptr) {
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    auto size = static_cast<long>(std::ftell(f));
    std::fseek(f, 0, SEEK_SET);
    if (size < 0) {
        std::fclose(f);
        return false;
    }
    out.reserve(static_cast<int>(size));
    auto read = size > 0 ? std::fread(out.data, 1, static_cast<unsigned>(size), f) : size_t{0};
    std::fclose(f);
    if (static_cast<long>(read) != size) {
        return false;
    }
    out.size = static_cast<int>(size);
    return true;
}

auto write_file(char const *path, void const *data, int size) -> bool {
    auto *f = std::fopen(path, "wb");
    if (f == nullptr) {
        return false;
    }
    auto written = size > 0 ? std::fwrite(data, 1, static_cast<unsigned>(size), f) : size_t{0};
    std::fclose(f);
    return static_cast<int>(written) == size;
}

auto create_directories(char const *path) -> bool {
    auto ec = std::error_code{};
    std::filesystem::create_directories(path, ec);
    return !ec;
}
