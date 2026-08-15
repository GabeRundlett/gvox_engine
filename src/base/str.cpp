#include "str.hpp"

#include <cstring>

Str::Str() = default;
Str::Str(char const *s) { assign(s, s ? static_cast<int>(std::strlen(s)) : 0); }
Str::Str(char const *s, int len) { assign(s, len); }
Str::Str(Str const &other) { assign(other.data, other.length); }
Str::Str(Str &&other) noexcept {
    data = other.data;
    length = other.length;
    capacity = other.capacity;
    other.data = nullptr;
    other.length = 0;
    other.capacity = 0;
}
Str::~Str() { delete[] data; }

void Str::assign(char const *s, int len) {
    if (len + 1 > capacity) {
        delete[] data;
        capacity = len + 1;
        data = new char[static_cast<unsigned>(capacity)];
    }
    if (len > 0) {
        std::memcpy(data, s, static_cast<unsigned>(len));
    }
    length = len;
    if (data != nullptr) {
        data[length] = '\0';
    }
}

auto Str::operator=(char const *s) -> Str & {
    assign(s, s ? static_cast<int>(std::strlen(s)) : 0);
    return *this;
}
auto Str::operator=(Str const &other) -> Str & {
    if (this == &other) {
        return *this;
    }
    assign(other.data, other.length);
    return *this;
}
auto Str::operator=(Str &&other) noexcept -> Str & {
    if (this == &other) {
        return *this;
    }
    delete[] data;
    data = other.data;
    length = other.length;
    capacity = other.capacity;
    other.data = nullptr;
    other.length = 0;
    other.capacity = 0;
    return *this;
}

auto Str::c_str() const -> char const * { return data ? data : ""; }
auto Str::empty() const -> bool { return length == 0; }
void Str::clear() {
    length = 0;
    if (data != nullptr) {
        data[0] = '\0';
    }
}
void Str::append(char const *s) {
    auto add_len = s ? static_cast<int>(std::strlen(s)) : 0;
    if (add_len == 0) {
        return;
    }
    auto new_len = length + add_len;
    if (new_len + 1 > capacity) {
        auto new_capacity = new_len + 1;
        auto *new_data = new char[static_cast<unsigned>(new_capacity)];
        if (length > 0) {
            std::memcpy(new_data, data, static_cast<unsigned>(length));
        }
        delete[] data;
        data = new_data;
        capacity = new_capacity;
    }
    std::memcpy(data + length, s, static_cast<unsigned>(add_len));
    length = new_len;
    data[length] = '\0';
}
void Str::append(Str const &other) { append(other.c_str()); }

void Str::append(unsigned long long value) {
    char buf[21];
    auto i = int{20};
    buf[i] = '\0';
    if (value == 0) {
        buf[--i] = '0';
    }
    while (value > 0) {
        buf[--i] = static_cast<char>('0' + (value % 10));
        value /= 10;
    }
    append(buf + i);
}

auto hash_key(Str const &s) -> unsigned long long {
    // FNV-1a, matching hash_bytes in hash_map.hpp
    auto h = 0xcbf29ce484222325ull;
    auto const *p = s.c_str();
    for (int i = 0; i < s.length; ++i) {
        h ^= static_cast<unsigned char>(p[i]);
        h *= 0x100000001b3ull;
    }
    return h;
}

auto Str::operator==(char const *other) const -> bool {
    return std::strcmp(c_str(), other ? other : "") == 0;
}
auto Str::operator==(Str const &other) const -> bool {
    return length == other.length && std::strcmp(c_str(), other.c_str()) == 0;
}
auto Str::compare(Str const &other) const -> int {
    return std::strcmp(c_str(), other.c_str());
}
