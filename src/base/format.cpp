#include "format.hpp"

#include <cstdarg>
#include <cstdio>

auto format(char const *fmt, ...) -> FormatBuffer {
    auto result = FormatBuffer{};
    va_list args;
    va_start(args, fmt);
    auto n = std::vsnprintf(result.data, sizeof(result.data), fmt, args);
    va_end(args);
    result.length = n < 0 ? 0 : (n >= static_cast<int>(sizeof(result.data)) ? static_cast<int>(sizeof(result.data)) - 1 : n);
    return result;
}
