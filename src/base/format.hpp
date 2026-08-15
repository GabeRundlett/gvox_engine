#pragma once

#if defined(__clang__) || defined(__GNUC__)
#define BASE_PRINTF_FORMAT(fmt_index, first_arg_index) [[gnu::format(printf, fmt_index, first_arg_index)]]
#else
#define BASE_PRINTF_FORMAT(fmt_index, first_arg_index)
#endif

// Fixed-capacity, stack-only formatted buffer. No heap, no STL.
// Truncates (like snprintf) if the result would exceed the buffer.
struct FormatBuffer {
    char data[512];
    int length = 0;

    operator char const *() const { return data; }
};

BASE_PRINTF_FORMAT(1, 2)
auto format(char const *fmt, ...) -> FormatBuffer;
