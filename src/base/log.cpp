#include "log.hpp"

#include <cstdarg>
#include <cstdio>

void log_error(char const *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::fputs("[error] ", stderr);
    std::vfprintf(stderr, fmt, args);
    std::fputc('\n', stderr);
    va_end(args);
}

void log_info(char const *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    std::fputc('\n', stderr);
    va_end(args);
}
