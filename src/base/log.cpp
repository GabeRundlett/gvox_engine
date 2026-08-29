#include "log.hpp"

#include <cstdarg>
#include <cstdio>

// Each message is formatted into one buffer and written with a single stdio
// call, so lines logged from worker threads (eg. the parallel shader compile
// and pipeline creation passes) cannot interleave mid-line with each other.
namespace {
    void log_line(char const *prefix, char const *fmt, va_list args) {
        char buf[2048];
        auto n = 0;
        if (prefix != nullptr) {
            n = std::snprintf(buf, sizeof(buf), "%s", prefix);
            if (n < 0) {
                return;
            }
        }
        auto const remaining = sizeof(buf) - static_cast<unsigned>(n);
        auto const written = std::vsnprintf(buf + n, remaining, fmt, args);
        if (written < 0) {
            return;
        }
        // vsnprintf returns the length it *would* have written; clamp so an
        // over-long message is truncated rather than read out of bounds.
        auto total = static_cast<unsigned>(n) + static_cast<unsigned>(written);
        if (total > sizeof(buf) - 2) {
            total = sizeof(buf) - 2;
        }
        buf[total] = '\n';
        buf[total + 1] = '\0';
        std::fputs(buf, stderr);
    }
} // namespace

void log_error(char const *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_line("[error] ", fmt, args);
    va_end(args);
}

void log_info(char const *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_line(nullptr, fmt, args);
    va_end(args);
}
