#pragma once

#include "format.hpp"

// Early-boot logging. Used by code that runs before debug_utils::Console
// exists (shader compilation happens during startup), so it just goes to
// stderr rather than the in-game console.
BASE_PRINTF_FORMAT(1, 2)
void log_error(char const *fmt, ...);

BASE_PRINTF_FORMAT(1, 2)
void log_info(char const *fmt, ...);
