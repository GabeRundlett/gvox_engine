#pragma once

#include "str.hpp"

// Filesystem path queries. No STL in this header (the .cpp uses <filesystem>).

auto path_exists(char const *path) -> bool;

// Creates the directory if it doesn't already exist. Returns false on failure.
auto path_create_directory(char const *path) -> bool;

// Last-write time in implementation-defined ticks; 0 if the file is missing.
// Only ever compared for equality against a previously recorded value.
auto path_modified_time(char const *path) -> unsigned long long;

// Collapses "foo/bar/../baz" to "foo/baz" and normalizes separators to '/',
// so the same file always maps to the same string (used as a cache key).
auto path_normalize(char const *path) -> Str;

// "a/b/c.glsl" -> "a/b" (no trailing slash). Returns "" if there's no directory part.
auto path_dir_part(char const *path) -> Str;

// Absolute, normalized path. Falls back to the input if it can't be resolved.
auto path_absolute(char const *path) -> Str;
