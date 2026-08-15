#pragma once

#include "vec.hpp"
#include "str.hpp"

// Reads the entire file as text. Returns false if it couldn't be opened.
auto read_file_to_string(char const *path, Str &out) -> bool;

// Reads the entire file as raw 32-bit words (used for .spv). Returns false if
// it couldn't be opened or the size isn't a multiple of 4.
auto read_file_to_u32s(char const *path, Vec<unsigned int> &out) -> bool;

// Reads the entire file as raw bytes. Unlike read_file_to_string this is safe
// for binary data (a Str would truncate at the first embedded null).
auto read_file_to_bytes(char const *path, Vec<char> &out) -> bool;

// Writes raw bytes, truncating/creating. Returns false on failure.
auto write_file(char const *path, void const *data, int size) -> bool;

// Creates a directory and any missing parents. Returns false on failure.
auto create_directories(char const *path) -> bool;
