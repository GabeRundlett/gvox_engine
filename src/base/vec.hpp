#pragma once

// <initializer_list> is not really "the STL": std::initializer_list is a
// compiler-intrinsic type the language requires for braced-init-lists, and the
// header is tiny. Without it, every `.defines = {{"X", "1"}}` call site would
// have to be rewritten into push_back calls.
#include <initializer_list>

// Minimal growable array. No STL. Deliberately not a drop-in std::vector:
// no iterators, no allocator support, no exceptions. T must be trivially
// relocatable (this type uses realloc-style growth, not move-construct).
template <typename T>
struct Vec {
    T *data = nullptr;
    int size = 0;
    int capacity = 0;

    Vec() = default;
    Vec(std::initializer_list<T> init) {
        reserve(static_cast<int>(init.size()));
        for (auto const &value : init) {
            data[size] = value;
            ++size;
        }
    }
    Vec(Vec const &other) { *this = other; }
    Vec(Vec &&other) noexcept {
        data = other.data;
        size = other.size;
        capacity = other.capacity;
        other.data = nullptr;
        other.size = 0;
        other.capacity = 0;
    }
    ~Vec() { delete[] data; }

    auto operator=(Vec const &other) -> Vec & {
        if (this == &other) {
            return *this;
        }
        delete[] data;
        data = nullptr;
        size = 0;
        capacity = 0;
        reserve(other.size);
        for (int i = 0; i < other.size; ++i) {
            data[i] = other.data[i];
        }
        size = other.size;
        return *this;
    }
    auto operator=(Vec &&other) noexcept -> Vec & {
        if (this == &other) {
            return *this;
        }
        delete[] data;
        data = other.data;
        size = other.size;
        capacity = other.capacity;
        other.data = nullptr;
        other.size = 0;
        other.capacity = 0;
        return *this;
    }

    void reserve(int new_capacity) {
        if (new_capacity <= capacity) {
            return;
        }
        auto *new_data = new T[static_cast<unsigned>(new_capacity)];
        for (int i = 0; i < size; ++i) {
            new_data[i] = static_cast<T &&>(data[i]);
        }
        delete[] data;
        data = new_data;
        capacity = new_capacity;
    }
    void push_back(T const &value) {
        if (size == capacity) {
            reserve(capacity == 0 ? 4 : capacity * 2);
        }
        data[size] = value;
        ++size;
    }
    void push_back(T &&value) {
        if (size == capacity) {
            reserve(capacity == 0 ? 4 : capacity * 2);
        }
        data[size] = static_cast<T &&>(value);
        ++size;
    }
    void pop_back() { --size; }
    void clear() { size = 0; }
    void erase(int index) {
        for (int i = index; i < size - 1; ++i) {
            data[i] = static_cast<T &&>(data[i + 1]);
        }
        --size;
    }
    void resize(int new_size) {
        reserve(new_size);
        for (int i = size; i < new_size; ++i) {
            data[i] = T{};
        }
        size = new_size;
    }
    void resize(int new_size, T const &value) {
        reserve(new_size);
        for (int i = size; i < new_size; ++i) {
            data[i] = value;
        }
        size = new_size;
    }
    auto empty() const -> bool { return size == 0; }

    auto operator[](int i) -> T & { return data[i]; }
    auto operator[](int i) const -> T const & { return data[i]; }
    auto back() -> T & { return data[size - 1]; }
    auto back() const -> T const & { return data[size - 1]; }

    auto begin() -> T * { return data; }
    auto end() -> T * { return data + size; }
    auto begin() const -> T const * { return data; }
    auto end() const -> T const * { return data + size; }
};
