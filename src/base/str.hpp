#pragma once

// Minimal owning, null-terminated growable string. No STL, no SSO.
struct Str {
    char *data = nullptr;
    int length = 0;
    int capacity = 0;

    Str();
    Str(char const *s);
    Str(char const *s, int len);
    Str(Str const &other);
    Str(Str &&other) noexcept;
    ~Str();

    auto operator=(char const *s) -> Str &;
    auto operator=(Str const &other) -> Str &;
    auto operator=(Str &&other) noexcept -> Str &;

    auto c_str() const -> char const *;
    auto empty() const -> bool;
    void clear();
    void append(char const *s);
    void append(Str const &other);

    void append(unsigned long long value);

    auto operator==(char const *other) const -> bool;
    auto operator==(Str const &other) const -> bool;
    auto operator!=(char const *other) const -> bool { return !(*this == other); }
    auto operator!=(Str const &other) const -> bool { return !(*this == other); }
    auto compare(Str const &other) const -> int;

    auto operator+=(char const *s) -> Str & {
        append(s);
        return *this;
    }
    auto operator+(char const *s) const -> Str {
        auto result = *this;
        result.append(s);
        return result;
    }

  private:
    void assign(char const *s, int len);
};

// Lets Str be used as a HashMap key (found via ADL at instantiation).
auto hash_key(Str const &s) -> unsigned long long;
