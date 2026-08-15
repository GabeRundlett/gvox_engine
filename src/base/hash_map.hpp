#pragma once

// Minimal open-addressing hash map. No STL. Keys must be equality-comparable
// and hashable via a `hash_key` overload (see below for the built-ins).
// Iteration order is unspecified. Not thread-safe.

inline auto hash_key(unsigned long long v) -> unsigned long long {
    // splitmix64 finalizer
    v += 0x9e3779b97f4a7c15ull;
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ull;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebull;
    return v ^ (v >> 31);
}

inline auto hash_bytes(char const *data, int len) -> unsigned long long {
    // FNV-1a
    auto h = 0xcbf29ce484222325ull;
    for (int i = 0; i < len; ++i) {
        h ^= static_cast<unsigned char>(data[i]);
        h *= 0x100000001b3ull;
    }
    return h;
}

template <typename K, typename V>
struct HashMap {
    struct Slot {
        K key{};
        V value{};
        // 0 = empty, 1 = occupied, 2 = tombstone
        unsigned char state = 0;
    };

    Slot *slots = nullptr;
    int capacity = 0;
    int count = 0;

    HashMap() = default;
    HashMap(HashMap const &other) { *this = other; }
    HashMap(HashMap &&other) noexcept {
        slots = other.slots;
        capacity = other.capacity;
        count = other.count;
        other.slots = nullptr;
        other.capacity = 0;
        other.count = 0;
    }
    ~HashMap() { delete[] slots; }

    auto operator=(HashMap const &other) -> HashMap & {
        if (this == &other) {
            return *this;
        }
        delete[] slots;
        slots = nullptr;
        capacity = 0;
        count = 0;
        if (other.capacity > 0) {
            slots = new Slot[static_cast<unsigned>(other.capacity)];
            for (int i = 0; i < other.capacity; ++i) {
                slots[i] = other.slots[i];
            }
            capacity = other.capacity;
            count = other.count;
        }
        return *this;
    }
    auto operator=(HashMap &&other) noexcept -> HashMap & {
        if (this == &other) {
            return *this;
        }
        delete[] slots;
        slots = other.slots;
        capacity = other.capacity;
        count = other.count;
        other.slots = nullptr;
        other.capacity = 0;
        other.count = 0;
        return *this;
    }

    void rehash(int new_capacity) {
        auto *old_slots = slots;
        auto old_capacity = capacity;
        slots = new Slot[static_cast<unsigned>(new_capacity)];
        capacity = new_capacity;
        count = 0;
        for (int i = 0; i < old_capacity; ++i) {
            if (old_slots[i].state == 1) {
                set(old_slots[i].key, old_slots[i].value);
            }
        }
        delete[] old_slots;
    }

    auto slot_index_of(K const &key) const -> int {
        if (capacity == 0) {
            return -1;
        }
        auto mask = static_cast<unsigned long long>(capacity - 1);
        auto i = hash_key(key) & mask;
        for (int probe = 0; probe < capacity; ++probe) {
            auto &s = slots[i];
            if (s.state == 0) {
                return -1;
            }
            if (s.state == 1 && s.key == key) {
                return static_cast<int>(i);
            }
            i = (i + 1) & mask;
        }
        return -1;
    }

    void set(K const &key, V const &value) {
        if (capacity == 0 || (count + 1) * 4 >= capacity * 3) {
            rehash(capacity == 0 ? 16 : capacity * 2);
        }
        auto mask = static_cast<unsigned long long>(capacity - 1);
        auto i = hash_key(key) & mask;
        auto first_tombstone = -1;
        for (int probe = 0; probe < capacity; ++probe) {
            auto &s = slots[i];
            if (s.state == 1 && s.key == key) {
                s.value = value;
                return;
            }
            if (s.state == 2 && first_tombstone < 0) {
                first_tombstone = static_cast<int>(i);
            }
            if (s.state == 0) {
                auto target = first_tombstone >= 0 ? first_tombstone : static_cast<int>(i);
                slots[target].key = key;
                slots[target].value = value;
                slots[target].state = 1;
                ++count;
                return;
            }
            i = (i + 1) & mask;
        }
    }

    auto get(K const &key) -> V * {
        auto i = slot_index_of(key);
        return i < 0 ? nullptr : &slots[i].value;
    }
    auto get(K const &key) const -> V const * {
        auto i = slot_index_of(key);
        return i < 0 ? nullptr : &slots[i].value;
    }
    auto contains(K const &key) const -> bool { return slot_index_of(key) >= 0; }

    void remove(K const &key) {
        auto i = slot_index_of(key);
        if (i >= 0) {
            // Clear key/value but leave a tombstone so probe chains stay intact.
            slots[i] = Slot{.state = 2};
            --count;
        }
    }

    void clear() {
        for (int i = 0; i < capacity; ++i) {
            slots[i] = Slot{};
        }
        count = 0;
    }

    // Range-for support; yields Slot& for occupied slots only.
    struct Iter {
        Slot *ptr;
        Slot *end_ptr;
        void skip_empty() {
            while (ptr != end_ptr && ptr->state != 1) {
                ++ptr;
            }
        }
        auto operator!=(Iter const &other) const -> bool { return ptr != other.ptr; }
        void operator++() {
            ++ptr;
            skip_empty();
        }
        auto operator*() const -> Slot & { return *ptr; }
    };
    auto begin() -> Iter {
        auto it = Iter{slots, slots + capacity};
        it.skip_empty();
        return it;
    }
    auto end() -> Iter { return Iter{slots + capacity, slots + capacity}; }

    struct ConstIter {
        Slot const *ptr;
        Slot const *end_ptr;
        void skip_empty() {
            while (ptr != end_ptr && ptr->state != 1) {
                ++ptr;
            }
        }
        auto operator!=(ConstIter const &other) const -> bool { return ptr != other.ptr; }
        void operator++() {
            ++ptr;
            skip_empty();
        }
        auto operator*() const -> Slot const & { return *ptr; }
    };
    auto begin() const -> ConstIter {
        auto it = ConstIter{slots, slots + capacity};
        it.skip_empty();
        return it;
    }
    auto end() const -> ConstIter { return ConstIter{slots + capacity, slots + capacity}; }
};
