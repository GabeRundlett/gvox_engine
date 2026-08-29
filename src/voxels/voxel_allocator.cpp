#include "voxel_allocator.hpp"
#include "voxel_brick.hpp"

// WARNING THIS IS COMPLETELY LLM GENERATED AND TEMPORARY
// TODO(grundlett): Rewrite this in the future

#include <voxels/voxel.inl>
#include <base/vec.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

namespace {

    // Slab allocator with per-thread free lists. The fast path is a couple of
    // pointer writes with no atomics at all; a thread only touches the shared
    // pool when its local list runs dry or grows past MAX_CACHED_BLOCKS, and
    // then it moves a whole batch at once so the mutex stays cold.
    constexpr size_t BLOCK_ALIGN = 64;
    constexpr size_t BATCH_BLOCKS = 64;
    constexpr size_t MAX_CACHED_BLOCKS = BATCH_BLOCKS * 2;
    constexpr size_t FIRST_SLAB_BYTES = 256 * 1024;
    constexpr size_t MAX_SLAB_BYTES = 8 * 1024 * 1024;

    // Free blocks are threaded through their own storage.
    struct FreeBlock {
        FreeBlock *next;
    };

    constexpr size_t align_up(size_t value, size_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    // The shared backing store for one block size. It is held by shared_ptr
    // because a thread cache can outlive the VoxelAllocator that created it:
    // the slabs have to stay mapped until the last cache holding blocks carved
    // out of them is gone.
    struct SharedPool {
        explicit SharedPool(size_t a_block_size)
            : block_size{align_up(std::max(a_block_size, sizeof(FreeBlock)), BLOCK_ALIGN)} {}

        ~SharedPool() {
            for (auto *slab : slabs)
                ::operator delete(slab, std::align_val_t{BLOCK_ALIGN});
        }

        SharedPool(const SharedPool &) = delete;
        SharedPool(SharedPool &&) = delete;
        auto operator=(const SharedPool &) -> SharedPool & = delete;
        auto operator=(SharedPool &&) -> SharedPool & = delete;

        const size_t block_size;
        std::mutex mutex;
        FreeBlock *free_list = nullptr;
        Vec<void *> slabs;
        size_t next_slab_bytes = FIRST_SLAB_BYTES;

        // Detaches up to `count` blocks from the shared list, carving a new slab
        // if it is empty. Returns the head of the chain and how many it took.
        FreeBlock *acquire(size_t count, size_t &out_count) {
            auto lock = std::lock_guard{mutex};
            if (free_list == nullptr)
                carve_slab();
            auto *head = free_list;
            auto *tail = head;
            size_t n = 1;
            while (n < count && tail->next != nullptr) {
                tail = tail->next;
                ++n;
            }
            free_list = tail->next;
            tail->next = nullptr;
            out_count = n;
            return head;
        }

        void release(FreeBlock *head, FreeBlock *tail) {
            auto lock = std::lock_guard{mutex};
            tail->next = free_list;
            free_list = head;
        }

        // Slabs grow geometrically so a small object costs little while a large
        // one stops paying for repeated slab carving.
        void carve_slab() {
            auto const block_count = std::max<size_t>(next_slab_bytes / block_size, 1);
            auto *slab = static_cast<uint8_t *>(::operator new(block_count * block_size, std::align_val_t{BLOCK_ALIGN}));
            slabs.push_back(slab);
            if (next_slab_bytes < MAX_SLAB_BYTES)
                next_slab_bytes *= 2;
            for (size_t i = 0; i < block_count; ++i) {
                auto *block = reinterpret_cast<FreeBlock *>(slab + i * block_size);
                block->next = free_list;
                free_list = block;
            }
        }
    };

    struct ThreadCache {
        ThreadCache() = default;
        ~ThreadCache() { flush_all(); }

        ThreadCache(ThreadCache &&other) noexcept
            : pool{std::move(other.pool)}, head{other.head}, count{other.count} {
            other.head = nullptr;
            other.count = 0;
        }
        ThreadCache(const ThreadCache &) = delete;
        auto operator=(const ThreadCache &) -> ThreadCache & = delete;
        auto operator=(ThreadCache &&) -> ThreadCache & = delete;

        std::shared_ptr<SharedPool> pool;
        FreeBlock *head = nullptr;
        size_t count = 0;

        // Splits `n` blocks off the front of the local list.
        void detach(size_t n, FreeBlock *&out_head, FreeBlock *&out_tail) {
            auto *tail = head;
            for (size_t i = 1; i < n; ++i)
                tail = tail->next;
            out_head = head;
            out_tail = tail;
            head = tail->next;
            tail->next = nullptr;
            count -= n;
        }

        void flush(size_t n) {
            FreeBlock *batch_head = nullptr;
            FreeBlock *batch_tail = nullptr;
            detach(n, batch_head, batch_tail);
            pool->release(batch_head, batch_tail);
        }

        void flush_all() {
            if (count != 0)
                flush(count);
        }

        // A pool id can be reused by a later allocator. If that happened, hand
        // the blocks we still hold back to the pool they actually came from
        // before binding to the new one.
        void rebind(const std::shared_ptr<SharedPool> &new_pool) {
            flush_all();
            pool = new_pool;
        }
    };

    // Pool ids index the per-thread cache table, so they are kept small and
    // dense by recycling them when an allocator is destroyed.
    std::mutex g_id_mutex;
    Vec<uint32_t> g_free_ids;
    uint32_t g_next_id = 0;

    uint32_t acquire_pool_id() {
        auto lock = std::lock_guard{g_id_mutex};
        if (!g_free_ids.empty()) {
            auto id = g_free_ids.back();
            g_free_ids.pop_back();
            return id;
        }
        return g_next_id++;
    }

    void release_pool_id(uint32_t id) {
        auto lock = std::lock_guard{g_id_mutex};
        g_free_ids.push_back(id);
    }

    struct Pool {
        explicit Pool(size_t block_size)
            : id{acquire_pool_id()}, shared{std::make_shared<SharedPool>(block_size)} {}
        ~Pool() { release_pool_id(id); }

        Pool(const Pool &) = delete;
        Pool(Pool &&) = delete;
        auto operator=(const Pool &) -> Pool & = delete;
        auto operator=(Pool &&) -> Pool & = delete;

        uint32_t id;
        std::shared_ptr<SharedPool> shared;
    };

    thread_local std::vector<ThreadCache> t_caches;

    ThreadCache &get_cache(const Pool &pool) {
        if (t_caches.size() <= pool.id)
            t_caches.resize(pool.id + 1);
        auto &cache = t_caches[pool.id];
        if (cache.pool.get() != pool.shared.get())
            cache.rebind(pool.shared);
        return cache;
    }

    void *pool_alloc(const Pool &pool) {
        auto &cache = get_cache(pool);
        if (cache.head == nullptr) {
            size_t n = 0;
            cache.head = cache.pool->acquire(BATCH_BLOCKS, n);
            cache.count = n;
        }
        auto *block = cache.head;
        cache.head = block->next;
        cache.count -= 1;
        return block;
    }

    void pool_free(const Pool &pool, void *ptr) {
        auto &cache = get_cache(pool);
        auto *block = static_cast<FreeBlock *>(ptr);
        block->next = cache.head;
        cache.head = block;
        cache.count += 1;
        if (cache.count >= MAX_CACHED_BLOCKS)
            cache.flush(BATCH_BLOCKS);
    }
} // namespace

struct VoxelAllocator {
    Pool brick_pool{sizeof(VoxelBrick)};
    Pool attrib_pool{sizeof(VoxelShadingAttribBrick)};
};

VoxelAllocator *create_voxel_allocator() {
    return new VoxelAllocator();
}

void destroy_voxel_allocator(VoxelAllocator *self) {
    delete self;
}

// The bricks are value-initialized to keep the zeroed-memory guarantee the
// plain `new VoxelBrick()` path gives callers.

VoxelBrick *alloc_brick(VoxelAllocator *self) {
    return new (pool_alloc(self->brick_pool)) VoxelBrick();
}

void free_brick(VoxelAllocator *self, VoxelBrick *brick) {
    if (brick != nullptr)
        pool_free(self->brick_pool, brick);
}

VoxelShadingAttribBrick *alloc_render_brick(VoxelAllocator *self) {
    return new (pool_alloc(self->attrib_pool)) VoxelShadingAttribBrick();
}

void free_render_brick(VoxelAllocator *self, VoxelShadingAttribBrick *render_brick) {
    if (render_brick != nullptr)
        pool_free(self->attrib_pool, render_brick);
}
