#include "voxel_app.hpp"
#include <filesystem>

#include <debug_utils.hpp>

void search_for_path_to_fix_working_directory(std::span<std::filesystem::path const> test_paths) {
    auto current_path = std::filesystem::current_path();
    while (true) {
        for (auto const &test_path : test_paths) {
            if (std::filesystem::exists(current_path / test_path)) {
                std::filesystem::current_path(current_path);
                return;
            }
        }
        if (!current_path.has_parent_path()) {
            break;
        }
        current_path = current_path.parent_path();
    }
}

auto main() -> int {
    search_for_path_to_fix_working_directory(std::array{
        std::filesystem::path{".out"},
        std::filesystem::path{"assets"},
    });

    auto global_console = debug_utils::Console{};
    auto global_debug_display = debug_utils::DebugDisplay{};

    auto app = VoxelApp{};
    app.run();
}
