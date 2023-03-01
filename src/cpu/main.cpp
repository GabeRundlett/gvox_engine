#include "voxel_app.hpp"

auto main() -> int {
    auto app = VoxelApp{};

    while (true) {
        if (app.update()) {
            break;
        }
    }
}
