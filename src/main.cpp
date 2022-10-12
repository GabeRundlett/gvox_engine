#define STB_IMAGE_IMPLEMENTATION

#include "app.hpp"

int main() {
    App app = {};
    while (true) {
        if (app.update())
            break;
    }
}
