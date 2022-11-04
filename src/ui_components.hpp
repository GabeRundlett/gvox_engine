#pragma once

#include <daxa/daxa.hpp>
using namespace daxa::types;

enum struct UiComponentID {
    COLOR,
    SLIDER_U32,
    SLIDER_F32,
    SLIDER_F32VEC3,
    LAST,
};

static constexpr std::array<std::string_view, static_cast<usize>(UiComponentID::LAST)> ui_component_strings{
    "color",
    "slider_u32",
    "slider_f32",
    "slider_f32vec3",
};

static constexpr std::array<usize, static_cast<usize>(UiComponentID::LAST)> ui_component_sizes{
    sizeof(f32vec3),
    sizeof(u32),
    sizeof(f32),
    sizeof(f32vec3),
};
