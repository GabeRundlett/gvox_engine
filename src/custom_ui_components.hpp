#pragma once

#include <daxa/daxa.hpp>
using namespace daxa::types;

enum struct UiComponentID {
    COLOR,
    SLIDER_I32,
    SLIDER_U32,
    SLIDER_F32,
    SLIDER_F32VEC3,
    INPUT_F32VEC3,
    LAST,
};

static constexpr std::array<std::string_view, static_cast<usize>(UiComponentID::LAST)> ui_component_strings{
    "color",
    "slider_i32",
    "slider_u32",
    "slider_f32",
    "slider_f32vec3",
    "input_f32vec3",
};

static constexpr std::array<usize, static_cast<usize>(UiComponentID::LAST)> ui_component_sizes{
    sizeof(f32vec3),
    sizeof(i32),
    sizeof(u32),
    sizeof(f32),
    sizeof(f32vec3),
    sizeof(f32vec3),
};

struct CustomBrush_color {
    f32vec3 default_value;
};
struct CustomBrush_slider_i32 {
    i32 default_value;
    i32 min, max;
};
struct CustomBrush_slider_u32 {
    u32 default_value;
    u32 min, max;
};
struct CustomBrush_slider_f32 {
    f32 default_value;
    f32 min, max;
};
struct CustomBrush_slider_f32vec3 {
    f32vec3 default_value;
    f32 min, max;
};
struct CustomBrush_input_f32vec3 {
    f32vec3 default_value;
};

using CustomBrushParameterTypeData = std::variant<
    CustomBrush_color,
    CustomBrush_slider_i32,
    CustomBrush_slider_u32,
    CustomBrush_slider_f32,
    CustomBrush_slider_f32vec3,
    CustomBrush_input_f32vec3>;

struct CustomBrushParameter {
    UiComponentID id;
    std::string name;
    CustomBrushParameterTypeData type_data;
};
