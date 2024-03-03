#pragma once

// https://media.contentapi.ea.com/content/dam/ea/seed/presentations/2019-ray-tracing-gems-chapter-20-akenine-moller-et-al.pdf
struct RayCone {
    float width;
    float spread_angle;
};

RayCone RayCone_from_spread_angle(float spread_angle) {
    RayCone res;
    res.width = 0.0;
    res.spread_angle = spread_angle;
    return res;
}

RayCone RayCone_from_width_spread_angle(float width, float spread_angle) {
    RayCone res;
    res.width = width;
    res.spread_angle = spread_angle;
    return res;
}

RayCone propagate(RayCone self, float surface_spread_angle, float hit_t) {
    RayCone res;
    res.width = self.spread_angle * hit_t + self.width;
    res.spread_angle = self.spread_angle + surface_spread_angle;
    return res;
}

float width_at_t(inout RayCone self, float hit_t) {
    return self.width + self.spread_angle * hit_t;
}

#include <application/input.inl>

float pixel_cone_spread_angle_from_image_height(daxa_BufferPtr(GpuInput) gpu_input, float image_height) {
    return atan(2.0 * deref(gpu_input).player.cam.clip_to_view[0][0] / image_height);
}

RayCone pixel_ray_cone_from_image_height(daxa_BufferPtr(GpuInput) gpu_input, float image_height) {
    RayCone res;
    res.width = 0.0;
    res.spread_angle = pixel_cone_spread_angle_from_image_height(gpu_input, image_height);
    return res;
}
