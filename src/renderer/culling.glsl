#ifndef RENDERER_CULLING_GLSL
#define RENDERER_CULLING_GLSL

bool is_texel_aabb_hiz_depth_occluded(
    vec2 min_texel_i, vec2 max_texel_i,
    float nearest_depth,
    uvec2 hiz_res,
    daxa_ImageViewIndex hiz) {
    const vec2 f_hiz_resolution = hiz_res;
    const float pixel_range = max(max_texel_i.x - min_texel_i.x + 1.0f, max_texel_i.y - min_texel_i.y + 1.0f);
    const float mip = ceil(log2(max(2.0f, pixel_range))) - 1 /* we want one mip lower, as we sample a quad */;

    // The calculation above gives us a mip level, in which the a 2x2 quad in that mip is just large enough to fit the ndc bounds.
    // When the ndc bounds are shofted from the alignment of that mip levels grid however, we need an even larger quad.
    // We check if the quad at its current position within that mip level fits that quad and if not we move up one mip.
    // This will give us the tightest fit.
    int imip = int(mip);
    const ivec2 min_corner_texel = ivec2(min_texel_i) >> imip;
    const ivec2 max_corner_texel = ivec2(max_texel_i) >> imip;
    if (any(greaterThan(max_corner_texel - min_corner_texel, ivec2(1)))) {
        imip += 1;
    }
    const ivec2 quad_corner_texel = ivec2(min_texel_i) >> imip;
    const ivec2 texel_bounds = max(ivec2(0, 0), (ivec2(f_hiz_resolution) >> imip) - 1);

    const vec4 fetch = vec4(
        texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + ivec2(0, 0), ivec2(0, 0), texel_bounds), imip).x,
        texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + ivec2(0, 1), ivec2(0, 0), texel_bounds), imip).x,
        texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + ivec2(1, 0), ivec2(0, 0), texel_bounds), imip).x,
        texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + ivec2(1, 1), ivec2(0, 0), texel_bounds), imip).x);
    const float conservative_depth = min(min(fetch.x, fetch.y), min(fetch.z, fetch.w));
    const bool depth_cull = nearest_depth < conservative_depth;
    return depth_cull;
}

bool is_ndc_aabb_hiz_depth_occluded(
    vec3 ndc_min, vec3 ndc_max,
    uvec2 viewport_size,
    uvec2 hiz_res,
    daxa_ImageViewIndex hiz) {
    const vec2 scale = vec2(viewport_size) * 0.5;
    const vec2 bias = vec2(viewport_size) * 0.5 + 0.5;
    ndc_min.xy = floor(ndc_min.xy * scale + bias);
    ndc_max.xy = floor(ndc_max.xy * scale + bias);
    ndc_min.xy = (ndc_min.xy - bias) / scale;
    ndc_max.xy = (ndc_max.xy - bias) / scale;

    const vec2 f_hiz_resolution = hiz_res;
    const vec2 min_uv = (ndc_min.xy + 1.0f) * 0.5f;
    const vec2 max_uv = (ndc_max.xy + 1.0f) * 0.5f;
    const vec2 min_texel_i = floor(clamp(f_hiz_resolution * min_uv, vec2(0.0f, 0.0f), f_hiz_resolution - 1.0f));
    const vec2 max_texel_i = floor(clamp(f_hiz_resolution * max_uv, vec2(0.0f, 0.0f), f_hiz_resolution - 1.0f));
    return is_texel_aabb_hiz_depth_occluded(min_texel_i, max_texel_i, ndc_max.z, hiz_res, hiz);
}

bool is_outside_frustum(vec2 ndc_min, vec2 ndc_max) {
    return any(greaterThan(ndc_min, vec2(1))) || any(lessThan(ndc_max, vec2(-1)));
}

bool is_between_raster_grid_lines(vec2 pixel_min, vec2 pixel_max) {
    // Cope epsilon to be conservative
    const float EPS = 1.0 / 256.0f;
    vec2 sample_grid_min = pixel_min - 0.5f - EPS;
    vec2 sample_grid_max = pixel_max - 0.5f + EPS;
    // Checks if the min and the max positions are right next to the same sample grid line.
    // If we are next to the same sample grid line in one dimension we are not rasterized.
    return any(equal(floor(sample_grid_max), floor(sample_grid_min)));
}

bool is_between_raster_grid_lines(vec2 ndc_min, vec2 ndc_max, vec2 resolution) {
    return is_between_raster_grid_lines((ndc_min * 0.5f + 0.5f) * resolution, (ndc_max * 0.5f + 0.5f) * resolution);
}

#endif // RENDERER_CULLING_GLSL
