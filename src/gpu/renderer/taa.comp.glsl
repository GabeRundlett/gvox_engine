#include <shared/app.inl>

#include <utils/math.glsl>

#define TAA_NONLINEARITY_TYPE 1
#define TAA_COLOR_MAPPING_MODE 1

#define FRAME_CONSTANTS_PRE_EXPOSURE_DELTA 1.0
#define SAMPLE_OFFSET_PIXELS f32vec2(deref(gpu_input).halton_jitter)

DAXA_DECL_PUSH_CONSTANT(TaaPush, push)

float linear_to_perceptual(float a) {
#if 0 == TAA_NONLINEARITY_TYPE
        return a;
#elif 1 == TAA_NONLINEARITY_TYPE
    return sqrt(max(0.0, a));
#elif 2 == TAA_NONLINEARITY_TYPE
    return max(0.0, log(1.0 + sqrt(max(0.0, a))));
#elif 3 == TAA_NONLINEARITY_TYPE
    return max(0.0, 1.0 - exp(-max(0.0, a)));
#elif 4 == TAA_NONLINEARITY_TYPE
    const float k = 0.25; // Linear part end
    return a < k ? max(0.0, a) : (k - 0.5 + sqrt(a - k + 0.25));
#else
    return 0;
#endif
}

float perceptual_to_linear(float a) {
#if 0 == TAA_NONLINEARITY_TYPE
        return a;
#elif 1 == TAA_NONLINEARITY_TYPE
    return a * a;
#elif 2 == TAA_NONLINEARITY_TYPE
    a = exp(a) - 1.0;
    return a * a;
#elif 3 == TAA_NONLINEARITY_TYPE
    return max(0.0, -log(1.0 - a));
#elif 4 == TAA_NONLINEARITY_TYPE
    const float k = 0.25; // Linear part end
    return a < k ? max(0.0, a) : (square(a - k + 0.5) + k - 0.25);
#else
    return 0;
#endif
}

float max_3(float a, float b, float c) {
    return max(a, max(b, c));
}

f32vec3 square(f32vec3 x) {
    return pow(x, f32vec3(2.0));
}

f32vec3 decode_rgb(f32vec3 v) {
#if 0 == TAA_COLOR_MAPPING_MODE
        return f32vec3(linear_to_perceptual(v.r), linear_to_perceptual(v.g), linear_to_perceptual(v.b));
#elif 1 == TAA_COLOR_MAPPING_MODE
    float max_comp = max_3(v.r, v.g, v.b);
    return v * linear_to_perceptual(max_comp) / max(1e-20, max_comp);
#endif
}

f32vec3 encode_rgb(f32vec3 v) {
#if 0 == TAA_COLOR_MAPPING_MODE
        return f32vec3(perceptual_to_linear(v.r), perceptual_to_linear(v.g), perceptual_to_linear(v.b));
#elif 1 == TAA_COLOR_MAPPING_MODE
    float max_comp = max_3(v.r, v.g, v.b);
    return v * perceptual_to_linear(max_comp) / max(1e-20, max_comp);
#endif
}

#if TAA_REPROJECT_COMPUTE

// Optimization: Try to skip velocity dilation if velocity diff is small
// around the pixel.
#define APPROX_SKIP_DILATION true

f32vec4 fetch_history(f32vec2 uv) {
    f32vec4 h = textureLod(daxa_sampler2D(history_tex, deref(gpu_input).sampler_lnc), uv, 0);
    return f32vec4(decode_rgb(h.xyz * FRAME_CONSTANTS_PRE_EXPOSURE_DELTA), h.w);
}

// struct HistoryRemap {
// };

// HistoryRemap HistoryRemap_create() {
//     HistoryRemap res;
//     return res;
// }

f32vec4 HistoryRemap_remap(f32vec4 v) {
    return f32vec4(decode_rgb(v.rgb * FRAME_CONSTANTS_PRE_EXPOSURE_DELTA), v.a);
}

f32 fetch_depth(i32vec2 px) {
    return texelFetch(daxa_texture2D(depth_image), px, 0).r;
}
f32vec4 fetch_reproj(i32vec2 px) {
    return texelFetch(daxa_texture2D(reprojection_map), px, 0);
}

#define REMAP_FUNC HistoryRemap_remap
f32vec4 image_sample_catmull_rom_approx(in daxa_ImageViewId tex, in daxa_SamplerId linearSampler, in f32vec2 uv, in f32vec2 texSize, bool useCornerTaps) { // , Remap remap = IdentityImageRemap::create()
    // https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1

    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    f32vec2 samplePos = uv * texSize;
    f32vec2 texPos1 = floor(samplePos - 0.5f) + 0.5f;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    f32vec2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    f32vec2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    f32vec2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    f32vec2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    f32vec2 w3 = f * f * (-0.5f + 0.5f * f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    f32vec2 w12 = w1 + w2;
    f32vec2 offset12 = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
    f32vec2 texPos0 = texPos1 - 1;
    f32vec2 texPos3 = texPos1 + 2;
    f32vec2 texPos12 = texPos1 + offset12;

    texPos0 /= texSize;
    texPos3 /= texSize;
    texPos12 /= texSize;

    f32vec4 result = f32vec4(0.0);

    if (useCornerTaps) {
        result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), f32vec2(texPos0.x, texPos0.y), 0.0f)) * w0.x * w0.y;
    }

    result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), f32vec2(texPos12.x, texPos0.y), 0.0f)) * w12.x * w0.y;

    if (useCornerTaps) {
        result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), f32vec2(texPos3.x, texPos0.y), 0.0f)) * w3.x * w0.y;
    }

    result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), f32vec2(texPos0.x, texPos12.y), 0.0f)) * w0.x * w12.y;
    result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), f32vec2(texPos12.x, texPos12.y), 0.0f)) * w12.x * w12.y;
    result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), f32vec2(texPos3.x, texPos12.y), 0.0f)) * w3.x * w12.y;

    if (useCornerTaps) {
        result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), f32vec2(texPos0.x, texPos3.y), 0.0f)) * w0.x * w3.y;
    }

    result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), f32vec2(texPos12.x, texPos3.y), 0.0f)) * w12.x * w3.y;

    if (useCornerTaps) {
        result += REMAP_FUNC(textureLod(daxa_sampler2D(tex, linearSampler), f32vec2(texPos3.x, texPos3.y), 0.0f)) * w3.x * w3.y;
    }

    if (!useCornerTaps) {
        result /= (w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y);
    }

    return result;
}
#undef REMAP_FUNC

f32vec4 image_sample_catmull_rom_5tap(daxa_ImageViewId tex, daxa_SamplerId linearSampler, in f32vec2 uv, in f32vec2 texSize) { // , Remap remap = IdentityImageRemap::create()
    return image_sample_catmull_rom_approx(
        tex, linearSampler, uv, texSize, false);
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    f32vec2 px = gl_GlobalInvocationID.xy;
    const f32vec2 input_resolution_scale = f32vec2(deref(gpu_input).render_res_scl); // input_tex_size.xy / output_tex_size.xy;
    const u32vec2 reproj_px = u32vec2((px + 0.5) * input_resolution_scale);

    f32vec4 output_tex_size = f32vec4(deref(gpu_input).output_resolution.xy, 0, 0);
    output_tex_size.zw = 1.0 / output_tex_size.xy;

    f32vec2 uv = get_uv(px, output_tex_size);
    u32vec2 closest_px = reproj_px;

#if APPROX_SKIP_DILATION
    // Find the bounding box of velocities around this 3x3 region
    f32vec2 vel_min;
    f32vec2 vel_max;
    {
        f32vec2 v = fetch_reproj(reproj_px + i32vec2(-1, -1)).xy;
        vel_min = v;
        vel_max = v;
    }
    {
        f32vec2 v = fetch_reproj(reproj_px + i32vec2(1, -1)).xy;
        vel_min = min(vel_min, v);
        vel_max = max(vel_max, v);
    }
    {
        f32vec2 v = fetch_reproj(reproj_px + i32vec2(-1, 1)).xy;
        vel_min = min(vel_min, v);
        vel_max = max(vel_max, v);
    }
    {
        f32vec2 v = fetch_reproj(reproj_px + i32vec2(1, 1)).xy;
        vel_min = min(vel_min, v);
        vel_max = max(vel_max, v);
    }

    bool should_dilate = any((vel_max - vel_min) > 0.1 * max(input_tex_size.zw, abs(vel_max + vel_min)));

    // Since we're only checking a few pixels, there's a chance we'll miss something.
    // Dilate in the wave to reduce the chance of that happening.
    // should_dilate |= subgroupBroadcast(should_dilate, gl_SubgroupInvocationID ^ 1);
    should_dilate |= subgroupBroadcast(should_dilate, gl_SubgroupInvocationID ^ 2);
    // should_dilate |= subgroupBroadcast(should_dilate, gl_SubgroupInvocationID ^ 8);
    should_dilate |= subgroupBroadcast(should_dilate, gl_SubgroupInvocationID ^ 16);

    // We want to find the velocity of the pixel which is closest to the camera,
    // which is critical to anti-aliased moving edges.
    // At the same time, when everything moves with roughly the same velocity
    // in the neighborhood of the pixel, we'd be performing this depth-based kernel
    // only to return the same value.
    // Therefore, we predicate the search on there being any appreciable
    // velocity difference around the target pixel. This ends up being faster on average.
    if (should_dilate)
#endif
    {
        float reproj_depth = fetch_depth(i32vec2(reproj_px));
        int k = 1;
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                float d = fetch_depth(i32vec2(reproj_px) + i32vec2(x, y));
                if (d > reproj_depth) {
                    reproj_depth = d;
                    closest_px = reproj_px + i32vec2(x, y);
                }
            }
        }
    }

    const f32vec2 reproj_xy = fetch_reproj(i32vec2(closest_px)).xy;
    imageStore(daxa_image2D(closest_velocity_img), i32vec2(px), f32vec4(reproj_xy, 0, 0));
    f32vec2 history_uv = uv + reproj_xy;

#if 0
    f32vec4 history_packed = image_sample_catmull_rom(
        TextureImage::from_parts(history_tex, output_tex_size.xy),
        history_uv,
        HistoryRemap::create()
    );
#elif 1
    f32vec4 history_packed = image_sample_catmull_rom_5tap(
        history_tex, deref(gpu_input).sampler_llc, history_uv, output_tex_size.xy);
#else
    f32vec4 history_packed = fetch_history(history_uv);
#endif

    f32vec3 history = history_packed.rgb;
    float history_coverage = max(0.0, history_packed.a);

    imageStore(daxa_image2D(reprojected_history_img), i32vec2(gl_GlobalInvocationID.xy), f32vec4(history, history_coverage));
}
#endif

#if TAA_FILTER_INPUT_COMPUTE

// struct InputRemap {
// };
// InputRemap InputRemap_create() {
//     InputRemap res;
//     return res;
// }

f32vec4 InputRemap_remap(f32vec4 v) {
    return f32vec4(sRGB_to_YCbCr(decode_rgb(v.rgb)), 1);
}

struct FilteredInput {
    f32vec3 clamped_ex;
    f32vec3 var;
};

f32 fetch_depth(i32vec2 px) {
    return texelFetch(daxa_texture2D(depth_image), px, 0).r;
}
f32vec4 fetch_input(i32vec2 px) {
    return texelFetch(daxa_texture2D(input_image), px, 0);
}

FilteredInput filter_input_inner(u32vec2 px, float center_depth, float luma_cutoff, float depth_scale) {
    f32vec3 iex = f32vec3(0);
    f32vec3 iex2 = f32vec3(0);
    float iwsum = 0;

    f32vec3 clamped_iex = f32vec3(0);
    float clamped_iwsum = 0;

    // InputRemap input_remap = InputRemap::create();

    const int k = 1;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            const i32vec2 spx_offset = i32vec2(x, y);
            const float distance_w = exp(-(0.8 / (k * k)) * dot(f32vec2(spx_offset), f32vec2(spx_offset)));

            const i32vec2 spx = i32vec2(px) + spx_offset;
            f32vec3 s = InputRemap_remap(fetch_input(spx)).rgb;

            const float depth = fetch_depth(spx);
            float w = 1;
            w *= exp2(-min(16, depth_scale * inverse_depth_relative_diff(center_depth, depth)));
            w *= distance_w;
            w *= pow(clamp(luma_cutoff / s.x, 0.0, 1.0), 8);

            clamped_iwsum += w;
            clamped_iex += s * w;

            iwsum += 1;
            iex += s;
            iex2 += s * s;
        }
    }

    clamped_iex /= clamped_iwsum;

    iex /= iwsum;
    iex2 /= iwsum;

    FilteredInput res;
    res.clamped_ex = clamped_iex;
    res.var = max(f32vec3(0.0), iex2 - iex * iex);

    return res;
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    i32vec2 px = i32vec2(gl_GlobalInvocationID.xy);
    const float center_depth = fetch_depth(px);

    // Filter the input, with a cross-bilateral weight based on depth
    FilteredInput filtered_input = filter_input_inner(px, center_depth, 1e10, 200);

    // Filter the input again, but add another cross-bilateral weight, reducing the weight of
    // inputs brighter than the just-estimated luminance mean. This clamps bright outliers in the input.
    FilteredInput clamped_filtered_input = filter_input_inner(px, center_depth, filtered_input.clamped_ex.x * 1.001, 200);

    imageStore(daxa_image2D(filtered_input_img), px, f32vec4(clamped_filtered_input.clamped_ex, 0.0));
    imageStore(daxa_image2D(filtered_input_deviation_img), px, f32vec4(sqrt(filtered_input.var), 0.0));
}

#endif

#if TAA_FILTER_HISTORY_COMPUTE

f32vec4 fetch_input(i32vec2 px) {
    return texelFetch(daxa_texture2D(reprojected_history_img), px, 0);
}

f32vec3 filter_input(f32vec2 uv, float luma_cutoff, int kernel_radius) {
    f32vec3 iex = f32vec3(0);
    float iwsum = 0;

    // Note: + epislon to counter precision loss, which manifests itself
    // as bad rounding in a 2x upscale, showing stair-stepping artifacts.
    i32vec2 src_px = i32vec2(floor(uv * push.input_tex_size.xy + 1e-3));

    const int k = kernel_radius;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            const i32vec2 spx_offset = i32vec2(x, y);
            const i32vec2 spx = i32vec2(src_px) + spx_offset;

            // TODO: consider a weight based on uv diffs between the low-res
            // output `uv` and the low-res input `spx`.
            const float distance_w = exp(-(0.8 / (k * k)) * dot(f32vec2(spx_offset), f32vec2(spx_offset)));

            f32vec3 s = sRGB_to_YCbCr(fetch_input(spx).rgb);

            float w = 1;
            w *= distance_w;
            w *= pow(clamp(luma_cutoff / s.x, 0.0, 1.0), 8);

            iwsum += w;
            iex += s * w;
        }
    }

    return iex / iwsum;
}

void filter_history(u32vec2 px, int kernel_radius) {
    f32vec2 uv = get_uv(px, push.output_tex_size);
    float filtered_luma = filter_input(uv, 1e10, kernel_radius).x;
    imageStore(daxa_image2D(filtered_history_img), i32vec2(px), f32vec4(filter_input(uv, filtered_luma * 1.001, kernel_radius), 0.0));
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 px = gl_GlobalInvocationID.xy;
    if (push.input_tex_size.x / push.output_tex_size.x > 1.75) {
        // If we're upscaling, history is at a higher resolution than
        // the new frame, so we need to filter history more.
        filter_history(px, 2);
    } else {
        filter_history(px, 1);
    }
}

#endif

#if TAA_INPUT_PROB_COMPUTE

// struct InputRemap {
// };
// static InputRemap create() {
//     InputRemap res;
//     return res;
// }

f32vec4 InputRemap_remap(f32vec4 v) {
    return f32vec4(sRGB_to_YCbCr(decode_rgb(v.rgb)), 1);
}

// struct HistoryRemap {
// };
// static HistoryRemap create() {
//     HistoryRemap res;
//     return res;
// }

f32vec4 HistoryRemap_remap(f32vec4 v) {
    return f32vec4(sRGB_to_YCbCr(v.rgb), 1);
}

f32vec4 fetch_filtered_input(i32vec2 px) {
    return texelFetch(daxa_texture2D(filtered_input_img), px, 0);
}

f32vec3 fetch_filtered_input_dev(i32vec2 px) {
    return texelFetch(daxa_texture2D(filtered_input_deviation_img), px, 0).rgb;
}

f32vec4 fetch_reproj(i32vec2 px) {
    return texelFetch(daxa_texture2D(reprojection_map), px, 0);
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    float input_prob = 0;
    i32vec2 px = i32vec2(gl_GlobalInvocationID.xy);

    {
        // InputRemap input_remap = InputRemap::create();

        // Estimate input variance from a pretty large spatial neighborhood
        // We'll combine it with a temporally-filtered variance estimate later.
        f32vec3 ivar = f32vec3(0);
        {
            const int k = 1;
            for (int y = -k; y <= k; ++y) {
                for (int x = -k; x <= k; ++x) {
                    ivar = max(ivar, fetch_filtered_input_dev(px + i32vec2(x, y) * 2));
                }
            }
            ivar = square(ivar);
        }

        const f32vec2 input_uv = (px + SAMPLE_OFFSET_PIXELS) * push.input_tex_size.zw;

        const f32vec4 closest_history = textureLod(daxa_sampler2D(filtered_history_img, deref(gpu_input).sampler_nnc), input_uv, 0);
        const f32vec3 closest_smooth_var = textureLod(daxa_sampler2D(smooth_var_history_tex, deref(gpu_input).sampler_lnc), input_uv + fetch_reproj(px).xy, 0).rgb;
        const f32vec2 closest_vel = textureLod(daxa_sampler2D(velocity_history_tex, deref(gpu_input).sampler_lnc), input_uv + fetch_reproj(px).xy, 0).xy * deref(gpu_input).delta_time;

        // Combine spaital and temporla variance. We generally want to use
        // the smoothed temporal estimate, but bound it by this frame's input,
        // to quickly react for large-scale temporal changes.
        const f32vec3 combined_var = min(closest_smooth_var, ivar * 10);
        // const f32vec3 combined_var = closest_smooth_var;
        // const f32vec3 combined_var = ivar;

        // Check this frame's input, and see how closely it resembles history,
        // taking the variance estimate into account.
        //
        // The idea here is that the new frames are samples from an unknown
        // function, while `closest_history` and `combined_var` are estimates
        // of its mean and variance. We find the probability that the given sample
        // belongs to the estimated distribution.
        //
        // Use a small neighborhood search, because the input may
        // flicker from frame to frame due to the temporal jitter.
        {
            int k = 1;
            for (int y = -k; y <= k; ++y) {
                for (int x = -k; x <= k; ++x) {
                    const f32vec3 s = fetch_filtered_input(px + i32vec2(x, y)).rgb;
                    const f32vec3 idiff = s - closest_history.rgb;

                    const f32vec2 vel = fetch_reproj(px + i32vec2(x, y)).xy;
                    const float vdiff = length((vel - closest_vel) / max(f32vec2(1.0), abs(vel + closest_vel)));

                    float prob = exp2(-1.0 * length(idiff * idiff / max(f32vec3(1e-6), combined_var)) - 1000 * vdiff);

                    input_prob = max(input_prob, prob);
                }
            }
        }
    }

    imageStore(daxa_image2D(input_prob_img), i32vec2(px), f32vec4(input_prob, 0, 0, 0));
}

#endif

#if TAA_PROB_FILTER_COMPUTE

f32 fetch_input(i32vec2 px) {
    return texelFetch(daxa_texture2D(input_prob_img), px, 0).r;
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    i32vec2 px = i32vec2(gl_GlobalInvocationID.xy);
    float prob = fetch_input(px);

    const int k = 1;
    {
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                float neighbor_prob = fetch_input(px + i32vec2(x, y));
                prob = max(prob, neighbor_prob);
            }
        }
    }

    imageStore(daxa_image2D(prob_filtered1_img), i32vec2(px), f32vec4(prob, 0, 0, 0));
}

#endif

#if TAA_PROB_FILTER2_COMPUTE

f32 fetch_input(i32vec2 px) {
    return texelFetch(daxa_texture2D(prob_filtered1_img), px, 0).r;
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    i32vec2 px = i32vec2(gl_GlobalInvocationID.xy);
    float prob = fetch_input(px);

    f32vec2 weighted_prob = f32vec2(0);
    const float SQUISH_STRENGTH = 10;

    const int k = 2;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float neighbor_prob = fetch_input(px + i32vec2(x, y) * 2);
            weighted_prob += f32vec2(exponential_squish(neighbor_prob, SQUISH_STRENGTH), 1);
        }
    }

    prob = exponential_unsquish(weighted_prob.x / weighted_prob.y, SQUISH_STRENGTH);

    // prob = min(prob, WaveReadLaneAt(prob, WaveGetLaneIndex() ^ 1));
    // prob = min(prob, WaveReadLaneAt(prob, WaveGetLaneIndex() ^ 8));

    imageStore(daxa_image2D(prob_filtered2_img), i32vec2(px), f32vec4(prob, 0, 0, 0));
}

#endif

#if TAA_COMPUTE

// Apply at spatial kernel to the current frame, "un-jittering" it.
#define FILTER_CURRENT_FRAME 1

#define USE_ACCUMULATION 1
#define RESET_ACCUMULATION 0
#define USE_NEIGHBORHOOD_CLAMPING 1
#define TARGET_SAMPLE_COUNT 8

// If 1, outputs the input verbatim
// if N > 1, exponentially blends approximately N frames together without any clamping
#define SHORT_CIRCUIT 0

// Whether to use the input probability calculated in `input_prob.hlsl` and the subsequent filters.
// Necessary for stability of temporal super-resolution.
#define USE_CONFIDENCE_BASED_HISTORY_BLEND 1

#define INPUT_TEX input_image
#define INPUT_REMAP InputRemap

// Draw a rectangle indicating the current frame index. Useful for debugging frame drops.
#define USE_FRAME_INDEX_INDICATOR_BAR 0

// struct InputRemap {
// };
// static InputRemap create() {
//     InputRemap res;
//     return res;
// }

f32vec4 InputRemap_remap(f32vec4 v) {
    return f32vec4(sRGB_to_YCbCr(decode_rgb(v.rgb)), 1);
}
f32vec4 fetch_history(i32vec2 px) {
    return texelFetch(daxa_texture2D(reprojected_history_img), px, 0);
}

f32vec4 fetch_blurred_history(i32vec2 px, int k, float sigma) {
    const f32vec3 center = fetch_history(px).rgb;

    f32vec4 csum = f32vec4(0.0);
    float wsum = 0;

    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            f32vec4 c = fetch_history(px + i32vec2(x, y));
            f32vec2 offset = f32vec2(x, y) * sigma;
            float w = exp(-dot(offset, offset));
            float color_diff =
                linear_to_perceptual(sRGB_to_luminance(c.rgb)) - linear_to_perceptual(sRGB_to_luminance(center));
            csum += c * w;
            wsum += w;
        }
    }

    return csum / wsum;
}

// struct HistoryRemap {
// };
// static HistoryRemap create() {
//     HistoryRemap res;
//     return res;
// }

f32vec4 HistoryRemap_remap(f32vec4 v) {
    return f32vec4(sRGB_to_YCbCr(v.rgb), 1);
}

struct UnjitteredSampleInfo {
    f32vec4 color;
    float coverage;
    f32vec3 ex;
    f32vec3 ex2;
};
struct UnjitterSettings {
    float kernel_scale;
    int kernel_half_width_pixels;
};

#define REMAP_FUNC HistoryRemap_remap
UnjitteredSampleInfo sample_image_unjitter_taa(
    daxa_ImageViewId img,
    i32vec2 output_px,
    f32vec2 output_tex_size,
    f32vec2 sample_offset_pixels,
    UnjitterSettings settings) {
    const f32vec2 input_tex_size = push.input_tex_size.xy; // f32vec2(img.size());
    const f32vec2 input_resolution_scale = input_tex_size / output_tex_size;
    const i32vec2 base_src_px = i32vec2((output_px + 0.5) * input_resolution_scale);

    // In pixel units of the destination (upsampled)
    const f32vec2 dst_sample_loc = output_px + 0.5;
    const f32vec2 base_src_sample_loc =
        (base_src_px + 0.5 + sample_offset_pixels * f32vec2(1, -1)) / input_resolution_scale;

    f32vec4 res = f32vec4(0.0);
    f32vec3 ex = f32vec3(0.0);
    f32vec3 ex2 = f32vec3(0.0);
    float dev_wt_sum = 0.0;
    float wt_sum = 0.0;

    // Stretch the kernel if samples become too sparse due to drastic upsampling
    // const float kernel_distance_mult = min(1.0, 1.2 * input_resolution_scale.x);

    const float kernel_distance_mult = 1.0 * settings.kernel_scale;
    // const float kernel_distance_mult = 0.3333 / 2;

    int k = settings.kernel_half_width_pixels;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            i32vec2 src_px = base_src_px + i32vec2(x, y);
            f32vec2 src_sample_loc = base_src_sample_loc + f32vec2(x, y) / input_resolution_scale;

            f32vec4 col = REMAP_FUNC(texelFetch(daxa_texture2D(img), src_px, 0));
            f32vec2 sample_center_offset = (src_sample_loc - dst_sample_loc) * kernel_distance_mult;

            float dist2 = dot(sample_center_offset, sample_center_offset);
            float dist = sqrt(dist2);

            // float wt = all(abs(sample_center_offset) < 0.83);//dist < 0.33;
            float dev_wt = exp2(-dist2 * input_resolution_scale.x);
            // float wt = mitchell_netravali(2.5 * dist * input_resolution_scale.x);
            float wt = exp2(-10 * dist2 * input_resolution_scale.x);
            // float wt = sinc(1 * dist * input_resolution_scale.x) * smoothstep(3, 0, dist * input_resolution_scale.x);
            // float wt = lanczos(2.2 * dist * input_resolution_scale.x, 3);
            // wt = max(wt, 0.0);

            res += col * wt;
            wt_sum += wt;

            ex += col.xyz * dev_wt;
            ex2 += col.xyz * col.xyz * dev_wt;
            dev_wt_sum += dev_wt;
        }
    }

    f32vec2 sample_center_offset = -sample_offset_pixels / input_resolution_scale * f32vec2(1, -1) - (base_src_sample_loc - dst_sample_loc);

    UnjitteredSampleInfo info;
    info.color = res;
    info.coverage = wt_sum;
    info.ex = ex / dev_wt_sum;
    info.ex2 = ex2 / dev_wt_sum;
    return info;
}
#undef REMAP_FUNC

f32vec4 fetch_reproj(i32vec2 px) {
    return texelFetch(daxa_texture2D(reprojection_map), px, 0);
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    i32vec2 px = i32vec2(gl_GlobalInvocationID.xy);
#if USE_FRAME_INDEX_INDICATOR_BAR
    if (px.y < 50) {
        f32vec4 val = 0;
        if (px.x < frame_constants.frame_index * 10 % uint(output_tex_size.x)) {
            val = 1;
        }
        temporal_output_tex[px] = val;
        output_tex[px] = val;
        return;
    }
#endif

    const f32vec2 input_resolution_fraction = push.input_tex_size.xy / push.output_tex_size.xy;
    const u32vec2 reproj_px = u32vec2((f32vec2(px) + 0.5) * input_resolution_fraction);
    // const u32vec2 reproj_px = u32vec2(px * input_resolution_fraction + 0.5);

#if SHORT_CIRCUIT
    temporal_output_tex[px] = mix(input_tex[reproj_px], f32vec4(encode_rgb(fetch_history(px).rgb), 1), 1.0 - 1.0 / SHORT_CIRCUIT);
    output_tex[px] = temporal_output_tex[px];
    return;
#endif

    f32vec2 uv = get_uv(px, push.output_tex_size);

    f32vec4 history_packed = fetch_history(px);
    f32vec3 history = history_packed.rgb;
    float history_coverage = max(0.0, history_packed.a);

    f32vec4 bhistory_packed = fetch_blurred_history(px, 2, 1);
    f32vec3 bhistory = bhistory_packed.rgb;
    f32vec3 bhistory_coverage = f32vec3(bhistory_packed.a);

    history = sRGB_to_YCbCr(history);
    bhistory = sRGB_to_YCbCr(bhistory);

    const f32vec4 reproj = fetch_reproj(i32vec2(reproj_px));
    const f32vec2 reproj_xy = texelFetch(daxa_texture2D(closest_velocity_img), px, 0).xy;

    UnjitterSettings unjitter_settings;
    unjitter_settings.kernel_scale = 1;
    unjitter_settings.kernel_half_width_pixels = 1;
    UnjitteredSampleInfo center_sample = sample_image_unjitter_taa(
        INPUT_TEX,
        px,
        push.output_tex_size.xy,
        SAMPLE_OFFSET_PIXELS,
        unjitter_settings);
    unjitter_settings.kernel_scale = 0.333;
    UnjitteredSampleInfo bcenter_sample = sample_image_unjitter_taa(
        INPUT_TEX,
        px,
        push.output_tex_size.xy,
        SAMPLE_OFFSET_PIXELS,
        unjitter_settings);

    float coverage = 1;
#if FILTER_CURRENT_FRAME
    f32vec3 center = center_sample.color.rgb;
    coverage = center_sample.coverage;
#else
    f32vec3 center = sRGB_to_YCbCr(decode_rgb(texelFetch(daxa_texture2D(INPUT_TEX), px, 0).rgb));
#endif

    f32vec3 bcenter = bcenter_sample.color.rgb / bcenter_sample.coverage;

    history = mix(history, bcenter, clamp(1.0 - history_coverage, 0.0, 1.0));
    bhistory = mix(bhistory, bcenter, clamp(1.0 - bhistory_coverage, 0.0, 1.0));

    const float input_prob = texelFetch(daxa_texture2D(input_prob_img), i32vec2(reproj_px), 0).r;

    f32vec3 ex = center_sample.ex;
    f32vec3 ex2 = center_sample.ex2;
    const f32vec3 var = max(0.0.xxx, ex2 - ex * ex);

    const f32vec3 prev_var = f32vec3(textureLod(daxa_sampler2D(smooth_var_history_tex, deref(gpu_input).sampler_lnc), uv + reproj_xy, 0).x);

    // TODO: factor-out camera-only velocity
    const f32vec2 vel_now = texelFetch(daxa_texture2D(closest_velocity_img), px, 0).xy / deref(gpu_input).delta_time;
    const f32vec2 vel_prev = textureLod(daxa_sampler2D(velocity_history_tex, deref(gpu_input).sampler_llc), uv + texelFetch(daxa_texture2D(closest_velocity_img), px, 0).xy, 0).xy;
    const float vel_diff = length((vel_now - vel_prev) / max(f32vec2(1.0), abs(vel_now + vel_prev)));
    const float var_blend = clamp(0.3 + 0.7 * (1 - reproj.z) + vel_diff, 0.0, 1.0);

    f32vec3 smooth_var = max(var, mix(prev_var, var, var_blend));

    const float var_prob_blend = clamp(input_prob, 0.0, 1.0);
    smooth_var = mix(var, smooth_var, var_prob_blend);

    const f32vec3 input_dev = sqrt(var);

    f32vec4 this_frame_result = f32vec4(0.0);
#define DEBUG_SHOW(value) \
    { this_frame_result = f32vec4((f32vec3)(value), 1); }

    f32vec3 clamped_history;

    // Perform neighborhood clamping / disocclusion rejection
    {
        // Use a narrow color bounding box to avoid disocclusions
        float box_n_deviations = 0.8;

        if (USE_CONFIDENCE_BASED_HISTORY_BLEND != 0) {
            // Expand the box based on input confidence.
            box_n_deviations = mix(box_n_deviations, 3, input_prob);
        }

        f32vec3 nmin = ex - input_dev * box_n_deviations;
        f32vec3 nmax = ex + input_dev * box_n_deviations;

#if USE_ACCUMULATION
#if USE_NEIGHBORHOOD_CLAMPING
        f32vec3 clamped_bhistory = clamp(bhistory, nmin, nmax);
#else
        f32vec3 clamped_bhistory = bhistory;
#endif

        const float clamping_event = length(max(f32vec3(0.0), max(bhistory - nmax, nmin - bhistory)) / max(f32vec3(0.01), ex));

        f32vec3 outlier3 = max(f32vec3(0.0), (max(nmin - history, history - nmax)) / (0.1 + max(max(abs(history), abs(ex)), 1e-5)));
        f32vec3 boutlier3 = max(f32vec3(0.0), (max(nmin - bhistory, bhistory - nmax)) / (0.1 + max(max(abs(bhistory), abs(ex)), 1e-5)));

        // Temporal outliers in sharp history
        float outlier = max(outlier3.x, max(outlier3.y, outlier3.z));
        // DEBUG_SHOW(outlier);

        // Temporal outliers in blurry history
        float boutlier = max(boutlier3.x, max(boutlier3.y, boutlier3.z));
        // DEBUG_SHOW(boutlier);

        const bool history_valid = all(bvec2(uv + reproj_xy == clamp(uv + reproj_xy, f32vec2(0.0), f32vec2(1.0))));

#if 1
        if (history_valid) {
            const float non_disoccluding_outliers = max(0.0, outlier - boutlier) * 10;
            // DEBUG_SHOW(non_disoccluding_outliers);

            const f32vec3 unclamped_history_detail = history - clamped_bhistory;

            // Temporal luminance diff, containing history edges, and peaking when
            // clamping happens.
            const float temporal_clamping_detail = length(unclamped_history_detail.x / max(1e-3, input_dev.x)) * 0.05;
            // DEBUG_SHOW(temporal_clamping_detail);

            // Close to 1.0 when temporal clamping is relatively low. Close to 0.0 when disocclusions happen.
            const float temporal_stability = clamp(1 - temporal_clamping_detail, 0.0, 1.0);
            // DEBUG_SHOW(temporal_stability);

            const float allow_unclamped_detail = clamp(non_disoccluding_outliers, 0.0, 1.0) * temporal_stability;
            // const float allow_unclamped_detail = saturate(non_disoccluding_outliers * exp2(-length(input_tex_size.xy * reproj_xy))) * temporal_stability;
            // DEBUG_SHOW(allow_unclamped_detail);

            // Clamping happens to blurry history because input is at lower fidelity (and potentially lower resolution)
            // than history (we don't have enough data to perform good clamping of high frequencies).
            // In order to keep high-resolution detail in the output, the high-frequency content is split from
            // low-frequency (`bhistory`), and then selectively re-added. The detail needs to be attenuated
            // in order not to cause false detail (which look like excessive sharpening artifacts).
            f32vec3 history_detail = history - bhistory;

            // Selectively stabilize some detail, allowing unclamped history
            history_detail = mix(history_detail, unclamped_history_detail, allow_unclamped_detail);

            // 0..1 value of how much clamping initially happened in the blurry history
            const float initial_bclamp_amount = clamp(dot(
                                                          clamped_bhistory - bhistory, bcenter - bhistory) /
                                                          max(1e-5, length(clamped_bhistory - bhistory) * length(bcenter - bhistory)),
                                                      0.0, 1.0);

            // Ditto, after adjusting for `allow_unclamped_detail`
            const float effective_clamp_amount = clamp(initial_bclamp_amount, 0.0, 1.0) * (1 - allow_unclamped_detail);
            // DEBUG_SHOW(effective_clamp_amount);

            // Where clamping happened to the blurry history, also remove the detail (history-bhistory)
            const float keep_detail = 1 - effective_clamp_amount;
            history_detail *= keep_detail;

            // Finally, construct the full-frequency output.
            clamped_history = clamped_bhistory + history_detail;

#if 1
            // TODO: figure out how not to over-do this with temporal super-resolution
            if (input_resolution_fraction.x < 1.0) {
                // When temporally upsampling, after a clamping event, there's pixellation
                // because we haven't accumulated enough samples yet from
                // the reduced-resolution input. Dampening history coverage when
                // clamping happens allows us to boost this convergence.

                history_coverage *= mix(
                    mix(0.0, 0.9, keep_detail), 1.0, clamp(10 * clamping_event, 0.0, 1.0));
            }
#endif
        } else {
            clamped_history = clamped_bhistory;
            coverage = 1;
            center = bcenter;
            history_coverage = 0;
        }
#else
        clamped_history = clamp(history, nmin, nmax);
#endif

        if (USE_CONFIDENCE_BASED_HISTORY_BLEND != 0) {
            // If input confidence is high, blend in unclamped history.
            clamped_history = mix(
                clamped_history,
                history,
                smoothstep(0.5, 1.0, input_prob));
        }
    }

#if RESET_ACCUMULATION
    history_coverage = 0;
#endif

    float total_coverage = max(1e-5, history_coverage + coverage);
    f32vec3 temporal_result = (clamped_history * history_coverage + center) / total_coverage;

    const float max_coverage = max(2, TARGET_SAMPLE_COUNT / (input_resolution_fraction.x * input_resolution_fraction.y));

    total_coverage = min(max_coverage, total_coverage);

    coverage = total_coverage;
#else
        f32vec3 temporal_result = center / coverage;
#endif
    imageStore(daxa_image2D(smooth_var_output_tex), px, f32vec4(smooth_var, 0.0));

    temporal_result = YCbCr_to_sRGB(temporal_result);
    temporal_result = encode_rgb(temporal_result);
    temporal_result = max(f32vec3(0.0), temporal_result);

    this_frame_result.rgb = mix(temporal_result, this_frame_result.rgb, this_frame_result.a);

    imageStore(daxa_image2D(temporal_output_tex), px, f32vec4(temporal_result, coverage));
    imageStore(daxa_image2D(this_frame_output_img), px, this_frame_result);

    f32vec2 vel_out = reproj_xy;
    float vel_out_depth = 0;

    // It's critical that this uses the closest depth since it's compared to closest depth
    imageStore(daxa_image2D(temporal_velocity_output_tex), px, f32vec4(texelFetch(daxa_texture2D(closest_velocity_img), px, 0).xy / deref(gpu_input).delta_time, 0.0, 0.0));
}

#endif
