#pragma once

#include <utils/math.glsl>

struct TreeSDF {
    f32 wood;
    f32 leaves;
};

void sd_branch(in out TreeSDF val, in f32vec3 p, in f32vec3 origin, in f32vec3 dir, in f32 scl) {
    f32vec3 bp0 = origin;
    f32vec3 bp1 = bp0 + dir + f32vec3(0, 0, 0.5);

    f32 r = rand(p + bp0) * 1;
    val.wood = min(val.wood, sd_capsule(p, bp0, bp1, 0.15));
    val.leaves = min(val.leaves, sd_sphere(p - bp1, 1.25 * scl) + r * 35);

    bp0 = bp1, bp1 = bp0 + dir + f32vec3(0, 0, 1.2);
    val.wood = min(val.wood, sd_capsule(p, bp0, bp1, 0.12));
    f32 sp = sd_sphere(p - (bp1 * 0.85 + bp0 * 0.15), 0.85 * scl);
    val.leaves = min(val.leaves, sp + r * 35);
}

TreeSDF sd_tree(in f32vec3 p, in f32vec3 seed) {
    TreeSDF val = TreeSDF(MAX_SD, MAX_SD);
    f32vec3 p0 = f32vec3(0, 0, 0);
    f32vec3 p1 = f32vec3(0, 0, 0);
    f32 radius = 0.6;
    for (u32 i = 0; i < 32; ++i) {
        f32 angle = 2.0 * PI * rand(seed + p0);
        p0 = p1;
        p1 += f32vec3(0, 0, 0.5) + f32vec3(cos(angle), sin(angle), +0.0) * 0.1;
        f32 prev_radius = radius;
        radius -= 0.012;
        val.wood = min(val.wood, sd_round_cone(p, p0, p1, prev_radius, radius));
        if (i > 12) {
            sd_branch(val, p, p1, normalize(f32vec3(cos(angle), sin(angle), +0.0)) * radius * 120 / (i + 1), 2);
        }
    }
    return val;
}

#if defined(CHUNKGEN)
#define OFFSET (f32vec3(32, 32, 0) * 0.5 / VOXEL_SCL)
#else
#define OFFSET f32vec3(0, 0, 0)
#endif

b32 custom_brush_should_edit(in BrushInput brush) {
    TreeSDF tree_sdf = sd_tree(brush.p - OFFSET, brush.origin);
    return min(tree_sdf.wood, tree_sdf.leaves) < 0.0;
}

u32 custom_brush_id_kernel(in BrushInput brush) {
    TreeSDF tree_sdf = sd_tree(brush.p - OFFSET, brush.origin);
    if (tree_sdf.wood < tree_sdf.leaves) {
        return BlockID_Log;
    } else {
        return BlockID_Leaves;
    }
}

Voxel custom_brush_kernel(in BrushInput brush) {
    Voxel result;
    result.block_id = custom_brush_id_kernel(brush);
    if (result.block_id == BlockID_Leaves) {
        f32 r = rand(brush.origin) * 0.1 + 0.01 + rand(brush.p) * 0.05;
        result.col = hsv2rgb(f32vec3(r, 0.99, 0.8));
    } else {
        result.col = block_color(result.block_id);
    }
    return result;
}
