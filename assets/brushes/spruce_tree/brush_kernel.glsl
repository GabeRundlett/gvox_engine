#pragma once

struct TreeSDF {
    f32 wood;
    f32 leaves;
};

void sd_branch(in out TreeSDF val, in f32vec3 p, in f32vec3 origin, in f32vec3 dir, in f32 scl) {
    f32vec3 bp0 = origin;
    f32vec3 bp1 = bp0 + dir;
    val.wood = min(val.wood, sd_capsule(p, bp0, bp1, 0.10));
    val.leaves = min(val.leaves, sd_sphere(p - bp1, 0.15 * scl));
    bp0 = bp1, bp1 = bp0 + dir * 0.5 + f32vec3(0, 0, 0.2);
    val.wood = min(val.wood, sd_capsule(p, bp0, bp1, 0.07));
    val.leaves = min(val.leaves, sd_sphere(p - bp1, 0.15 * scl));
}

TreeSDF sd_spruce_tree(in f32vec3 p, in f32vec3 seed) {
    TreeSDF val = TreeSDF(MAX_SD, MAX_SD);
    val.wood = min(val.wood, sd_capsule(p, f32vec3(0, 0, 0), f32vec3(0, 0, 4.5), 0.15));
    val.leaves = min(val.leaves, sd_capsule(p, f32vec3(0, 0, 4.5), f32vec3(0, 0, 5.0), 0.15));
    for (u32 i = 0; i < 5; ++i) {
        f32 scl = 1.0 / (1.0 + i * 0.5);
        f32 scl2 = 1.0 / (1.0 + i * 0.1);
        u32 branch_n = 8 - i;
        for (u32 branch_i = 0; branch_i < branch_n; ++branch_i) {
            f32 angle = (1.0 / branch_n * branch_i) * 2.0 * PI + rand(seed + i + 1.0 * branch_i) * 0.5;
            sd_branch(val, p, f32vec3(0, 0, 1.0 + i * 0.8) * 1.0, normalize(f32vec3(cos(angle), sin(angle), +0.0)) * scl, scl2 * 1.5);
        }
    }
    return val;
}

b32 custom_brush_should_edit(in BrushInput brush) {
    TreeSDF tree_sdf = sd_spruce_tree(brush.p, brush.origin);
    return min(tree_sdf.wood, tree_sdf.leaves) < 0.0;
}

u32 custom_brush_id_kernel(in BrushInput brush) {
    TreeSDF tree_sdf = sd_spruce_tree(brush.p, brush.origin);
    TreeSDF slope_t0 = sd_spruce_tree(brush.p + f32vec3(1, 0, 0) / VOXEL_SCL * 0.01, brush.origin);
    TreeSDF slope_t1 = sd_spruce_tree(brush.p + f32vec3(0, 1, 0) / VOXEL_SCL * 0.01, brush.origin);
    TreeSDF slope_t2 = sd_spruce_tree(brush.p + f32vec3(0, 0, 1) / VOXEL_SCL * 0.01, brush.origin);
    f32vec3 leaves_nrm = normalize(f32vec3(slope_t0.leaves, slope_t1.leaves, slope_t2.leaves) - tree_sdf.leaves);
    f32 leaves_upwards = max(dot(leaves_nrm, f32vec3(0, 0, 1)), 0);
    if (tree_sdf.wood < tree_sdf.leaves) {
        return BlockID_Log;
    } else {
        if (leaves_upwards > 0.5) {
            return BlockID_Snow;
        } else {
            return BlockID_Leaves;
        }
    }
}

Voxel custom_brush_kernel(in BrushInput brush) {
    u32 result_id = custom_brush_id_kernel(brush);
    return Voxel(block_color(result_id), result_id);
}
