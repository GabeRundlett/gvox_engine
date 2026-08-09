#include "animation_playground.inl"
#include <voxels/pack_unpack.inl>

DAXA_DECL_PUSH_CONSTANT(AnimationPlaygroundGenPush, push)

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#define PI 3.14159265359
#define HALFPI 1.57079632679

float sdCircle(vec2 p, float r) {
    return length(p) - r;
}

float smin(float d1, float d2, float k) {
    k *= 4.0;
    float h = max(k - abs(d1 - d2), 0.0);
    return min(d1, d2) - h * h * 0.25 / k;
}

float dot2(in vec2 v) { return dot(v, v); }
float sdRoundCone(vec2 p, vec2 a, vec2 b, float r1, float r2) { // from tdXGWr, only converted to 2D
    // sampling independent computations (only depend on shape)
    vec2 ba = b - a;
    float l2 = dot(ba, ba);
    float rr = r1 - r2;
    float a2 = l2 - rr * rr;
    float il2 = 1.0 / l2;

    // sampling dependant computations
    vec2 pa = p - a;
    float y = dot(pa, ba);
    float z = y - l2;
    float x2 = dot2(pa * l2 - ba * y);
    float y2 = y * y * l2;
    float z2 = z * z * l2;

    // single square root!
    float k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k)
        return sqrt(x2 + z2) * il2 - r2;
    if (sign(y) * a2 * y2 < k)
        return sqrt(x2 + y2) * il2 - r1;
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

float sdSegment(in vec2 p, in vec2 a, in vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

// parametric circle
vec2 circle(in float angle, in float r) {
    float t = angle * PI;
    return r * vec2(cos(t), sin(t));
}

float nextBranchAngle(in float angle_range, in float branch_count, // these should be constant over a tree
                      in float branch_id, in float last_branch_orientation) {
    return last_branch_orientation + angle_range - (2. * angle_range * branch_id / (branch_count - 1.));
}

#define BRANCH_COUNT 3

float sdTree(in vec2 p, in int dep, float iTime) {
    const int branch_count = BRANCH_COUNT;
    const float fbranch_count = float(BRANCH_COUNT);
    float angle_range = cos(iTime / 2.);
    // float angle_range = 2./3.; //cool pattern with 3 branches
    const float thickness = .2;
    float initial_d = length(p); // (; https://www.youtube.com/watch?v=dv13gl0a-FA

    float depth = 1.; // additional float accumulator because int -> float castings are slow af and float for loops are no no

    float d = initial_d - thickness;
    vec2 current_pos = vec2(0.);
    float current_orientation = .5;

    vec2 working_pos = vec2(0.);
    float working_orientation = .0;
    float working_d = 0.;

    vec2 chosen_pos = current_pos;
    float chosen_orientation = .5;
    float chosen_d = 0.;

    for (int idepth = 0; idepth < dep; idepth++) {
        // by putting a condition to pseudorandomly skip iterations based on branch ID
        // you could get more realistic looking trees where branches aren't all the same length

        // this section chooses the position of the next branch on the tree
        chosen_orientation = nextBranchAngle(angle_range, fbranch_count, 0., current_orientation);
        chosen_pos = current_pos + circle(chosen_orientation, 1. / (depth));
        chosen_d = sdRoundCone(p, chosen_pos, current_pos + circle(chosen_orientation, .0001 / (depth)), thickness / (depth + 1.), thickness / depth);
        // chosen_d = sdSegment(p, chosen_pos, current_pos + circle(chosen_orientation, .1/(depth))) - thickness/depth;
        // chosen_d = length(p - chosen_pos) - thickness/depth;

        for (int branch = 1; branch <= branch_count - 1; branch++) {
            working_orientation = nextBranchAngle(angle_range, fbranch_count, float(branch), current_orientation);
            working_pos = current_pos + circle(working_orientation, 1. / (depth));
            working_d = sdRoundCone(p, working_pos, current_pos + circle(working_orientation, .0001 / (depth)), thickness / (depth + 1.), thickness / depth);
            // working_d = sdSegment(p, working_pos, current_pos + circle(working_orientation, .1/(depth))) - thickness/depth;
            // working_d = length(p - working_pos) - thickness/depth;

            if (working_d < chosen_d) {
                chosen_orientation = working_orientation;
                chosen_pos = working_pos;
                chosen_d = working_d;
            }
        }

        d = smin(d, chosen_d, 0.03);
        current_pos = chosen_pos;
        current_orientation = chosen_orientation;

        chosen_d = initial_d;
        depth++;
    }

    return d;
}

void main() {
    ivec3 brick_i = ivec3(gl_GlobalInvocationID.xyz);
    int frame_index = brick_i.z / push.grid_dims_bricks.z;
    brick_i.z = brick_i.z % push.grid_dims_bricks.z;

    int bricks_per_frame = push.grid_dims_bricks.x * push.grid_dims_bricks.y * push.grid_dims_bricks.z;
    int brick_index_in_frame = brick_i.x + brick_i.y * push.grid_dims_bricks.x + brick_i.z * push.grid_dims_bricks.x * push.grid_dims_bricks.y;
    int brick_index = frame_index * bricks_per_frame + brick_index_in_frame;

    daxa_RWBufferPtr(BrickPrimitive) brick_ptr = advance(push.bricks, brick_index);
    daxa_RWBufferPtr(VoxelShadingAttribBrick) attribs_ptr = advance(push.brick_attribs, brick_index);

    ivec3 grid_voxel_dims = push.grid_dims_bricks * BRICK_SIZE;
    vec3 grid_center = vec3(grid_voxel_dims) * 0.5;
    float max_radius = float(min(min(grid_voxel_dims.x, grid_voxel_dims.y), grid_voxel_dims.z)) * 0.5;

    for (int z = 0; z < BRICK_SIZE; ++z) {
        for (int y = 0; y < BRICK_SIZE; ++y) {
            uint local_byte = 0;
            for (int x = 0; x < BRICK_SIZE; ++x) {
                ivec3 local_voxel = ivec3(x, y, z);
                ivec3 voxel_i = brick_i * BRICK_SIZE + local_voxel;
                vec3 p = vec3(voxel_i) + 0.5 - grid_center;

                float radius = max_radius * (0.5 + 0.4 * sin(float(frame_index) * 0.8));
                bool solid = length(p) < radius;

                vec3 nrm = normalize(p + vec3(1.0e-5, 0.0, 0.0));
                // vec3 col = nrm * 0.5 + 0.5;
                vec3 col = vec3(1, 0, 0);
                // solid = mandelbulb(p * 0.04, col, frame_index);
                // col = clamp(col, vec3(0), vec3(1));
                solid = sdTree(p.xz * 0.1 + vec2(0, 0.8), 5, float(frame_index) / 8 + 2.5) < 0 && abs(p.y) < 4;

                Voxel voxel = Voxel(col, nrm, 0.6, 0u);
                deref(attribs_ptr).voxels[x + y * BRICK_SIZE + z * BRICK_SIZE * BRICK_SIZE] = pack_voxel(voxel);

                if (solid) {
                    local_byte |= (1u << x);
                }
            }
            deref(brick_ptr).bitmap[z * BRICK_SIZE + y] = uint8_t(local_byte);
        }
    }
}
