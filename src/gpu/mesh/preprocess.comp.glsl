#include <shared/utils/mesh_model.inl>

DAXA_DECL_PUSH_CONSTANT(MeshPreprocessPush, daxa_push_constant)
#define VERTS(i) deref(daxa_push_constant.vertex_buffer[i])
#define NORMALS(i) deref(daxa_push_constant.normal_buffer[i])
#define INPUT deref(daxa_push_constant.gpu_input)
#define VOXELS(i) deref(daxa_push_constant.voxel_buffer[i])

vec3 map_range(vec3 x, vec3 domain_min, vec3 domain_max, vec3 range_min, vec3 range_max) {
    vec3 domain_size = domain_max - domain_min;
    vec3 range_size = range_max - range_min;
    x = (x - domain_min) / domain_size; // x is 0 to 1
    x = (x * range_size) + range_min;
    return x;
}
void rescale_pos(inout vec3 pos) {
    vec3 bound_min = INPUT.bound_min;
    vec3 bound_max = INPUT.bound_max;
    vec3 bound_range = bound_max - bound_min;
    float max_extent = max(bound_range.x, max(bound_range.y, bound_range.z));
    vec3 bound_c = (bound_min + bound_max) * 0.5;
    bound_min = bound_c - max_extent * 0.5;
    bound_max = bound_c + max_extent * 0.5;
    // bound_max = bound_min + max_extent;
    pos = map_range(pos, bound_min, bound_max, vec3(-1), vec3(1));
}

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint tri_i = gl_GlobalInvocationID.x;
    if (tri_i >= daxa_push_constant.triangle_count) {
        return;
    }
    uint vert_off = tri_i * 3;
    vec3 pos[3];
    pos[0] = (daxa_push_constant.modl_mat * vec4(VERTS(vert_off + 0).pos, 1)).xyz;
    pos[1] = (daxa_push_constant.modl_mat * vec4(VERTS(vert_off + 1).pos, 1)).xyz;
    pos[2] = (daxa_push_constant.modl_mat * vec4(VERTS(vert_off + 2).pos, 1)).xyz;

    rescale_pos(pos[0]);
    rescale_pos(pos[1]);
    rescale_pos(pos[2]);

    vec3 del_a = pos[1] - pos[0];
    vec3 del_b = pos[2] - pos[0];
    vec3 nrm = normalize(cross(del_a, del_b));

    float dx = abs(dot(nrm, vec3(1, 0, 0)));
    float dy = abs(dot(nrm, vec3(0, 1, 0)));
    float dz = abs(dot(nrm, vec3(0, 0, 1)));

    uint side = 0;
    if (dx > dy) {
        if (dx > dz) {
            side = 0;
        } else {
            side = 2;
        }
    } else {
        if (dy > dz) {
            side = 1;
        } else {
            side = 2;
        }
    }
    VERTS(vert_off + 0).rotation = side;
    VERTS(vert_off + 1).rotation = side;
    VERTS(vert_off + 2).rotation = side;

    switch (side) {
    case 0:
        VERTS(vert_off + 0).pos = pos[0].zyx;
        VERTS(vert_off + 1).pos = pos[1].zyx;
        VERTS(vert_off + 2).pos = pos[2].zyx;
        break;
    case 1:
        VERTS(vert_off + 0).pos = pos[0].xzy;
        VERTS(vert_off + 1).pos = pos[1].xzy;
        VERTS(vert_off + 2).pos = pos[2].xzy;
        break;
    case 2:
        VERTS(vert_off + 0).pos = pos[0].xyz;
        VERTS(vert_off + 1).pos = pos[1].xyz;
        VERTS(vert_off + 2).pos = pos[2].xyz;
        break;
    }

    NORMALS(tri_i) = -nrm;
}
