#pragma once

u32 bitpacking_create_mask(in u32 X, in u32 BIT_N) {
    u32 MASK = 0;
    for (u32 i = 0; i < BIT_N; ++i)
        MASK |= (1u << i);
    MASK = MASK << X;
    return MASK;
}

#define BITPACKING_MAP_UNMAP_RANGE_TEMPLATE(VectorT)  \
    VectorT bitpacking_map_range(                     \
        in VectorT min_val,                           \
        in VectorT max_val,                           \
        in VectorT val) {                             \
        return (val - min_val) / (max_val - min_val); \
    }                                                 \
    VectorT bitpacking_unmap_range(                   \
        in VectorT min_val,                           \
        in VectorT max_val,                           \
        in VectorT val) {                             \
        return val * (max_val - min_val) + min_val;   \
    }
BITPACKING_MAP_UNMAP_RANGE_TEMPLATE(f32)
BITPACKING_MAP_UNMAP_RANGE_TEMPLATE(f32vec2)
BITPACKING_MAP_UNMAP_RANGE_TEMPLATE(f32vec3)
BITPACKING_MAP_UNMAP_RANGE_TEMPLATE(f32vec4)

u32 bitpacking_pack_f32(in u32 X, in u32 BIT_N, in u32 packed_data, in f32 value) {
    value = sqrt(value);
    u32 MASK = ~(bitpacking_create_mask(X, BIT_N));
    packed_data &= MASK;
    packed_data |= u32(value * ((1u << BIT_N) - 1)) << X;
    return packed_data;
}
f32 bitpacking_unpack_f32(in u32 X, in u32 BIT_N, in u32 packed_data) {
    f32 value;
    u32 MASK = bitpacking_create_mask(X, BIT_N);
    value = ((packed_data & MASK) >> X) * 1.0 / ((1u << BIT_N) - 1);
    return value * value;
}
u32 bitpacking_pack_f32vec2(in u32 X, in u32 BIT_N, in u32 packed_data, in f32vec2 value) {
    packed_data = bitpacking_pack_f32(X + BIT_N * 0, BIT_N, packed_data, value.x);
    packed_data = bitpacking_pack_f32(X + BIT_N * 1, BIT_N, packed_data, value.y);
    return packed_data;
}
f32vec2 bitpacking_unpack_f32vec2(in u32 X, in u32 BIT_N, in u32 packed_data) {
    f32vec2 value;
    value.x = bitpacking_unpack_f32(X + BIT_N * 0, BIT_N, packed_data);
    value.y = bitpacking_unpack_f32(X + BIT_N * 1, BIT_N, packed_data);
    return value;
}
u32 bitpacking_pack_f32vec3(in u32 X, in u32 BIT_N, in u32 packed_data, in f32vec3 value) {
    packed_data = bitpacking_pack_f32(X + BIT_N * 0, BIT_N, packed_data, value.x);
    packed_data = bitpacking_pack_f32(X + BIT_N * 1, BIT_N, packed_data, value.y);
    packed_data = bitpacking_pack_f32(X + BIT_N * 2, BIT_N, packed_data, value.z);
    return packed_data;
}
f32vec3 bitpacking_unpack_f32vec3(in u32 X, in u32 BIT_N, in u32 packed_data) {
    f32vec3 value;
    value.x = bitpacking_unpack_f32(X + BIT_N * 0, BIT_N, packed_data);
    value.y = bitpacking_unpack_f32(X + BIT_N * 1, BIT_N, packed_data);
    value.z = bitpacking_unpack_f32(X + BIT_N * 2, BIT_N, packed_data);
    return value;
}
u32 bitpacking_pack_f32vec4(in u32 X, in u32 BIT_N, in u32 packed_data, in f32vec4 value) {
    packed_data = bitpacking_pack_f32(X + BIT_N * 0, BIT_N, packed_data, value.x);
    packed_data = bitpacking_pack_f32(X + BIT_N * 1, BIT_N, packed_data, value.y);
    packed_data = bitpacking_pack_f32(X + BIT_N * 2, BIT_N, packed_data, value.z);
    packed_data = bitpacking_pack_f32(X + BIT_N * 3, BIT_N, packed_data, value.w);
    return packed_data;
}
f32vec4 bitpacking_unpack_f32vec4(in u32 X, in u32 BIT_N, in u32 packed_data) {
    f32vec4 value;
    value.x = bitpacking_unpack_f32(X + BIT_N * 0, BIT_N, packed_data);
    value.y = bitpacking_unpack_f32(X + BIT_N * 1, BIT_N, packed_data);
    value.z = bitpacking_unpack_f32(X + BIT_N * 2, BIT_N, packed_data);
    value.w = bitpacking_unpack_f32(X + BIT_N * 3, BIT_N, packed_data);
    return value;
}

u32 bitpacking_pack_u32(in u32 X, in u32 BIT_N, in u32 packed_data, in u32 value) {
    u32 MASK = bitpacking_create_mask(X, BIT_N);
    packed_data &= ~MASK;
    packed_data |= (value << X) & MASK;
    return packed_data;
}
u32 bitpacking_unpack_u32(in u32 X, in u32 BIT_N, in u32 packed_data) {
    u32 value;
    u32 MASK = bitpacking_create_mask(X, BIT_N);
    value = (packed_data & MASK) >> X;
    return value;
}
