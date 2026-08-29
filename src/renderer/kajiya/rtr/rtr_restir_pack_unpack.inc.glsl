#ifndef RENDERER_KAJIYA_RTR_RTR_RESTIR_PACK_UNPACK_INC_GLSL
#define RENDERER_KAJIYA_RTR_RTR_RESTIR_PACK_UNPACK_INC_GLSL

struct RtrRestirRayOrigin {
    vec3 ray_origin_eye_offset_ws;
    float roughness;
    uint frame_index_mod4;
};

RtrRestirRayOrigin RtrRestirRayOrigin_from_raw(vec4 raw) {
    RtrRestirRayOrigin res;
    res.ray_origin_eye_offset_ws = raw.xyz;

    vec2 misc = unpackHalf2x16(floatBitsToUint(raw.w));
    res.roughness = misc.x;
    res.frame_index_mod4 = uint(misc.y) & 3;
    return res;
}

vec4 to_raw(RtrRestirRayOrigin self) {
    return vec4(
        self.ray_origin_eye_offset_ws,
        uintBitsToFloat(packHalf2x16(vec2(self.roughness, self.frame_index_mod4))));
}

#endif // RENDERER_KAJIYA_RTR_RTR_RESTIR_PACK_UNPACK_INC_GLSL
