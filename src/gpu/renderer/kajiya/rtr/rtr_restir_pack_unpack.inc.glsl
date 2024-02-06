struct RtrRestirRayOrigin {
    vec3 ray_origin_eye_offset_ws;
    float roughness;
    uint frame_index_mod4;

    static RtrRestirRayOrigin from_raw(vec4 raw) {
        RtrRestirRayOrigin res;
        res.ray_origin_eye_offset_ws = raw.xyz;

        vec2 misc = unpack_2x16f_uint(asuint(raw.w));
        res.roughness = misc.x;
        res.frame_index_mod4 = uint(misc.y) & 3;
        return res;
    }

    vec4 to_raw() {
        return vec4(
            ray_origin_eye_offset_ws,
            asfloat(pack_2x16f_uint(vec2(roughness, frame_index_mod4)))
        );
    }
};
