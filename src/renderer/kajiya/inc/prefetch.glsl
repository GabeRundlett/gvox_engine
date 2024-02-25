#pragma once

// NOTE(grundlett): Some operations, especially convolutions, will benefit heavily from
// prefetching a range of values. I add this header to make such operations faster.

#if !defined(PREFETCH_RADIUS) || !defined(PREFETCH_GROUP_SIZE)
#error "Bad prefetch configuration"
#endif

#if !defined(ENABLE_PREFETCH)
#define ENABLE_PREFETCH 1
#endif

#if ENABLE_PREFETCH
const uint TILE_DIM = PREFETCH_GROUP_SIZE + PREFETCH_RADIUS * 2;
shared FetchResult tile_input[TILE_DIM * TILE_DIM];
#endif

void do_prefetch() {
#if ENABLE_PREFETCH
    if (gl_LocalInvocationIndex < TILE_DIM * TILE_DIM / 4) {
        const ivec2 anchor = ivec2(gl_WorkGroupID.xy * PREFETCH_GROUP_SIZE) - int(PREFETCH_RADIUS);

        const uint out_index1 = gl_LocalInvocationIndex;
        const uint out_index2 = gl_LocalInvocationIndex + TILE_DIM * TILE_DIM / 4;
        const uint out_index3 = gl_LocalInvocationIndex + TILE_DIM * TILE_DIM / 2;
        const uint out_index4 = gl_LocalInvocationIndex + TILE_DIM * TILE_DIM * 3 / 4;

        const ivec2 coord1 = anchor + ivec2(out_index1 % TILE_DIM, out_index1 / TILE_DIM);
        const ivec2 coord2 = anchor + ivec2(out_index2 % TILE_DIM, out_index2 / TILE_DIM);
        const ivec2 coord3 = anchor + ivec2(out_index3 % TILE_DIM, out_index3 / TILE_DIM);
        const ivec2 coord4 = anchor + ivec2(out_index4 % TILE_DIM, out_index4 / TILE_DIM);

        tile_input[out_index1] = do_fetch(coord1);
        tile_input[out_index2] = do_fetch(coord2);
        tile_input[out_index3] = do_fetch(coord3);
        tile_input[out_index4] = do_fetch(coord4);
    }
    memoryBarrierShared();
    barrier();
#endif
}

FetchResult prefetch_tap(ivec2 px) {
#if ENABLE_PREFETCH
    px += ivec2(PREFETCH_RADIUS) - ivec2(gl_WorkGroupID.xy * PREFETCH_GROUP_SIZE);
    return tile_input[px.x + TILE_DIM * px.y];
#else
    return do_fetch(px);
#endif
}
