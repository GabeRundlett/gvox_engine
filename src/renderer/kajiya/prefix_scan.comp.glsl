#include <renderer/kajiya/prefix_scan.inl>
#include <utils/common.glsl>

#define THREAD_GROUP_SIZE 512
#define SEGMENT_SIZE (THREAD_GROUP_SIZE * 2)

#if PrefixScan1ComputeShader

DAXA_DECL_PUSH_CONSTANT(PrefixScan1ComputePush, push)
daxa_RWBufferPtr(daxa_u32) inout_buf = push.uses.inout_buf;

shared uint shared_data[SEGMENT_SIZE];

uvec2 load_input2(uint idx, uint segment) {
    uvec2 result = uvec2(0);
    uint i = idx + segment * SEGMENT_SIZE;
    if (i + 1 < push.element_n) {
        result.x = deref(inout_buf[i]);
        result.y = deref(inout_buf[i + 1]);
    } else if (i < push.element_n) {
        result.x = deref(inout_buf[i]);
    }
    return result;
}

void store_output2(uint idx, uint segment, uvec2 val) {
    uint i = idx + segment * SEGMENT_SIZE;
    if (i + 1 < push.element_n) {
        deref(inout_buf[i]) = val.x;
        deref(inout_buf[i + 1]) = val.y;
    } else if (i < push.element_n) {
        deref(inout_buf[i]) = val.x;
    }
}

layout(local_size_x = THREAD_GROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint idx = gl_LocalInvocationID.x;
    uint segment = gl_WorkGroupID.x;
    const uint STEP_COUNT = uint(log2(THREAD_GROUP_SIZE)) + 1;

    const uvec2 input2 = load_input2(idx * 2, segment);
    shared_data[idx * 2] = input2.x;
    shared_data[idx * 2 + 1] = input2.y;

    barrier();
    memoryBarrierShared();

    for (uint step = 0; step < STEP_COUNT; step++) {
        uint mask = (1u << step) - 1;
        uint rd_idx = ((idx >> step) << (step + 1)) + mask;
        uint wr_idx = rd_idx + 1 + (idx & mask);

        shared_data[wr_idx] += shared_data[rd_idx];

        barrier();
        memoryBarrierShared();
    }

    store_output2(idx * 2, segment, uvec2(shared_data[idx * 2], shared_data[idx * 2 + 1]));
}

#endif

#if PrefixScan2ComputeShader

DAXA_DECL_PUSH_CONSTANT(PrefixScan2ComputePush, push)
daxa_BufferPtr(daxa_u32) input_buf = push.uses.input_buf;
daxa_RWBufferPtr(daxa_u32) output_buf = push.uses.output_buf;

shared uint shared_data[SEGMENT_SIZE];

uint load_input(uint idx) {
    const uint segment_sum_idx = idx * SEGMENT_SIZE + SEGMENT_SIZE - 1;
    if (segment_sum_idx < push.element_n) {
        return deref(input_buf[segment_sum_idx]);
    } else {
        return 0;
    }
}

void store_output2(uint idx, uvec2 val) {
    deref(output_buf[idx]) = val.x;
    deref(output_buf[idx + 1]) = val.y;
}

layout(local_size_x = THREAD_GROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint idx = gl_LocalInvocationID.x;
    uint segment = gl_WorkGroupID.x;
    const uint STEP_COUNT = uint(log2(THREAD_GROUP_SIZE)) + 1;

    shared_data[idx * 2] = load_input(idx * 2);
    shared_data[idx * 2 + 1] = load_input(idx * 2 + 1);

    barrier();
    memoryBarrierShared();

    for (uint step = 0; step < STEP_COUNT; step++) {
        uint mask = (1u << step) - 1;
        uint rd_idx = ((idx >> step) << (step + 1)) + mask;
        uint wr_idx = rd_idx + 1 + (idx & mask);

        shared_data[wr_idx] += shared_data[rd_idx];

        barrier();
        memoryBarrierShared();
    }

    store_output2(idx * 2, uvec2(shared_data[idx * 2], shared_data[idx * 2 + 1]));
}

#endif

#if PrefixScanMergeComputeShader

DAXA_DECL_PUSH_CONSTANT(PrefixScanMergeComputePush, push)
daxa_RWBufferPtr(daxa_u32) inout_buf = push.uses.inout_buf;
daxa_BufferPtr(daxa_u32) segment_sum_buf = push.uses.segment_sum_buf;

uvec2 load_input2(uint idx, uint segment) {
    uvec2 internal_sum = uvec2(0);
    uint i = idx + segment * SEGMENT_SIZE;
    if (i + 1 < push.element_n) {
        internal_sum.x = deref(inout_buf[i]);
        internal_sum.y = deref(inout_buf[i + 1]);
    } else if (i < push.element_n) {
        internal_sum.x = deref(inout_buf[i]);
    }

    uint prev_segment_sum = 0;
    if (segment != 0) {
        prev_segment_sum = deref(segment_sum_buf[segment - 1]);
    }
    return internal_sum + prev_segment_sum;
}

void store_output2(uint idx, uint segment, uvec2 val) {
    uint i = idx + segment * SEGMENT_SIZE;
    if (i + 1 < push.element_n) {
        deref(inout_buf[i]) = val.x;
        deref(inout_buf[i + 1]) = val.y;
    } else if (i < push.element_n) {
        deref(inout_buf[i]) = val.x;
    }
}

layout(local_size_x = THREAD_GROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint idx = gl_LocalInvocationID.x;
    uint segment = gl_WorkGroupID.x;
    store_output2(idx * 2, segment, load_input2(idx * 2, segment));
}

#endif
