#include <shared/shared.inl>

#include <utils/math.glsl>

#if RTDGI_TEMPORAL_COMPUTE
DAXA_DECL_PUSH_CONSTANT(RtdgiRestirResolvePush, push)

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
}
#endif

#if RTDGI_SPATIAL_COMPUTE
DAXA_DECL_PUSH_CONSTANT(RtdgiPush, push)

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
}
#endif

#if RTDGI_REPROJECT_COMPUTE
DAXA_DECL_PUSH_CONSTANT(RtdgiPush, push)

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
}
#endif

#if RTDGI_VALIDATE_COMPUTE
DAXA_DECL_PUSH_CONSTANT(RtdgiPush, push)

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
}
#endif

#if RTDGI_TRACE_COMPUTE
DAXA_DECL_PUSH_CONSTANT(RtdgiPush, push)

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
}
#endif

#if RTDGI_VALIDITY_INTEGRATE_COMPUTE
DAXA_DECL_PUSH_CONSTANT(RtdgiPush, push)

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
}
#endif

#if RTDGI_RESTIR_TEMPORAL_COMPUTE
DAXA_DECL_PUSH_CONSTANT(RtdgiPush, push)

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
}
#endif

#if RTDGI_RESTIR_SPATIAL_COMPUTE
DAXA_DECL_PUSH_CONSTANT(RtdgiRestirSpatialPush, push)

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
}
#endif

#if RTDGI_RESTIR_RESOLVE_COMPUTE
DAXA_DECL_PUSH_CONSTANT(RtdgiRestirResolvePush, push)

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
}
#endif
