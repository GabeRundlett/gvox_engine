#pragma once

#include <shared/input.inl>
#include <shared/globals.inl>
#include <shared/voxels.inl>

#include <shared/core.inl>

#if STARTUP_COMPUTE || defined(__cplusplus)
#include <shared/tasks/startup.inl>
#endif
#if PERFRAME_COMPUTE || defined(__cplusplus)
#include <shared/tasks/perframe.inl>
#endif
#if CHUNK_EDIT_COMPUTE || defined(__cplusplus)
#include <shared/tasks/chunk_edit.inl>
#endif
#if CHUNK_OPT_COMPUTE || defined(__cplusplus)
#include <shared/tasks/chunk_opt.inl>
#endif
#if CHUNK_ALLOC_COMPUTE || defined(__cplusplus)
#include <shared/tasks/chunk_alloc.inl>
#endif
#if PER_CHUNK_COMPUTE || defined(__cplusplus)
#include <shared/tasks/per_chunk.inl>
#endif
#if VOXEL_PARTICLE_SIM_COMPUTE || defined(__cplusplus)
#include <shared/tasks/voxel_particle_sim.inl>
#endif

#if CALCULATE_REPROJECTION_MAP_COMPUTE || defined(__cplusplus)
#include <shared/tasks/renderer/calculate_reprojection_map.inl>
#endif
#if TRACE_DEPTH_PREPASS_COMPUTE || defined(__cplusplus)
#include <shared/tasks/renderer/trace_depth_prepass.inl>
#endif
#if TRACE_PRIMARY_COMPUTE || defined(__cplusplus)
#include <shared/tasks/renderer/trace_primary.inl>
#endif
#if DOWNSCALE_COMPUTE || defined(__cplusplus)
#include <shared/tasks/renderer/downscale.inl>
#endif
#if SSAO_COMPUTE || defined(__cplusplus)
#include <shared/tasks/renderer/ssao.inl>
#endif
#if SSAO_SPATIAL_FILTER_COMPUTE || defined(__cplusplus)
#include <shared/tasks/renderer/ssao/spatial_filter.inl>
#endif
#if SSAO_UPSAMPLE_COMPUTE || defined(__cplusplus)
#include <shared/tasks/renderer/ssao/upsample.inl>
#endif
#if SSAO_TEMPORAL_FILTER_COMPUTE || defined(__cplusplus)
#include <shared/tasks/renderer/ssao/temporal_filter.inl>
#endif
#if TRACE_SECONDARY_COMPUTE || defined(__cplusplus)
#include <shared/tasks/renderer/trace_secondary.inl>
#endif
#if UPSCALE_RECONSTRUCT_COMPUTE || defined(__cplusplus)
#include <shared/tasks/renderer/upscale_reconstruct.inl>
#endif
#if POSTPROCESSING_RASTER || defined(__cplusplus)
#include <shared/tasks/renderer/postprocessing.inl>
#endif
#if VOXEL_PARTICLE_RASTER || defined(__cplusplus)
#include <shared/tasks/renderer/voxel_particle_raster.inl>
#endif
