// Helper defines
#define ALBEDO_TEXTURE 0
#define ALBEDO_DEBUG_POS 1
#define ALBEDO_DEBUG_NRM 2
#define ALBEDO_DEBUG_DIST 4
#define ALBEDO_DEBUG_RANDOM 5
#define ALBEDO_DEBUG_BLOCKID 6

// Which information to show as the albedo
#define ALBEDO ALBEDO_TEXTURE
// Whether to disable everything else and draw just the complexity
#define VISUALIZE_STEP_COMPLEXITY 0
#define VISUALIZE_SUBGRID 0
// Whether to cast shadow rays
#define ENABLE_SHADOWS 1
#define ENABLE_FAKE_SKY_LIGHTING 1
#define ENABLE_REFLECTIONS 1
// Camera settings
#define LENS_TYPE_DEFAULT 0
#define LENS_TYPE_FISHEYE 1
#define LENS_TYPE_EQUIRECTANGULAR 2

#define LENS_TYPE LENS_TYPE_DEFAULT
// Whether to visualize the position that the view ray intersects
#define SHOW_UI 1
#define SHOW_PICK_POS 1
#define SHOW_DEBUG_BLOCKS 0
// Whether to variate the sample-space coordinates based on time
#define JITTER_VIEW 0
// Number of samples per axis (so a value of 4 means 16 samples)
#define SUBSAMPLE_N 1

#define BLOCKEDIT_RADIUS 0

#define ENABLE_TAA 1
#define TAA_MIXING 0.1

#define MAX_STEPS (BLOCK_NX + BLOCK_NY + BLOCK_NZ)

#define DISABLE_DRIVER_BREAKING_CODE 1
