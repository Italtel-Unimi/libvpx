/*
    Cuda accelerated motion estimation for VP8 libvpx encoder
    by Pietro Paglierani, Giuliano Grossi, Federico Pedersini and Alessandro Petrini

    for Italtel and Universita' degli Studi di Milano
    2015-2016, Milano
*/

#ifndef TYPEDEF_CUDA_H_
#define TYPEDEF_CUDA_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>

#include <stdint.h>
#include "vpx_config.h"
#include "vp8/common/mv.h"


// -----------------------------------
// Cuda devices and their capabilities
// -----------------------------------
struct gpu_status_t {
	int have_cuda_capable_devices;
	int current_GPU;
};

struct cuda_devices_t {
	char **names;            // Capable device names
	int ndevs;               // number of CUDA Capable devices
	int *major;              // Capability Major version number
	int *minor;              // Capability Minor version number
	int *threads_multiproc;  // max threads x multiprocessor
	size_t *local_mem;       // local memory amount
	int * maxTexture1D;      // max texture mem 1D
};


// ------------
// kernel names
// ------------
typedef enum ME_KERNELS_GPU {
	ME_NULL_KERNEL = 0,	// == me on cpu
	ME_FAST_KERNEL = 1,
	ME_SPLITMV_KERNEL = 2,
	ME_TEX_KERNEL = 3
} ME_KERNELS_GPU;


// ----------------
// Frame types enum
// ----------------
typedef enum FRAME_TYPE_GPU {
	NEW_FRAME_GPU = 0,
	LST_FRAME_GPU = 1,
	GLD_FRAME_GPU = 2,
	ALT_FRAME_GPU = 3,
	RAW_FRAME_GPU = 4
} FRAME_TYPE_GPU;             // GPU frame types

typedef enum FRAME_TYPE_CPU {
	NEW_FRAME_CPU = 0,
	LST_FRAME_CPU = 1,
	GLD_FRAME_CPU = 2,
	ALT_FRAME_CPU = 3,
	RAW_FRAME_CPU = 4,
	KEY_FRAME_CPU = 5
} FRAME_TYPE_CPU;             // GPU frame types

typedef enum flags_ref_frame_gpu {
  GPUFLAG_LAST_FRAME = 1,
  GPUFLAG_GOLD_FRAME = 2,
  GPUFLAG_ALTR_FRAME = 4
} flags_ref_frame_gpu_t;

typedef enum splitmv_on {
	SPLITMV_OFF = 0,
	SPLITMV_ON  = 1
} splitmv_on_t;


// ---------------------
// Host and device frame
// ---------------------
typedef struct vpx_frames_GPU {
	unsigned int 	width;           	// orig. frame width
	unsigned int	height;          	// orig. frame height
	unsigned int 	height_ext;      	// ext. frame height
	unsigned int	stride;         	// ext. frame stride
	int 			num_MB_width;       // number of macro-blocks for width
	int 			num_MB_height;      // number of macro-blocks for height
	int				num_mv;				// number of motion vectors per frame (as no, == number of MB per frame)
    int_mv	 		*MVs_g[3];          // gpu motion vectors
	int_mv			*MVs_split_g;		// gpu motion vectors (split) (*)
	int				mbrow;				// row number of current mb	(for splitmv)
	int				mbcol;				// col number of current mb	(for splitmv)
	int				refframe;			// reference frame of current mode	(for splitmv)

	// linear array stuff
	uint8_t 		*raw_current_fb_g;	// raw frame (BORDERLESS!!!)
	uint8_t			*yv12_fb_g[4];		// gpu framebuffers

	// Texture stuff
	struct cudaChannelFormatDesc	channelDesc;
	cudaArray_t						rawfb_arr;
	cudaArray_t						yv12_arr_g[4];
	struct cudaResourceDesc			rawResDesc;
	struct cudaResourceDesc			resDesc[4];
	struct cudaTextureDesc			texDesc;
	struct cudaResourceViewDesc		rawResViewDesc;
	struct cudaResourceViewDesc		resViewDesc;
	cudaTextureObject_t				rawFbTex;
	cudaTextureObject_t				fbTex[4];

} vpx_frames_GPU_t;


typedef struct vpx_frames_CPU {
	int_mv	 		*MVs_h[3];			// host motion vectors
	int_mv	 		*MVs_split_h;		// host motion vectors (split) (*)
} vpx_frames_CPU_t;

// (*): salvati sequenzialmente:
//			MVs_split_h[0][0 ..16] = MB 0, split 4x4
//			MVs_split_h[0][17..20] = MB 0, split 8x8
//			MVs_split_h[0][21..22] = MB 0, split 8x16
//			MVs_split_h[0][23..24] = MB 0, split 16x8
//			MVs_split_h[0][25..40] = MB 1, split 4x4
//			ecc...


// --------------------------------
// GPU block/grid and stream config
// --------------------------------
typedef struct GPU_config {
	dim3 			gridDim;
	dim3			blockDim;
	int				num_mb16th;			// num streams (16 mb each)

	struct {
		cudaStream_t *frame;			// streams for frame ME
	} streams;

	struct {
		cudaEvent_t start;				// start time
		cudaEvent_t stop;				// stop time
	} events;

} GPU_config_t;


// --------------------
// | MV float variant |
// --------------------
typedef struct {
    float row;
    float col;
} MV_ref;


#endif /* TYPEDEF_CUDA_H_ */
