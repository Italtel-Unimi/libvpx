/*
    Cuda accelerated motion estimation for VP8 libvpx encoder
    by Pietro Paglierani, Giuliano Grossi, Federico Pedersini and Alessandro Petrini

    for Italtel and Universita' degli Studi di Milano
    2015-2016, Milano
*/

#ifndef FRAME_CUDA_H_
#define FRAME_CUDA_H_

#include "vp8/common/onyxc_int.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include "cuda/define_cuda.h"
#include "cuda/typedef_cuda.h"
#include "cuda/me_cuda.h"


#ifdef __cplusplus
extern "C" {
#endif

static inline void GPU_sync_stream_frame( VP8_COMMON *common, int streamID) {
	CHECK(cudaStreamSynchronize(common->GPU.streams.frame[streamID]));
}

// Only for texure kernel copy to CudaArray
static inline void copy_raw_frame_to_GPU_tex( VP8_COMMON * const cm, uint8_t * const y_plane ) {
	int frame_size_raw = ( cm->gpu_frame.width  + cm->gpu_frame.width  % 16 ) * ( cm->gpu_frame.height + cm->gpu_frame.height % 16 );
	CHECK( cudaMemcpyToArray( (cm->gpu_frame.rawfb_arr), 0, 0, y_plane, frame_size_raw * sizeof(uint8_t), cudaMemcpyHostToDevice ));
}
static inline void copy_new_frame_to_GPU_tex( VP8_COMMON * const cm, uint8_t * const y_plane, int new_fb_idx ) {
	int frame_size = cm->gpu_frame.stride * cm->gpu_frame.height_ext;
	CHECK( cudaMemcpyToArray( (cm->gpu_frame.yv12_arr_g)[new_fb_idx], 0, 0, y_plane, frame_size * sizeof(uint8_t), cudaMemcpyHostToDevice ));
}


// Only for fast e splitmv kernels: copy to linear arrays
static inline void copy_raw_frame_to_GPU( VP8_COMMON * const cm, uint8_t * const y_plane ) {
	int frame_size_raw = ( cm->gpu_frame.width  + cm->gpu_frame.width  % 16 ) * ( cm->gpu_frame.height + cm->gpu_frame.height % 16 );
	CHECK( cudaMemcpy(cm->gpu_frame.raw_current_fb_g, y_plane, frame_size_raw, cudaMemcpyHostToDevice) );
}
static inline void copy_new_frame_to_GPU( VP8_COMMON * const cm, uint8_t * const y_plane, int new_fb_idx ) {
	int frame_size = cm->gpu_frame.stride * cm->gpu_frame.height_ext;
	CHECK(cudaMemcpy( (cm->gpu_frame.yv12_fb_g)[new_fb_idx], y_plane, frame_size, cudaMemcpyHostToDevice) );
}


#ifdef __cplusplus
}
#endif

#endif /* FRAME_TYPE_GPU */
