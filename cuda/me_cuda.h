/*
    Cuda accelerated motion estimation for VP8 libvpx encoder
    by Pietro Paglierani, Giuliano Grossi, Federico Pedersini and Alessandro Petrini

    for Italtel and Universita' degli Studi di Milano
    2015-2016, Milano
*/

#ifndef ME_CUDA_H_
#define ME_CUDA_H_

#include "vp8/common/onyxc_int.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include "cuda/define_cuda.h"
#include "cuda/typedef_cuda.h"
#include "cuda/frame_cuda.h"


#ifdef __cplusplus
extern "C" {
#endif

// public kernels
__global__ void me_cuda_fast (const uint8_t * __restrict__ const in_frame, const uint8_t * __restrict__ const ref_frame,
				int const streamID, int const streamSize, int const stride, int const width, int const num_MB_width,
				int_mv * __restrict__ const MVs_g );

__global__ void me_cuda_split (const uint8_t * __restrict__ const in_frame, const uint8_t * __restrict__ const ref_frame,
		int const streamID, int const streamSize, int const stride, int const width, int const num_MB_width, int const split_on,
		int_mv * __restrict__ const MVs_g, int_mv * __restrict__ const MVs_split_g );

__global__ void me_cuda_tex ( const cudaTextureObject_t in_tex, const cudaTextureObject_t ref_tex,
				int const streamID, int const streamSize, int const stride, int const width, int const num_MB_width, int const split_on,
				int_mv * __restrict__ const MVs_g, int_mv * __restrict__ const MVs_split_g );


void me_cuda_launch_interleaved_fast( VP8_COMMON * const cm, int fb_idx, int ref_frame_flags );
void me_cuda_launch_interleaved_split( VP8_COMMON * const cm, int fb_idx, int ref_frame_flags );
void me_cuda_launch_interleaved_tex( VP8_COMMON * const cm, int fb_idx, int ref_frame_flags );


void me_kernel_launch_fast( VP8_COMMON * const common, const uint8_t * const in_frame, const uint8_t * const ref_frame,
		int const streamID, int_mv * const MVs );
void me_kernel_launch_split( VP8_COMMON * const common, const uint8_t * const in_frame, const uint8_t * const ref_frame,
		int const streamID, int const split_on, int_mv * const MVs, int_mv * const MVs_split );
void me_kernel_launch_tex( VP8_COMMON * const common, const cudaTextureObject_t in_tex, const cudaTextureObject_t ref_tex,
		int const streamID, int const split_on, int_mv * const MVs, int_mv * const MVs_split );

void setup_constant_mem_fast(int img_stride);
void setup_constant_mem_split(int img_stride);

#ifdef __cplusplus
}
#endif

#endif /* ME_CUDA_H */
