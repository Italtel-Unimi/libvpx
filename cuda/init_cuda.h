/*
    Cuda accelerated motion estimation for VP8 libvpx encoder
    by Pietro Paglierani, Giuliano Grossi, Federico Pedersini and Alessandro Petrini

    for Italtel and Universita' degli Studi di Milano
    2015-2016, Milano
*/

#ifndef INIT_CUDA_H_
#define INIT_CUDA_H_

#include "cuda/define_cuda.h"
#include "vp8/common/onyxc_int.h"
#include "cuda/typedef_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

// functions prototypes
void init_cuda(void);
void GPU_setup( GPU_config_t * GPU_config, int gpu_frame_width, int gpu_frame_height );
void memory_setup_CPU_GPU( VP8_COMMON *cm );
void GPU_destroy( VP8_COMMON *cm );
void GPUstreamReorder( VP8_COMMON * const cm );


#ifdef __cplusplus
}
#endif

#endif /* INIT_CUDA_H_ */
