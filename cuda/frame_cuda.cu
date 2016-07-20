/*
    Cuda accelerated motion estimation for VP8 libvpx encoder
    by Pietro Paglierani, Giuliano Grossi, Federico Pedersini and Alessandro Petrini

    for Italtel and Universita' degli Studi di Milano
    2015-2016, Milano
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <locale.h>

#include "vpx_config.h"
#include "cuda/frame_cuda.h"
#include "cuda/me_cuda.h"
//#include "misc/STATS.h"

#ifdef __cplusplus
extern "C" {
#endif

// all but print_cuda_ctx functions have been inlined in frame_cuda.h

/*
 * Print context stored on GPU
 */
__global__ void print_cuda_ctx(vpx_frames_GPU_t fr, FRAME_TYPE_GPU FRAME, int fb_idx) {
	int w = fr.width;
	int h = fr.height;
	printf("[GPU] Frame size: w = %d, h = %d\n", w, h);
	if ( FRAME == RAW_FRAME_GPU ) {
		printf("[GPU] Stampa frame CURRENT\n");
		//printf("[GPU] val0 = %d, val1 = %d\n", fr.raw_current_fb_g[0],fr.raw_current_fb_g[0]);
	} else {
		printf("[GPU] Stampa frame LAST\n");
		//printf("[GPU] val0 = %d, val1 = %d\n", fr.yv12_fb_g[fb_idx][0],fr.yv12_fb_g[fb_idx][0]);
	}
}

#ifdef __cplusplus
}
#endif
