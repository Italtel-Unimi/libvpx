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
#include "cuda/typedef_cuda.h"
#include "cuda/me_cuda.h"
//#include "cuda/me_diamonds.h"

#ifdef __cplusplus
extern "C" {
#endif
#if HAVE_CUDA_ENABLED_DEVICE

__device__ __constant__ MV MV_16x12_lookup_fast[] = {                     {-12,-2}, {-12, 0}, {-12, 2},		// Unit: pixel
		                                              {-10,-5}, {-10,-3}, {-10,-1}, {-10, 1}, {-10, 3}, {-10, 5},
		                                 {-8,-8},  {-8,-6},  {-8,-4},  {-8,-2},  {-8, 0},  {-8, 2},  {-8, 4},  {-8, 6},  {-8, 8},
		                           {-6,-9},  {-6,-7},  {-6,-5},  {-6,-3},  {-6,-1},  {-6, 1},  {-6, 3},  {-6, 5},  {-6, 7},  {-6, 9},
		           {-4,-12},  {-4,-10},  {-4,-8},  {-4,-6},  {-4,-4},  {-4,-2},  {-4, 0},  {-4, 2},  {-4, 4},  {-4, 6},  {-4, 8},  {-4,10},  {-4,12},
	        {-2,-13},   {-2,-11},  {-2,-9},  {-2,-7},  {-2,-5},  {-2,-3},  {-2,-1},  {-2, 1},  {-2, 3},  {-2, 5},  {-2, 7},  {-2, 9},  {-2,11},  {-2,13},
{0,-16},  {0,-14},  {0,-12},   {0,-10},   {0,-8},   {0,-6},   {0,-4},   {0,-2},   {0, 0},   {0, 2},   {0, 4},   {0, 6},   {0, 8},   {0,10},   {0,12},   {0,14},   {0,16},
		     {2,-13},    {2,-11},   {2,-9},   {2,-7},   {2,-5},   {2,-3},   {2,-1},   {2, 1},   {2, 3},   {2, 5},   {2, 7},   {2, 9},   {2,11},   {2,13},
		            {4,-12},   {4,-10},   {4,-8},   {4,-6},   {4,-4},   {4,-2},   {4, 0},   {4, 2},   {4, 4},   {4, 6},   {4, 8},   {4,10},   {4,12},
		                            {6,-9},   {6,-7},   {6,-5},   {6,-3},   {6,-1},   {6, 1},   {6, 3},   {6, 5},   {6, 7},   {6, 9},
		                                  {8,-8},   {8,-6},   {8,-4},   {8,-2},   {8, 0},   {8, 2},   {8, 4},   {8, 6},   {8, 8},
		                                               {10,-5},  {10,-3},   {10,-1},  {10, 1},  {10, 3},  {10, 5},
		                                                               {12,-2},  {12, 0},  {12, 2},
		{0, 0} }; // 127 + 1 candidati

__device__ __constant__ MV MV_lookup_refin_fast[] = {
	                    {-2, 0} ,		// Unit: pixel
	{-1, -2}, {-1, -1}, {-1, 0}, {-1, 1}, {-1, 2},
	          { 0, -1}, { 0, 0}, { 0, 1},
	{ 1, -2}, { 1, -1}, { 1, 0}, { 1, 1}, { 1, 2},
	                    { 2, 0},
	{0, 0} };  // one more to reach 16

__constant__ int offset_16x12[128];
__constant__ int offset_16x12_refin[16];

void setup_constant_mem_fast(int img_stride) {
	int I = img_stride;
	int off_16x12[] = { -12*I-2, -12*I, -12*I+2,		// Offsets
			-10*I-5, -10*I-3, -10*I-1,  -10*I+1,  -10*I+3, -10*I+5,
			-8*I-8,  -8*I-6, -8*I-4,  -8*I-2,  -8*I, -8*I+2, -8*I+4, -8*I+6, -8*I+8,
			-6*I-9,  -6*I-7, -6*I-5,  -6*I-3, -6*I-1, -6*I+1, -6*I+3, -6*I+5, -6*I+7, -6*I+9,
			-4*I-12, -4*I-10, -4*I-8,  -4*I-6, -4*I-4, -4*I-2, -4*I, -4*I+2, -4*I+4, -4*I+6, -4*I+8, -4*I+10, -4*I+12,
			-2*I-13, -2*I-11, -2*I-9,  -2*I-7, -2*I-5, -2*I-3, -2*I-1, -2*I+1, -2*I+3, -2*I+5, -2*I+7, -2*I+9, -2*I+11, -2*I+13,
			-16,  -14,  -12,  -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16,
			2*I-13, 2*I-11, 2*I-9,  2*I-7, 2*I-5, 2*I-3, 2*I-1, 2*I+1, 2*I+3, 2*I+5, 2*I+7, 2*I+9, 2*I+11, 2*I+13,
			4*I-12, 4*I-10, 4*I-8,  4*I-6, 4*I-4, 4*I-2, 4*I,  4*I+2, 4*I+4, 4*I+6, 4*I+8, 4*I+10, 4*I+12,
			6*I-9,  6*I-7,  6*I-5,  6*I-3, 6*I-1, 6*I+1, 6*I+3, 6*I+5, 6*I+7, 6*I+9,
			8*I-8,  8*I-6,  8*I-4,  8*I-2, 8*I, 8*I+2, 8*I+4, 8*I+6, 8*I+8,
			10*I-5, 10*I-3, 10*I-1, 10*I+1, 10*I+3, 10*I+5,
			12*I-2, 12*I, 12*I+2,
			0 };		// one more to reach 128
	int off_16x12_refin[] = { -2*I ,
			-I-2, -I-1, -I, -I+1, -I+2,
			-1,  0,  1,
			-I-2, I-1, I, I+1, I+2,
			2*I,
			0 };

	// copy to device constant memory
	CHECK(cudaMemcpyToSymbol(offset_16x12, off_16x12, 128*sizeof(int)));
	CHECK(cudaMemcpyToSymbol(offset_16x12_refin, off_16x12_refin, 16*sizeof(int)));
}


__global__ void me_cuda_fast (const uint8_t * __restrict__ const in_frame, const uint8_t * __restrict__ const ref_frame,
				int const streamID, int const stride, int const width, int const num_MB_width,
				int_mv * __restrict__ const MVs_g ) {

	__shared__ int diff[128][64];
	int TID = threadIdx.x*blockDim.y + threadIdx.y;	// Thread Index (0..127)
	int i, j;

	int MBoffset = streamID*16 + blockIdx.x;
	int blockX = MBoffset / num_MB_width;		// row?
	int blockY = MBoffset % num_MB_width;		// column?
	int im_offset = 16*(blockX*stride + blockY) + (threadIdx.x*stride + threadIdx.y); 	// That takes into account the 2 macroblocks border
	int im_offset_raw = 16*(blockX*width + blockY) + (threadIdx.x*width + threadIdx.y);

	const uint8_t *refptr = ref_frame + 32*(stride+1) + im_offset;
	const uint8_t *imptr  = in_frame + im_offset_raw;

	// Compute pixel differences: //////////
	for( i=0; i<128; i+=2 )		// the 127+1 positions in the 16x12 diamond: (64 threads/position x 2 positions)/warp
	{
		j = i + (TID/64);
		{
			diff[j][TID%64]  = abs( (int)*(imptr)			- (int)*(refptr + offset_16x12[j]) );
			diff[j][TID%64] += abs( (int)*(imptr + 4*width) - (int)*(refptr + offset_16x12[j] + 4*stride) );
			diff[j][TID%64] += abs( (int)*(imptr + 8*width) - (int)*(refptr + offset_16x12[j] + 8*stride) );
			diff[j][TID%64] += abs( (int)*(imptr +12*width) - (int)*(refptr + offset_16x12[j] +12*stride) );
		}
	}
	__syncthreads();

	// Compute SUM: even if 63 total warps, this is much faster than with: for( i=1..64 )
	//
	// Sum: 64 --> 32 (32 warps)
	for( i=0; i<32; i++ )
		diff[TID] [i] += diff[TID] [i + 32];
	__syncthreads();

	// Sum 32 --> 16 (16 warps)
	for( i=0; i<16; i++ )
		diff[TID] [i] += diff[TID] [i + 16];
	__syncthreads();

	// Sum 16 --> 8 (8 warp)
	for( i=0; i<8; i++ )
		diff[TID] [i] += diff[TID] [i + 8];
	__syncthreads();

	// Sum 8 --> 4 (4 warp)
	for( i=0; i<8; i++ )
		diff[TID] [i] += diff[TID] [i + 4];
	__syncthreads();

	// Sum 4 --> 2 (2 warp)
	diff[TID] [0] += diff[TID] [2];
	diff[TID] [1] += diff[TID] [3];
	__syncthreads();

	// Sum 2 --> 1 (1 warp)
	diff[TID] [0] += diff[TID] [1];
	__syncthreads();

	// Find MINIMUM (and corresponding best MV) of 128 Pts - 128 threads ////////////////////////
	//
	__shared__ int minpos[128];

	minpos[TID] = TID;	// All 128 threads!

	if( TID < 64 )	// 64 threads
		if( diff[TID][0] > diff[TID + 64][0] )	{
			diff[TID][0] = diff[TID + 64][0];
			minpos[TID] = minpos[TID + 64];
		}
	__syncthreads();

	if( TID < 32 )	// 32 threads
		if( diff[TID][0] > diff[TID + 32][0] )	{
			diff[TID][0] = diff[TID + 32][0];
			minpos[TID] = minpos[TID + 32];
		}
	__syncthreads();

	if( TID < 16 )	// 16 threads
		if( diff[TID][0] > diff[TID + 16][0] )	{
			diff[TID][0] = diff[TID + 16][0];
			minpos[TID] = minpos[TID + 16];
		}
	__syncthreads();

	if( TID < 8 )	// 8 threads
		if( diff[TID][0] > diff[TID + 8][0] )	{
			diff[TID][0] = diff[TID + 8][0];
			minpos[TID] = minpos[TID + 8];
		}
	__syncthreads();

	if( TID < 4 )	// 4 threads
		if( diff[TID][0] > diff[TID + 4][0] )	{
			diff[TID][0] = diff[TID + 4][0];
			minpos[TID] = minpos[TID + 4];
		}
	__syncthreads();

	if( TID < 2 )	// 2 threads
		if( diff[TID][0] > diff[TID + 2][0] )	{
			diff[TID][0] = diff[TID + 2][0];
			minpos[TID] = minpos[TID + 2];
		}
	__syncthreads();

	if( TID == 0 )	// Only thread 0
	{
		if( diff[0][0] > diff[1][0] )	{
			diff[0][0] = diff[1][0];
			minpos[0] = minpos[1];
		}
		// And finally assign resulting MV

		MVs_g[MBoffset].as_mv = MV_16x12_lookup_fast[ minpos[0] ];

	}
	__syncthreads();

	///////////////////////////////////////////////////////////////////////////////////////////
	// STEP 2: pixel-scale Motion Vector Search

	// Update RefPointer to the best motion vector
	refptr += offset_16x12[ minpos[0] ] ;

	// Compute pixel differences: //////////
	//
	for( i=0; i<16; i+=2 )		// the 15+1 positions in the STEP 2 of the 16x12 diamond: 64 threads x 2 positions/warp
	{
		j = i + (TID/64);
		{
			diff[j][TID%64]  = abs( (int)*(imptr)			   - (int)*(refptr + offset_16x12_refin[j]) );
			diff[j][TID%64] += abs( (int)*(imptr + 4*width) - (int)*(refptr + offset_16x12_refin[j] + 4*stride) );
			diff[j][TID%64] += abs( (int)*(imptr + 8*width) - (int)*(refptr + offset_16x12_refin[j] + 8*stride) );
			diff[j][TID%64] += abs( (int)*(imptr +12*width) - (int)*(refptr + offset_16x12_refin[j] +12*stride) );
		}
	}
	__syncthreads();

	// Sum 64 --> 32 (128 threads, 4 warps)
	for( i=0; i<16; i+=4 )
		diff[i + (TID/32)] [TID%32] += diff[i + (TID/32)] [(TID%32) + 32];
	__syncthreads();

	// Sum 32 --> 16 (128 threads, 2 warps)
	for( i=0; i<16; i+=8 )
		diff[i + (TID/16)] [TID%16] += diff[i + (TID/16)] [(TID%16) + 16];
	__syncthreads();

	// Sum 16 --> 8 (128 threads, 1 warp)
	diff[threadIdx.y] [threadIdx.x] += diff[threadIdx.y] [threadIdx.x + 8];
	__syncthreads();

	// Sum 8 --> 4 (64 threads, 1 warp)
	if( threadIdx.x < 4 )
		diff[threadIdx.y] [threadIdx.x] += diff[threadIdx.y] [threadIdx.x + 4];
	__syncthreads();

	// Sum 4 --> 2 (32 threads, 1 warp)
	if( threadIdx.x < 2 )
		diff[threadIdx.y] [threadIdx.x] += diff[threadIdx.y] [threadIdx.x + 2];
	__syncthreads();

	// Sum 2 --> 1 (16 threads, 1 warp)
	if( threadIdx.x==0 )
		diff[threadIdx.y] [0] += diff[threadIdx.y] [1];
	__syncthreads();


	// Find MINIMUM (and corresponding best MV) of 16 Pts - 16 threads ////////////////////////
	//
	if( TID < 16 )
		minpos[TID] = TID;
	__syncthreads();

	if( threadIdx.x==0 )
	{
		if( threadIdx.y < 8 )
			if( diff[threadIdx.y][0] > diff[threadIdx.y + 8][0] )	{
				diff[threadIdx.y][0] = diff[threadIdx.y + 8][0];
				minpos[threadIdx.y] = minpos[threadIdx.y + 8];
			}
		__syncthreads();

		if( threadIdx.y < 4 )
			if( diff[threadIdx.y][0] > diff[threadIdx.y + 4][0] )	{
				diff[threadIdx.y][0] = diff[threadIdx.y + 4][0];
				minpos[threadIdx.y] = minpos[threadIdx.y + 4];
			}
		__syncthreads();

		if( threadIdx.y < 2 )
			if( diff[threadIdx.y][0] > diff[threadIdx.y + 2][0] )	{
				diff[threadIdx.y][0] = diff[threadIdx.y + 2][0];
				minpos[threadIdx.y] = minpos[threadIdx.y + 2];
			}
		__syncthreads();

		if( threadIdx.y == 0 )
		{
			if( diff[0][0] > diff[1][0] )	{
				diff[0][0] = diff[1][0];
				minpos[0] = minpos[1];
			}
			// And finally assign resulting MV
			MVs_g[MBoffset].as_mv.row += MV_lookup_refin_fast[ minpos[0] ].row;	// Added to the previous MV
			MVs_g[MBoffset].as_mv.col += MV_lookup_refin_fast[ minpos[0] ].col;	// Added to the previous MV
			MVs_g[MBoffset].as_mv.row <<= 3;
			MVs_g[MBoffset].as_mv.col <<= 3;
		}
	}
	__syncthreads();

}


inline void me_kernel_launch_fast( VP8_COMMON * const common, const uint8_t * const in_frame, const uint8_t * const ref_frame,
		int const streamID, int_mv * const MVs ) {

#if CUDA_VERBOSE
	float elapsedTime;
	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	CHECK(cudaEventRecord(start));
#endif

	me_cuda_fast <<< common->GPU.gridDim, common->GPU.blockDim, 0, common->GPU.streams.frame[streamID] >>> (in_frame, ref_frame,
			streamID, common->gpu_frame.stride, common->gpu_frame.width, common->gpu_frame.num_MB_width, MVs );

#if CUDA_VERBOSE
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("\n[GPU] ME elapsed time streams[%d]:  %.4f ms\n",streamID,elapsedTime);
	CHECK(cudaEventDestroy(start));
	CHECK(cudaEventDestroy(stop));
	add_STATS((double)elapsedTime,0);
#endif

}

void me_cuda_launch_interleaved_fast( VP8_COMMON * const cm, int fb_idx, int ref_frame_flags ) {

	int MV_size_16 = 16*sizeof(int_mv);
	// for printing informations about reference frame flags and thei usage, I left a commented prinft at line 3625
	// at the beginning of encode_frame_to_data_rate(..) in onyx_if.c

	for (int s = 0; s < cm->GPU.num_mb16th; s++) {

		int offset = 16*s;
		// bugfix per immagini il cui n di mb non e' divisibile per 16
		// prima venivano lanciati troppi processi e cudaMemcpyAsync andava a leggere oltre i limiti degli array
		if (offset + 16 > cm->gpu_frame.num_mv) {
			MV_size_16 = ( offset + 16 - cm->gpu_frame.num_mv ) * sizeof( int_mv );
		}

		if ((ref_frame_flags & GPUFLAG_LAST_FRAME) && (cm->yv12_fb[cm->lst_fb_idx].flags & GPUFLAG_LAST_FRAME)) {
			me_kernel_launch_fast(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->lst_fb_idx], s, (cm->gpu_frame.MVs_g)[0] );
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[0][offset],		&(cm->gpu_frame.MVs_g)[0][offset],		MV_size_16,      cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}

		// Se ref_frame_flags indica la presenza di un gold e se il flag del fb puntato da gld_fb_idx indica che e' gold, allora...
		if ((ref_frame_flags & GPUFLAG_GOLD_FRAME) && (cm->yv12_fb[cm->gld_fb_idx].flags & GPUFLAG_GOLD_FRAME)) {
			me_kernel_launch_fast(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->gld_fb_idx], s, (cm->gpu_frame.MVs_g)[1] );
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[1][offset],		&(cm->gpu_frame.MVs_g)[1][offset],		MV_size_16,      cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}

		// Se ref_frame_flags indica la presenza di un altref e se il flag del fb puntato da alt_fb_idx indica che e' altref, allora...
		if ((ref_frame_flags & GPUFLAG_ALTR_FRAME) && (cm->yv12_fb[cm->alt_fb_idx].flags & GPUFLAG_ALTR_FRAME)) {
			me_kernel_launch_fast(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->alt_fb_idx], s, (cm->gpu_frame.MVs_g)[2] );
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[2][offset],		&(cm->gpu_frame.MVs_g)[2][offset],		MV_size_16,		 cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}
	}
}


void me_cuda_launch_not_interleaved_fast( VP8_COMMON * const cm, int fb_idx, int ref_frame_flags ) {

	int MV_size_16 = 16*sizeof(int_mv);
	// for printing informations about reference frame flags and thei usage, I left a commented prinft at line 3625
	// at the beginning of encode_frame_to_data_rate(..) in onyx_if.c

	for (int s = 0; s < cm->GPU.num_mb16th; s++) {

		int offset = 16*s;
		// bugfix per immagini il cui n di mb non e' divisibile per 16
		// prima venivano lanciati troppi processi e cudaMemcpyAsync andava a leggere oltre i limiti degli array
		if (offset + 16 > cm->gpu_frame.num_mv) {
			MV_size_16 = ( offset + 16 - cm->gpu_frame.num_mv ) * sizeof( int_mv );
		}

		if ((ref_frame_flags & GPUFLAG_LAST_FRAME) && (cm->yv12_fb[cm->lst_fb_idx].flags & GPUFLAG_LAST_FRAME)) {
			me_kernel_launch_fast(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->lst_fb_idx], s, (cm->gpu_frame.MVs_g)[0] );
		}

		// Se ref_frame_flags indica la presenza di un gold e se il flag del fb puntato da gld_fb_idx indica che e' gold, allora...
		if ((ref_frame_flags & GPUFLAG_GOLD_FRAME) && (cm->yv12_fb[cm->gld_fb_idx].flags & GPUFLAG_GOLD_FRAME)) {
			me_kernel_launch_fast(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->gld_fb_idx], s, (cm->gpu_frame.MVs_g)[1] );
		}

		// Se ref_frame_flags indica la presenza di un altref e se il flag del fb puntato da alt_fb_idx indica che e' altref, allora...
		if ((ref_frame_flags & GPUFLAG_ALTR_FRAME) && (cm->yv12_fb[cm->alt_fb_idx].flags & GPUFLAG_ALTR_FRAME)) {
			me_kernel_launch_fast(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->alt_fb_idx], s, (cm->gpu_frame.MVs_g)[2] );
		}

		if (ref_frame_flags & GPUFLAG_LAST_FRAME) {
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[0][offset],		&(cm->gpu_frame.MVs_g)[0][offset],		MV_size_16,      cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}

		if ((ref_frame_flags & GPUFLAG_GOLD_FRAME) && (cm->yv12_fb[cm->gld_fb_idx].flags & GPUFLAG_GOLD_FRAME)) {
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[1][offset],		&(cm->gpu_frame.MVs_g)[1][offset],		MV_size_16,      cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}

		if ((ref_frame_flags & GPUFLAG_ALTR_FRAME) && (cm->yv12_fb[cm->alt_fb_idx].flags & GPUFLAG_ALTR_FRAME)) {
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[2][offset],		&(cm->gpu_frame.MVs_g)[2][offset],		MV_size_16,		 cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}
	}
}


#endif  /* HAVE_CUDA_ENABLED_DEVICE */
#ifdef __cplusplus
}
#endif
