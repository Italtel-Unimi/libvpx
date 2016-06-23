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
#include "cuda/me_diamonds.h"

#ifdef __cplusplus
extern "C" {
#endif
#if HAVE_CUDA_ENABLED_DEVICE


extern __constant__ int offset_16x12[128];
extern __constant__ int offset_16x12_refin[16];


__inline__ __device__ uint32_t __vabsdiff4( uint32_t u, uint32_t v )
{
	uint32_t w = 0;
	asm volatile("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(w) : "r"(u), "r"(v), "r"(w));
	return w;
}


__global__ void me_cuda_split (const uint8_t * __restrict__ const in_frame, const uint8_t * __restrict__ const ref_frame,
				int const streamID, int const stride, int const width, int const num_MB_width,
				int_mv * __restrict__ const MVs_g ) {

	__shared__ int diff[128][32];
	__shared__ int minpos[128];

	// configurazione di lancio: blocks:  16 x 1 x 1
	//							 threads:  4 x 8 x 1

	int TID = threadIdx.y * blockDim.x + threadIdx.x;	// Thread Index (0..32)
	int i;
	int sad_result;

	int MBoffset = streamID * 16 + blockIdx.x;
	int blockX = MBoffset % num_MB_width;		// colonna MB
	int blockY = MBoffset / num_MB_width;		// riga MB
	// Occhio: immagine di riferimento ha cornice (larghezza tot = stride) mentre immagine input no (largh tot = width)
	int im_offset     = 16 * (blockY * stride + blockX) + (2 * threadIdx.y * stride + 4 * threadIdx.x) + 32 * (stride + 1);
	int im_offset_raw = 16 * (blockY * width  + blockX) + (2 * threadIdx.y * width  + 4 * threadIdx.x);

	const uint8_t *refptr = ref_frame + im_offset;
	const uint8_t *imptr  = in_frame  + im_offset_raw;
	int delta_img = (1 * width);	// riga successiva
	int delta_ref = (1 * stride);

	unsigned int img0    = (uint32_t)  ( (*(imptr + 3) << 24) | (*(imptr + 2) << 16) | (*(imptr + 1) << 8) | *(imptr) );
	unsigned int img1    = (uint32_t)  ( (*(imptr + delta_img + 3) << 24) | (*(imptr + delta_img + 2) << 16) | (*(imptr + delta_img + 1) << 8) | *(imptr + delta_img) );

	unsigned int ref0, ref1;

	for (i = 0; i < 128; i++){
		const uint8_t *refp = refptr + offset_16x12[i];
		ref0 = (uint32_t)( *(refp + 3) << 24 | *(refp + 2) << 16 | *(refp + 1) << 8 | *(refp) );
		ref1 = (uint32_t)( *(refp + delta_ref + 3) << 24 | *(refp + delta_ref + 2) << 16 | *(refp + delta_ref + 1) << 8 | *(refp + delta_ref) );
		sad_result  = __vabsdiff4( img0, ref0 );
		sad_result += __vabsdiff4( img1, ref1 );
		diff[i][TID]  = sad_result;
	}
	__syncthreads();

	// Accumulazione
	for (i = 0; i < 16; i++) {
		diff[TID][i]    += diff[TID][i+16];
		diff[TID+32][i] += diff[TID+32][i+16];
		diff[TID+64][i] += diff[TID+64][i+16];
		diff[TID+96][i] += diff[TID+96][i+16];
	}
	__syncthreads();

	for (i = 0; i < 8; i++) {
		diff[TID][i]    += diff[TID][i+8];
		diff[TID+32][i] += diff[TID+32][i+8];
		diff[TID+64][i] += diff[TID+64][i+8];
		diff[TID+96][i] += diff[TID+96][i+8];
	}
	__syncthreads();

	for (i = 0; i < 4; i++) {
		diff[TID][i]    += diff[TID][i+4];
		diff[TID+32][i] += diff[TID+32][i+4];
		diff[TID+64][i] += diff[TID+64][i+4];
		diff[TID+96][i] += diff[TID+96][i+4];
	}
	__syncthreads();

	diff[TID][0]    += (diff[TID][1]    + diff[TID][2]    + diff[TID][3]);
	diff[TID+32][0] += (diff[TID+32][1] + diff[TID+32][2] + diff[TID+32][3]);
	diff[TID+64][0] += (diff[TID+64][1] + diff[TID+64][2] + diff[TID+64][3]);
	diff[TID+96][0] += (diff[TID+96][1] + diff[TID+96][2] + diff[TID+96][3]);
	__syncthreads();

	// Find MINIMUM (and corresponding best MV) of 128 Pts - 32 threads ////////////////////////
	//
	minpos[TID]    = TID;
	minpos[TID+32] = TID+32;
	minpos[TID+64] = TID+64;
	minpos[TID+96] = TID+96;
	__syncthreads();

	if( diff[TID][0] > diff[TID+32][0] ) {
		diff[TID][0] = diff[TID+32][0];
		minpos[TID] = minpos[TID+32];
	}
	if( diff[TID][0] > diff[TID+64][0] ) {
		diff[TID][0] = diff[TID+64][0];
		minpos[TID] = minpos[TID+64];
	}
	if( diff[TID][0] > diff[TID+96][0] ) {
		diff[TID][0] = diff[TID+96][0];
		minpos[TID] = minpos[TID+96];
	}
	__syncthreads();

	if( TID < 16 )	// 16 threads
		if( diff[TID][0] > diff[TID + 16][0] )	{
			diff[TID][0] = diff[TID + 16][0];
			minpos[TID] = minpos[TID + 16];
		}
	__syncthreads();

	if( TID < 8 )
		if( diff[TID][0] > diff[TID + 8][0] )	{
			diff[TID][0] = diff[TID + 8][0];
			minpos[TID] = minpos[TID + 8];
		}
	__syncthreads();

	if( TID < 4 )
		if( diff[TID][0] > diff[TID + 4][0] )	{
			diff[TID][0] = diff[TID + 4][0];
			minpos[TID] = minpos[TID + 4];
		}
	__syncthreads();

	if( TID < 2 )
		if( diff[TID][0] > diff[TID + 2][0] )	{
			diff[TID][0] = diff[TID + 2][0];
			minpos[TID] = minpos[TID + 2];
		}
	__syncthreads();

	if( TID == 0 )	{
		if( diff[0][0] > diff[1][0] )	{
			diff[0][0] = diff[1][0];
			minpos[0] = minpos[1];
		}
		MVs_g[MBoffset].as_mv = MV_16x12_lookup[ minpos[0] ];
	}
	__syncthreads();

	///////////////////////////////////////////////////////////////////////////////////////////
	// STEP 2: pixel-scale Motion Vector Search

	// Update RefPointer to the best motion vector
	refptr += offset_16x12[ minpos[0] ] ;

	// Compute pixel differences: //////////
	//

	for (i = 0; i < 16; i++){
		const uint8_t *refp = refptr + offset_16x12_refin[i];
  		ref0 = (uint32_t)( *(refp + 3) << 24 | *(refp + 2) << 16 | *(refp + 1) << 8 | *(refp) );
  		ref1 = (uint32_t)( *(refp + delta_ref + 3) << 24 | *(refp + delta_ref + 2) << 16 | *(refp + delta_ref + 1) << 8 | *(refp + delta_ref) );
  		sad_result  = __vabsdiff4( img0, ref0 );
  		sad_result += __vabsdiff4( img1, ref1 );
  		diff[i][TID] = sad_result;
	}
	__syncthreads();

	// accumulazione su 32 thread
	// (16 calcolati inutilmente)
	for (i=0; i<16; i++)
		diff[TID][i] += diff[TID][i+16];
	__syncthreads();
	for (i=0; i<8; i++)
		diff[TID][i] += diff[TID][i+8];
	__syncthreads();
	for (i=0; i<4; i++)
		diff[TID][i] += diff[TID][i+4];
	__syncthreads();
	diff[TID][0]    += (diff[TID][1] + diff[TID][2] + diff[TID][3]);
	__syncthreads();

	// Find MINIMUM (and corresponding best MV) of 16 Pts - 16 threads ////////////////////////
	//
	minpos[TID] = TID;
	__syncthreads();

    if( TID < 8 )	// 8 threads
		if( diff[TID][0] > diff[TID+8][0] )	{
			diff[TID][0] = diff[TID+8][0];
			minpos[TID] = minpos[TID+8];
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
		MVs_g[MBoffset].as_mv.row += MV_lookup_refin_fast[ minpos[0] ].row;	// Added to the previous MV
		MVs_g[MBoffset].as_mv.col += MV_lookup_refin_fast[ minpos[0] ].col;	// Added to the previous MV
		MVs_g[MBoffset].as_mv.row <<= 3;
		MVs_g[MBoffset].as_mv.col <<= 3;

	}
}


inline void me_kernel_launch_split( VP8_COMMON * const common, const uint8_t * const in_frame, const uint8_t * const ref_frame,
		int const streamID, int_mv * const MVs ) {

#if CUDA_VERBOSE
	float elapsedTime;
	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	CHECK(cudaEventRecord(start));
#endif

	me_cuda_split <<< common->GPU.gridDim, common->GPU.blockDim, 0, common->GPU.streams.frame[streamID] >>> (in_frame, ref_frame,
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

void me_cuda_launch_interleaved_split( VP8_COMMON * const cm, int fb_idx, int ref_frame_flags ) {

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
			me_kernel_launch_split(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->lst_fb_idx], s, (cm->gpu_frame.MVs_g)[0] );
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[0][offset],		&(cm->gpu_frame.MVs_g)[0][offset],		MV_size_16,      cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}

		// Se ref_frame_flags indica la presenza di un gold e se il flag del fb puntato da gld_fb_idx indica che e' gold, allora...
		if ((ref_frame_flags & GPUFLAG_GOLD_FRAME) && (cm->yv12_fb[cm->gld_fb_idx].flags & GPUFLAG_GOLD_FRAME)) {
			me_kernel_launch_split(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->gld_fb_idx], s, (cm->gpu_frame.MVs_g)[1] );
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[1][offset],		&(cm->gpu_frame.MVs_g)[1][offset],		MV_size_16,      cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}

		// Se ref_frame_flags indica la presenza di un altref e se il flag del fb puntato da alt_fb_idx indica che e' altref, allora...
		if ((ref_frame_flags & GPUFLAG_ALTR_FRAME) && (cm->yv12_fb[cm->alt_fb_idx].flags & GPUFLAG_ALTR_FRAME)) {
			me_kernel_launch_split(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->alt_fb_idx], s, (cm->gpu_frame.MVs_g)[2] );
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[2][offset],		&(cm->gpu_frame.MVs_g)[2][offset],		MV_size_16,		 cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}
	}
}


void me_cuda_launch_not_interleaved_split( VP8_COMMON * const cm, int fb_idx, int ref_frame_flags ) {

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
			me_kernel_launch_split(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->lst_fb_idx], s, (cm->gpu_frame.MVs_g)[0] );
		}

		// Se ref_frame_flags indica la presenza di un gold e se il flag del fb puntato da gld_fb_idx indica che e' gold, allora...
		if ((ref_frame_flags & GPUFLAG_GOLD_FRAME) && (cm->yv12_fb[cm->gld_fb_idx].flags & GPUFLAG_GOLD_FRAME)) {
			me_kernel_launch_split(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->gld_fb_idx], s, (cm->gpu_frame.MVs_g)[1] );
		}

		// Se ref_frame_flags indica la presenza di un altref e se il flag del fb puntato da alt_fb_idx indica che e' altref, allora...
		if ((ref_frame_flags & GPUFLAG_ALTR_FRAME) && (cm->yv12_fb[cm->alt_fb_idx].flags & GPUFLAG_ALTR_FRAME)) {
			me_kernel_launch_split(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->alt_fb_idx], s, (cm->gpu_frame.MVs_g)[2] );
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
