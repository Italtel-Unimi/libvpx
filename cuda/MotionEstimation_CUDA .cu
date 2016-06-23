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

#ifdef __cplusplus
extern "C" {
#endif
#if HAVE_CUDA_ENABLED_DEVICE

__constant__ int MV_offset4[128];
__constant__ int MV_offset_refin[32];

void setup_constant_mem(int img_stride) {
	int I = img_stride;
	int MV_off4[128] = {                     -22*I,
	                                         -20*I,
	                                 -18*I-4,-18*I,-18*I+4,
	                                 -16*I-4,-16*I,-16*I+4,
	                         -14*I-8,-14*I-4,-14*I,-14*I+4,-14*I+8,
	                         -12*I-8,-12*I-4,-12*I,-12*I+4,-12*I+8,
	                -10*I-12,-10*I-8,-10*I-4,-10*I,-10*I+4,-10*I+8,-10*I+12,
	                 -8*I-12, -8*I-8, -8*I-4, -8*I, -8*I+4, -8*I+8, -8*I+12,
	                 -6*I-12, -6*I-8, -6*I-4, -6*I, -6*I+4, -6*I+8, -6*I+12,
	        -4*I-16, -4*I-12, -4*I-8, -4*I-4, -4*I, -4*I+4, -4*I+8, -4*I+12, -4*I+16,
	        -2*I-16, -2*I-12, -2*I-8, -2*I-4, -2*I, -2*I+4, -2*I+8, -2*I+12, -2*I+16,
	-24, -20,   -16,     -12,     -8,     -4,    0,      4,      8,      12,      16,  20,  24,
	         2*I-16,  2*I-12,  2*I-8,  2*I-4,  2*I,  2*I+4,  2*I+8,  2*I+12,  2*I+16,
	         4*I-16,  4*I-12,  4*I-8,  4*I-4,  4*I,  4*I+4,  4*I+8,  4*I+12,  4*I+16,
	                  6*I-12,  6*I-8,  6*I-4,  6*I,  6*I+4,  6*I+8,  6*I+12,
	                  8*I-12,  8*I-8,  8*I-4,  8*I,  8*I+4,  8*I+8,  8*I+12,
	                  10*I-12,10*I-8, 10*I-4, 10*I, 10*I+4, 10*I+8, 10*I+12,
	                          12*I-8, 12*I-4, 12*I, 12*I+4, 12*I+8,
	                          14*I-8, 14*I-4, 14*I, 14*I+4, 14*I+8,
	                                  16*I-4, 16*I, 16*I+4,
	                                  18*I-4, 18*I, 18*I+4,
	                                          20*I,
	                                          22*I, 22*I+4,
	                    };
    int MV_refin[32] = {
											 -3*I,
							 -2*I-2, -2*I-1, -2*I, -2*I+1, -2*I+2,
						-I-3,  -I-2,   -I-1,   -I,   -I+1,   -I+2,  -I+3,
						  -3,    -2,     -1,            1,      2,     3,
						 I-3,   I-2,    I-1,    I,    I+1,    I+2,   I+3,
							  2*I-2,  2*I-1,  2*I,  2*I+1,  2*I+2,
											  3*I
    };

    CHECK(cudaMemcpyToSymbol(MV_offset4,   MV_off4, 128*sizeof(int)));
    CHECK(cudaMemcpyToSymbol(MV_offset_refin, MV_refin, 32*sizeof(int)));
}


__device__ __constant__ MV MV_lookup4[128] =   {	// Unit: pixel
                                                     {-22,0},
                                                     {-20,0},
                                            {-18,-4},{-18,0},{-18,4},
                                            {-16,-4},{-16,0},{-16,4},
                                   {-14,-8},{-14,-4},{-14,0},{-14,4},{-14,8},
                                   {-12,-8},{-12,-4},{-12,0},{-12,4},{-12,8},
                         {-10,-12},{-10,-8},{-10,-4},{-10,0},{-10,4},{-10,8},{-10,12},
                         { -8,-12},{ -8,-8},{ -8,-4},{ -8,0},{ -8,4},{ -8,8},{ -8,12},
                         { -6,-12},{ -6,-8},{ -6,-4},{ -6,0},{ -6,4},{ -6,8},{ -6,12},
                {-4,-16},{ -4,-12},{ -4,-8},{ -4,-4},{ -4,0},{ -4,4},{ -4,8},{ -4,12},{ -4,16},
                {-2,-16},{ -2,-12},{ -2,-8},{ -2,-4},{ -2,0},{ -2,4},{ -2,8},{ -2,12},{ -2,16},
{0,-24},{0,-20},{ 0,-16},{  0,-12},{  0,-8},{  0,-4},{  0,0},{  0,4},{  0,8},{  0,12},{  0,16},{  0,20},{  0,24},
                { 2,-16},{  2,-12},{  2,-8},{  2,-4},{  2,0},{  2,4},{  2,8},{  2,12},{  2,16},
                { 4,-16},{  4,-12},{  4,-8},{  4,-4},{  4,0},{  4,4},{  4,8},{  4,12},{  4,16},
                         {  6,-12},{  6,-8},{  6,-4},{  6,0},{  6,4},{  6,8},{  6,12},
                         {  8,-12},{  8,-8},{  8,-4},{  8,0},{  8,4},{  8,8},{  8,12},
                         { 10,-12},{ 10,-8},{ 10,-4},{ 10,0},{ 10,4},{ 10,8},{ 10,12},
                                   { 12,-8},{ 12,-4},{ 12,0},{ 12,4},{ 12,8},
                                   { 14,-8},{ 14,-4},{ 14,0},{ 14,4},{ 14,8},
                                            { 16,-4},{ 16,0},{ 16,4},
                                            { 18,-4},{ 18,0},{ 18,4},
                                                     { 20,0},
                                                     { 22,0},{ 22,4}
						};

// Ne basterebbero molte meno (17, per la precisione), ma cosi' riempiamo un warp
__device__ __constant__ MV MV_lookup_refin[32] = {
		                              {-3, 0},
		          {-2, -2}, {-2, -1}, {-2, 0}, {-2, 1}, {-2, 2},
		{-1, -3}, {-1, -2}, {-1, -1}, {-1, 0}, {-1, 1}, {-1, 2}, {-1, 3},
		{ 0, -3}, { 0, -2}, { 0, -1},          { 0, 1}, { 0, 2}, { 0, 3},
		{ 1, -3}, { 1, -2}, { 1, -1}, { 1, 0}, { 1, 1}, { 1, 2}, { 1, 3},
		          { 2, -2}, { 2, -1}, { 2, 0}, { 2, 1}, { 2, 2},
                                      { 3, 0}
};


__inline__ __device__ uint32_t __vabsdiff4( uint32_t u, uint32_t v )
{
	uint32_t w = 0;
	uint32_t ww = 0;
	asm volatile("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(ww) : "r"(u), "r"(v), "r"(w));
	return ww;
}

// Called with a (ROWS,COLS,1) GRID of (8x16x1) blocks
// block: 4x8x1, ogni blocco calcola la me per un MB
// grid: 16 ?
__global__ void ME_CUDA_p ( const uint8_t * const in_frame, const uint8_t * const ref_frame,
                            int const streamID, int const stride, int const width, int const num_MB_width,
                            int_mv * const MVs_g, int * const MV_vars_g )
{

	__shared__ uint32_t sad[128][32];
	__shared__  int32_t minpos[128];
	uint32_t i;

	int32_t TID = threadIdx.y * blockDim.x + threadIdx.x;
	// 16 blocks per grid (16,1,1)
	int32_t MBoffset = streamID * 16 + blockIdx.x;
	int32_t blockY = MBoffset / num_MB_width;
	int32_t blockX = MBoffset % num_MB_width;

	//if (( MBoffset == 3010 ) && ( TID == 10 ))
	//	printf( "%d %d ", blockX, blockY);

    // Occhio alle dimensioni: ref_frame ha la cornice, raw_frame no
	int32_t img_offset = 16 * (blockY * width  + blockX) + 2 * threadIdx.y * width  + threadIdx.x;
	int32_t ref_offset = 16 * (blockY * stride + blockX) + 2 * threadIdx.y * stride + threadIdx.x;

    uint8_t * img = (uint8_t *) in_frame  + img_offset;
	uint8_t * ref = (uint8_t *) ref_frame + ref_offset + 32 * (stride + 1);

	// one thread loads two quad pixels, one 4x8 block covers one img MB
	// ind0: 0-31, relative position of the first quad with respect to the first MB pixel
	int32_t delta_img = (1 * width);
	int32_t delta_ref = (1 * stride);
    // Senor... no capito, Senor...

    // Valori dell'immagine di input
    // Ogni thread carica 4 pixel (in un int) del MB di riferimento
    uint32_t img0    = (uint32_t)  ( (*img << 24) | (*(img + 1) << 16) | (*(img + 2) << 8) | *(img + 3) ); //*img;
	uint32_t img1    = (uint32_t)  ( (*(img + delta_img) << 24) | (*(img + delta_img + 1) << 16) | (*(img + delta_img + 2) << 8) | *(img + delta_img + 3) ); //*(img + delta_img);
    //uint8_t *imgd = img + delta_img;
    //uint32_t img0    = *( (uint32_t *)(img) );
	//uint32_t img1    = *( (uint32_t *)(img) );// + delta_img) );


    // Puntatori e valori dell'immagine di riferimento (no init)
    //uint8_t *refp;
    //uint8_t *refpd;2 *
    uint32_t ref0;
    uint32_t ref1;

    // Valori di out calcolati dalle sad
	uint32_t result;
    //uint32_t result1;

//	ref0=0x01020304;
//	ref1=0x05060708;8
//	img0=0x01010101;
//	img1=0x01010101;
	// Compute pixel differences: //
    asm(".reg .u64	ss<4>;"::);
    asm(".reg .u32	st<4>;"::);
    asm(".reg .u32	rr<2>;"::);
	asm("	mov.u32		st0, %0;"::"r"(img0));
	//asm( "	mov.u32		st1, %0;"::"r"(img1));

    //asm("	mov.u64		ss0, %0;"::"l"(img));
    //asm("	mov.u64		ss1, %0;"::"l"(img));
    //asm("	ld.global.u32	st0, [ss0];"::);
    //asm("	ld.global.u32	st1, [ss1];"::);
    // ss0 : *img0
    // ss1 : *img1
    // ss1 : *ref0
    // ss3 : *ref1
    // st0 : img0
	// st1 : img1
	// st1 : ref0
	// st3 : ref1
    // rr0 : risult
    // rr1 : risult1

	for(i=0; i < 128; i++)
	{
		const uint8_t *refp = ref + MV_offset4[i];
		//refpd = refp + delta_ref;

		//result  = abs( refp[0] - img[0] ) + abs( refp[1] - img[1] ) + abs( refp[2] - img[2] ) + abs( refp[3] - img[3] );
		//result += abs( refpd[0] - imgd[0] ) + abs( refpd[1] - imgd[1] ) + abs( refpd[2] - imgd[2] ) + abs( refpd[3] - imgd[3] );
		ref0 = (uint32_t)( *(refp) << 24 | *(refp + 1) << 16 | *(refp + 2) << 8 | *(refp + 3) );
		ref1 = (uint32_t)( *(refp + delta_ref) << 24 | *(refp + delta_ref + 1) << 16 | *(refp + delta_ref + 2) << 8 | *(refp + delta_ref + 3) );
		//asm("	mov.u64		ss2, %0;"::"l"(ref));
		//asm("	mov.u64		ss3, %0;"::"l"(ref));
		//asm(" 	mov.u32		rr0, 0;"::);
		//asm(" 	mov.u32		rr1, 0;"::);
		//asm("	ld.global.u32	st2, [ss2];"::);
		//asm("	ld.global.u32	st3, [ss3];"::);
		//asm("	mov.u32		st2, %0;"::"r"(ref0));
		//asm("	mov.u32		st3, %0;"::"r"(ref1));result
		//asm("	vabsdiff4.u32.u32.u32.add rr0, st0, st2, rr1;"::);
		//asm("	vabsdiff4.u32.u32.u32.add rr1, st1, st3, rr0;"::);

		//uint32_t result1;
		//asm("	mov.u32	%0, rr0;":"=r"(result):);
		//ref0 = *( (uint32_t *)(ref) );// + MV_offset4[i]
		//ref1 = *( (uint32_t *)(ref) );// + MV_offset4[i] + delta_ref) );


        //result  = 0;
		//result1 = 0;

		//asm(" .reg .u32 r1;\n\t");
		//asm(" vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;\n\t": "=r"(result) : "r" (img0), "r" (ref0), "r" (result1));
		//asm(" vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;\n\t": "=r"(result1) : "r" (img1), "r" (ref1), "r" (result));
		//" vabsdiff4.u32.u32.u32.add %0, %3, %4, r1;\n\t"
		//" vabsdiff4.u32.u32.u32.add r1, %3, %4, r1;\n\t"
		//" mov.u32 %0, r1;\n\t"

		result = 0;
		result += abs( *(refp) - *(img));
		result += abs( *(refp + 1) - *(img + 1));
		result += abs( *(refp + 2) - *(img + 2));
		result += abs( *(refp + 3) - *(img + 3));
		result += abs( *(refp + delta_ref) - *(img + delta_img));
		result += abs( *(refp + 1 + delta_ref) - *(img + 1 + delta_img));
		result += abs( *(refp + 2 + delta_ref) - *(img + 3 + delta_img));
		result += abs( *(refp + 2 + delta_ref) - *(img + 3 + delta_img));

		//result  = __vabsdiff4( img0, ref0 );
		//result += __vabsdiff4( img1, ref1 );


		sad[i][TID] = result;
	}
	__syncthreads();

	// accumulate diff, 32 -> 16: 128 righe e 32 colonne
	for (i=0; i<16; i++)
		sad[TID][i]+=sad[TID][i+16];
	__syncthreads();

	for (i=0; i<16; i++)
		sad[TID+32][i]+=sad[TID + 32][i+16];
	__syncthreads();

	for (i=0; i<16; i++)
		sad[TID+64][i]+=sad[TID+64][i+16];
	__syncthreads();

	for (i=0; i<16; i++)
		sad[TID+96][i]+=sad[TID+96][i+16];
	__syncthreads();

	// accumulate diff, 16 -> 8: 128 righe e 16 colonne
	for (i=0; i<8; i++)
		sad[TID][i]+=sad[TID][i+8];
	__syncthreads();

	for (i=0; i<8; i++)
		sad[TID+32][i]+=sad[TID+32][i+8];
	__syncthreads();

	for (i=0; i<8; i++)
		sad[TID+64][i]+=sad[TID+64][i+8];
	__syncthreads();

	for (i=0; i<8; i++)
		sad[TID+96][i]+=sad[TID+96][i+8];
	__syncthreads();


	// accumulate diff, 8 -> 4: 128 righe e 8 colonne
	for (i=0; i<4; i++)
		sad[TID][i]+=sad[TID][i+4];
	__syncthreads();

	for (i=0; i<4; i++)
		sad[TID+32][i]+=sad[TID+32][i+4];
	__syncthreads();

	for (i=0; i<4; i++)
		sad[TID+64][i]+=sad[TID+64][i+4];
	__syncthreads();

	for (i=0; i<4; i++)
		sad[TID+96][i]+=sad[TID+96][i+4];
	__syncthreads();

	// accumulate diff, 4 -> 2 128 righe e 4 colonne
	for (i=0; i<2; i++)
		sad[TID][i]+=sad[TID][i+2];
	__syncthreads();

	for (i=0; i<2; i++)
		sad[TID+32][i]+=sad[TID+32][i+2];
	__syncthreads();

	for (i=0; i<2; i++)
		sad[TID+64][i]+=sad[TID+64][i+2];
	__syncthreads();

	for (i=0; i<2; i++)
		sad[TID+96][i]+=sad[TID+96][i+2];
	__syncthreads();

	// accumulate diff, 2 -> 1
	for (i=0; i<1; i++)
		sad[TID][i]+=sad[TID][i+1];
	__syncthreads();

	for (i=0; i<1; i++)
		sad[TID+32][i]+=sad[TID+32][i+1];
	__syncthreads();

	for (i=0; i<1; i++)
		sad[TID+64][i]+=sad[TID+64][i+1];
	__syncthreads();

	for (i=0; i<1; i++)
		sad[TID+96][i]+=sad[TID+96][i+1];
	__syncthreads();

	// Find MINIMUM (and corresponding best MV) of 128 Pts - 32 threads //
    //
	minpos[TID] = TID;
	//__syncthreads();	// serve?
	minpos[32+TID]=32+TID;
	//__syncthreads();	// SERVE?
	minpos[64+TID]=64+TID;
	//__syncthreads();	// SERVEEEEEE??????
	minpos[96+TID]=96+TID;
	__syncthreads();


	if( sad[TID][0] < sad[TID+32][0] )
	{
		sad[TID][0] = sad[TID+32][0];
		minpos[TID] = minpos[TID+32];
	}
	__syncthreads();

	if( sad[TID][0] < sad[TID+64][0] )
	{
		sad[TID][0] = sad[TID+64][0];
		minpos[TID] = minpos[TID+64];
	}
	__syncthreads();

	if( sad[TID][0] < sad[TID+96][0] )
	{
		sad[TID][0] = sad[TID+96][0];
		minpos[TID] = minpos[TID+96];
	}
	__syncthreads();



	if( TID < 16 )	// 16 threads
		if( sad[TID][0] < sad[TID+16][0] )	{
			sad[TID][0] = sad[TID+16][0];
			minpos[TID] = minpos[TID+16];
		}
	__syncthreads();

	if( TID < 8 )	// 8 threads
		if( sad[TID][0] < sad[TID+8][0] )	{
			sad[TID][0] = sad[TID+8][0];
			minpos[TID] = minpos[TID+8];
		}
	__syncthreads();

	if( TID < 4 )	// 4 threads
		if( sad[TID][0] < sad[TID + 4][0] )	{
			sad[TID][0] = sad[TID + 4][0];
			minpos[TID] = minpos[TID + 4];
		}
	__syncthreads();

	if( TID < 2 )	// 2 threads
		if( sad[TID][0] < sad[TID + 2][0] )	{
			sad[TID][0] = sad[TID + 2][0];
			minpos[TID] = minpos[TID + 2];
		}
	__syncthreads();

	int minsad;
	if( TID == 0 )	// Only thread 0
	{
		if( sad[0][0] < sad[1][0] )	{
			sad[0][0] = sad[1][0];
			minpos[0] = minpos[1];
		}
		// And finally assign resulting MV
		//MVs_g[MBoffset].as_mv = MV_lookup4[ minpos[0] ];
		MVs_g[MBoffset].as_mv.row = MV_lookup4[ minpos[0] ].row;
		MVs_g[MBoffset].as_mv.col = MV_lookup4[ minpos[0] ].col;
		minsad = sad[0][0];
	}


	// Refining search su diamante interno.
	ref += MV_offset4[minpos[0]];

    // calcolo matrice delle sad
	for(i=0; i < 16; i++) {
		const uint8_t *refp = ref + MV_offset_refin[i];
		//ref0 = (uint32_t)( *(refp) << 24 | *(refp + 1) << 16 | *(refp + 2) << 8 | *(refp + 3) );
		//ref1 = (uint32_t)( *(refp + delta_ref) << 24 | *(refp + delta_ref + 1) << 16 | *(refp + delta_ref + 2) << 8 | *(refp + delta_ref + 3) );
		//result = __vabsdiff4( img0, ref0 );
		//result += __vabsdiff4( img1, ref1 );
		result = 0;
		result += abs( *(refp) - *(img));
		result += abs( *(refp + 1) - *(img + 1));
		result += abs( *(refp + 2) - *(img + 2));
		result += abs( *(refp + 3) - *(img + 3));
		result += abs( *(refp + delta_ref) - *(img + delta_img));
		result += abs( *(refp + 1 + delta_ref) - *(img + 1 + delta_img));
		result += abs( *(refp + 2 + delta_ref) - *(img + 3 + delta_img));
		result += abs( *(refp + 2 + delta_ref) - *(img + 3 + delta_img));
		sad[i][TID] = result;
	}
	__syncthreads();

    // Accumulazione
	// non serve controllo "if TID < 32" perche' thread sono sempre 32
	for (i=0; i<16; i++)
    		sad[TID][i]+=sad[TID][i+16];
    __syncthreads();

	for (i=0; i<8; i++)
    		sad[TID][i]+=sad[TID][i+8];
    __syncthreads();

    for (i=0; i<4; i++)
    		sad[TID][i]+=sad[TID][i+4];
    __syncthreads();

    sad[TID][0] += ( sad[TID][1] + sad[TID][2] + sad[TID][3] );
    __syncthreads();


    // Ricerca del minimo
    minpos[TID] = TID;
    __syncthreads();

    if( TID < 16 )	// 16 threads
		if( sad[TID][0] < sad[TID+16][0] )	{
			sad[TID][0] = sad[TID+16][0];
			minpos[TID] = minpos[TID+16];
		}
	__syncthreads();

    if( TID < 8 )	// 8 threads
		if( sad[TID][0] < sad[TID+8][0] )	{
			sad[TID][0] = sad[TID+8][0];
			minpos[TID] = minpos[TID+8];
		}
	__syncthreads();

	if( TID < 4 )	// 4 threads
		if( sad[TID][0] < sad[TID + 4][0] )	{
			sad[TID][0] = sad[TID + 4][0];
			minpos[TID] = minpos[TID + 4];
		}
	__syncthreads();

	if( TID < 2 )	// 2 threads
		if( sad[TID][0] < sad[TID + 2][0] )	{
			sad[TID][0] = sad[TID + 2][0];
			minpos[TID] = minpos[TID + 2];
		}
	__syncthreads();

    if( TID == 0 )	// Only thread 0
	{
		if( sad[0][0] < sad[1][0] )	{
			sad[0][0] = sad[1][0];
			minpos[0] = minpos[1];
		}

        if ( sad[0][0] < minsad ) {
            MVs_g[MBoffset].as_mv.row += MV_lookup_refin[ minpos[0] ].row;
            MVs_g[MBoffset].as_mv.col += MV_lookup_refin[ minpos[0] ].col;
        }
	}
}


void me_kernel_launch( VP8_COMMON * const common, const uint8_t * const in_frame, const uint8_t * const ref_frame,
		int const streamID, int_mv * const MVs, int * const MV_vars ) {

    ME_CUDA_p <<< common->GPU.gridDim, common->GPU.blockDim, 0, common->GPU.streams.frame[streamID] >>> (in_frame, ref_frame,
		          streamID, common->gpu_frame.stride, common->gpu_frame.width, common->gpu_frame.num_MB_width, MVs, MV_vars );

}

void me_cuda_launch_interleaved( VP8_COMMON * const cm, int fb_idx, int ref_frame_flags ) {

	int MV_size_16 = 16*sizeof(int_mv);
	int MV_vars_size_16 = 16*sizeof(int);
	// for printing informations about reference frame flags and thei usage, I left a commented prinft at line 3625
	// at the beginning of encode_frame_to_data_rate(..) in onyx_if.c

	for (int s = 0; s < cm->GPU.num_mb16th; s++) {

		int offset = 16*s;
		// bugfix per immagini il cui n di mb non e' divisibile per 16
		// prima venivano lanciati troppi processi e cudaMemcpyAsync andava a leggere oltre i limiti degli array
		if (offset + 16 > cm->gpu_frame.num_mv) {
			MV_size_16 = ( offset + 16 - cm->gpu_frame.num_mv ) * sizeof( int_mv );
			MV_vars_size_16 = ( offset + 16 - cm->gpu_frame.num_mv ) * sizeof( int );
		}

		if ((ref_frame_flags & GPUFLAG_LAST_FRAME) && (cm->yv12_fb[cm->lst_fb_idx].flags & GPUFLAG_LAST_FRAME)) {
			me_kernel_launch(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->lst_fb_idx], s, (cm->gpu_frame.MVs_g)[0], (cm->gpu_frame.MV_vars_g)[0] );
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[0][offset],		&(cm->gpu_frame.MVs_g)[0][offset],		MV_size_16,      cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
			//CHECK(cudaMemcpyAsync( &(cm->host_frame.MV_vars_h)[0][offset],	&(cm->gpu_frame.MV_vars_g)[0][offset],	MV_vars_size_16, cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}

		// Se ref_frame_flags indica la presenza di un gold e se il flag del fb puntato da gld_fb_idx indica che e' gold, allora...
		if ((ref_frame_flags & GPUFLAG_GOLD_FRAME) && (cm->yv12_fb[cm->gld_fb_idx].flags & GPUFLAG_GOLD_FRAME)) {
			me_kernel_launch(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->gld_fb_idx], s, (cm->gpu_frame.MVs_g)[1], (cm->gpu_frame.MV_vars_g)[1] );
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[1][offset],		&(cm->gpu_frame.MVs_g)[1][offset],		MV_size_16,      cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
			//CHECK(cudaMemcpyAsync( &(cm->host_frame.MV_vars_h)[1][offset],	&(cm->gpu_frame.MV_vars_g)[1][offset],	MV_vars_size_16, cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}

		// Se ref_frame_flags indica la presenza di un altref e se il flag del fb puntato da alt_fb_idx indica che e' altref, allora...
		if ((ref_frame_flags & GPUFLAG_ALTR_FRAME) && (cm->yv12_fb[cm->alt_fb_idx].flags & GPUFLAG_ALTR_FRAME)) {
			me_kernel_launch(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->alt_fb_idx], s, (cm->gpu_frame.MVs_g)[2], (cm->gpu_frame.MV_vars_g)[2] );
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[2][offset],		&(cm->gpu_frame.MVs_g)[2][offset],		MV_size_16,		 cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
			//CHECK(cudaMemcpyAsync( &(cm->host_frame.MV_vars_h)[2][offset],	&(cm->gpu_frame.MV_vars_g)[2][offset],	MV_vars_size_16, cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}
	}
}


void me_cuda_launch_not_interleaved( VP8_COMMON * const cm, int fb_idx, int ref_frame_flags ) {

	int MV_size_16 = 16*sizeof(int_mv);
	int MV_vars_size_16 = 16*sizeof(int);
	// for printing informations about reference frame flags and thei usage, I left a commented prinft at line 3625
	// at the beginning of encode_frame_to_data_rate(..) in onyx_if.c

	for (int s = 0; s < cm->GPU.num_mb16th; s++) {

		int offset = 16*s;
		// bugfix per immagini il cui n di mb non e' divisibile per 16
		// prima venivano lanciati troppi processi e cudaMemcpyAsync andava a leggere oltre i limiti degli array
		if (offset + 16 > cm->gpu_frame.num_mv) {
			MV_size_16 = ( offset + 16 - cm->gpu_frame.num_mv ) * sizeof( int_mv );
			MV_vars_size_16 = ( offset + 16 - cm->gpu_frame.num_mv ) * sizeof( int );
		}

		if ((ref_frame_flags & GPUFLAG_LAST_FRAME) && (cm->yv12_fb[cm->lst_fb_idx].flags & GPUFLAG_LAST_FRAME)) {
			me_kernel_launch(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->lst_fb_idx], s, (cm->gpu_frame.MVs_g)[0], (cm->gpu_frame.MV_vars_g)[0] );
		}

		// Se ref_frame_flags indica la presenza di un gold e se il flag del fb puntato da gld_fb_idx indica che e' gold, allora...
		if ((ref_frame_flags & GPUFLAG_GOLD_FRAME) && (cm->yv12_fb[cm->gld_fb_idx].flags & GPUFLAG_GOLD_FRAME)) {
			me_kernel_launch(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->gld_fb_idx], s, (cm->gpu_frame.MVs_g)[1], (cm->gpu_frame.MV_vars_g)[1] );
		}

		// Se ref_frame_flags indica la presenza di un altref e se il flag del fb puntato da alt_fb_idx indica che e' altref, allora...
		if ((ref_frame_flags & GPUFLAG_ALTR_FRAME) && (cm->yv12_fb[cm->alt_fb_idx].flags & GPUFLAG_ALTR_FRAME)) {
			me_kernel_launch(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->alt_fb_idx], s, (cm->gpu_frame.MVs_g)[2], (cm->gpu_frame.MV_vars_g)[2] );
		}

		if (ref_frame_flags & GPUFLAG_LAST_FRAME) {
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[0][offset],		&(cm->gpu_frame.MVs_g)[0][offset],		MV_size_16,      cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
			//CHECK(cudaMemcpyAsync( &(cm->host_frame.MV_vars_h)[0][offset],	&(cm->gpu_frame.MV_vars_g)[0][offset],	MV_vars_size_16, cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}

		if ((ref_frame_flags & GPUFLAG_GOLD_FRAME) && (cm->yv12_fb[cm->gld_fb_idx].flags & GPUFLAG_GOLD_FRAME)) {
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[1][offset],		&(cm->gpu_frame.MVs_g)[1][offset],		MV_size_16,      cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
			//CHECK(cudaMemcpyAsync( &(cm->host_frame.MV_vars_h)[1][offset],	&(cm->gpu_frame.MV_vars_g)[1][offset],	MV_vars_size_16, cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}

		if ((ref_frame_flags & GPUFLAG_ALTR_FRAME) && (cm->yv12_fb[cm->alt_fb_idx].flags & GPUFLAG_ALTR_FRAME)) {
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[2][offset],		&(cm->gpu_frame.MVs_g)[2][offset],		MV_size_16,		 cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
			//CHECK(cudaMemcpyAsync( &(cm->host_frame.MV_vars_h)[2][offset],	&(cm->gpu_frame.MV_vars_g)[2][offset],	MV_vars_size_16, cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}
	}
}


#endif  /* HAVE_CUDA_ENABLED_DEVICE */
#ifdef __cplusplus
}
#endif
