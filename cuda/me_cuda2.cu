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


__device__ __constant__ MV MV_16x12_lookup_split[] = {                      {-12,-2}, {-12, 0}, {-12, 2},		// Unit: pixel
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

__device__ __constant__ MV MV_lookup_refin_split[] = {
	                    {-2, 0} ,		// Unit: pixel
	{-1, -2}, {-1, -1}, {-1, 0}, {-1, 1}, {-1, 2},
	          { 0, -1}, { 0, 0}, { 0, 1},
	{ 1, -2}, { 1, -1}, { 1, 0}, { 1, 1}, { 1, 2},
	                    { 2, 0},
	{0, 0} };  // in piu', per arrivare a 16

__constant__ int offset_16x12__[128];
__constant__ int offset_16x12_refin__[16];

void setup_constant_mem_split(int img_stride) {
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
			0 };		// in piu', per arrivare a 128
	int off_16x12_refin[] = { -2*I ,
			-I-2, -I-1, -I, -I+1, -I+2,
			-1,  0,  1,
			-I-2, I-1, I, I+1, I+2,
			2*I,
			0 };  // in piu', per arrivare a 16

	// copy to device constant memory
	CHECK(cudaMemcpyToSymbol(offset_16x12__, off_16x12, 128*sizeof(int)));
	CHECK(cudaMemcpyToSymbol(offset_16x12_refin__, off_16x12_refin, 16*sizeof(int)));
}


__inline__ __device__ uint32_t __vabsdiff4( uint32_t u, uint32_t v )
{
	uint32_t w = 0;
	asm volatile("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(w) : "r"(u), "r"(v), "r"(w));
	return w;
}


__global__ void me_cuda_split (const uint8_t * __restrict__ const in_frame, const uint8_t * __restrict__ const ref_frame,
				int const streamID, int const streamSize, int const stride, int const width, int const num_MB_width, int const split_on,
				int_mv * __restrict__ const MVs_g, int_mv * __restrict__ const MVs_split_g ) {

	__shared__ uint16_t diff[128][32]; // Risky! It might overflow in one pathologic instance
	__shared__ uint8_t minpos[32];
	__shared__ uint8_t minpos_refin[32];

	// configurazione di lancio: blocks per grid:   16 x 1 x 1
	//							 threads per block:  4 x 8 x 1

	int32_t TID = threadIdx.y * blockDim.x + threadIdx.x;	// Thread Index (0..32)
	int32_t i, j;

	int32_t MBoffset = streamID * streamSize + blockIdx.x;
	int32_t blockX = MBoffset % num_MB_width;		// colonna
	int32_t blockY = MBoffset / num_MB_width;		// riga
	// Occhio: immagine di riferimento ha cornice (larghezza tot = stride) mentre immagine input no (largh tot = width)
	int32_t im_offset     = 16 * (blockY * stride + blockX) + (2 * threadIdx.y * stride + 4 * threadIdx.x) + 32 * (stride + 1);
	int32_t im_offset_raw = 16 * (blockY * width  + blockX) + (2 * threadIdx.y * width  + 4 * threadIdx.x);

	const uint8_t *refptr = ref_frame + im_offset;
	const uint8_t *imptr  = in_frame  + im_offset_raw;
	uint32_t delta_img = (1 * width);
	uint32_t delta_ref = (1 * stride);

	uint32_t img0    = (uint32_t)  ( (*(imptr + 3) << 24)             | (*(imptr + 2) << 16)             | (*(imptr + 1) << 8)             | *(imptr) );
	uint32_t img1    = (uint32_t)  ( (*(imptr + delta_img + 3) << 24) | (*(imptr + delta_img + 2) << 16) | (*(imptr + delta_img + 1) << 8) | *(imptr + delta_img) );
	uint32_t ref0, ref1;

	/* Organizzazione dei thread all'interno del macroblocco. Ogni thread considera 4 pixel e i 4 immediatamente sottostanti.
		Accesso a memoria globale non e' ottimale (coalescenza a gruppi di quattro), ma questo schema permette di raggruppare
		le sad in somme parziali per calcolare tutte le splitmv.
	 ╔══════════╦══════════╦══════════╦══════════╗
	 ║  TID  0  ║  TID  1  ║  TID  2  ║  TID  3  ║
     ╟──────────╫──────────╫──────────╫──────────╢
     ║  TID  4  ║  TID  5  ║  TID  6  ║  TID  7  ║
     ╠══════════╬══════════╬══════════╬══════════╣
	 ║  TID  9  ║  TID  9  ║  TID 10  ║  TID 11  ║
     ╟──────────╫──────────╫──────────╫──────────╢
     ║  TID 12  ║  TID 13  ║  TID 14  ║  TID 15  ║
     ╠══════════╬══════════╬══════════╬══════════╣
	 ║  TID 16  ║  TID 17  ║  TID 18  ║  TID 19  ║
     ╟──────────╫──────────╫──────────╫──────────╢
     ║  TID 20  ║  TID 21  ║  TID 22  ║  TID 23  ║
     ╠══════════╬══════════╬══════════╬══════════╣
	 ║  TID 24  ║  TID 25  ║  TID 26  ║  TID 27  ║
     ╟──────────╫──────────╫──────────╫──────────╢
     ║  TID 28  ║  TID 29  ║  TID 30  ║  TID 31  ║
     ╚══════════╩══════════╩══════════╩══════════╝
	 */


	 /*
		Calcolo delle sad, risultati memorizzati nella matrice diff.
                32          32 TID = 32 sotto blocchi, ognuno contenente sad parziali
		 /             \
		┌───────────────┐
		│               │
		│               │
		│               │
		│ diff[128][32] │   128 candidati mv
		│               │
		│               │
		│               │
		└───────────────┘
		Ogni thread si fa carico si un sottoblocco di 8 pixel e calcola la sad per ogni
		candidato mv
	 */
	for (i = 0; i < 128; i++){
		const uint8_t *refp = refptr + offset_16x12__[i];
		int32_t sad_result;
		ref0 = (uint32_t)( *(refp + 3) << 24             | *(refp + 2) << 16             | *(refp + 1) << 8             | *(refp) );
		ref1 = (uint32_t)( *(refp + delta_ref + 3) << 24 | *(refp + delta_ref + 2) << 16 | *(refp + delta_ref + 1) << 8 | *(refp + delta_ref) );
		sad_result  = __vabsdiff4( img0, ref0 );
		sad_result += __vabsdiff4( img1, ref1 );
		diff[i][TID]  = sad_result;
	}
	__syncthreads();

	// Accumulazione delle colonne di diff in modo da formare sad di blocchi per ogni candidato mv
	// Prima reduction, generazione 16 sad 4x4
	// 0   1   2   3   |   8   9  10  11  |  16  17  18  19  |  24  25  26  27  <- j
	// ^   ^   ^   ^   |   ^   ^   ^   ^  |   ^   ^   ^   ^  |   ^   ^   ^   ^
	// 4   5   6   7   |  12  13  14  15  |  20  21  22  23  |  28  29  30  31  <- j + 4
	for (i = 0; i < 16; i++) {
		j = i + (i / 4) * 4;
		diff[TID   ][j] += diff[TID   ][j+4];
		diff[TID+32][j] += diff[TID+32][j+4];
		diff[TID+64][j] += diff[TID+64][j+4];
		diff[TID+96][j] += diff[TID+96][j+4];
	}
	__syncthreads();

	// Seconda reduction, generazione 4 sad 8x8
	//       4        |       12        |        20        |        28        <- (8 * i) + 4
	//       ^        |        ^        |         ^        |         ^
	// 0   1   8   9  |  2   3  10  11  |  16  17  24  25  |  18  19  26  27  <- [j j+1 j+8 j+9]
	for (i = 0; i < 4; i++) {
		j = 2 * i + (i / 2) * 12;			// genera 0, 2, 16, 18 per i = 0 .. 3
		diff[TID   ][(8 * i) + 4] = diff[TID   ][j] + diff[TID   ][j + 1] + diff[TID   ][j + 8] + diff[TID   ][j + 9];
		diff[TID+32][(8 * i) + 4] = diff[TID+32][j] + diff[TID+32][j + 1] + diff[TID+32][j + 8] + diff[TID+32][j + 9];
		diff[TID+64][(8 * i) + 4] = diff[TID+64][j] + diff[TID+64][j + 1] + diff[TID+64][j + 8] + diff[TID+64][j + 9];
		diff[TID+96][(8 * i) + 4] = diff[TID+96][j] + diff[TID+96][j + 1] + diff[TID+96][j + 8] + diff[TID+96][j + 9];
	}
	__syncthreads();

	// Terza reduction (a), generazione 2 sad 8x16
	//         8x16
	//    22    |    30		<- 22 + (i * 8)
	//     ^    |     ^
	//   4  20  |  12  28
	for (i = 0; i < 2; i++) {
		j = 4 + (8 * i);				// genera 4, 12 per i = 0..1
		diff[TID   ][22 + (i * 8)] = diff[TID   ][j] + diff[TID   ][j + 16];
		diff[TID+32][22 + (i * 8)] = diff[TID+32][j] + diff[TID+32][j + 16];
		diff[TID+64][22 + (i * 8)] = diff[TID+64][j] + diff[TID+64][j + 16];
		diff[TID+96][22 + (i * 8)] = diff[TID+96][j] + diff[TID+96][j + 16];
	}
	__syncthreads(); // potrebbe non servire!

	// Terza reduction (b), generazione 2 sad 16x8
	//       16x8
	//    6    |    14		<- 6*(i+1) + 2*i = 8 * i + 6
	//    ^    |     ^
	//  4  12  |  20  28	<- [j j+8]
	for (i = 0; i < 2; i++) {
		j = 4 + (16 * i);					// genera 4, 20 per i = 0..1
		diff[TID   ][8 * i + 6] = diff[TID   ][j] + diff[TID   ][j + 8];
		diff[TID+32][8 * i + 6] = diff[TID+32][j] + diff[TID+32][j + 8];
		diff[TID+64][8 * i + 6] = diff[TID+64][j] + diff[TID+64][j + 8];
		diff[TID+96][8 * i + 6] = diff[TID+96][j] + diff[TID+96][j + 8];
	}
	__syncthreads();

	// Quarta reduction, generazione 1 sad 16x16
	//    31
	//     ^
	//  6    14
	diff[TID   ][31] = diff[TID   ][6] + diff[TID   ][14];
	diff[TID+32][31] = diff[TID+32][6] + diff[TID+32][14];
	diff[TID+64][31] = diff[TID+64][6] + diff[TID+64][14];
	diff[TID+96][31] = diff[TID+96][6] + diff[TID+96][14];
	__syncthreads();



	// Ricerca del minimo di ogni colonna. A noi interessano 25 delle 32 colonne,
	// ma per non creare divergenza tra i thread eseguiamo la ricerca anche dove non serve
	minpos[TID] = 0;
	__syncthreads();

	// 32 thread, ognuno ricerca il minimo lungo una colonna
	for( i = 1; i < 128; i++ ){
		if ( diff[0][TID] > diff[i][TID] ) {
			diff[0][TID] = diff[i][TID];
			minpos[TID] = i;
		}
	}

	// Salva mv 16x16
	// Questo potrebbe essere fatto meglio, conj 25 thread che lavorano contemporaneamente,
	// ma devo studiare come indicizzare l'accesso alla matrice globale.
	if ( TID == 31 ) {
		MVs_g[MBoffset].as_mv.row = MV_16x12_lookup_split[ minpos[TID] ].row * 8;
		MVs_g[MBoffset].as_mv.col = MV_16x12_lookup_split[ minpos[TID] ].col * 8;
	}

	if (split_on == SPLITMV_ON) {

		// salva mv 4x4
		if ( TID < 16 ) {
			MVs_split_g[MBoffset*24 + TID].as_mv.row  = MV_16x12_lookup_split[ minpos[TID + (TID / 4) * 4] ].row * 8;
			MVs_split_g[MBoffset*24 + TID].as_mv.col  = MV_16x12_lookup_split[ minpos[TID + (TID / 4) * 4] ].col * 8;
		}
		// salva mv 8x8
		if ( TID < 4 ) {
			MVs_split_g[MBoffset*24 + 16 + TID].as_mv.row  = MV_16x12_lookup_split[ minpos[8 * TID + 4] ].row * 8;
			MVs_split_g[MBoffset*24 + 16 + TID].as_mv.col  = MV_16x12_lookup_split[ minpos[8 * TID + 4] ].col * 8;
		}
		// salva mv 8x16 e 16x8
		if ( TID < 2 ) {
			MVs_split_g[MBoffset*24 + 20 + TID].as_mv.row  = MV_16x12_lookup_split[ minpos[8 * TID + 22] ].row * 8;
			MVs_split_g[MBoffset*24 + 22 + TID].as_mv.row  = MV_16x12_lookup_split[ minpos[8 * TID + 6] ].row * 8;
			MVs_split_g[MBoffset*24 + 20 + TID].as_mv.col  = MV_16x12_lookup_split[ minpos[8 * TID + 22] ].col * 8;
			MVs_split_g[MBoffset*24 + 22 + TID].as_mv.col  = MV_16x12_lookup_split[ minpos[8 * TID + 6] ].col * 8;
		}
		__syncthreads();
	}




	///////////////////////////////////////////////////////////////////////////////////////////
	// STEP 2: pixel-scale Motion Vector Search

	// 1.
	// Ricerca di un MV per ogni blocco 4x4
	// 16 blocchi, 2 thread per blocco. Stesso schema per decidere TID => thread 0 e 4 fanno 1 blocco; 1 e 5 il secondo, ecc...
	// Risultati sad memorizzati in diff[i][TID] con 0 < i < 15
	// Questa volta non possiamo piu' sfruttare che refptr punti alla stesso indice, quindi posso
	// calcolare contemporaneamente ogni sad per tid e accumulare, ma posso sfruttare il
	// parallelismo tra mv dello stesso tipo: prima calcolo in parall tutte le 4x4, poi le 8x8, ecc...

	if (split_on == SPLITMV_ON) {

	// Update refpointer al miglior mv
	j = (TID % 4) + (TID / 8) * 8;	// Genera 0 1 2 3 0 1 2 3 8 9 10 11 8 9 10 11 16 17...
									// perche' TID 0 e 4 vengono traslati dello stesso mv corrispondente
									// a quello ora presente in colonna 0 di minpos
	refptr += offset_16x12__[minpos[j]];

	for (i = 0; i < 16; i++) {
		const uint8_t *refp = refptr + offset_16x12_refin__[i];
  		int32_t sad_result;
  		ref0 = (uint32_t)( *(refp + 3) << 24             | *(refp + 2) << 16             | *(refp + 1) << 8             | *(refp) );
  		ref1 = (uint32_t)( *(refp + delta_ref + 3) << 24 | *(refp + delta_ref + 2) << 16 | *(refp + delta_ref + 1) << 8 | *(refp + delta_ref) );
  		sad_result  = __vabsdiff4( img0, ref0 );
  		sad_result += __vabsdiff4( img1, ref1 );
  		diff[i][TID] = sad_result;
	}
	__syncthreads();

	if ( TID < 16 ) {
		for (i = 0; i < 16; i++) {
			j = i + (i / 4) * 4;
			diff[TID][j] += diff[TID][j+4];
		}
	}

	minpos_refin[TID] = 0;
	__syncthreads();

	for( i = 1; i < 16; i++ ){
		if ( diff[0][TID] > diff[i][TID] ) {
			diff[0][TID] = diff[i][TID];
			minpos_refin[TID] = i;
		}
	}

	// salva MV della split 4x4
	if ( TID < 16 ) {
		MVs_split_g[MBoffset*24 + TID].as_mv.row += (MV_lookup_refin_split[ minpos_refin[TID + (TID / 4) * 4] ].row * 8);
		MVs_split_g[MBoffset*24 + TID].as_mv.col += (MV_lookup_refin_split[ minpos_refin[TID + (TID / 4) * 4] ].col * 8);
	}

	// 2.
	// Ricerca di un mv per ogni blocco 8x8
	// Procedura esattamente identica alla precedente: TID che elaborano stesso blocco avranno
	// mv impostato coerentemente. Differente accumulazione (per blocco 0: TID 0 1 4 5 8 9 12 13)

	// Update refpointer al miglior mv
	j = (TID / 8) * 8 + 4;	// Genera 4 4 4 4 4 4 4 4 12 12 12 12 12 12 12 12 20 20 20 20...
	refptr = ref_frame + im_offset + offset_16x12__[ minpos[j] ];

	for (i = 0; i < 16; i++) {
		const uint8_t *refp = refptr + offset_16x12_refin__[i];
  		int32_t sad_result;
  		ref0 = (uint32_t)( *(refp + 3) << 24             | *(refp + 2) << 16             | *(refp + 1) << 8             | *(refp) );
  		ref1 = (uint32_t)( *(refp + delta_ref + 3) << 24 | *(refp + delta_ref + 2) << 16 | *(refp + delta_ref + 1) << 8 | *(refp + delta_ref) );
  		sad_result  = __vabsdiff4( img0, ref0 );
  		sad_result += __vabsdiff4( img1, ref1 );
  		diff[i][TID] = sad_result;
	}
	__syncthreads();

	// Sono pigro, copio e incollo la stessa manfrina
	if ( TID < 16 ) {
		for (i = 0; i < 16; i++) {
			j = i + (i / 4) * 4;
			diff[TID][j] += diff[TID][j+4];
		}
	}
	__syncthreads();
	if ( TID < 16 ) {
		for (i = 0; i < 4; i++) {
			j = 2 * i + (i / 2) * 12;
			diff[TID][(8 * i) + 4] = diff[TID][j] + diff[TID][j + 1] + diff[TID][j + 8] + diff[TID][j + 9];
		}
	}
	__syncthreads();

	minpos_refin[TID] = 0;
	__syncthreads();

	// 32 thread, ognuno ricerca il minimo lungo ogni colonna
	// anche se le colonne interessanti sono solo la 4, 12, 20 e 28
	for( i = 1; i < 16; i++ ){
		if ( diff[0][TID] > diff[i][TID] ) {
			diff[0][TID] = diff[i][TID];
			minpos_refin[TID] = i;
		}
	}
	__syncthreads();

	// Salva i MV della split 8x8
	if ( TID < 4 ) {
		MVs_split_g[MBoffset*24 + 16 + TID].as_mv.row += (MV_lookup_refin_split[ minpos_refin[8 * TID + 4] ].row * 8);
		MVs_split_g[MBoffset*24 + 16 + TID].as_mv.col += (MV_lookup_refin_split[ minpos_refin[8 * TID + 4] ].col * 8);

	}

	}

	// 5.
	// Refining search su blocco 16x16
	// Update RefPointer to the best motion vector
	refptr = ref_frame + im_offset + offset_16x12__[ minpos[31] ];

	for (i = 0; i < 16; i++) {
		const uint8_t *refp = refptr + offset_16x12_refin__[i];
  		int32_t sad_result;
  		ref0 = (uint32_t)( *(refp + 3) << 24             | *(refp + 2) << 16             | *(refp + 1) << 8             | *(refp) );
  		ref1 = (uint32_t)( *(refp + delta_ref + 3) << 24 | *(refp + delta_ref + 2) << 16 | *(refp + delta_ref + 1) << 8 | *(refp + delta_ref) );
  		sad_result  = __vabsdiff4( img0, ref0 );
  		sad_result += __vabsdiff4( img1, ref1 );
  		diff[i][TID] = sad_result;
	}
	__syncthreads();

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

	minpos[TID] = TID;
	__syncthreads();

	// qui posso lasciare il vecchio modo di trovare il minimo, funziona lo stesso
    if( TID < 8 )
		if( diff[TID][0] > diff[TID+8][0] )	{
			diff[TID][0] = diff[TID+8][0];
			minpos[TID] = minpos[TID+8];
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
    if( TID == 0 )
	{
		if( diff[0][0] > diff[1][0] )	{
			diff[0][0] = diff[1][0];
			minpos[0] = minpos[1];
		}
		MVs_g[MBoffset].as_mv.row += (MV_lookup_refin_split[ minpos[0] ].row * 8);
		MVs_g[MBoffset].as_mv.col += (MV_lookup_refin_split[ minpos[0] ].col * 8);

	}

}


void me_kernel_launch_split( VP8_COMMON * const common, const uint8_t * const in_frame, const uint8_t * const ref_frame,
		int const streamID, int const split_on, int_mv * const MVs, int_mv * const MVs_split ) {

#if CUDA_VERBOSE
	float elapsedTime;
	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	CHECK(cudaEventRecord(start));
#endif

	me_cuda_split <<< common->GPU.gridDim, common->GPU.blockDim, 0, common->GPU.streams.frame[streamID] >>> (in_frame, ref_frame,
			streamID, common->GPU.streamSize, common->gpu_frame.stride, common->gpu_frame.width, common->gpu_frame.num_MB_width, split_on, MVs, MVs_split );

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

	//int MV_size_16 = 16*sizeof(int_mv);
	int MV_size_16 = cm->GPU.streamSize * sizeof(int_mv);
	// for printing informations about reference frame flags and thei usage, I left a commented prinft at line 3625
	// at the beginning of encode_frame_to_data_rate(..) in onyx_if.c

	for (int t = 0; t < cm->GPU.num_mb16th; t++) {

		int s = cm->GPU.streamLaunchOrder[t];
		//int offset = 16*s;
		int offset = cm->GPU.streamSize * s;
		// bugfix per immagini il cui n di mb non e' divisibile per 16
		// prima venivano lanciati troppi processi e cudaMemcpyAsync andava a leggere oltre i limiti degli array
		if (offset + cm->GPU.streamSize > cm->gpu_frame.num_mv)
			MV_size_16 = ( offset + cm->GPU.streamSize - cm->gpu_frame.num_mv ) * sizeof( int_mv );


		if ((ref_frame_flags & GPUFLAG_LAST_FRAME) && (cm->yv12_fb[cm->lst_fb_idx].flags & GPUFLAG_LAST_FRAME)) {
			me_kernel_launch_split(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->lst_fb_idx], s, SPLITMV_ON, (cm->gpu_frame.MVs_g)[0], cm->gpu_frame.MVs_split_g );
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[0][offset],		&(cm->gpu_frame.MVs_g)[0][offset],		MV_size_16,      cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_split_h)[offset],&(cm->gpu_frame.MVs_split_g)[offset],24 * MV_size_16, cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}

		// Se ref_frame_flags indica la presenza di un gold e se il flag del fb puntato da gld_fb_idx indica che e' gold, allora...
		if ((ref_frame_flags & GPUFLAG_GOLD_FRAME) && (cm->yv12_fb[cm->gld_fb_idx].flags & GPUFLAG_GOLD_FRAME)) {
			me_kernel_launch_split(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->gld_fb_idx], s, SPLITMV_OFF, (cm->gpu_frame.MVs_g)[1], 0 );
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[1][offset],		&(cm->gpu_frame.MVs_g)[1][offset],		MV_size_16,      cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}

		// Se ref_frame_flags indica la presenza di un altref e se il flag del fb puntato da alt_fb_idx indica che e' altref, allora...
		if ((ref_frame_flags & GPUFLAG_ALTR_FRAME) && (cm->yv12_fb[cm->alt_fb_idx].flags & GPUFLAG_ALTR_FRAME)) {
			me_kernel_launch_split(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->alt_fb_idx], s, SPLITMV_OFF, (cm->gpu_frame.MVs_g)[2], 0 );
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[2][offset],		&(cm->gpu_frame.MVs_g)[2][offset],		MV_size_16,		 cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}
	}
}


void me_cuda_launch_not_interleaved_split( VP8_COMMON * const cm, int fb_idx, int ref_frame_flags ) {

	//int MV_size_16 = 16*sizeof(int_mv);
	int MV_size_16 = cm->GPU.streamSize * sizeof(int_mv);
	// for printing informations about reference frame flags and thei usage, I left a commented prinft at line 3625
	// at the beginning of encode_frame_to_data_rate(..) in onyx_if.c

	for (int t = 0; t < cm->GPU.num_mb16th; t++) {

		int s = cm->GPU.streamLaunchOrder[t];
		//int offset = 16*s;
		int offset = cm->GPU.streamSize * s;
		// bugfix per immagini il cui n di mb non e' divisibile per 16
		// prima venivano lanciati troppi processi e cudaMemcpyAsync andava a leggere oltre i limiti degli array
		if (offset + cm->GPU.streamSize > cm->gpu_frame.num_mv)
			MV_size_16 = ( offset + cm->GPU.streamSize - cm->gpu_frame.num_mv ) * sizeof( int_mv );


		if ((ref_frame_flags & GPUFLAG_LAST_FRAME) && (cm->yv12_fb[cm->lst_fb_idx].flags & GPUFLAG_LAST_FRAME))
			me_kernel_launch_split(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->lst_fb_idx], s, SPLITMV_ON, (cm->gpu_frame.MVs_g)[0], cm->gpu_frame.MVs_split_g );

		// Se ref_frame_flags indica la presenza di un gold e se il flag del fb puntato da gld_fb_idx indica che e' gold, allora...
		if ((ref_frame_flags & GPUFLAG_GOLD_FRAME) && (cm->yv12_fb[cm->gld_fb_idx].flags & GPUFLAG_GOLD_FRAME))
			me_kernel_launch_split(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->gld_fb_idx], s, SPLITMV_OFF, (cm->gpu_frame.MVs_g)[1], 0 );

		// Se ref_frame_flags indica la presenza di un altref e se il flag del fb puntato da alt_fb_idx indica che e' altref, allora...
		if ((ref_frame_flags & GPUFLAG_ALTR_FRAME) && (cm->yv12_fb[cm->alt_fb_idx].flags & GPUFLAG_ALTR_FRAME))
			me_kernel_launch_split(cm, cm->gpu_frame.raw_current_fb_g, (cm->gpu_frame.yv12_fb_g)[cm->alt_fb_idx], s, SPLITMV_OFF, (cm->gpu_frame.MVs_g)[2], 0 );

		if (ref_frame_flags & GPUFLAG_LAST_FRAME) {
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[0][offset],		&(cm->gpu_frame.MVs_g)[0][offset],		MV_size_16,cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_split_h)[offset],&(cm->gpu_frame.MVs_split_g)[offset],24 * MV_size_16, cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}

		if ((ref_frame_flags & GPUFLAG_GOLD_FRAME) && (cm->yv12_fb[cm->gld_fb_idx].flags & GPUFLAG_GOLD_FRAME)) {
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[1][offset],		&(cm->gpu_frame.MVs_g)[1][offset],		MV_size_16, cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}

		if ((ref_frame_flags & GPUFLAG_ALTR_FRAME) && (cm->yv12_fb[cm->alt_fb_idx].flags & GPUFLAG_ALTR_FRAME)) {
			CHECK(cudaMemcpyAsync( &(cm->host_frame.MVs_h)[2][offset],		&(cm->gpu_frame.MVs_g)[2][offset],		MV_size_16,	cudaMemcpyDeviceToHost, cm->GPU.streams.frame[s]));
		}

	}
}


#endif  /* HAVE_CUDA_ENABLED_DEVICE */
#ifdef __cplusplus
}
#endif
