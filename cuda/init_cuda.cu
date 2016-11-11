/*
    Cuda accelerated motion estimation for VP8 libvpx encoder
    by Pietro Paglierani, Giuliano Grossi, Federico Pedersini and Alessandro Petrini

    for Italtel and Universita' degli Studi di Milano
    2015-2016, Milano
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "cuda/init_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif


/*
 * print CUDA capable device info
 */
void print_cuda_info( cuda_devices_t cuda_devices ) {
	printf( "Detected %d CUDA Capable device(s)\n", cuda_devices.ndevs );
	for (int i = 0; i < cuda_devices.ndevs; i++) {
		cudaSetDevice( i );                           // set current device
		printf( "Device[%d]:\n", i );
		printf( "   |          device name: \"%s\"\n", cuda_devices.names[i] );
		printf( "   |   compute capability: %d.%d\n", cuda_devices.major[i], cuda_devices.minor[i] );
		printf( "   |      constant memory: %d (KB)\n", (int)(cuda_devices.local_mem[i] / 1024) );
		printf( "   |    texture 1D memory: %d (KB)\n", (int)(cuda_devices.maxTexture1D[i] / 1024) );
		printf( "   | max num threads x MP: %d\n", (cuda_devices.threads_multiproc[i]) );

	}
}

/*
 * check and initializes CUDA capable devices
 */
void init_cuda( void ) {
	int ndevs;
	gpu_status_t gpu_status;
	cuda_devices_t cuda_devices;

	CHECK( cudaGetDeviceCount( &ndevs ) );
	cuda_devices.ndevs = ndevs;
	cuda_devices.major = new int[ndevs];
	cuda_devices.minor = new int[ndevs];
	cuda_devices.local_mem = new size_t[ndevs];
	cuda_devices.threads_multiproc = new int[ndevs];
	cuda_devices.maxTexture1D = new int[ndevs];
	cuda_devices.names = new char*[ndevs];
	if (ndevs > 0) {
		gpu_status.have_cuda_capable_devices = 1;
		for (int i = 0; i < ndevs; i++) {
			cudaSetDevice( i );                 // set current device
			cudaDeviceProp devProp;
			cudaGetDeviceProperties( &devProp, i );
			cuda_devices.major[i] = devProp.major;
			cuda_devices.minor[i] = devProp.minor;
			cuda_devices.local_mem[i] = devProp.totalConstMem;
			cuda_devices.threads_multiproc[i] = devProp.maxThreadsPerMultiProcessor;
			cuda_devices.maxTexture1D[i] = devProp.maxTexture1D;
			cuda_devices.names[i] = (char *)malloc( (strlen( devProp.name ) + 1)*sizeof( char ) );
			strcpy( cuda_devices.names[i], devProp.name );
		}
		gpu_status.current_GPU = 0;                // TODO: default device 0
	}
	int curr_dev = gpu_status.current_GPU;
	CHECK( cudaSetDevice( curr_dev ) );

	print_cuda_info( cuda_devices );

	delete[] cuda_devices.names;
	delete[] cuda_devices.maxTexture1D;
	delete[] cuda_devices.threads_multiproc;
	delete[] cuda_devices.local_mem;
	delete[] cuda_devices.minor;
	delete[] cuda_devices.major;
}


/*
 * setup device grid, memory, streams, events, ...
 */
void GPU_setup( GPU_config_t * GPU_config, int gpu_frame_width, int gpu_frame_height ) {
	int H16th = ceil( gpu_frame_height / 16.0 );
	int W16th = ceil( gpu_frame_width / 16.0 );
	GPU_config->streamSize = W16th * 2;		// Ogni riga e' presa in carico da uno stream. Per ora.
	//int num_mb16th = ceil(H16th*W16th/16.0);
	//int num_mb16th = ceil((H16th*W16th)/(float)W16th);
	int num_mb16th = ceil( H16th * W16th / (float)GPU_config->streamSize );

	// stream creation
	GPU_config->streams.frame     = (cudaStream_t *)malloc( num_mb16th * sizeof(cudaStream_t) );
	GPU_config->streamLaunchOrder = (int *)malloc( num_mb16th * sizeof(int) );
	for (int i = 0; i < num_mb16th; i++) {
		CHECK(cudaStreamCreateWithFlags( &(GPU_config->streams.frame[i]), cudaStreamNonBlocking ));
		GPU_config->streamLaunchOrder[i] = i;
	}
	GPU_config->num_mb16th = num_mb16th;

	printf("\nNUM streams = %d\n", num_mb16th);
	printf("H16: %d, W16: %d\n", H16th, W16th);

	// event creation
	CHECK(cudaEventCreate( &(GPU_config->events.start) ));
	CHECK(cudaEventCreate( &(GPU_config->events.stop) ));

	//memory_setup_CPU_GPU();

}


/*
 * setup host pinned mem and device mem
 */
void memory_setup_CPU_GPU( VP8_COMMON *cm ) {

	cm->gpu_frame.num_MB_width  = ceil(cm->gpu_frame.width  / 16.0);
	cm->gpu_frame.num_MB_height = ceil(cm->gpu_frame.height / 16.0);

	// setup host&dev memory for MVs
	int num_MV = cm->gpu_frame.num_MB_height * cm->gpu_frame.num_MB_width;
	cm->gpu_frame.num_mv = num_MV;
	for ( int i = 0; i < 3; i++ ) {
		CHECK(cudaMallocHost(	&(cm->host_frame.MVs_h)[i],	num_MV * sizeof(int_mv)));
		CHECK(cudaMalloc(		&(cm->gpu_frame.MVs_g)[i],	num_MV * sizeof(int_mv)));
	}
	// How many splitmv vectors?
	// For every MB: 16 4x4, 4 8x8, 2 8x16, 2 16x8 => 24 x num_MV
	CHECK(cudaMallocHost(	&(cm->host_frame.MVs_split_h),	24 * num_MV * sizeof(int_mv)));
	CHECK(cudaMalloc(		&(cm->gpu_frame.MVs_split_g),	24 * num_MV * sizeof(int_mv)));

	switch (cm->cuda_me_enabled) {
		case ME_TEX_KERNEL:
			// Textures!
			cm->gpu_frame.channelDesc = cudaCreateChannelDesc( 8, 0, 0, 0, cudaChannelFormatKindUnsigned );

			CHECK(cudaMallocArray( &(cm->gpu_frame.rawfb_arr), &(cm->gpu_frame.channelDesc), (cm->gpu_frame.width  + cm->gpu_frame.width  % 16), (cm->gpu_frame.height + cm->gpu_frame.height % 16), 0 ));
			for ( int i = 0; i < NUM_YV12_BUFFERS; i++ )
				CHECK(cudaMallocArray( &(cm->gpu_frame.yv12_arr_g)[i], &(cm->gpu_frame.channelDesc), cm->gpu_frame.stride, cm->gpu_frame.height_ext, 0 ));

			memset( &(cm->gpu_frame.rawResDesc), 0, sizeof(cudaResourceDesc) );
			cm->gpu_frame.rawResDesc.resType = cudaResourceTypeArray;
			cm->gpu_frame.rawResDesc.res.array.array = cm->gpu_frame.rawfb_arr;
			for ( int i = 0; i < NUM_YV12_BUFFERS; i++ ) {
				memset( &(cm->gpu_frame.resDesc[i]), 0, sizeof(cudaResourceDesc) );
				cm->gpu_frame.resDesc[i].resType = cudaResourceTypeArray;
				cm->gpu_frame.resDesc[i].res.array.array = cm->gpu_frame.yv12_arr_g[i];
			}

			memset( &(cm->gpu_frame.texDesc), 0, sizeof(cudaTextureDesc) );
			cm->gpu_frame.texDesc.addressMode[0] = cudaAddressModeBorder;
			cm->gpu_frame.texDesc.addressMode[1] = cudaAddressModeBorder;
			cm->gpu_frame.texDesc.filterMode = cudaFilterModeLinear;
			cm->gpu_frame.texDesc.readMode = cudaReadModeNormalizedFloat;
			cm->gpu_frame.texDesc.normalizedCoords = 0;

			memset( &(cm->gpu_frame.rawResViewDesc), 0, sizeof(cudaResourceViewDesc) );
			cm->gpu_frame.rawResViewDesc.format = cudaResViewFormatUnsignedChar1;
			cm->gpu_frame.rawResViewDesc.width  = (cm->gpu_frame.width  + cm->gpu_frame.width  % 16);
			cm->gpu_frame.rawResViewDesc.height = (cm->gpu_frame.height + cm->gpu_frame.height % 16);
			cm->gpu_frame.rawResViewDesc.depth  = 0;
			memset( &(cm->gpu_frame.resViewDesc), 0, sizeof(cudaResourceViewDesc) );
			cm->gpu_frame.resViewDesc.format = cudaResViewFormatUnsignedChar1;
			cm->gpu_frame.resViewDesc.width  = cm->gpu_frame.stride;
			cm->gpu_frame.resViewDesc.height = cm->gpu_frame.height_ext;
			cm->gpu_frame.resViewDesc.depth  = 0;

			CHECK( cudaCreateTextureObject( &(cm->gpu_frame.rawFbTex), &(cm->gpu_frame.rawResDesc), &(cm->gpu_frame.texDesc), &(cm->gpu_frame.rawResViewDesc) ) );
			for ( int i = 0; i < NUM_YV12_BUFFERS; i++ ) {
				CHECK( cudaCreateTextureObject( &(cm->gpu_frame.fbTex[i]), &(cm->gpu_frame.resDesc[i]), &(cm->gpu_frame.texDesc), &(cm->gpu_frame.resViewDesc) ) );
			}

			//cm->GPU.gridDim = dim3(16,1,1);
			//cm->GPU.gridDim = dim3(120, 1, 1);
			cm->GPU.gridDim = dim3( cm->GPU.streamSize, 1, 1 );
			cm->GPU.blockDim = dim3(4,8,1);
			break;
		case ME_FAST_KERNEL: {
				int frame_size = cm->gpu_frame.stride * cm->gpu_frame.height_ext;
				int frame_size_raw = ( cm->gpu_frame.width  + cm->gpu_frame.width  % 16 ) * ( cm->gpu_frame.height + cm->gpu_frame.height % 16 );

				CHECK(cudaMalloc( (void **)&(cm->gpu_frame.raw_current_fb_g), frame_size_raw * sizeof(uint8_t)));
				for ( int i = 0; i < NUM_YV12_BUFFERS; i++ ) {
					CHECK(cudaMalloc( &(cm->gpu_frame.yv12_fb_g)[i], frame_size * sizeof(uint8_t) ));
				}
			}
			//cm->GPU.gridDim = dim3(16,1,1);
			//cm->GPU.gridDim = dim3(120, 1, 1);
			cm->GPU.gridDim = dim3( cm->GPU.streamSize, 1, 1 );
			cm->GPU.blockDim = dim3(8,16,1);
			break;
		case ME_SPLITMV_KERNEL:	{
				int frame_size = cm->gpu_frame.stride * cm->gpu_frame.height_ext;
				int frame_size_raw = ( cm->gpu_frame.width  + cm->gpu_frame.width  % 16 ) * ( cm->gpu_frame.height + cm->gpu_frame.height % 16 );

				CHECK(cudaMalloc( (void **)&(cm->gpu_frame.raw_current_fb_g), frame_size_raw * sizeof(uint8_t)));
				for ( int i = 0; i < NUM_YV12_BUFFERS; i++ ) {
					CHECK(cudaMalloc( &(cm->gpu_frame.yv12_fb_g)[i], frame_size * sizeof(uint8_t) ));
				}
				//cm->GPU.gridDim = dim3(16,1,1);
				//cm->GPU.gridDim = dim3(120, 1, 1);
				cm->GPU.gridDim = dim3( cm->GPU.streamSize, 1, 1 );
				cm->GPU.blockDim = dim3(4,8,1);
				break;
			}
		default:
			printf("Geeze!/n");
	}




}

void GPU_destroy( VP8_COMMON *cm ) {

	// Destroy the streams and free associate array
	for( int i = 0; i < cm->GPU.num_mb16th; i++ ) {
		CHECK( cudaStreamDestroy( cm->GPU.streams.frame[i] ) );
	}
	free( cm->GPU.streams.frame );
	if (cm->GPU.multiThreadEnabled)
		free( cm->GPU.streamLaunchOrder );

	// Destroy cudaEvents
	CHECK( cudaEventDestroy( cm->GPU.events.stop  ) );
	CHECK( cudaEventDestroy( cm->GPU.events.start ) );

	// free MV arrays from host and GPU, and texture obj and cudaArray
	for ( int i = 0; i < 3; i++ ) {
		CHECK( cudaFreeHost( cm->host_frame.MVs_h[i] ) );
		CHECK( cudaFree( cm->gpu_frame.MVs_g[i] ) );
	}
	CHECK( cudaFreeHost( cm->host_frame.MVs_split_h ) );
	CHECK( cudaFree( cm->gpu_frame.MVs_split_g ) );

	switch (cm->cuda_me_enabled) {
		case ME_TEX_KERNEL:
			for ( int i = 0; i < 3; i++ ) {
				CHECK( cudaDestroyTextureObject(cm->gpu_frame.fbTex[i]) );
				CHECK( cudaFreeArray(cm->gpu_frame.yv12_arr_g[i]) );
			}
			CHECK( cudaDestroyTextureObject(cm->gpu_frame.rawFbTex) );
			CHECK( cudaFreeArray(cm->gpu_frame.rawfb_arr) );
			break;
		case ME_FAST_KERNEL:
		case ME_SPLITMV_KERNEL:
			for ( int i = 0; i < 3; i++ ) {
				CHECK( cudaFree( cm->gpu_frame.yv12_fb_g[i] ) );
			}
			CHECK( cudaFree( cm->gpu_frame.raw_current_fb_g ) );
			break;
		default:
			break;
	}

	printf("calling cudaDeviceReset()\n");
	cudaDeviceReset();
}

void GPUstreamReorder( VP8_COMMON * const cm ) {
	int nStreams  = cm->GPU.num_mb16th;
	int * strm    = cm->GPU.streamLaunchOrder;
	int mbW       = cm->gpu_frame.num_MB_width;
	int mbH       = cm->gpu_frame.num_MB_height;
	int streamSz = cm->GPU.streamSize;
	int nthreads = cm->GPU.nEncodingThreads + 1;

	int id = 0;
	int multip = mbW / streamSz;					// Numero di stream interessati da un CPU thread
	int streamPacket = (mbW * nthreads) / streamSz;	// Numero di stream interessati da tutti i CPU thread

	if (streamSz <= mbW) {
		int i;
		for (i = 0; i < ceil( mbH / (float)nthreads ); i++) {
			if (((i + 1) * streamPacket) > nStreams)	// gestisce ultimo pacchettino di stream [vedi sotto]
				break;
			for (int k = 0; k < streamPacket; k++) {
				if (id < nStreams)
					strm[id++] = (k * multip) % streamPacket + k / nthreads + i * streamPacket;
			}
		}
		for (int k = i * streamPacket; k < nStreams; k++)	// Stream avanzati, assegnati sequenzialmente
			strm[id] = id++;

	} else {
		for (int i = 0; i < nStreams; i++)
			strm[i] = i;
	}
	printf( "nstreams = %d  --  id = %d\n", nStreams, id );
	for (int i = 0; i < nStreams; i++)
		printf( "%d ", strm[i] );
}



#ifdef __cplusplus
}
#endif
