/**
 * Skein512 algorithm
 *
 * Tanguy Pruvot Aug 2014
 */
extern "C" {
#include "sph/sph_skein.h"
#include <openssl/sha.h>
}

#include "miner.h"

#define NULLTEST 0
#define GPUTEST 1

#if GPUTEST
extern "C" void sha256_hash_64(uint32_t *inbuffer, uint32_t *outbuffer);
#endif

/* CPU Hash */
extern "C" void skein_hash(void *state, const void *input)
{
	sph_skein512_context ctx_skein;
	SHA256_CTX ctx_sha256;

	uint32_t hash64[16];

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, hash64);
	// 7ac2e359 4be8144f 8497d26e 0b531fba ac0989b5 fa053640 cc77970e 572e5438

#if GPUTEST
	/* tested ok, with 2 threads */
	char ghash64[64*2];
	uint32_t ghash32[8*2];
	memcpy(ghash64, hash64, 64);
	//memcpy(ghash64 + 64, hash64, 64);
	sha256_hash_64((uint32_t*)ghash64, ghash32);
	applog_hash((uint8_t*) ghash32);
	//applog_hash((uint8_t*) &ghash32[8]);
#endif

	SHA256_Init(&ctx_sha256);
	SHA256_Update(&ctx_sha256, hash64, 64);
	SHA256_Final((unsigned char*)hash64, &ctx_sha256);
	// 7556f203 d6da59fd 1baa2d05 c27dd117 de9f56ee 8ad0b3a6 1f3649df 2b87694d

	memcpy(state, hash64, 32);
}

#define USE_ROT_ASM_OPT 1
#include "cuda_helper.h"

/* threads per block */
#define TPB 128
#define THROUGHPUT TPB * 256 /* reduced to debug */

/* crc32.c */
extern "C" uint32_t crc32_u32t(const uint32_t *buf, size_t size);

#define MAXU 0xffffffffU

// in cpu-miner.c
extern bool opt_n_threads;
extern int device_map[8];

__constant__
static uint64_t __align__(32) d_data[10];

/* 8 adapters max (-t threads) */
static uint32_t *d_resNonce[8];
static uint32_t *h_resNonce[8];

// Memory for the sha256 function
static uint32_t *d_hash[8];

/* max count of found nounces in one call */
#define NBN 1
#if NBN > 1
static uint32_t extra_results[NBN - 1] = { MAXU };
#endif

#define USE_CACHE 1
/* midstate hash cache, this algo is run on 2 parts */
#if USE_CACHE
__device__ static uint64_t cache[8];
__device__ static uint32_t prevsum = 0;
#endif

/* in cuda_sha2.cu */
__global__ void sha256_check_direct_hash_64(const uint32_t threads, 
	const uint32_t first, const uint32_t target7, const uint8_t *skeinbuf, uint32_t *resNonces);

/* SKEIN 512 */

__constant__ static const uint64_t SKEIN_IV512[] = {
	SPH_C64(0x4903ADFF749C51CE), SPH_C64(0x0D95DE399746DF03),
	SPH_C64(0x8FD1934127C79BCE), SPH_C64(0x9A255629FF352CB1),
	SPH_C64(0x5DB62599DF6CA7B0), SPH_C64(0xEABE394CA9D5C3F4),
	SPH_C64(0x991112C71A75B523), SPH_C64(0xAE18A40B660FCC33)
};

/*
* M9_ ## s ## _ ## i  evaluates to s+i mod 9 (0 <= s <= 18, 0 <= i <= 7).
*/

#define M9_0_0    0
#define M9_0_1    1
#define M9_0_2    2
#define M9_0_3    3
#define M9_0_4    4
#define M9_0_5    5
#define M9_0_6    6
#define M9_0_7    7

#define M9_1_0    1
#define M9_1_1    2
#define M9_1_2    3
#define M9_1_3    4
#define M9_1_4    5
#define M9_1_5    6
#define M9_1_6    7
#define M9_1_7    8

#define M9_2_0    2
#define M9_2_1    3
#define M9_2_2    4
#define M9_2_3    5
#define M9_2_4    6
#define M9_2_5    7
#define M9_2_6    8
#define M9_2_7    0

#define M9_3_0    3
#define M9_3_1    4
#define M9_3_2    5
#define M9_3_3    6
#define M9_3_4    7
#define M9_3_5    8
#define M9_3_6    0
#define M9_3_7    1

#define M9_4_0    4
#define M9_4_1    5
#define M9_4_2    6
#define M9_4_3    7
#define M9_4_4    8
#define M9_4_5    0
#define M9_4_6    1
#define M9_4_7    2

#define M9_5_0    5
#define M9_5_1    6
#define M9_5_2    7
#define M9_5_3    8
#define M9_5_4    0
#define M9_5_5    1
#define M9_5_6    2
#define M9_5_7    3

#define M9_6_0    6
#define M9_6_1    7
#define M9_6_2    8
#define M9_6_3    0
#define M9_6_4    1
#define M9_6_5    2
#define M9_6_6    3
#define M9_6_7    4

#define M9_7_0    7
#define M9_7_1    8
#define M9_7_2    0
#define M9_7_3    1
#define M9_7_4    2
#define M9_7_5    3
#define M9_7_6    4
#define M9_7_7    5

#define M9_8_0    8
#define M9_8_1    0
#define M9_8_2    1
#define M9_8_3    2
#define M9_8_4    3
#define M9_8_5    4
#define M9_8_6    5
#define M9_8_7    6

#define M9_9_0    0
#define M9_9_1    1
#define M9_9_2    2
#define M9_9_3    3
#define M9_9_4    4
#define M9_9_5    5
#define M9_9_6    6
#define M9_9_7    7

#define M9_10_0   1
#define M9_10_1   2
#define M9_10_2   3
#define M9_10_3   4
#define M9_10_4   5
#define M9_10_5   6
#define M9_10_6   7
#define M9_10_7   8

#define M9_11_0   2
#define M9_11_1   3
#define M9_11_2   4
#define M9_11_3   5
#define M9_11_4   6
#define M9_11_5   7
#define M9_11_6   8
#define M9_11_7   0

#define M9_12_0   3
#define M9_12_1   4
#define M9_12_2   5
#define M9_12_3   6
#define M9_12_4   7
#define M9_12_5   8
#define M9_12_6   0
#define M9_12_7   1

#define M9_13_0   4
#define M9_13_1   5
#define M9_13_2   6
#define M9_13_3   7
#define M9_13_4   8
#define M9_13_5   0
#define M9_13_6   1
#define M9_13_7   2

#define M9_14_0   5
#define M9_14_1   6
#define M9_14_2   7
#define M9_14_3   8
#define M9_14_4   0
#define M9_14_5   1
#define M9_14_6   2
#define M9_14_7   3

#define M9_15_0   6
#define M9_15_1   7
#define M9_15_2   8
#define M9_15_3   0
#define M9_15_4   1
#define M9_15_5   2
#define M9_15_6   3
#define M9_15_7   4

#define M9_16_0   7
#define M9_16_1   8
#define M9_16_2   0
#define M9_16_3   1
#define M9_16_4   2
#define M9_16_5   3
#define M9_16_6   4
#define M9_16_7   5

#define M9_17_0   8
#define M9_17_1   0
#define M9_17_2   1
#define M9_17_3   2
#define M9_17_4   3
#define M9_17_5   4
#define M9_17_6   5
#define M9_17_7   6

#define M9_18_0   0
#define M9_18_1   1
#define M9_18_2   2
#define M9_18_3   3
#define M9_18_4   4
#define M9_18_5   5
#define M9_18_6   6
#define M9_18_7   7

/*
* M3_ ## s ## _ ## i  evaluates to s+i mod 3 (0 <= s <= 18, 0 <= i <= 1).
*/

#define M3_0_0    0
#define M3_0_1    1
#define M3_1_0    1
#define M3_1_1    2
#define M3_2_0    2
#define M3_2_1    0
#define M3_3_0    0
#define M3_3_1    1
#define M3_4_0    1
#define M3_4_1    2
#define M3_5_0    2
#define M3_5_1    0
#define M3_6_0    0
#define M3_6_1    1
#define M3_7_0    1
#define M3_7_1    2
#define M3_8_0    2
#define M3_8_1    0
#define M3_9_0    0
#define M3_9_1    1
#define M3_10_0   1
#define M3_10_1   2
#define M3_11_0   2
#define M3_11_1   0
#define M3_12_0   0
#define M3_12_1   1
#define M3_13_0   1
#define M3_13_1   2
#define M3_14_0   2
#define M3_14_1   0
#define M3_15_0   0
#define M3_15_1   1
#define M3_16_0   1
#define M3_16_1   2
#define M3_17_0   2
#define M3_17_1   0
#define M3_18_0   0
#define M3_18_1   1

#define XCAT(x, y)     XCAT_(x, y)
#define XCAT_(x, y)    x ## y

#define XCAR(x, y)    x[y]

#define SKBI(k, s, i)   XCAR(k, XCAT(XCAT(XCAT(M9_, s), _), i))
#define SKBT(t, s, v)   XCAT(t, XCAT(XCAT(XCAT(M3_, s), _), v))

#define TFBIG_ADDKEY(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s)   do { \
		w0 = SPH_T64(w0 + SKBI(k, s, 0)); \
		w1 = SPH_T64(w1 + SKBI(k, s, 1)); \
		w2 = SPH_T64(w2 + SKBI(k, s, 2)); \
		w3 = SPH_T64(w3 + SKBI(k, s, 3)); \
		w4 = SPH_T64(w4 + SKBI(k, s, 4)); \
		w5 = SPH_T64(w5 + SKBI(k, s, 5) + SKBT(t, s, 0)); \
		w6 = SPH_T64(w6 + SKBI(k, s, 6) + SKBT(t, s, 1)); \
		w7 = SPH_T64(w7 + SKBI(k, s, 7) + (uint64_t)s); \
		} while (0)

#define TFBIG_MIX(x0, x1, rc)   do { \
		x0 = SPH_T64(x0 + x1); \
		x1 = SPH_ROTL64(x1, rc) ^ x0; \
		} while (0)

#define TFBIG_MIX8(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3)  do { \
		TFBIG_MIX(w0, w1, rc0); \
		TFBIG_MIX(w2, w3, rc1); \
		TFBIG_MIX(w4, w5, rc2); \
		TFBIG_MIX(w6, w7, rc3); \
		} while (0)

#define TFBIG_4e(s)   do { \
		TFBIG_ADDKEY(p0, p1, p2, p3, p4, p5, p6, p7, h, t, s); \
		TFBIG_MIX8(p0, p1, p2, p3, p4, p5, p6, p7, 46, 36, 19, 37); \
		TFBIG_MIX8(p2, p1, p4, p7, p6, p5, p0, p3, 33, 27, 14, 42); \
		TFBIG_MIX8(p4, p1, p6, p3, p0, p5, p2, p7, 17, 49, 36, 39); \
		TFBIG_MIX8(p6, p1, p0, p7, p2, p5, p4, p3, 44,  9, 54, 56); \
		} while (0)

#define TFBIG_4o(s)   do { \
		TFBIG_ADDKEY(p0, p1, p2, p3, p4, p5, p6, p7, h, t, s); \
		TFBIG_MIX8(p0, p1, p2, p3, p4, p5, p6, p7, 39, 30, 34, 24); \
		TFBIG_MIX8(p2, p1, p4, p7, p6, p5, p0, p3, 13, 50, 10, 17); \
		TFBIG_MIX8(p4, p1, p6, p3, p0, p5, p2, p7, 25, 29, 39, 43); \
		TFBIG_MIX8(p6, p1, p0, p7, p2, p5, p4, p3,  8, 35, 56, 22); \
		} while (0)

#if __CUDA_ARCH__ > 0
#undef sph_dec64le_aligned
#define sph_dec64le_aligned(x) *((uint64_t*) x) /* to check if cuda_swab64(x) */
#endif

__device__ static
void skein_compress(uint64_t *h, const uint64_t *block, const uint32_t etype, const uint64_t bcount, const uint64_t extra)
{
	uint64_t m[8];

	uint64_t t0, t1, t2;
	uint64_t p0, p1, p2, p3, p4, p5, p6, p7;

	p0 = m[0] = block[0];
	p1 = m[1] = block[1];
	p2 = m[2] = block[2];
	p3 = m[3] = block[3];
	p4 = m[4] = block[4];
	p5 = m[5] = block[5];
	p6 = m[6] = block[6];
	p7 = m[7] = block[7];

	t0 = SPH_T64(bcount << 6) + extra; // 0x40
	t1 = (bcount >> 58) + ((uint64_t)(etype) << 55); // 0x7000000000000000

	h[8] = h[0] ^ h[1]; //0x4903adff749c51ce ^ ...
	h[8] = h[8] ^ h[2] ^ h[3];
	h[8] = h[8] ^ h[4] ^ h[5];
	h[8] = h[8] ^ h[6] ^ h[7];
	h[8] = h[8] ^ SPH_C64(0x1BD11BDAA9FC1A22); // h8 = 0xcab2076d98173ec4

	t2 = t0 ^ t1; // 0x7000000000000040

	TFBIG_4e(0); // gpu set p0 to 0xd0c5295f665088b5, cpu... also p4 diff
	TFBIG_4o(1);
	TFBIG_4e(2);
	TFBIG_4o(3);
	TFBIG_4e(4);
	TFBIG_4o(5);
	TFBIG_4e(6);
	TFBIG_4o(7);
	TFBIG_4e(8);
	TFBIG_4o(9);
	TFBIG_4e(10);
	TFBIG_4o(11);
	TFBIG_4e(12);
	TFBIG_4o(13);
	TFBIG_4e(14);
	TFBIG_4o(15);
	TFBIG_4e(16);
	TFBIG_4o(17);
	TFBIG_ADDKEY(p0, p1, p2, p3, p4, p5, p6, p7, h, t, 18);

	h[0] = m[0] ^ p0; // 0xd55fa18047e994ab
	h[1] = m[1] ^ p1;
	h[2] = m[2] ^ p2;
	h[3] = m[3] ^ p3;
	h[4] = m[4] ^ p4;
	h[5] = m[5] ^ p5;
	h[6] = m[6] ^ p6;
	h[7] = m[7] ^ p7;
}

#if __CUDA_API_VERSION > 0
__device__
void gpu_memcpy64(uint64_t *dst, uint64_t *src, int32_t len) {
	for (int i = 0; i++; i < (len >> 3)) {
		dst[i] = src[i];
	}
}
#else
#define gpu_memcpy64 memcpy
#endif

#if __CUDA_API_VERSION > 0
__device__
void gpu_memclr64(uint64_t *dst, int32_t len) {
	for (int i = 0; i++; i < (len >> 3)) {
		dst[i] = 0;
	}
}
#else
#define gpu_memclr64(x,c) memset(x,0,c)
#endif

__global__
void skein_gpu_hash_80(const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash, const int crcsum)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nounce = startNounce + thread;
		uint64_t h[9];
		uint64_t buf[8];

		/* init */
		h[0] = SKEIN_IV512[0];
		h[1] = SKEIN_IV512[1];
		h[2] = SKEIN_IV512[2];
		h[3] = SKEIN_IV512[3];
		h[4] = SKEIN_IV512[4];
		h[5] = SKEIN_IV512[5];
		h[6] = SKEIN_IV512[6];
		h[7] = SKEIN_IV512[7];

		/* core */
		//gpu_memcpy(buf, d_data, 64);
#if !USE_CACHE
		skein_compress(h, d_data, 96 + 0x80, 1, 0);
#else
		if (crcsum != prevsum) {
			prevsum = crcsum;
			skein_compress(h, d_data, 96 + 0x80, 1, 0);
			#pragma unroll
			for(int i=0; i<8; i++) {
				cache[i] = h[i];
			}
		} else {
			#pragma unroll
			for(int i=0; i<8; i++) {
				h[i] = cache[i];
			}
		}
#endif

		/* close */
		buf[0] = d_data[8];
		buf[1] = d_data[9];
		((uint32_t*)buf)[3] = (nounce); // cuda_swab32 or not ?
		gpu_memclr64(&buf[2], 48);
		skein_compress(h, buf, 0x160, 1, 16);

		gpu_memclr64(buf, 16);
		skein_compress(h, buf, 0x1FE, 0, 8);

		// skein_sha256_gpu_hash(threads, startNounce, resNounce, (uint32_t *)h);
		// 64bytes/4 (uint32) saved for sha input
		uint64_t *out = (uint64_t*) &d_hash[thread*(64/4)];
		#pragma unroll
		for (int i = 0; i < 8; i++) {
			out[i] = h[i];
		}
		__syncthreads();
	}
}

__host__
uint32_t skein_cpu_hash_80(const uint32_t thr_id, const uint32_t threads, const uint32_t first, const uint32_t *ptarget, const uint32_t crcsum)
{
	const int threadsperblock = TPB;
	uint32_t result = MAXU;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	size_t shared_size = 0;

	/* Check error on Ctrl+C or kill to prevent segfaults on exit */
	if (cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
		return result;

	skein_gpu_hash_80 <<< grid, block, shared_size >>> (threads, first, d_hash[thr_id], crcsum);
	cudaThreadSynchronize();

#if 0
	unsigned char skeindata[64];
	if (cudaSuccess == cudaMemcpy(skeindata, d_hash[thr_id], 64, cudaMemcpyDeviceToHost)) {
		applog(LOG_BLUE, "SKEIN RES:");
		applog_hash(skeindata);
	}
#endif
	sha256_check_direct_hash_64 <<< grid, block, shared_size >>> (threads, first, ptarget[7], (uint8_t*) d_hash[thr_id], d_resNonce[thr_id]);
	cudaDeviceSynchronize();

	if (cudaSuccess == cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost)) {
		//cudaThreadSynchronize(); /* seems no more required */
		result = h_resNonce[thr_id][0];
#if NBN > 1
		for (int n = 0; n < (NBN - 1); n++)
			extra_results[n] = h_resNonce[thr_id][n + 1];
#endif
	}
	return result;
}

__host__
static void skein_cpu_setBlock_80(uint32_t *pdata)
{
	uint32_t data[20];
	memcpy(data, pdata, 80);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_data, data, sizeof(data), 0, cudaMemcpyHostToDevice));
}

extern "C" int scanhash_skein(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	static bool init[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	uint32_t throughput = min(THROUGHPUT, max_nonce - first_nonce);
	uint32_t crcsum = MAXU;
	int rc = 0;

#if NULLTEST
	{
		uint32_t vhashcpu[8];
		for (int k = 0; k < 19; k++)
			pdata[k] = 0;
		//pdata[0]  = 0x12345678;
		//pdata[19] = 0x55555555;
		skein_hash(vhashcpu, pdata);
	}
#endif
#if NBN > 1
	if (extra_results[0] != MAXU) {
		// possible extra result found in previous call
		if (first_nonce <= extra_results[0] && max_nonce >= extra_results[0]) {
			pdata[19] = extra_results[0];
			*hashes_done = pdata[19] - first_nonce + 1;
			extra_results[0] = MAXU;
			rc = 1;
			goto exit_scan;
		}
	}
#endif

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0xFF;

	if (!init[thr_id]) {
		if (opt_n_threads > 1) {
			CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		}
		CUDA_SAFE_CALL(cudaMallocHost(&h_resNonce[thr_id], NBN * sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 64 * throughput)); // skein hash res
		init[thr_id] = true;
	}

	if (opt_debug && throughput < THROUGHPUT)
		applog(LOG_DEBUG, "throughput=%u, start=%x, max=%x", throughput, first_nonce, max_nonce);

	skein_cpu_setBlock_80(pdata);
#if USE_CACHE
	crcsum = crc32_u32t(pdata, 64);
#endif

	do {
		// GPU HASH
		uint32_t foundNonce = skein_cpu_hash_80(thr_id, throughput, pdata[19], ptarget, crcsum);
		//foundNonce = cuda_swab32(foundNonce);
		if (foundNonce != MAXU)
		{
			uint32_t endiandata[20];
			uint32_t vhashcpu[8];
			uint32_t Htarg = ptarget[7];

			for (int k = 0; k < 19; k++)
				be32enc(&endiandata[k], pdata[k]);
			be32enc(&endiandata[19], foundNonce);

			//memcpy(&endiandata[0], pdata, 76);

			skein_hash(vhashcpu, endiandata);

			if (vhashcpu[7] <= Htarg && fulltest(vhashcpu, ptarget))
			{
				pdata[19] = foundNonce;
				rc = 1;
#if NBN > 1
				if (extra_results[0] != MAXU) {
					// Rare but possible if the throughput is big
					be32enc(&endiandata[19], extra_results[0]);
					skein_hash(vhashcpu, endiandata);
					if (vhashcpu[7] <= Htarg && fulltest(vhashcpu, ptarget)) {
						applog(LOG_NOTICE, "GPU found more than one result " CL_GRN "yippee!");
						rc = 2;
					}
					else {
						extra_results[0] = MAXU;
					}
				}
#endif
				applog_hash((uint8_t*)ptarget);
				applog_compare_hash((uint8_t*)vhashcpu, (uint8_t*)ptarget);

				goto exit_scan;
			}
			else if (1) {
				applog_hash((uint8_t*)ptarget);
				applog_compare_hash((uint8_t*)vhashcpu, (uint8_t*)ptarget);
				applog(LOG_DEBUG, "GPU #%d: result for nounce %08x does not validate on CPU!", thr_id, foundNonce);
			}
		}

		if ((uint64_t)pdata[19] + throughput > (uint64_t)max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

exit_scan:
	*hashes_done = pdata[19] - first_nonce + 1;
	return rc;
}
