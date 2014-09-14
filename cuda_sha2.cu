/**
 * SHA256 algo by Tanguy Pruvot <tpruvot@github> - Aug 2014
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "miner.h"

/* not used in sha256_direct_hash_64() */
#define BLOCKS 1
#define THREADS 2

#include "cuda_helper.h"

#define SWAP(u32) cuda_swab32(u32)
#define ROTR(x,n) ((x >> n) | (x << (32-n)))

#define Ch(x,y,z)  ((x & y) ^ ( (~x) & z))
#define Maj(x,y,z) ((x & y) ^ (x & z) ^ (y & z))

#define SIGMA0(x) ((ROTR(x,2))  ^ (ROTR(x,13)) ^ (ROTR(x,22)))
#define SIGMA1(x) ((ROTR(x,6))  ^ (ROTR(x,11)) ^ (ROTR(x,25)))
#define sigma0(x) ((ROTR(x,7))  ^ (ROTR(x,18)) ^ (x >> 3))
#define sigma1(x) ((ROTR(x,17)) ^ (ROTR(x,19)) ^ (x >> 10))

//#define MIN(x,y) ((x) < (y) ? (x) : (y))
//#define MAX(x,y) ((x) > (y) ? (x) : (y))

typedef struct {
	uint32_t h[8];
	uint32_t total;
	uint32_t buflen;
	uint8_t  buf[64];
} sha256_ctx;

__device__ __constant__
static uint32_t k[] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__
static void init_ctx(sha256_ctx *ctx)
{
	ctx->h[0] = 0x6a09e667;
	ctx->h[1] = 0xbb67ae85;
	ctx->h[2] = 0x3c6ef372;
	ctx->h[3] = 0xa54ff53a;
	ctx->h[4] = 0x510e527f;
	ctx->h[5] = 0x9b05688c;
	ctx->h[6] = 0x1f83d9ab;
	ctx->h[7] = 0x5be0cd19;
	ctx->total = 0;
	ctx->buflen = 0;
}

__device__
static void insert_to_buf(sha256_ctx *ctx, const uint8_t*data, uint8_t len)
{
	int i = len;
	uint8_t *d = &ctx->buf[ctx->buflen];
	while (i--) {
		*(d++) = *(data++);
	}
	ctx->buflen += len;
}

__device__
static void sha256_block(sha256_ctx *ctx)
{
	int i;
	uint32_t a = ctx->h[0];
	uint32_t b = ctx->h[1];
	uint32_t c = ctx->h[2];
	uint32_t d = ctx->h[3];
	uint32_t e = ctx->h[4];
	uint32_t f = ctx->h[5];
	uint32_t g = ctx->h[6];
	uint32_t h = ctx->h[7];
	uint32_t w[16];
	uint32_t *data = (uint32_t *) ctx->buf;

    #pragma unroll 16
	  for (i = 0; i < 16; i++)
		w[i] = SWAP(data[i]);

	uint32_t t1, t2;
	for (i = 0; i < 16; i++) {
		t1 = k[i] + w[i] + h + SIGMA1(e) + Ch(e, f, g);
		t2 = Maj(a, b, c) + SIGMA0(a);

		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	for (i = 16; i < 64; i++) {

		w[i & 15] =
		    sigma1(w[(i - 2) & 15]) + sigma0(w[(i - 15) & 15]) + w[(i -
			16) & 15] + w[(i - 7) & 15];
		t1 = k[i] + w[i & 15] + h + SIGMA1(e) + Ch(e, f, g);
		t2 = Maj(a, b, c) + SIGMA0(a);

		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	ctx->h[0] += a;
	ctx->h[1] += b;
	ctx->h[2] += c;
	ctx->h[3] += d;
	ctx->h[4] += e;
	ctx->h[5] += f;
	ctx->h[6] += g;
	ctx->h[7] += h;

}

__device__
static void ctx_update(sha256_ctx *ctx, const uint8_t *data, uint8_t len)
{
	ctx->total += len;
	uint8_t startpos = ctx->buflen;
	uint8_t partsize;
	if (startpos + len <= 64) {
		partsize = len;
	} else
		partsize = 64 - startpos;

	insert_to_buf(ctx, data, partsize);
	if (ctx->buflen == 64) {
		uint8_t offset = 64 - startpos;
		sha256_block(ctx);
		ctx->buflen = 0;
		insert_to_buf(ctx, (const uint8_t*) (data + offset),
		    len - offset);
	}
}

/**
 * Add 0x80 byte to ctx->buf and clean the rest of it
 */
__device__
static void ctx_append_1(sha256_ctx *ctx)
{
	int i = 63 - ctx->buflen;
	uint8_t *d = &ctx->buf[ctx->buflen];
	*d++ = 0x80;
	while (i--)
	{
	  *d++ = 0;
	}

}

/**
 * Add ctx->bufflen at the end of ctx->buf
 */
__device__
static void ctx_add_length(sha256_ctx *ctx)
{
	uint32_t *blocks = (uint32_t *) ctx->buf;
	blocks[15] = SWAP(ctx->total * 8);
}

__device__
static void finish_ctx(sha256_ctx *ctx)
{
	ctx_append_1(ctx);
	ctx_add_length(ctx);
	ctx->buflen = 0;
}

__device__
static void clear_ctx_buf(sha256_ctx *ctx)
{
	uint32_t *w = (uint32_t *) ctx->buf;
#pragma unroll 16
	for (int i = 0; i < 16; i++)
		w[i] = 0;
	ctx->buflen = 0;

}

/**
 * SHA256_Final
 */
__device__
static void sha256_digest(sha256_ctx *ctx, uint32_t * result, bool last_only)
{
	uint8_t i;
	if (ctx->buflen <= 55) {	//data+0x80+datasize fits in one 512bit block
		finish_ctx(ctx);
		sha256_block(ctx);
	} else {
		uint8_t moved = 1;
		if (ctx->buflen < 64) {	//data and 0x80 fits in one block
			ctx_append_1(ctx);
			moved = 0;
		}
		sha256_block(ctx);
		clear_ctx_buf(ctx);
		if (moved)
			ctx->buf[0] = 0x80;	//append 1,the rest is already clean
		ctx_add_length(ctx);
		sha256_block(ctx);
	}
	if (last_only) {
		result[6] = SWAP(ctx->h[6]);
		result[7] = SWAP(ctx->h[7]);
		return;
	}
	#pragma unroll 8
	for (i = 0; i < 8; i++)
		result[i] = SWAP(ctx->h[i]);
}

#if 1
__device__
static void sha256_gpu_hash(const uint32_t idx, const uint8_t *data, uint8_t inlen, uint32_t *out)
{
	uint32_t sha_hash[8];

	sha256_ctx ctx;
	init_ctx(&ctx);

	ctx_update(&ctx, data, inlen);
	sha256_digest(&ctx, sha_hash, false);

	__syncthreads();
	#pragma unroll 8
	for (int i = 0; i < 8; i++)
		out[i] = sha_hash[i];
}

__global__
void kernel_sha256_hash_64(uint32_t *inbuf, uint32_t *outbuf)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	sha256_gpu_hash(idx, (const uint8_t*) &inbuf[idx * 16], 64, &outbuf[idx * 8]);
}

/* not required except for tests */
extern "C" __host__ void cryptsha256_init(int gpuid)
{
	int count;
	CUDA_SAFE_CALL(cudaGetDeviceCount(&count));
	if (gpuid < count)
		cudaSetDevice(gpuid);
	else {
		printf("Invalid CUDA device id = %d\n", gpuid);
		exit(1);
	}
}

extern "C" void sha256_hash_64(uint32_t *inbuf, uint32_t *outbuf)
{
	const size_t insize = 64 * THREADS * BLOCKS;
	const size_t outsize = 32 * THREADS * BLOCKS;

	dim3 dimGrid(BLOCKS);
	dim3 dimBlock(THREADS);

	uint32_t *cuda_inbuf;
	uint32_t *cuda_outbuf;

	CUDA_SAFE_CALL(cudaMalloc(&cuda_inbuf, insize));
	CUDA_SAFE_CALL(cudaMalloc(&cuda_outbuf, outsize));
	CUDA_SAFE_CALL(cudaMemcpy(cuda_inbuf, inbuf, insize, cudaMemcpyHostToDevice));

	kernel_sha256_hash_64 <<< dimGrid, dimBlock >>> (cuda_inbuf, cuda_outbuf);
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy(outbuf, cuda_outbuf, outsize, cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaFree(cuda_inbuf));
	CUDA_SAFE_CALL(cudaFree(cuda_outbuf));
}
#endif

__global__
/* skein final direct call */
void sha256_check_direct_hash_64(const uint32_t threads, const uint32_t first, const uint32_t target7, const uint8_t *skeinbuf, uint32_t *resNonces)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		sha256_ctx ctx;
		init_ctx(&ctx);
		ctx_update(&ctx, &skeinbuf[thread*64], 64);

		uint32_t sha_hash[8];
		sha256_digest(&ctx, sha_hash, true);

		uint32_t nounce = first + thread;
		if (sha_hash[7] <= target7 && resNonces[0] > nounce) {
			resNonces[0] = nounce;
			printf("%08x %08x <= %x\n", sha_hash[6], sha_hash[7], target7);
		}
	}
}
