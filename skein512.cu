/*
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

/* CPU Hash */
extern "C" void skein_hash(void *state, const void *input)
{
	sph_skein512_context ctx_skein;
	SHA256_CTX ctx_sha256;

	uint64_t hash[8];

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, hash);

	SHA256_Init(&ctx_sha256);
	SHA256_Update(&ctx_sha256, hash, 64);
	//applog_hash((uint8_t*)(ctx_sha256.h));
	//  c3e143a5 13cff155 0d2276a7 45978724 a1481917 efe0bc5f f0a97882 1d1b684d

	SHA256_Final((unsigned char*)hash, &ctx_sha256);

	//applog_hash((uint8_t*)hash);
	// [2014-09-12 18:33:43] 7556f203 d6da59fd 1baa2d05 c27dd117 de9f56ee 8ad0b3a6 1f3649df 2b87694d

	memcpy(state, hash, 32);
}

#define USE_ROT_ASM_OPT 1
#include "cuda_helper.h"

/* threads per block */
#define TPB 128
#define THROUGHPUT TPB * 1 /* reduced to debug */

/* crc32.c */
extern "C" uint32_t crc32_u32t(const uint32_t *buf, size_t size);

#define MAXU 0xffffffffU

// in cpu-miner.c
extern bool opt_n_threads;
extern int device_map[8];

__constant__
static uint32_t __align__(32) c_Target[8];

__constant__
static uint64_t __align__(32) c_data[10];

/* 8 adapters max (-t threads) */
static uint32_t *d_resNonce[8];
static uint32_t *h_resNonce[8];

/* max count of found nounces in one call */
#define NBN 1
#if NBN > 1
static uint32_t extra_results[NBN-1] = { MAXU };
#endif

#define USE_CACHE 0
/* midstate hash cache, this algo is run on 2 parts */
#if USE_CACHE
__device__ static uint32_t cache[8];
__device__ static uint32_t prevsum = 0;
#endif

/* SHA 256 */

__device__ __constant__ static uint32_t sha256_gpu_blockHeader[16]; // 2x512 Bit Message
__device__ __constant__ static uint32_t sha256_gpu_register[8];

__device__ __constant__ static uint32_t sha256_table[] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

__device__ __constant__ static uint32_t c_sha256[64];
static const uint32_t h_sha256[] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

#define SHR(x, n) ((x) >> (n))

#define bitselect(cmp, b, a) (a & cmp) | (b & ~cmp)

#define S0(x) (ROTL32(x, 25) ^ ROTL32(x, 14) ^  SHR(x, 3))
#define S1(x) (ROTL32(x, 15) ^ ROTL32(x, 13) ^  SHR(x, 10))
#define S2(x) (ROTL32(x, 30) ^ ROTL32(x, 19) ^ ROTL32(x, 10))
#define S3(x) (ROTL32(x, 26) ^ ROTL32(x, 21) ^ ROTL32(x, 7))

#define F0(y, x, z) bitselect(z, y, z ^ x)
#define F1(x, y, z) bitselect(z, y, x)

#define R0 (W0 = S1(W14) + W9 + S0(W1) + W0)
#define R1 (W1 = S1(W15) + W10 + S0(W2) + W1)
#define R2 (W2 = S1(W0) + W11 + S0(W3) + W2)
#define R3 (W3 = S1(W1) + W12 + S0(W4) + W3)
#define R4 (W4 = S1(W2) + W13 + S0(W5) + W4)
#define R5 (W5 = S1(W3) + W14 + S0(W6) + W5)
#define R6 (W6 = S1(W4) + W15 + S0(W7) + W6)
#define R7 (W7 = S1(W5) + W0 + S0(W8) + W7)
#define R8 (W8 = S1(W6) + W1 + S0(W9) + W8)
#define R9 (W9 = S1(W7) + W2 + S0(W10) + W9)
#define R10 (W10 = S1(W8) + W3 + S0(W11) + W10)
#define R11 (W11 = S1(W9) + W4 + S0(W12) + W11)
#define R12 (W12 = S1(W10) + W5 + S0(W13) + W12)
#define R13 (W13 = S1(W11) + W6 + S0(W14) + W13)
#define R14 (W14 = S1(W12) + W7 + S0(W15) + W14)
#define R15 (W15 = S1(W13) + W8 + S0(W0) + W15)

#define RD14 (S1(W12) + W7 + S0(W15) + W14)
#define RD15 (S1(W13) + W8 + S0(W0) + W15)

#define P(a,b,c,d,e,f,g,h,x,K) { \
	temp1 = h + S3(e) + F1(e,f,g) + (K + x); \
	d += temp1; h = temp1 + S2(a) + F0(a,b,c); \
}

#define PLAST(a,b,c,d,e,f,g,h,x,K) { \
	d += h + S3(e) + F1(e,f,g) + (x + K); \
}

/* SHA not ok for the moment... */

__device__
static uint32_t skein_check_sha256(const uint32_t *skeinHash, uint32_t *outSHA)
{
	uint32_t temp1;
	uint32_t W0 = cuda_swab32(skeinHash[0]);
	uint32_t W1 = cuda_swab32(skeinHash[1]);
	uint32_t W2 = cuda_swab32(skeinHash[2]);
	uint32_t W3 = cuda_swab32(skeinHash[3]);
	uint32_t W4 = cuda_swab32(skeinHash[4]);
	uint32_t W5 = cuda_swab32(skeinHash[5]);
	uint32_t W6 = cuda_swab32(skeinHash[6]);
	uint32_t W7 = cuda_swab32(skeinHash[7]);
	uint32_t W8 = cuda_swab32(skeinHash[8]);
	uint32_t W9 = cuda_swab32(skeinHash[9]);
	uint32_t W10 = cuda_swab32(skeinHash[0xA]);
	uint32_t W11 = cuda_swab32(skeinHash[0xB]);
	uint32_t W12 = cuda_swab32(skeinHash[0xC]);
	uint32_t W13 = cuda_swab32(skeinHash[0xD]);
	uint32_t W14 = cuda_swab32(skeinHash[0xE]);
	uint32_t W15 = cuda_swab32(skeinHash[0xF]);

	uint32_t v0 = 0x6A09E667U;
	uint32_t v1 = 0xBB67AE85U;
	uint32_t v2 = 0x3C6EF372U;
	uint32_t v3 = 0xA54FF53AU;
	uint32_t v4 = 0x510E527FU;
	uint32_t v5 = 0x9B05688CU;
	uint32_t v6 = 0x1F83D9ABU;
	uint32_t v7 = 0x5BE0CD19U;

	P(v0, v1, v2, v3, v4, v5, v6, v7, W0, 0x428A2F98);
	P(v7, v0, v1, v2, v3, v4, v5, v6, W1, 0x71374491);
	P(v6, v7, v0, v1, v2, v3, v4, v5, W2, 0xB5C0FBCF);
	P(v5, v6, v7, v0, v1, v2, v3, v4, W3, 0xE9B5DBA5);
	P(v4, v5, v6, v7, v0, v1, v2, v3, W4, 0x3956C25B);
	P(v3, v4, v5, v6, v7, v0, v1, v2, W5, 0x59F111F1);
	P(v2, v3, v4, v5, v6, v7, v0, v1, W6, 0x923F82A4);
	P(v1, v2, v3, v4, v5, v6, v7, v0, W7, 0xAB1C5ED5);
	P(v0, v1, v2, v3, v4, v5, v6, v7, W8, 0xD807AA98);
	P(v7, v0, v1, v2, v3, v4, v5, v6, W9, 0x12835B01);
	P(v6, v7, v0, v1, v2, v3, v4, v5, W10, 0x243185BE);
	P(v5, v6, v7, v0, v1, v2, v3, v4, W11, 0x550C7DC3);
	P(v4, v5, v6, v7, v0, v1, v2, v3, W12, 0x72BE5D74);
	P(v3, v4, v5, v6, v7, v0, v1, v2, W13, 0x80DEB1FE);
	P(v2, v3, v4, v5, v6, v7, v0, v1, W14, 0x9BDC06A7);
	P(v1, v2, v3, v4, v5, v6, v7, v0, W15, 0xC19BF174);

	P(v0, v1, v2, v3, v4, v5, v6, v7, R0, 0xE49B69C1);
	P(v7, v0, v1, v2, v3, v4, v5, v6, R1, 0xEFBE4786);
	P(v6, v7, v0, v1, v2, v3, v4, v5, R2, 0x0FC19DC6);
	P(v5, v6, v7, v0, v1, v2, v3, v4, R3, 0x240CA1CC);
	P(v4, v5, v6, v7, v0, v1, v2, v3, R4, 0x2DE92C6F);
	P(v3, v4, v5, v6, v7, v0, v1, v2, R5, 0x4A7484AA);
	P(v2, v3, v4, v5, v6, v7, v0, v1, R6, 0x5CB0A9DC);
	P(v1, v2, v3, v4, v5, v6, v7, v0, R7, 0x76F988DA);
	P(v0, v1, v2, v3, v4, v5, v6, v7, R8, 0x983E5152);
	P(v7, v0, v1, v2, v3, v4, v5, v6, R9, 0xA831C66D);
	P(v6, v7, v0, v1, v2, v3, v4, v5, R10, 0xB00327C8);
	P(v5, v6, v7, v0, v1, v2, v3, v4, R11, 0xBF597FC7);
	P(v4, v5, v6, v7, v0, v1, v2, v3, R12, 0xC6E00BF3);
	P(v3, v4, v5, v6, v7, v0, v1, v2, R13, 0xD5A79147);
	P(v2, v3, v4, v5, v6, v7, v0, v1, R14, 0x06CA6351);
	P(v1, v2, v3, v4, v5, v6, v7, v0, R15, 0x14292967);

	P(v0, v1, v2, v3, v4, v5, v6, v7, R0, 0x27B70A85);
	P(v7, v0, v1, v2, v3, v4, v5, v6, R1, 0x2E1B2138);
	P(v6, v7, v0, v1, v2, v3, v4, v5, R2, 0x4D2C6DFC);
	P(v5, v6, v7, v0, v1, v2, v3, v4, R3, 0x53380D13);
	P(v4, v5, v6, v7, v0, v1, v2, v3, R4, 0x650A7354);
	P(v3, v4, v5, v6, v7, v0, v1, v2, R5, 0x766A0ABB);
	P(v2, v3, v4, v5, v6, v7, v0, v1, R6, 0x81C2C92E);
	P(v1, v2, v3, v4, v5, v6, v7, v0, R7, 0x92722C85);
	P(v0, v1, v2, v3, v4, v5, v6, v7, R8, 0xA2BFE8A1);
	P(v7, v0, v1, v2, v3, v4, v5, v6, R9, 0xA81A664B);
	P(v6, v7, v0, v1, v2, v3, v4, v5, R10, 0xC24B8B70);
	P(v5, v6, v7, v0, v1, v2, v3, v4, R11, 0xC76C51A3);
	P(v4, v5, v6, v7, v0, v1, v2, v3, R12, 0xD192E819);
	P(v3, v4, v5, v6, v7, v0, v1, v2, R13, 0xD6990624);
	P(v2, v3, v4, v5, v6, v7, v0, v1, R14, 0xF40E3585);
	P(v1, v2, v3, v4, v5, v6, v7, v0, R15, 0x106AA070);

	P(v0, v1, v2, v3, v4, v5, v6, v7, R0, 0x19A4C116);
	P(v7, v0, v1, v2, v3, v4, v5, v6, R1, 0x1E376C08);
	P(v6, v7, v0, v1, v2, v3, v4, v5, R2, 0x2748774C);
	P(v5, v6, v7, v0, v1, v2, v3, v4, R3, 0x34B0BCB5);
	P(v4, v5, v6, v7, v0, v1, v2, v3, R4, 0x391C0CB3);
	P(v3, v4, v5, v6, v7, v0, v1, v2, R5, 0x4ED8AA4A);
	P(v2, v3, v4, v5, v6, v7, v0, v1, R6, 0x5B9CCA4F);
	P(v1, v2, v3, v4, v5, v6, v7, v0, R7, 0x682E6FF3);
	P(v0, v1, v2, v3, v4, v5, v6, v7, R8, 0x748F82EE);
	P(v7, v0, v1, v2, v3, v4, v5, v6, R9, 0x78A5636F);
	P(v6, v7, v0, v1, v2, v3, v4, v5, R10, 0x84C87814);
	P(v5, v6, v7, v0, v1, v2, v3, v4, R11, 0x8CC70208);
	P(v4, v5, v6, v7, v0, v1, v2, v3, R12, 0x90BEFFFA);
	P(v3, v4, v5, v6, v7, v0, v1, v2, R13, 0xA4506CEB);
	P(v2, v3, v4, v5, v6, v7, v0, v1, RD14, 0xBEF9A3F7);
	P(v1, v2, v3, v4, v5, v6, v7, v0, RD15, 0xC67178F2);

	v0 += 0x6A09E667;
	v1 += 0xBB67AE85;
	v2 += 0x3C6EF372;
	v3 += 0xA54FF53A;
	v4 += 0x510E527F;
	v5 += 0x9B05688C;
	v6 += 0x1F83D9AB;
	v7 += 0x5BE0CD19;
	uint32_t s7 = v7;

	P(v0, v1, v2, v3, v4, v5, v6, v7, 0x80000000, 0x428A2F98);
	P(v7, v0, v1, v2, v3, v4, v5, v6, 0, 0x71374491);
	P(v6, v7, v0, v1, v2, v3, v4, v5, 0, 0xB5C0FBCF);
	P(v5, v6, v7, v0, v1, v2, v3, v4, 0, 0xE9B5DBA5);
	P(v4, v5, v6, v7, v0, v1, v2, v3, 0, 0x3956C25B);
	P(v3, v4, v5, v6, v7, v0, v1, v2, 0, 0x59F111F1);
	P(v2, v3, v4, v5, v6, v7, v0, v1, 0, 0x923F82A4);
	P(v1, v2, v3, v4, v5, v6, v7, v0, 0, 0xAB1C5ED5);
	P(v0, v1, v2, v3, v4, v5, v6, v7, 0, 0xD807AA98);
	P(v7, v0, v1, v2, v3, v4, v5, v6, 0, 0x12835B01);
	P(v6, v7, v0, v1, v2, v3, v4, v5, 0, 0x243185BE);
	P(v5, v6, v7, v0, v1, v2, v3, v4, 0, 0x550C7DC3);
	P(v4, v5, v6, v7, v0, v1, v2, v3, 0, 0x72BE5D74);
	P(v3, v4, v5, v6, v7, v0, v1, v2, 0, 0x80DEB1FE);
	P(v2, v3, v4, v5, v6, v7, v0, v1, 0, 0x9BDC06A7);
	P(v1, v2, v3, v4, v5, v6, v7, v0, 512, 0xC19BF174);

	P(v0, v1, v2, v3, v4, v5, v6, v7, 0x80000000U, 0xE49B69C1U);
	P(v7, v0, v1, v2, v3, v4, v5, v6, 0x01400000U, 0xEFBE4786U);
	P(v6, v7, v0, v1, v2, v3, v4, v5, 0x00205000U, 0x0FC19DC6U);
	P(v5, v6, v7, v0, v1, v2, v3, v4, 0x00005088U, 0x240CA1CCU);
	P(v4, v5, v6, v7, v0, v1, v2, v3, 0x22000800U, 0x2DE92C6FU);
	P(v3, v4, v5, v6, v7, v0, v1, v2, 0x22550014U, 0x4A7484AAU);
	P(v2, v3, v4, v5, v6, v7, v0, v1, 0x05089742U, 0x5CB0A9DCU);
	P(v1, v2, v3, v4, v5, v6, v7, v0, 0xa0000020U, 0x76F988DAU);
	P(v0, v1, v2, v3, v4, v5, v6, v7, 0x5a880000U, 0x983E5152U);
	P(v7, v0, v1, v2, v3, v4, v5, v6, 0x005c9400U, 0xA831C66DU);
	P(v6, v7, v0, v1, v2, v3, v4, v5, 0x0016d49dU, 0xB00327C8U);
	P(v5, v6, v7, v0, v1, v2, v3, v4, 0xfa801f00U, 0xBF597FC7U);
	P(v4, v5, v6, v7, v0, v1, v2, v3, 0xd33225d0U, 0xC6E00BF3U);
	P(v3, v4, v5, v6, v7, v0, v1, v2, 0x11675959U, 0xD5A79147U);
	P(v2, v3, v4, v5, v6, v7, v0, v1, 0xf6e6bfdaU, 0x06CA6351U);
	P(v1, v2, v3, v4, v5, v6, v7, v0, 0xb30c1549U, 0x14292967U);
	P(v0, v1, v2, v3, v4, v5, v6, v7, 0x08b2b050U, 0x27B70A85U);
	P(v7, v0, v1, v2, v3, v4, v5, v6, 0x9d7c4c27U, 0x2E1B2138U);
	P(v6, v7, v0, v1, v2, v3, v4, v5, 0x0ce2a393U, 0x4D2C6DFCU);
	P(v5, v6, v7, v0, v1, v2, v3, v4, 0x88e6e1eaU, 0x53380D13U);
	P(v4, v5, v6, v7, v0, v1, v2, v3, 0xa52b4335U, 0x650A7354U);
	P(v3, v4, v5, v6, v7, v0, v1, v2, 0x67a16f49U, 0x766A0ABBU);
	P(v2, v3, v4, v5, v6, v7, v0, v1, 0xd732016fU, 0x81C2C92EU);
	P(v1, v2, v3, v4, v5, v6, v7, v0, 0x4eeb2e91U, 0x92722C85U);
	P(v0, v1, v2, v3, v4, v5, v6, v7, 0x5dbf55e5U, 0xA2BFE8A1U);
	P(v7, v0, v1, v2, v3, v4, v5, v6, 0x8eee2335U, 0xA81A664BU);
	P(v6, v7, v0, v1, v2, v3, v4, v5, 0xe2bc5ec2U, 0xC24B8B70U);
	P(v5, v6, v7, v0, v1, v2, v3, v4, 0xa83f4394U, 0xC76C51A3U);
	P(v4, v5, v6, v7, v0, v1, v2, v3, 0x45ad78f7U, 0xD192E819U);
	P(v3, v4, v5, v6, v7, v0, v1, v2, 0x36f3d0cdU, 0xD6990624U);
	P(v2, v3, v4, v5, v6, v7, v0, v1, 0xd99c05e8U, 0xF40E3585U);
	P(v1, v2, v3, v4, v5, v6, v7, v0, 0xb0511dc7U, 0x106AA070U);
	P(v0, v1, v2, v3, v4, v5, v6, v7, 0x69bc7ac4U, 0x19A4C116U);
	P(v7, v0, v1, v2, v3, v4, v5, v6, 0xbd11375bU, 0x1E376C08U);
	P(v6, v7, v0, v1, v2, v3, v4, v5, 0xe3ba71e5U, 0x2748774CU);
	P(v5, v6, v7, v0, v1, v2, v3, v4, 0x3b209ff2U, 0x34B0BCB5U);
	P(v4, v5, v6, v7, v0, v1, v2, v3, 0x18feee17U, 0x391C0CB3U);
	P(v3, v4, v5, v6, v7, v0, v1, v2, 0xe25ad9e7U, 0x4ED8AA4AU);
	P(v2, v3, v4, v5, v6, v7, v0, v1, 0x13375046U, 0x5B9CCA4FU);
	P(v1, v2, v3, v4, v5, v6, v7, v0, 0x0515089dU, 0x682E6FF3U);
	P(v0, v1, v2, v3, v4, v5, v6, v7, 0x4f0d0f04U, 0x748F82EEU);
	P(v7, v0, v1, v2, v3, v4, v5, v6, 0x2627484eU, 0x78A5636FU);
	P(v6, v7, v0, v1, v2, v3, v4, v5, 0x310128d2U, 0x84C87814U);
	P(v5, v6, v7, v0, v1, v2, v3, v4, 0xc668b434U, 0x8CC70208U);
	PLAST(v4, v5, v6, v7, v0, v1, v2, v3, 0x420841ccU, 0x90BEFFFAU);

	outSHA[0] = v0;
	outSHA[1] = v1;
	outSHA[2] = v2;
	outSHA[3] = v3;
	outSHA[4] = v4;
	outSHA[5] = v5;
	outSHA[6] = v6;
	outSHA[7] = v7;

	return v7 + s7;
}

/* draft for second method, not finished yet ... */
#undef S0
#undef S1
#define S(x, n)			(((x) >> (n)) | ((x) << (32 - (n))))
#define Ch(x, y, z)		((x & (y ^ z)) ^ z)
#define Maj(x, y, z)	((x & (y | z)) | (y & z))
#define S0(x)			(S(x, 2) ^ S(x, 13) ^ S(x, 22))
#define S1(x)			(S(x, 6) ^ S(x, 11) ^ S(x, 25))
#define s0(x)			(S(x, 7) ^ S(x, 18) ^ SHR(x, 3))
#define s1(x)			(S(x, 17) ^ S(x, 19) ^ SHR(x, 10))

__device__
static void skein_sha256_gpu_hash(const uint32_t threads, const uint32_t startNounce, uint32_t *outputHash, const uint32_t *skeinHash)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		//uint32_t nounce = startNounce + thread;
		// nonceVector[thread] = nounce;

		uint32_t W1[16];
		uint32_t W2[16];
		uint32_t regs[8];
		uint32_t hash[8];
#if NULLTEST
		if (thread > 0) return;
		//nounce = 0;
#endif
		// pre
		#pragma unroll 8
		for (int k=0; k < 8; k++)
		{
			//regs[k] = sha256_gpu_register[k];
			regs[k] = sha256_table[k];
			hash[k] = regs[k];
		}

		#pragma unroll 16
		for(int k=0;k<16;k++)
			W1[k] = skeinHash[k];

		//SWAB to check....
		#pragma unroll 8
//		for (int i = 0; i < 8; i++)
//			W1[i] = SWAB32(W1[i]);
//		W1[3] = SWAB32(nounce);

		// Progress W1
		#pragma unroll 16
		for (int j = 0; j<16; j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + c_sha256[j] + W1[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

			#pragma unroll 7
			for (int k = 6; k >= 0; k--)
				regs[k + 1] = regs[k];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}

		// Progress W2...W3
		#pragma unroll 3
		for(int k=0;k<3;k++)
		{
			#pragma unroll 2
			for(int j=0;j<2;j++)
				W2[j] = s1(W1[14+j]) + W1[9+j] + s0(W1[1+j]) + W1[j];
			#pragma unroll 5
			for(int j=2;j<7;j++)
				W2[j] = s1(W2[j-2]) + W1[9+j] + s0(W1[1+j]) + W1[j];

			#pragma unroll 8
			for(int j=7;j<15;j++)
				W2[j] = s1(W2[j-2]) + W2[j-7] + s0(W1[1+j]) + W1[j];

			W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

			// Rundenfunktion
			#pragma unroll 16
			for(int j=0;j<16;j++)
			{
				uint32_t T1, T2;
				T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + c_sha256[j + 16 * (k+1)] + W2[j];
				T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

				#pragma unroll 7
				for (int l=6; l >= 0; l--) regs[l+1] = regs[l];
				regs[0] = T1 + T2;
				regs[4] += T1;
			}

			#pragma unroll 16
			for(int j=0;j<16;j++)
				W1[j] = W2[j];
		}


/*
		for(int j=16;j<64;j++)
			W[j] = s1(W[j-2]) + W[j-7] + s0(W[j-15]) + W[j-16];

#pragma unroll 64
		for(int j=0;j<64;j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + c_sha256[j] + W[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

			#pragma unroll 7
			for (int k=6; k >= 0; k--) regs[k+1] = regs[k];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}
*/

		#pragma unroll 8
		for(int k=0;k<8;k++)
			hash[k] += regs[k];

		#pragma unroll 8
		for(int k=0;k<8;k++)
			outputHash[k] = cuda_swab32(hash[k]); // 8*thread+k

		// we need c3e143a5 13cff155 0d2276a7 45978724 a1481917 efe0bc5f f0a97882 1d1b684d
	}
}

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
void gpu_memcpy(uint64_t *dst, uint64_t *src, int32_t len) {
	for (int i=0; i++; i < (len>>3)) {
		dst[i] = src[i];
	}
}
#else
#define gpu_memcpy memcpy
#endif

#if __CUDA_API_VERSION > 0
__device__
void gpu_memclr(uint64_t *dst, int32_t len) {
	for (int i=0; i++; i < (len>>3)) {
		dst[i] = 0;
	}
}
#else
#define gpu_memclr(x,c) memset(x,0,c)
#endif

__global__
void skein_gpu_hash_80(const uint32_t threads, const uint32_t startNounce, uint32_t *resNounce, const int crcsum)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
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

#if 0
		/* core */
		size_t len = 80;
		size_t ptr = 0;
		uint64_t bcount = 0;
		uint32_t first = 0x80;
		do {
			size_t clen;

			if (ptr == 64) {
				bcount++;
				//gpu_memclr(buf, 64);
				skein_compress(h, buf, 96 + first, bcount, 0);
				first = 0;
				ptr = 0;
			}
			clen = 64 - ptr;
			if (clen > len)
				clen = len;
			gpu_memcpy(buf, c_data + ptr, clen);
			ptr += clen;
			len -= clen;
		} while (len > 0);

#else /* simplified */
		gpu_memcpy(buf, c_data, 64);
		skein_compress(h, buf, 96 + 0x80, 1, 0);
#endif
		gpu_memcpy(buf, &c_data[8], 16);
		gpu_memclr(&buf[4], 48);
		buf[3] = nounce; // ? swab or not ?
		skein_compress(h, buf, 0x160, 1, 16);
		gpu_memclr(buf, 64);
		skein_compress(h, buf, 0x1FE, 0, 8);

		// skein_sha256_gpu_hash(threads, startNounce, resNounce, (uint32_t *)h);
		uint32_t outSHA[8];
		uint32_t res_sha = skein_check_sha256((uint32_t*)h, outSHA);
		//printf("%08x %08x %08x %08x.", outSHA[0], outSHA[1], outSHA[2], outSHA[3]);
#if 0
		//for (int i=0; i < 64; i += 4) {
		int i = 0;
			printf("%08x %08x %08x %08x ",
				((uint32_t*)h)[i], ((uint32_t*)h)[i+1], ((uint32_t*)h)[i+2], ((uint32_t*)h)[i+3]);
		//}
#endif
		// to check...
		//if (cuda_swab64(h[7]) <= c_Target[7])
		if (res_sha & 0xc0ffffff)
			resNounce[0] = nounce;
	}
}

__host__
uint32_t skein_cpu_hash_80(int thr_id, const uint32_t threads, const uint32_t startNounce, const uint32_t crcsum)
{
	const int threadsperblock = TPB;
	uint32_t result = MAXU;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	size_t shared_size = 0;

	/* Check error on Ctrl+C or kill to prevent segfaults on exit */
	if (cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
		return result;

	skein_gpu_hash_80 <<<grid, block, shared_size>>> (threads, startNounce, d_resNonce[thr_id], crcsum);
	cudaDeviceSynchronize();
	if (cudaSuccess == cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost)) {
		//cudaThreadSynchronize(); /* seems no more required */
		result = h_resNonce[thr_id][0];
#if NBN > 1
		for (int n=0; n < (NBN-1); n++)
			extra_results[n] = h_resNonce[thr_id][n+1];
#endif
	}
	return result;
}


__host__
static void skein_cpu_setBlock_80(uint32_t *pdata, const uint32_t *ptarget)
{
	uint32_t data[20];
	memcpy(data, pdata, 80);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_data, data, sizeof(data), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Target, ptarget, 32, 0, cudaMemcpyHostToDevice));
	/* sha */
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_sha256, h_sha256, sizeof(uint32_t) * 64));
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
		((uint32_t*)ptarget)[7] = 0x00000f;

	if (!init[thr_id]) {
		if (opt_n_threads > 1) {
			CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		}
		CUDA_SAFE_CALL(cudaMallocHost(&h_resNonce[thr_id], NBN * sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
		init[thr_id] = true;
	}

	if (opt_debug && throughput < THROUGHPUT)
		applog(LOG_DEBUG, "throughput=%u, start=%x, max=%x", throughput, first_nonce, max_nonce);

	skein_cpu_setBlock_80(pdata, ptarget);
#if USE_CACHE
	crcsum = crc32_u32t(pdata, 64);
#endif

	do {
		// GPU HASH
		uint32_t foundNonce = skein_cpu_hash_80(thr_id, throughput, pdata[19], crcsum);

		if (foundNonce != MAXU)
		{
			uint32_t endiandata[20];
			uint32_t vhashcpu[8];
			uint32_t Htarg = ptarget[7];

			for (int k=0; k < 19; k++)
				be32enc(&endiandata[k], pdata[k]);
			be32enc(&endiandata[19], foundNonce);

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
					} else {
						extra_results[0] = MAXU;
					}
				}
#endif
				goto exit_scan;
			}
			else if (opt_debug) {
				applog(LOG_DEBUG, "GPU #%d: result for nounce %08x does not validate on CPU!", thr_id, foundNonce);
			}
			else {
				applog_hash((uint8_t*)ptarget);
				applog_compare_hash((uint8_t*)vhashcpu, (uint8_t*)ptarget);
				scanf("%d",&max_nonce);
			}
		}

		if ((uint64_t) pdata[19] + throughput > (uint64_t) max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

exit_scan:
	*hashes_done = pdata[19] - first_nonce + 1;
	return rc;
}
