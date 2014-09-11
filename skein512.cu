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

#include "cuda_helper.h"

extern "C" void skein_hash(void *state, const void *input)
{
	sph_skein512_context ctx_skein;
	SHA256_CTX sha256;

	uint8_t hash[64];

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, hash);

	SHA256_Init(&sha256);
	SHA256_Update(&sha256, hash, 64);
	SHA256_Final((unsigned char*) hash, &sha256);

	memcpy(state, hash, 32);
}

extern "C" int scanhash_skein(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	int rc = 0;
	uint32_t n = pdata[19] - 1;
	uint32_t endiandata[20];
	uint32_t hash64[8];

	for (int k=0; k < 20; k++) {
		be32enc(&endiandata[k], pdata[k]);
	}

	do {
		//const uint32_t Htarg = ptarget[7];
		pdata[19] = ++n;
		be32enc(&endiandata[19], n);
		skein_hash(hash64, endiandata);
		if (((hash64[7]&0xFFFFFF00)==0) &&
				fulltest(hash64, ptarget)) {
			*hashes_done = n - first_nonce + 1;
			return rc;
		}
	} while (n < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = n - first_nonce + 1;
	pdata[19] = n;

	return rc;
}
