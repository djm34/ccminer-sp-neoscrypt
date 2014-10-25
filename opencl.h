#ifndef __GPU_H__
#define __GPU_H__

#define OUTPUT_SIZE 32
#define MAX_GPU 8

#include <stdbool.h>
#include <stdio.h>
#include <CL/cl.h>

typedef struct {
	cl_device_id device;
	cl_context context;
	cl_command_queue commandQueue;

	cl_kernel kernel;
	cl_kernel initKernel;
	cl_kernel init2Kernel;
	cl_kernel rndKernel;
	cl_kernel mixinKernel;
	cl_kernel resultKernel;

	cl_program program;
	cl_mem inputBuffer;
	cl_mem outputBuffer;
	cl_mem stateBuffer;

	cl_mem extraBuffer;
	uint32_t extraBufferSz;

	uint32_t *output;
	uint32_t threadNumber;
	const char * kernel_name;

} CLGPU;

CLGPU* cl_gpu_init(uint32_t id, const char* kernel);
void cl_gpu_run(CLGPU* gpu, uint32_t work_size, size_t offset, cl_ulong target);
void cl_gpu_release(CLGPU* gpu);

void CopyBufferToDevice(cl_command_queue queue, cl_mem buffer, void* h_Buffer, size_t size);
void CopyBufferToHost  (cl_command_queue queue, cl_mem buffer, void* h_Buffer, size_t size);

/* algos */
int opencl_scan_blake256(int thr_id, CLGPU *gpu, uint32_t *pdata, const uint32_t *ptarget, uint32_t max_nonce, unsigned long *hashes_done);

#endif /* __GPU_H__ */
