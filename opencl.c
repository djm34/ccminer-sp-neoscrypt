#include "miner.h"

#include <stdlib.h>
#include <string.h>

#include "sph/sph_blake.h"

enum platform_type {
	AMD,
	NVIDIA,
	OTHER
};

extern uint32_t opt_work_size;
extern const char* kernel_path;

void CHECK_OPENCL_ERROR(cl_int err, uint32_t id)
{
	if (err != CL_SUCCESS)
	{
		switch (err)
		{
		case CL_DEVICE_NOT_FOUND:
			applog(LOG_ERR, "[GPU%u] error:  CL_DEVICE_NOT_FOUND", id); exit(1);
		case CL_DEVICE_NOT_AVAILABLE:
			applog(LOG_ERR, "[GPU%u] error:  CL_DEVICE_NOT_AVAILABLE", id); exit(1);
		case CL_COMPILER_NOT_AVAILABLE:
			applog(LOG_ERR, "[GPU%u] error:  CL_COMPILER_NOT_AVAILABLE", id); exit(1);
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			applog(LOG_ERR, "[GPU%u] error:  CL_MEM_OBJECT_ALLOCATION_FAILURE", id); exit(1);
		case CL_OUT_OF_RESOURCES:
			applog(LOG_ERR, "[GPU%u] error:  CL_OUT_OF_RESOURCES", id); exit(1);
		case CL_OUT_OF_HOST_MEMORY:
			applog(LOG_ERR, "[GPU%u] error:  CL_OUT_OF_HOST_MEMORY", id); exit(1);
		case CL_PROFILING_INFO_NOT_AVAILABLE:
			applog(LOG_ERR, "[GPU%u] error:  CL_PROFILING_INFO_NOT_AVAILABLE", id); exit(1);
		case CL_MEM_COPY_OVERLAP:
			applog(LOG_ERR, "[GPU%u] error:  CL_MEM_COPY_OVERLAP", id); exit(1);
		case CL_IMAGE_FORMAT_MISMATCH:
			applog(LOG_ERR, "[GPU%u] error:  CL_IMAGE_FORMAT_MISMATCH", id); exit(1);
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:
			applog(LOG_ERR, "[GPU%u] error:  CL_IMAGE_FORMAT_NOT_SUPPORTED", id); exit(1);
		case CL_BUILD_PROGRAM_FAILURE:
			applog(LOG_ERR, "[GPU%u] error:  CL_BUILD_PROGRAM_FAILURE", id); exit(1);
		case CL_MAP_FAILURE:
			applog(LOG_ERR, "[GPU%u] error:  CL_MAP_FAILURE", id); exit(1);

		case CL_INVALID_VALUE:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_VALUE", id); exit(1);
		case CL_INVALID_DEVICE_TYPE:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_DEVICE_TYPE", id); exit(1);
		case CL_INVALID_PLATFORM:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_PLATFORM", id); exit(1);
		case CL_INVALID_DEVICE:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_DEVICE", id); exit(1);
		case CL_INVALID_CONTEXT:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_CONTEXT", id); exit(1);
		case CL_INVALID_QUEUE_PROPERTIES:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_QUEUE_PROPERTIES", id); exit(1);
		case CL_INVALID_COMMAND_QUEUE:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_COMMAND_QUEUE", id); exit(1);
		case CL_INVALID_HOST_PTR:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_HOST_PTR", id); exit(1);
		case CL_INVALID_MEM_OBJECT:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_MEM_OBJECT", id); exit(1);
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", id); exit(1);
		case CL_INVALID_IMAGE_SIZE:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_IMAGE_SIZE", id); exit(1);
		case CL_INVALID_SAMPLER:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_SAMPLER", id); exit(1);
		case CL_INVALID_BINARY:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_BINARY", id); exit(1);
		case CL_INVALID_BUILD_OPTIONS:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_BUILD_OPTIONS", id); exit(1);
		case CL_INVALID_PROGRAM:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_PROGRAM", id); exit(1);
		case CL_INVALID_PROGRAM_EXECUTABLE:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_PROGRAM_EXECUTABLE", id); exit(1);
		case CL_INVALID_KERNEL_NAME:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_KERNEL_NAME", id); exit(1);
		case CL_INVALID_KERNEL_DEFINITION:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_KERNEL_DEFINITION", id); exit(1);
		case CL_INVALID_KERNEL:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_KERNEL", id); exit(1);
		case CL_INVALID_ARG_INDEX:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_ARG_INDEX", id); exit(1);
		case CL_INVALID_ARG_VALUE:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_ARG_VALUE", id); exit(1);
		case CL_INVALID_ARG_SIZE:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_ARG_SIZE", id); exit(1);
		case CL_INVALID_KERNEL_ARGS:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_KERNEL_ARGS", id); exit(1);
		case CL_INVALID_WORK_DIMENSION:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_WORK_DIMENSION", id); exit(1);
		case CL_INVALID_WORK_GROUP_SIZE:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_WORK_GROUP_SIZE", id); exit(1);
		case CL_INVALID_WORK_ITEM_SIZE:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_WORK_ITEM_SIZE", id); exit(1);
		case CL_INVALID_GLOBAL_OFFSET:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_GLOBAL_OFFSET", id); exit(1);
		case CL_INVALID_EVENT_WAIT_LIST:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_EVENT_WAIT_LIST", id); exit(1);
		case CL_INVALID_EVENT:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_EVENT", id); exit(1);
		case CL_INVALID_OPERATION:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_OPERATION", id); exit(1);
		case CL_INVALID_GL_OBJECT:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_GL_OBJECT", id); exit(1);
		case CL_INVALID_BUFFER_SIZE:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_BUFFER_SIZE", id); exit(1);
		case CL_INVALID_MIP_LEVEL:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_MIP_LEVEL", id); exit(1);
		case CL_INVALID_GLOBAL_WORK_SIZE:
			applog(LOG_ERR, "[GPU%u] error:  CL_INVALID_GLOBAL_WORK_SIZE", id); exit(1);
		}
	}
}

/* Convert the kernel file into a string */
char * kernelToString(const char *filename)
{
	FILE *fp;
	long lSize;
	char *buffer;
	char filen[512];

	fp = fopen(filename, "rb");
	if (!fp) {
		sprintf(filen, "%s/%s", kernel_path, filename);
		fp = fopen(filen, "rb");
	}
	if (!fp)
		perror(filen),exit(1);

	fseek( fp , 0L , SEEK_END);
	lSize = ftell( fp );
	rewind( fp );

	/* allocate memory for entire content */
	buffer = calloc( 1, lSize+1 );
	if( !buffer ) fclose(fp),fputs("memory alloc fails",stderr),exit(1);

	/* copy the file into the buffer */
	if( 1!=fread( buffer , lSize, 1 , fp) )
	  fclose(fp),free(buffer),fputs("entire read fails",stderr),exit(1);

	fclose(fp);

	return buffer;
}

enum platform_type PrintPlatformInfo(unsigned int id, cl_platform_id platformId)
{
	enum platform_type result;
	// Get Required Size
	size_t length;
	clGetPlatformInfo(platformId, CL_PLATFORM_NAME, 0, NULL, &length);
	char* sInfo = malloc(length);
	clGetPlatformInfo(platformId, CL_PLATFORM_NAME, length, sInfo, NULL);
	applog(LOG_INFO, "[GPU%u] Platform: %s", id, sInfo);
	if (strstr(sInfo, "NVIDIA") != NULL)
		result = NVIDIA;
	else if (strstr(sInfo, "AMD") != NULL)
		result = AMD;
	else
		result = OTHER;
	free(sInfo);

	return result;
}

void PrintDeviceInfo(unsigned int id, cl_device_id deviceId, cl_ulong *maxMem, cl_ulong *maxBuffer)
{
	// Get Required Size
	size_t length;
	clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 0, NULL, &length);
	// Get actual device name
	char* sInfo = malloc(length);
	clGetDeviceInfo(deviceId, CL_DEVICE_NAME, length, sInfo, NULL);
	applog(LOG_INFO, "[GPU%u] Device: %s", id, sInfo);

	clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), maxMem, NULL);
	applog(LOG_INFO, "[GPU%u] Total device memory: %d MB", id, *maxMem >> 20);

	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), maxBuffer, NULL);
	applog(LOG_INFO, "[GPU%u] Maximum buffer size: %d MB", id, *maxBuffer >> 20);

	free(sInfo);
}

cl_kernel GetKernel(cl_program program, const char* sKernelName)
{
	cl_int err;
	cl_kernel kernel = clCreateKernel(program, sKernelName, &err);
	CHECK_OPENCL_ERROR(err, 0);

	return kernel;
}

cl_mem DeviceMalloc(cl_context m_context, size_t size)
{
	cl_int err;
	cl_mem mem = clCreateBuffer(m_context, CL_MEM_READ_WRITE, size, NULL, &err);
	CHECK_OPENCL_ERROR(err, 0);

	return mem;
}

void CopyBufferToDevice(cl_command_queue queue, cl_mem buffer, void* h_Buffer, size_t size)
{
	/* to check: clEnqueueMapBuffer */
	cl_int err = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, size, h_Buffer, 0, NULL, NULL);
	CHECK_OPENCL_ERROR(err, 0);
}

void CopyBufferToHost(cl_command_queue queue, cl_mem buffer, void* h_Buffer, size_t size)
{
	cl_int err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, size, h_Buffer, 0, NULL, NULL);
	CHECK_OPENCL_ERROR(err, 0);
}

static device_init[8] = { 0 };

CLGPU* cl_gpu_init(uint32_t id, const char* kernel) {
	applog(LOG_DEBUG, "[GPU%u] Init", id);
	CLGPU* gpu = (CLGPU*)calloc(1, sizeof(CLGPU));
	gpu->threadNumber = id;
	gpu->kernel_name = kernel;

	/* Step1: Getting platforms and choose an available one. */
	cl_uint numPlatforms;	//the NO. of platforms
	cl_platform_id platform = NULL;	//the chosen platform
	cl_int	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS)
	{
		applog(LOG_ERR, "Error %d getting platform list!", status);
		return NULL;
	}

	/* Step 2: Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device. */
	cl_uint				numDevices = 0;
	cl_device_id        *devices;
	cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms* sizeof(cl_platform_id));
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	unsigned int gpuCount = 0;
	for (int i = 0; i < numPlatforms; i++) {
		platform = platforms[i];
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
		if (numDevices > 0) {
			if (id - gpuCount < numDevices) {
				devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
				status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
				gpu->device = devices[id - gpuCount];
				break;
			}
			else
				gpuCount += numDevices;
		}
	}
	free(platforms);

	if (gpu->device == NULL) {
		applog(LOG_INFO, "[GPU%u] Device not found", id);
		return NULL;
	}

	enum platform_type platformType = PrintPlatformInfo(id, platform);
	cl_ulong maxMem, maxBuffer;
	PrintDeviceInfo(id, gpu->device, &maxMem, &maxBuffer);

	gpu->context = clCreateContext(NULL, 1, &gpu->device, NULL, NULL, NULL);
	gpu->commandQueue = clCreateCommandQueue(gpu->context, gpu->device, 0, NULL);

	const char *filename = gpu->kernel_name == NULL ? "blake256.cl" : gpu->kernel_name;
	char *source = kernelToString(filename);
	size_t sourceSize[] = { strlen(source) };

	applog(LOG_DEBUG, "Creating the program: %s, size %d", filename, *sourceSize);
	gpu->program = clCreateProgramWithSource(gpu->context, 1, (const char**)&source, sourceSize, &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d creating the program: %s", status, filename);
		return NULL;
	}
	free(source);

	char *CompilerOptions = (char *)calloc(1, 256);
	memset(CompilerOptions, 0, 256);
/*
	sprintf(CompilerOptions, "-D OUTPUT_SIZE=%u ", OUTPUT_SIZE);
	if (platformType == AMD)
		strcat(CompilerOptions, "-D RADEON=1 ");
	else if (platformType == NVIDIA)
*/
	strcat(CompilerOptions, "-DNVIDIA=1 ");

	if (strlen(CompilerOptions) > 0)
		applog(LOG_DEBUG, "Compiler options: %s", CompilerOptions);

	status = clBuildProgram(gpu->program, 1, &gpu->device, CompilerOptions, NULL, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d building the program: %s", status, filename);

		if (opt_debug) {
			char *programLog;
			size_t logSize;
			cl_int error = 0;
			cl_build_status buildStatus;

			// check build error and build status first
			clGetProgramBuildInfo(gpu->program, gpu->device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &buildStatus, NULL);
			// check build log
			clGetProgramBuildInfo(gpu->program, gpu->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
			programLog = (char*) calloc (logSize+1, sizeof(char));
			clGetProgramBuildInfo(gpu->program, gpu->device, CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
			printf("Build failed; error=%d, status=%d, programLog:\n\n%s", error, buildStatus, programLog);
			free(programLog);
		}
		CHECK_OPENCL_ERROR(status, gpu->threadNumber);
	}

	if (strcmp(gpu->kernel_name, "blake256.cl") == 0)
		gpu->kernel = GetKernel(gpu->program, "search");
	else if (strcmp(gpu->kernel_name, "wild_keccak.cl") == 0)
		gpu->kernel = GetKernel(gpu->program, "search");
	else {
		applog(LOG_ERR, "[GPU%u] kernel %s not supported", id, gpu->kernel_name);
		exit(1);
	}
	gpu->inputBuffer = DeviceMalloc(gpu->context, 128); /* 80 req for blake */
	gpu->outputBuffer = DeviceMalloc(gpu->context, OUTPUT_SIZE + sizeof(uint32_t) /* result count */);

	gpu->output = (uint32_t*) malloc(OUTPUT_SIZE);

	applog(LOG_INFO, "[GPU%u] initialized successfully", id);

	device_init[id] = true;

	return gpu;
}

void cl_gpu_release(CLGPU* gpu)
{

}

static void opencl_run_blake256(CLGPU* gpu, uint32_t work_size, size_t offset, cl_ulong target)
{
	if (opt_debug)
		applog(LOG_INFO, "[GPU%u] run work = %u, offset = %u", gpu->threadNumber, work_size, offset);

	size_t off = offset;
	size_t num = work_size;
	int p = 0;
	cl_int err;
	cl_ulong targetArg = target;

	err = clSetKernelArg(gpu->kernel, p++, sizeof(cl_mem), &gpu->outputBuffer);
	CHECK_OPENCL_ERROR(err, gpu->threadNumber);
	err = clSetKernelArg(gpu->kernel, p++, sizeof(cl_mem), &gpu->inputBuffer);
	CHECK_OPENCL_ERROR(err, gpu->threadNumber);
	err = clSetKernelArg(gpu->kernel, p++, sizeof(targetArg), &targetArg);
	CHECK_OPENCL_ERROR(err, gpu->threadNumber);

#if 0
	/* specific params */
	if (strcmp(gpu->kernel_name, "blake256.cl") == 0) {
	}

	if (strcmp(gpu->kernel_name, "wild_keccak.cl") == 0) {
		gpu->extraBufferSz = 4;
		gpu->extraBuffer = DeviceMalloc(gpu->context, gpu->extraBufferSz);
		err = clSetKernelArg(gpu->kernel, p++, sizeof(cl_mem), &gpu->extraBuffer);
		CHECK_OPENCL_ERROR(err, gpu->threadNumber);
		err = clSetKernelArg(gpu->kernel, p++, sizeof(cl_int), &gpu->extraBufferSz);
		CHECK_OPENCL_ERROR(err, gpu->threadNumber);
	}
#endif

	err = clEnqueueNDRangeKernel(gpu->commandQueue, gpu->kernel, 1, &off, &num, NULL, 0, NULL, NULL);
	CHECK_OPENCL_ERROR(err, gpu->threadNumber);

	err = clFinish(gpu->commandQueue);
	CHECK_OPENCL_ERROR(err, gpu->threadNumber);
}

int opencl_scan_blake256(int thr_id, CLGPU *gpu, uint32_t *pdata, const uint32_t *ptarget, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *nonceptr = (uint32_t*) (((char*)pdata) + 1);
	uint32_t n = *nonceptr;
	const uint32_t first_nonce = n;
	uint32_t hash[32 / 4] __attribute__((aligned(32)));
	cl_int err;

	char buf[80];
	sph_blake256_context ctx;
	sph_blake256_init(&ctx);
	sph_blake256(&ctx, pdata, 64);
	sph_blake256_close(&ctx, buf);

	memcpy(&buf[32], pdata + 16, 12);

	//applog_hash(buf);
	//applog_hash(buf+32);
	do {
		CopyBufferToDevice(gpu->commandQueue, gpu->inputBuffer, buf, 80);

		if (first_nonce == n) {
			memset(gpu->output, 0, OUTPUT_SIZE + 4);
			CopyBufferToDevice(gpu->commandQueue, gpu->outputBuffer, gpu->output, OUTPUT_SIZE + 4);
		} else {
			memset(gpu->output, 0, 4);
			CopyBufferToDevice(gpu->commandQueue, gpu->outputBuffer, gpu->output, 4);
		}

		opencl_run_blake256(gpu, opt_work_size, n, *((uint64_t*)&ptarget[6]));

		CopyBufferToHost(gpu->commandQueue, gpu->outputBuffer, gpu->output, OUTPUT_SIZE);
		for (uint32_t i = 1; i < ((uint32_t*)gpu->output)[0]; i++) {
			applog_hash(gpu->output);
			uint64_t found_nonce = gpu->output[i];
			*nonceptr = found_nonce;
			blake256hash(pdata, hash, 14);
			if (unlikely(hash[7] < ptarget[7])) {
				*hashes_done = n - first_nonce;
				return true;
			}
			else
				applog(LOG_ERR, "[GPU%u] share doesn't validate on CPU, hash=%08x, target=%08x", gpu->threadNumber, hash[7], ptarget[7]);
		}
		n += opt_work_size;

		if ((uint64_t) first_nonce + n > (uint64_t) max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

	} while (!work_restart[thr_id].restart);

	pdata[19] = n;
	*hashes_done = n - first_nonce;

	cl_gpu_release(gpu);

	return 0;
}