#include <stdio.h>
extern "C" __declspec(dllexport) void init_cuda_device();
extern "C" __declspec(dllexport) void *alloc_cuda_mem(size_t byte_count);
extern "C" __declspec(dllexport) void *cuda_free();
extern "C" __declspec(dllexport) void cuda_mem_copy_to_device(void *dst, void *src, size_t byte_count);
extern "C" __declspec(dllexport) void cuda_mem_copy_to_host(void *dst, void *src, size_t byte_count);
extern "C" __declspec(dllexport) void add(float *a, float *b, size_t len, float *out);

void init_cuda_device()
{
}

void *alloc_cuda_mem(size_t byte_count)
{
    void *result = NULL;
    cudaError_t error = cudaMalloc(&result, byte_count);
    return result;
}

void cuda_free(void *mem)
{
    cudaFree(mem);
}

void cuda_mem_copy_to_device(void *dst, void *src, size_t byte_count)
{
    cudaMemcpy(dst, src, byte_count, cudaMemcpyHostToDevice);
}

void cuda_mem_copy_to_host(void *dst, void *src, size_t byte_count)
{
    cudaMemcpy(dst, src, byte_count, cudaMemcpyDeviceToHost);
}

__global__ void add_kernel(float *a, float *b, size_t len, float *out)
{
    for (int i = 0; i < len; i++)
    {
        out[i] = a[i] + b[i];
    }
}

/* TODO: delete me. dumb test kernel */
void add(float *a, float *b, size_t len, float *out)
{
    add_kernel<<<1, 1>>>(a, b, len, out);
}
