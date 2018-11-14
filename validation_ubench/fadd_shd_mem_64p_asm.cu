#include <stdio.h>
#include <iostream>
#include <cuda_profiler_api.h>
//#include <cutil.h>
#include <cuda_runtime.h>

#define GPUJOULE_DIR ""

#define SHARED_MEM_ELEMENTS 1024

int num_blocks;
int num_threads_per_block;
int num_iterations;
int divergence;

float* h_A;
float* h_B;
float* h_C;
float* h_res;
float* d_A;
float* d_B;
float* d_C;
float* d_res;

__global__
//void compute(const float* A, const float* B, const float* C, float* D, int n) {
void compute(float* D, int n, int div, int stride) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float I1 = tid * 2.0;

    int thread_id = threadIdx.x % 32;

    __shared__ unsigned long long sdata[SHARED_MEM_ELEMENTS];

    __shared__ void **tmp_ptr;

    __shared__ void *arr[SHARED_MEM_ELEMENTS];

    if (threadIdx.x == 0) {
        for (int i = 0; i < SHARED_MEM_ELEMENTS; i++) {
            arr[i] = (void *)&sdata[i];
        }

        for (int i = 0; i < (SHARED_MEM_ELEMENTS - 1); i++) {
            sdata[i] = (unsigned long long) arr[i + 1];
        }

        sdata[SHARED_MEM_ELEMENTS - 1] = (unsigned long long) arr[0];
    }

    __syncthreads();

    tmp_ptr = (void **)(&(arr[(threadIdx.x + stride) % SHARED_MEM_ELEMENTS]));

        double f1, f2, f3;
        f1 = 1.1;
        f2 = 2.5;
    if (thread_id < div) {
/*        __asm volatile (
                ".reg .f32 %r14;\n\t"
                "mov.f32 %r14, 2.2;\n\t"
                );
*/

        for (int k = 0; k < n; k++) {
/*           __asm volatile (
                    "add.rn.f32 %r14, %r11, %r14;\n\t"
                   );
*/           
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + k;
        }
    }
//    __syncthreads();

    //    if ((blockDim.x * blockIdx.x + threadIdx.x) == 0)
    *D = f1 * tid;

//    __syncthreads();
}

void usage() {
    std::cout << "Usage ./binary <num_blocks> <num_threads_per_block> <iterations>" "threads active per warp" << std::endl;
}

int main(int argc, char **argv)
{
    if (argc != 6) {
        usage();
        exit(1);
    }

    int num_blocks = atoi(argv[1]);
    int num_threads_per_block = atoi(argv[2]);
    int iterations = atoi(argv[3]);
    int divergence = atoi(argv[4]);
    int stride = atoi(argv[5]);

//    h_A = new float(2.0);
//    h_B = new float(3.0);
//    h_C = new float(4.0);

//    cudaMalloc((void**)&d_A, sizeof(float));
//    cudaMalloc((void**)&d_B, sizeof(float));
//    cudaMalloc((void**)&d_C, sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));
  
//    cudaMemcpy(d_A, h_A, sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_B, h_B, sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_C, h_C, sizeof(float), cudaMemcpyHostToDevice);
     
    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::string cmd = "GPUJOULE_DIR/nvml/example/power_monitor 5 > GPUJOULE_DIR/energy_model_ubench/energy_model_data/combined_inst_validation_data/fadd_shd_mem_64p_asm_power.txt &";
    std::system(cmd.c_str());
    std::system("sleep 5");

    cudaEventRecord(start, 0);
    cudaProfilerStart();
    
//    compute<<<num_blocks, num_threads_per_block>>>(d_A, d_B, d_C, d_res, iterations);
    compute<<<num_blocks, num_threads_per_block>>>(d_res, iterations, divergence, stride);

    cudaProfilerStop();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);

    std::system("killall power_monitor");
    std::cout << "GPU Elapsed Time = " << time << std::endl;
  
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();

    cudaMemcpy(h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}
