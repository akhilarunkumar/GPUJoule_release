#include <stdio.h>
#include <iostream>
#include <cuda_profiler_api.h>
//#include <cutil.h>
#include <cuda_runtime.h>

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
void compute(float* D, int n, int div) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float I1 = tid * 2.0;

    int thread_id = threadIdx.x % 32;

    if (thread_id < div) {
        __asm volatile (
                " .reg .f32 %r12;\n\t"
                " .reg .f32 %r13;\n\t"
                " .reg .f32 %r14;\n\t"
                " .reg .f32 %r15;\n\t"
                " .reg .f32 %r16;\n\t"
                " .reg .f32 %r17;\n\t"
                " .reg .f32 %r18;\n\t"
                " .reg .f32 %r19;\n\t"
                " .reg .f32 %r20;\n\t"
                " .reg .f32 %r21;\n\t"
                " .reg .f32 %r22;\n\t"
                " .reg .f32 %r23;\n\t"
                " .reg .f32 %r24;\n\t"
                " .reg .f32 %r25;\n\t"
                " .reg .f32 %r26;\n\t"
                " .reg .f32 %r27;\n\t"
                " .reg .f32 %r28;\n\t"
                "mov.f32 %r12, 4.4;\n\t"
                "mov.f32 %r13, %r12;\n\t"
                "mov.f32 %r14, 2.2;\n\t"
                "mov.f32 %r15, 3.3;\n\t"
                "mov.f32 %r16, 1.23;\n\t"
                "mov.f32 %r17, 2.42;\n\t"
                "mov.f32 %r18, 3.34;\n\t"
                "mov.f32 %r19, 5.62;\n\t"
                "mov.f32 %r20, 2.56;\n\t"
                "mov.f32 %r21, 1.56;\n\t"
                "mov.f32 %r22, 2.56;\n\t"
                "mov.f32 %r23, 5.56;\n\t"
                "mov.f32 %r24, 8.56;\n\t"
                "mov.f32 %r25, 3.56;\n\t"
                "mov.f32 %r26, 5.56;\n\t"
                "mov.f32 %r27, 6.56;\n\t"
                "mov.f32 %r28, 5.6;\n\t"

                );
        for (int k = 0; k < n; k++) {
            __asm volatile (
                    "lg2.approx.f32 %r13, %r13;\n\t" 
                    "lg2.approx.f32 %r14, %r14;\n\t" 
                    "lg2.approx.f32 %r15, %r15;\n\t" 
                    "lg2.approx.f32 %r16, %r16;\n\t" 
                    "lg2.approx.f32 %r17, %r17;\n\t" 
                    "lg2.approx.f32 %r18, %r18;\n\t" 
                    "lg2.approx.f32 %r19, %r19;\n\t" 
                    "lg2.approx.f32 %r20, %r20;\n\t" 
                    "lg2.approx.f32 %r21, %r21;\n\t" 
                    "lg2.approx.f32 %r22, %r22;\n\t" 
                    "lg2.approx.f32 %r23, %r23;\n\t" 
                    "lg2.approx.f32 %r24, %r24;\n\t" 
                    "lg2.approx.f32 %r25, %r25;\n\t" 
                    "lg2.approx.f32 %r26, %r26;\n\t" 
                    "lg2.approx.f32 %r27, %r27;\n\t" 
                    "lg2.approx.f32 %r28, %r28;\n\t" 
                    "lg2.approx.f32 %r13, %r13;\n\t" 
                    "lg2.approx.f32 %r14, %r14;\n\t" 
                    "lg2.approx.f32 %r15, %r15;\n\t" 
                    "lg2.approx.f32 %r16, %r16;\n\t" 
                    "lg2.approx.f32 %r17, %r17;\n\t" 
                    "lg2.approx.f32 %r18, %r18;\n\t" 
                    "lg2.approx.f32 %r19, %r19;\n\t" 
                    "lg2.approx.f32 %r20, %r20;\n\t" 
                    "lg2.approx.f32 %r21, %r21;\n\t" 
                    "lg2.approx.f32 %r22, %r22;\n\t" 
                    "lg2.approx.f32 %r23, %r23;\n\t" 
                    "lg2.approx.f32 %r24, %r24;\n\t" 
                    "lg2.approx.f32 %r25, %r25;\n\t" 
                    "lg2.approx.f32 %r26, %r26;\n\t" 
                    "lg2.approx.f32 %r27, %r27;\n\t" 
                    "lg2.approx.f32 %r28, %r28;\n\t" 
                    "lg2.approx.f32 %r13, %r13;\n\t" 
                    "lg2.approx.f32 %r14, %r14;\n\t" 
                    "lg2.approx.f32 %r15, %r15;\n\t" 
                    "lg2.approx.f32 %r16, %r16;\n\t" 
                    "lg2.approx.f32 %r17, %r17;\n\t" 
                    "lg2.approx.f32 %r18, %r18;\n\t" 
                    "lg2.approx.f32 %r19, %r19;\n\t" 
                    "lg2.approx.f32 %r20, %r20;\n\t" 
                    "lg2.approx.f32 %r21, %r21;\n\t" 
                    "lg2.approx.f32 %r22, %r22;\n\t" 
                    "lg2.approx.f32 %r23, %r23;\n\t" 
                    "lg2.approx.f32 %r24, %r24;\n\t" 
                    "lg2.approx.f32 %r25, %r25;\n\t" 
                    "lg2.approx.f32 %r26, %r26;\n\t" 
                    "lg2.approx.f32 %r27, %r27;\n\t" 
                    "lg2.approx.f32 %r28, %r28;\n\t" 
                    "lg2.approx.f32 %r13, %r13;\n\t" 
                    "lg2.approx.f32 %r14, %r14;\n\t" 
                    "lg2.approx.f32 %r15, %r15;\n\t" 
                    "lg2.approx.f32 %r16, %r16;\n\t" 
                    "lg2.approx.f32 %r17, %r17;\n\t" 
                    "lg2.approx.f32 %r18, %r18;\n\t" 
                    "lg2.approx.f32 %r19, %r19;\n\t" 
                    "lg2.approx.f32 %r20, %r20;\n\t" 
                    "lg2.approx.f32 %r21, %r21;\n\t" 
                    "lg2.approx.f32 %r22, %r22;\n\t" 
                    "lg2.approx.f32 %r23, %r23;\n\t" 
                    "lg2.approx.f32 %r24, %r24;\n\t" 
                    "lg2.approx.f32 %r25, %r25;\n\t" 
                    "lg2.approx.f32 %r26, %r26;\n\t" 
                    "lg2.approx.f32 %r27, %r27;\n\t" 
                    "lg2.approx.f32 %r28, %r28;\n\t" 
                    "lg2.approx.f32 %r13, %r13;\n\t" 
                    "lg2.approx.f32 %r14, %r14;\n\t" 
                    "lg2.approx.f32 %r15, %r15;\n\t" 
                    "lg2.approx.f32 %r16, %r16;\n\t" 
                    "lg2.approx.f32 %r17, %r17;\n\t" 
                    "lg2.approx.f32 %r18, %r18;\n\t" 
                    "lg2.approx.f32 %r19, %r19;\n\t" 
                    "lg2.approx.f32 %r20, %r20;\n\t" 
                    "lg2.approx.f32 %r21, %r21;\n\t" 
                    "lg2.approx.f32 %r22, %r22;\n\t" 
                    "lg2.approx.f32 %r23, %r23;\n\t" 
                    "lg2.approx.f32 %r24, %r24;\n\t" 
                    "lg2.approx.f32 %r25, %r25;\n\t" 
                    "lg2.approx.f32 %r26, %r26;\n\t" 
                    "lg2.approx.f32 %r27, %r27;\n\t" 
                    "lg2.approx.f32 %r28, %r28;\n\t" 
                    "lg2.approx.f32 %r13, %r13;\n\t" 
                    "lg2.approx.f32 %r14, %r14;\n\t" 
                    "lg2.approx.f32 %r15, %r15;\n\t" 
                    "lg2.approx.f32 %r16, %r16;\n\t" 
                    "lg2.approx.f32 %r17, %r17;\n\t" 
                    "lg2.approx.f32 %r18, %r18;\n\t" 
                    "lg2.approx.f32 %r19, %r19;\n\t" 
                    "lg2.approx.f32 %r20, %r20;\n\t" 
                    "lg2.approx.f32 %r21, %r21;\n\t" 
                    "lg2.approx.f32 %r22, %r22;\n\t" 
                    "lg2.approx.f32 %r23, %r23;\n\t" 
                    "lg2.approx.f32 %r24, %r24;\n\t" 
                    "lg2.approx.f32 %r25, %r25;\n\t" 
                    "lg2.approx.f32 %r26, %r26;\n\t" 
                    "lg2.approx.f32 %r27, %r27;\n\t" 
                    "lg2.approx.f32 %r28, %r28;\n\t" 
                    "lg2.approx.f32 %r13, %r13;\n\t" 
                    "lg2.approx.f32 %r14, %r14;\n\t" 
                    "lg2.approx.f32 %r15, %r15;\n\t" 
                    "lg2.approx.f32 %r16, %r16;\n\t" 
                    "lg2.approx.f32 %r17, %r17;\n\t" 
                    "lg2.approx.f32 %r18, %r18;\n\t" 
                    "lg2.approx.f32 %r19, %r19;\n\t" 
                    "lg2.approx.f32 %r20, %r20;\n\t" 
                    "lg2.approx.f32 %r21, %r21;\n\t" 
                    "lg2.approx.f32 %r22, %r22;\n\t" 
                    "lg2.approx.f32 %r23, %r23;\n\t" 
                    "lg2.approx.f32 %r24, %r24;\n\t" 
                    "lg2.approx.f32 %r25, %r25;\n\t" 
                    "lg2.approx.f32 %r26, %r26;\n\t" 
                    "lg2.approx.f32 %r27, %r27;\n\t" 
                    "lg2.approx.f32 %r28, %r28;\n\t" 
                    "lg2.approx.f32 %r13, %r13;\n\t" 
                    "lg2.approx.f32 %r14, %r14;\n\t" 
                    "lg2.approx.f32 %r15, %r15;\n\t" 
                    "lg2.approx.f32 %r16, %r16;\n\t" 
                    "lg2.approx.f32 %r17, %r17;\n\t" 
                    "lg2.approx.f32 %r18, %r18;\n\t" 
                    "lg2.approx.f32 %r19, %r19;\n\t" 
                    "lg2.approx.f32 %r20, %r20;\n\t" 
                    "lg2.approx.f32 %r21, %r21;\n\t" 
                    "lg2.approx.f32 %r22, %r22;\n\t" 
                    "lg2.approx.f32 %r23, %r23;\n\t" 
                    "lg2.approx.f32 %r24, %r24;\n\t" 
                    "lg2.approx.f32 %r25, %r25;\n\t" 
                    "lg2.approx.f32 %r26, %r26;\n\t" 
                    "lg2.approx.f32 %r27, %r27;\n\t" 
                    "lg2.approx.f32 %r28, %r28;\n\t" 
                    "lg2.approx.f32 %r13, %r13;\n\t" 
                    "lg2.approx.f32 %r14, %r14;\n\t" 
                    "lg2.approx.f32 %r15, %r15;\n\t" 
                    "lg2.approx.f32 %r16, %r16;\n\t" 
                    "lg2.approx.f32 %r17, %r17;\n\t" 
                    "lg2.approx.f32 %r18, %r18;\n\t" 
                    "lg2.approx.f32 %r19, %r19;\n\t" 
                    "lg2.approx.f32 %r20, %r20;\n\t" 
                    "lg2.approx.f32 %r21, %r21;\n\t" 
                    "lg2.approx.f32 %r22, %r22;\n\t" 
                    "lg2.approx.f32 %r23, %r23;\n\t" 
                    "lg2.approx.f32 %r24, %r24;\n\t" 
                    "lg2.approx.f32 %r25, %r25;\n\t" 
                    "lg2.approx.f32 %r26, %r26;\n\t" 
                    "lg2.approx.f32 %r27, %r27;\n\t" 
                    "lg2.approx.f32 %r28, %r28;\n\t" 
                    "lg2.approx.f32 %r13, %r13;\n\t" 
                    "lg2.approx.f32 %r14, %r14;\n\t" 
                    "lg2.approx.f32 %r15, %r15;\n\t" 
                    "lg2.approx.f32 %r16, %r16;\n\t" 
                    "lg2.approx.f32 %r17, %r17;\n\t" 
                    "lg2.approx.f32 %r18, %r18;\n\t" 
                    "lg2.approx.f32 %r19, %r19;\n\t" 
                    "lg2.approx.f32 %r20, %r20;\n\t" 
                    "lg2.approx.f32 %r21, %r21;\n\t" 
                    "lg2.approx.f32 %r22, %r22;\n\t" 
                    "lg2.approx.f32 %r23, %r23;\n\t" 
                    "lg2.approx.f32 %r24, %r24;\n\t" 
                    "lg2.approx.f32 %r25, %r25;\n\t" 
                    "lg2.approx.f32 %r26, %r26;\n\t" 
                    "lg2.approx.f32 %r27, %r27;\n\t" 
                    "lg2.approx.f32 %r28, %r28;\n\t" 
                    "lg2.approx.f32 %r13, %r13;\n\t" 
                    "lg2.approx.f32 %r14, %r14;\n\t" 
                    "lg2.approx.f32 %r15, %r15;\n\t" 
                    "lg2.approx.f32 %r16, %r16;\n\t" 
                    "lg2.approx.f32 %r17, %r17;\n\t" 
                    "lg2.approx.f32 %r18, %r18;\n\t" 
                    "lg2.approx.f32 %r19, %r19;\n\t" 
                    "lg2.approx.f32 %r20, %r20;\n\t" 
                    "lg2.approx.f32 %r21, %r21;\n\t" 
                    "lg2.approx.f32 %r22, %r22;\n\t" 
                    "lg2.approx.f32 %r23, %r23;\n\t" 
                    "lg2.approx.f32 %r24, %r24;\n\t" 
                    "lg2.approx.f32 %r25, %r25;\n\t" 
                    "lg2.approx.f32 %r26, %r26;\n\t" 
                    "lg2.approx.f32 %r27, %r27;\n\t" 
                    "lg2.approx.f32 %r28, %r28;\n\t" 
                    "lg2.approx.f32 %r13, %r13;\n\t" 
                    "lg2.approx.f32 %r14, %r14;\n\t" 
                    "lg2.approx.f32 %r15, %r15;\n\t" 
                    "lg2.approx.f32 %r16, %r16;\n\t" 
                    "lg2.approx.f32 %r17, %r17;\n\t" 
                    "lg2.approx.f32 %r18, %r18;\n\t" 
                    "lg2.approx.f32 %r19, %r19;\n\t" 
                    "lg2.approx.f32 %r20, %r20;\n\t" 
                    "lg2.approx.f32 %r21, %r21;\n\t" 
                    "lg2.approx.f32 %r22, %r22;\n\t" 
                    "lg2.approx.f32 %r23, %r23;\n\t" 
                    "lg2.approx.f32 %r24, %r24;\n\t" 
                    "lg2.approx.f32 %r25, %r25;\n\t" 
                    "lg2.approx.f32 %r26, %r26;\n\t" 
                    "lg2.approx.f32 %r27, %r27;\n\t" 
                    "lg2.approx.f32 %r28, %r28;\n\t" 
                    "lg2.approx.f32 %r13, %r13;\n\t" 
                    "lg2.approx.f32 %r14, %r14;\n\t" 
                    "lg2.approx.f32 %r15, %r15;\n\t" 
                    "lg2.approx.f32 %r16, %r16;\n\t" 
                    "lg2.approx.f32 %r17, %r17;\n\t" 
                    "lg2.approx.f32 %r18, %r18;\n\t" 
                    "lg2.approx.f32 %r19, %r19;\n\t" 
                    "lg2.approx.f32 %r20, %r20;\n\t" 
                    "lg2.approx.f32 %r21, %r21;\n\t" 
                    "lg2.approx.f32 %r22, %r22;\n\t" 
                    "lg2.approx.f32 %r23, %r23;\n\t" 
                    "lg2.approx.f32 %r24, %r24;\n\t" 
                    "lg2.approx.f32 %r25, %r25;\n\t" 
                    "lg2.approx.f32 %r26, %r26;\n\t" 
                    "lg2.approx.f32 %r27, %r27;\n\t" 
                    "lg2.approx.f32 %r28, %r28;\n\t" 
                    "lg2.approx.f32 %r13, %r13;\n\t" 
                    "lg2.approx.f32 %r14, %r14;\n\t" 
                    "lg2.approx.f32 %r15, %r15;\n\t" 
                    "lg2.approx.f32 %r16, %r16;\n\t" 
                    "lg2.approx.f32 %r17, %r17;\n\t" 
                    "lg2.approx.f32 %r18, %r18;\n\t" 
                    "lg2.approx.f32 %r19, %r19;\n\t" 
                    "lg2.approx.f32 %r20, %r20;\n\t" 
                    "lg2.approx.f32 %r21, %r21;\n\t" 
                    "lg2.approx.f32 %r22, %r22;\n\t" 
                    "lg2.approx.f32 %r23, %r23;\n\t" 
                    "lg2.approx.f32 %r24, %r24;\n\t" 
                    "lg2.approx.f32 %r25, %r25;\n\t" 
                    "lg2.approx.f32 %r26, %r26;\n\t" 
                    "lg2.approx.f32 %r27, %r27;\n\t" 
                    "lg2.approx.f32 %r28, %r28;\n\t" 
                    "lg2.approx.f32 %r13, %r13;\n\t" 
                    "lg2.approx.f32 %r14, %r14;\n\t" 
                    "lg2.approx.f32 %r15, %r15;\n\t" 
                    "lg2.approx.f32 %r16, %r16;\n\t" 
                    "lg2.approx.f32 %r17, %r17;\n\t" 
                    "lg2.approx.f32 %r18, %r18;\n\t" 
                    "lg2.approx.f32 %r19, %r19;\n\t" 
                    "lg2.approx.f32 %r20, %r20;\n\t" 
                    "lg2.approx.f32 %r21, %r21;\n\t" 
                    "lg2.approx.f32 %r22, %r22;\n\t" 
                    "lg2.approx.f32 %r23, %r23;\n\t" 
                    "lg2.approx.f32 %r24, %r24;\n\t" 
                    "lg2.approx.f32 %r25, %r25;\n\t" 
                    "lg2.approx.f32 %r26, %r26;\n\t" 
                    "lg2.approx.f32 %r27, %r27;\n\t" 
                    "lg2.approx.f32 %r28, %r28;\n\t" 
                    "lg2.approx.f32 %r13, %r13;\n\t" 
                    "lg2.approx.f32 %r14, %r14;\n\t" 
                    "lg2.approx.f32 %r15, %r15;\n\t" 
                    "lg2.approx.f32 %r16, %r16;\n\t" 
                    "lg2.approx.f32 %r17, %r17;\n\t" 
                    "lg2.approx.f32 %r18, %r18;\n\t" 
                    "lg2.approx.f32 %r19, %r19;\n\t" 
                    "lg2.approx.f32 %r20, %r20;\n\t" 
                    "lg2.approx.f32 %r21, %r21;\n\t" 
                    "lg2.approx.f32 %r22, %r22;\n\t" 
                    "lg2.approx.f32 %r23, %r23;\n\t" 
                    "lg2.approx.f32 %r24, %r24;\n\t" 
                    "lg2.approx.f32 %r25, %r25;\n\t" 
                    "lg2.approx.f32 %r26, %r26;\n\t" 
                    "lg2.approx.f32 %r27, %r27;\n\t" 
                    "lg2.approx.f32 %r28, %r28;\n\t" 
                    );
        }
    }
//    __syncthreads();

    //    if ((blockDim.x * blockIdx.x + threadIdx.x) == 0)
    *D = I1;

//    __syncthreads();
}

void usage() {
    std::cout << "Usage ./binary <num_blocks> <num_threads_per_block> <iterations>" "threads active per warp" << std::endl;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        usage();
        exit(1);
    }

    int num_blocks = atoi(argv[1]);
    int num_threads_per_block = atoi(argv[2]);
    int iterations = atoi(argv[3]);
    int divergence = atoi(argv[4]);

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

    cudaEventRecord(start, 0);
    cudaProfilerStart();
    
//    compute<<<num_blocks, num_threads_per_block>>>(d_A, d_B, d_C, d_res, iterations);
    compute<<<num_blocks, num_threads_per_block>>>(d_res, iterations, divergence);

    cudaProfilerStop();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);

    std::cout << "GPU Elapsed Time = " << time << std::endl;
  
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();

    cudaMemcpy(h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}
