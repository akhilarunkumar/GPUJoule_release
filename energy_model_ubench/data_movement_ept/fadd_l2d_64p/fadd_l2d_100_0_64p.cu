#include <stdio.h>
#include <iostream>
#include <cuda_profiler_api.h>
//#include <cutil.h>
#include <cuda_runtime.h>
#include <string>

#define GPUJOULE_DIR ""

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
void shared_latency(float* D, int n, int div) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    double I1 = tid * 2.0;

    int thread_id = threadIdx.x % 32;

    if (thread_id < div) {
        __asm volatile (
                " .reg .f64 %r29;\n\t"
                " .reg .f64 %r13;\n\t"
                " .reg .f64 %r14;\n\t"
                " .reg .f64 %r15;\n\t"
                " .reg .f64 %r16;\n\t"
                " .reg .f64 %r17;\n\t"
                " .reg .f64 %r18;\n\t"
                " .reg .f64 %r19;\n\t"
                " .reg .f64 %r20;\n\t"
                " .reg .f64 %r21;\n\t"
                " .reg .f64 %r22;\n\t"
                " .reg .f64 %r23;\n\t"
                " .reg .f64 %r24;\n\t"
                " .reg .f64 %r25;\n\t"
                " .reg .f64 %r26;\n\t"
                " .reg .f64 %r27;\n\t"
                " .reg .f64 %r28;\n\t"
                "mov.f64 %r29, 4.4;\n\t"
                "mov.f64 %r13, %r29;\n\t"
                "mov.f64 %r14, 2.2;\n\t"
                "mov.f64 %r15, 3.3;\n\t"
                "mov.f64 %r16, 1.23;\n\t"
                "mov.f64 %r17, 2.42;\n\t"
                "mov.f64 %r18, 3.34;\n\t"
                "mov.f64 %r19, 5.62;\n\t"
                "mov.f64 %r20, 2.56;\n\t"
                "mov.f64 %r21, 1.56;\n\t"
                "mov.f64 %r22, 2.56;\n\t"
                "mov.f64 %r23, 5.56;\n\t"
                "mov.f64 %r24, 8.56;\n\t"
                "mov.f64 %r25, 3.56;\n\t"
                "mov.f64 %r26, 5.56;\n\t"
                "mov.f64 %r27, 6.56;\n\t"
                "mov.f64 %r28, 0.56;\n\t"

                );
        for (int k = 0; k < n; k++) {
            __asm volatile (
                    "add.rn.f64 %r21, %r29, %r21;\n\t" 
                    "add.rn.f64 %r22, %r29, %r22;\n\t" 
                    "add.rn.f64 %r23, %r29, %r23;\n\t" 
                    "add.rn.f64 %r24, %r29, %r24;\n\t" 
                    "add.rn.f64 %r25, %r29, %r25;\n\t" 
                    "add.rn.f64 %r26, %r29, %r26;\n\t" 
                    "add.rn.f64 %r27, %r29, %r27;\n\t" 
                    "add.rn.f64 %r28, %r29, %r28;\n\t" 
                    "add.rn.f64 %r13, %r29, %r13;\n\t" 
                    "add.rn.f64 %r14, %r29, %r14;\n\t" 
                    "add.rn.f64 %r15, %r29, %r15;\n\t" 
                    "add.rn.f64 %r16, %r29, %r16;\n\t" 
                    "add.rn.f64 %r17, %r29, %r17;\n\t" 
                    "add.rn.f64 %r18, %r29, %r18;\n\t" 
                    "add.rn.f64 %r19, %r29, %r19;\n\t" 
                    "add.rn.f64 %r20, %r29, %r20;\n\t" 
                    "add.rn.f64 %r21, %r29, %r21;\n\t" 
                    "add.rn.f64 %r22, %r29, %r22;\n\t" 
                    "add.rn.f64 %r23, %r29, %r23;\n\t" 
                    "add.rn.f64 %r24, %r29, %r24;\n\t" 
                    "add.rn.f64 %r25, %r29, %r25;\n\t" 
                    "add.rn.f64 %r26, %r29, %r26;\n\t" 
                    "add.rn.f64 %r27, %r29, %r27;\n\t" 
                    "add.rn.f64 %r28, %r29, %r28;\n\t" 
                    "add.rn.f64 %r13, %r29, %r13;\n\t" 
                    "add.rn.f64 %r14, %r29, %r14;\n\t" 
                    "add.rn.f64 %r15, %r29, %r15;\n\t" 
                    "add.rn.f64 %r16, %r29, %r16;\n\t" 
                    "add.rn.f64 %r17, %r29, %r17;\n\t" 
                    "add.rn.f64 %r18, %r29, %r18;\n\t" 
                    "add.rn.f64 %r19, %r29, %r19;\n\t" 
                    "add.rn.f64 %r20, %r29, %r20;\n\t" 
                    "add.rn.f64 %r21, %r29, %r21;\n\t" 
                    "add.rn.f64 %r22, %r29, %r22;\n\t" 
                    "add.rn.f64 %r23, %r29, %r23;\n\t" 
                    "add.rn.f64 %r24, %r29, %r24;\n\t" 
                    "add.rn.f64 %r25, %r29, %r25;\n\t" 
                    "add.rn.f64 %r26, %r29, %r26;\n\t" 
                    "add.rn.f64 %r27, %r29, %r27;\n\t" 
                    "add.rn.f64 %r28, %r29, %r28;\n\t" 
                    "add.rn.f64 %r13, %r29, %r13;\n\t" 
                    "add.rn.f64 %r14, %r29, %r14;\n\t" 
                    "add.rn.f64 %r15, %r29, %r15;\n\t" 
                    "add.rn.f64 %r16, %r29, %r16;\n\t" 
                    "add.rn.f64 %r17, %r29, %r17;\n\t" 
                    "add.rn.f64 %r18, %r29, %r18;\n\t" 
                    "add.rn.f64 %r19, %r29, %r19;\n\t" 
                    "add.rn.f64 %r20, %r29, %r20;\n\t" 
                    "add.rn.f64 %r21, %r29, %r21;\n\t" 
                    "add.rn.f64 %r22, %r29, %r22;\n\t" 
                    "add.rn.f64 %r23, %r29, %r23;\n\t" 
                    "add.rn.f64 %r24, %r29, %r24;\n\t" 
                    "add.rn.f64 %r25, %r29, %r25;\n\t" 
                    "add.rn.f64 %r26, %r29, %r26;\n\t" 
                    "add.rn.f64 %r27, %r29, %r27;\n\t" 
                    "add.rn.f64 %r28, %r29, %r28;\n\t" 
                    "add.rn.f64 %r13, %r29, %r13;\n\t" 
                    "add.rn.f64 %r14, %r29, %r14;\n\t" 
                    "add.rn.f64 %r15, %r29, %r15;\n\t" 
                    "add.rn.f64 %r16, %r29, %r16;\n\t" 
                    "add.rn.f64 %r17, %r29, %r17;\n\t" 
                    "add.rn.f64 %r18, %r29, %r18;\n\t" 
                    "add.rn.f64 %r19, %r29, %r19;\n\t" 
                    "add.rn.f64 %r20, %r29, %r20;\n\t" 
                    "add.rn.f64 %r21, %r29, %r21;\n\t" 
                    "add.rn.f64 %r22, %r29, %r22;\n\t" 
                    "add.rn.f64 %r23, %r29, %r23;\n\t" 
                    "add.rn.f64 %r24, %r29, %r24;\n\t" 
                    "add.rn.f64 %r25, %r29, %r25;\n\t" 
                    "add.rn.f64 %r26, %r29, %r26;\n\t" 
                    "add.rn.f64 %r27, %r29, %r27;\n\t" 
                    "add.rn.f64 %r28, %r29, %r28;\n\t" 
                    "add.rn.f64 %r13, %r29, %r13;\n\t" 
                    "add.rn.f64 %r14, %r29, %r14;\n\t" 
                    "add.rn.f64 %r15, %r29, %r15;\n\t" 
                    "add.rn.f64 %r16, %r29, %r16;\n\t" 
                    "add.rn.f64 %r17, %r29, %r17;\n\t" 
                    "add.rn.f64 %r18, %r29, %r18;\n\t" 
                    "add.rn.f64 %r19, %r29, %r19;\n\t" 
                    "add.rn.f64 %r20, %r29, %r20;\n\t" 
                    "add.rn.f64 %r21, %r29, %r21;\n\t" 
                    "add.rn.f64 %r22, %r29, %r22;\n\t" 
                    "add.rn.f64 %r23, %r29, %r23;\n\t" 
                    "add.rn.f64 %r24, %r29, %r24;\n\t" 
                    "add.rn.f64 %r25, %r29, %r25;\n\t" 
                    "add.rn.f64 %r26, %r29, %r26;\n\t" 
                    "add.rn.f64 %r27, %r29, %r27;\n\t" 
                    "add.rn.f64 %r28, %r29, %r28;\n\t" 
                    "add.rn.f64 %r13, %r29, %r13;\n\t" 
                    "add.rn.f64 %r14, %r29, %r14;\n\t" 
                    "add.rn.f64 %r15, %r29, %r15;\n\t" 
                    "add.rn.f64 %r16, %r29, %r16;\n\t" 
                    "add.rn.f64 %r17, %r29, %r17;\n\t" 
                    "add.rn.f64 %r18, %r29, %r18;\n\t" 
                    "add.rn.f64 %r19, %r29, %r19;\n\t" 
                    "add.rn.f64 %r20, %r29, %r20;\n\t" 
                    "add.rn.f64 %r21, %r29, %r21;\n\t" 
                    "add.rn.f64 %r22, %r29, %r22;\n\t" 
                    "add.rn.f64 %r23, %r29, %r23;\n\t" 
                    "add.rn.f64 %r24, %r29, %r24;\n\t" 
                    "add.rn.f64 %r25, %r29, %r25;\n\t" 
                    "add.rn.f64 %r26, %r29, %r26;\n\t" 
                    "add.rn.f64 %r27, %r29, %r27;\n\t" 
                    "add.rn.f64 %r28, %r29, %r28;\n\t" 
                    "add.rn.f64 %r13, %r29, %r13;\n\t" 
                    "add.rn.f64 %r14, %r29, %r14;\n\t" 
                    "add.rn.f64 %r15, %r29, %r15;\n\t" 
                    "add.rn.f64 %r16, %r29, %r16;\n\t" 
                    "add.rn.f64 %r17, %r29, %r17;\n\t" 
                    "add.rn.f64 %r18, %r29, %r18;\n\t" 
                    "add.rn.f64 %r19, %r29, %r19;\n\t" 
                    "add.rn.f64 %r20, %r29, %r20;\n\t" 
                    "add.rn.f64 %r21, %r29, %r21;\n\t" 
                    "add.rn.f64 %r22, %r29, %r22;\n\t" 
                    "add.rn.f64 %r23, %r29, %r23;\n\t" 
                    "add.rn.f64 %r24, %r29, %r24;\n\t" 
                    "add.rn.f64 %r25, %r29, %r25;\n\t" 
                    "add.rn.f64 %r26, %r29, %r26;\n\t" 
                    "add.rn.f64 %r27, %r29, %r27;\n\t" 
                    "add.rn.f64 %r28, %r29, %r28;\n\t" 
                    "add.rn.f64 %r13, %r29, %r13;\n\t" 
                    "add.rn.f64 %r14, %r29, %r14;\n\t" 
                    "add.rn.f64 %r15, %r29, %r15;\n\t" 
                    "add.rn.f64 %r16, %r29, %r16;\n\t" 
                    "add.rn.f64 %r17, %r29, %r17;\n\t" 
                    "add.rn.f64 %r18, %r29, %r18;\n\t" 
                    "add.rn.f64 %r19, %r29, %r19;\n\t" 
                    "add.rn.f64 %r20, %r29, %r20;\n\t" 
                    "add.rn.f64 %r21, %r29, %r21;\n\t" 
                    "add.rn.f64 %r22, %r29, %r22;\n\t" 
                    "add.rn.f64 %r23, %r29, %r23;\n\t" 
                    "add.rn.f64 %r24, %r29, %r24;\n\t" 
                    "add.rn.f64 %r25, %r29, %r25;\n\t" 
                    "add.rn.f64 %r26, %r29, %r26;\n\t" 
                    "add.rn.f64 %r27, %r29, %r27;\n\t" 
                    "add.rn.f64 %r28, %r29, %r28;\n\t" 
                    "add.rn.f64 %r13, %r29, %r13;\n\t" 
                    "add.rn.f64 %r14, %r29, %r14;\n\t" 
                    "add.rn.f64 %r15, %r29, %r15;\n\t" 
                    "add.rn.f64 %r16, %r29, %r16;\n\t" 
                    "add.rn.f64 %r17, %r29, %r17;\n\t" 
                    "add.rn.f64 %r18, %r29, %r18;\n\t" 
                    "add.rn.f64 %r19, %r29, %r19;\n\t" 
                    "add.rn.f64 %r20, %r29, %r20;\n\t" 
                    "add.rn.f64 %r21, %r29, %r21;\n\t" 
                    "add.rn.f64 %r22, %r29, %r22;\n\t" 
                    "add.rn.f64 %r23, %r29, %r23;\n\t" 
                    "add.rn.f64 %r24, %r29, %r24;\n\t" 
                    "add.rn.f64 %r25, %r29, %r25;\n\t" 
                    "add.rn.f64 %r26, %r29, %r26;\n\t" 
                    "add.rn.f64 %r27, %r29, %r27;\n\t" 
                    "add.rn.f64 %r28, %r29, %r28;\n\t" 
                    "add.rn.f64 %r13, %r29, %r13;\n\t" 
                    "add.rn.f64 %r14, %r29, %r14;\n\t" 
                    "add.rn.f64 %r15, %r29, %r15;\n\t" 
                    "add.rn.f64 %r16, %r29, %r16;\n\t" 
                    "add.rn.f64 %r17, %r29, %r17;\n\t" 
                    "add.rn.f64 %r18, %r29, %r18;\n\t" 
                    "add.rn.f64 %r19, %r29, %r19;\n\t" 
                    "add.rn.f64 %r20, %r29, %r20;\n\t" 
                    "add.rn.f64 %r21, %r29, %r21;\n\t" 
                    "add.rn.f64 %r22, %r29, %r22;\n\t" 
                    "add.rn.f64 %r23, %r29, %r23;\n\t" 
                    "add.rn.f64 %r24, %r29, %r24;\n\t" 
                    "add.rn.f64 %r25, %r29, %r25;\n\t" 
                    "add.rn.f64 %r26, %r29, %r26;\n\t" 
                    "add.rn.f64 %r27, %r29, %r27;\n\t" 
                    "add.rn.f64 %r28, %r29, %r28;\n\t" 
                    "add.rn.f64 %r13, %r29, %r13;\n\t" 
                    "add.rn.f64 %r14, %r29, %r14;\n\t" 
                    "add.rn.f64 %r15, %r29, %r15;\n\t" 
                    "add.rn.f64 %r16, %r29, %r16;\n\t" 
                    "add.rn.f64 %r17, %r29, %r17;\n\t" 
                    "add.rn.f64 %r18, %r29, %r18;\n\t" 
                    "add.rn.f64 %r19, %r29, %r19;\n\t" 
                    "add.rn.f64 %r20, %r29, %r20;\n\t" 
                    "add.rn.f64 %r21, %r29, %r21;\n\t" 
                    "add.rn.f64 %r22, %r29, %r22;\n\t" 
                    "add.rn.f64 %r23, %r29, %r23;\n\t" 
                    "add.rn.f64 %r24, %r29, %r24;\n\t" 
                    "add.rn.f64 %r25, %r29, %r25;\n\t" 
                    "add.rn.f64 %r26, %r29, %r26;\n\t" 
                    "add.rn.f64 %r27, %r29, %r27;\n\t" 
                    "add.rn.f64 %r28, %r29, %r28;\n\t" 
                    "add.rn.f64 %r13, %r29, %r13;\n\t" 
                    "add.rn.f64 %r14, %r29, %r14;\n\t" 
                    "add.rn.f64 %r15, %r29, %r15;\n\t" 
                    "add.rn.f64 %r16, %r29, %r16;\n\t" 
                    "add.rn.f64 %r17, %r29, %r17;\n\t" 
                    "add.rn.f64 %r18, %r29, %r18;\n\t" 
                    "add.rn.f64 %r19, %r29, %r19;\n\t" 
                    "add.rn.f64 %r20, %r29, %r20;\n\t" 
                    "add.rn.f64 %r21, %r29, %r21;\n\t" 
                    "add.rn.f64 %r22, %r29, %r22;\n\t" 
                    "add.rn.f64 %r23, %r29, %r23;\n\t" 
                    "add.rn.f64 %r24, %r29, %r24;\n\t" 
                    "add.rn.f64 %r25, %r29, %r25;\n\t" 
                    "add.rn.f64 %r26, %r29, %r26;\n\t" 
                    "add.rn.f64 %r27, %r29, %r27;\n\t" 
                    "add.rn.f64 %r28, %r29, %r28;\n\t" 
                    );
        }
   
//        double temp; 
//        float output = 0.0;
//        asm("add.rn.f64 %0, r13, r14" : "=d"(temp));
//        asm("cvt.rn.f32.f64 %0, %1" : "=f"(output) : "d"(temp));
//        printf("%lf \n", output);
    }
    __syncthreads();

    //    if ((blockDim.x * blockIdx.x + threadIdx.x) == 0)
    *D = I1;

    __syncthreads();
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
    cudaMalloc((void**)&d_res, sizeof(double));
  
//    cudaMemcpy(d_A, h_A, sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_B, h_B, sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_C, h_C, sizeof(float), cudaMemcpyHostToDevice);
     
    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::string cmd = "GPUJOULE_DIR/nvml/example/power_monitor 5 > GPUJOULE_DIR/energy_model_ubench/energy_model_data/data_movement_energy/l2_cache/fadd_l2d_100_0_64p_asm_power.txt &";
    std::system(cmd.c_str());
    std::system("sleep 5");
   
    cudaEventRecord(start, 0);
    cudaProfilerStart();
    
//    compute<<<num_blocks, num_threads_per_block>>>(d_A, d_B, d_C, d_res, iterations);
    shared_latency<<<num_blocks, num_threads_per_block>>>(d_res, iterations, divergence);

    cudaDeviceSynchronize();
    cudaProfilerStop();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);

    std::system("killall power_monitor");
    std::cout << time << std::endl;
  
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaMemcpy(h_res, d_res, sizeof(double), cudaMemcpyDeviceToHost);

    return 0;
}
