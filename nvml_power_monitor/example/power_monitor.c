/***************************************************************************\
  |*                                                                           *|
  |*      Copyright 2010-2016 NVIDIA Corporation.  All rights reserved.        *|
  |*                                                                           *|
  |*   NOTICE TO USER:                                                         *|
  |*                                                                           *|
  |*   This source code is subject to NVIDIA ownership rights under U.S.       *|
  |*   and international Copyright laws.  Users and possessors of this         *|
  |*   source code are hereby granted a nonexclusive, royalty-free             *|
  |*   license to use this code in individual and commercial software.         *|
  |*                                                                           *|
  |*   NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE     *|
  |*   CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR         *|
  |*   IMPLIED WARRANTY OF ANY KIND. NVIDIA DISCLAIMS ALL WARRANTIES WITH      *|
  |*   REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF         *|
  |*   MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR          *|
  |*   PURPOSE. IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL,            *|
  |*   INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES          *|
  |*   WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN      *|
  |*   AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING     *|
  |*   OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE      *|
  |*   CODE.                                                                   *|
  |*                                                                           *|
  |*   U.S. Government End Users. This source code is a "commercial item"      *|
  |*   as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting       *|
  |*   of "commercial computer  software" and "commercial computer software    *|
  |*   documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)   *|
  |*   and is provided to the U.S. Government only as a commercial end item.   *|
  |*   Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through        *|
  |*   227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the       *|
  |*   source code with only those rights set forth herein.                    *|
  |*                                                                           *|
  |*   Any use of this source code in individual and commercial software must  *| 
  |*   include, in the user documentation and internal comments to the code,   *|
  |*   the above Disclaimer and U.S. Government End Users Notice.              *|
  |*                                                                           *|
  |*                                                                           *|
  \***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <nvml.h>
#include <signal.h>

#define GPU_NAME ""

void monitor_power(nvmlDevice_t device)
{
    nvmlReturn_t result;
    unsigned int device_count, i;

    result = nvmlDeviceGetCount(&device_count);
    if (NVML_SUCCESS != result)
    { 
        printf("Failed to query device count: %s\n", nvmlErrorString(result));
        goto Error;
    }

    unsigned int power;

    result = nvmlDeviceGetPowerUsage(device, &power);

    if (NVML_ERROR_NOT_SUPPORTED == result)
        printf("This does not support power measurement\n");
    else if (NVML_SUCCESS != result)
    {
        printf("Failed to get power for device %i: %s\n", i, nvmlErrorString(result));
        goto Error;
    }
    
    unsigned int temp;
    result = nvmlDeviceGetTemperature(device, 0, &temp);

    if (NVML_ERROR_NOT_SUPPORTED == result)
        printf("This does not support temperature measurement\n");
    else if (NVML_SUCCESS != result)
    {
        printf("Failed to get temperature for device %i: %s\n", i, nvmlErrorString(result));
        goto Error;
    }

    unsigned int sm_clock;
    result = nvmlDeviceGetClockInfo(device, 1, &sm_clock);
    
    if (NVML_ERROR_NOT_SUPPORTED == result)
        printf("This does not support SM Clock measurement\n");
    else if (NVML_SUCCESS != result)
    {
        printf("Failed to get SM Clock for device %i: %s\n", i, nvmlErrorString(result));
        goto Error;
    }
    
    unsigned int graphics_clock;
    result = nvmlDeviceGetClockInfo(device, 0, &graphics_clock);
    
    if (NVML_ERROR_NOT_SUPPORTED == result)
        printf("This does not support Graphics Clock  measurement\n");
    else if (NVML_SUCCESS != result)
    {
        printf("Failed to get Graphics Clock for device %i: %s\n", i, nvmlErrorString(result));
        goto Error;
    }

    unsigned int mem_clock;
    result = nvmlDeviceGetClockInfo(device, 2, &mem_clock);
    
    if (NVML_ERROR_NOT_SUPPORTED == result)
        printf("This does not support Mem Clock  measurement\n");
    else if (NVML_SUCCESS != result)
    {
        printf("Failed to get Mem Clock for device %i: %s\n", i, nvmlErrorString(result));
        goto Error;
    }

    printf("%d,%d,%d,%d,%d\n", power, temp, sm_clock, graphics_clock, mem_clock);
    return;


Error:
    result = nvmlShutdown();
    if (NVML_SUCCESS != result)
        printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));

    exit(1);
}

void initialize_nvml()
{
    nvmlReturn_t result;
    // First initialize NVML library
    result = nvmlInit();
    if (NVML_SUCCESS != result)
    { 
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));

        exit(1);
    }
}

void shutdown_nvml()
{
    nvmlReturn_t result;
    result = nvmlShutdown();
    if (NVML_SUCCESS != result)
        printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
}

void usage() {
    printf("Usage: binary <monitoring period (ms)>\n");
}

volatile sig_atomic_t flag = 0;

void end_monitoring(int sig) {
    flag = 1;
}

int main(int argc, char** argv) 
{
    if (argc < 2) 
    {
        usage();
        return 0;
    }
    
    unsigned int sleep_useconds = atoi(argv[1])*1000;
    initialize_nvml();

    nvmlReturn_t result;
    nvmlDevice_t device_of_interest;
    unsigned int device_count, i;
    result = nvmlDeviceGetCount(&device_count);
    if(NVML_SUCCESS != result)
    {
        printf("Failed to query device count: %s\n", nvmlErrorString(result));
        shutdown_nvml();
        exit(1);
    }
    
    for (i = 0; i < device_count; i++)
    {
        nvmlDevice_t device;
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (NVML_SUCCESS != result)
        { 
            printf("Failed to get handle for device %i: %s\n", i, nvmlErrorString(result));
            shutdown_nvml();
            exit(1);
        }

        result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        if (NVML_SUCCESS != result)
        { 
            printf("Failed to get name of device %i: %s\n", i, nvmlErrorString(result));
            shutdown_nvml();
            exit(1);
        }
    
        if (strcmp(name, GPU_NAME)) {
            continue;
        }
        else
        {
            device_of_interest = device;
            break;
        }
    }

    signal(SIGINT, end_monitoring);

    while (1)
    {
        if (flag) {
            break;
        }
        monitor_power(device_of_interest);
        usleep(sleep_useconds);
    }

    shutdown_nvml();
    return 0;
}
