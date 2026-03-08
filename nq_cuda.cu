#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nq_types.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>

#define CUDA_BLOCK_SIZE 256
#define DEFAULT_GPU_BATCH_TASKS 500000

static __device__ SOLUTIONTYPE solveTaskDevice(int board_minus,
                                               BITFIELDTYPE mask,
                                               NqueenTask task)
{
    BITFIELDTYPE aQueenBitCol[MAX_BOARDSIZE];
    BITFIELDTYPE aQueenBitPosDiag[MAX_BOARDSIZE];
    BITFIELDTYPE aQueenBitNegDiag[MAX_BOARDSIZE];
    BITFIELDTYPE aStack[MAX_BOARDSIZE + 1];
    BITFIELDTYPE bitfield;
    BITFIELDTYPE lsb;
    int numrows;
    int stack_count;
    SOLUTIONTYPE solutions = 0;

    numrows = task.start_numrows;
    stack_count = 0;
    bitfield = task.bitfield;
    aQueenBitCol[numrows] = task.col;
    aQueenBitPosDiag[numrows] = task.pos_diag;
    aQueenBitNegDiag[numrows] = task.neg_diag;

    for (;;)
    {
        if (0 == bitfield)
        {
            if (0 == stack_count)
            {
                break;
            }
            bitfield = aStack[--stack_count];
            --numrows;
            continue;
        }

        lsb = bitfield & (~bitfield + 1);
        bitfield &= ~lsb;

        if (numrows < board_minus)
        {
            int n = numrows++;
            aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
            aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1;
            aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1;
            aStack[stack_count++] = bitfield;
            bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
            continue;
        }

        ++solutions;
        if (0 == stack_count)
        {
            break;
        }
        bitfield = aStack[--stack_count];
        --numrows;
    }

    return solutions;
}

static __global__ void solveTasksKernel(int board_minus,
                                        BITFIELDTYPE mask,
                                        const NqueenTask* tasks,
                                        int task_count,
                                        SOLUTIONTYPE* block_sums)
{
    __shared__ SOLUTIONTYPE shared_sums[CUDA_BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    SOLUTIONTYPE local = 0;

    if (idx < task_count)
    {
        local = solveTaskDevice(board_minus, mask, tasks[idx]);
    }
    shared_sums[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared_sums[threadIdx.x] += shared_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (0 == threadIdx.x)
    {
        block_sums[blockIdx.x] = shared_sums[0];
    }
}

static __global__ void reduceSumsKernel(const SOLUTIONTYPE* in_values,
                                        int in_count,
                                        SOLUTIONTYPE* out_values)
{
    __shared__ SOLUTIONTYPE shared_sums[CUDA_BLOCK_SIZE];
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    SOLUTIONTYPE local = 0;

    if (global_index < in_count)
    {
        local = in_values[global_index];
    }
    shared_sums[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared_sums[threadIdx.x] += shared_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (0 == threadIdx.x)
    {
        out_values[blockIdx.x] = shared_sums[0];
    }
}

static int shouldPrintGpuTiming(void)
{
    const char* value = getenv("NQ_GPU_TIMING");

    if (NULL == value || '\0' == *value)
    {
        return 0;
    }
    if (0 == strcmp(value, "0"))
    {
        return 0;
    }

    return 1;
}

static int getGpuBatchTasks(void)
{
    const char* value = getenv("NQ_GPU_BATCH_TASKS");
    int batch_tasks = DEFAULT_GPU_BATCH_TASKS;

    if (value != NULL && *value != '\0')
    {
        char* endp = NULL;
        long parsed = strtol(value, &endp, 10);
        if (endp != value && parsed >= 1)
        {
            batch_tasks = (int)parsed;
        }
    }

    if (batch_tasks < 1)
    {
        batch_tasks = 1;
    }
    return batch_tasks;
}

extern "C" int solveTasksGpuCuda(int board_minus,
                                 BITFIELDTYPE mask,
                                 const NqueenTask* tasks,
                                 int task_count,
                                 SOLUTIONTYPE* total_out)
{
    int batch_capacity;
    int max_blocks;
    size_t task_buffer_bytes;
    size_t partial_bytes;
    NqueenTask* d_tasks = NULL;
    SOLUTIONTYPE* d_partial_a = NULL;
    SOLUTIONTYPE* d_partial_b = NULL;
    SOLUTIONTYPE total = 0;
    SOLUTIONTYPE batch_total = 0;
    cudaError_t err;
    cudaEvent_t ev_start = NULL;
    cudaEvent_t ev_stop = NULL;
    float h2d_total_ms = 0.0f;
    float kernel_total_ms = 0.0f;
    float d2h_total_ms = 0.0f;
    float phase_ms = 0.0f;
    int print_timing = shouldPrintGpuTiming();
    int batch_tasks = getGpuBatchTasks();
    int batches = 0;
    int total_reduction_passes = 0;
    int max_reduction_passes = 0;
    int success = 0;
    int offset;

    if (NULL == total_out)
    {
        return 0;
    }

    *total_out = 0;
    if (task_count <= 0)
    {
        return 1;
    }

    batch_capacity = (batch_tasks < task_count) ? batch_tasks : task_count;
    max_blocks = (batch_capacity + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    task_buffer_bytes = (size_t)batch_capacity * sizeof(NqueenTask);
    partial_bytes = (size_t)max_blocks * sizeof(SOLUTIONTYPE);

    if (print_timing)
    {
        if (cudaSuccess != cudaEventCreate(&ev_start) ||
            cudaSuccess != cudaEventCreate(&ev_stop))
        {
            fprintf(stderr, "CUDA warning: failed to create timing events.\n");
            if (ev_start) cudaEventDestroy(ev_start);
            if (ev_stop) cudaEventDestroy(ev_stop);
            ev_start = ev_stop = NULL;
            print_timing = 0;
        }
    }

    err = cudaMalloc((void**)&d_tasks, task_buffer_bytes);
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA error (cudaMalloc d_tasks): %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMalloc((void**)&d_partial_a, partial_bytes);
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA error (cudaMalloc d_partial_a): %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    if (max_blocks > 1)
    {
        err = cudaMalloc((void**)&d_partial_b, partial_bytes);
        if (cudaSuccess != err)
        {
            fprintf(stderr, "CUDA error (cudaMalloc d_partial_b): %s\n", cudaGetErrorString(err));
            goto cleanup;
        }
    }

    for (offset = 0; offset < task_count; offset += batch_capacity)
    {
        int current_tasks = task_count - offset;
        int blocks;
        int current_count;
        const SOLUTIONTYPE* d_reduce_in;
        SOLUTIONTYPE* d_reduce_out;
        const SOLUTIONTYPE* d_final;
        int reduction_passes = 0;

        if (current_tasks > batch_capacity)
        {
            current_tasks = batch_capacity;
        }
        blocks = (current_tasks + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        ++batches;

        if (print_timing)
        {
            cudaEventRecord(ev_start, 0);
        }

        err = cudaMemcpy(d_tasks,
                         tasks + offset,
                         (size_t)current_tasks * sizeof(NqueenTask),
                         cudaMemcpyHostToDevice);
        if (cudaSuccess != err)
        {
            fprintf(stderr, "CUDA error (cudaMemcpy H2D tasks): %s\n", cudaGetErrorString(err));
            goto cleanup;
        }

        if (print_timing)
        {
            cudaEventRecord(ev_stop, 0);
            cudaEventSynchronize(ev_stop);
            cudaEventElapsedTime(&phase_ms, ev_start, ev_stop);
            h2d_total_ms += phase_ms;
        }

        if (print_timing)
        {
            cudaEventRecord(ev_start, 0);
        }

        solveTasksKernel<<<blocks, CUDA_BLOCK_SIZE>>>(board_minus, mask, d_tasks, current_tasks, d_partial_a);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "CUDA error (kernel launch): %s\n", cudaGetErrorString(err));
            goto cleanup;
        }

        current_count = blocks;
        d_reduce_in = d_partial_a;
        while (current_count > 1)
        {
            int out_count = (current_count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            if (NULL == d_partial_b)
            {
                fprintf(stderr, "CUDA error: reduction buffer not allocated.\n");
                goto cleanup;
            }

            d_reduce_out = (d_reduce_in == d_partial_a) ? d_partial_b : d_partial_a;
            reduceSumsKernel<<<out_count, CUDA_BLOCK_SIZE>>>(d_reduce_in, current_count, d_reduce_out);
            err = cudaGetLastError();
            if (cudaSuccess != err)
            {
                fprintf(stderr, "CUDA error (reduction kernel launch): %s\n", cudaGetErrorString(err));
                goto cleanup;
            }
            d_reduce_in = d_reduce_out;
            current_count = out_count;
            ++reduction_passes;
        }
        d_final = d_reduce_in;

        err = cudaDeviceSynchronize();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "CUDA error (kernel sync): %s\n", cudaGetErrorString(err));
            goto cleanup;
        }

        if (print_timing)
        {
            cudaEventRecord(ev_stop, 0);
            cudaEventSynchronize(ev_stop);
            cudaEventElapsedTime(&phase_ms, ev_start, ev_stop);
            kernel_total_ms += phase_ms;
        }

        if (print_timing)
        {
            cudaEventRecord(ev_start, 0);
        }

        err = cudaMemcpy(&batch_total, d_final, sizeof(SOLUTIONTYPE), cudaMemcpyDeviceToHost);
        if (cudaSuccess != err)
        {
            fprintf(stderr, "CUDA error (cudaMemcpy D2H total): %s\n", cudaGetErrorString(err));
            goto cleanup;
        }

        if (print_timing)
        {
            cudaEventRecord(ev_stop, 0);
            cudaEventSynchronize(ev_stop);
            cudaEventElapsedTime(&phase_ms, ev_start, ev_stop);
            d2h_total_ms += phase_ms;
        }

        total += batch_total;
        total_reduction_passes += reduction_passes;
        if (reduction_passes > max_reduction_passes)
        {
            max_reduction_passes = reduction_passes;
        }
    }

    *total_out = total;

    if (print_timing)
    {
        fprintf(stderr,
                "GPU timing (ms): H2D=%.3f kernel+reduce=%.3f D2H=%.3f tasks=%d batch_tasks=%d batches=%d max_blocks=%d total_passes=%d max_passes=%d\n",
                h2d_total_ms,
                kernel_total_ms,
                d2h_total_ms,
                task_count,
                batch_capacity,
                batches,
                max_blocks,
                total_reduction_passes,
                max_reduction_passes);
    }

    success = 1;

cleanup:
    if (d_partial_b) cudaFree(d_partial_b);
    if (d_partial_a) cudaFree(d_partial_a);
    if (d_tasks) cudaFree(d_tasks);
    if (ev_start) cudaEventDestroy(ev_start);
    if (ev_stop) cudaEventDestroy(ev_stop);
    return success;
}

#else

int solveTasksGpuCuda(int board_minus,
                      BITFIELDTYPE mask,
                      const NqueenTask* tasks,
                      int task_count,
                      SOLUTIONTYPE* total_out)
{
    (void)board_minus;
    (void)mask;
    (void)tasks;
    (void)task_count;
    (void)total_out;
    return 0;
}

#endif
