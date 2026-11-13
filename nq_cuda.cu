#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nq_types.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>

#define CUDA_BLOCK_SIZE 64
#define GPU_KERNEL_DEFAULT_SMEM_LIMIT (48 * 1024)
#define DEFAULT_GPU_BATCH_TASKS 500000
#define DEFAULT_GPU_TARGET_TASKS_PER_BLOCK 8192
#define DEFAULT_GPU_AUTOTUNE_SAMPLE_TASKS 8192
#define DEFAULT_GPU_PIPELINE_SLOTS 2
#define MAX_GPU_PIPELINE_SLOTS 2

/*
 * Interchangeable subtree solvers (NQ_GPU_KERNEL=legacy|peel|peel-local):
 *   legacy  solveTaskDevice below: 64-bit masks, all DFS state in shared memory,
 *           one leaf counted per iteration.
 *   peel    nqPeelSolve from nq_peel.h: 32-bit masks, state in registers, last
 *           four rows peeled with a popcount at the bottom. Needs less shared
 *           memory per thread (one 16-byte frame per stacked row).
 *   peel-local  the same solver with its frames in thread-local memory, so
 *           shared memory no longer limits how many blocks fit on an SM.
 */
#define NQ_PEEL_FN static __device__ __forceinline__
#define NQ_PEEL_POPCOUNT(x) __popc(x)
#include "nq_peel.h"

#define GPU_KERNEL_LEGACY 0
#define GPU_KERNEL_PEEL 1
#define GPU_KERNEL_PEEL_LOCAL 2

static __device__ SOLUTIONTYPE solveTaskDevice(int start_offset,
                                               int last_row,
                                               BITFIELDTYPE mask,
                                               BITFIELDTYPE* s_col,
                                               BITFIELDTYPE* s_pos,
                                               BITFIELDTYPE* s_neg,
                                               BITFIELDTYPE* s_stack,
                                               NqueenTask task)
{
    BITFIELDTYPE bitfield;
    BITFIELDTYPE lsb;
    int row;
    int stack_count;
    SOLUTIONTYPE solutions = 0;

    row = task.start_numrows - start_offset;
    stack_count = 0;
    bitfield = task.bitfield;
    s_col[row] = task.col;
    s_pos[row] = task.pos_diag;
    s_neg[row] = task.neg_diag;

    for (;;)
    {
        if (0 == bitfield)
        {
            if (0 == stack_count)
            {
                break;
            }
            bitfield = s_stack[--stack_count];
            --row;
            continue;
        }

        lsb = bitfield & (~bitfield + 1);
        bitfield &= ~lsb;

        if (row < last_row)
        {
            int n = row++;
            s_col[row] = s_col[n] | lsb;
            s_neg[row] = (s_neg[n] | lsb) >> 1;
            s_pos[row] = (s_pos[n] | lsb) << 1;
            s_stack[stack_count++] = bitfield;
            bitfield = mask & ~(s_col[row] | s_neg[row] | s_pos[row]);
            continue;
        }

        ++solutions;
        if (0 == stack_count)
        {
            break;
        }
        bitfield = s_stack[--stack_count];
        --row;
    }

    return solutions;
}

/*
 * depth_stride is the per-thread slot count in the dynamic shared-memory
 * region, in units that depend on kernel_mode:
 *   legacy: 4 arrays of depth_stride 64-bit words (32 * depth_stride bytes)
 *   peel:   depth_stride NqPeelFrame entries       (16 * depth_stride bytes)
 *   peel-local: nothing (frames are a local array in the kernel)
 * getKernelDepthStride()/getKernelSharedBytes() below are the single source
 * of truth for both.
 */
static __global__ void solveTasksKernel(int kernel_mode,
                                        int start_offset,
                                        int last_row,
                                        BITFIELDTYPE mask,
                                        int depth_stride,
                                        const NqueenTask* tasks,
                                        int task_count,
                                        unsigned int* next_task,
                                        SOLUTIONTYPE* block_sums)
{
    extern __shared__ unsigned long long s_dyn[];
    __shared__ SOLUTIONTYPE shared_sums[CUDA_BLOCK_SIZE];

    SOLUTIONTYPE local = 0;

    if (GPU_KERNEL_LEGACY != kernel_mode)
    {
        NqPeelFrame local_stk[NQ_PEEL_STACK_FRAMES(NQ_PEEL_MAX_BOARD)];
        NqPeelFrame* stk = (GPU_KERNEL_PEEL_LOCAL == kernel_mode)
            ? local_stk
            : (NqPeelFrame*)s_dyn + (size_t)threadIdx.x * depth_stride;
        const uint32_t mask32 = (uint32_t)mask;
        const uint32_t board_minus = (uint32_t)(last_row + start_offset);

        for (;;)
        {
            unsigned int idx = atomicAdd(next_task, 1u);
            if (idx >= (unsigned int)task_count)
            {
                break;
            }
            /* Narrowing to 32 bits is exact: only pos_diag bits already shifted
               past the board are dropped, and they can never shift back. */
            local += nqPeelSolve(mask32,
                                 board_minus,
                                 (uint32_t)tasks[idx].start_numrows,
                                 (uint32_t)tasks[idx].col,
                                 (uint32_t)tasks[idx].pos_diag,
                                 (uint32_t)tasks[idx].neg_diag,
                                 (uint32_t)tasks[idx].bitfield,
                                 stk);
        }
    }
    else
    {
        BITFIELDTYPE* s_thread = (BITFIELDTYPE*)s_dyn + (size_t)threadIdx.x * 4 * depth_stride;
        BITFIELDTYPE* s_col   = s_thread;
        BITFIELDTYPE* s_pos   = s_thread + depth_stride;
        BITFIELDTYPE* s_neg   = s_thread + 2 * depth_stride;
        BITFIELDTYPE* s_stack = s_thread + 3 * depth_stride;

        for (;;)
        {
            unsigned int idx = atomicAdd(next_task, 1u);
            if (idx >= (unsigned int)task_count)
            {
                break;
            }
            local += solveTaskDevice(start_offset, last_row, mask, s_col, s_pos, s_neg, s_stack, tasks[idx]);
        }
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

/* NQ_GPU_KERNEL=peel or peel-local selects a peeled kernel; anything else is legacy. */
static int getGpuKernelMode(int board_size)
{
    const char* value = getenv("NQ_GPU_KERNEL");

    if (value != NULL && (0 == strcmp(value, "peel") || 0 == strcmp(value, "peel-local")))
    {
        if (board_size <= NQ_PEEL_MAX_BOARD)
        {
            return (0 == strcmp(value, "peel")) ? GPU_KERNEL_PEEL : GPU_KERNEL_PEEL_LOCAL;
        }
        fprintf(stderr, "CUDA warning: the peel kernels support boards up to %d; using legacy kernel.\n",
                NQ_PEEL_MAX_BOARD);
    }
    return GPU_KERNEL_LEGACY;
}

/* max_rows = most rows any task still has to fill (board_size - smallest start row). */
static int getKernelDepthStride(int kernel_mode, int max_rows)
{
    int stride = (GPU_KERNEL_LEGACY != kernel_mode) ? NQ_PEEL_STACK_FRAMES(max_rows) : max_rows;
    return (stride < 1) ? 1 : stride;
}

static size_t getKernelSharedBytes(int kernel_mode, int depth_stride)
{
    size_t per_thread;

    if (GPU_KERNEL_PEEL_LOCAL == kernel_mode)
    {
        return 0;
    }
    per_thread = (GPU_KERNEL_PEEL == kernel_mode)
        ? sizeof(NqPeelFrame) * (size_t)depth_stride
        : (size_t)4 * sizeof(BITFIELDTYPE) * (size_t)depth_stride;
    return per_thread * (size_t)CUDA_BLOCK_SIZE;
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

static int getGpuPersistentBlocks(void)
{
    const char* value = getenv("NQ_GPU_PERSIST_BLOCKS");
    int blocks = 0;

    if (value != NULL && *value != '\0')
    {
        char* endp = NULL;
        long parsed = strtol(value, &endp, 10);
        if (endp != value && parsed >= 1)
        {
            blocks = (int)parsed;
        }
    }

    return blocks;
}

static int getGpuTargetTasksPerBlock(int* was_overridden)
{
    const char* value = getenv("NQ_GPU_TARGET_TASKS_PER_BLOCK");
    int target = DEFAULT_GPU_TARGET_TASKS_PER_BLOCK;

    if (was_overridden != NULL)
    {
        *was_overridden = 0;
    }

    if (value != NULL && *value != '\0')
    {
        char* endp = NULL;
        long parsed = strtol(value, &endp, 10);
        if (endp != value && parsed >= 1)
        {
            target = (int)parsed;
            if (was_overridden != NULL)
            {
                *was_overridden = 1;
            }
        }
    }

    if (target < 1)
    {
        target = 1;
    }
    return target;
}

static int shouldAutoTuneGpuTarget(void)
{
    const char* value = getenv("NQ_GPU_AUTOTUNE_TARGET");

    if (NULL == value || '\0' == *value)
    {
        return 0;
    }

    if (0 == strcmp(value, "1") ||
        0 == strcmp(value, "true") ||
        0 == strcmp(value, "TRUE") ||
        0 == strcmp(value, "on") ||
        0 == strcmp(value, "ON"))
    {
        return 1;
    }

    return 0;
}

static int getGpuAutoTuneSampleTasks(void)
{
    const char* value = getenv("NQ_GPU_AUTOTUNE_SAMPLE_TASKS");
    int sample_tasks = DEFAULT_GPU_AUTOTUNE_SAMPLE_TASKS;

    if (value != NULL && *value != '\0')
    {
        char* endp = NULL;
        long parsed = strtol(value, &endp, 10);
        if (endp != value && parsed >= 1)
        {
            sample_tasks = (int)parsed;
        }
    }

    if (sample_tasks < 1)
    {
        sample_tasks = 1;
    }
    return sample_tasks;
}

typedef struct
{
    cudaStream_t stream;
    cudaEvent_t done_event;
    cudaEvent_t h2d_start;
    cudaEvent_t h2d_stop;
    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;
    cudaEvent_t d2h_start;
    cudaEvent_t d2h_stop;
    NqueenTask* d_tasks;
    SOLUTIONTYPE* d_partial_a;
    SOLUTIONTYPE* d_partial_b;
    unsigned int* d_next_task;
    int in_use;
    int reduction_passes;
} GpuPipelineSlot;

static int getGpuPipelineSlots(void)
{
    const char* value = getenv("NQ_GPU_PIPELINE_SLOTS");
    int slots = DEFAULT_GPU_PIPELINE_SLOTS;

    if (value != NULL && *value != '\0')
    {
        char* endp = NULL;
        long parsed = strtol(value, &endp, 10);
        if (endp != value && parsed >= 1)
        {
            slots = (int)parsed;
        }
    }

    if (slots < 1)
    {
        slots = 1;
    }
    if (slots > MAX_GPU_PIPELINE_SLOTS)
    {
        slots = MAX_GPU_PIPELINE_SLOTS;
    }
    return slots;
}

static int finalizePipelineSlot(GpuPipelineSlot* slot,
                                int slot_index,
                                SOLUTIONTYPE* h_batch_totals,
                                int print_timing,
                                float* h2d_total_ms,
                                float* kernel_total_ms,
                                float* d2h_total_ms,
                                SOLUTIONTYPE* total,
                                int* total_reduction_passes,
                                int* max_reduction_passes)
{
    cudaError_t err;

    if (!slot->in_use)
    {
        return 1;
    }

    err = cudaEventSynchronize(slot->done_event);
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA error (stream sync): %s\n", cudaGetErrorString(err));
        return 0;
    }

    *total += h_batch_totals[slot_index];
    *total_reduction_passes += slot->reduction_passes;
    if (slot->reduction_passes > *max_reduction_passes)
    {
        *max_reduction_passes = slot->reduction_passes;
    }

    if (print_timing)
    {
        float phase_ms = 0.0f;

        err = cudaEventElapsedTime(&phase_ms, slot->h2d_start, slot->h2d_stop);
        if (cudaSuccess == err)
        {
            *h2d_total_ms += phase_ms;
        }

        err = cudaEventElapsedTime(&phase_ms, slot->kernel_start, slot->kernel_stop);
        if (cudaSuccess == err)
        {
            *kernel_total_ms += phase_ms;
        }

        err = cudaEventElapsedTime(&phase_ms, slot->d2h_start, slot->d2h_stop);
        if (cudaSuccess == err)
        {
            *d2h_total_ms += phase_ms;
        }
    }

    slot->in_use = 0;
    slot->reduction_passes = 0;
    return 1;
}

static int getLaunchBlocksCap(int current_tasks)
{
    int blocks_by_cap = (current_tasks + (CUDA_BLOCK_SIZE * 8) - 1) / (CUDA_BLOCK_SIZE * 8);

    if (blocks_by_cap < 1)
    {
        blocks_by_cap = 1;
    }
    return blocks_by_cap;
}

static int getPersistentLaunchBlocks(int current_tasks,
                                     int sm_count,
                                     int override_blocks,
                                     int target_tasks_per_block)
{
    int blocks_by_cap;
    int blocks_by_grain;
    int min_blocks_by_sm;
    int blocks;

    if (override_blocks > 0)
    {
        return override_blocks;
    }

    blocks_by_cap = getLaunchBlocksCap(current_tasks);

    blocks_by_grain = (current_tasks + target_tasks_per_block - 1) / target_tasks_per_block;
    if (blocks_by_grain < 1)
    {
        blocks_by_grain = 1;
    }

    if (sm_count > 0)
    {
        min_blocks_by_sm = sm_count * 2;
        if (min_blocks_by_sm < 1)
        {
            min_blocks_by_sm = 1;
        }
    }
    else
    {
        min_blocks_by_sm = 1;
    }

    blocks = blocks_by_grain;
    if (blocks < min_blocks_by_sm)
    {
        blocks = min_blocks_by_sm;
    }
    if (blocks > blocks_by_cap)
    {
        blocks = blocks_by_cap;
    }
    if (blocks < 1)
    {
        blocks = 1;
    }

    return blocks;
}

static int addUniqueCandidate(int* values, int* count, int value, int max_value)
{
    int i;

    if (value < 1)
    {
        value = 1;
    }
    if (value > max_value)
    {
        value = max_value;
    }

    for (i = 0; i < *count; ++i)
    {
        if (values[i] == value)
        {
            return 1;
        }
    }

    values[*count] = value;
    ++(*count);
    return 1;
}

static int shouldRunAutotuneSample(int sample_max_blocks, int baseline_blocks)
{
    if (sample_max_blocks < 8)
    {
        return 0;
    }

    if (baseline_blocks > 0 && (sample_max_blocks * 2) < baseline_blocks)
    {
        return 0;
    }

    return 1;
}

static int autotuneTargetTasksPerBlock(int kernel_mode,
                                       int start_offset,
                                       int last_row,
                                       BITFIELDTYPE mask,
                                       int depth_stride,
                                       size_t dyn_shared_bytes,
                                       const NqueenTask* d_tasks,
                                       int sample_tasks,
                                       int sm_count,
                                       int base_target,
                                       int sample_max_blocks,
                                       unsigned int* d_next_task,
                                       SOLUTIONTYPE* d_partial_a,
                                       int* tuned_target_out,
                                       int* tuned_blocks_out,
                                       float* tune_ms_total_out)
{
    int base_blocks;
    int candidates[10];
    int candidate_count = 0;
    int best_blocks = 0;
    float best_ms = 0.0f;
    float total_ms = 0.0f;
    cudaEvent_t ev_start = NULL;
    cudaEvent_t ev_stop = NULL;
    int had_error = 0;
    int i;

    if (sample_tasks < 1024 ||
        sample_max_blocks < 1 ||
        NULL == tuned_target_out ||
        NULL == tuned_blocks_out ||
        NULL == tune_ms_total_out)
    {
        return 0;
    }

    base_blocks = getPersistentLaunchBlocks(sample_tasks, sm_count, 0, base_target);

    addUniqueCandidate(candidates, &candidate_count, base_blocks / 2, sample_max_blocks);
    addUniqueCandidate(candidates, &candidate_count, (base_blocks * 3) / 4, sample_max_blocks);
    addUniqueCandidate(candidates, &candidate_count, base_blocks, sample_max_blocks);
    addUniqueCandidate(candidates, &candidate_count, (base_blocks * 5) / 4, sample_max_blocks);
    addUniqueCandidate(candidates, &candidate_count, (base_blocks * 3) / 2, sample_max_blocks);
    addUniqueCandidate(candidates, &candidate_count, base_blocks * 2, sample_max_blocks);
    if (sm_count > 0)
    {
        addUniqueCandidate(candidates, &candidate_count, sm_count, sample_max_blocks);
        addUniqueCandidate(candidates, &candidate_count, sm_count * 2, sample_max_blocks);
        addUniqueCandidate(candidates, &candidate_count, sm_count * 3, sample_max_blocks);
    }

    if (candidate_count < 2)
    {
        return 0;
    }

    if (cudaSuccess != cudaEventCreate(&ev_start) ||
        cudaSuccess != cudaEventCreate(&ev_stop))
    {
        if (ev_start) cudaEventDestroy(ev_start);
        if (ev_stop) cudaEventDestroy(ev_stop);
        return 0;
    }

    for (i = 0; i < candidate_count; ++i)
    {
        int launch_blocks = candidates[i];
        float elapsed_ms = 0.0f;
        cudaError_t err;

        err = cudaMemset(d_next_task, 0, sizeof(unsigned int));
        if (cudaSuccess != err)
        {
            had_error = 1;
            break;
        }

        cudaEventRecord(ev_start, 0);
        solveTasksKernel<<<launch_blocks, CUDA_BLOCK_SIZE, dyn_shared_bytes>>>(kernel_mode, start_offset, last_row, mask, depth_stride, d_tasks, sample_tasks, d_next_task, d_partial_a);
        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            had_error = 1;
            break;
        }
        cudaEventRecord(ev_stop, 0);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
        total_ms += elapsed_ms;

        if (0 == i || elapsed_ms < best_ms)
        {
            best_ms = elapsed_ms;
            best_blocks = launch_blocks;
        }
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    if (had_error || best_blocks < 1)
    {
        return 0;
    }

    *tuned_blocks_out = best_blocks;
    *tuned_target_out = (base_target * base_blocks + best_blocks - 1) / best_blocks;
    if (*tuned_target_out < 1)
    {
        *tuned_target_out = 1;
    }
    *tune_ms_total_out = total_ms;
    return 1;
}

extern "C" int solveTasksGpuCuda(int board_minus,
                                 BITFIELDTYPE mask,
                                 const NqueenTask* tasks,
                                 int task_count,
                                 SOLUTIONTYPE* total_out)
{
    int batch_capacity;
    int max_launch_blocks;
    size_t task_buffer_bytes;
    size_t partial_bytes;
    GpuPipelineSlot slots[MAX_GPU_PIPELINE_SLOTS];
    SOLUTIONTYPE h_batch_totals[MAX_GPU_PIPELINE_SLOTS];
    SOLUTIONTYPE total = 0;
    cudaError_t err;
    cudaDeviceProp device_prop;
    float h2d_total_ms = 0.0f;
    float kernel_total_ms = 0.0f;
    float d2h_total_ms = 0.0f;
    int print_timing = shouldPrintGpuTiming();
    int batch_tasks = getGpuBatchTasks();
    int persist_blocks = getGpuPersistentBlocks();
    int target_override = 0;
    int target_tasks_per_block = getGpuTargetTasksPerBlock(&target_override);
    int autotune_target = shouldAutoTuneGpuTarget();
    int autotune_sample_tasks = getGpuAutoTuneSampleTasks();
    int autotune_applied = 0;
    int autotune_best_blocks = 0;
    float autotune_kernel_ms = 0.0f;
    int slot_count = getGpuPipelineSlots();
    int sm_count = 0;
    int batches = 0;
    int total_reduction_passes = 0;
    int max_reduction_passes = 0;
    int max_blocks_used = 0;
    int success = 0;
    int offset;
    int i;
    int depth_stride;
    int start_offset;
    int last_row;
    int kernel_mode = getGpuKernelMode(board_minus + 1);
    size_t dyn_shared_bytes;

    memset(slots, 0, sizeof(slots));
    memset(h_batch_totals, 0, sizeof(h_batch_totals));

    if (NULL == total_out)
    {
        return 0;
    }

    *total_out = 0;
    if (task_count <= 0)
    {
        return 1;
    }

    err = cudaGetDeviceProperties(&device_prop, 0);
    if (cudaSuccess == err)
    {
        sm_count = device_prop.multiProcessorCount;
    }
    else
    {
        fprintf(stderr, "CUDA warning: failed to query device properties: %s\n", cudaGetErrorString(err));
    }

    start_offset = tasks[0].start_numrows;
    for (i = 1; i < task_count; ++i)
    {
        if (tasks[i].start_numrows < start_offset)
        {
            start_offset = tasks[i].start_numrows;
        }
    }
    if (start_offset < 0)
    {
        start_offset = 0;
    }
    if (start_offset > board_minus)
    {
        start_offset = board_minus;
    }
    last_row = board_minus - start_offset;
    depth_stride = getKernelDepthStride(kernel_mode, last_row + 1);
    dyn_shared_bytes = getKernelSharedBytes(kernel_mode, depth_stride);

    if (dyn_shared_bytes > (size_t)GPU_KERNEL_DEFAULT_SMEM_LIMIT)
    {
        int device_max_optin = GPU_KERNEL_DEFAULT_SMEM_LIMIT;
        cudaError_t attr_err;

        cudaDeviceGetAttribute(&device_max_optin,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               0);

        if (dyn_shared_bytes > (size_t)device_max_optin)
        {
            fprintf(stderr,
                    "CUDA error: required dynamic shared memory %zu bytes exceeds device opt-in cap %d bytes (board_size=%d). Reduce CUDA_BLOCK_SIZE or board size.\n",
                    dyn_shared_bytes,
                    device_max_optin,
                    board_minus + 1);
            return 0;
        }

        attr_err = cudaFuncSetAttribute(solveTasksKernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)dyn_shared_bytes);
        cudaGetLastError();
        if (cudaSuccess != attr_err)
        {
            fprintf(stderr,
                    "CUDA error: cudaFuncSetAttribute(MaxDynamicSharedMemorySize=%zu) failed: %s\n",
                    dyn_shared_bytes,
                    cudaGetErrorString(attr_err));
            return 0;
        }
    }

    batch_capacity = (batch_tasks < task_count) ? batch_tasks : task_count;
    max_launch_blocks = getLaunchBlocksCap(batch_capacity);
    task_buffer_bytes = (size_t)batch_capacity * sizeof(NqueenTask);
    partial_bytes = (size_t)max_launch_blocks * sizeof(SOLUTIONTYPE);

    for (i = 0; i < slot_count; ++i)
    {
        err = cudaStreamCreateWithFlags(&slots[i].stream, cudaStreamNonBlocking);
        if (cudaSuccess != err)
        {
            fprintf(stderr, "CUDA error (cudaStreamCreate): %s\n", cudaGetErrorString(err));
            goto cleanup;
        }

        err = cudaEventCreateWithFlags(&slots[i].done_event, cudaEventDisableTiming);
        if (cudaSuccess != err)
        {
            fprintf(stderr, "CUDA error (cudaEventCreate done_event): %s\n", cudaGetErrorString(err));
            goto cleanup;
        }

        if (print_timing)
        {
            err = cudaEventCreate(&slots[i].h2d_start);
            if (cudaSuccess != err)
            {
                fprintf(stderr, "CUDA error (cudaEventCreate h2d_start): %s\n", cudaGetErrorString(err));
                goto cleanup;
            }
            err = cudaEventCreate(&slots[i].h2d_stop);
            if (cudaSuccess != err)
            {
                fprintf(stderr, "CUDA error (cudaEventCreate h2d_stop): %s\n", cudaGetErrorString(err));
                goto cleanup;
            }
            err = cudaEventCreate(&slots[i].kernel_start);
            if (cudaSuccess != err)
            {
                fprintf(stderr, "CUDA error (cudaEventCreate kernel_start): %s\n", cudaGetErrorString(err));
                goto cleanup;
            }
            err = cudaEventCreate(&slots[i].kernel_stop);
            if (cudaSuccess != err)
            {
                fprintf(stderr, "CUDA error (cudaEventCreate kernel_stop): %s\n", cudaGetErrorString(err));
                goto cleanup;
            }
            err = cudaEventCreate(&slots[i].d2h_start);
            if (cudaSuccess != err)
            {
                fprintf(stderr, "CUDA error (cudaEventCreate d2h_start): %s\n", cudaGetErrorString(err));
                goto cleanup;
            }
            err = cudaEventCreate(&slots[i].d2h_stop);
            if (cudaSuccess != err)
            {
                fprintf(stderr, "CUDA error (cudaEventCreate d2h_stop): %s\n", cudaGetErrorString(err));
                goto cleanup;
            }
        }

        err = cudaMalloc((void**)&slots[i].d_tasks, task_buffer_bytes);
        if (cudaSuccess != err)
        {
            fprintf(stderr, "CUDA error (cudaMalloc d_tasks): %s\n", cudaGetErrorString(err));
            goto cleanup;
        }

        err = cudaMalloc((void**)&slots[i].d_partial_a, partial_bytes);
        if (cudaSuccess != err)
        {
            fprintf(stderr, "CUDA error (cudaMalloc d_partial_a): %s\n", cudaGetErrorString(err));
            goto cleanup;
        }

        if (max_launch_blocks > 1)
        {
            err = cudaMalloc((void**)&slots[i].d_partial_b, partial_bytes);
            if (cudaSuccess != err)
            {
                fprintf(stderr, "CUDA error (cudaMalloc d_partial_b): %s\n", cudaGetErrorString(err));
                goto cleanup;
            }
        }

        err = cudaMalloc((void**)&slots[i].d_next_task, sizeof(unsigned int));
        if (cudaSuccess != err)
        {
            fprintf(stderr, "CUDA error (cudaMalloc d_next_task): %s\n", cudaGetErrorString(err));
            goto cleanup;
        }
    }

    if (0 == persist_blocks && 0 == target_override && autotune_target)
    {
        int sample_tasks = task_count;
        int sample_max_blocks;
        int baseline_blocks;
        int tuned_target = target_tasks_per_block;
        int tuned_blocks = 0;
        float tuned_ms = 0.0f;

        if (sample_tasks > batch_capacity)
        {
            sample_tasks = batch_capacity;
        }
        if (sample_tasks > autotune_sample_tasks)
        {
            sample_tasks = autotune_sample_tasks;
        }
        sample_max_blocks = getLaunchBlocksCap(sample_tasks);
        baseline_blocks = getPersistentLaunchBlocks(batch_capacity, sm_count, 0, target_tasks_per_block);

        if (!shouldRunAutotuneSample(sample_max_blocks, baseline_blocks))
        {
            if (print_timing)
            {
                fprintf(stderr,
                        "CUDA info: skipping autotune (sample_blocks=%d baseline_blocks=%d). Increase NQ_GPU_AUTOTUNE_SAMPLE_TASKS to enable.\n",
                        sample_max_blocks,
                        baseline_blocks);
            }
        }
        else
        {
            err = cudaMemcpy(slots[0].d_tasks,
                             tasks,
                             (size_t)sample_tasks * sizeof(NqueenTask),
                             cudaMemcpyHostToDevice);
            if (cudaSuccess != err)
            {
                fprintf(stderr, "CUDA warning: autotune H2D copy failed: %s\n", cudaGetErrorString(err));
            }
            else if (autotuneTargetTasksPerBlock(kernel_mode,
                                                 start_offset,
                                                 last_row,
                                                 mask,
                                                 depth_stride,
                                                 dyn_shared_bytes,
                                                 slots[0].d_tasks,
                                                 sample_tasks,
                                                 sm_count,
                                                 target_tasks_per_block,
                                                 sample_max_blocks,
                                                 slots[0].d_next_task,
                                                 slots[0].d_partial_a,
                                                 &tuned_target,
                                                 &tuned_blocks,
                                                 &tuned_ms))
            {
                target_tasks_per_block = tuned_target;
                autotune_best_blocks = tuned_blocks;
                autotune_kernel_ms = tuned_ms;
                autotune_applied = 1;
            }
        }
    }

    for (offset = 0; offset < task_count; offset += batch_capacity)
    {
        int slot_index = (offset / batch_capacity) % slot_count;
        GpuPipelineSlot* slot = &slots[slot_index];
        int current_tasks = task_count - offset;
        int launch_blocks;
        int current_count;
        const SOLUTIONTYPE* d_reduce_in;
        SOLUTIONTYPE* d_reduce_out;
        const SOLUTIONTYPE* d_final;
        int reduction_passes = 0;
        size_t copy_bytes;

        if (!finalizePipelineSlot(slot,
                                  slot_index,
                                  h_batch_totals,
                                  print_timing,
                                  &h2d_total_ms,
                                  &kernel_total_ms,
                                  &d2h_total_ms,
                                  &total,
                                  &total_reduction_passes,
                                  &max_reduction_passes))
        {
            goto cleanup;
        }

        if (current_tasks > batch_capacity)
        {
            current_tasks = batch_capacity;
        }
        copy_bytes = (size_t)current_tasks * sizeof(NqueenTask);
        launch_blocks = getPersistentLaunchBlocks(current_tasks, sm_count, persist_blocks, target_tasks_per_block);
        if (launch_blocks < 1)
        {
            launch_blocks = 1;
        }
        if (launch_blocks > max_launch_blocks)
        {
            /* d_partial_* hold max_launch_blocks sums; an NQ_GPU_PERSIST_BLOCKS
               override must not run past them. */
            launch_blocks = max_launch_blocks;
        }
        if (launch_blocks > max_blocks_used)
        {
            max_blocks_used = launch_blocks;
        }
        ++batches;

        if (print_timing)
        {
            cudaEventRecord(slot->h2d_start, slot->stream);
        }

        err = cudaMemcpyAsync(slot->d_tasks,
                              tasks + offset,
                              copy_bytes,
                              cudaMemcpyHostToDevice,
                              slot->stream);
        if (cudaSuccess != err)
        {
            fprintf(stderr, "CUDA error (cudaMemcpy H2D tasks): %s\n", cudaGetErrorString(err));
            goto cleanup;
        }

        if (print_timing)
        {
            cudaEventRecord(slot->h2d_stop, slot->stream);
        }

        if (print_timing)
        {
            cudaEventRecord(slot->kernel_start, slot->stream);
        }

        err = cudaMemsetAsync(slot->d_next_task, 0, sizeof(unsigned int), slot->stream);
        if (cudaSuccess != err)
        {
            fprintf(stderr, "CUDA error (cudaMemset d_next_task): %s\n", cudaGetErrorString(err));
            goto cleanup;
        }

        solveTasksKernel<<<launch_blocks, CUDA_BLOCK_SIZE, dyn_shared_bytes, slot->stream>>>(kernel_mode,
                                                                               start_offset,
                                                                               last_row,
                                                                               mask,
                                                                               depth_stride,
                                                                               slot->d_tasks,
                                                                               current_tasks,
                                                                               slot->d_next_task,
                                                                               slot->d_partial_a);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "CUDA error (kernel launch): %s\n", cudaGetErrorString(err));
            goto cleanup;
        }

        current_count = launch_blocks;
        d_reduce_in = slot->d_partial_a;
        while (current_count > 1)
        {
            int out_count = (current_count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            if (NULL == slot->d_partial_b)
            {
                fprintf(stderr, "CUDA error: reduction buffer not allocated.\n");
                goto cleanup;
            }

            d_reduce_out = (d_reduce_in == slot->d_partial_a) ? slot->d_partial_b : slot->d_partial_a;
            reduceSumsKernel<<<out_count, CUDA_BLOCK_SIZE, 0, slot->stream>>>(d_reduce_in, current_count, d_reduce_out);
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

        if (print_timing)
        {
            cudaEventRecord(slot->kernel_stop, slot->stream);
        }

        if (print_timing)
        {
            cudaEventRecord(slot->d2h_start, slot->stream);
        }

        err = cudaMemcpyAsync(&h_batch_totals[slot_index],
                              d_final,
                              sizeof(SOLUTIONTYPE),
                              cudaMemcpyDeviceToHost,
                              slot->stream);
        if (cudaSuccess != err)
        {
            fprintf(stderr, "CUDA error (cudaMemcpy D2H total): %s\n", cudaGetErrorString(err));
            goto cleanup;
        }

        if (print_timing)
        {
            cudaEventRecord(slot->d2h_stop, slot->stream);
        }

        err = cudaEventRecord(slot->done_event, slot->stream);
        if (cudaSuccess != err)
        {
            fprintf(stderr, "CUDA error (cudaEventRecord done_event): %s\n", cudaGetErrorString(err));
            goto cleanup;
        }

        slot->reduction_passes = reduction_passes;
        slot->in_use = 1;
    }

    for (i = 0; i < slot_count; ++i)
    {
        if (!finalizePipelineSlot(&slots[i],
                                  i,
                                  h_batch_totals,
                                  print_timing,
                                  &h2d_total_ms,
                                  &kernel_total_ms,
                                  &d2h_total_ms,
                                  &total,
                                  &total_reduction_passes,
                                  &max_reduction_passes))
        {
            goto cleanup;
        }
    }

    *total_out = total;

    if (print_timing)
    {
        fprintf(stderr,
                "GPU timing (ms): kernel=%s smem_per_block=%zu H2D=%.3f kernel+reduce=%.3f D2H=%.3f tasks=%d batch_tasks=%d batches=%d pipeline_slots=%d max_blocks=%d target_tasks_per_block=%d persist_override=%d autotune_target=%d autotune_blocks=%d autotune_ms=%.3f total_passes=%d max_passes=%d\n",
                (GPU_KERNEL_PEEL == kernel_mode) ? "peel" : (GPU_KERNEL_PEEL_LOCAL == kernel_mode) ? "peel-local" : "legacy",
                dyn_shared_bytes,
                h2d_total_ms,
                kernel_total_ms,
                d2h_total_ms,
                task_count,
                batch_capacity,
                batches,
                slot_count,
                max_blocks_used,
                target_tasks_per_block,
                persist_blocks,
                autotune_applied,
                autotune_best_blocks,
                autotune_kernel_ms,
                total_reduction_passes,
                max_reduction_passes);
    }

    success = 1;

cleanup:
    for (i = 0; i < MAX_GPU_PIPELINE_SLOTS; ++i)
    {
        if (slots[i].stream) cudaStreamSynchronize(slots[i].stream);
        if (slots[i].d_next_task) cudaFree(slots[i].d_next_task);
        if (slots[i].d_partial_b) cudaFree(slots[i].d_partial_b);
        if (slots[i].d_partial_a) cudaFree(slots[i].d_partial_a);
        if (slots[i].d_tasks) cudaFree(slots[i].d_tasks);
        if (slots[i].d2h_stop) cudaEventDestroy(slots[i].d2h_stop);
        if (slots[i].d2h_start) cudaEventDestroy(slots[i].d2h_start);
        if (slots[i].kernel_stop) cudaEventDestroy(slots[i].kernel_stop);
        if (slots[i].kernel_start) cudaEventDestroy(slots[i].kernel_start);
        if (slots[i].h2d_stop) cudaEventDestroy(slots[i].h2d_stop);
        if (slots[i].h2d_start) cudaEventDestroy(slots[i].h2d_start);
        if (slots[i].done_event) cudaEventDestroy(slots[i].done_event);
        if (slots[i].stream) cudaStreamDestroy(slots[i].stream);
    }
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
