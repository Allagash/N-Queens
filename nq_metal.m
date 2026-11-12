/*
 * nq_metal.m - Metal GPU backend for nq (macOS / Apple Silicon).
 *
 * Entry point: solveTasksGpuMetal(), the USE_METAL counterpart of
 * solveTasksGpuCuda(). Contract (see nq.c): count every leaf under every task
 * once (vertical-mirror-only convention, NOT doubled - Nqueen() does the x2),
 * store the sum in *total_out and return nonzero; or return 0 having counted
 * nothing, so the caller solves everything on the CPU.
 *
 * Strategy (two-level split):
 *   1. nq.c hands over a shallow frontier (default depth 6). That list is small
 *      (under 10M tasks even at N=23) but its subtrees are too deep for one GPU
 *      thread at N >= 20.
 *   2. This file expands each task on the CPU down to a target depth T and
 *      streams the resulting 16-byte partial placements into a fixed-size
 *      chunk buffer. Memory stays bounded regardless of board size.
 *   3. Each full chunk is flushed to the GPU: one thread per partial placement
 *      solves rows T..N-1 (kernel `nqueen` in nq_metal.metal, 4-row leaf peel)
 *      and writes its count to results[]; the CPU sums them.
 *
 * INVARIANTS - measured on an Apple M1 Max, 2026-03-19/20. Do not relax these
 * without re-measuring:
 *
 *   1. No sliding window on one command queue. Commit a command buffer,
 *      waitUntilCompleted, collect, and only then submit the next. A single
 *      MTLCommandQueue serializes command buffers anyway, and when one fails
 *      (watchdog, internal error) every command buffer queued behind it on the
 *      same queue fails too, with a misleading "Internal Error (0000000e)".
 *      Several dispatches inside ONE command buffer are fine.
 *
 *   2. At most 13 residual rows per GPU thread (9 stack rows + 4 peeled).
 *      14 rows hit the GPU watchdog at N=20, 21 and 22. Hence T = N - 13 for
 *      N >= 20. 65536 threads per dispatch is safe; 131072 crashes.
 *
 *   3. Function constants change codegen. BOARD_SIZE and START_ROW are the
 *      only function constants; a loop budget passed that way crashed at
 *      values that worked as a #define.
 *
 * The limits in (2) can be overridden at run time for other GPUs:
 *   NQ_METAL_MAX_ROWS            residual rows per thread     (default 13)
 *   NQ_METAL_BATCH               threads per dispatch         (default 65536)
 *   NQ_METAL_DISPATCHES_PER_CB   dispatches per command buf   (default 16)
 *   NQ_METAL_CHUNK               partials buffered per flush  (default 8000000)
 *   NQ_METALLIB                  path to nq_metal.metallib    (default: next
 *                                to the executable)
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach-o/dyld.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nq_types.h"

/* Largest board the 32-bit kernel is validated for here (bit masks would work to 31). */
#define METAL_MAX_BOARD          27
/* The kernel's stack is stk[23 + 2]; it pushes at most (residual rows - 4)
   frames starting at index 1, so 23 residual rows is the hard ceiling. */
#define METAL_MAX_ROWS_CEILING   23
#define DEFAULT_MAX_ROWS         13
#define DEFAULT_BATCH_SIZE       65536
#define DEFAULT_DISPATCHES_PER_CB 16
/* 8 M entries x (16 + 8) bytes = 192 MB of shared buffers. */
#define DEFAULT_CHUNK_SIZE       8000000
/* A repeated GPU failure at or above this board size aborts instead of
   silently restarting on the CPU (which would take hours to days). */
#define FATAL_FAILURE_MIN_BOARD  20

/* Must match struct PartialPlacement in nq_metal.metal. */
typedef struct {
    uint32_t col;
    uint32_t pos_diag;
    uint32_t neg_diag;
    uint32_t init_bitfield;
} PartialPlacement;

_Static_assert(sizeof(PartialPlacement) == 16, "PartialPlacement size mismatch");

typedef struct {
    id<MTLDevice>               device;
    id<MTLComputePipelineState> pipeline;
    id<MTLCommandQueue>         queue;
    id<MTLBuffer>               partialsBuf;
    id<MTLBuffer>               resultsBuf;
    NSUInteger                  tgSize;

    PartialPlacement*           partials;      /* = [partialsBuf contents] */
    int64_t                     num_partials;  /* filled entries in the current chunk */

    int64_t                     chunk_size;
    int64_t                     batch_size;
    int                         dispatches_per_cb;

    int                         board_size;
    int                         target_depth;
    uint32_t                    mask;

    uint64_t                    total;         /* leaves counted so far */
    int                         failed;        /* set on unrecoverable GPU error */

    /* progress */
    int                         task_index;    /* shallow task being expanded */
    int                         task_count;
    int64_t                     partials_done;
    time_t                      start_time;
    time_t                      last_progress;
} MetalCtx;

/* ---- Small helpers ------------------------------------------------------- */

static int64_t envInt(const char* name, int64_t def, int64_t lo, int64_t hi)
{
    const char* value = getenv(name);
    if (value != NULL && *value != '\0') {
        char* endp = NULL;
        long long parsed = strtoll(value, &endp, 10);
        if (endp != value && parsed >= lo) {
            return parsed > hi ? hi : (int64_t)parsed;
        }
    }
    return def;
}

static NSString* metallibPath(void)
{
    const char* env = getenv("NQ_METALLIB");
    if (env != NULL && *env != '\0') {
        return [NSString stringWithUTF8String:env];
    }

    char     buf[4096];
    uint32_t size = (uint32_t)sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) != 0) {
        return @"nq_metal.metallib";
    }
    NSString* exe = [[NSString stringWithUTF8String:buf] stringByResolvingSymlinksInPath];
    return [[exe stringByDeletingLastPathComponent]
               stringByAppendingPathComponent:@"nq_metal.metallib"];
}

static void printProgress(MetalCtx* c)
{
    time_t now;
    time(&now);
    if (difftime(now, c->last_progress) < 60.0) return;
    c->last_progress = now;

    int secs = (int)difftime(now, c->start_time);
    double pct = c->task_count > 0 ? 100.0 * (double)c->task_index / (double)c->task_count : 0.0;

    fprintf(stderr, "[%dh%02dm%02ds] %.1f%% of %d frontier tasks, %lld GPU threads run, %llu leaves counted\n",
            secs / 3600, (secs % 3600) / 60, secs % 60, pct, c->task_count,
            (long long)c->partials_done, (unsigned long long)c->total);
}

/* ---- GPU dispatch -------------------------------------------------------- */

/*
 * Run batches [first_batch, first_batch + count) of the current chunk in ONE
 * command buffer and wait for it. Returns 1 on success; on failure returns 0
 * and copies the Metal error text into err.
 */
static int runCommandBuffer(MetalCtx* c, int64_t first_batch, int count,
                            char* err, size_t err_size)
{
    int ok = 1;
    @autoreleasepool {
        id<MTLCommandBuffer>         cb  = [c->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:c->pipeline];
        for (int64_t b = first_batch; b < first_batch + count; b++) {
            int64_t start = b * c->batch_size;
            int64_t n     = c->num_partials - start;
            if (n > c->batch_size) n = c->batch_size;

            [enc setBuffer:c->partialsBuf
                    offset:(NSUInteger)start * sizeof(PartialPlacement) atIndex:0];
            [enc setBuffer:c->resultsBuf
                    offset:(NSUInteger)start * sizeof(uint64_t) atIndex:1];
            [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(c->tgSize, 1, 1)];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];   /* invariant 1: never queue a second CB behind this one */

        if (cb.status != MTLCommandBufferStatusCompleted) {
            const char* text = [cb.error.localizedDescription UTF8String];
            snprintf(err, err_size, "%s", text != NULL ? text : "unknown Metal error");
            ok = 0;
        }
    }
    return ok;
}

/*
 * Solve the partials buffered in the current chunk and add their counts to
 * c->total. On an unrecoverable GPU error: exits for N >= 20, otherwise sets
 * c->failed. c->total is only ever advanced by fully completed work.
 */
static void flushToGpu(MetalCtx* c)
{
    int64_t num = c->num_partials;
    if (num == 0 || c->failed) return;

    uint64_t* results     = (uint64_t*)[c->resultsBuf contents];
    int64_t   num_batches = (num + c->batch_size - 1) / c->batch_size;
    char      err[256];

    memset(results, 0, (size_t)num * sizeof(uint64_t));

    for (int64_t bidx = 0; bidx < num_batches; ) {
        int64_t remaining = num_batches - bidx;
        int     count     = remaining > c->dispatches_per_cb ? c->dispatches_per_cb : (int)remaining;
        int64_t first     = bidx * c->batch_size;
        int64_t last      = (bidx + count) * c->batch_size;
        if (last > num) last = num;

        if (!runCommandBuffer(c, bidx, count, err, sizeof(err))) {
            /* Retry once: fresh queue (the old one may be poisoned), one
               dispatch per command buffer. */
            fprintf(stderr, "Metal: command buffer failed (%s); retrying on a fresh queue.\n", err);
            c->queue = [c->device newCommandQueue];
            memset(results + first, 0, (size_t)(last - first) * sizeof(uint64_t));

            for (int64_t b = bidx; b < bidx + count; b++) {
                if (runCommandBuffer(c, b, 1, err, sizeof(err))) continue;

                fprintf(stderr,
                        "Metal: GPU error persists after retry: %s\n"
                        "  board size %d, frontier task %d of %d, %llu leaves counted before the failure\n",
                        err, c->board_size, c->task_index, c->task_count,
                        (unsigned long long)c->total);
                if (c->board_size >= FATAL_FAILURE_MIN_BOARD) {
                    fprintf(stderr, "  Aborting: no CPU fallback for board sizes >= %d.\n",
                            FATAL_FAILURE_MIN_BOARD);
                    exit(1);
                }
                fprintf(stderr, "  Discarding GPU work and falling back to the CPU solver.\n");
                c->failed = 1;
                return;
            }
        }

        for (int64_t i = first; i < last; i++) c->total += results[i];
        c->partials_done += last - first;
        bidx += count;
        printProgress(c);
    }

    c->num_partials = 0;
}

/* ---- CPU expansion of a shallow task to the target depth ----------------- */

static inline void emitPartial(MetalCtx* c, uint32_t col, uint32_t pos, uint32_t neg,
                               uint32_t bitfield)
{
    if (bitfield == 0u) return;   /* dead end: no GPU thread needed */
    c->partials[c->num_partials++] = (PartialPlacement){ col, pos, neg, bitfield };
    if (c->num_partials == c->chunk_size) flushToGpu(c);
}

/* `row` rows are placed; `bitfield` holds the free squares of row `row`. */
static void expand(MetalCtx* c, int row, uint32_t col, uint32_t pos, uint32_t neg,
                   uint32_t bitfield)
{
    if (row == c->target_depth) {
        emitPartial(c, col, pos, neg, bitfield);
        return;
    }
    while (bitfield != 0u && !c->failed) {
        uint32_t lsb = bitfield & (~bitfield + 1u);
        bitfield &= ~lsb;
        uint32_t nc = col | lsb;
        uint32_t np = (pos | lsb) << 1;
        uint32_t nn = (neg | lsb) >> 1;
        expand(c, row + 1, nc, np, nn, c->mask & ~(nc | np | nn));
    }
}

/* ---- Setup --------------------------------------------------------------- */

static int setupMetal(MetalCtx* c, int64_t chunk_capacity)
{
    NSError* error = nil;

    c->device = MTLCreateSystemDefaultDevice();
    if (c->device == nil) {
        fprintf(stderr, "Metal: no device found; using CPU solver.\n");
        return 0;
    }

    NSString*      libPath = metallibPath();
    id<MTLLibrary> library = [c->device newLibraryWithURL:[NSURL fileURLWithPath:libPath]
                                                    error:&error];
    if (library == nil) {
        fprintf(stderr, "Metal: cannot load %s (%s); using CPU solver.\n"
                        "  Build it with `make metal`, keep it next to the binary, or set NQ_METALLIB.\n",
                [libPath UTF8String], [[error localizedDescription] UTF8String]);
        return 0;
    }

    /* Specialize the kernel for this board size and start row (invariant 3). */
    MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
    uint32_t cv_board = (uint32_t)c->board_size;
    uint32_t cv_start = (uint32_t)c->target_depth;
    [cv setConstantValue:&cv_board type:MTLDataTypeUInt atIndex:0];
    [cv setConstantValue:&cv_start type:MTLDataTypeUInt atIndex:1];

    id<MTLFunction> kernel = [library newFunctionWithName:@"nqueen" constantValues:cv error:&error];
    if (kernel == nil) {
        fprintf(stderr, "Metal: kernel 'nqueen' not found (%s); using CPU solver.\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }

    c->pipeline = [c->device newComputePipelineStateWithFunction:kernel error:&error];
    if (c->pipeline == nil) {
        fprintf(stderr, "Metal: pipeline error (%s); using CPU solver.\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }

    c->queue       = [c->device newCommandQueue];
    c->partialsBuf = [c->device newBufferWithLength:(NSUInteger)chunk_capacity * sizeof(PartialPlacement)
                                            options:MTLResourceStorageModeShared];
    c->resultsBuf  = [c->device newBufferWithLength:(NSUInteger)chunk_capacity * sizeof(uint64_t)
                                            options:MTLResourceStorageModeShared];
    if (c->queue == nil || c->partialsBuf == nil || c->resultsBuf == nil) {
        fprintf(stderr, "Metal: failed to allocate queue/buffers; using CPU solver.\n");
        return 0;
    }

    c->partials   = (PartialPlacement*)[c->partialsBuf contents];
    c->chunk_size = chunk_capacity;
    c->tgSize     = c->pipeline.maxTotalThreadsPerThreadgroup;
    if (c->tgSize > 512) c->tgSize = 512;
    return 1;
}

/* ---- Entry point --------------------------------------------------------- */

int solveTasksGpuMetal(int board_minus,
                       BITFIELDTYPE mask,
                       const NqueenTask* tasks,
                       int task_count,
                       SOLUTIONTYPE* total_out)
{
    int board_size = board_minus + 1;
    int handled    = 0;

    if (board_size < 2 || board_size > METAL_MAX_BOARD || task_count < 0) {
        return 0;
    }
    if (task_count == 0) {
        *total_out = 0;
        return 1;
    }

    @autoreleasepool {
        MetalCtx ctx = {0};   /* all object fields start nil */
        MetalCtx* c = &ctx;

        /* Target depth: deep enough that no thread exceeds the watchdog cap,
           and no shallower than the deepest task handed to us. */
        int max_rows  = (int)envInt("NQ_METAL_MAX_ROWS", DEFAULT_MAX_ROWS, 1, METAL_MAX_ROWS_CEILING);
        int max_start = 0;
        for (int i = 0; i < task_count; i++) {
            if (tasks[i].start_numrows > max_start) max_start = tasks[i].start_numrows;
        }
        int target = board_size - max_rows;
        if (target < max_start) target = max_start;
        if (max_start < 1 || target > board_minus) {
            return 0;   /* malformed frontier; let the CPU handle it */
        }

        c->board_size        = board_size;
        c->target_depth      = target;
        c->mask              = (uint32_t)mask;
        c->batch_size        = envInt("NQ_METAL_BATCH", DEFAULT_BATCH_SIZE, 1, 1 << 24);
        c->dispatches_per_cb = (int)envInt("NQ_METAL_DISPATCHES_PER_CB", DEFAULT_DISPATCHES_PER_CB, 1, 4096);
        c->task_count        = task_count;

        /* When nothing needs expanding, the chunk never holds more than task_count entries. */
        int64_t chunk = envInt("NQ_METAL_CHUNK", DEFAULT_CHUNK_SIZE, 1, 1LL << 28);
        if (target == max_start && chunk > task_count) {
            int needs_expansion = 0;
            for (int i = 0; i < task_count; i++) {
                if (tasks[i].start_numrows != target) { needs_expansion = 1; break; }
            }
            if (!needs_expansion) chunk = task_count;
        }

        if (!setupMetal(c, chunk)) {
            return 0;
        }

        time(&c->start_time);
        c->last_progress = c->start_time;

        /*
         * Narrowing the 64-bit task fields to 32 bits is exact: col, neg_diag
         * and bitfield never exceed the board mask, and the pos_diag bits lost
         * (>= 32) have already shifted past the board and can never return.
         */
        for (int i = 0; i < task_count && !c->failed; i++) {
            const NqueenTask* t = &tasks[i];
            c->task_index = i;
            expand(c, t->start_numrows,
                   (uint32_t)t->col, (uint32_t)t->pos_diag, (uint32_t)t->neg_diag,
                   (uint32_t)t->bitfield);
        }
        c->task_index = task_count;
        flushToGpu(c);

        if (!c->failed) {
            *total_out = (SOLUTIONTYPE)c->total;   /* raw leaf sum; Nqueen() doubles it */
            handled = 1;
        }
    }

    return handled;
}
