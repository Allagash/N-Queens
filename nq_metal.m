/*
 * nq_metal.m – N-Queens solver using Metal compute shaders.
 *
 * Strategy:
 *   1. CPU enumerates valid queen placements for rows 0..K-1, streaming them
 *      to the GPU in fixed-size chunks.  Memory usage is bounded at ~192 MB
 *      regardless of board size, so N=22 and N=23 are solvable on 32 GB machines.
 *   2. Each partial placement becomes one GPU thread that solves rows K..N-1
 *      using iterative backtracking identical to nq.c.
 *   3. Each thread writes its local count to a results[] buffer.
 *   4. CPU accumulates all chunk totals and multiplies by 2 (mirror symmetry).
 *
 * K selection (pre-enumeration depth):
 *   Keep (N - K) ≈ 13 remaining rows per GPU thread so the worst-case subtree
 *   stays under the ~10 s macOS GPU watchdog.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#define MAX_BOARDSIZE 23

/* Fixed-size chunk buffer reused across GPU dispatches.
   8 M entries × (16 + 8) bytes = 192 MB — fits comfortably in 32 GB. */
#define CHUNK_SIZE  8000000

/* Max partial placements per GPU command buffer (GPU watchdog safety). */
#define BATCH_SIZE  32000

/* Sliding-window depth: command buffers in flight simultaneously. */
#define WINDOW_SIZE 10

typedef struct {
    uint32_t col;
    uint32_t pos_diag;
    uint32_t neg_diag;
    uint32_t init_bitfield;
} PartialPlacement;

/* ---- Globals shared between enumerate() and flush_to_gpu() --------------- */

static PartialPlacement *g_partials     = NULL;
static int               g_num_partials = 0;
static uint64_t          g_total        = 0;

static id<MTLComputePipelineState> g_pipeline;
static id<MTLCommandQueue>         g_queue;
static id<MTLBuffer>               g_partialsBuf;
static id<MTLBuffer>               g_resultsBuf;
static id<MTLBuffer>               g_bsBuf;
static id<MTLBuffer>               g_srBuf;
static NSUInteger                  g_tgSize;

/* ---- Flush one chunk of partial placements to the GPU -------------------- */

static void flush_to_gpu(void)
{
    int num = g_num_partials;
    if (num == 0) return;

    memcpy([g_partialsBuf contents], g_partials,
           (size_t)num * sizeof(PartialPlacement));
    memset([g_resultsBuf contents], 0, (size_t)num * sizeof(uint64_t));

    uint64_t *results   = (uint64_t *)[g_resultsBuf contents];
    int       num_batches = (num + BATCH_SIZE - 1) / BATCH_SIZE;

    id<MTLCommandBuffer> window[WINDOW_SIZE];
    int wStart[WINDOW_SIZE], wCount[WINDOW_SIZE];
    memset(window, 0, sizeof(window));

    int submitted = 0, completed = 0;

#define SUBMIT_ONE(slot, bidx) do { \
        int _bs = (bidx) * BATCH_SIZE; \
        int _bc = num - _bs; if (_bc > BATCH_SIZE) _bc = BATCH_SIZE; \
        NSUInteger _po = (NSUInteger)_bs * sizeof(PartialPlacement); \
        NSUInteger _ro = (NSUInteger)_bs * sizeof(uint64_t); \
        id<MTLCommandBuffer>         _cb  = [g_queue commandBuffer]; \
        id<MTLComputeCommandEncoder> _enc = [_cb computeCommandEncoder]; \
        [_enc setComputePipelineState:g_pipeline]; \
        [_enc setBuffer:g_partialsBuf offset:_po atIndex:0]; \
        [_enc setBuffer:g_resultsBuf  offset:_ro atIndex:1]; \
        [_enc setBuffer:g_bsBuf       offset:0   atIndex:2]; \
        [_enc setBuffer:g_srBuf       offset:0   atIndex:3]; \
        [_enc dispatchThreads:MTLSizeMake((NSUInteger)_bc,1,1) \
             threadsPerThreadgroup:MTLSizeMake(g_tgSize,1,1)]; \
        [_enc endEncoding]; [_cb commit]; \
        window[(slot)] = _cb; wStart[(slot)] = _bs; wCount[(slot)] = _bc; \
    } while(0)

    while (submitted < num_batches && submitted - completed < WINDOW_SIZE) {
        SUBMIT_ONE(submitted % WINDOW_SIZE, submitted); submitted++;
    }
    while (completed < num_batches) {
        int slot = completed % WINDOW_SIZE;
        [window[slot] waitUntilCompleted];
        if (window[slot].status == MTLCommandBufferStatusError) {
            fprintf(stderr, "GPU error: %s\n",
                    [window[slot].error.localizedDescription UTF8String]);
            exit(1);
        }
        int bs = wStart[slot], bc = wCount[slot];
        for (int i = bs; i < bs + bc; i++) g_total += results[i];
        window[slot] = nil;
        completed++;
        if (submitted < num_batches) {
            SUBMIT_ONE(slot, submitted); submitted++;
        }
    }
#undef SUBMIT_ONE

    g_num_partials = 0;
}

/* ---- CPU pre-enumeration ------------------------------------------------- */

static void enumerate(int row, int target_depth, int board_size,
                      uint32_t col, uint32_t pos, uint32_t neg,
                      uint32_t bitfield, uint32_t mask)
{
    if (row == target_depth) {
        g_partials[g_num_partials++] = (PartialPlacement){
            col, pos, neg,
            mask & ~(col | pos | neg)
        };
        if (g_num_partials == CHUNK_SIZE)
            flush_to_gpu();
        return;
    }
    uint32_t bf = bitfield;
    while (bf) {
        uint32_t lsb = (uint32_t)(-(int32_t)bf) & bf;
        bf &= ~lsb;
        uint32_t nc = col | lsb;
        uint32_t np = (pos | lsb) << 1;
        uint32_t nn = (neg | lsb) >> 1;
        enumerate(row + 1, target_depth, board_size,
                  nc, np, nn,
                  mask & ~(nc | np | nn), mask);
    }
}

/* ---- main ---------------------------------------------------------------- */

int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("N Queens (Metal) by Jeff Somers / Metal port.\n");
        printf("Usage: nq_metal <board_size>\n");
        return 0;
    }

    int board_size = atoi(argv[1]);
    if (board_size < 2 || board_size > MAX_BOARDSIZE) {
        printf("Board size must be between 2 and %d.\n", MAX_BOARDSIZE);
        return 0;
    }

    /* Choose pre-enumeration depth K so that (N - K) ≈ 13 remaining rows,
       keeping worst-case GPU thread time under the ~10 s watchdog. */
    int K;
    if      (board_size <= 5)  K = 1;
    else if (board_size <= 9)  K = 2;
    else if (board_size <= 13) K = 3;
    else if (board_size <= 16) K = 4;
    else if (board_size <= 19) K = 6;
    else if (board_size <= 21) K = 8;
    else if (board_size == 22) K = 10;
    else                       K = 11;  /* N=23 */
    if (K >= board_size) K = board_size - 1;

    /* Fixed-size chunk buffer reused across GPU dispatches. */
    g_partials = malloc((size_t)CHUNK_SIZE * sizeof(PartialPlacement));
    if (!g_partials) { fprintf(stderr, "OOM\n"); return 1; }

    uint32_t mask = (1u << board_size) - 1u;
    int      half = board_size / 2;
    int      odd  = board_size & 1;

    time_t t1, t2;
    time(&t1);
    printf("N Queens (Metal) – board size %d\n", board_size);

    /* ---- Metal setup ----------------------------------------------------- */

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { fprintf(stderr, "No Metal device found.\n"); return 1; }
    printf("GPU: %s\n", [[device name] UTF8String]);

    NSString *exe    = [NSString stringWithUTF8String:argv[0]];
    NSString *dir    = [exe stringByDeletingLastPathComponent];
    if ([dir length] == 0) dir = @".";
    NSString *libPath = [dir stringByAppendingPathComponent:@"nq.metallib"];
    NSURL    *libURL  = [NSURL fileURLWithPath:libPath];

    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&error];
    if (!library) {
        fprintf(stderr, "Failed to load nq.metallib: %s\n",
                [[error localizedDescription] UTF8String]);
        return 1;
    }

    id<MTLFunction> kernelFn = [library newFunctionWithName:@"nqueen"];
    if (!kernelFn) { fprintf(stderr, "Kernel 'nqueen' not found.\n"); return 1; }

    g_pipeline = [device newComputePipelineStateWithFunction:kernelFn error:&error];
    if (!g_pipeline) {
        fprintf(stderr, "Pipeline error: %s\n",
                [[error localizedDescription] UTF8String]);
        return 1;
    }

    g_queue = [device newCommandQueue];

    /* Reusable shared buffers sized for one CHUNK_SIZE chunk. */
    g_partialsBuf = [device newBufferWithLength:sizeof(PartialPlacement) * CHUNK_SIZE
                                        options:MTLResourceStorageModeShared];
    g_resultsBuf  = [device newBufferWithLength:sizeof(uint64_t) * CHUNK_SIZE
                                        options:MTLResourceStorageModeShared];

    uint32_t bs = (uint32_t)board_size;
    uint32_t sr = (uint32_t)K;
    g_bsBuf = [device newBufferWithBytes:&bs length:4
                                 options:MTLResourceStorageModeShared];
    g_srBuf = [device newBufferWithBytes:&sr length:4
                                 options:MTLResourceStorageModeShared];

    g_tgSize = g_pipeline.maxTotalThreadsPerThreadgroup;
    if (g_tgSize > 512) g_tgSize = 512;

    printf("Streaming partial placements to GPU (K=%d, chunk=%dK)\n",
           K, CHUNK_SIZE / 1000);

    /* ---- Enumerate + stream to GPU --------------------------------------- */

    /* Pass 1: row-0 queen in columns 0..half-1. */
    uint32_t row0_bf = (1u << half) - 1u;
    enumerate(0, K, board_size, 0, 0, 0, row0_bf, mask);

    /* Pass 2 (odd boards): row-0 queen at the centre column. */
    if (odd) {
        uint32_t center = 1u << (board_size / 2);
        uint32_t c_pos  = center << 1;
        uint32_t c_neg  = center >> 1;
        uint32_t avail1 = mask & ~(center | c_pos | c_neg);
        uint32_t restricted = avail1 & ((1u << half) - 1u);

        if (K == 1) {
            g_partials[g_num_partials++] = (PartialPlacement){
                center, c_pos, c_neg, restricted
            };
            if (g_num_partials == CHUNK_SIZE) flush_to_gpu();
        } else {
            enumerate(1, K, board_size, center, c_pos, c_neg, restricted, mask);
        }
    }

    flush_to_gpu();  /* flush any remaining partial placements */

    /* ---- Collect results ------------------------------------------------- */

    g_total *= 2;  /* mirror symmetry */

    time(&t2);
    double secs    = difftime(t2, t1);
    int    intsecs = (int)secs;
    int    hours   = intsecs / 3600;
    int    mins    = (intsecs % 3600) / 60;
    int    sec2    = intsecs % 60;

    printf("Calculations took %d second%s", intsecs, intsecs == 1 ? "" : "s");
    if (hours > 0 || mins > 0)
        printf(" (%dh %dm %ds)", hours, mins, sec2);
    printf(".\n");

    if (g_total)
        printf("For board size %d, %llu solution%s found.\n",
               board_size, g_total, g_total == 1 ? "" : "s");
    else
        printf("No solutions found.\n");

    return 0;
}
