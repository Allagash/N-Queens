/*  Jeff Somers
 *
 *  Copyright (c) 2002
 *
 *  jsomers@alumni.williams.edu
 *  or
 *  allagash98@yahoo.com
 *
 *  April, 2002
 *  
 *  Program:  nq
 *  
 *  Program to find number of solutions to the N queens problem.
 *  This program assumes a twos complement architecture.
 *
 *  For example, you can arrange 4 queens on 4 x 4 chess so that
 *  none of the queens can attack each other:
 *
 *  Two solutions:
 *     _ Q _ _        _ _ Q _
 *     _ _ _ Q        Q _ _ _
 *     Q _ _ _        _ _ _ Q
 *     _ _ Q _    and _ Q _ _
 *    
 *  Note that these are separate solutions, even though they
 *  are mirror images of each other.
 *
 *  Likewise, a 8 x 8 chess board has 92 solutions to the 8 queens
 *  problem.
 *
 *  Command Line Usage:
 *
 *          nq N
 *
 *       where N is the size of the N x N board.  For example,
 *       nq 4 will find the 4 queen solution for the 4 x 4 chess
 *       board.
 *
 *  By default, this program will only print the number of solutions,
 *  not board arrangements which are the solutions.  To print the 
 *  boards, uncomment the call to printtable in the Nqueen function.
 *  Note that printing the board arrangements slows down the program
 *  quite a bit, unless you pipe the output to a text file:
 *
 *  nq 10 > output.txt
 *
 *
 *  The number of solutions for the N queens problems are known for
 *  boards up to 23 x 23.  With this program, I've calculated the
 *  results for boards up to 21 x 21, and that took over a week on
 *  an 800 MHz PC.  The algorithm is approximated O(n!) (i.e. slow), 
 *  and calculating the results for a 22 x 22 board will take about 8.5 
 *  times the amount of time for the 21 x 21 board, or over 8 1/2 weeks.
 *  Even with a 10 GHz machine, calculating the results for a 23 x 23 
 *  board would take over a month.  Of course, setting up a cluster of
 *  machines (or a distributed client) would do the work in less time.
 *  
 *  (from Sloane's On-Line Encyclopedia of Integer Sequences,
 *   Sequence A000170
 *   http://www.research.att.com/cgi-bin/access.cgi/as/njas/sequences/eisA.cgi?Anum=000170
 *   )
 *
 *   Board Size:       Number of Solutions to          Time to calculate 
 *   (length of one        N queens problem:              on 800MHz PC
 *    side of N x N                                    (Hours:Mins:Secs)
 *    chessboard)
 *
 *     1                                  1                    n/a
 *     2                                  0                   < 0 seconds
 *     3                                  0                   < 0 seconds
 *     4                                  2                   < 0 seconds
 *     5                                 10                   < 0 seconds
 *     6                                  4                   < 0 seconds
 *     7                                 40                   < 0 seconds
 *     8                                 92                   < 0 seconds
 *     9                                352                   < 0 seconds 
 *    10                                724                   < 0 seconds
 *    11                               2680                   < 0 seconds
 *    12                              14200                   < 0 seconds
 *    13                              73712                   < 0 seconds
 *    14                             365596                  00:00:01
 *    15                            2279184                  00:00:04
 *    16                           14772512                  00:00:23
 *    17                           95815104                  00:02:38
 *    18                          666090624                  00:19:26
 *    19                         4968057848                  02:31:24
 *    20                        39029188884                  20:35:06
 *    21                       314666222712                 174:53:45
 *    22                      2691008701644                     ?
 *    23                     24233937684440                     ?
 *    24                                  ?                     ?
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef WIN32
#include <pthread.h>
#include <unistd.h>
#endif

#include "nq_types.h"

#define MIN_BOARDSIZE 2
#define DEFAULT_FRONTIER_DEPTH 6
#define TASK_BUCKET_POP_RANGE (MAX_BOARDSIZE + 1)
#define TASK_BUCKET_DEPTH_RANGE MAX_BOARDSIZE
#define TASK_BUCKET_COUNT (TASK_BUCKET_DEPTH_RANGE * TASK_BUCKET_POP_RANGE)

SOLUTIONTYPE g_numsolutions = 0; 


/* Print a chess table with queens positioned for a solution */
/* This is not a critical path function & I didn't try to optimize it. */
void printtable(int boardsize, BITFIELDTYPE* aQueenBitRes, SOLUTIONTYPE numSolution)
{
    int i, j, k;
    BITFIELDTYPE row;

    /*  We only calculated half the solutions, because we can derive
        the other half by reflecting the solution across the "Y axis". */
    for (k = 0; k < 2; ++k)
    {
            printf("*** Solution #: " SOLUTION_PRINTF " ***\n", 2 * numSolution + k - 1);
        for ( i = 0; i < boardsize; i++)
        {
            BITFIELDTYPE bitf;
            /*
               Get the column that was set (i.e. find the
               first, least significant, bit set).
               If aQueenBitRes[i] = 011010b, then
               bitf = 000010b
            */
            bitf = aQueenBitRes[i];

            row = bitf ^ (bitf & (bitf - 1)); /* get least significant bit */
            for ( j = 0; j < boardsize; j++)
            {
                /* keep shifting row over to the right until we find the one '1' in
                   the binary representation.  There will only be one '1'. */
                if (0 == k && ((row >> j) & 1))
                {
                    printf("Q ");
                }
                else if (1 == k && (row & (((BITFIELDTYPE)1) << (boardsize - j - 1)))) /* this is the board reflected across the "Y axis" */
                {
                    printf("Q ");
                }
                else
                {
                    printf(". ");
                }
            }
            printf("\n");
        }
        printf("\n");
    }
}

typedef struct
{
    NqueenTask* tasks;
    int count;
    int capacity;
} FrontierTaskList;

#ifndef WIN32
typedef struct
{
    const NqueenTask* tasks;
    int task_count;
    int thread_id;
    int thread_count;
    int board_minus;
    BITFIELDTYPE mask;
    SOLUTIONTYPE subtotal;
    int started;
} WorkerContext;
#endif

#ifdef USE_CUDA
int solveTasksGpuCuda(int board_minus,
                      BITFIELDTYPE mask,
                      const NqueenTask* tasks,
                      int task_count,
                      SOLUTIONTYPE* total_out);
#endif

static int bitFieldIndex(BITFIELDTYPE bit)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(bit);
#else
    int idx = 0;
    while (0 == (bit & (BITFIELDTYPE)1))
    {
        bit >>= 1;
        ++idx;
    }
    return idx;
#endif
}

static void initTaskList(FrontierTaskList* list)
{
    list->tasks = NULL;
    list->count = 0;
    list->capacity = 0;
}

static void freeTaskList(FrontierTaskList* list)
{
    free(list->tasks);
    list->tasks = NULL;
    list->count = 0;
    list->capacity = 0;
}

static int appendTask(FrontierTaskList* list, const NqueenTask* task)
{
    if (list->count == list->capacity)
    {
        int new_capacity = (0 == list->capacity) ? 1024 : list->capacity * 2;
        NqueenTask* new_tasks = (NqueenTask*)realloc(list->tasks, (size_t)new_capacity * sizeof(NqueenTask));
        if (NULL == new_tasks)
        {
            return 0;
        }
        list->tasks = new_tasks;
        list->capacity = new_capacity;
    }

    list->tasks[list->count++] = *task;
    return 1;
}

/*
 * Compute orbit metrics under the dihedral group D4 for a complete column-vector
 * solution `positions` on an N x N board.
 *
 *   *s_out:  orbit size (number of distinct D4 transforms)
 *   *q_out:  number of distinct orbit elements that lie in the pruned search
 *            domain, i.e., col_0 in [0, N/2) AND row_of_col0 >= col_0
 *
 * The eight D4 transforms of a column-vector sigma (sigma[r] = column of queen
 * in row r) are: e, m_v, m_h, r^2, d_main, d_anti, r, r^3, expressed via the
 * inverse permutation `inv` (inv[c] = row of queen in column c).
 */
static void computeOrbitMetricsPruned(const unsigned char* positions,
                                      int N,
                                      int* s_out,
                                      int* q_out)
{
    unsigned char transforms[8][MAX_BOARDSIZE];
    unsigned char inv[MAX_BOARDSIZE];
    int distinct_idx[8];
    int distinct_count = 0;
    int half = N >> 1;
    int q = 0;
    int i;
    int t;
    int j;

    for (i = 0; i < N; ++i)
    {
        inv[positions[i]] = (unsigned char)i;
    }

    for (i = 0; i < N; ++i)
    {
        unsigned char p = positions[i];
        unsigned char p_rev = positions[N - 1 - i];
        unsigned char inv_i = inv[i];
        unsigned char inv_rev = inv[N - 1 - i];
        transforms[0][i] = p;                                /* e        */
        transforms[1][i] = (unsigned char)(N - 1 - p);       /* m_v      */
        transforms[2][i] = p_rev;                            /* m_h      */
        transforms[3][i] = (unsigned char)(N - 1 - p_rev);   /* r^2      */
        transforms[4][i] = inv_i;                            /* d_main   */
        transforms[5][i] = (unsigned char)(N - 1 - inv_rev); /* d_anti   */
        transforms[6][i] = (unsigned char)(N - 1 - inv_i);   /* r        */
        transforms[7][i] = inv_rev;                          /* r^3      */
    }

    for (t = 0; t < 8; ++t)
    {
        int is_dup = 0;
        for (j = 0; j < distinct_count; ++j)
        {
            if (0 == memcmp(transforms[t], transforms[distinct_idx[j]], (size_t)N))
            {
                is_dup = 1;
                break;
            }
        }
        if (!is_dup)
        {
            distinct_idx[distinct_count++] = t;
        }
    }

    for (j = 0; j < distinct_count; ++j)
    {
        const unsigned char* p = transforms[distinct_idx[j]];
        int col_0 = p[0];
        int row_of_col0 = -1;
        int in_search_domain = 0;

        if (col_0 < half)
        {
            in_search_domain = 1;
        }
        else if ((N & 1) && col_0 == half && p[1] < half)
        {
            /* Odd-N center-column case: m_v breaks the col_0 = half tie via col_1 < half. */
            in_search_domain = 1;
        }
        if (!in_search_domain)
        {
            continue;
        }

        for (i = 0; i < N; ++i)
        {
            if (0 == p[i])
            {
                row_of_col0 = i;
                break;
            }
        }
        if (row_of_col0 >= col_0)
        {
            ++q;
        }
    }

    *s_out = distinct_count;
    *q_out = q;
}

static SOLUTIONTYPE solveTask(int board_minus, BITFIELDTYPE mask, const NqueenTask* task)
{
    BITFIELDTYPE aQueenBitCol[MAX_BOARDSIZE];
    BITFIELDTYPE aQueenBitPosDiag[MAX_BOARDSIZE];
    BITFIELDTYPE aQueenBitNegDiag[MAX_BOARDSIZE];
    BITFIELDTYPE aStack[MAX_BOARDSIZE + 1];
    unsigned char positions[MAX_BOARDSIZE];
    BITFIELDTYPE bitfield;
    BITFIELDTYPE lsb;
    int numrows;
    int stack_count;
    int N = board_minus + 1;
    int col_0_idx = task->col_0_idx;
    SOLUTIONTYPE weighted_total = 0;

    numrows = task->start_numrows;
    stack_count = 0;
    bitfield = task->bitfield;
    aQueenBitCol[numrows] = task->col;
    aQueenBitPosDiag[numrows] = task->pos_diag;
    aQueenBitNegDiag[numrows] = task->neg_diag;
    memcpy(positions, task->seed_positions, sizeof(positions));

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

        /* get least-significant set bit */
        lsb = bitfield & (~bitfield + 1);
        bitfield &= ~lsb;

        /* d_main symmetry-breaking prune: queen in column 0 must be at row >= col_0_idx. */
        if (1 == lsb && numrows < col_0_idx)
        {
            continue;
        }

        if (numrows < board_minus)
        {
            int n = numrows++;
            aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
            aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1;
            aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1;
            aStack[stack_count++] = bitfield;
            positions[n] = (unsigned char)bitFieldIndex(lsb);
            bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
            continue;
        }

        /* leaf reached: queen at row board_minus has been chosen; compute orbit weight */
        positions[numrows] = (unsigned char)bitFieldIndex(lsb);
        {
            int leaf_row_of_col0 = -1;
            int i;
            for (i = 0; i < N; ++i)
            {
                if (0 == positions[i])
                {
                    leaf_row_of_col0 = i;
                    break;
                }
            }
            if (leaf_row_of_col0 >= col_0_idx)
            {
                /* This leaf is in the pruned domain; weight by orbit metrics. */
                int s;
                int q;
                computeOrbitMetricsPruned(positions, N, &s, &q);
                if (q > 0)
                {
                    weighted_total += (SOLUTIONTYPE)((12 * s) / q);
                }
            }
        }
        if (0 == stack_count)
        {
            break;
        }
        bitfield = aStack[--stack_count];
        --numrows;
    }

    return weighted_total;
}

#ifdef USE_CUDA
static int popcountBitfield(BITFIELDTYPE value)
{
#if defined(__GNUC__) || defined(__clang__)
    return (int)__builtin_popcountll((unsigned long long)value);
#else
    int count = 0;
    while (value != 0)
    {
        value &= (value - 1);
        ++count;
    }
    return count;
#endif
}

static int getTaskBucketIndex(int board_minus, const NqueenTask* task)
{
    int remaining_rows = board_minus - task->start_numrows;
    int candidate_count = popcountBitfield(task->bitfield);

    if (remaining_rows < 0)
    {
        remaining_rows = 0;
    }
    if (remaining_rows >= TASK_BUCKET_DEPTH_RANGE)
    {
        remaining_rows = TASK_BUCKET_DEPTH_RANGE - 1;
    }

    if (candidate_count < 0)
    {
        candidate_count = 0;
    }
    if (candidate_count > MAX_BOARDSIZE)
    {
        candidate_count = MAX_BOARDSIZE;
    }

    return remaining_rows * TASK_BUCKET_POP_RANGE + candidate_count;
}

static int bucketFrontierTasks(FrontierTaskList* list, int board_minus)
{
    int bucket_counts[TASK_BUCKET_COUNT];
    int bucket_offsets[TASK_BUCKET_COUNT];
    int bucket_cursor[TASK_BUCKET_COUNT];
    NqueenTask* reordered;
    int i;
    int offset = 0;

    if (list->count <= 1)
    {
        return 1;
    }

    memset(bucket_counts, 0, sizeof(bucket_counts));

    for (i = 0; i < list->count; ++i)
    {
        int bucket = getTaskBucketIndex(board_minus, &list->tasks[i]);
        ++bucket_counts[bucket];
    }

    /* Sort into descending hardness buckets to reduce long-tail imbalance. */
    for (i = TASK_BUCKET_COUNT - 1; i >= 0; --i)
    {
        bucket_offsets[i] = offset;
        offset += bucket_counts[i];
    }
    memcpy(bucket_cursor, bucket_offsets, sizeof(bucket_offsets));

    reordered = (NqueenTask*)malloc((size_t)list->count * sizeof(NqueenTask));
    if (NULL == reordered)
    {
        return 0;
    }

    for (i = 0; i < list->count; ++i)
    {
        int bucket = getTaskBucketIndex(board_minus, &list->tasks[i]);
        int position = bucket_cursor[bucket]++;
        reordered[position] = list->tasks[i];
    }

    memcpy(list->tasks, reordered, (size_t)list->count * sizeof(NqueenTask));
    free(reordered);
    return 1;
}
#endif

static int getFrontierDepth(int board_size)
{
    const char* env = getenv("NQ_FRONTIER_DEPTH");
    int depth = DEFAULT_FRONTIER_DEPTH;

    if (env != NULL && *env != '\0')
    {
        char* endp = NULL;
        long parsed = strtol(env, &endp, 10);
        if (endp != env && parsed > 0)
        {
            depth = (int)parsed;
        }
    }

    if (depth < 1)
    {
        depth = 1;
    }
    if (depth > board_size - 1)
    {
        depth = board_size - 1;
    }

    return depth;
}

static int generateFrontierFromSeed(int board_minus,
                                    BITFIELDTYPE mask,
                                    int frontier_depth,
                                    const NqueenTask* seed,
                                    FrontierTaskList* out)
{
    BITFIELDTYPE aQueenBitCol[MAX_BOARDSIZE];
    BITFIELDTYPE aQueenBitPosDiag[MAX_BOARDSIZE];
    BITFIELDTYPE aQueenBitNegDiag[MAX_BOARDSIZE];
    BITFIELDTYPE aStack[MAX_BOARDSIZE + 1];
    unsigned char positions[MAX_BOARDSIZE];
    BITFIELDTYPE bitfield;
    BITFIELDTYPE lsb;
    int numrows;
    int stack_count;

    numrows = seed->start_numrows;
    stack_count = 0;
    bitfield = seed->bitfield;
    aQueenBitCol[numrows] = seed->col;
    aQueenBitPosDiag[numrows] = seed->pos_diag;
    aQueenBitNegDiag[numrows] = seed->neg_diag;
    memcpy(positions, seed->seed_positions, sizeof(positions));

    for (;;)
    {
        if (numrows >= frontier_depth)
        {
            NqueenTask task;
            task.start_numrows = numrows;
            task.col_0_idx = seed->col_0_idx;
            task.col = aQueenBitCol[numrows];
            task.pos_diag = aQueenBitPosDiag[numrows];
            task.neg_diag = aQueenBitNegDiag[numrows];
            task.bitfield = bitfield;
            memcpy(task.seed_positions, positions, sizeof(task.seed_positions));

            if (!appendTask(out, &task))
            {
                return 0;
            }

            if (0 == stack_count)
            {
                break;
            }
            bitfield = aStack[--stack_count];
            --numrows;
            continue;
        }

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

        /*
         * NOTE: the d_main symmetry prune is applied at solveTask leaf time, not here,
         * to keep the emitted frontier compatible with the legacy GPU kernel which
         * still uses the vertical-mirror-only convention. Once the GPU is migrated
         * to the orbit-weighted convention, the prune can move up to here for an
         * additional speedup at the frontier-generation step.
         */

        if (numrows < board_minus)
        {
            int n = numrows++;
            aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
            aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1;
            aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1;
            aStack[stack_count++] = bitfield;
            positions[n] = (unsigned char)bitFieldIndex(lsb);
            bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
            continue;
        }

        /* Leaf solution reached before frontier depth clamp. */
        if (0 == stack_count)
        {
            break;
        }
        bitfield = aStack[--stack_count];
        --numrows;
    }

    return 1;
}

static int generateFrontier(int board_size,
                            int board_minus,
                            BITFIELDTYPE mask,
                            int frontier_depth,
                            FrontierTaskList* out)
{
    int odd = board_size & 1;
    int half = board_size >> 1;
    BITFIELDTYPE bitfield;
    BITFIELDTYPE lsb;

    bitfield = (((BITFIELDTYPE)1) << half) - 1;
    while (bitfield != 0)
    {
        NqueenTask seed;

        lsb = bitfield & (~bitfield + 1);
        bitfield &= ~lsb;

        seed.start_numrows = 1;
        seed.col_0_idx = bitFieldIndex(lsb);
        seed.col = lsb;
        seed.neg_diag = (lsb >> 1);
        seed.pos_diag = (lsb << 1);
        seed.bitfield = mask & ~(seed.col | seed.neg_diag | seed.pos_diag);
        memset(seed.seed_positions, 0, sizeof(seed.seed_positions));
        seed.seed_positions[0] = (unsigned char)seed.col_0_idx;

        if (!generateFrontierFromSeed(board_minus, mask, frontier_depth, &seed, out))
        {
            return 0;
        }
    }

    if (odd)
    {
        BITFIELDTYPE middle = ((BITFIELDTYPE)1) << half;
        bitfield = (middle - 1) >> 1;

        while (bitfield != 0)
        {
            BITFIELDTYPE col;
            BITFIELDTYPE neg_diag;
            BITFIELDTYPE pos_diag;
            NqueenTask seed;

            lsb = bitfield & (~bitfield + 1);
            bitfield &= ~lsb;

            col = middle | lsb;
            neg_diag = ((middle >> 1) | lsb) >> 1;
            pos_diag = ((middle << 1) | lsb) << 1;

            seed.start_numrows = 2;
            seed.col_0_idx = half;  /* row 0's queen sits on the center column for the odd-N branch. */
            seed.col = col;
            seed.neg_diag = neg_diag;
            seed.pos_diag = pos_diag;
            seed.bitfield = mask & ~(col | neg_diag | pos_diag);
            memset(seed.seed_positions, 0, sizeof(seed.seed_positions));
            seed.seed_positions[0] = (unsigned char)half;
            seed.seed_positions[1] = (unsigned char)bitFieldIndex(lsb);

            if (!generateFrontierFromSeed(board_minus, mask, frontier_depth, &seed, out))
            {
                return 0;
            }
        }
    }

    return 1;
}

#ifndef WIN32
static int getThreadCount(int task_count)
{
    long cores;

    if (task_count <= 1)
    {
        return 1;
    }

    cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (cores < 1)
    {
        cores = 1;
    }
    if (cores > task_count)
    {
        cores = task_count;
    }
    return (int)cores;
}

static void* NqueenWorker(void* arg)
{
    WorkerContext* context = (WorkerContext*)arg;
    int i;
    SOLUTIONTYPE subtotal = 0;

    for (i = context->thread_id; i < context->task_count; i += context->thread_count)
    {
        subtotal += solveTask(context->board_minus, context->mask, &context->tasks[i]);
    }

    context->subtotal = subtotal;
    return NULL;
}
#endif

static SOLUTIONTYPE solveTasksCpu(int board_minus,
                                  BITFIELDTYPE mask,
                                  const NqueenTask* tasks,
                                  int task_count)
{
    int i;
    SOLUTIONTYPE total = 0;

    if (task_count <= 0)
    {
        return 0;
    }

#ifndef WIN32
    {
        int thread_count = getThreadCount(task_count);

        if (thread_count > 1)
        {
            pthread_t* threads = (pthread_t*)malloc((size_t)thread_count * sizeof(pthread_t));
            WorkerContext* contexts = (WorkerContext*)malloc((size_t)thread_count * sizeof(WorkerContext));

            if (threads != NULL && contexts != NULL)
            {
                for (i = 0; i < thread_count; ++i)
                {
                    contexts[i].tasks = tasks;
                    contexts[i].task_count = task_count;
                    contexts[i].thread_id = i;
                    contexts[i].thread_count = thread_count;
                    contexts[i].board_minus = board_minus;
                    contexts[i].mask = mask;
                    contexts[i].subtotal = 0;
                    contexts[i].started = 0;

                    if (0 == pthread_create(&threads[i], NULL, NqueenWorker, &contexts[i]))
                    {
                        contexts[i].started = 1;
                    }
                    else
                    {
                        /* If this thread fails to start, run its share inline. */
                        NqueenWorker(&contexts[i]);
                    }
                }

                for (i = 0; i < thread_count; ++i)
                {
                    if (contexts[i].started)
                    {
                        pthread_join(threads[i], NULL);
                    }
                    total += contexts[i].subtotal;
                }

                free(contexts);
                free(threads);
                return total;
            }

            free(contexts);
            free(threads);
        }
    }
#endif

    for (i = 0; i < task_count; ++i)
    {
        total += solveTask(board_minus, mask, &tasks[i]);
    }

    return total;
}

#ifdef USE_CUDA
static int getGpuMinTasks(void)
{
    const char* env = getenv("NQ_GPU_MIN_TASKS");
    int min_tasks = 600000;

    if (env != NULL && *env != '\0')
    {
        char* endp = NULL;
        long parsed = strtol(env, &endp, 10);
        if (endp != env && parsed >= 1)
        {
            min_tasks = (int)parsed;
        }
    }

    return min_tasks;
}

static int shouldTryGpuSolver(int task_count)
{
    const char* mode = getenv("NQ_SOLVER_MODE");

    if (mode != NULL)
    {
        if (0 == strcmp(mode, "cpu"))
        {
            return 0;
        }
        if (0 == strcmp(mode, "gpu"))
        {
            return 1;
        }
    }

    return task_count >= getGpuMinTasks();
}

static int shouldBucketGpuTasks(void)
{
    const char* value = getenv("NQ_GPU_BUCKET_TASKS");

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
#endif

/* The function which calculates the N queen solutions.
   We calculate one-half the solutions, then flip the results over
   the "Y axis" of the board.  Every solution can be reflected that
   way to generate another unique solution (assuming the board size
   isn't 1 x 1).  That's because a solution cannot be symmetrical
   across the Y-axis (because you can't have two queens in the same
   horizontal row).  A solution also cannot consist of queens 
   down the middle column of a board with an odd number of columns,
   since you can't have two queens in the same vertical row.

   This is a backtracking algorithm.  We place a queen in the top
   row, then note the column and diagonals it occupies.  We then 
   place a queen in the next row down, taking care not to place it
   in the same column or diagonal.  We then update the occupied
   columns & diagonals & move on to the next row.  If no position
   is open in the next row, we back track to the previous row & move
   the queen over to the next available spot in its row & the process
   starts over again.
*/
void Nqueen(int board_size)
{
    FrontierTaskList frontier;
    int frontier_depth;
    int board_minus = board_size - 1; /* board size - 1 */
    BITFIELDTYPE mask = (((BITFIELDTYPE)1) << board_size) - 1; /* if board size is N, mask consists of N 1's */
    SOLUTIONTYPE total;

    initTaskList(&frontier);
    frontier_depth = getFrontierDepth(board_size);
    if (!generateFrontier(board_size, board_minus, mask, frontier_depth, &frontier))
    {
        fprintf(stderr, "Failed to allocate frontier tasks.\n");
        freeTaskList(&frontier);
        g_numsolutions = 0;
        return;
    }

    /*
     * `gpu_path_taken` distinguishes the two scaling conventions in flight while the
     * symmetry-extension Phase 1 lands on CPU first:
     *   - CPU (this build): solveTasksCpu accumulates 12 * (orbit_size / |orbit ∩ S'|)
     *     per leaf. Divide by 12 to recover the integer total.
     *   - GPU: solveTasksGpuCuda still uses the prior vertical-mirror-only convention
     *     (one count per leaf). Multiply by 2 to recover the total.
     * When the GPU path is also migrated to the orbit-weighted convention, both
     * branches will use /12 and this distinction can be removed.
     */
    int gpu_path_taken = 0;
#ifdef USE_CUDA
    int try_gpu = shouldTryGpuSolver(frontier.count);

    if (try_gpu && shouldBucketGpuTasks())
    {
        if (!bucketFrontierTasks(&frontier, board_minus))
        {
            fprintf(stderr, "Warning: failed to bucket GPU tasks; continuing without bucketing.\n");
        }
    }

    if (try_gpu &&
        solveTasksGpuCuda(board_minus, mask, frontier.tasks, frontier.count, &total))
    {
        gpu_path_taken = 1;
    }
    else
    {
        total = solveTasksCpu(board_minus, mask, frontier.tasks, frontier.count);
    }
#else
    total = solveTasksCpu(board_minus, mask, frontier.tasks, frontier.count);
#endif
    freeTaskList(&frontier);

    if (gpu_path_taken)
    {
        g_numsolutions = total * 2;
    }
    else
    {
        g_numsolutions = total / 12;
    }
}

/* Print the results at the end of the run */
void printResults(time_t* pt1, time_t* pt2)
{
    double secs;
    int hours , mins, intsecs;

    printf("End: \t%s", ctime(pt2));
    secs = difftime(*pt2, *pt1);
    intsecs = (int)secs;
    printf("Calculations took %d second%s.\n", intsecs, (intsecs == 1 ? "" : "s"));

    /* Print hours, minutes, seconds */
    hours = intsecs/3600;
    intsecs -= hours * 3600;
    mins = intsecs/60;
    intsecs -= mins * 60;
    if (hours > 0 || mins > 0) 
    {
        printf("Equals ");
        if (hours > 0) 
        {
            printf("%d hour%s, ", hours, (hours == 1) ? "" : "s");
        }
        if (mins > 0)
        {           
            printf("%d minute%s and ", mins, (mins == 1) ? "" : "s");
        }
        printf("%d second%s.\n", intsecs, (intsecs == 1 ? "" : "s"));

    }
}

/* main routine for N Queens program.*/
int main(int argc, char** argv)
{
    time_t t1, t2;
    int boardsize;

    if (argc != 2) {
        printf("N Queens program by Jeff Somers.\n");
        printf("\tallagash98@yahoo.com or jsomers@alumni.williams.edu\n");
        printf("This program calculates the total number of solutions to the N Queens problem.\n");
        printf("Usage: nq <width of board>\n"); /* user must pass in size of board */
        return 0;
    }

    boardsize = atoi(argv[1]);

    /* check size of board is within correct range */
    if (MIN_BOARDSIZE > boardsize || MAX_BOARDSIZE < boardsize)
    {
        printf("Width of board must be between %d and %d, inclusive.\n", 
                                            MIN_BOARDSIZE, MAX_BOARDSIZE );
        return 0;
    }

    time(&t1);
    printf("N Queens program by Jeff Somers.\n");
    printf("\tallagash98@yahoo.com or jsomers@alumni.williams.edu\n");
    printf("Start: \t %s", ctime(&t1));

    Nqueen(boardsize); /* find solutions */
    time(&t2);

    printResults(&t1, &t2);

    if (g_numsolutions != 0)
    {
        printf("For board size %d, " SOLUTION_PRINTF " solution%s found.\n", boardsize, g_numsolutions, (g_numsolutions == 1 ? "" : "s"));
    }
    else
    {
        printf("No solutions found.\n");
    }

    return 0;
}
