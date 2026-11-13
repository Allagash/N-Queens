/*
 * test_peel.c - host test for nq_peel.h (run via `make test-peel`).
 *
 * The CUDA kernel calls nqPeelSolve() once per frontier task. This harness
 * does the same on the CPU: it builds the mirror-half frontier at several
 * depths (so every "rows remaining" branch of the solver is the entry point
 * somewhere), sums nqPeelSolve() over it, doubles, and compares with the
 * known counts. It also checks the solver stays inside the number of stack
 * frames NQ_PEEL_STACK_FRAMES() promises, which is what the CUDA backend
 * sizes its shared memory from.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nq_peel.h"

#define TEST_MAX_N 15
#define CANARY 0xDEADBEEFu

static const unsigned long long EXPECTED[TEST_MAX_N + 1] = {
    0, 1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184
};

static unsigned long long g_sum;
static unsigned long long g_tasks;
static int g_overrun;

static void solveLeafTask(int N, int row, uint32_t col, uint32_t pos, uint32_t neg, uint32_t bitfield)
{
    NqPeelFrame stk[NQ_PEEL_MAX_BOARD + 2];
    int frames = NQ_PEEL_STACK_FRAMES(N - row);
    uint32_t mask = (1u << N) - 1u;

    /* Guard frame directly after the promised capacity. */
    stk[frames].bf = stk[frames].col = stk[frames].pos = stk[frames].neg = CANARY;

    g_sum += nqPeelSolve(mask, (uint32_t)(N - 1), (uint32_t)row, col, pos, neg, bitfield, stk);
    ++g_tasks;

    if (stk[frames].bf != CANARY || stk[frames].col != CANARY ||
        stk[frames].pos != CANARY || stk[frames].neg != CANARY)
    {
        g_overrun = 1;
    }
}

/* `row` rows are placed; `bitfield` holds the free squares of row `row`. */
static void expand(int N, int depth, int row, uint32_t col, uint32_t pos, uint32_t neg, uint32_t bitfield)
{
    uint32_t mask = (1u << N) - 1u;

    if (row >= depth)
    {
        solveLeafTask(N, row, col, pos, neg, bitfield);
        return;
    }
    while (bitfield)
    {
        uint32_t lsb = NQ_PEEL_LSB(bitfield);
        uint32_t c = col | lsb;
        uint32_t p = (pos | lsb) << 1;
        uint32_t n = (neg | lsb) >> 1;
        bitfield &= ~lsb;
        expand(N, depth, row + 1, c, p, n, mask & ~(c | p | n));
    }
}

/* Mirror-half frontier, same seeding as generateFrontier() in nq.c. */
static unsigned long long countBoard(int N, int depth)
{
    uint32_t mask = (1u << N) - 1u;
    int half = N / 2;
    uint32_t first = (1u << half) - 1u;

    g_sum = 0;

    while (first)
    {
        uint32_t lsb = NQ_PEEL_LSB(first);
        uint32_t p = lsb << 1;
        uint32_t n = lsb >> 1;
        first &= ~lsb;
        expand(N, depth, 1, lsb, p, n, mask & ~(lsb | p | n));
    }

    if (N & 1)
    {
        uint32_t middle = 1u << half;
        uint32_t second = (middle - 1u) >> 1;
        while (second)
        {
            uint32_t lsb = NQ_PEEL_LSB(second);
            uint32_t c = middle | lsb;
            uint32_t n = ((middle >> 1) | lsb) >> 1;
            uint32_t p = ((middle << 1) | lsb) << 1;
            second &= ~lsb;
            expand(N, depth, 2, c, p, n, mask & ~(c | p | n));
        }
    }

    return g_sum * 2;
}

int main(void)
{
    int N;
    int cases = 0;
    int failures = 0;

    for (N = 4; N <= TEST_MAX_N; ++N)
    {
        int depth;
        /* Every start depth up to N-1, so each branch of the solver is an entry point. */
        for (depth = 1; depth <= N - 1; ++depth)
        {
            unsigned long long got;

            if (N >= 15 && depth > 8 && depth < N - 1)
            {
                continue; /* keep the run short; deep starts are covered at smaller N */
            }

            g_tasks = 0;
            g_overrun = 0;
            got = countBoard(N, depth);
            ++cases;

            if (got != EXPECTED[N] || g_overrun)
            {
                printf("FAIL  N=%d depth=%d: got %llu, expected %llu%s\n",
                       N, depth, got, EXPECTED[N], g_overrun ? " (stack overrun)" : "");
                ++failures;
            }
        }
    }

    if (failures)
    {
        printf("FAILURES: %d of %d cases.\n", failures, cases);
        return 1;
    }
    printf("All %d cases passed.\n", cases);
    return 0;
}
